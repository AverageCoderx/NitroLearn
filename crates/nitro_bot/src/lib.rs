//! `nitro_bot`: plays an exported NitroLearn policy in RLBot v5.
//!
//! Every packet becomes a `GameSnapshot`; on the training rhythm (every `tick_skip` ticks, applied after
//! `action_delay` ticks) the bot builds the observation and action mask with the very code training used
//! (`nitro_core::obs`, `nitro_core::actions`), runs the policy on the CPU and sends the controls. A policy
//! with an attention encoder plays any number of cars a side: in a game bigger than its room, the
//! observation gets one place per car present (`ObsKind::build_cars`). A policy trained with ball
//! prediction reads RLBot's own prediction at the same times ahead as in training.

pub mod angles;
pub mod decider;
pub mod packet;
pub mod rhythm;

use nitro_core::actions::ActionTable;
use nitro_core::ball_prediction::{self, BallPath};
use nitro_core::policy::{Policy, SelectorState};
use nitro_core::snapshot::{Body, Controls, GameSnapshot};
use nitro_core::team::Team;
use rlbot::flat::{
    BallPrediction, ControllerState, FieldInfo, GamePacket, MatchPhase, PredictionSlice,
};
use std::sync::Arc;
use std::time::Instant;

use decider::Seat;
use packet::{JumpStarts, PadMap};
use rhythm::Rhythm;

/// How long the bot took to decide: a decision must reach the game before the tick it is first
/// used on, `action_delay + 1` ticks after its packet (no extra latency), or it acts late. Deciding
/// off the packet's thread, the time runs from the packet to the decider's answer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DecisionTimes {
    pub decisions: u64,
    pub total_ms: f64,
    pub slowest_ms: f64,
    /// Decisions slower than `budget_ms`.
    pub over_budget: u64,
    /// The time the action delay leaves a decision.
    pub budget_ms: f64,
    /// Packets that had to wait for a decision still being made off the packet's thread: their
    /// controls went out late.
    pub waited: u64,
    /// Decisions made in one pass with another car's.
    pub batched: u64,
}

/// How the bot picks an action from the policy's probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Always the most likely allowed action (steady play; the default).
    MostLikely,
    /// Random in proportion to the probabilities, as during training.
    Sample,
}

/// Decisions in a row without a ball prediction before the bot says it is coasting for want of one.
const UNPREDICTED_NOTE: u32 = 3;

/// RLBot's prediction slices kept beyond the furthest decision ahead: for a prediction this many ticks
/// older than its packet.
const SPARE_SLICES: usize = 4;

pub struct NitroBot {
    policy: Arc<Policy>,
    table: ActionTable,
    pads: PadMap,
    jumps: JumpStarts,
    rhythm: Rhythm,
    /// This bot's index in the packet's player list.
    index: usize,
    team_size: usize,
    expected_cars: usize,
    /// Players a side the observation has places for (0 = exactly the trained game).
    room: usize,
    /// The policy reads any number of cars (an attention encoder), so it plays games beyond its room.
    any_size: bool,
    choice: Choice,
    rng: fastrand::Rng,
    snapshot: GameSnapshot,
    obs: Vec<f32>,
    /// The observation in a game beyond the room: one place per car present.
    cars_obs: Vec<f32>,
    mask: Vec<u8>,
    /// For a multi-model bot: which model drives this car; restarts with the rhythm.
    selection: SelectorState,
    /// For a multi-model bot whose models see the game their own way: each view's observation and mask.
    views: Vec<(Vec<f32>, Vec<u8>)>,
    /// Each view's ball prediction (decisions ahead; empty = none).
    view_steps: Vec<Vec<u32>>,
    complained: bool,
    /// The note that the match is bigger than the room was printed.
    told_size: bool,
    /// Whether the last packet handled took a decision (for the match log).
    decided: bool,
    times: DecisionTimes,
    /// Threads one decision's maths is split across (`--threads`); the same answer with any number.
    threads: usize,
    /// The decider this car hands its decisions to; None decides on the packet's own thread.
    seat: Option<Seat>,
    /// A policy trained with ball prediction: the decisions ahead it reads (empty = none), how far any reader
    /// (it or a model's view) needs the path, every decision up to there, the path a decision apart now,
    /// RLBot's latest prediction, the bot's own (with `--meshes`) for when RLBot's is missing or stale, and
    /// the bodies picked for the decision.
    prediction_steps: Vec<u32>,
    prediction_len: u32,
    every: Vec<u32>,
    path_now: Vec<Body>,
    rlbot_prediction: Vec<PredictionSlice>,
    own_prediction: Option<BallPath>,
    predicted: Vec<Body>,
    /// Decisions in a row without a prediction, and whether the note saying so was printed.
    unpredicted: u32,
    told_prediction: bool,
    /// The frame of the last packet in play: a count that goes back starts the rhythm afresh.
    last_frame: Option<u32>,
}

impl NitroBot {
    /// `latency`: ticks the game takes to apply a sent control (0 measured; `Rhythm::with_latency`).
    pub fn new(
        policy: impl Into<Arc<Policy>>,
        field: &FieldInfo,
        index: usize,
        choice: Choice,
        seed: u64,
        latency: u32,
    ) -> Result<Self, String> {
        let policy = policy.into();
        let game = &policy.header().game;
        let room = game.obs_team_size;
        if !game.ball_prediction.is_empty() && !game.observation.pads() {
            return Err(
                "the policy file asks for ball prediction with an observation that has none".into(),
            );
        }
        let prediction_steps = ball_prediction::decisions(&game.ball_prediction, game.tick_skip)?;
        let layout = game
            .observation
            .layout(game.team_size, game.opponents, room)
            .with_ball_prediction(prediction_steps.len());
        let view_steps = policy
            .views()
            .iter()
            .map(|view| view.prediction(game.tick_skip))
            .collect::<Result<Vec<_>, _>>()?;
        if layout.size != policy.obs_size() {
            return Err(format!(
                "the policy file says its observation has {} values, but {:?} for {}v{} (room for {}) has {}",
                policy.obs_size(),
                game.observation,
                game.team_size,
                game.team_size,
                room,
                layout.size
            ));
        }
        let table = ActionTable::new();
        if policy.num_actions() != table.len() {
            return Err(format!(
                "the policy chooses among {} actions, but this bot's action table has {}",
                policy.num_actions(),
                table.len()
            ));
        }
        let expected_cars = game.team_size * if game.opponents { 2 } else { 1 };
        // Only NitroLearn's own observations have a place per car to give a bigger game.
        let any_size = policy.reads_any_cars() && game.observation.pads();
        // The furthest decision ahead any reader needs: this policy or one of its models' views.
        let prediction_len = prediction_steps
            .iter()
            .chain(view_steps.iter().flatten())
            .copied()
            .max()
            .unwrap_or(0);
        Ok(Self {
            pads: PadMap::new(field)?,
            jumps: JumpStarts::default(),
            rhythm: Rhythm::with_latency(game.tick_skip, game.action_delay, latency),
            mask: vec![0; table.len()],
            selection: SelectorState::default(),
            views: policy
                .views()
                .iter()
                .map(|view| Ok((vec![0.0; view.obs_size()?], vec![0; table.len()])))
                .collect::<Result<_, String>>()?,
            view_steps,
            prediction_len,
            every: (1..=prediction_len).collect(),
            path_now: Vec::new(),
            table,
            index,
            team_size: game.team_size,
            expected_cars,
            room,
            any_size,
            choice,
            rng: fastrand::Rng::with_seed(seed),
            snapshot: GameSnapshot::default(),
            obs: vec![0.0; layout.size],
            cars_obs: Vec::new(),
            complained: false,
            told_size: false,
            decided: false,
            times: DecisionTimes {
                budget_ms: f64::from(game.action_delay.saturating_sub(latency) + 1) * 1000.0
                    / 120.0,
                ..DecisionTimes::default()
            },
            threads: 1,
            seat: None,
            predicted: Vec::with_capacity(prediction_steps.len()),
            prediction_steps,
            rlbot_prediction: Vec::new(),
            own_prediction: None,
            unpredicted: 0,
            told_prediction: false,
            last_frame: None,
            policy,
        })
    }

    /// The observation of the latest decision in a game of the trained size (for checks).
    pub fn observation(&self) -> &[f32] {
        &self.obs
    }

    /// Whether the policy reads the ball's predicted path, so RLBot should send its prediction.
    pub fn wants_ball_prediction(&self) -> bool {
        self.prediction_len > 0
    }

    /// RLBot's newest ball prediction (a slice per tick, each with its game time; the first a tick after the
    /// packet it was made with), read at the next decision: only the slices a decision can read are kept (as
    /// far as the furthest time, and a few ticks for a prediction a little older than its packet), copied into
    /// a buffer reused tick after tick.
    pub fn set_ball_prediction(&mut self, prediction: &BallPrediction) {
        if self.wants_ball_prediction() {
            let tick_skip = self.policy.header().game.tick_skip;
            let kept = (self.prediction_len * tick_skip) as usize + SPARE_SLICES;
            self.rlbot_prediction.clear();
            self.rlbot_prediction
                .extend_from_slice(&prediction.slices[..prediction.slices.len().min(kept)]);
        }
    }

    /// Predicts the ball itself when RLBot's prediction is missing (`--meshes`): the collision meshes must
    /// be loaded. RocketSim's physics, as training's prediction.
    pub fn predict_ball_itself(&mut self) {
        if self.wants_ball_prediction() {
            let tick_skip = self.policy.header().game.tick_skip;
            self.own_prediction = Some(BallPath::for_observation(tick_skip, self.prediction_len));
        }
    }

    /// The ball's predicted path a decision apart for a decision on `packet`, and the bodies this policy
    /// reads from it: from RLBot's prediction at whole decisions ahead, else from the bot's own; false when
    /// there is neither.
    fn predict_ball(&mut self, packet: &GamePacket) -> bool {
        let tick_skip = self.policy.header().game.tick_skip;
        let now = ball_prediction::Now {
            seconds: packet.match_info.seconds_elapsed,
            ball: self.snapshot.ball,
        };
        // A stale prediction (from before a touch or a reset, or older than its packet by more than a
        // tick or two) does not fit the packet's ball and is not used.
        let from_rlbot = ball_prediction::pick_timed(
            &self.rlbot_prediction,
            |slice| slice.game_seconds,
            |slice| packet::body(&slice.physics),
            now,
            &self.every,
            tick_skip,
            &mut self.path_now,
        );
        if from_rlbot.is_none() {
            let Some(path) = &mut self.own_prediction else {
                return false;
            };
            self.path_now.clear();
            self.path_now
                .extend_from_slice(path.update(self.snapshot.ball, 1));
        }
        ball_prediction::pick(&self.path_now, &self.prediction_steps, &mut self.predicted);
        true
    }

    /// Hands this car's decisions to `seat`'s decider: every packet is answered at once and the
    /// network runs until the decision is needed, `action_delay` ticks later; the same decisions.
    pub fn use_decider(&mut self, seat: Seat) {
        self.seat = Some(seat);
    }

    /// Splits each decision's maths across `threads` threads (the calling one included): the same
    /// decisions, sooner for a large network.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Coasts in a match bigger than the policy trained for even when it reads any number of cars
    /// (`nitro_bot --trained-size-only`), as every bot did before attention bots played any size.
    pub fn keep_to_trained_size(&mut self) {
        self.any_size = false;
    }

    pub fn rhythm_stats(&self) -> rhythm::RhythmStats {
        self.rhythm.stats
    }

    /// True when the last packet handled took a decision.
    pub fn decided(&self) -> bool {
        self.decided
    }

    pub fn decision_times(&self) -> DecisionTimes {
        self.times
    }

    /// The controls to send for `packet`. Outside live play (countdown, goal replays, pauses) the car
    /// coasts and the rhythm restarts, as a training episode starts afresh.
    pub fn controls(&mut self, packet: &GamePacket) -> Controls {
        let decisions = self.rhythm.stats.decisions;
        let controls = self.live_controls(packet);
        self.decided = self.rhythm.stats.decisions > decisions;
        controls
    }

    fn record_time(&mut self, ms: f64) {
        let times = &mut self.times;
        times.decisions += 1;
        times.total_ms += ms;
        times.slowest_ms = times.slowest_ms.max(ms);
        times.over_budget += u64::from(ms > times.budget_ms);
    }

    /// Starts afresh, as a training episode does: the rhythm, the selector, any decision in flight.
    fn restart(&mut self) {
        self.rhythm.reset();
        self.selection.reset();
        if let Some(path) = &mut self.own_prediction {
            path.forget();
        }
        if let Some(seat) = &mut self.seat {
            if self.choice == Choice::Sample {
                // Drawn and thrown away, so the random numbers run as when deciding on this thread.
                if let Some((answer, _, _)) = seat.wait() {
                    Policy::sample(&answer.logits, &self.mask, &mut self.rng);
                }
            } else {
                seat.forget();
            }
        }
    }

    /// Waits for the decision in flight at the decider and hands its controls to the rhythm.
    fn finish_decision(&mut self) {
        let Some(seat) = &mut self.seat else {
            return;
        };
        let Some((answer, took, waited)) = seat.wait() else {
            if !self.complained {
                println!("nitro_bot: the decider stopped; coasting.");
                self.complained = true;
            }
            self.rhythm.supply(Controls::default());
            return;
        };
        let action = match self.choice {
            Choice::MostLikely => Policy::argmax(&answer.logits, &self.mask),
            Choice::Sample => Policy::sample(&answer.logits, &self.mask, &mut self.rng),
        };
        self.rhythm.supply(self.table.controls(action));
        self.times.waited += u64::from(waited);
        self.times.batched += u64::from(answer.batch > 1);
        self.record_time(took.as_secs_f64() * 1000.0);
    }

    fn live_controls(&mut self, packet: &GamePacket) -> Controls {
        let info = &packet.match_info;
        if !matches!(info.match_phase, MatchPhase::Kickoff | MatchPhase::Active) {
            self.restart();
            return Controls::default();
        }
        // The teams must be the trained shape too: 3 Blue + 1 Orange is not 2v2. A padded observation
        // takes any game with up to `room` players a side, and an attention policy any game.
        let my_team = packet.players.get(self.index).map(|p| p.team);
        let same = packet
            .players
            .iter()
            .filter(|p| Some(p.team) == my_team)
            .count();
        let other = packet.players.len() - same;
        let expected_other = self.expected_cars - self.team_size;
        let fits = if self.room > 0 {
            same <= self.room && other <= self.room
        } else {
            same == self.team_size && other == expected_other
        };
        let trained = || {
            if self.room > 0 {
                format!("games of up to {0}v{0}", self.room)
            } else {
                format!("{}v{}", self.team_size, expected_other)
            }
        };
        if my_team.is_some() && !fits && self.any_size && !self.told_size {
            println!(
                "nitro_bot: the match has {same} players on this bot's team and {other} on the other, more \
                 than the policy trained for ({}); its attention network reads one place per car, so it \
                 plays, but how well at this size is not measured (--trained-size-only makes it coast instead).",
                trained()
            );
            self.told_size = true;
        }
        if my_team.is_none() || !(fits || self.any_size) {
            if !self.complained {
                let trained = trained();
                println!(
                    "nitro_bot: the match has {same} players on this bot's team and {other} on the other, \
                     but the policy was trained for {trained}; coasting."
                );
                self.complained = true;
            }
            self.restart();
            return Controls::default();
        }
        // A multi-model bot's models each see the match their own way: one without room for fewer players
        // needs a full side of them.
        if let Some(problem) = self
            .policy
            .views()
            .iter()
            .find_map(|view| view.match_problem(same, other))
        {
            if !self.complained {
                println!("nitro_bot: {problem}; coasting.");
                self.complained = true;
            }
            self.restart();
            return Controls::default();
        }
        if let Err(problem) =
            packet::snapshot(packet, &self.pads, &mut self.jumps, &mut self.snapshot)
        {
            if !self.complained {
                println!("nitro_bot: {problem}; coasting.");
                self.complained = true;
            }
            return Controls::default();
        }
        let frame = info.frame_num;
        // The game's frame count started over in play: the rhythm's ticks belong to the old count.
        if self
            .last_frame
            .replace(frame)
            .is_some_and(|last| frame < last)
        {
            self.restart();
        }
        if self.rhythm.needs(frame) {
            self.finish_decision();
        }
        if let Some(previous) = self.rhythm.begin(frame) {
            let began = Instant::now();
            if self.wants_ball_prediction() && !self.predict_ball(packet) {
                // RLBot may send its prediction after the packet it goes with: a first decision without one
                // is normal, a few in a row are not.
                self.unpredicted += 1;
                if self.unpredicted >= UNPREDICTED_NOTE && !self.told_prediction {
                    println!(
                        "nitro_bot: this bot was trained with ball prediction, but RLBot sent none and no \
                         --meshes folder was given to compute one; coasting until a prediction comes."
                    );
                    self.told_prediction = true;
                }
                self.rhythm.supply(Controls::default());
                return self.rhythm.current(frame);
            }
            self.unpredicted = 0;
            let game = &self.policy.header().game;
            let index = self.index;
            let cars = if fits {
                game.observation.build_predicted(
                    &self.snapshot,
                    index,
                    previous,
                    game.obs_team_size,
                    &self.predicted,
                    &mut self.obs,
                );
                None
            } else {
                Some(game.observation.build_cars(
                    &self.snapshot,
                    index,
                    previous,
                    &self.predicted,
                    &mut self.cars_obs,
                ))
            };
            self.table.write_mask(
                game.action_masks,
                &self.snapshot.cars[index],
                &mut self.mask,
            );
            for ((view, steps), (obs, mask)) in self
                .policy
                .views()
                .iter()
                .zip(&self.view_steps)
                .zip(&mut self.views)
            {
                view.write(
                    &self.snapshot,
                    index,
                    previous,
                    steps,
                    &self.path_now,
                    &self.table,
                    obs,
                    mask,
                );
            }
            if let Some(seat) = &mut self.seat {
                let obs = if cars.is_some() {
                    &self.cars_obs
                } else {
                    &self.obs
                };
                seat.send(obs.clone(), cars);
                if self.rhythm.needs(frame) {
                    // No action delay: the decision is in effect at once.
                    self.finish_decision();
                }
            } else {
                let action = self.decide_here(cars);
                self.rhythm.supply(self.table.controls(action));
                self.record_time(began.elapsed().as_secs_f64() * 1000.0);
            }
        }
        self.rhythm.current(frame)
    }

    /// The action for the observation just built (`cars`: its car blocks when not the trained size), on
    /// this thread (`--no-decider`).
    fn decide_here(&mut self, cars: Option<(usize, usize)>) -> usize {
        let Self {
            policy,
            obs,
            cars_obs,
            mask,
            selection,
            views,
            rng,
            choice,
            threads,
            ..
        } = self;
        let mut pick = |logits: &[f32], allowed: &[u8]| match choice {
            Choice::MostLikely => Policy::argmax(logits, allowed),
            Choice::Sample => Policy::sample(logits, allowed, rng),
        };
        let mut logits = vec![0.0f32; policy.num_actions()];
        match cars {
            None if policy.is_selector() || *threads <= 1 => {
                let seen: Vec<(&[f32], &[u8])> = views
                    .iter()
                    .map(|(o, m)| (o.as_slice(), m.as_slice()))
                    .collect();
                policy.decide_seeing(obs, mask, &seen, selection, &mut pick)
            }
            None => {
                Team::run(*threads, |team| policy.logits_team(obs, &mut logits, team));
                pick(&logits, mask)
            }
            Some((teammates, opponents)) => {
                Team::run(*threads, |team| {
                    policy.logits_cars_team(cars_obs, teammates, opponents, &mut logits, team)
                });
                pick(&logits, mask)
            }
        }
    }
}

/// Whether `policy` reads the ball's predicted path, so its program must ask RLBot for its ball prediction.
pub fn wants_ball_prediction(policy: &Policy) -> bool {
    !policy.header().game.ball_prediction.is_empty()
        || policy
            .views()
            .iter()
            .any(|view| !view.ball_prediction.is_empty())
}

/// RLBot's controller state for our controls (`use_item` is for rumble, never used).
pub fn controller_state(controls: Controls) -> ControllerState {
    ControllerState {
        throttle: controls.throttle,
        steer: controls.steer,
        pitch: controls.pitch,
        yaw: controls.yaw,
        roll: controls.roll,
        jump: controls.jump,
        boost: controls.boost,
        handbrake: controls.handbrake,
        use_item: false,
    }
}
