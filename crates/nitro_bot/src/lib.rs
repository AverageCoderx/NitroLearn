//! `nitro_bot`: plays an exported NitroLearn policy in RLBot v5.
//!
//! Every packet becomes a `GameSnapshot`; on the training rhythm (every `tick_skip` ticks, applied after
//! `action_delay` ticks) the bot builds the observation and action mask with the very code training used
//! (`nitro_core::obs`, `nitro_core::actions`), runs the policy on the CPU and sends the controls. A policy
//! with an attention encoder plays any number of cars a side: in a game bigger than its room, the
//! observation gets one place per car present (`ObsKind::build_cars`).

pub mod angles;
pub mod decider;
pub mod packet;
pub mod rhythm;

use nitro_core::actions::ActionTable;
use nitro_core::policy::Policy;
use nitro_core::snapshot::{Controls, GameSnapshot};
use nitro_core::team::Team;
use rlbot::flat::{ControllerState, FieldInfo, GamePacket, MatchPhase};
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
        let layout = game
            .observation
            .layout(game.team_size, game.opponents, room);
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
        let any_size = policy.reads_any_cars();
        Ok(Self {
            pads: PadMap::new(field)?,
            jumps: JumpStarts::default(),
            rhythm: Rhythm::with_latency(game.tick_skip, game.action_delay, latency),
            mask: vec![0; table.len()],
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
            policy,
        })
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

    /// Starts afresh, as a training episode does: the rhythm and any decision in flight.
    fn restart(&mut self) {
        self.rhythm.reset();
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
        if self.rhythm.needs(frame) {
            self.finish_decision();
        }
        if let Some(previous) = self.rhythm.begin(frame) {
            let began = Instant::now();
            let game = &self.policy.header().game;
            let index = self.index;
            let cars = if fits {
                game.observation.build(
                    &self.snapshot,
                    index,
                    previous,
                    game.obs_team_size,
                    &mut self.obs,
                );
                None
            } else {
                Some(game.observation.build_cars(
                    &self.snapshot,
                    index,
                    previous,
                    &mut self.cars_obs,
                ))
            };
            self.table.write_mask(
                game.action_masks,
                &self.snapshot.cars[index],
                &mut self.mask,
            );
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
            None if *threads <= 1 => policy.decide(obs, mask, &mut pick),
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
