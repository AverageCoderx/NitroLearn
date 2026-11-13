//! One arena plus everything needed to run episodes in it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::actions::{ActionTable, MaskRule};
use crate::capture::Capture;
use crate::mechanics::Mechanics;
use crate::obs::ObsKind;
use crate::rewards::{RewardSet, RewardStats, RewardTerm};
use crate::setters::{SetterKind, SetterMix};
use crate::sim::Sim;
use crate::snapshot::{CarSnapshot, Controls, GameSnapshot, MAX_CARS, StepEvents, Team};
use crate::terminals::{Ending, EpisodeClock, TerminalConfig};

const TICK_RATE: f32 = 120.0;
/// When the ball's position is recorded after the episode's first touch (kickoff outcome).
const AFTER_TOUCH_TICKS: u32 = 240;
/// RocketSim's kickoff spots as (|x|, |y|) and their class: the two diagonal ones, the two off-centre ones
/// and the one straight back (Orange's are Blue's mirrored through the centre).
const KICKOFF_SPOTS: [(f32, f32, i8); 3] = [
    (2048.0, 2560.0, KICKOFF_DIAGONAL),
    (256.0, 3840.0, KICKOFF_OFF_CENTRE),
    (0.0, 4608.0, KICKOFF_BACK),
];
pub const KICKOFF_DIAGONAL: i8 = 0;
pub const KICKOFF_OFF_CENTRE: i8 = 1;
pub const KICKOFF_BACK: i8 = 2;
/// A car within this (uu, across the floor) of a spot started there: above the kickoff setter's largest fuzz
/// (300) and below half the gap between the off-centre and back spots (809 / 2), so fuzz never mixes classes.
const KICKOFF_SPOT_REACH: f32 = 400.0;

/// Settings shared by every environment of a pool.
pub struct EnvRules {
    pub team_size: usize,
    pub opponents: bool,
    pub tick_skip: u32,
    /// Ticks between deciding on an action and the action taking effect (0..tick_skip).
    pub action_delay: u32,
    pub obs: ObsKind,
    /// Players per team the observation has places for (0 = the game's own).
    pub obs_team_size: usize,
    pub masks: MaskRule,
    pub terminals: TerminalConfig,
    pub rewards: Vec<RewardTerm>,
    pub setters: SetterMix,
    pub actions: ActionTable,
    /// Each episode every car takes a random place among its team's in the padded observation.
    pub shuffle_slots: bool,
}

impl EnvRules {
    pub fn obs_size(&self) -> usize {
        self.obs
            .layout(self.team_size, self.opponents, self.obs_team_size)
            .size
    }

    pub fn cars_per_env(&self) -> usize {
        self.team_size * if self.opponents { 2 } else { 1 }
    }
}

/// What one car did in one finished episode: the practice-task measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeRecord {
    /// The environment's id, so evaluation can count each environment's episodes in order.
    pub env: u32,
    /// Which state setter started the episode (its index in the settings' list, for the per-setter figures).
    pub setter: u8,
    pub car: u32,
    pub seconds: f32,
    /// Ended by the game or the task (a goal, a touch, a landing) rather than a time limit.
    pub terminal: bool,
    /// Seconds until this car first touched the ball (NaN if it never did).
    pub first_touch: f32,
    /// Seconds until this car was first on its wheels after a step (NaN if never).
    pub landed: f32,
    /// Boost picked up from pads, 0..100 per full tank.
    pub boost_collected: f32,
    /// The ball's distance along the field toward this car's opponents' goal (uu, negative toward its
    /// own) 2 s after the episode's first touch by any car, or when the episode ended if that came
    /// sooner (a goal); NaN if nobody touched the ball.
    pub ball_y_after_touch: f32,
    /// How the episode ended for this car's team: 1 scored, -1 conceded, 0 no goal (a time limit or a
    /// practice-task ending). Self-play ratings count every goal as one game.
    pub goal: i8,
    /// Seconds until any car first touched the ball (NaN if none did); the same in every car's record.
    pub ball_first_touch: f32,
    /// The episode's kickoff, when the kickoff setter started it: `KICKOFF_DIAGONAL`, `KICKOFF_OFF_CENTRE`
    /// or `KICKOFF_BACK`, the most forward spot any car started on (the one nearest the ball; RocketSim
    /// gives both teams the same spots); -1 for another setter. The same in every car's record.
    pub kickoff: i8,
}

/// The class of the kickoff the cars start from (`EpisodeRecord::kickoff`): the most forward spot taken,
/// each car matched to the nearest spot within `KICKOFF_SPOT_REACH` (kickoff fuzz); -1 if none is on one.
pub fn kickoff_class(cars: &[CarSnapshot]) -> i8 {
    cars.iter()
        .filter_map(|car| {
            let (x, y) = (car.body.pos.x.abs(), car.body.pos.y.abs());
            KICKOFF_SPOTS
                .iter()
                .find(|&&(sx, sy, _)| (x - sx).hypot(y - sy) <= KICKOFF_SPOT_REACH)
                .map(|&(_, _, class)| class)
        })
        .min()
        .unwrap_or(-1)
}

/// What environments did since their stats were last taken.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvStats {
    pub rewards: RewardStats,
    /// What the counted cars did, summed over their steps (gameplay statistics; idea credited to RLMind).
    pub play: PlayStats,
    pub steps: u64,
    /// Wall time inside `Env::step`, and the part of it spent in the physics engine.
    pub step_time: Duration,
    pub physics_time: Duration,
    pub episodes: Vec<EpisodeRecord>,
    /// With mixed game sizes: `play` per players a side (index = the size); empty otherwise.
    pub play_by_size: Vec<PlayStats>,
}

impl EnvStats {
    pub fn new(reward_terms: usize) -> Self {
        Self {
            rewards: RewardStats::new(reward_terms),
            ..Self::default()
        }
    }

    /// Back to `EnvStats::new(reward_terms)`, keeping the vectors' memory.
    fn reset(&mut self, reward_terms: usize) {
        let Self {
            rewards,
            play,
            steps,
            step_time,
            physics_time,
            episodes,
            play_by_size,
        } = self;
        rewards.reset(reward_terms);
        *play = PlayStats::default();
        *steps = 0;
        *step_time = Duration::ZERO;
        *physics_time = Duration::ZERO;
        episodes.clear();
        play_by_size.clear();
    }

    pub fn merge(&mut self, other: &Self) {
        self.rewards.merge(&other.rewards);
        self.play.merge(&other.play);
        self.steps += other.steps;
        self.step_time += other.step_time;
        self.physics_time += other.physics_time;
        self.episodes.extend_from_slice(&other.episodes);
        if self.play_by_size.len() < other.play_by_size.len() {
            self.play_by_size
                .resize(other.play_by_size.len(), PlayStats::default());
        }
        for (mine, theirs) in self.play_by_size.iter_mut().zip(&other.play_by_size) {
            mine.merge(theirs);
        }
    }
}

/// Gameplay statistics: sums over the counted cars' steps, at the end of each step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayStats {
    pub car_steps: u64,
    /// Steps ending with the car off the ground.
    pub in_air: u64,
    /// Steps in which the car touched the ball.
    pub touches: u64,
    pub demolished: u64,
    pub supersonic: u64,
    /// Sums of the car's speed and of its speed toward the ball (uu/s), and of its boost (0 to 100).
    pub speed: f64,
    pub toward_ball: f64,
    pub boost: f64,
    /// Mechanics (`crate::mechanics`): how many of each, and the steps with the ball on the car.
    pub aerial_touches: u64,
    pub dribbles: u64,
    pub ball_on_car: u64,
    pub flicks: u64,
    pub double_taps: u64,
    pub flip_resets: u64,
    /// Sums over the flip resets of the ball's centre height and its distance from the opponents' goal line (uu).
    pub flip_reset_height: f64,
    pub flip_reset_goal_dist: f64,
    /// Reset musty goals (a goal within 6 s of the car's flip reset, its back flip between), and the sum of
    /// the ball's speed (uu/s) at them.
    pub reset_musty_goals: u64,
    pub reset_musty_goal_speed: f64,
    pub shots: u64,
    pub saves: u64,
    pub demos: u64,
}

impl PlayStats {
    pub fn merge(&mut self, other: &Self) {
        self.car_steps += other.car_steps;
        self.in_air += other.in_air;
        self.touches += other.touches;
        self.demolished += other.demolished;
        self.supersonic += other.supersonic;
        self.speed += other.speed;
        self.toward_ball += other.toward_ball;
        self.boost += other.boost;
        self.aerial_touches += other.aerial_touches;
        self.dribbles += other.dribbles;
        self.ball_on_car += other.ball_on_car;
        self.flicks += other.flicks;
        self.double_taps += other.double_taps;
        self.flip_resets += other.flip_resets;
        self.flip_reset_height += other.flip_reset_height;
        self.flip_reset_goal_dist += other.flip_reset_goal_dist;
        self.reset_musty_goals += other.reset_musty_goals;
        self.reset_musty_goal_speed += other.reset_musty_goal_speed;
        self.shots += other.shots;
        self.saves += other.saves;
        self.demos += other.demos;
    }

    /// Adds the counted cars of a step that ended in `snap`.
    pub fn add(&mut self, snap: &GameSnapshot, events: &StepEvents, counted: &[bool]) {
        for (car, state) in snap.cars.iter().enumerate() {
            if !counted.get(car).copied().unwrap_or(true) {
                continue;
            }
            self.car_steps += 1;
            self.in_air += u64::from(!state.on_ground && !state.is_demoed);
            self.touches += u64::from(events.cars.get(car).is_some_and(|e| e.touched_ball));
            self.demolished += u64::from(state.is_demoed);
            self.supersonic += u64::from(state.is_supersonic);
            self.speed += f64::from(state.body.vel.length());
            let to_ball = (snap.ball.pos - state.body.pos).normalize_or_zero();
            self.toward_ball += f64::from(state.body.vel.dot(to_ball));
            self.boost += f64::from(state.boost);
        }
    }
}

/// The physics of one step, as `Env::step_begin` plans it: `delay` ticks with the controls the cars
/// have, then every car in `drive` takes its new controls, then the other `ticks - delay` ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PhysicsInput {
    ticks: u32,
    delay: u32,
    /// One bit per car given an action.
    drive: u8,
    controls: [Controls; MAX_CARS],
}

/// Measurements of the episode in progress.
#[derive(Clone, Debug, Default)]
struct EpisodeTracker {
    first_touch: Vec<Option<u32>>,
    landed: Vec<Option<u32>>,
    boost_collected: Vec<f32>,
    any_touch: Option<u32>,
    ball_y_after_touch: Option<f32>,
    /// `EpisodeRecord::kickoff`.
    kickoff: i8,
}

impl EpisodeTracker {
    fn restart(&mut self, cars: usize, kickoff: i8) {
        self.kickoff = kickoff;
        self.first_touch.clear();
        self.first_touch.resize(cars, None);
        self.landed.clear();
        self.landed.resize(cars, None);
        self.boost_collected.clear();
        self.boost_collected.resize(cars, 0.0);
        self.any_touch = None;
        self.ball_y_after_touch = None;
    }

    fn update(&mut self, ticks: u32, prev: &GameSnapshot, cur: &GameSnapshot, events: &StepEvents) {
        for (car, now) in cur.cars.iter().enumerate() {
            if events.cars[car].touched_ball {
                self.first_touch[car].get_or_insert(ticks);
                self.any_touch.get_or_insert(ticks);
            }
            if now.on_ground {
                self.landed[car].get_or_insert(ticks);
            }
            self.boost_collected[car] += (now.boost - prev.cars[car].boost).max(0.0);
        }
        if let Some(touch) = self.any_touch
            && ticks >= touch + AFTER_TOUCH_TICKS
        {
            self.ball_y_after_touch.get_or_insert(cur.ball.pos.y);
        }
    }

    fn finish(
        &self,
        env: u32,
        setter: u8,
        ticks: u32,
        terminal: bool,
        snap: &GameSnapshot,
        out: &mut Vec<EpisodeRecord>,
    ) {
        let seconds = |t: Option<u32>| t.map_or(f32::NAN, |t| t as f32 / TICK_RATE);
        // An episode that ends within 2 s of the touch (a goal) reports where the ball ended up.
        let ball_y = self
            .any_touch
            .map(|_| self.ball_y_after_touch.unwrap_or(snap.ball.pos.y));
        for (car, state) in snap.cars.iter().enumerate() {
            // Blue attacks +y, so Orange's view of the field is flipped.
            let toward_opponents = if state.team == Team::Blue { 1.0 } else { -1.0 };
            out.push(EpisodeRecord {
                env,
                setter,
                car: car as u32,
                seconds: ticks as f32 / TICK_RATE,
                terminal,
                first_touch: seconds(self.first_touch[car]),
                landed: seconds(self.landed[car]),
                boost_collected: self.boost_collected[car],
                ball_y_after_touch: ball_y.map_or(f32::NAN, |y| y * toward_opponents),
                goal: snap
                    .goal_scored
                    .map_or(0, |scorer| if scorer == state.team { 1 } else { -1 }),
                ball_first_touch: seconds(self.any_touch),
                kickoff: self.kickoff,
            });
        }
    }
}

/// Per-car output slices of one environment inside the shared buffers.
pub struct AgentRows<'a> {
    pub obs: &'a mut [f32],
    pub mask: &'a mut [u8],
    pub reward: &'a mut [f32],
    pub terminal: &'a mut [u8],
    pub truncated: &'a mut [u8],
    /// Identity of the observation now in `obs`: episode id and step number inside that episode.
    pub episode: &'a mut [i64],
    pub step: &'a mut [i32],
    /// For truncated cars: the last observation before the reset (value bootstrap); untouched otherwise.
    pub final_obs: &'a mut [f32],
}

pub struct Env {
    rules: Arc<EnvRules>,
    sim: Sim,
    snap: GameSnapshot,
    prev: GameSnapshot,
    events: StepEvents,
    last_actions: Vec<Controls>,
    rewards: RewardSet,
    /// Which cars' rewards enter the statistics (`set_counted`); all of them unless self-play says otherwise.
    counted: Vec<bool>,
    stats: EnvStats,
    clock: EpisodeClock,
    tracker: EpisodeTracker,
    /// Detects mechanics for the gameplay statistics.
    mechanics: Mechanics,
    rng: fastrand::Rng,
    id: u32,
    episodes: u32,
    step_in_episode: i32,
    /// Which state setter started the current episode (its index in the settings' list).
    setter: u8,
    /// Frames for the viewer every few ticks; None while training.
    capture: Option<Capture>,
    /// The physics of the step in progress (`step_begin`).
    input: PhysicsInput,
    /// With `shuffle_slots`: per car, its place among its team's in the observation this episode;
    /// empty otherwise. Drawn from a generator of its own, so the games play as without it.
    slots: Vec<u8>,
    slot_rng: fastrand::Rng,
}

impl Env {
    /// Chooses which cars' rewards enter the statistics (one flag per car, in the arena's car order).
    pub fn set_counted(&mut self, counted: &[bool]) -> Result<(), String> {
        if counted.len() != self.counted.len() {
            return Err(format!(
                "expected {} flags (one per car), got {}",
                self.counted.len(),
                counted.len()
            ));
        }
        self.counted.copy_from_slice(counted);
        Ok(())
    }

    /// `id` must be unique within the pool; it prefixes every episode id this environment produces.
    pub fn new(rules: Arc<EnvRules>, id: u32, seed: u64) -> Self {
        let team_size = rules.team_size;
        Self::sized(rules, id, seed, team_size)
    }

    /// An environment of `team_size` a side, whatever `rules.team_size` says (mixed game sizes).
    pub fn sized(rules: Arc<EnvRules>, id: u32, seed: u64, team_size: usize) -> Self {
        let sim = Sim::new(team_size, rules.opponents, seed);
        Self::with_sim(rules, id, seed, sim)
    }

    fn with_sim(rules: Arc<EnvRules>, id: u32, seed: u64, sim: Sim) -> Self {
        let cars = sim.num_cars();
        let mut env = Self {
            rewards: RewardSet::new(rules.rewards.clone(), rules.tick_skip),
            counted: vec![true; cars],
            stats: EnvStats::new(rules.rewards.len()),
            clock: EpisodeClock::new(rules.terminals),
            tracker: EpisodeTracker::default(),
            mechanics: Mechanics::new(rules.tick_skip),
            rules,
            sim,
            snap: GameSnapshot::default(),
            prev: GameSnapshot::default(),
            events: StepEvents::default(),
            last_actions: vec![Controls::default(); cars],
            rng: fastrand::Rng::with_seed(seed ^ 0x9E37_79B9_7F4A_7C15),
            id,
            episodes: 0,
            step_in_episode: 0,
            setter: 0,
            capture: None,
            input: PhysicsInput::default(),
            slots: Vec::new(),
            slot_rng: fastrand::Rng::with_seed(seed ^ 0x51_07_5A_17),
        };
        env.start_episode();
        env
    }

    fn start_episode(&mut self) {
        let setter = self
            .rules
            .setters
            .apply(&mut self.sim, &mut self.rng, &mut self.prev);
        self.setter = setter as u8;
        for car in 0..self.sim.num_cars() {
            self.sim.set_controls(car, Controls::default());
        }
        self.last_actions.fill(Controls::default());
        self.mechanics.reset();
        self.sim.forget();
        self.sim.details(&mut self.events);
        self.events.prev_details.clone_from(&self.events.details);
        self.events.tick = self.sim.tick();
        self.events.last_touch = None;
        self.rewards.reset();
        self.sim.read(&mut self.snap);
        self.prev.clone_from(&self.snap);
        if let Some(capture) = &mut self.capture {
            capture.restart(&mut self.sim, &self.last_actions);
        }
        self.clock.restart(&self.snap);
        // Read here, after the setter: where the cars start.
        let kickoff = match self.rules.setters.kind(setter) {
            SetterKind::Kickoff { .. } => kickoff_class(&self.snap.cars),
            _ => -1,
        };
        self.tracker.restart(self.snap.cars.len(), kickoff);
        self.episodes += 1;
        self.step_in_episode = 0;
        if self.rules.shuffle_slots {
            self.shuffle_slots();
        }
    }

    /// Gives every car a random place among its team's `obs_team_size` places.
    fn shuffle_slots(&mut self) {
        let room = self.rules.obs_team_size;
        self.slots.resize(self.snap.cars.len(), 0);
        for team in [Team::Blue, Team::Orange] {
            let mut places: Vec<u8> = (0..room as u8).collect();
            self.slot_rng.shuffle(&mut places);
            let cars = self.snap.cars.iter().enumerate();
            for ((car, _), place) in cars.filter(|(_, c)| c.team == team).zip(places) {
                self.slots[car] = place;
            }
        }
    }

    /// Players a side in this arena (mixed game sizes).
    pub fn team_size(&self) -> usize {
        self.sim.num_cars() / if self.rules.opponents { 2 } else { 1 }
    }

    fn episode_id(&self) -> i64 {
        (i64::from(self.id) << 32) | i64::from(self.episodes)
    }

    /// The current state of the game.
    pub fn snapshot(&self) -> &GameSnapshot {
        &self.snap
    }

    /// Starts writing a viewer frame every `every_ticks` ticks (`capture`), beginning with the current state.
    pub fn start_capture(&mut self, every_ticks: u32) {
        self.capture = Some(Capture::new(every_ticks, &mut self.sim, &self.last_actions));
    }

    /// The frames written since the last call (empty when capturing is off).
    pub fn take_capture(&mut self) -> Vec<f32> {
        self.capture
            .as_mut()
            .map(|capture| std::mem::take(&mut capture.frames))
            .unwrap_or_default()
    }

    /// Each boost pad's position and whether it is a big one, in the snapshot's order.
    pub fn pad_layout(&self) -> Vec<[f32; 4]> {
        self.sim.pad_layout()
    }

    fn run_ticks(&mut self, ticks: u32) {
        if self.capture.is_none() {
            self.sim.step(ticks, &mut self.events);
            return;
        }
        for _ in 0..ticks {
            self.sim.step(1, &mut self.events);
            if let Some(capture) = &mut self.capture {
                capture.after_tick(&mut self.sim, &self.events, &self.last_actions);
            }
        }
    }

    /// Writes observations, masks and episode identity for the current state.
    pub fn write_observations(&self, rows: &mut AgentRows<'_>) {
        let size = self.rules.obs_size();
        let actions = self.rules.actions.len();
        for car in 0..self.snap.cars.len() {
            self.rules.obs.build_slotted(
                &self.snap,
                car,
                self.last_actions[car],
                self.rules.obs_team_size,
                &self.slots,
                &mut rows.obs[car * size..(car + 1) * size],
            );
            self.rules.actions.write_mask(
                self.rules.masks,
                &self.snap.cars[car],
                &mut rows.mask[car * actions..(car + 1) * actions],
            );
        }
        rows.episode.fill(self.episode_id());
        rows.step.fill(self.step_in_episode);
    }

    /// Plays one step with `actions` (indices into the action table, one per car).
    pub fn step(&mut self, actions: &[i32], rows: &mut AgentRows<'_>) {
        self.step_begin(actions);
        self.step_physics();
        self.step_end(rows);
    }

    /// The part of a step before the physics: what happened is cleared, the action delay drawn and
    /// the actions turned into controls.
    fn step_begin(&mut self, actions: &[i32]) {
        let start = Instant::now();
        let rules = &self.rules;
        self.events.clear(self.sim.num_cars());
        self.events.step_seconds = rules.tick_skip as f32 / 120.0;
        if let Some(capture) = &mut self.capture {
            capture.step_started();
        }
        self.input = PhysicsInput {
            ticks: rules.tick_skip,
            delay: rules.action_delay,
            ..PhysicsInput::default()
        };
        for (car, &action) in actions.iter().enumerate() {
            self.input.controls[car] = rules.actions.controls(action as usize);
            self.input.drive |= 1 << car;
        }
        self.stats.step_time += start.elapsed();
    }

    /// The step's physics as `step_begin` planned it.
    fn step_physics(&mut self) {
        let start = Instant::now();
        let input = self.input;
        if input.delay > 0 {
            self.run_ticks(input.delay);
        }
        self.take_actions();
        self.run_ticks(input.ticks - input.delay);
        self.add_physics_time(start.elapsed());
    }

    /// The cars given an action take it, at tick `delay` of the step: their controls and `last_actions`.
    fn take_actions(&mut self) {
        let input = self.input;
        for (car, &controls) in input.controls.iter().enumerate() {
            if input.drive & (1 << car) != 0 {
                self.sim.set_controls(car, controls);
                self.last_actions[car] = controls;
            }
        }
    }

    /// Time spent on this arena's physics.
    fn add_physics_time(&mut self, spent: Duration) {
        self.stats.physics_time += spent;
        self.stats.step_time += spent;
    }

    /// The part of a step after the physics: the state it reached is read, rewards and endings
    /// written, an ended episode replaced by a new one, and the next observations written.
    fn step_end(&mut self, rows: &mut AgentRows<'_>) {
        let start = Instant::now();
        let ending = self.settle(rows);
        self.conclude(ending, rows);
        self.stats.steps += 1;
        self.stats.step_time += start.elapsed();
    }

    /// Simulates the step and writes rewards and endings; the state is still the one the step reached.
    #[cfg(test)]
    fn advance(&mut self, actions: &[i32], rows: &mut AgentRows<'_>) -> Ending {
        self.step_begin(actions);
        self.step_physics();
        self.settle(rows)
    }

    /// Reads the state the step's physics reached and writes rewards and endings.
    fn settle(&mut self, rows: &mut AgentRows<'_>) -> Ending {
        let rules = &self.rules;
        std::mem::swap(&mut self.prev, &mut self.snap);
        self.sim.read(&mut self.snap);
        std::mem::swap(&mut self.events.details, &mut self.events.prev_details);
        self.sim.details(&mut self.events);
        self.events.tick = self.sim.tick();
        let ending = self
            .clock
            .advance(rules.tick_skip, &self.snap, &self.events, &mut self.rng);
        self.tracker
            .update(self.clock.ticks(), &self.prev, &self.snap, &self.events);
        self.stats.play.add(&self.snap, &self.events, &self.counted);
        self.mechanics.update(
            &self.prev,
            &self.snap,
            &self.events,
            &self.counted,
            &mut self.stats.play,
        );
        if let Some(capture) = &mut self.capture {
            capture.mark(self.mechanics.marks());
        }
        if ending != Ending::Continues {
            self.tracker.finish(
                self.id,
                self.setter,
                self.clock.ticks(),
                ending == Ending::Terminal,
                &self.snap,
                &mut self.stats.episodes,
            );
        }
        self.rewards.compute(
            &self.prev,
            &self.snap,
            &self.events,
            &self.counted,
            rows.reward,
            &mut self.stats.rewards,
        );
        rows.terminal.fill(u8::from(ending == Ending::Terminal));
        rows.truncated.fill(u8::from(ending == Ending::Truncated));
        ending
    }

    /// Saves the final observation of a truncated episode, starts a new episode if one ended, and
    /// writes the observations the next action will be chosen from.
    fn conclude(&mut self, ending: Ending, rows: &mut AgentRows<'_>) {
        match ending {
            Ending::Continues => self.step_in_episode += 1,
            Ending::Truncated => {
                let size = self.rules.obs_size();
                for car in 0..self.snap.cars.len() {
                    self.rules.obs.build_slotted(
                        &self.snap,
                        car,
                        self.last_actions[car],
                        self.rules.obs_team_size,
                        &self.slots,
                        &mut rows.final_obs[car * size..(car + 1) * size],
                    );
                }
                self.start_episode();
            }
            Ending::Terminal => self.start_episode(),
        }
        self.write_observations(rows);
    }

    /// Statistics since the last call.
    pub fn take_stats(&mut self) -> EnvStats {
        let fresh = self.fresh_stats();
        std::mem::replace(&mut self.stats, fresh)
    }

    /// Adds the statistics since the last call to `total` and starts them again: `take_stats` without
    /// making and freeing a set of vectors per environment each time (the pool's, every iteration).
    pub fn add_stats_to(&mut self, total: &mut EnvStats) {
        total.merge(&self.stats);
        self.stats.reset(self.rules.rewards.len());
    }

    fn fresh_stats(&self) -> EnvStats {
        EnvStats::new(self.rules.rewards.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::NUM_ACTIONS;

    #[test]
    fn gameplay_statistics_count_only_the_counted_cars() {
        use crate::snapshot::{Body, CarEvents, CarSnapshot};
        use rocketsim::{Mat3A, Vec3A};
        let car = |x: f32, vx: f32, on_ground: bool| CarSnapshot {
            team: Team::Blue,
            body: Body {
                pos: Vec3A::new(x, 0.0, 17.0),
                rot: Mat3A::IDENTITY,
                vel: Vec3A::new(vx, 0.0, 0.0),
                ang_vel: Vec3A::ZERO,
            },
            boost: 40.0,
            on_ground,
            has_jumped: false,
            has_double_jumped: false,
            has_flipped: false,
            can_flip: true,
            is_demoed: false,
            is_supersonic: vx.abs() >= 2200.0,
            demo_timer: 0.0,
            flip_time_left: 0.0,
        };
        let snap = GameSnapshot {
            cars: vec![car(-1000.0, 2300.0, false), car(1000.0, 500.0, true)],
            ball: Body {
                pos: Vec3A::new(0.0, 0.0, 17.0),
                ..Body::RESTING
            },
            ..GameSnapshot::default()
        };
        let mut events = StepEvents::default();
        events.clear(2);
        events.cars[0] = CarEvents {
            touched_ball: true,
            ..CarEvents::default()
        };
        let mut play = PlayStats::default();
        play.add(&snap, &events, &[true, false]);
        assert_eq!(
            (play.car_steps, play.in_air, play.touches, play.supersonic),
            (1, 1, 1, 1)
        );
        assert!(
            (play.toward_ball - 2300.0).abs() < 1e-3,
            "driving straight at the ball"
        );
        play.add(&snap, &events, &[true, true]);
        assert_eq!(play.car_steps, 3);
        assert!(
            (play.toward_ball - (4600.0 - 500.0)).abs() < 1e-2,
            "the second car drives away"
        );
        assert_eq!(play.boost, 120.0);
    }

    use rocketsim::Vec3A;

    use crate::pool::{BufferShape, OwnedBuffers};
    use crate::snapshot::Body;
    use crate::testing::{meshes_loaded, pool_config};

    fn rows(owned: &mut OwnedBuffers) -> AgentRows<'_> {
        AgentRows {
            obs: &mut owned.obs,
            mask: &mut owned.mask,
            reward: &mut owned.reward,
            terminal: &mut owned.terminal,
            truncated: &mut owned.truncated,
            episode: &mut owned.episode,
            step: &mut owned.step,
            final_obs: &mut owned.final_obs,
        }
    }

    /// A step written in one piece, to compare with `Env::step`'s three parts (`step_begin`, `step_physics`
    /// and `step_end`).
    fn step_legacy(env: &mut Env, actions: &[i32], rows: &mut AgentRows<'_>) {
        let start = Instant::now();
        let rules = Arc::clone(&env.rules);
        env.events.clear(env.sim.num_cars());
        env.events.step_seconds = rules.tick_skip as f32 / 120.0;
        if let Some(capture) = &mut env.capture {
            capture.step_started();
        }
        let physics_start = Instant::now();
        let delay = rules.action_delay;
        if delay > 0 {
            env.run_ticks(delay);
        }
        for (car, &action) in actions.iter().enumerate() {
            let controls = rules.actions.controls(action as usize);
            env.sim.set_controls(car, controls);
            env.last_actions[car] = controls;
        }
        env.run_ticks(rules.tick_skip - delay);
        env.stats.physics_time += physics_start.elapsed();

        let ending = env.settle(rows);
        env.conclude(ending, rows);
        env.stats.steps += 1;
        env.stats.step_time += start.elapsed();
    }

    /// Every buffer's values as bits (so a NaN compares too).
    fn buffer_bits(owned: &OwnedBuffers) -> Vec<u64> {
        let floats = owned
            .obs
            .iter()
            .chain(&owned.reward)
            .chain(&owned.final_obs);
        let floats = floats.map(|v| u64::from(v.to_bits()));
        let bytes = owned
            .mask
            .iter()
            .chain(&owned.terminal)
            .chain(&owned.truncated);
        let bytes = bytes.map(|&v| u64::from(v));
        let ints = owned.episode.iter().map(|&v| v as u64);
        let ints = ints.chain(owned.step.iter().map(|&v| u64::from(v as u32)));
        floats.chain(bytes).chain(ints).collect()
    }

    /// A step in its three parts (before the physics, the physics, after it) gives exactly the step written
    /// in one piece, 2v2 with action delay, with and without a capture (which steps tick by tick).
    #[test]
    fn the_step_in_parts_matches_the_step_in_one_piece() {
        if !meshes_loaded() {
            return;
        }
        let mut config = pool_config(2, 1, 1, 3);
        config.team_size = 2;
        config.action_delay = 2;
        let rules = Arc::new(config.rules().unwrap());
        let shape = BufferShape {
            agents: 4,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        };
        let mut endings = 0;
        for capture in [false, true] {
            let (mut split, mut old) = (
                Env::new(Arc::clone(&rules), 3, 21),
                Env::new(Arc::clone(&rules), 3, 21),
            );
            if capture {
                split.start_capture(3);
                old.start_capture(3);
            }
            let (mut a, mut b) = (OwnedBuffers::new(shape), OwnedBuffers::new(shape));
            let mut rng = fastrand::Rng::with_seed(4);
            for step in 0..400 {
                let actions: Vec<i32> = (0..4).map(|_| rng.i32(0..NUM_ACTIONS as i32)).collect();
                split.step(&actions, &mut rows(&mut a));
                step_legacy(&mut old, &actions, &mut rows(&mut b));
                assert!(
                    buffer_bits(&a) == buffer_bits(&b),
                    "step {step}, capture {capture}"
                );
                let (frames_a, frames_b) = (split.take_capture(), old.take_capture());
                assert_eq!(frames_a.len(), frames_b.len());
                assert!(
                    frames_a
                        .iter()
                        .zip(&frames_b)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                endings += a
                    .terminal
                    .iter()
                    .chain(&a.truncated)
                    .filter(|&&e| e != 0)
                    .count();
            }
            let (stats_a, stats_b) = (split.take_stats(), old.take_stats());
            assert_eq!(stats_a.rewards, stats_b.rewards);
            assert_eq!(stats_a.steps, stats_b.steps);
            assert!(
                stats_a.physics_time > Duration::ZERO && stats_a.physics_time < stats_a.step_time
            );
        }
        assert!(endings > 20, "only {endings} endings");
    }

    #[test]
    fn a_capture_writes_a_frame_every_few_ticks_and_leaves_the_game_unchanged() {
        use crate::capture::{CAR, FORMAT, HEAD, PADS_AT, RESTART, SCORE_AT};
        if !meshes_loaded() {
            return;
        }
        let rules = Arc::new(pool_config(2, 1, 1, 5).rules().unwrap());
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 2,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        });
        let drive = (0..NUM_ACTIONS)
            .find(|&i| {
                let c = rules.actions.controls(i);
                c.throttle == 1.0 && c.steer == 0.0 && !c.boost && !c.jump && !c.handbrake
            })
            .unwrap() as i32;
        let (mut plain, mut watched) = (
            Env::new(Arc::clone(&rules), 0, 5),
            Env::new(Arc::clone(&rules), 0, 5),
        );
        assert!(plain.take_capture().is_empty(), "off unless started");
        watched.start_capture(2);
        let pads = watched.pad_layout();
        assert_eq!(
            pads.iter().filter(|pad| pad[3] == 1.0).count(),
            6,
            "six big pads"
        );
        let width = HEAD + 2 * CAR;
        let first = watched.take_capture();
        assert_eq!((first.len(), first[0]), (width, RESTART));
        assert_eq!((FORMAT, width), (2, 52 + 2 * 30));
        assert!(
            first[PADS_AT..HEAD].iter().all(|&wait| wait == 0.0),
            "every pad up at a kickoff"
        );
        assert_eq!(first[SCORE_AT..PADS_AT], [0.0, 0.0], "no goals yet");
        assert_eq!(
            (first[7..10].to_vec(), first[10..13].to_vec()),
            (vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]),
            "the ball rests the right way up"
        );
        for _ in 0..20 {
            plain.step(&[drive, drive], &mut rows(&mut owned));
            watched.step(&[drive, drive], &mut rows(&mut owned));
            let frames = watched.take_capture();
            assert_eq!(frames.len() % width, 0);
            let last = &frames[frames.len() - width..];
            for car in 0..2 {
                let at = HEAD + car * CAR;
                assert_eq!(last[at], car as f32, "Blue, then Orange");
                assert_eq!(last[at + 18], 1.0, "the throttle in use");
                assert_eq!(last[at + 26] as u32 & 15, 15, "four wheels on the floor");
                assert_eq!(
                    last[at + 28..at + 30],
                    [0.0, 0.0],
                    "an Octane body, not demolished"
                );
                assert_eq!(
                    last[at + 1..at + 4],
                    plain.snap.cars[car].body.pos.to_array(),
                    "capturing does not change the game"
                );
            }
        }
    }

    #[test]
    fn the_ball_landing_cut_off_fires_on_the_step_of_the_first_floor_contact() {
        if !meshes_loaded() {
            return;
        }
        let mut config = pool_config(1, 1, 1, 6);
        config.opponents = false;
        config.terminals = TerminalConfig {
            truncate_on_ball_landing: 1.0,
            ..config_terminals()
        };
        let rules = Arc::new(config.rules().unwrap());
        let idle = (0..NUM_ACTIONS)
            .find(|&i| {
                let c = rules.actions.controls(i);
                c.throttle == 0.0
                    && c.steer == 0.0
                    && c.pitch == 0.0
                    && c.yaw == 0.0
                    && c.roll == 0.0
                    && !c.jump
                    && !c.boost
                    && !c.handbrake
            })
            .unwrap() as i32;
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 1,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        });
        // Eight drop heights, so the first contact falls on every phase of the 8-tick step.
        for k in 0..8 {
            let height = 400.0 + 75.0 * k as f32;
            let mut env = Env::new(Arc::clone(&rules), 0, 6);
            let ball = Body {
                pos: Vec3A::new(0.0, 2000.0, height),
                ..Body::RESTING
            };
            let far_away = Body {
                pos: Vec3A::new(-3000.0, -3000.0, 17.0),
                ..Body::RESTING
            };
            env.sim.place(ball, &[(far_away, 0.0, true)]);
            env.sim.read(&mut env.snap);
            env.clock.restart(&env.snap);
            let mut ending = Ending::Continues;
            for _ in 0..150 {
                ending = env.advance(&[idle], &mut rows(&mut owned));
                if ending != Ending::Continues {
                    break;
                }
            }
            assert_eq!(ending, Ending::Truncated, "drop from {height}");
            // Free fall of the centre to 100 uu under 650 uu/s²; the ball's drag makes it a little slower.
            // Missing the first bounce would show as hundreds of ticks more (the ball rises again).
            let fall_ticks = (2.0 * (height - 100.0) / 650.0).sqrt() * 120.0;
            let ticks = env.clock.ticks() as f32;
            assert!(
                ticks >= fall_ticks - 8.0 && ticks <= fall_ticks + 24.0,
                "drop from {height}: cut after {ticks} ticks, first contact expected near {fall_ticks:.0}"
            );
        }
    }

    #[test]
    fn truncation_keeps_the_reached_state_for_bootstrapping_and_goals_do_not() {
        if !meshes_loaded() {
            return;
        }
        let config = pool_config(1, 1, 1, 4);
        let rules = Arc::new(config.rules().unwrap());
        let size = rules.obs_size();
        let mut env = Env::new(Arc::clone(&rules), 0, 4);
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 2,
            obs_size: size,
            num_actions: rules.actions.len(),
        });
        let mut rng = fastrand::Rng::with_seed(3);
        let (mut truncations, mut goals) = (0, 0);
        for i in 0..6000 {
            if i % 97 == 0 {
                // Random play rarely scores, so now and then send the ball into the orange net.
                let shot = Body {
                    pos: Vec3A::new(0.0, 4900.0, 300.0),
                    vel: Vec3A::new(0.0, 2500.0, 0.0),
                    ..Body::RESTING
                };
                let cars: Vec<_> = env
                    .snap
                    .cars
                    .iter()
                    .map(|c| (c.body, c.boost, c.on_ground))
                    .collect();
                env.sim.place(shot, &cars);
            }
            let actions = [
                rng.i32(0..NUM_ACTIONS as i32),
                rng.i32(0..NUM_ACTIONS as i32),
            ];
            owned.final_obs.fill(f32::NAN);
            let ending = env.advance(&actions, &mut rows(&mut owned));
            let mut reached = vec![0.0; 2 * size];
            for car in 0..2 {
                rules.obs.build(
                    &env.snap,
                    car,
                    env.last_actions[car],
                    0,
                    &mut reached[car * size..(car + 1) * size],
                );
            }
            let episode_before = env.episode_id();
            env.conclude(ending, &mut rows(&mut owned));
            match ending {
                Ending::Continues => {
                    assert_eq!(owned.obs, reached);
                    assert_eq!(owned.episode, [episode_before; 2]);
                }
                Ending::Truncated => {
                    truncations += 1;
                    assert_eq!(
                        owned.final_obs, reached,
                        "bootstrap must use the state the step reached"
                    );
                    assert_eq!(owned.truncated, [1, 1]);
                    assert_ne!(owned.episode[0], episode_before);
                }
                Ending::Terminal => {
                    goals += 1;
                    assert!(
                        owned.final_obs.iter().all(|v| v.is_nan()),
                        "nothing to bootstrap after a goal"
                    );
                    assert_eq!(owned.terminal, [1, 1]);
                    assert_ne!(owned.episode[0], episode_before);
                }
            }
        }
        assert!(truncations > 10, "only {truncations} truncations");
        assert!(
            goals > 30,
            "only {goals} goals; the terminal branch went untested"
        );
    }

    /// Plays a lone car from `car` (ball at `ball`) with full throttle until the episode ends. The meshes
    /// must be loaded first (`meshes_loaded`).
    fn practice_episode(terminals: TerminalConfig, ball: Body, car: Body) -> EpisodeRecord {
        let mut config = pool_config(1, 1, 1, 5);
        config.opponents = false;
        config.terminals = terminals;
        let rules = Arc::new(config.rules().unwrap());
        let throttle = (0..NUM_ACTIONS)
            .find(|&i| {
                let c = rules.actions.controls(i);
                c.throttle == 1.0 && c.steer == 0.0 && !c.boost && !c.jump && !c.handbrake
            })
            .unwrap() as i32;
        let mut env = Env::new(Arc::clone(&rules), 7, 5);
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 1,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        });
        env.sim.place(ball, &[(car, 0.0, false)]);
        for _ in 0..150 {
            if env.advance(&[throttle], &mut rows(&mut owned)) != Ending::Continues {
                return env.take_stats().episodes[0];
            }
        }
        panic!("the episode never ended");
    }

    #[test]
    fn episode_records_measure_landing_and_touching() {
        if !meshes_loaded() {
            return;
        }
        let ball_far = Body {
            pos: Vec3A::new(0.0, 3000.0, 93.0),
            ..Body::RESTING
        };
        let landing = practice_episode(
            TerminalConfig {
                end_on_landing: true,
                ..config_terminals()
            },
            ball_far,
            Body {
                pos: Vec3A::new(1500.0, -1500.0, 80.0),
                ..Body::RESTING
            },
        );
        assert!(landing.terminal && landing.env == 7 && landing.car == 0);
        assert!((0.1..1.0).contains(&landing.landed), "{landing:?}");
        assert!(landing.first_touch.is_nan() && landing.ball_y_after_touch.is_nan());
        assert_eq!(landing.boost_collected, 0.0);

        let touch = practice_episode(
            TerminalConfig {
                end_on_touch: true,
                ..config_terminals()
            },
            Body {
                pos: Vec3A::new(0.0, 300.0, 93.0),
                ..Body::RESTING
            },
            Body {
                pos: Vec3A::new(0.0, 0.0, 17.0),
                rot: rocketsim::Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2),
                vel: Vec3A::new(0.0, 1500.0, 0.0),
                ang_vel: Vec3A::ZERO,
            },
        );
        assert!(touch.terminal, "{touch:?}");
        assert!((0.0..0.5).contains(&touch.first_touch) && touch.first_touch == touch.seconds);
    }

    /// The same situation seen from the other side: rotated 180Â° about the vertical axis.
    fn mirrored(body: Body) -> Body {
        let flip = Vec3A::new(-1.0, -1.0, 1.0);
        Body {
            pos: body.pos * flip,
            rot: rocketsim::Mat3A::from_cols(
                body.rot.x_axis * flip,
                body.rot.y_axis * flip,
                body.rot.z_axis * flip,
            ),
            vel: body.vel * flip,
            ang_vel: body.ang_vel * flip,
        }
    }

    fn observe(env: &Env, car: usize) -> Vec<f32> {
        let mut out = vec![0.0; env.rules.obs_size()];
        env.rules
            .obs
            .build(&env.snap, car, env.last_actions[car], 0, &mut out);
        out
    }

    #[test]
    fn a_mirrored_game_gives_each_car_its_counterparts_observations_and_motion() {
        if !meshes_loaded() {
            return;
        }
        let config = pool_config(1, 1, 1, 9);
        let rules = Arc::new(config.rules().unwrap());
        let (mut game, mut mirror) = (
            Env::new(Arc::clone(&rules), 0, 9),
            Env::new(Arc::clone(&rules), 1, 9),
        );
        let ball = Body {
            pos: Vec3A::new(700.0, 400.0, 300.0),
            vel: Vec3A::new(-300.0, 200.0, 100.0),
            ..Body::RESTING
        };
        let blue = Body {
            pos: Vec3A::new(-1200.0, -2000.0, 17.0),
            rot: rocketsim::Mat3A::from_rotation_z(0.9),
            vel: Vec3A::new(500.0, 900.0, 0.0),
            ang_vel: Vec3A::ZERO,
        };
        let orange = Body {
            pos: Vec3A::new(1500.0, 2500.0, 17.0),
            rot: rocketsim::Mat3A::from_rotation_z(-2.0),
            vel: Vec3A::new(-400.0, -800.0, 0.0),
            ang_vel: Vec3A::ZERO,
        };
        // Cars are ordered [blue, orange]; in the mirror each team takes the other's place.
        game.sim
            .place(ball, &[(blue, 40.0, true), (orange, 70.0, true)]);
        mirror.sim.place(
            mirrored(ball),
            &[(mirrored(orange), 70.0, true), (mirrored(blue), 40.0, true)],
        );
        for env in [&mut game, &mut mirror] {
            env.sim.read(&mut env.snap);
        }
        assert_eq!(
            observe(&game, 0),
            observe(&mirror, 1),
            "blue here = orange there"
        );
        assert_eq!(
            observe(&game, 1),
            observe(&mirror, 0),
            "orange here = blue there"
        );

        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 2,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        });
        let mut rng = fastrand::Rng::with_seed(4);
        // The physics is mirror-symmetric up to float rounding (about 4e-6 after one step), not bit-exact.
        for step in 0..15 {
            let (a, b) = (
                rng.i32(0..NUM_ACTIONS as i32),
                rng.i32(0..NUM_ACTIONS as i32),
            );
            game.advance(&[a, b], &mut rows(&mut owned));
            mirror.advance(&[b, a], &mut rows(&mut owned));
            for (here, there) in [(0, 1), (1, 0)] {
                let (x, y) = (observe(&game, here), observe(&mirror, there));
                let worst = x
                    .iter()
                    .zip(&y)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f32::max);
                assert!(
                    worst < 1e-4,
                    "step {step}: car {here} differs from its mirror by {worst}"
                );
            }
        }
    }

    fn config_terminals() -> TerminalConfig {
        TerminalConfig {
            no_touch_seconds: 0.0,
            max_episode_seconds: 10.0,
            end_on_touch: false,
            end_on_landing: false,
            truncate_on_ball_landing: 0.0,
        }
    }

    /// Statistics added to a total and started again in place are, take after take, exactly what
    /// `take_stats` gives (rewards, play and episodes, in the same order); the environment's statistics are then
    /// as new, in the same memory.
    #[test]
    fn statistics_taken_in_place_are_those_take_stats_gives() {
        if !meshes_loaded() {
            return;
        }
        let mut config = pool_config(1, 1, 1, 9);
        config.terminals.no_touch_seconds = 1.0;
        let rules = Arc::new(config.rules().unwrap());
        let shape = BufferShape {
            agents: 2,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        };
        let (mut a, mut b) = (OwnedBuffers::new(shape), OwnedBuffers::new(shape));
        let terms = rules.rewards.len();
        let (mut in_place, mut replaced) =
            (Env::new(Arc::clone(&rules), 0, 3), Env::new(rules, 0, 3));
        let mut rng = fastrand::Rng::with_seed(4);
        // As text, so NaNs (never touched, never succeeded) compare too.
        let text = |stats: &EnvStats| format!("{stats:?}");
        let untimed = |mut stats: EnvStats| {
            (stats.step_time, stats.physics_time) = (Duration::ZERO, Duration::ZERO);
            text(&stats)
        };
        let mut episodes = 0;
        for take in 0..8 {
            for _ in 0..70 {
                let actions: Vec<i32> = (0..2).map(|_| rng.i32(0..NUM_ACTIONS as i32)).collect();
                in_place.step(&actions, &mut rows(&mut a));
                replaced.step(&actions, &mut rows(&mut b));
            }
            // What `take_stats` would hand over: this environment's statistics as they are.
            let mut wanted = EnvStats::new(terms);
            wanted.merge(&in_place.stats.clone());
            let memory = (
                in_place.stats.rewards.raw.as_ptr(),
                in_place.stats.episodes.as_ptr(),
            );
            let mut total = EnvStats::new(terms);
            in_place.add_stats_to(&mut total);
            assert_eq!(text(&total), text(&wanted), "take {take}");
            assert_eq!(
                text(&in_place.stats),
                text(&in_place.fresh_stats()),
                "take {take}"
            );
            assert_eq!(
                (
                    in_place.stats.rewards.raw.as_ptr(),
                    in_place.stats.episodes.as_ptr()
                ),
                memory,
                "take {take}: in place"
            );
            // The same games' statistics the old way, from a twin (their step times differ).
            let mut old = EnvStats::new(terms);
            old.merge(&replaced.take_stats());
            assert_eq!(untimed(total.clone()), untimed(old), "take {take}");
            episodes += total.episodes.len();
        }
        assert!(episodes > 3, "{episodes} episodes");
    }

    fn kickoff_car(x: f32, y: f32, team: Team) -> CarSnapshot {
        CarSnapshot {
            team,
            body: Body {
                pos: Vec3A::new(x, y, 17.0),
                ..Body::RESTING
            },
            ..CarSnapshot::default()
        }
    }

    /// RocketSim's five kickoff spots, Blue's and Orange's (mirrored through the centre), each moved up to
    /// the kickoff setter's largest fuzz (300 uu) in sixteen directions, keep their class.
    #[test]
    fn every_kickoff_spot_is_classed_on_both_teams_through_the_fuzz() {
        use rocketsim::GameMode;
        use rocketsim::consts::car::spawn::get_kickoff_spawn_locations;
        let spots = get_kickoff_spawn_locations(GameMode::Soccar);
        assert_eq!(spots.len(), 5);
        let mut seen = [0; 3];
        for spot in spots {
            let class = match spot.x.abs() {
                2048.0 => KICKOFF_DIAGONAL,
                256.0 => KICKOFF_OFF_CENTRE,
                0.0 => KICKOFF_BACK,
                other => panic!("a kickoff spot at x {other}"),
            };
            seen[class as usize] += 1;
            for team in [Team::Blue, Team::Orange] {
                let side = if team == Team::Blue { 1.0 } else { -1.0 };
                for turn in 0..16 {
                    let angle = turn as f32 * std::f32::consts::TAU / 16.0;
                    for fuzz in [0.0, 150.0, 300.0] {
                        let (x, y) = (
                            side * (spot.x + fuzz * angle.cos()),
                            side * (spot.y + fuzz * angle.sin()),
                        );
                        assert_eq!(
                            kickoff_class(&[kickoff_car(x, y, team)]),
                            class,
                            "{team:?} at ({x}, {y})"
                        );
                    }
                }
            }
        }
        assert_eq!(seen, [2, 2, 1], "two diagonal, two off-centre, one back");
        // 2v2: the most forward spot taken (each team has the same spots).
        let two = [
            kickoff_car(0.0, -4608.0, Team::Blue),
            kickoff_car(-256.0, -3840.0, Team::Blue),
            kickoff_car(0.0, 4608.0, Team::Orange),
            kickoff_car(256.0, 3840.0, Team::Orange),
        ];
        assert_eq!(kickoff_class(&two), KICKOFF_OFF_CENTRE);
        assert_eq!(kickoff_class(&two[..1]), KICKOFF_BACK);
        assert_eq!(
            kickoff_class(&[kickoff_car(1000.0, -1000.0, Team::Blue)]),
            -1
        );
        assert_eq!(
            kickoff_class(&[kickoff_car(1000.0, -3200.0, Team::Blue)]),
            -1,
            "between spots"
        );
        assert_eq!(kickoff_class(&[]), -1);
    }

    /// Real kickoffs in 1v1, 2v2 and 3v3, exact and with the largest fuzz: every start gets the class of
    /// the car nearest the ball (exact) or some class (fuzz), every class that can turns up, and other
    /// setters get none.
    #[test]
    fn kickoff_starts_carry_their_class() {
        if !meshes_loaded() {
            return;
        }
        for team_size in 1..=3 {
            for fuzz_uu in [0.0, 300.0] {
                let mut config = pool_config(1, 1, 1, 11);
                config.team_size = team_size;
                config.state_setters[0].kind = SetterKind::Kickoff { fuzz_uu };
                let rules = Arc::new(config.rules().unwrap());
                let mut env = Env::new(Arc::clone(&rules), 0, 13);
                let mut seen = [0; 3];
                for _ in 0..200 {
                    env.start_episode();
                    let class = env.tracker.kickoff;
                    if !matches!(
                        rules.setters.kind(usize::from(env.setter)),
                        SetterKind::Kickoff { .. }
                    ) {
                        assert_eq!(class, -1);
                        continue;
                    }
                    assert!((0..3).contains(&class), "{team_size}v{team_size}: {class}");
                    seen[class as usize] += 1;
                    if fuzz_uu == 0.0 {
                        let nearest = env
                            .snap
                            .cars
                            .iter()
                            .map(|car| car.body.pos.distance(env.snap.ball.pos))
                            .fold(f32::INFINITY, f32::min);
                        let wanted = [3278.0, 3848.0, 4608.0]
                            .iter()
                            .position(|d| (nearest - d).abs() < 50.0)
                            .unwrap() as i8;
                        assert_eq!(class, wanted, "{team_size}v{team_size}");
                    }
                }
                // With two or three cars a side on distinct spots, one of them is always further forward
                // than the straight-back spot.
                let back = if team_size == 1 {
                    seen[2] > 0
                } else {
                    seen[2] == 0
                };
                assert!(
                    seen[0] > 0 && seen[1] > 0 && back,
                    "{team_size}v{team_size}: {seen:?}"
                );
            }
        }
    }

    /// A finished kickoff episode's records all carry its class and when the ball was first touched by
    /// any car (the earliest of the cars' own first touches).
    #[test]
    fn kickoff_records_carry_the_class_and_the_first_touch_by_any_car() {
        if !meshes_loaded() {
            return;
        }
        let mut config = pool_config(1, 1, 1, 3);
        config.terminals.no_touch_seconds = 6.0;
        config.state_setters[0].kind = SetterKind::Kickoff { fuzz_uu: 300.0 };
        let rules = Arc::new(config.rules().unwrap());
        let boost = (0..NUM_ACTIONS)
            .find(|&i| {
                let c = rules.actions.controls(i);
                c.throttle == 1.0 && c.steer == 0.0 && c.boost && !c.jump && !c.handbrake
            })
            .unwrap() as i32;
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 2,
            obs_size: rules.obs_size(),
            num_actions: rules.actions.len(),
        });
        let mut env = Env::new(Arc::clone(&rules), 0, 5);
        let mut records = Vec::new();
        for _ in 0..8000 {
            env.step(&[boost, boost], &mut rows(&mut owned));
            records.extend(env.take_stats().episodes);
        }
        let (mut kickoffs, mut touched) = (0, 0);
        for episode in records.chunks(2) {
            assert_eq!((episode[0].car, episode[1].car), (0, 1));
            assert_eq!(episode[0].kickoff, episode[1].kickoff);
            let first = episode[0].first_touch.min(episode[1].first_touch);
            for record in episode {
                assert!(
                    record.ball_first_touch == first
                        || (record.ball_first_touch.is_nan() && first.is_nan()),
                    "{episode:?}"
                );
            }
            if episode[0].setter == 0 {
                assert!((0..3).contains(&episode[0].kickoff), "{episode:?}");
                kickoffs += 1;
                touched += usize::from(!first.is_nan());
            } else {
                assert_eq!(episode[0].kickoff, -1);
            }
        }
        assert!(
            kickoffs > 10 && touched > 2,
            "{kickoffs} kickoffs, {touched} touched"
        );
    }
}
