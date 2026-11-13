//! The built-in rewards, chosen and weighted in the settings file.
//!
//! Seven follow RLGym's common rewards (RLGym 1.2.2, `rlgym/utils/reward_functions/common_rewards/`,
//! Copyright 2021 Lucas Emery; rlgym-sim has the same files), licensed under the Apache License, Version 2.0
//! (http://www.apache.org/licenses/LICENSE-2.0; the full text is in LICENSES/Apache-2.0.txt):
//! `face_ball` (FaceBallReward), `velocity_player_to_ball` (VelocityPlayerToBallReward) and `touch_ball`
//! (TouchBallReward) in `player_ball_rewards.py`;
//! `velocity_ball_to_goal` (VelocityBallToGoalReward) in `ball_goal_rewards.py`; `save_boost`
//! (SaveBoostReward), `demo` and `goal` (EventReward's demo, team_goal and concede) in `misc_rewards.py`.
//! Modified: written in Rust; a direction of zero length gives 0, not NaN; touch_ball has no aerial weight;
//! save_boost's power is a setting (0.5, the original's square root, by default); demo and goal pay on the
//! step of the event, not on a change in the match's counters.
//! The others (`air`, `strong_touch`, `pickup_boost`, `bump`, `alignment`, `all_of`) and the zero-sum sharing
//! are NitroLearn's own; each one's formula is written beside it.

use rocketsim::Vec3A;
use serde::Deserialize;

use crate::snapshot::{GameSnapshot, StepEvents, Team};

const CAR_MAX_SPEED: f32 = 2300.0;
const BALL_MAX_SPEED: f32 = 6000.0;
const BACK_NET_Y: f32 = 6000.0;
const GOAL_HEIGHT: f32 = 642.775;
// Rewards paid every step are defined per 8-tick step and scaled to the actual step.
const REFERENCE_TICK_SKIP: f32 = 8.0;

fn kph_to_uu(kph: f32) -> f32 {
    kph * 250.0 / 9.0
}

// Variants without parameters are written `Air {}`: serde ignores unknown keys on unit variants of a
// tagged enum, and a misspelled parameter must be an error, not a silent default.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum RewardKind {
    /// Every step the car is off the ground (no wheel on the floor, a wall or the ceiling): 1, else 0.
    Air {},
    /// Every step: the cosine between the car's nose and the direction from the car to the ball, -1 to 1
    /// (RLGym's FaceBallReward).
    FaceBall {},
    /// Every step: the car's velocity along the direction to the ball over the car's top speed (2300 uu/s),
    /// -1 to 1 (RLGym's VelocityPlayerToBallReward).
    VelocityPlayerToBall {},
    /// Every step: the ball's velocity along the direction to the back of the opponents' net, half the goal's
    /// height up, over the ball's top speed (6000 uu/s), -1 to 1 (RLGym's VelocityBallToGoalReward).
    VelocityBallToGoal {},
    /// 1 on a step the car touches the ball (RLGym's TouchBallReward).
    TouchBall {},
    /// On a step the car touches the ball: how hard, as the change in the ball's velocity over the step on a
    /// straight ramp, 0 at `min_speed_kph` (and below) to 1 at `max_speed_kph` (and above).
    StrongTouch {
        min_speed_kph: f32,
        max_speed_kph: f32,
    },
    /// When the car's boost goes up: the rise in √(boost / 100) over the step, so the same pad is worth more
    /// to an emptier tank (a big pad from empty pays 1, a small pad from empty about 0.35). rlgym-tools'
    /// BoostChangeReward gives the same with a gain weight of 1 and a lose weight of 0.
    PickupBoost {},
    /// Every step: (boost / 100) to the power `exponent` (RLGym's SaveBoostReward, √ boost, at 0.5).
    SaveBoost { exponent: f32 },
    /// 1 on a step the car bumps an opponent.
    Bump {},
    /// 1 on a step the car demolishes an opponent (RLGym's EventReward, demo).
    Demo {},
    /// On a goal: 1 to every car of the scoring team, `concede_scale` to every car of the other (RLGym's
    /// EventReward, team_goal and concede).
    Goal { concede_scale: f32 },
    /// Every step: the cosine between the direction from the car to the ball and the direction from the ball
    /// to the back of the opponents' net: 1 with the ball straight between the car and that net, -1 with the
    /// car straight between them.
    Alignment {},
    /// The "AND" of its parts (the KRC of the Lucy-SKG paper): the geometric mean of the parts when every part
    /// is positive; 0 when any part is 0; otherwise minus the geometric mean of their sizes.
    AllOf { parts: Vec<RewardKind> },
    /// A reward of one's own, compiled in from a Rust file: `kind` names it, `settings` holds its own
    /// settings.
    Custom(crate::custom::CustomReward),
}

impl RewardKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Air {} => "air",
            Self::FaceBall {} => "face_ball",
            Self::VelocityPlayerToBall {} => "velocity_player_to_ball",
            Self::VelocityBallToGoal {} => "velocity_ball_to_goal",
            Self::TouchBall {} => "touch_ball",
            Self::StrongTouch { .. } => "strong_touch",
            Self::PickupBoost {} => "pickup_boost",
            Self::SaveBoost { .. } => "save_boost",
            Self::Bump {} => "bump",
            Self::Demo {} => "demo",
            Self::Goal { .. } => "goal",
            Self::Alignment {} => "alignment",
            Self::AllOf { .. } => "all_of",
            Self::Custom(custom) => custom.name(),
        }
    }

    /// Paid every step (and so scaled to the tick skip); a group is, when every part is.
    fn is_continuous(&self) -> bool {
        match self {
            Self::Air {}
            | Self::FaceBall {}
            | Self::VelocityPlayerToBall {}
            | Self::VelocityBallToGoal {}
            | Self::SaveBoost { .. }
            | Self::Alignment {} => true,
            Self::AllOf { parts } => parts.iter().all(Self::is_continuous),
            Self::Custom(custom) => custom.every_step(),
            _ => false,
        }
    }

    /// How many numbers per car the reward keeps between steps (only rewards of one's own do).
    fn memory(&self) -> usize {
        match self {
            Self::Custom(custom) => custom.memory(),
            _ => 0,
        }
    }

    /// What is wrong with the kind's own parameters, if anything.
    fn problem(&self) -> Option<&'static str> {
        match self {
            Self::StrongTouch {
                min_speed_kph,
                max_speed_kph,
            } if !((0.0..=f32::MAX).contains(min_speed_kph)
                && max_speed_kph.is_finite()
                && max_speed_kph > min_speed_kph) =>
            {
                Some("strong_touch needs 0 <= min_speed_kph < max_speed_kph")
            }
            Self::SaveBoost { exponent } if !(exponent.is_finite() && *exponent > 0.0) => {
                Some("save_boost exponent must be above 0")
            }
            Self::Goal { concede_scale } if !concede_scale.is_finite() => {
                Some("goal concede_scale must be a finite number")
            }
            Self::AllOf { parts } if parts.len() < 2 => Some("all_of needs at least two parts"),
            Self::AllOf { parts } if parts.iter().any(|p| matches!(p, Self::AllOf { .. })) => {
                Some("an all_of group cannot hold another all_of group")
            }
            Self::AllOf { parts } if parts.iter().any(|p| p.memory() > 0) => {
                Some("a reward that remembers from step to step cannot be part of an all_of group")
            }
            Self::AllOf { parts } => parts.iter().find_map(Self::problem),
            _ => None,
        }
    }

    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        let happened = &events.cars[car];
        let to_ball = (cur.ball.pos - me.body.pos).normalize_or_zero();
        match *self {
            Self::Air {} => {
                if me.on_ground {
                    0.0
                } else {
                    1.0
                }
            }
            Self::FaceBall {} => me.body.forward().dot(to_ball),
            Self::VelocityPlayerToBall {} => to_ball.dot(me.body.vel / CAR_MAX_SPEED),
            Self::VelocityBallToGoal {} => {
                let goal_y = if me.team == Team::Blue {
                    BACK_NET_Y
                } else {
                    -BACK_NET_Y
                };
                let to_goal =
                    (Vec3A::new(0.0, goal_y, GOAL_HEIGHT / 2.0) - cur.ball.pos).normalize_or_zero();
                to_goal.dot(cur.ball.vel / BALL_MAX_SPEED)
            }
            Self::TouchBall {} => f32::from(u8::from(happened.touched_ball)),
            Self::StrongTouch {
                min_speed_kph,
                max_speed_kph,
            } => {
                if !happened.touched_ball {
                    return 0.0;
                }
                // How hard the touch was: the change in the ball's velocity over the step (uu/s).
                let hit = (cur.ball.vel - prev.ball.vel).length();
                let (low, high) = (kph_to_uu(min_speed_kph), kph_to_uu(max_speed_kph));
                ((hit - low) / (high - low)).clamp(0.0, 1.0)
            }
            Self::PickupBoost {} => {
                let tank = |boost: f32| (boost / 100.0).clamp(0.0, 1.0).sqrt();
                (tank(me.boost) - tank(prev.cars[car].boost)).max(0.0)
            }
            Self::SaveBoost { exponent } => (me.boost / 100.0).powf(exponent).clamp(0.0, 1.0),
            Self::Bump {} => f32::from(u8::from(happened.bumped_opponent)),
            Self::Demo {} => f32::from(u8::from(happened.demoed_opponent)),
            Self::Goal { concede_scale } => match cur.goal_scored {
                Some(team) if team == me.team => 1.0,
                Some(_) => concede_scale,
                None => 0.0,
            },
            Self::Alignment {} => {
                let goal_y = if me.team == Team::Blue {
                    BACK_NET_Y
                } else {
                    -BACK_NET_Y
                };
                let ball_to_goal =
                    (Vec3A::new(0.0, goal_y, GOAL_HEIGHT / 2.0) - cur.ball.pos).normalize_or_zero();
                to_ball.dot(ball_to_goal)
            }
            Self::AllOf { ref parts } => {
                krc(parts.iter().map(|part| part.value(prev, cur, events, car)))
            }
            Self::Custom(ref custom) => custom.value(prev, cur, events, car),
        }
    }
}

/// The KRC of the Lucy-SKG paper: the parts combined as an "AND". Positive only when every part is positive (then their
/// geometric mean); 0 when any part is 0; otherwise minus the geometric mean of their sizes. One pass, no
/// allocation (it runs for every car on every step).
fn krc(values: impl IntoIterator<Item = f32>) -> f32 {
    let (mut count, mut log_sum, mut negative) = (0u32, 0.0f32, false);
    for value in values {
        if value == 0.0 {
            return 0.0;
        }
        count += 1;
        log_sum += value.abs().ln();
        negative |= value < 0.0;
    }
    if count == 0 {
        return 0.0;
    }
    let magnitude = (log_sum / count as f32).exp();
    if negative { -magnitude } else { magnitude }
}

/// Shares a reward with teammates and subtracts what the opponents earn, so a reward that helps nobody
/// cannot be farmed: each car gets `own·(1 − team_spirit) + team_average·team_spirit −
/// opponent_average·opponent_scale`, the averages over the cars of each team (the "team spirit" of
/// OpenAI Five).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZeroSum {
    pub team_spirit: f32,
    pub opponent_scale: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardTerm {
    pub kind: RewardKind,
    pub weight: f32,
    pub zero_sum: Option<ZeroSum>,
}

impl RewardTerm {
    /// What is wrong with these values, if anything (the Python loader checks the same, with line numbers).
    pub fn problem(&self) -> Option<&'static str> {
        let within = |value: f32, low: f32, high: f32| (low..=high).contains(&value);
        let finite = |value: f32| within(value, f32::MIN, f32::MAX);
        if !finite(self.weight) {
            return Some("reward weights must be finite numbers");
        }
        if let Some(zs) = self.zero_sum
            && !(within(zs.team_spirit, 0.0, 1.0) && within(zs.opponent_scale, 0.0, f32::MAX))
        {
            return Some("zero_sum needs team_spirit 0 to 1 and opponent_scale 0 or more");
        }
        self.kind.problem()
    }
}

/// Sums over every agent-step since the last read, per term.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewardStats {
    pub agent_steps: u64,
    /// Before zero-sum and weight (per-step terms already scaled to the tick skip): what the agents
    /// are actually achieving.
    pub raw: Vec<f64>,
    /// After zero-sum and weight: what the learner is paid.
    pub weighted: Vec<f64>,
    pub weighted_abs: Vec<f64>,
    /// Weighted but before zero-sum, by size: what the agents earn (or are charged) themselves, before
    /// teammates' and opponents' pay is mixed in (the dashboard shows it beside `weighted_abs`).
    pub own_abs: Vec<f64>,
    /// Agent steps on which the term's value before zero-sum and weight was non-zero: how often it fires.
    pub fired: Vec<u64>,
}

impl RewardStats {
    pub fn new(terms: usize) -> Self {
        Self {
            agent_steps: 0,
            raw: vec![0.0; terms],
            weighted: vec![0.0; terms],
            weighted_abs: vec![0.0; terms],
            own_abs: vec![0.0; terms],
            fired: vec![0; terms],
        }
    }

    /// Back to `RewardStats::new(terms)`, keeping the vectors' memory.
    pub fn reset(&mut self, terms: usize) {
        let Self {
            agent_steps,
            raw,
            weighted,
            weighted_abs,
            own_abs,
            fired,
        } = self;
        *agent_steps = 0;
        for sums in [raw, weighted, weighted_abs, own_abs] {
            sums.clear();
            sums.resize(terms, 0.0);
        }
        fired.clear();
        fired.resize(terms, 0);
    }

    pub fn merge(&mut self, other: &Self) {
        self.agent_steps += other.agent_steps;
        for (mine, theirs) in [
            (&mut self.raw, &other.raw),
            (&mut self.weighted, &other.weighted),
            (&mut self.weighted_abs, &other.weighted_abs),
            (&mut self.own_abs, &other.own_abs),
        ] {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a += b;
            }
        }
        for (a, b) in self.fired.iter_mut().zip(&other.fired) {
            *a += b;
        }
    }
}

#[derive(Clone)]
pub struct RewardSet {
    terms: Vec<RewardTerm>,
    step_scale: f32,
    raw: Vec<f32>,
    /// Per term, the numbers each car's reward keeps between steps (rewards that remember).
    memory: Vec<Vec<f32>>,
}

impl RewardSet {
    pub fn new(terms: Vec<RewardTerm>, tick_skip: u32) -> Self {
        let memory = vec![Vec::new(); terms.len()];
        Self {
            terms,
            step_scale: tick_skip as f32 / REFERENCE_TICK_SKIP,
            raw: Vec::new(),
            memory,
        }
    }

    /// A new episode: what the rewards remember starts again from 0.
    pub fn reset(&mut self) {
        for memory in &mut self.memory {
            memory.fill(0.0);
        }
    }

    /// Writes every car's reward for the step into `out`; only cars with `counted[car]` enter `stats` (the
    /// frozen opponents of self-play are simulated and paid like anyone else but are nobody's learning
    /// progress).
    pub fn compute(
        &mut self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        counted: &[bool],
        out: &mut [f32],
        stats: &mut RewardStats,
    ) {
        let cars = cur.cars.len();
        debug_assert_eq!(counted.len(), cars);
        out.fill(0.0);
        stats.agent_steps += counted.iter().filter(|&&c| c).count() as u64;
        for (t, term) in self.terms.iter().enumerate() {
            let scale = if term.kind.is_continuous() {
                self.step_scale
            } else {
                1.0
            };
            let weight = term.weight;
            self.raw.clear();
            if let (RewardKind::Custom(custom), size @ 1..) = (&term.kind, term.kind.memory()) {
                let memory = &mut self.memory[t];
                memory.resize(cars * size, 0.0);
                self.raw.extend(
                    memory
                        .chunks_exact_mut(size)
                        .enumerate()
                        .map(|(car, kept)| {
                            custom.value_remembering(prev, cur, events, car, kept) * scale
                        }),
                );
            } else {
                self.raw
                    .extend((0..cars).map(|car| term.kind.value(prev, cur, events, car) * scale));
            }
            let mut team_sum = [0.0f32; 2];
            let mut team_count = [0u32; 2];
            for (car, &value) in self.raw.iter().enumerate() {
                let team = cur.cars[car].team.index();
                team_sum[team] += value;
                team_count[team] += 1;
            }
            let team_avg = |team: usize| team_sum[team] / team_count[team].max(1) as f32;
            for (car, &value) in self.raw.iter().enumerate() {
                let shaped = match term.zero_sum {
                    None => value,
                    Some(zs) => {
                        let team = cur.cars[car].team.index();
                        let spirit = zs.team_spirit;
                        value * (1.0 - spirit) + team_avg(team) * spirit
                            - team_avg(1 - team) * zs.opponent_scale
                    }
                };
                let weighted = shaped * weight;
                out[car] += weighted;
                if !counted[car] {
                    continue;
                }
                stats.raw[t] += f64::from(value);
                stats.weighted[t] += f64::from(weighted);
                stats.weighted_abs[t] += f64::from(weighted.abs());
                stats.own_abs[t] += f64::from((value * weight).abs());
                stats.fired[t] += u64::from(value != 0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{Body, CarEvents, CarSnapshot};

    fn game(ball_y: f32) -> GameSnapshot {
        let car = |team| CarSnapshot {
            team,
            boost: 50.0,
            ..CarSnapshot::default()
        };
        GameSnapshot {
            ball: Body {
                pos: Vec3A::new(0.0, ball_y, 93.0),
                ..Body::RESTING
            },
            cars: vec![car(Team::Blue), car(Team::Orange)],
            ..GameSnapshot::default()
        }
    }

    fn rewards(
        terms: Vec<RewardTerm>,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
    ) -> (Vec<f32>, RewardStats) {
        let mut set = RewardSet::new(terms, 8);
        let mut out = vec![0.0; cur.cars.len()];
        let mut stats = RewardStats::new(set.terms.len());
        let counted = vec![true; cur.cars.len()];
        set.compute(prev, cur, events, &counted, &mut out, &mut stats);
        (out, stats)
    }

    /// Counts its car's touches this episode and pays the count on each touch (a reward's memory).
    struct Touches;

    impl crate::custom::Reward for Touches {
        fn memory(&self) -> usize {
            1
        }

        fn value_remembering(
            &self,
            _: &GameSnapshot,
            _: &GameSnapshot,
            events: &StepEvents,
            car: usize,
            memory: &mut [f32],
        ) -> f32 {
            if !events.cars[car].touched_ball {
                return 0.0;
            }
            memory[0] += 1.0;
            memory[0]
        }
    }

    #[test]
    fn a_reward_remembers_per_car_until_the_episode_starts_again() {
        let kind = RewardKind::Custom(crate::custom::CustomReward::of("touches", Touches));
        let mut set = RewardSet::new(vec![plain(kind.clone(), 1.0)], 8);
        let g = game(0.0);
        let mut touched = no_events(2);
        touched.cars[0].touched_ball = true;
        let mut both = touched.clone();
        both.cars[1].touched_ball = true;
        let mut out = vec![0.0; 2];
        let mut stats = RewardStats::new(1);
        let mut paid = |set: &mut RewardSet, events: &StepEvents| {
            set.compute(&g, &g, events, &[true, true], &mut out, &mut stats);
            out.clone()
        };
        assert_eq!(paid(&mut set, &touched), vec![1.0, 0.0]);
        assert_eq!(paid(&mut set, &no_events(2)), vec![0.0, 0.0]);
        assert_eq!(
            paid(&mut set, &both),
            vec![2.0, 1.0],
            "each car its own count"
        );
        set.reset();
        assert_eq!(
            paid(&mut set, &touched),
            vec![1.0, 0.0],
            "a new episode counts from 0"
        );
        let group = plain(
            RewardKind::AllOf {
                parts: vec![kind, RewardKind::Air {}],
            },
            1.0,
        );
        assert!(group.problem().unwrap().contains("remembers"));
    }

    fn no_events(cars: usize) -> StepEvents {
        StepEvents {
            cars: vec![CarEvents::default(); cars],
            ..StepEvents::default()
        }
    }

    fn plain(kind: RewardKind, weight: f32) -> RewardTerm {
        RewardTerm {
            kind,
            weight,
            zero_sum: None,
        }
    }

    #[test]
    fn alignment_is_the_cosine_between_car_to_ball_and_ball_to_goal() {
        // Blue attacks +Y. Ball at the centre; a car behind it (-Y) lines up, one beyond it does not.
        let mut cur = game(0.0);
        cur.ball.pos = Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0);
        cur.cars[0].body.pos = Vec3A::new(0.0, -1000.0, GOAL_HEIGHT / 2.0);
        cur.cars[1].body.pos = Vec3A::new(0.0, -1000.0, GOAL_HEIGHT / 2.0); // Orange, attacking -Y
        let (out, _) = rewards(
            vec![plain(RewardKind::Alignment {}, 1.0)],
            &cur,
            &cur,
            &no_events(2),
        );
        assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}");
        assert!((out[1] + 1.0).abs() < 1e-6, "{out:?}");
        cur.cars[0].body.pos = Vec3A::new(1000.0, 0.0, GOAL_HEIGHT / 2.0); // beside the ball
        let (out, _) = rewards(
            vec![plain(RewardKind::Alignment {}, 1.0)],
            &cur,
            &cur,
            &no_events(2),
        );
        assert!(out[0].abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn all_of_is_the_geometric_mean_positive_only_when_every_part_is() {
        assert!((krc([0.5, 0.5]) - 0.5).abs() < 1e-6);
        assert!((krc([0.25, 1.0]) - 0.5).abs() < 1e-6, "geometric mean");
        assert!(
            (krc([0.25, -1.0]) + 0.5).abs() < 1e-6,
            "one bad part makes it negative"
        );
        assert!(
            (krc([-0.25, -1.0]) + 0.5).abs() < 1e-6,
            "two bad parts stay negative"
        );
        assert_eq!(krc([0.5, 0.0]), 0.0);
        assert_eq!(krc([0.5, -0.0]), 0.0, "minus zero is zero");
        let tiny = krc([1e-30, 1e-30]);
        assert!(
            tiny > 0.0 && (tiny - 1e-30).abs() < 1e-34,
            "small parts do not underflow: {tiny}"
        );
        assert_eq!(krc([0.0f32; 0]), 0.0);
        // In a game: a car driving at the ball while lined up with the goal earns the AND of both.
        let mut cur = game(0.0);
        cur.ball.pos = Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0);
        cur.cars[0].body.pos = Vec3A::new(0.0, -1000.0, GOAL_HEIGHT / 2.0);
        cur.cars[0].body.vel = Vec3A::new(0.0, CAR_MAX_SPEED / 4.0, 0.0);
        let group = RewardKind::AllOf {
            parts: vec![
                RewardKind::VelocityPlayerToBall {},
                RewardKind::Alignment {},
            ],
        };
        let (out, stats) = rewards(vec![plain(group.clone(), 2.0)], &cur, &cur, &no_events(2));
        assert!(
            (out[0] - 2.0 * 0.5).abs() < 1e-6,
            "{out:?}: sqrt(0.25 * 1) weighted by 2"
        );
        assert!(
            (stats.raw[0] - 0.5).abs() < 1e-6,
            "the raw value is logged before the weight"
        );
        // Every part paid per step: the group is scaled to the tick skip like them.
        let mut set = RewardSet::new(vec![plain(group.clone(), 1.0)], 16);
        let mut out = vec![0.0; 2];
        let mut stats = RewardStats::new(1);
        set.compute(
            &cur,
            &cur,
            &no_events(2),
            &[true, true],
            &mut out,
            &mut stats,
        );
        assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}: 0.5 scaled by 16/8");
        // With an event part the group is an event: unscaled, and 0 without the event.
        let touch_group = RewardKind::AllOf {
            parts: vec![RewardKind::TouchBall {}, RewardKind::Alignment {}],
        };
        let (out, _) = rewards(vec![plain(touch_group, 1.0)], &cur, &cur, &no_events(2));
        assert_eq!(out[0], 0.0);
        assert_eq!(plain(group, 1.0).problem(), None);
        assert!(
            plain(RewardKind::AllOf { parts: vec![] }, 1.0)
                .problem()
                .unwrap()
                .contains("two parts")
        );
        let nested = RewardKind::AllOf {
            parts: vec![
                RewardKind::Alignment {},
                RewardKind::AllOf {
                    parts: vec![RewardKind::Air {}, RewardKind::FaceBall {}],
                },
            ],
        };
        assert!(
            plain(nested, 1.0)
                .problem()
                .unwrap()
                .contains("another all_of")
        );
        let bad_part = RewardKind::AllOf {
            parts: vec![
                RewardKind::Alignment {},
                RewardKind::SaveBoost { exponent: -1.0 },
            ],
        };
        assert!(
            plain(bad_part, 1.0)
                .problem()
                .unwrap()
                .contains("save_boost")
        );
    }

    #[test]
    fn goal_rewards_the_scorers_and_charges_the_conceders() {
        let mut cur = game(5300.0);
        cur.goal_scored = Some(Team::Blue);
        let term = RewardTerm {
            kind: RewardKind::Goal {
                concede_scale: -0.8,
            },
            weight: 150.0,
            zero_sum: None,
        };
        let (out, _) = rewards(vec![term], &game(0.0), &cur, &no_events(2));
        assert_eq!(out, vec![150.0, -120.0]);
    }

    #[test]
    fn zero_sum_subtracts_the_opponents_and_logs_the_raw_value() {
        let mut events = no_events(2);
        events.cars[0].bumped_opponent = true;
        let term = RewardTerm {
            kind: RewardKind::Bump {},
            weight: 20.0,
            zero_sum: Some(ZeroSum {
                team_spirit: 0.5,
                opponent_scale: 1.0,
            }),
        };
        let (out, stats) = rewards(vec![term], &game(0.0), &game(0.0), &events);
        // 1v1: own·0.5 + own·0.5 − opponent = 1 for the bumper, −1 for the victim; times 20.
        assert_eq!(out, vec![20.0, -20.0]);
        assert_eq!(stats.raw, vec![1.0]);
        assert_eq!(stats.weighted, vec![0.0]);
        assert_eq!(stats.weighted_abs, vec![40.0]);
        assert_eq!(
            stats.own_abs,
            vec![20.0],
            "before zero-sum only the bumper earned"
        );
        assert_eq!(
            stats.fired,
            vec![1],
            "the victim's zero-sum share is no trigger"
        );
    }

    #[test]
    fn strong_touch_scales_with_the_speed_change() {
        let prev = game(0.0);
        let mut cur = game(0.0);
        cur.ball.vel = Vec3A::new(0.0, kph_to_uu(50.0), 0.0);
        let mut events = no_events(2);
        events.cars[0].touched_ball = true;
        let term = RewardTerm {
            kind: RewardKind::StrongTouch {
                min_speed_kph: 20.0,
                max_speed_kph: 100.0,
            },
            weight: 1.0,
            zero_sum: None,
        };
        // A straight ramp from 20 km/h (0) to 100 km/h (1): 50 km/h is 30/80 of the way.
        let (out, _) = rewards(vec![term.clone()], &prev, &cur, &events);
        assert!((out[0] - 0.375).abs() < 1e-6 && out[1] == 0.0, "{out:?}");
        cur.ball.vel = Vec3A::new(0.0, kph_to_uu(10.0), 0.0);
        assert_eq!(rewards(vec![term.clone()], &prev, &cur, &events).0[0], 0.0);
        cur.ball.vel = Vec3A::new(0.0, kph_to_uu(20.0), 0.0);
        assert!(
            rewards(vec![term.clone()], &prev, &cur, &events).0[0].abs() < 1e-6,
            "no jump at the lower end"
        );
        cur.ball.vel = Vec3A::new(kph_to_uu(150.0), 0.0, 0.0);
        assert_eq!(rewards(vec![term.clone()], &prev, &cur, &events).0[0], 1.0);
        assert_eq!(
            rewards(vec![term], &prev, &cur, &no_events(2)).0[0],
            0.0,
            "no touch, no pay"
        );
    }

    fn value_of(
        kind: RewardKind,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
    ) -> Vec<f32> {
        rewards(vec![plain(kind, 1.0)], prev, cur, events).0
    }

    #[test]
    fn the_rlgym_rewards_follow_their_formulas() {
        let mut cur = game(0.0);
        cur.ball.pos = Vec3A::new(0.0, 0.0, 93.0);
        cur.cars[0].body.pos = Vec3A::new(0.0, -1000.0, 93.0); // Blue, behind the ball
        cur.cars[1].body.pos = Vec3A::new(1000.0, 0.0, 93.0); // Orange, beside it
        let events = no_events(2);
        // face_ball: cosine between the nose (+x when the car is unrotated) and the direction to the ball.
        let face = value_of(RewardKind::FaceBall {}, &cur, &cur, &events);
        assert!(
            face[0].abs() < 1e-6 && (face[1] + 1.0).abs() < 1e-6,
            "{face:?}"
        );
        // velocity_player_to_ball: velocity along the direction to the ball over 2300.
        cur.cars[0].body.vel = Vec3A::new(500.0, 1150.0, 0.0);
        let toward = value_of(RewardKind::VelocityPlayerToBall {}, &cur, &cur, &events);
        assert!((toward[0] - 0.5).abs() < 1e-6, "{toward:?}");
        // velocity_ball_to_goal: the ball's velocity towards the back of the net each team attacks, over 6000.
        cur.ball.pos = Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0);
        cur.ball.vel = Vec3A::new(0.0, 3000.0, 0.0);
        let goalward = value_of(RewardKind::VelocityBallToGoal {}, &cur, &cur, &events);
        assert!(
            (goalward[0] - 0.5).abs() < 1e-6 && (goalward[1] + 0.5).abs() < 1e-6,
            "{goalward:?}"
        );
        // touch_ball, demo and bump: 1 on the step of the event, for that car only.
        let mut happened = no_events(2);
        happened.cars[0].touched_ball = true;
        happened.cars[1].demoed_opponent = true;
        happened.cars[1].bumped_opponent = true;
        assert_eq!(
            value_of(RewardKind::TouchBall {}, &cur, &cur, &happened),
            vec![1.0, 0.0]
        );
        assert_eq!(
            value_of(RewardKind::Demo {}, &cur, &cur, &happened),
            vec![0.0, 1.0]
        );
        assert_eq!(
            value_of(RewardKind::Bump {}, &cur, &cur, &happened),
            vec![0.0, 1.0]
        );
        // save_boost: (boost / 100) ^ exponent; 0.5 is RLGym's square root.
        cur.cars[0].boost = 25.0;
        let saved = value_of(RewardKind::SaveBoost { exponent: 0.5 }, &cur, &cur, &events);
        assert!(
            (saved[0] - 0.5).abs() < 1e-6 && (saved[1] - 0.5f32.sqrt()).abs() < 1e-6,
            "{saved:?}"
        );
        // air: 1 while off the ground; a car exactly at the ball's centre faces nothing (0, not NaN).
        cur.cars[0].on_ground = true;
        cur.cars[1].on_ground = false;
        assert_eq!(
            value_of(RewardKind::Air {}, &cur, &cur, &events),
            vec![0.0, 1.0]
        );
        cur.cars[1].body.pos = cur.ball.pos;
        assert_eq!(
            value_of(RewardKind::FaceBall {}, &cur, &cur, &events)[1],
            0.0
        );
    }

    #[test]
    fn pickup_boost_uses_the_square_root_gain() {
        let prev = game(0.0);
        let mut cur = game(0.0);
        cur.cars[0].boost = 100.0;
        let term = RewardTerm {
            kind: RewardKind::PickupBoost {},
            weight: 1.0,
            zero_sum: None,
        };
        let (out, _) = rewards(vec![term.clone()], &prev, &cur, &no_events(2));
        assert!((out[0] - (1.0 - 0.5f32.sqrt())).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
        // A big pad from empty pays 1, a small one (12) about 0.35; spending boost pays nothing.
        let (mut empty, mut small) = (game(0.0), game(0.0));
        empty.cars[0].boost = 0.0;
        small.cars[0].boost = 12.0;
        let (out, _) = rewards(vec![term.clone()], &empty, &cur, &no_events(2));
        assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}");
        let (out, _) = rewards(vec![term.clone()], &empty, &small, &no_events(2));
        assert!((out[0] - 0.12f32.sqrt()).abs() < 1e-6, "{out:?}");
        let (out, _) = rewards(vec![term], &cur, &prev, &no_events(2));
        assert_eq!(out[0], 0.0, "boost spent");
    }

    #[test]
    fn continuous_rewards_scale_with_tick_skip() {
        let mut cur = game(0.0);
        cur.cars[0].on_ground = false;
        let term = RewardTerm {
            kind: RewardKind::Air {},
            weight: 1.0,
            zero_sum: None,
        };
        let mut set = RewardSet::new(vec![term], 4);
        let mut out = vec![0.0; 2];
        set.compute(
            &cur,
            &cur,
            &no_events(2),
            &[true, true],
            &mut out,
            &mut RewardStats::new(1),
        );
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn uncounted_cars_are_paid_but_leave_no_trace_in_the_statistics() {
        let mut cur = game(0.0);
        cur.cars[0].on_ground = false;
        cur.cars[1].on_ground = false;
        let term = RewardTerm {
            kind: RewardKind::Air {},
            weight: 2.0,
            zero_sum: None,
        };
        let mut set = RewardSet::new(vec![term], 8);
        let mut out = vec![0.0; 2];
        let mut stats = RewardStats::new(1);
        set.compute(
            &cur,
            &cur,
            &no_events(2),
            &[true, false],
            &mut out,
            &mut stats,
        );
        assert_eq!(out, vec![2.0, 2.0], "both cars are paid");
        assert_eq!(stats.agent_steps, 1);
        assert_eq!(
            (stats.raw[0], stats.weighted[0], stats.weighted_abs[0]),
            (1.0, 2.0, 2.0)
        );
    }

    #[test]
    fn terms_deserialize_from_settings_json() {
        let json = r#"[{"kind": {"name": "strong_touch", "min_speed_kph": 20, "max_speed_kph": 100}, "weight": 60, "zero_sum": null},
                       {"kind": {"name": "velocity_ball_to_goal"}, "weight": 2, "zero_sum": {"team_spirit": 1, "opponent_scale": 1}},
                       {"kind": {"name": "all_of", "parts": [{"name": "velocity_player_to_ball"}, {"name": "alignment"}]}, "weight": 1, "zero_sum": null}]"#;
        let terms: Vec<RewardTerm> = serde_json::from_str(json).unwrap();
        assert_eq!(
            terms[0].kind,
            RewardKind::StrongTouch {
                min_speed_kph: 20.0,
                max_speed_kph: 100.0
            }
        );
        assert_eq!(
            terms[2].kind,
            RewardKind::AllOf {
                parts: vec![
                    RewardKind::VelocityPlayerToBall {},
                    RewardKind::Alignment {}
                ]
            }
        );
        for typo in [
            r#"{"name": "air", "oops": 1}"#,
            r#"{"name": "save_boost", "exponent": 1, "oops": 1}"#,
        ] {
            let json = format!(r#"[{{"kind": {typo}, "weight": 1, "zero_sum": null}}]"#);
            assert!(
                serde_json::from_str::<Vec<RewardTerm>>(&json).is_err(),
                "{typo} was accepted"
            );
        }
    }
}
