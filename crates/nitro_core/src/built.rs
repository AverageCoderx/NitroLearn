//! Rewards built without code: a reward is a sum of terms, each term a set of conditions that must
//! all hold, the product (or the smallest) of one or more measures, and a weight. Read from the settings
//! (`name = "built"`), checked once, then interpreted over a fixed set of building blocks: nothing is
//! compiled. The block names here are the settings' names; `python/nitro/config/built.py` lists the same
//! ones with their help text.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use rocketsim::Vec3A;
use serde_json::Value;

use crate::custom::kit::{attack_goal, attack_sign, closest_to_ball, cosine, is_kickoff, kph};
use crate::custom::{CustomReward, Reward};
use expr::{Ctx, NEVER, Node, Remember, Scope, Stateful, Ty, Uses};

pub mod expr;
use crate::snapshot::{GameSnapshot, StepEvents};

const CAR_MAX_SPEED: f32 = 2300.0;
const BALL_MAX_SPEED: f32 = 6000.0;
const CEILING_Z: f32 = 2044.0;
const BACK_NET_Y: f32 = 6000.0;
const GOAL_HEIGHT: f32 = 642.775;
const GOAL_LINE_Y: f32 = 5120.0;
/// The field's length, goal line to goal line: `distance_to_ball` and `closeness_to_ball` are shares of it.
const FIELD_LENGTH: f32 = 10240.0;
const MAX_AIR_TIME: f32 = 1.75;
/// The step continuous rewards are written for (8 ticks), as `rewards::REFERENCE_TICK_SKIP`.
const REFERENCE_STEP_SECONDS: f32 = 8.0 / 120.0;
/// Up-vector z below which a car on its wheels is on a wall (or the ceiling) rather than the floor.
const WALL_UP_Z: f32 = 0.7;

/// A condition on car `car` at the step's end (or on what happened during the step).
#[derive(Clone, Debug, PartialEq)]
enum Cond {
    InAir,
    OnGround,
    OnFloor,
    OnWall,
    TouchingBall,
    HasFlip,
    Supersonic,
    Boosting,
    OwnHalf,
    OpponentHalf,
    BallOwnHalf,
    BallOpponentHalf,
    Kickoff,
    ClosestToBall,
    LastToTouch,
    BallAbove(f32),
    BallBelow(f32),
    CarAbove(f32),
    CarBelow(f32),
    BallCloserThan(f32),
    BallFartherThan(f32),
    /// Cosines, from the degrees written.
    FacingBallWithin(f32),
    FacingGoalWithin(f32),
    BoostAbove(f32),
    BoostBelow(f32),
    /// uu/s, from the km/h written.
    SpeedAbove(f32),
    SpeedBelow(f32),
    BallSpeedAbove(f32),
    BallSpeedBelow(f32),
    Not(Box<Cond>),
}

const FLAGS: &[(&str, Cond)] = &[
    ("in_air", Cond::InAir),
    ("on_ground", Cond::OnGround),
    ("on_floor", Cond::OnFloor),
    ("on_wall", Cond::OnWall),
    ("touching_ball", Cond::TouchingBall),
    ("has_flip", Cond::HasFlip),
    ("supersonic", Cond::Supersonic),
    ("boosting", Cond::Boosting),
    ("own_half", Cond::OwnHalf),
    ("opponent_half", Cond::OpponentHalf),
    ("ball_own_half", Cond::BallOwnHalf),
    ("ball_opponent_half", Cond::BallOpponentHalf),
    ("kickoff", Cond::Kickoff),
    ("closest_to_ball", Cond::ClosestToBall),
    ("last_to_touch", Cond::LastToTouch),
];

const NUMBERED: &[&str] = &[
    "ball_above",
    "ball_below",
    "car_above",
    "car_below",
    "ball_closer_than",
    "ball_farther_than",
    "facing_ball_within",
    "facing_goal_within",
    "boost_above",
    "boost_below",
    "speed_above",
    "speed_below",
    "ball_speed_above",
    "ball_speed_below",
];

impl Cond {
    fn parse(written: &Value) -> Result<Self, String> {
        match written {
            Value::String(name) => FLAGS
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, c)| c.clone())
                .ok_or_else(|| format!("there is no condition called '{name}'")),
            Value::Object(map) if map.len() == 1 => {
                let (key, value) = map.iter().next().expect("one entry");
                if key == "not" {
                    return Ok(Self::Not(Box::new(Self::parse(value)?)));
                }
                if !NUMBERED.contains(&key.as_str()) {
                    return Err(format!("there is no condition called '{key}'"));
                }
                // Finite as the core holds it, a 32-bit number (1e300 is not).
                let n = value
                    .as_f64()
                    .map(|n| n as f32)
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| format!("{key} needs a number, not {value}"))?;
                let degrees = |n: f32| {
                    if (0.0..=180.0).contains(&n) {
                        Ok(n.to_radians().cos())
                    } else {
                        Err(format!("{key} needs 0 to 180 degrees"))
                    }
                };
                Ok(match key.as_str() {
                    "ball_above" => Self::BallAbove(n),
                    "ball_below" => Self::BallBelow(n),
                    "car_above" => Self::CarAbove(n),
                    "car_below" => Self::CarBelow(n),
                    "ball_closer_than" => Self::BallCloserThan(n),
                    "ball_farther_than" => Self::BallFartherThan(n),
                    "facing_ball_within" => Self::FacingBallWithin(degrees(n)?),
                    "facing_goal_within" => Self::FacingGoalWithin(degrees(n)?),
                    "boost_above" => Self::BoostAbove(n),
                    "boost_below" => Self::BoostBelow(n),
                    "speed_above" => Self::SpeedAbove(kph(n)),
                    "speed_below" => Self::SpeedBelow(kph(n)),
                    "ball_speed_above" => Self::BallSpeedAbove(kph(n)),
                    _ => Self::BallSpeedBelow(kph(n)),
                })
            }
            _ => Err(format!(
                "a condition is a name in quotes or {{ name = number }}, not {written}"
            )),
        }
    }

    /// It depends on what happened during the step, not on the state alone.
    fn is_event(&self) -> bool {
        match self {
            Self::TouchingBall => true,
            Self::Not(inner) => inner.is_event(),
            _ => false,
        }
    }

    fn holds(&self, cur: &GameSnapshot, events: &StepEvents, car: usize) -> bool {
        let me = &cur.cars[car];
        let ball = &cur.ball;
        let sign = attack_sign(me.team);
        match *self {
            Self::InAir => !me.on_ground,
            Self::OnGround => me.on_ground,
            Self::OnFloor => me.on_ground && me.body.up().z >= WALL_UP_Z,
            Self::OnWall => me.on_ground && me.body.up().z < WALL_UP_Z,
            Self::TouchingBall => events.cars[car].touched_ball,
            Self::HasFlip => me.can_flip,
            Self::Supersonic => me.is_supersonic,
            Self::Boosting => events.detail(car).is_boosting,
            Self::OwnHalf => me.body.pos.y * sign < 0.0,
            Self::OpponentHalf => me.body.pos.y * sign > 0.0,
            Self::BallOwnHalf => ball.pos.y * sign < 0.0,
            Self::BallOpponentHalf => ball.pos.y * sign > 0.0,
            Self::Kickoff => is_kickoff(cur),
            Self::ClosestToBall => closest_to_ball(cur, me.team) == Some(car),
            Self::LastToTouch => events.last_touch == Some(car),
            Self::BallAbove(z) => ball.pos.z > z,
            Self::BallBelow(z) => ball.pos.z < z,
            Self::CarAbove(z) => me.body.pos.z > z,
            Self::CarBelow(z) => me.body.pos.z < z,
            Self::BallCloserThan(d) => me.body.pos.distance(ball.pos) < d,
            Self::BallFartherThan(d) => me.body.pos.distance(ball.pos) > d,
            Self::FacingBallWithin(cos) => cosine(me.body.forward(), ball.pos - me.body.pos) >= cos,
            Self::FacingGoalWithin(cos) => {
                cosine(me.body.forward(), attack_goal(me.team) - me.body.pos) >= cos
            }
            Self::BoostAbove(b) => me.boost > b,
            Self::BoostBelow(b) => me.boost < b,
            Self::SpeedAbove(v) => me.body.vel.length() > v,
            Self::SpeedBelow(v) => me.body.vel.length() < v,
            Self::BallSpeedAbove(v) => ball.vel.length() > v,
            Self::BallSpeedBelow(v) => ball.vel.length() < v,
            Self::Not(ref inner) => !inner.holds(cur, events, car),
        }
    }
}

/// What a term pays while its conditions hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Measure {
    // Of the state (paid every step, scaled to the tick skip).
    One,
    SpeedTowardBall,
    BallSpeedTowardGoal,
    CarSpeed,
    BallSpeed,
    CarHeight,
    BallHeight,
    Boost,
    FacingBall,
    Alignment,
    DistanceToBall,
    ClosenessToBall,
    AirTime,
    // Of the step's change.
    BoostUsed,
    BoostGained,
    BallSpeedChange,
    // One-off events (1 when they happen).
    Touch,
    Goal,
    Concede,
    Shot,
    Demo,
    Demoed,
    Bump,
    Jump,
    Flip,
}

const MEASURES: &[(&str, Measure)] = &[
    ("one", Measure::One),
    ("speed_toward_ball", Measure::SpeedTowardBall),
    ("ball_speed_toward_goal", Measure::BallSpeedTowardGoal),
    ("car_speed", Measure::CarSpeed),
    ("ball_speed", Measure::BallSpeed),
    ("car_height", Measure::CarHeight),
    ("ball_height", Measure::BallHeight),
    ("boost", Measure::Boost),
    ("facing_ball", Measure::FacingBall),
    ("alignment", Measure::Alignment),
    ("distance_to_ball", Measure::DistanceToBall),
    ("closeness_to_ball", Measure::ClosenessToBall),
    ("air_time", Measure::AirTime),
    ("boost_used", Measure::BoostUsed),
    ("boost_gained", Measure::BoostGained),
    ("ball_speed_change", Measure::BallSpeedChange),
    ("touch", Measure::Touch),
    ("goal", Measure::Goal),
    ("concede", Measure::Concede),
    ("shot", Measure::Shot),
    ("demo", Measure::Demo),
    ("demoed", Measure::Demoed),
    ("bump", Measure::Bump),
    ("jump", Measure::Jump),
    ("flip", Measure::Flip),
];

impl Measure {
    fn parse(written: &Value) -> Result<Self, String> {
        let name = written
            .as_str()
            .ok_or_else(|| format!("a measure is a name in quotes, not {written}"))?;
        MEASURES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, m)| m)
            .ok_or_else(|| format!("there is no measure called '{name}'"))
    }

    /// A measure of the state alone (paid every step).
    fn is_state(self) -> bool {
        (self as u8) <= (Self::AirTime as u8)
    }

    fn value(
        self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        let ball = &cur.ball;
        let to_ball = (ball.pos - me.body.pos).normalize_or_zero();
        let bit = |b: bool| f32::from(u8::from(b));
        match self {
            Self::One => 1.0,
            Self::SpeedTowardBall => to_ball.dot(me.body.vel / CAR_MAX_SPEED),
            Self::BallSpeedTowardGoal => {
                let net = Vec3A::new(0.0, attack_sign(me.team) * BACK_NET_Y, GOAL_HEIGHT / 2.0);
                (net - ball.pos)
                    .normalize_or_zero()
                    .dot(ball.vel / BALL_MAX_SPEED)
            }
            Self::CarSpeed => me.body.vel.length() / CAR_MAX_SPEED,
            Self::BallSpeed => ball.vel.length() / BALL_MAX_SPEED,
            Self::CarHeight => me.body.pos.z / CEILING_Z,
            Self::BallHeight => ball.pos.z / CEILING_Z,
            Self::Boost => me.boost / 100.0,
            Self::FacingBall => me.body.forward().dot(to_ball),
            Self::Alignment => {
                let net = Vec3A::new(0.0, attack_sign(me.team) * BACK_NET_Y, GOAL_HEIGHT / 2.0);
                to_ball.dot((net - ball.pos).normalize_or_zero())
            }
            Self::DistanceToBall => me.body.pos.distance(ball.pos) / FIELD_LENGTH,
            Self::ClosenessToBall => 1.0 - (me.body.pos.distance(ball.pos) / FIELD_LENGTH).min(1.0),
            Self::AirTime => events.detail(car).air_time.min(MAX_AIR_TIME) / MAX_AIR_TIME,
            Self::BoostUsed => (prev.cars[car].boost - me.boost).max(0.0) / 100.0,
            Self::BoostGained => (me.boost - prev.cars[car].boost).max(0.0) / 100.0,
            Self::BallSpeedChange => (ball.vel - prev.ball.vel).length() / BALL_MAX_SPEED,
            Self::Touch => bit(events.cars[car].touched_ball),
            Self::Goal => bit(cur.goal_scored == Some(me.team)),
            Self::Concede => bit(cur.goal_scored == Some(me.team.opposite())),
            Self::Shot => bit(events.cars[car].touched_ball
                && crate::mechanics::heading_into_goal(ball, attack_sign(me.team) * GOAL_LINE_Y)),
            Self::Demo => bit(events.cars[car].demoed_opponent),
            Self::Demoed => bit(me.is_demoed && !prev.cars[car].is_demoed),
            Self::Bump => bit(events.cars[car].bumped_opponent),
            Self::Jump => bit(me.has_jumped && !prev.cars[car].has_jumped),
            Self::Flip => bit(me.has_flipped && !prev.cars[car].has_flipped),
        }
    }
}

/// One of a term's conditions: a block, or a formula (true when not 0).
#[derive(Clone, Debug, PartialEq)]
enum When {
    Flag(Cond),
    Formula(Node),
}

/// What a term pays while its conditions hold.
#[derive(Clone, Debug, PartialEq)]
enum Pay {
    /// Blocks, multiplied (or the smallest with `min`).
    Measures(Vec<Measure>, bool),
    Formula(Node),
}

#[derive(Clone, Debug, PartialEq)]
struct Term {
    when: Vec<When>,
    pay: Pay,
    weight: f32,
    /// Paid at most once between two of the car's ball touches.
    once_per_touch: bool,
    /// Paid on the first step it pays in an episode only.
    once_per_episode: bool,
    /// The most the term pays (or charges) in one step, after its weight.
    cap_step: Option<f32>,
    /// The most the term pays in one episode, after its weight (charges are not capped).
    cap_episode: Option<f32>,
    /// Of the state alone (or marked `every_step`) and paid as is: scaled to the tick skip.
    continuous: bool,
    /// Memory slots: the touch lock, the episode's one payment, the episode's pay so far.
    lock: Option<usize>,
    done: Option<usize>,
    paid: Option<usize>,
}

const TERM_KEYS: &[&str] = &[
    "when",
    "measure",
    "value",
    "combine",
    "weight",
    "once_per_touch",
    "once_per_episode",
    "cap_step",
    "cap_episode",
    "every_step",
];

fn list(map: &serde_json::Map<String, Value>, key: &str) -> Vec<Value> {
    match map.get(key) {
        None => Vec::new(),
        Some(Value::Array(items)) => items.clone(),
        Some(one) => vec![one.clone()],
    }
}

fn flag(map: &serde_json::Map<String, Value>, key: &str) -> Result<bool, String> {
    match map.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("{key} is true or false")),
    }
}

fn number(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f32>, String> {
    match map.get(key) {
        None => Ok(None),
        // Finite as the core holds it, a 32-bit number: 1e300 is finite as written but not as that.
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite())
            .map(Some)
            .ok_or_else(|| format!("{key} must be a number")),
    }
}

impl Term {
    fn parse(written: &Value, scope: &Scope, stateful: &mut Vec<Stateful>) -> Result<Self, String> {
        let map = written.as_object().ok_or(
            "a term is a table: { measure = ... (or value = \"formula\"), when = [...], weight = ... }",
        )?;
        if map.contains_key("potential") {
            return Err("potential terms are not in this version of NitroLearn".into());
        }
        for key in map.keys() {
            if !TERM_KEYS.contains(&key.as_str()) {
                return Err(format!("'{key}' is not a setting of a term"));
            }
        }
        let (conditions, measures) = (list(map, "when"), list(map, "measure"));
        if conditions.len() > MAX_CONDITIONS || measures.len() > MAX_CONDITIONS {
            return Err(format!(
                "a term has at most {MAX_CONDITIONS} conditions and {MAX_CONDITIONS} measures"
            ));
        }
        let mut uses = Uses::default();
        let mut when = Vec::new();
        // Blocks of the state: paid every step, scaled to the tick skip.
        let mut flags_state = true;
        for (i, item) in conditions.iter().enumerate() {
            match item {
                Value::String(text) if FLAGS.iter().any(|(n, _)| n == text) => {
                    let cond = Cond::parse(item)?;
                    flags_state &= !cond.is_event();
                    when.push(When::Flag(cond));
                }
                Value::String(text) => {
                    let word = text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                    let made = expr::compile(text, scope, stateful).map_err(|e| {
                        if word {
                            format!("when[{i}]: there is no condition called '{text}'")
                        } else {
                            format!("when[{i}]: {e}")
                        }
                    })?;
                    if made.ty != Ty::Num {
                        return Err(format!(
                            "when[{i}]: a condition must be true or false, not a direction"
                        ));
                    }
                    uses = uses.with(made.uses);
                    when.push(When::Formula(made.node));
                }
                _ => {
                    let cond = Cond::parse(item).map_err(|e| format!("when[{i}]: {e}"))?;
                    flags_state &= !cond.is_event();
                    when.push(When::Flag(cond));
                }
            }
        }
        let pay = match (map.get("value"), measures.is_empty()) {
            (Some(_), false) => return Err("a term has a measure or a value, not both".into()),
            (None, true) => {
                return Err(
                    "a term needs a measure (\"one\" pays 1 while its conditions hold) or a value formula"
                        .into(),
                );
            }
            (Some(Value::String(text)), true) => {
                let made =
                    expr::compile(text, scope, stateful).map_err(|e| format!("value: {e}"))?;
                if made.ty != Ty::Num {
                    return Err(
                        "value: a term pays a number, not a direction (use length(...) or .x)"
                            .into(),
                    );
                }
                uses = uses.with(made.uses);
                Pay::Formula(made.node)
            }
            (Some(_), true) => return Err("value is a formula in quotes".into()),
            (None, false) => {
                let parsed = measures
                    .iter()
                    .map(Measure::parse)
                    .collect::<Result<Vec<_>, _>>()?;
                flags_state &= parsed.iter().all(|m| m.is_state());
                let min = match map.get("combine").map(|c| c.as_str()) {
                    None | Some(Some("product")) => false,
                    Some(Some("min")) => true,
                    _ => return Err("combine is \"product\" or \"min\"".into()),
                };
                Pay::Measures(parsed, min)
            }
        };
        if matches!(pay, Pay::Formula(_)) && map.contains_key("combine") {
            return Err(
                "combine goes with measures; a value formula says itself how it combines".into(),
            );
        }
        let weight = number(map, "weight")
            .map_err(|_| "a term's weight must be a number".to_string())?
            .unwrap_or(1.0);
        let once_per_touch = flag(map, "once_per_touch")?;
        let once_per_episode = flag(map, "once_per_episode")?;
        let positive = |key: &str| -> Result<Option<f32>, String> {
            match number(map, key)? {
                Some(n) if n <= 0.0 => Err(format!("{key} must be above 0")),
                other => Ok(other),
            }
        };
        let (cap_step, cap_episode) = (positive("cap_step")?, positive("cap_episode")?);
        let state = flags_state && uses.state_only();
        let every_step = flag(map, "every_step")?;
        let plain =
            !(once_per_touch || once_per_episode || cap_step.is_some() || cap_episode.is_some());
        Ok(Self {
            when,
            pay,
            weight,
            once_per_touch,
            once_per_episode,
            cap_step,
            cap_episode,
            continuous: plain && (state || every_step),
            lock: None,
            done: None,
            paid: None,
        })
    }

    fn holds(&self, ctx: &Ctx) -> bool {
        let game = if ctx.past { ctx.prev } else { ctx.cur };
        self.when.iter().all(|w| match w {
            When::Flag(c) => c.holds(game, ctx.events, ctx.car),
            When::Formula(node) => ctx.num(node) != 0.0,
        })
    }

    /// The term's value before its weight: 0 where its conditions do not hold.
    fn value(&self, ctx: &Ctx) -> f32 {
        if !self.holds(ctx) {
            return 0.0;
        }
        match &self.pay {
            Pay::Formula(node) => ctx.num(node),
            Pay::Measures(measures, min) => {
                let values = measures
                    .iter()
                    .map(|m| m.value(ctx.prev, ctx.cur, ctx.events, ctx.car));
                if *min {
                    values.fold(f32::INFINITY, f32::min)
                } else {
                    values.product()
                }
            }
        }
    }
}

/// A built reward, checked and ready: paid through the same path as a reward of one's own.
pub struct Built {
    terms: Vec<Term>,
    /// Variables: each one's value when an episode starts and its formula for the next step.
    vars: Vec<(f32, Node)>,
    /// The remembering functions, and for each variable how many of them its formula holds (they come
    /// first, in the variables' order; the terms' follow).
    stateful: Vec<Stateful>,
    var_stateful: Vec<usize>,
    every_step: bool,
    memory: usize,
    /// Where the blocks of memory start: the variables (after a slot saying they were set), their values at
    /// the step before, the remembering functions' outputs, and their own numbers (two each).
    vars_at: usize,
    prev_vars_at: usize,
    states_at: usize,
    state_slots_at: usize,
}

/// Most variables, remembering functions and terms one built reward may have.
const MAX_VARS: usize = 16;
const MAX_STATEFUL: usize = 32;
const MAX_TERMS: usize = 32;
/// Most conditions (and measures) one term may have: with the limits above, the work per car and step is
/// bounded however a shared settings file is written.
const MAX_CONDITIONS: usize = 16;

impl Built {
    pub fn new(terms: &[Value]) -> Result<Self, String> {
        Self::with_vars(&[], terms)
    }

    pub fn with_vars(vars: &[Value], terms: &[Value]) -> Result<Self, String> {
        if terms.is_empty() {
            return Err("a built reward needs at least one term".into());
        }
        if terms.len() > MAX_TERMS || vars.len() > MAX_VARS {
            return Err(format!(
                "a built reward has at most {MAX_TERMS} terms and {MAX_VARS} variables"
            ));
        }
        let mut names: Vec<String> = Vec::new();
        let mut written = Vec::new();
        for (i, var) in vars.iter().enumerate() {
            let map = var.as_object().ok_or_else(|| {
                format!(
                    "variable {}: a table {{ name = ..., start = ..., next = \"formula\" }}",
                    i + 1
                )
            })?;
            for key in map.keys() {
                if !["name", "start", "next"].contains(&key.as_str()) {
                    return Err(format!(
                        "variable {}: '{key}' is not a setting of a variable",
                        i + 1
                    ));
                }
            }
            let name = map
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| {
                    n.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                        && n.chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                })
                .ok_or_else(|| {
                    format!("variable {}: a name of small letters, digits and _", i + 1)
                })?;
            if expr::reserved(name) || names.iter().any(|n| n == name) {
                return Err(format!("variable {}: the name '{name}' is taken", i + 1));
            }
            let start = number(map, "start")
                .map_err(|e| format!("variable '{name}': {e}"))?
                .unwrap_or(0.0);
            let next = match map.get("next") {
                Some(Value::String(text)) => text.clone(),
                _ => return Err(format!("variable '{name}': next = \"formula\" is needed")),
            };
            names.push(name.to_string());
            written.push((start, next));
        }
        let scope = Scope { vars: &names };
        let mut stateful = Vec::new();
        let mut var_stateful = Vec::new();
        let mut parsed_vars = Vec::new();
        for ((start, next), name) in written.iter().zip(&names) {
            let before = stateful.len();
            let made = expr::compile(next, &scope, &mut stateful)
                .map_err(|e| format!("variable '{name}': {e}"))?;
            if made.ty != Ty::Num {
                return Err(format!("variable '{name}': a variable holds a number"));
            }
            var_stateful.push(stateful.len() - before);
            parsed_vars.push((*start, made.node));
        }
        let mut terms = terms
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Term::parse(t, &scope, &mut stateful).map_err(|e| format!("term {}: {e}", i + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if stateful.len() > MAX_STATEFUL {
            return Err(format!(
                "a built reward has at most {MAX_STATEFUL} since/held/count/once"
            ));
        }
        let (v, n) = (parsed_vars.len(), stateful.len());
        let mut memory = if v + n > 0 { 1 + 2 * v + 3 * n } else { 0 };
        let mut slot = || {
            memory += 1;
            Some(memory - 1)
        };
        for term in &mut terms {
            if term.once_per_touch {
                term.lock = slot();
            }
            if term.once_per_episode {
                term.done = slot();
            }
            if term.cap_episode.is_some() {
                term.paid = slot();
            }
        }
        Ok(Self {
            every_step: terms.iter().all(|t| t.continuous),
            terms,
            vars: parsed_vars,
            stateful,
            var_stateful,
            memory,
            vars_at: 1,
            prev_vars_at: 1 + v,
            states_at: 1 + 2 * v,
            state_slots_at: 1 + 2 * v + n,
        })
    }

    fn ctx<'a>(
        &self,
        prev: &'a GameSnapshot,
        cur: &'a GameSnapshot,
        events: &'a StepEvents,
        car: usize,
        memory: &'a [f32],
    ) -> Ctx<'a> {
        let (v, n) = (self.vars.len(), self.stateful.len());
        let block = |at: usize, len: usize| memory.get(at..at + len).unwrap_or(&[]);
        Ctx {
            prev,
            cur,
            events,
            car,
            it: car,
            past: false,
            vars: block(self.vars_at, v),
            prev_vars: block(self.prev_vars_at, v),
            states: block(self.states_at, n),
        }
    }

    /// Steps the remembering function `i` on.
    fn remember(&self, i: usize, ctx_cond: bool, dt: f32, memory: &mut [f32]) {
        let (out, a) = (self.states_at + i, self.state_slots_at + 2 * i);
        let output = match self.stateful[i].kind {
            Remember::Since => {
                if ctx_cond {
                    memory[a] = 1.0;
                    memory[a + 1] = 0.0;
                } else if memory[a] != 0.0 {
                    memory[a + 1] += dt;
                }
                if memory[a] != 0.0 {
                    memory[a + 1]
                } else {
                    NEVER
                }
            }
            Remember::Held => {
                memory[a] = if ctx_cond { memory[a] + dt } else { 0.0 };
                memory[a]
            }
            Remember::Count => {
                memory[a] += f32::from(u8::from(ctx_cond));
                memory[a]
            }
            Remember::Once => {
                let first = ctx_cond && memory[a] == 0.0;
                if first {
                    memory[a] = 1.0;
                }
                f32::from(u8::from(first))
            }
        };
        memory[out] = output;
    }

    fn step_memory(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) {
        let v = self.vars.len();
        if memory[0] == 0.0 {
            memory[0] = 1.0;
            for (i, (start, _)) in self.vars.iter().enumerate() {
                memory[self.vars_at + i] = *start;
            }
        }
        memory.copy_within(self.vars_at..self.vars_at + v, self.prev_vars_at);
        let dt = events.step_seconds;
        let mut next_state = 0;
        let mut run = |count: usize, memory: &mut [f32]| {
            for i in next_state..next_state + count {
                let cond = self
                    .ctx(prev, cur, events, car, memory)
                    .num(&self.stateful[i].cond)
                    != 0.0;
                self.remember(i, cond, dt, memory);
            }
            next_state += count;
        };
        for (i, (_, next)) in self.vars.iter().enumerate() {
            run(self.var_stateful[i], memory);
            let value = self.ctx(prev, cur, events, car, memory).num(next);
            memory[self.vars_at + i] = value;
        }
        run(
            self.stateful.len() - self.var_stateful.iter().sum::<usize>(),
            memory,
        );
    }

    fn total(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        mut memory: Option<&mut [f32]>,
    ) -> f32 {
        if let Some(memory) = memory.as_deref_mut()
            && (!self.vars.is_empty() || !self.stateful.is_empty())
        {
            self.step_memory(prev, cur, events, car, memory);
        }
        // A reward paid every step as a whole is scaled by the reward set; in a mixed one, the continuous
        // terms are scaled here so they mean the same at any tick skip.
        let step_scale = if self.every_step {
            1.0
        } else {
            events.step_seconds / REFERENCE_STEP_SECONDS
        };
        let touched = events.cars[car].touched_ball;
        let mut total = 0.0;
        for term in &self.terms {
            let mut slots = memory.as_deref_mut();
            if let (Some(s), Some(memory)) = (term.lock, slots.as_deref_mut()) {
                if touched {
                    memory[s] = 0.0;
                }
                if memory[s] != 0.0 {
                    continue;
                }
            }
            if let (Some(s), Some(memory)) = (term.done, slots.as_deref_mut())
                && memory[s] != 0.0
            {
                continue;
            }
            let empty: &[f32] = &[];
            let ctx = self.ctx(prev, cur, events, car, slots.as_deref().unwrap_or(empty));
            let value = term.value(&ctx);
            if !value.is_finite() || value == 0.0 {
                continue;
            }
            let mut pay = term.weight * value * if term.continuous { step_scale } else { 1.0 };
            if let Some(cap) = term.cap_step {
                pay = pay.clamp(-cap, cap);
            }
            // A big weight times a big value can pass what a 32-bit number holds: never paid as infinity.
            if !pay.is_finite() {
                continue;
            }
            if let Some(memory) = slots {
                if let Some(s) = term.lock {
                    memory[s] = 1.0;
                }
                if let Some(s) = term.done {
                    memory[s] = 1.0;
                }
                if let (Some(s), Some(cap)) = (term.paid, term.cap_episode)
                    && pay > 0.0
                {
                    pay = pay.min((cap - memory[s]).max(0.0));
                    memory[s] += pay;
                }
            }
            total += pay;
        }
        if total.is_finite() { total } else { 0.0 }
    }
}

impl Reward for Built {
    fn every_step(&self) -> bool {
        self.every_step
    }

    fn memory(&self) -> usize {
        self.memory
    }

    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        self.total(prev, cur, events, car, None)
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        self.total(prev, cur, events, car, Some(memory))
    }
}

/// The block names this core knows: (plain conditions, numbered conditions, measures with whether each is
/// of the state alone), for the settings check's own list to be tested against.
pub fn blocks() -> (
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<(&'static str, bool)>,
) {
    (
        FLAGS.iter().map(|(name, _)| *name).collect(),
        NUMBERED.to_vec(),
        MEASURES
            .iter()
            .map(|&(name, m)| (name, m.is_state()))
            .collect(),
    )
}

/// A built reward's settings checked as the core reads them (`{"title", "vars", "terms"}`, the tag left
/// out): the first problem found, or whether it is paid every step and how many numbers it remembers.
pub fn check(settings: &str) -> Result<(bool, usize), String> {
    let settings: serde_json::Map<String, Value> =
        serde_json::from_str(settings).map_err(|e| e.to_string())?;
    let made = make(&settings)?;
    Ok((made.every_step(), made.memory()))
}

/// The reward's title as a label that lives as long as the program (labels are `&'static str`); each
/// title is stored once however often settings are read.
fn intern(title: &str) -> &'static str {
    static TITLES: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut titles = TITLES.lock().unwrap_or_else(|e| e.into_inner());
    let titles = titles.get_or_insert_with(HashSet::new);
    if let Some(&kept) = titles.get(title) {
        return kept;
    }
    let kept: &'static str = Box::leak(title.to_owned().into_boxed_str());
    titles.insert(kept);
    kept
}

/// `{ "name": "built", "title": ..., "terms": [...] }` (the tag already read) as a reward the core pays.
pub fn make(settings: &serde_json::Map<String, Value>) -> Result<CustomReward, String> {
    for key in settings.keys() {
        if !["title", "terms", "vars"].contains(&key.as_str()) {
            return Err(format!("'{key}' is not a setting of a built reward"));
        }
    }
    let title = settings
        .get("title")
        .and_then(Value::as_str)
        .filter(|t| {
            !t.is_empty()
                && t.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        })
        .ok_or("a built reward needs a title of small letters, digits and _")?;
    let terms = match settings.get("terms") {
        Some(Value::Array(terms)) => terms.as_slice(),
        _ => return Err("a built reward needs terms = [ ... ]".into()),
    };
    let vars = match settings.get("vars") {
        None => &[][..],
        Some(Value::Array(vars)) => vars.as_slice(),
        _ => return Err("vars is a list of [[rewards.vars]] tables".into()),
    };
    let built = Built::with_vars(vars, terms)?;
    Ok(CustomReward::built(
        intern(title),
        Value::Object(settings.clone()),
        Arc::new(built),
    ))
}

#[cfg(test)]
mod advanced_tests;
#[cfg(test)]
mod tests;
