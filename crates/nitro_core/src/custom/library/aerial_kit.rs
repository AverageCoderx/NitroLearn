//! Helpers shared by the trick-move rewards (air dribble, flip reset, reset musty, Mawkzy flick, speed
//! flip): an episode clock in physics ticks, named numbers in a reward's memory, the settings' defaults.
//! No reward here.

use crate::custom::prelude::*;

/// RLGym's ball radius, which these rewards were written with (RocketSim's collision radius, BALL_RADIUS, is
/// 91.25).
pub const RLGYM_BALL_RADIUS: f32 = 92.75;
pub const GRAVITY_Z: f32 = -650.0;
/// The Octane's hitbox (RocketSim's CarConfig): its size, and its centre's offset from the car's origin.
pub const HITBOX_SIZE: Vec3A = Vec3A::new(120.507, 86.6994, 38.6591);
pub const HITBOX_OFFSET: Vec3A = Vec3A::new(13.87566, 0.0, 20.755);

/// Close to the floor, ceiling, side walls or back walls (straight-plane approximation of the arena; corners and
/// goals are curved or recessed, which only makes this slightly generous near them).
pub fn near_arena_surface(pos: Vec3A, margin: f32) -> bool {
    pos.z < margin
        || pos.z > CEILING_Z - margin
        || pos.x.abs() > SIDE_WALL_X - margin
        || pos.y.abs() > BACK_WALL_Y - margin
}

/// Wheels on the arena, not on the ball (`on_ground` is also true with 3+ wheels on the ball).
pub fn on_arena_surface(car: &CarSnapshot) -> bool {
    car.on_ground && near_arena_surface(car.body.pos, 120.0)
}

/// A flip reset during the step: the car had no flip left and, away from every arena surface, its jump and flip
/// came back (only 3+ wheels touching something, here the ball, does that) with the ball close by. Compares
/// states, so it works at any tick skip.
pub fn flip_reset_happened(
    prev: &CarSnapshot,
    cur: &CarSnapshot,
    ball_pos: Vec3A,
    max_ball_dist: f32,
) -> bool {
    if cur.is_demoed || prev.is_demoed {
        return false;
    }
    if near_arena_surface(prev.body.pos, 120.0) || near_arena_surface(cur.body.pos, 120.0) {
        return false;
    }
    let had_no_flip = !prev.on_ground && !prev.can_flip;
    let reset = !cur.has_jumped && !cur.has_flipped && !cur.has_double_jumped;
    had_no_flip && reset && cur.body.pos.distance(ball_pos) <= max_ball_dist
}

/// `flip_reset_happened` that also sees a reset taking two steps: in RocketSim the dodge flag can clear a
/// step after the wheels reach the ball, while they are still on it (RocketSim's on the ground), which the
/// one-step test misses (a third of the scripted resets at tick skip 4). `waiting` is the car's own memory: it
/// had no flip left in the air and has not since been near a surface, demolished, or counted a reset. Returns
/// (a reset this step, the new `waiting`).
pub fn flip_reset_over_steps(
    waiting: bool,
    prev: &CarSnapshot,
    cur: &CarSnapshot,
    ball_pos: Vec3A,
    max_ball_dist: f32,
) -> (bool, bool) {
    let clear = |c: &CarSnapshot| !c.is_demoed && !near_arena_surface(c.body.pos, 120.0);
    let waiting = clear(prev) && (waiting || (!prev.on_ground && !prev.can_flip));
    let reset = waiting
        && clear(cur)
        && !cur.has_jumped
        && !cur.has_flipped
        && !cur.has_double_jumped
        && cur.body.pos.distance(ball_pos) <= max_ball_dist;
    (reset, waiting && !reset && clear(cur))
}

/// The car can't flip or jump now: a reset would give it one.
pub fn needs_reset(car: &CarSnapshot) -> bool {
    !car.on_ground && !car.can_flip
}

/// Air rolling: the roll input at least half on, not dodging, and the car turning about its nose at least
/// `min_rate` rad/s.
pub fn air_rolling(car: &CarSnapshot, detail: &CarDetail, min_rate: f32) -> bool {
    detail.controls.roll.abs() >= 0.5
        && !detail.is_flipping
        && car.body.ang_vel.dot(car.body.forward()).abs() >= min_rate
}

/// `v` in the car's own frame (forward, right, up).
pub fn local(body: &Body, v: Vec3A) -> Vec3A {
    body.rot.transpose() * v
}

/// `v` clamped to 0..1, and 0 for NaN (a ramp whose ends are equal).
pub fn clamp01(v: f32) -> f32 {
    if v >= 1.0 {
        1.0
    } else if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// 0 at `lo`, 1 at `hi`, clamped (either way round).
pub fn ramp(v: f32, lo: f32, hi: f32) -> f32 {
    clamp01((v - lo) / (hi - lo))
}

/// A 0-1 share `x` bent so the pay for each step of it changes by e^`curve` from the first step to the last
/// (`(e^(curve x) - 1) / (e^curve - 1)`): above 0 the last steps pay most, below 0 the first; 0: `x` itself (a
/// straight line).
pub fn speed_curve(x: f32, curve: f32) -> f32 {
    if curve.abs() < 1e-6 {
        x
    } else {
        ((curve * x).exp() - 1.0) / (curve.exp() - 1.0)
    }
}

/// The horizontal direction of `v` (zero when it has none).
pub fn dir2d(v: Vec3A) -> Vec3A {
    let flat = Vec3A::new(v.x, v.y, 0.0);
    let len = flat.length();
    if len > 1e-4 { flat / len } else { Vec3A::ZERO }
}

/// The horizontal length of `v`.
pub fn hspeed(v: Vec3A) -> f32 {
    (v.x * v.x + v.y * v.y).sqrt()
}

/// Radians to degrees.
pub fn deg(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

/// `v` if it is a number, else 0.
pub fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Physics ticks in one step (the step's length, 8 ticks when unknown).
pub fn step_ticks(events: &StepEvents) -> f32 {
    (step_seconds(events) * 120.0).round()
}

/// The step's length in seconds (8 ticks when unknown).
pub fn step_seconds(events: &StepEvents) -> f32 {
    if events.step_seconds > 0.0 {
        events.step_seconds
    } else {
        8.0 / 120.0
    }
}

/// A number kept in a reward's memory, from one or more of its f32s.
pub trait Kept: Copy {
    const SIZE: usize;
    fn read(from: &[f32]) -> Self;
    fn write(self, to: &mut [f32]);
}

impl Kept for f32 {
    const SIZE: usize = 1;
    fn read(from: &[f32]) -> Self {
        from[0]
    }
    fn write(self, to: &mut [f32]) {
        to[0] = self;
    }
}

impl Kept for bool {
    const SIZE: usize = 1;
    fn read(from: &[f32]) -> Self {
        from[0] != 0.0
    }
    fn write(self, to: &mut [f32]) {
        to[0] = f32::from(u8::from(self));
    }
}

/// Small whole numbers (counts, phases): exact in an f32.
impl Kept for i32 {
    const SIZE: usize = 1;
    fn read(from: &[f32]) -> Self {
        from[0] as i32
    }
    fn write(self, to: &mut [f32]) {
        to[0] = self as f32;
    }
}

impl Kept for Vec3A {
    const SIZE: usize = 3;
    fn read(from: &[f32]) -> Self {
        Vec3A::new(from[0], from[1], from[2])
    }
    fn write(self, to: &mut [f32]) {
        to[..3].copy_from_slice(&[self.x, self.y, self.z]);
    }
}

/// A struct of `Kept` fields that reads itself from, and writes itself to, a reward's memory: `SIZE` numbers.
#[allow(unused_macros)]
macro_rules! kept {
    ($(#[$meta:meta])* $vis:vis struct $name:ident { $($(#[$fmeta:meta])* $field:ident: $ty:ty,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        $vis struct $name { $($(#[$fmeta])* pub $field: $ty,)* }

        impl $name {
            /// How many numbers of a reward's memory it takes.
            pub const SIZE: usize = 0 $(+ <$ty as Kept>::SIZE)*;

            /// Read from the start of `memory`.
            pub fn load(memory: &[f32]) -> Self {
                let mut at = 0;
                $(
                    let $field = <$ty as Kept>::read(&memory[at..]);
                    at += <$ty as Kept>::SIZE;
                )*
                let _ = at;
                Self { $($field,)* }
            }

            /// Written to the start of `memory`.
            pub fn store(&self, memory: &mut [f32]) {
                let mut at = 0;
                $(
                    <$ty as Kept>::write(self.$field, &mut memory[at..]);
                    at += <$ty as Kept>::SIZE;
                )*
                let _ = at;
            }
        }
    };
}

/// Each setting's default as a function of its own name (what `#[serde(default = "name")]` calls), and the reward's
/// `Default` (every setting at its default).
#[allow(unused_macros)]
macro_rules! defaults {
    ($name:ident { $($field:ident: $ty:ty = $value:expr,)* }) => {
        $(fn $field() -> $ty { $value })*

        impl Default for $name {
            fn default() -> Self {
                Self { $($field: $field(),)* }
            }
        }
    };
}

// The reward files take both macros in with `use crate::custom::library::aerial_kit::*`.
pub(crate) use {defaults, kept};

kept! {
    /// The start of every such reward's memory: whether the episode's first step has been seen, and the
    /// episode's clock in physics ticks (0 at its start; f32 counts ticks exactly for 38 hours).
    pub struct Clock {
        started: bool,
        now: f32,
    }
}

impl Clock {
    /// Advances the clock by one step; true on the episode's first step (the reward then sets itself up from
    /// `prev`, the episode's starting state).
    pub fn advance(&mut self, events: &StepEvents) -> bool {
        let first = !self.started;
        self.started = true;
        self.now += step_ticks(events);
        first
    }
}

/// The car's latest ball touch, in RocketSim's terms: `tick` on the episode clock as RocketSim counts it (the tick
/// during which it touched: Nitro records the tick count after it, one more), the ball's centre then, and how many
/// ticks back from the step's end the car's pose must be wound to meet that ball position (Nitro records the ball
/// at the end of the touching tick, so this is one tick less than RocketSim's).
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub tick: f32,
    pub ball_pos: Vec3A,
    pub rewind_ticks: f32,
}

/// Car `car`'s latest touch this episode (None when it has none), against the episode clock at `now`.
pub fn last_hit(events: &StepEvents, car: usize, now: f32) -> Option<Hit> {
    let detail = events.detail(car);
    let at = detail.last_hit_tick?;
    let rewind_ticks = events.tick.saturating_sub(at) as f32;
    Some(Hit {
        tick: now - rewind_ticks - 1.0,
        ball_pos: detail.last_hit_ball_pos,
        rewind_ticks,
    })
}

/// For the rewards' tests: `name` is built in as the dashboard reads it, described, with `settings` settings, each
/// optional (a settings file can name the reward alone) and described.
#[cfg(test)]
pub fn assert_listed(name: &str, settings: usize) {
    let info = crate::custom::REWARDS
        .iter()
        .find(|info| info.name == name)
        .unwrap_or_else(|| panic!("{name} is built in"));
    assert!(!info.doc.is_empty() && !info.every_step, "{name}");
    assert_eq!(info.params.len(), settings, "{name}");
    for param in info.params {
        assert!(
            param.optional && !param.doc.is_empty(),
            "{name}.{}",
            param.name
        );
    }
    assert!(crate::custom::CustomReward::new(name, serde_json::json!({})).is_ok());
}

#[cfg(test)]
mod tests {
    use super::*;

    kept! {
        struct Sample {
            count: i32,
            flag: bool,
            at: Vec3A,
            seconds: f32,
        }
    }

    #[test]
    fn kept_numbers_round_trip_and_the_clock_counts_ticks() {
        let sample = Sample {
            count: 7,
            flag: true,
            at: Vec3A::new(1.0, -2.0, 3.5),
            seconds: 0.25,
        };
        let mut memory = vec![0.0; Sample::SIZE];
        sample.store(&mut memory);
        assert_eq!(Sample::SIZE, 6);
        assert_eq!(Sample::load(&memory), sample);
        let events = crate::custom::kit::testing::quiet(1);
        let mut clock = Clock::default();
        assert!(clock.advance(&events) && !clock.advance(&events));
        assert_eq!(clock.now, 16.0);
        assert!(clamp01(f32::NAN) == 0.0 && ramp(5.0, 10.0, 0.0) == 0.5);
    }
}
