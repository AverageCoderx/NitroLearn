//! The game state as both training and the RLBot bot see it.
//!
//! `CarSnapshot`, `GameSnapshot::ball` and `GameSnapshot::pads` hold only values an RLBot v5 `GamePacket`
//! provides directly or through a documented derivation, so observations and action masks built from
//! them are identical in training and in a real match. `StepEvents` are training-only (rewards).

use rocketsim::{Mat3A, Vec3A};

pub const NUM_BOOST_PADS: usize = 34;
pub const MAX_CARS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Team {
    #[default]
    Blue,
    Orange,
}

impl Team {
    pub const fn opposite(self) -> Self {
        match self {
            Self::Blue => Self::Orange,
            Self::Orange => Self::Blue,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    /// Team whose half of the field contains `y` (Blue defends −Y).
    pub fn from_field_y(y: f32) -> Self {
        if y <= 0.0 { Self::Blue } else { Self::Orange }
    }
}

/// Position (uu), orientation (columns: forward, right, up), velocity (uu/s), angular velocity (rad/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub pos: Vec3A,
    pub rot: Mat3A,
    pub vel: Vec3A,
    pub ang_vel: Vec3A,
}

impl Body {
    pub const RESTING: Self = Self {
        pos: Vec3A::ZERO,
        rot: Mat3A::IDENTITY,
        vel: Vec3A::ZERO,
        ang_vel: Vec3A::ZERO,
    };

    pub fn forward(&self) -> Vec3A {
        self.rot.x_axis
    }

    pub fn up(&self) -> Vec3A {
        self.rot.z_axis
    }
}

/// Controller inputs, the same eight values RLBot's `ControllerState` carries (minus `use_item`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Controls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
}

impl Controls {
    /// `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`, booleans as 0/1.
    pub fn to_floats(self) -> [f32; 8] {
        [
            self.throttle,
            self.steer,
            self.pitch,
            self.yaw,
            self.roll,
            f32::from(u8::from(self.jump)),
            f32::from(u8::from(self.boost)),
            f32::from(u8::from(self.handbrake)),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CarSnapshot {
    pub team: Team,
    pub body: Body,
    /// 0..100.
    pub boost: f32,
    pub on_ground: bool,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    /// RLBot: `has_dodged`.
    pub has_flipped: bool,
    /// A jump, double jump or flip can be started on the next tick. RLBot side: `on_ground || (!has_dodged &&
    /// !has_double_jumped && (!has_jumped || window_open(dodge_timeout)))`; sim side: RocketSim's
    /// `has_flip_or_jump()`, which is the same rule.
    pub can_flip: bool,
    pub is_demoed: bool,
    pub is_supersonic: bool,
    /// Seconds until a demolished car respawns; 0 while alive. RocketSim: `demo_respawn_timer`; RLBot:
    /// `demolished_timeout` (−1 while alive, so clamped at 0).
    pub demo_timer: f32,
    /// Seconds the double-jump/flip window stays open: 0 when no jump or flip is available, the full
    /// 1.25 s while the car is on the ground or holds a flip reset (no window running), otherwise what is
    /// left of the window after the first jump. RocketSim: `1.25 − air_time_since_jump`; RLBot:
    /// `dodge_timeout`. During the first jump's button hold (≤ 0.2 s) RLBot may report up to 0.2 s more.
    pub flip_time_left: f32,
}

/// RocketSim's `DOUBLEJUMP_MAX_DELAY`: how long after the first jump a double jump or flip is allowed.
pub const FLIP_WINDOW_SECONDS: f32 = 1.25;

/// Whether a first jump's window with `flip_time_left` seconds left still allows a double jump or flip on
/// the next tick: RocketSim 0.3 counts that tick's time first (`air_time_since_jump + 1/120 < 1.25`;
/// 0.2.9 said yes one tick too long).
pub fn window_open(flip_time_left: f32) -> bool {
    flip_time_left > 1.0 / 120.0
}

impl CarSnapshot {
    /// `flip_time_left` from the jump flags and the time already spent in the air since the first jump.
    pub fn flip_time_left_from(
        on_ground: bool,
        has_jumped: bool,
        can_flip: bool,
        air_time_since_jump: f32,
    ) -> f32 {
        if !can_flip {
            0.0
        } else if !on_ground && has_jumped {
            (FLIP_WINDOW_SECONDS - air_time_since_jump).max(0.0)
        } else {
            FLIP_WINDOW_SECONDS
        }
    }
}

impl Default for Body {
    fn default() -> Self {
        Self::RESTING
    }
}

impl CarSnapshot {
    /// Upside down near the floor. Derived from position and orientation only, so training and the
    /// bot agree exactly (RocketSim's world-contact flag has no RLBot equivalent).
    pub fn is_turtled(&self) -> bool {
        !self.on_ground && self.body.up().z < -0.9 && self.body.pos.z < 60.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadSnapshot {
    pub active: bool,
    /// Seconds until the pad is active again (0 while active). RLBot v5's `BoostPadState::timer` counts
    /// the other way (seconds since pickup); the bot converts it (crates/nitro_bot/src/packet.rs).
    pub cooldown: f32,
}

impl Default for PadSnapshot {
    fn default() -> Self {
        Self {
            active: true,
            cooldown: 0.0,
        }
    }
}

/// What happened to one car during the last step (training only).
///
/// The contact fields are latched tick by tick in `Sim::step`, so a contact of a single tick is never
/// missed. "Arena" is the static field: floor, walls, ceiling and goals. Wheels on the ball (a flip reset)
/// or on another car are never arena contact, unlike `CarSnapshot::on_ground`, which counts any 3 wheels.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CarEvents {
    pub touched_ball: bool,
    pub bumped_opponent: bool,
    pub demoed_opponent: bool,
    /// Bumped a teammate, or was bumped by one (both cars of the bump).
    pub bumped_teammate: bool,
    /// On some tick the car was on the floor: a wheel on an arena surface facing up (normal z > 0.7) less
    /// than 60 uu high, or its body against one less than 40 uu high. The floor of a goal counts; a wall,
    /// the ceiling and the curve up to a wall above those heights do not.
    pub on_floor: bool,
    /// On some tick a wheel or the body touched any arena surface.
    pub on_arena: bool,
    /// The car touched the ball on a tick when it was on the floor.
    pub touched_ball_on_floor: bool,
    /// The car touched the ball on a tick when it touched no arena surface.
    pub touched_ball_in_air: bool,
    /// The arena ticks (counted like `StepEvents::tick`) of the car's first and last ball touch in the step.
    pub first_touch_tick: Option<u64>,
    pub last_touch_tick: Option<u64>,
    /// The arena tick of the car's first touch in the air in the step.
    pub first_air_touch_tick: Option<u64>,
    /// The arena tick of the car's first touch in the air with the ball off the field's floor on that tick
    /// (the touch that can start play in the air; one on a ball on the floor is a setup pop).
    pub air_start_tick: Option<u64>,
    /// The arena tick of the car's first arena contact in the step.
    pub first_arena_tick: Option<u64>,
    /// The arena tick of the car's first arena contact after `air_start_tick` in the step. With the ticks
    /// above, it orders touches and contacts inside one step.
    pub arena_after_air_start_tick: Option<u64>,
    /// Some ticks of the step came after the ball was scored (`StepEvents::goal_tick`); a touch on
    /// them set `touched_ball` only.
    pub past_goal: bool,
}

impl CarEvents {
    /// Latches the touch and contact fields for physics tick `tick`: whether the car `touched` the ball, was
    /// on the floor and touched any arena surface (the floor counts as one) on it, and whether the ball was on
    /// the field's floor on it (`ball_down`, as `StepEvents::ball_on_field_floor`).
    pub fn add_tick(
        &mut self,
        tick: u64,
        touched: bool,
        on_floor: bool,
        on_arena: bool,
        ball_down: bool,
    ) {
        let on_arena = on_arena || on_floor;
        self.on_floor |= on_floor;
        self.on_arena |= on_arena;
        if on_arena {
            self.first_arena_tick.get_or_insert(tick);
            if self.air_start_tick.is_some() {
                self.arena_after_air_start_tick.get_or_insert(tick);
            }
        }
        if touched {
            self.touched_ball = true;
            self.first_touch_tick.get_or_insert(tick);
            self.last_touch_tick = Some(tick);
            self.touched_ball_on_floor |= on_floor;
            if !on_arena {
                self.touched_ball_in_air = true;
                self.first_air_touch_tick.get_or_insert(tick);
                if !ball_down {
                    self.air_start_tick.get_or_insert(tick);
                }
            }
        }
    }

    /// Latches a physics tick after the step's goal (`StepEvents::goal_tick`): a touch still sets
    /// `touched_ball`, but no tick-ordered field, so play in the air is judged up to the goal (in Rocket
    /// League the ball explodes there).
    pub fn add_tick_after_goal(&mut self, touched: bool) {
        self.past_goal = true;
        self.touched_ball |= touched;
    }
}

/// Below this height (uu) the ball's centre is on the floor (its resting height is 93.15).
pub const BALL_FLOOR_Z: f32 = 100.0;

/// More of a car's state at the end of the step, for rewards of one's own (training only): what
/// RocketSim keeps about each car beyond the snapshot. The RLBot bot never fills it.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CarDetail {
    /// The controls the car was driven with this step.
    pub controls: Controls,
    /// Which wheels touch something; the first two are the front wheels.
    pub wheel_contacts: [bool; 4],
    /// The normal of the surface the car's body (not a wheel) touches, when it touches one.
    pub world_contact_normal: Option<Vec3A>,
    pub is_jumping: bool,
    pub is_flipping: bool,
    pub is_auto_flipping: bool,
    pub is_boosting: bool,
    /// RocketSim's jump time: seconds since the current jump began (read it while `is_jumping`).
    pub jump_time: f32,
    /// RocketSim's flip clock (s): from a flip's start while it flips, then from the flip's end; it keeps
    /// counting in the air and on the ground (RocketSim 0.3; 0.2.9 held it at 0 on the ground). Read it
    /// with `has_flipped` or `is_flipping`: a car that has not flipped this episode starts at 3 s.
    pub flip_time: f32,
    /// Seconds in the air.
    pub air_time: f32,
    /// Seconds in the air since the jump ended (0 when the car never jumped).
    pub air_time_since_jump: f32,
    /// Seconds of continuous boosting.
    pub boosting_time: f32,
    /// The flip's torque direction in the car's own frame (a forward flip has positive y).
    pub flip_rel_torque: Vec3A,
    /// The handbrake, smoothed: 0 to 1.
    pub handbrake: f32,
    /// The arena tick of this car's latest touch of the ball this episode (`StepEvents::tick` counts the
    /// same ticks), and where the ball was then.
    pub last_hit_tick: Option<u64>,
    pub last_hit_ball_pos: Vec3A,
    /// The car this one last bumped or was bumped by this episode (either team), and the arena tick.
    pub last_contact_car: Option<usize>,
    pub last_contact_tick: Option<u64>,
    /// Seconds until this car can bump or demolish again.
    pub bump_cooldown: f32,
}

#[derive(Clone, Debug, Default)]
pub struct StepEvents {
    pub cars: Vec<CarEvents>,
    /// The ball was on the floor at some tick of the step. Latched tick by tick, because a bounce keeps
    /// the ball down for only a few ticks and a look at the step's end alone misses most of them.
    pub ball_on_floor: bool,
    /// The ball touched the field's floor at some tick of the step: its centre below `BALL_FLOOR_Z`,
    /// or a contact with an arena surface facing up (normal z > 0.7) less than 60 uu high. Not inside a
    /// goal (|y| > 5120): a ball dropping into the net may touch the goal's floor before it counts as scored.
    pub ball_on_field_floor: bool,
    /// The last arena tick of the step on which `ball_on_field_floor` held: ordered against the cars' touch
    /// ticks, it tells whether the ball was down before or after a touch that started play in the air.
    pub ball_field_floor_last_tick: Option<u64>,
    /// Each car's state beyond the snapshot at the step's end, and at the end of the step before
    /// (training only; `detail` and `prev_detail` read them).
    pub details: Vec<CarDetail>,
    pub prev_details: Vec<CarDetail>,
    /// The car that touched the ball last this episode: kept from step to step, cleared when an episode
    /// starts.
    pub last_touch: Option<usize>,
    /// The arena tick on which the ball was first seen scored during the step; the cars' tick-ordered
    /// contact fields stop there. None without a goal, and in the RocketSim 2.2.1 build.
    pub goal_tick: Option<u64>,
    /// How long a step lasts, in seconds (the tick skip over 120).
    pub step_seconds: f32,
    /// The arena's tick count at the step's end (120 a second; it counts on across episodes).
    pub tick: u64,
}

impl StepEvents {
    /// Ready for the next step: what happened is cleared; the details, `last_touch`, `step_seconds` and
    /// `tick` stay until the step fills them.
    pub fn clear(&mut self, num_cars: usize) {
        self.cars.clear();
        self.cars.resize(num_cars, CarEvents::default());
        self.ball_on_floor = false;
        self.ball_on_field_floor = false;
        self.ball_field_floor_last_tick = None;
        self.goal_tick = None;
    }

    /// Car `car`'s details at the step's end (all defaults where nothing filled them, as in a test).
    pub fn detail(&self, car: usize) -> CarDetail {
        self.details.get(car).copied().unwrap_or_default()
    }

    /// Car `car`'s details at the end of the step before.
    pub fn prev_detail(&self, car: usize) -> CarDetail {
        self.prev_details.get(car).copied().unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub ball: Body,
    pub cars: Vec<CarSnapshot>,
    /// Sorted by y, then x (RocketSim's order; a 180° turn maps pad i to pad 33 − i). RLBot lists two
    /// pairs differently, so the bot maps packet pads to this order by position.
    pub pads: [PadSnapshot; NUM_BOOST_PADS],
    /// Team that scored during the last step.
    pub goal_scored: Option<Team>,
}

impl Default for GameSnapshot {
    fn default() -> Self {
        Self {
            ball: Body::RESTING,
            cars: Vec::with_capacity(MAX_CARS),
            pads: [PadSnapshot::default(); NUM_BOOST_PADS],
            goal_scored: None,
        }
    }
}
