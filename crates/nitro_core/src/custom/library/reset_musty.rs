//! A musty flick made off a high flip reset (a sling paying double a smack), and shaping for the turn
//! from the reset into the musty pose.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::aerial_kit::*;
use crate::custom::library::flick_kit::{has_dodge, project_to_goal_line};

/// The flip reset reward, stepped here to judge resets (same settings).
use crate::custom::library::flip_reset::{FlipReset, holds_reset_at_start};

/// A musty flick off a flip reset the car still held when it dodged: MustyFlickReward's pay (up to ~2.3 for an
/// aerial musty on net) times 0.5 for a smack rising to 1 for a sling; nothing for any other musty; a backward
/// shot can cost up to 0.5.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetMusty {
    /// Pay only mustys made with a counted flip reset (false: every musty).
    #[serde(default = "reset_only")]
    pub reset_only: bool,
    /// The ball must stay this far (its surface) off the side and back walls, and off the ceiling, from the
    /// counted reset to the outcome, 0 = off.
    #[serde(default = "wall_margin")]
    pub wall_margin: f32,
    /// A smack (the ball leaving the roof where it landed) pays this share...
    #[serde(default = "sling_floor")]
    pub sling_floor: f32,
    /// ...rising from a ride along the roof of this many uu...
    #[serde(default = "sling_ride_from")]
    pub sling_ride_from: f32,
    /// ...to the full pay (a sling) at this many.
    #[serde(default = "sling_ride_full")]
    pub sling_ride_full: f32,
    /// A roof push (the ball leaving the roof behind its middle, or the car still steep at the release) pays this
    /// share of a full scoop (the ball rolled off the front of the roof with the car near level), rising with the
    /// scoop; lower it to favour scoop mustys. 0.5 as musty_flick.
    #[serde(default = "scoop_floor")]
    pub scoop_floor: f32,
    /// A paid musty off a counted reset also earns this times the ball's horizontal speed toward the net as it left
    /// the car (the last contact's, so a scoop counts from its release; gravity taken out), 0 at
    /// release_speed_zero to 1 at release_speed_full, bent by release_speed_curve as stretch_shot_curve; paid at
    /// once when the musty is judged, goal or not, so the speed is rewarded apart from the shot (0: off).
    #[serde(default = "release_speed_weight")]
    pub release_speed_weight: f32,
    /// ...from this goalward speed (uu/s; 1,389 = 50 km/h)...
    #[serde(default = "release_speed_zero")]
    pub release_speed_zero: f32,
    /// ...full at this one (4,167 = 150 km/h)...
    #[serde(default = "release_speed_full")]
    pub release_speed_full: f32,
    /// ...and this bend (0 a straight line).
    #[serde(default = "release_speed_curve")]
    pub release_speed_curve: f32,
    /// The musty's training option: extra for a shot on target (MustyFlickReward's default: 0).
    #[serde(default = "on_net_bonus")]
    pub on_net_bonus: f32,
    /// ...this share of it flat, the rest scaling with the shot's goalward speed (default: 1).
    #[serde(default = "on_net_power_mix")]
    pub on_net_power_mix: f32,
    /// The musty's training option: extra for power on net (default: 0)...
    #[serde(default = "stretch_shot_weight")]
    pub stretch_shot_weight: f32,
    /// ...from this goalward ball speed...
    #[serde(default = "stretch_shot_from")]
    pub stretch_shot_from: f32,
    /// ...full at this one.
    #[serde(default = "stretch_shot_full")]
    pub stretch_shot_full: f32,
    /// Bends the shot's speed share between stretch_shot_from and stretch_shot_full: above 0 the fastest shots gain
    /// most, below 0 the first speed above stretch_shot_from pays most (0: a straight line).
    #[serde(default = "stretch_shot_curve")]
    pub stretch_shot_curve: f32,
    /// The musty's training option: extra for the goalward ball speed added (default: 0)...
    #[serde(default = "stretch_gain_weight")]
    pub stretch_gain_weight: f32,
    /// ...from this much...
    #[serde(default = "stretch_gain_from")]
    pub stretch_gain_from: f32,
    /// ...full at this much.
    #[serde(default = "stretch_gain_full")]
    pub stretch_gain_full: f32,
    /// The flip_reset reward's min_ball_height (keep them the same).
    #[serde(default = "reset_min_ball_height")]
    pub reset_min_ball_height: f32,
    /// The flip_reset reward's min_air_time (keep them the same).
    #[serde(default = "reset_min_air_time")]
    pub reset_min_air_time: f32,
    /// The flip_reset reward's min_use_ball_height: the musty's hit needs the ball this high (runs: 600).
    #[serde(default = "reset_min_use_ball_height")]
    pub reset_min_use_ball_height: f32,
    /// The flip_reset reward's start_reset_counts: a start in the air holding its flip under the ball holds a
    /// counted reset (keep them the same).
    #[serde(default = "start_reset_counts")]
    pub start_reset_counts: bool,
    /// A flip reset also counts when RocketSim gives the flip back a step after the wheels reach the ball, while
    /// they are still on it (the one-step test misses about a third of real resets at tick skip 4); false: the
    /// one-step test only.
    #[serde(default)]
    pub two_step_resets: bool,
    /// The flip_reset reward's reset_min_air_touches: separate touches since leaving the floor before a reset
    /// counts, 0 = off (keep them the same).
    #[serde(default = "reset_min_air_touches")]
    pub reset_min_air_touches: i32,
    /// The flip_reset reward's reset_max_goal_dist: a reset counts only with the ball this near the opponents' goal
    /// line (uu), 0 = off (keep them the same).
    #[serde(default = "reset_max_goal_dist")]
    pub reset_max_goal_dist: f32,
    /// The flip_reset reward's reset_max_fall_speed: a reset counts only with car and ball falling slower than this
    /// (uu/s), 0 = off (keep them the same).
    #[serde(default = "reset_max_fall_speed")]
    pub reset_max_fall_speed: f32,
    /// The flip_reset reward's reset_min_closing_speed: a reset counts only with the car moving into the ball this
    /// fast (uu/s), 0 = off (keep them the same).
    #[serde(default = "reset_min_closing_speed")]
    pub reset_min_closing_speed: f32,
}

defaults! { ResetMusty {
    reset_only: bool = true,
    wall_margin: f32 = 0.0,
    sling_floor: f32 = 0.5,
    sling_ride_from: f32 = 15.0,
    sling_ride_full: f32 = 70.0,
    scoop_floor: f32 = 0.5,
    release_speed_weight: f32 = 0.0,
    release_speed_zero: f32 = 1389.0,
    release_speed_full: f32 = 4167.0,
    release_speed_curve: f32 = 0.0,
    on_net_bonus: f32 = 0.4,
    on_net_power_mix: f32 = 0.5,
    stretch_shot_weight: f32 = 0.25,
    stretch_shot_from: f32 = 1600.0,
    stretch_shot_full: f32 = 3000.0,
    stretch_shot_curve: f32 = 0.0,
    stretch_gain_weight: f32 = 0.15,
    stretch_gain_from: f32 = 500.0,
    stretch_gain_full: f32 = 1500.0,
    reset_min_ball_height: f32 = 300.0,
    reset_min_air_time: f32 = 0.0,
    reset_min_use_ball_height: f32 = 0.0,
    start_reset_counts: bool = false,
    two_step_resets: bool = false,
    reset_min_air_touches: i32 = 0,
    reset_max_goal_dist: f32 = 0.0,
    reset_max_fall_speed: f32 = 0.0,
    reset_min_closing_speed: f32 = 0.0,
} }

/// Every step while the car holds a counted flip reset: potential-based shaping (gamma x phi' - phi) toward the
/// musty pose (nose down past vertical, roof to the ball, close), phi 0-1; kept when it dodges into a musty, paid
/// back otherwise (at most 1 a dodge).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetMustyPoseShaping {
    /// The pose counts within this car-ball distance...
    #[serde(default = "max_dist")]
    pub max_dist: f32,
    /// ...fully within this one.
    #[serde(default = "full_dist")]
    pub full_dist: f32,
    /// The ball centre at least this high above the car's.
    #[serde(default = "min_ball_above")]
    pub min_ball_above: f32,
    /// Full nose credit this far down past horizontal, toward the ball (deg).
    #[serde(default = "nose_full_deg")]
    pub nose_full_deg: f32,
    /// The roof toward the ball (up . ball direction): 0 credit at this...
    #[serde(default = "roof_zero")]
    pub roof_zero: f32,
    /// ...full credit from this.
    #[serde(default = "roof_full")]
    pub roof_full: f32,
    /// The training discount per step (0.995 at tick skip 4 is a good start).
    #[serde(default = "gamma")]
    pub gamma: f32,
    /// The flip_reset reward's min_ball_height (keep them the same).
    #[serde(default = "pose_reset_min_ball_height")]
    pub reset_min_ball_height: f32,
    /// The flip_reset reward's min_air_time (keep them the same).
    #[serde(default = "pose_reset_min_air_time")]
    pub reset_min_air_time: f32,
    /// The flip_reset reward's min_use_ball_height (keep them the same).
    #[serde(default = "pose_reset_min_use_ball_height")]
    pub reset_min_use_ball_height: f32,
    /// The flip_reset reward's start_reset_counts (keep them the same).
    #[serde(default = "pose_start_reset_counts")]
    pub start_reset_counts: bool,
    /// A flip reset also counts when RocketSim gives the flip back a step after the wheels reach the ball, while
    /// they are still on it (the one-step test misses about a third of real resets at tick skip 4); false: the
    /// one-step test only.
    #[serde(default)]
    pub two_step_resets: bool,
    /// The flip_reset reward's reset_min_air_touches (keep them the same).
    #[serde(default = "pose_reset_min_air_touches")]
    pub reset_min_air_touches: i32,
    /// The flip_reset reward's reset_max_goal_dist (keep them the same).
    #[serde(default = "pose_reset_max_goal_dist")]
    pub reset_max_goal_dist: f32,
    /// The flip_reset reward's reset_max_fall_speed (keep them the same).
    #[serde(default = "pose_reset_max_fall_speed")]
    pub reset_max_fall_speed: f32,
    /// The flip_reset reward's reset_min_closing_speed (keep them the same).
    #[serde(default = "pose_reset_max_fall_speed")]
    pub reset_min_closing_speed: f32,
}

fn pose_reset_min_air_touches() -> i32 {
    0
}
fn pose_reset_max_goal_dist() -> f32 {
    0.0
}
fn pose_reset_max_fall_speed() -> f32 {
    0.0
}

fn max_dist() -> f32 {
    450.0
}
fn full_dist() -> f32 {
    250.0
}
fn min_ball_above() -> f32 {
    -100.0
}
fn nose_full_deg() -> f32 {
    115.0
}
fn roof_zero() -> f32 {
    -1.0
}
fn roof_full() -> f32 {
    0.6
}
fn gamma() -> f32 {
    0.995
}
fn pose_reset_min_ball_height() -> f32 {
    300.0
}
fn pose_reset_min_air_time() -> f32 {
    0.0
}
fn pose_reset_min_use_ball_height() -> f32 {
    0.0
}
fn pose_start_reset_counts() -> bool {
    false
}

/// The flip reset reward these read, with the run's reset settings and its defaults otherwise.
pub(crate) fn resets(
    min_ball_height: f32,
    min_air_time: f32,
    min_use_ball_height: f32,
    start_reset_counts: bool,
    two_step_resets: bool,
    (reset_min_air_touches, reset_max_goal_dist, reset_max_fall_speed, reset_min_closing_speed): (
        i32,
        f32,
        f32,
        f32,
    ),
) -> FlipReset {
    FlipReset {
        min_ball_height,
        min_air_time,
        min_use_ball_height,
        start_reset_counts,
        two_step_resets,
        reset_min_air_touches,
        reset_max_goal_dist,
        reset_max_fall_speed,
        reset_min_closing_speed,
        ..FlipReset::default()
    }
}

// ---- MustyFlickReward, as ResetMustyReward uses it: its settings at their defaults (ResetMusty's own settings
// carry the training options) ----

// ARM: the dodge's start
const ARM_MAX_TORQUE_Y: f32 = -0.4;
const ARM_MIN_NOSE_DOWN_DEG: f32 = 88.0;
const ARM_MAX_NOSE_DOWN_DEG: f32 = 140.0;
const ARM_MAX_BALL_DIST: f32 = 350.0;
const ARM_MIN_BALL_ABOVE: f32 = -100.0;
const ARM_MIN_UP_DOT_BALL: f32 = -0.35;
const ARM_MAX_LATERAL: f32 = 1.0;
// STRIKE: the first ball contact after the dodge
const STRIKE_WINDOW: f32 = 0.7;
const TOP_MIN_UP: f32 = 28.0;
// The outcome
const MIN_IMPULSE: f32 = 300.0;
const FULL_IMPULSE: f32 = 2000.0;
const MIN_LIFT: f32 = 0.0;
const FULL_LIFT: f32 = 600.0;
const FULL_GAIN: f32 = 1000.0;
const FULL_LAUNCH_SPEED: f32 = 2400.0;
const SCOOP_REAR_X: f32 = -30.0;
const SCOOP_FRONT_X: f32 = 40.0;
const SCOOP_LEVEL_DEG: f32 = 30.0;
const SCOOP_ZERO_DEG: f32 = 75.0;
const SHOT_ON_TARGET: f32 = 0.5;
const CORNER_FROM_X: f32 = 250.0;
const CORNER_FULL_X: f32 = 760.0;
const SHOT_MAX_TIME: f32 = 5.0;
const W_POWER: f32 = 0.35;
const W_IMPULSE: f32 = 0.1;
const W_LIFT: f32 = 0.2;
const W_SHOT: f32 = 0.35;
const AIM_COS_ZERO: f32 = 0.0;
const AIM_COS_FULL: f32 = 0.85;
const ELEV_ZERO_LOW: f32 = 0.0;
const ELEV_FULL_LOW: f32 = 8.0;
const ELEV_FULL_HIGH: f32 = 45.0;
const ELEV_ZERO_HIGH: f32 = 70.0;
const AERIAL_ELEV_ZERO_LOW: f32 = -45.0;
const AERIAL_ELEV_FULL_LOW: f32 = -20.0;
const BASE_CREDIT: f32 = 0.25;
const AERIAL_BONUS: f32 = 0.5;
const FLICK_WEIGHT: f32 = 1.0;
const BACKWARD_PENALTY: f32 = 0.5;
const MIN_PENALTY_IMPULSE: f32 = 500.0;
// Cancellations and the rest (arm_credit is 0: nothing is paid when a musty arms)
const CANCEL_ON_CAR_CONTACT: bool = true;
const CAR_CONTACT_MARGIN: f32 = 40.0;
const CAR_CONTACT_LOOKBACK: f32 = 1.0;
const SEPARATION_DIST: f32 = 300.0;
const RESOLVE_TIME: f32 = 0.6;
const RELEASE_STEPS: i32 = 2;
const RIDE_HOVER_GAP: f32 = 45.0;
const RIDE_LEAVE_SPEED: f32 = 500.0;
const MUSTY_WALL_MARGIN: f32 = 200.0;
const SURFACE_MARGIN: f32 = 150.0;
const START_JUMP_MARGIN: f32 = 450.0;
const RESET_MAX_BALL_DIST: f32 = 350.0;

// Where the car's flip came from.
const GROUND: i32 = 0;
const WALL: i32 = 1;
const CEILING: i32 = 2;
const FLIP_RESET: i32 = 3;
const AERIAL_OTHER: i32 = 4;

// The musty attempt's stage.
const IDLE: i32 = 0;
const ARMED: i32 = 1;
const STRUCK: i32 = 2;

kept! {
    /// MustyFlickReward's state for the car (ticks on the episode clock).
    struct MustyTrack {
        /// The last arena surface the wheels touched (GROUND, WALL, CEILING), or AERIAL_OTHER.
        surface: i32,
        /// A flip reset off the ball since then.
        reset_since_surface: bool,
        stage: i32,
        /// The attempt in progress: where its flip came from, when it armed, another car near it.
        context: i32,
        arm_tick: f32,
        car_contact: bool,
        /// The contact point's forward coordinate at the first hit, and at the release (the last contact on the
        /// roof), the car's pitch then and how far the contact rolled along the roof.
        contact_x: f32,
        release_x: f32,
        release_pitch_deg: f32,
        ride_dist: f32,
        strike_tick: f32,
        ball_vel0: Vec3A,
        last_hit_tick: f32,
        /// The ball just after the last touch: velocity (gravity since the hit undone) and position.
        launch_vel: Vec3A,
        touch_ball_pos: Vec3A,
        steps_without_touch: i32,
        last_car_contact_tick: f32,
        any_car_contact: bool,
    }
}

kept! {
    /// ResetMustyReward's own state for the car.
    struct Own {
        /// The car's latest dodge start, and whether it held a counted reset then.
        has_dodge: bool,
        dodge_tick: f32,
        dodge_held_reset: bool,
        /// The ball's height at the first touch after that dodge (-1 before it).
        strike_ball_z: f32,
        /// The last counted reset, and whether the ball has stayed off the walls since it.
        has_reset: bool,
        reset_tick: f32,
        off_walls: bool,
    }
}

impl MustyTrack {
    /// The attempt's own record cleared (MustyFlickReward's fresh MustyEvent).
    fn clear_attempt(&mut self) {
        self.context = GROUND;
        self.arm_tick = 0.0;
        self.car_contact = false;
        self.contact_x = 0.0;
        self.release_x = 0.0;
        self.release_pitch_deg = 0.0;
        self.ride_dist = 0.0;
        self.strike_tick = 0.0;
    }
}

/// A musty's launch, as ResetMustyReward checks it.
#[derive(Clone, Copy)]
struct Launch {
    context: i32,
    arm_tick: f32,
    ride_dist: f32,
    /// The ball's horizontal speed toward the net it attacks as it left the car (uu/s; the last contact's, so a
    /// scoop that rolls along the roof counts from its release).
    goal_speed: f32,
}

/// The nose's angle below horizontal measured toward horizontal direction `d`: 0 level pointing at it, 90 straight
/// down, 180 level pointing away; negative = nose up.
fn nose_down_deg(rot: &Mat3A, d: Vec3A) -> f32 {
    deg((-rot.x_axis.z).atan2(rot.x_axis.dot(dir2d(d))))
}

/// The arena surface the car's wheels are on (3+ wheels down near it), or AERIAL_OTHER.
fn surface_of(car: &CarSnapshot) -> i32 {
    if car.on_ground {
        surface_within(car, SURFACE_MARGIN)
    } else {
        AERIAL_OTHER
    }
}

/// The surface the car is within `m` of (the floor, the ceiling, then the walls), or AERIAL_OTHER.
fn surface_within(car: &CarSnapshot, m: f32) -> i32 {
    let p = car.body.pos;
    if p.z < m {
        GROUND
    } else if p.z > CEILING_Z - m {
        CEILING
    } else if p.x.abs() > SIDE_WALL_X - m || p.y.abs() > BACK_WALL_Y - m {
        WALL
    } else {
        AERIAL_OTHER // Wheels on the ball (a flip reset), not an arena surface
    }
}

/// The surface to credit for a car first seen in this state: the one it is on or, in the air during its first
/// jump, the one it is still within START_JUMP_MARGIN of.
fn start_surface_of(car: &CarSnapshot) -> i32 {
    let c = surface_of(car);
    if c != AERIAL_OTHER
        || car.on_ground
        || !car.has_jumped
        || car.has_flipped
        || car.has_double_jumped
    {
        return c;
    }
    surface_within(car, START_JUMP_MARGIN)
}

/// A car first seen in the air with its flip and no jump used, away from the surfaces and near the ball, not long
/// off its last wheel contact: its flip came back off the ball.
fn starts_after_flip_reset(car: &CarSnapshot, detail: &CarDetail, ball: Vec3A) -> bool {
    surface_of(car) == AERIAL_OTHER && holds_reset_at_start(car, detail, ball)
}

/// The ARM test at a dodge's start: a backflip or back diagonal in the air, no touch this step, the nose 88-140 deg
/// down toward the ball, the ball close, not below the car, on the roof side.
pub(crate) fn arm_pose_ok(car: &CarSnapshot, torque: Vec3A, touched: bool, ball: Vec3A) -> bool {
    if car.on_ground || touched || torque.y > ARM_MAX_TORQUE_Y {
        return false;
    }
    let to_ball = ball - car.body.pos;
    let dist = to_ball.length();
    let nose_down = nose_down_deg(&car.body.rot, to_ball);
    let (up_dot_ball, lateral) = if dist > 1.0 {
        (
            car.body.up().dot(to_ball / dist),
            car.body.rot.y_axis.dot(to_ball / dist).abs(),
        )
    } else {
        (0.0, 0.0)
    };
    (ARM_MIN_NOSE_DOWN_DEG..=ARM_MAX_NOSE_DOWN_DEG).contains(&nose_down)
        && dist <= ARM_MAX_BALL_DIST
        && to_ball.z >= ARM_MIN_BALL_ABOVE
        && up_dot_ball >= ARM_MIN_UP_DOT_BALL
        && lateral <= ARM_MAX_LATERAL
}

/// Do two oriented boxes overlap (the separating-axis test, 15 axes)?
fn boxes_overlap(ca: Vec3A, ra: &Mat3A, ha: Vec3A, cb: Vec3A, rb: &Mat3A, hb: Vec3A) -> bool {
    let (a_axes, b_axes) = (
        [ra.x_axis, ra.y_axis, ra.z_axis],
        [rb.x_axis, rb.y_axis, rb.z_axis],
    );
    let (a, b) = (ha.to_array(), hb.to_array());
    let mut r = [[0.0f32; 3]; 3];
    let mut ar = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a_axes[i].dot(b_axes[j]);
            ar[i][j] = r[i][j].abs() + 1e-6;
        }
    }
    let d = cb - ca;
    let t = [d.dot(a_axes[0]), d.dot(a_axes[1]), d.dot(a_axes[2])];
    for i in 0..3 {
        if t[i].abs() > a[i] + b[0] * ar[i][0] + b[1] * ar[i][1] + b[2] * ar[i][2] {
            return false;
        }
    }
    for j in 0..3 {
        let along = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        if along.abs() > a[0] * ar[0][j] + a[1] * ar[1][j] + a[2] * ar[2][j] + b[j] {
            return false;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
            let ra_ = a[i1] * ar[i2][j] + a[i2] * ar[i1][j];
            let rb_ = b[j1] * ar[i][j2] + b[j2] * ar[i][j1];
            if (t[i2] * r[i1][j] - t[i1] * r[i2][j]).abs() > ra_ + rb_ {
                return false;
            }
        }
    }
    true
}

/// Another car within CAR_CONTACT_MARGIN of the hitbox, or a bump between them still cooling down.
fn car_contact(game: &GameSnapshot, events: &StepEvents, car: usize) -> bool {
    let half = HITBOX_SIZE * 0.5 + Vec3A::splat(CAR_CONTACT_MARGIN * 0.5);
    let centre = |q: &CarSnapshot| q.body.pos + q.body.rot * HITBOX_OFFSET;
    let p = &game.cars[car];
    let mine = events.detail(car);
    (0..game.cars.len()).any(|other| {
        let q = &game.cars[other];
        if other == car || q.is_demoed {
            return false;
        }
        let theirs = events.detail(other);
        (theirs.last_contact_car == Some(car) && theirs.bump_cooldown > 0.0)
            || (mine.last_contact_car == Some(other) && mine.bump_cooldown > 0.0)
            || boxes_overlap(centre(p), &p.body.rot, half, centre(q), &q.body.rot, half)
    })
}

/// The car's pose at its last ball hit (the step-end pose wound back with constant linear and angular velocity),
/// and the hitbox point closest to the ball then, in the car's frame; with the car's pitch then (deg).
fn hit_contact_local(car: &Body, hit: &Hit) -> (Vec3A, f32) {
    let back = hit.rewind_ticks * TICK_SECONDS;
    let pos = car.pos - car.vel * back;
    let w = car.ang_vel * -back;
    let angle = w.length();
    let rot = if angle > 1e-6 {
        Mat3A::from_axis_angle((w / angle).into(), angle) * car.rot
    } else {
        car.rot
    };
    let pitch = deg(rot.x_axis.z.clamp(-1.0, 1.0).asin());
    let d = hit.ball_pos - pos;
    let ball_local =
        Vec3A::new(d.dot(rot.x_axis), d.dot(rot.y_axis), d.dot(rot.z_axis)) - HITBOX_OFFSET;
    let half = HITBOX_SIZE * 0.5;
    (ball_local.clamp(-half, half) + HITBOX_OFFSET, pitch)
}

/// The gap between the ball and the car's top face now (f32::MAX when the ball is not over it), and how fast the
/// ball moves away from the face's nearest point (which moves as the car spins).
fn roof_gap(car: &Body, ball: &Body) -> (f32, f32) {
    let r = &car.rot;
    let d = ball.pos - car.pos;
    let l = Vec3A::new(d.dot(r.x_axis), d.dot(r.y_axis), d.dot(r.z_axis)) - HITBOX_OFFSET;
    let half = HITBOX_SIZE * 0.5;
    let closest = l.clamp(-half, half);
    if closest.z + HITBOX_OFFSET.z < TOP_MIN_UP {
        return (f32::MAX, 0.0);
    }
    let n = l - closest;
    let len = n.length();
    let n = if len > 1e-3 { n / len } else { Vec3A::Z };
    let c = closest + HITBOX_OFFSET;
    let point_off = r.x_axis * c.x + r.y_axis * c.y + r.z_axis * c.z;
    let point_vel = car.vel + car.ang_vel.cross(point_off);
    let away = (ball.vel - point_vel).dot(r.x_axis * n.x + r.y_axis * n.y + r.z_axis * n.z);
    (len - RLGYM_BALL_RADIUS, away)
}

/// Side and back walls only (an aerial musty under the ceiling is still attributable).
fn musty_near_wall(p: Vec3A) -> bool {
    let m = MUSTY_WALL_MARGIN + RLGYM_BALL_RADIUS;
    p.x.abs() > SIDE_WALL_X - m
        || (p.y.abs() > BACK_WALL_Y - m && (p.x.abs() > GOAL_HALF_WIDTH || p.z > GOAL_HEIGHT))
}

/// The ball just after a touch this step: its velocity with the gravity since the hit undone, and where it is.
fn record_touch(mt: &mut MustyTrack, ball: &Body, now: f32) {
    let since_hit = (now - mt.last_hit_tick).max(0.0);
    mt.launch_vel = ball.vel - Vec3A::new(0.0, 0.0, GRAVITY_Z * since_hit * TICK_SECONDS);
    mt.touch_ball_pos = ball.pos;
}

/// How an attempt ends.
#[derive(PartialEq)]
enum Outcome {
    Launch,
    CarContact,
    Nothing,
}

impl ResetMusty {
    fn sling_mult(&self, ride_dist: f32) -> f32 {
        self.sling_floor
            + (1.0 - self.sling_floor) * ramp(ride_dist, self.sling_ride_from, self.sling_ride_full)
    }

    fn ball_off_walls(&self, b: Vec3A) -> bool {
        let m = self.wall_margin + RLGYM_BALL_RADIUS;
        let goal_mouth = b.x.abs() < GOAL_HALF_WIDTH && b.z < GOAL_HEIGHT;
        b.x.abs() <= SIDE_WALL_X - m
            && (b.y.abs() <= BACK_WALL_Y - m || goal_mouth)
            && b.z < CEILING_Z - RLGYM_BALL_RADIUS - 10.0
    }

    /// Ends the attempt: a launch (or one cancelled by another car, judged but not paid) is paid on the ball as it
    /// left the car.
    fn finish(
        &self,
        mt: &mut MustyTrack,
        team: Team,
        ball: &Body,
        outcome: Outcome,
    ) -> (f32, Option<Launch>) {
        let goal_dir = dir2d(attack_net_back(team) - mt.touch_ball_pos);
        let launch = Launch {
            context: mt.context,
            arm_tick: mt.arm_tick,
            ride_dist: mt.ride_dist,
            goal_speed: Vec3A::new(mt.launch_vel.x, mt.launch_vel.y, 0.0).dot(goal_dir),
        };
        let done = *mt;
        mt.stage = IDLE;
        mt.clear_attempt();
        if outcome == Outcome::Nothing {
            return (0.0, None);
        }
        let goal_dir = dir2d(attack_net_back(team) - done.touch_ball_pos);
        let v1 = done.launch_vel;
        let dv = v1 - done.ball_vel0;
        let horiz_speed = hspeed(v1);
        let flat = |v: Vec3A| Vec3A::new(v.x, v.y, 0.0);
        let impulse = dv.length();
        let lift = dv.z;
        let gain = flat(dv).dot(goal_dir);
        let final_goal_speed = flat(v1).dot(goal_dir);
        let elevation = deg(v1.z.atan2(horiz_speed));
        let aim_cos = if horiz_speed > 1.0 {
            flat(v1).dot(goal_dir) / horiz_speed
        } else {
            0.0
        };
        let power_term =
            (ramp(gain, 0.0, FULL_GAIN) * clamp01(final_goal_speed / FULL_LAUNCH_SPEED)).sqrt();
        let impulse_term = ramp(impulse, MIN_IMPULSE, FULL_IMPULSE);
        let lift_term = ramp(lift, MIN_LIFT, FULL_LIFT);
        let scoop_term = ramp(done.release_x, SCOOP_REAR_X, SCOOP_FRONT_X)
            * (1.0
                - ramp(
                    done.release_pitch_deg.abs(),
                    SCOOP_LEVEL_DEG,
                    SCOOP_ZERO_DEG,
                ));
        // The shot from the ball's state now
        let shot = project_to_goal_line(ball.pos, ball.vel, team, SHOT_MAX_TIME);
        let (on_target, shot_x) = (shot.on_target, shot.x);
        let shot_term = if on_target {
            SHOT_ON_TARGET
                + (1.0 - SHOT_ON_TARGET) * ramp(shot_x.abs(), CORNER_FROM_X, CORNER_FULL_X)
        } else {
            0.0
        };
        let quality = W_POWER * power_term
            + W_IMPULSE * impulse_term
            + W_LIFT * lift_term
            + W_SHOT * shot_term;
        let aim = ramp(aim_cos, AIM_COS_ZERO, AIM_COS_FULL);
        let in_air = matches!(done.context, CEILING | FLIP_RESET | AERIAL_OTHER); // A downward shot is fine
        let (lo0, lo1) = if in_air {
            (AERIAL_ELEV_ZERO_LOW, AERIAL_ELEV_FULL_LOW)
        } else {
            (ELEV_ZERO_LOW, ELEV_FULL_LOW)
        };
        let elev = if elevation < lo1 {
            ramp(elevation, lo0, lo1)
        } else {
            clamp01((ELEV_ZERO_HIGH - elevation) / (ELEV_ZERO_HIGH - ELEV_FULL_HIGH))
        };
        let context_mult = if done.context != GROUND {
            1.0 + AERIAL_BONUS
        } else {
            1.0
        };
        let scoop_mult = self.scoop_floor + (1.0 - self.scoop_floor) * scoop_term;
        let success = impulse >= MIN_IMPULSE && final_goal_speed > 0.0 && gain > -MIN_IMPULSE;
        let mut reward = 0.0;
        if success {
            reward = FLICK_WEIGHT * (BASE_CREDIT + (1.0 - BASE_CREDIT) * quality) * aim * elev;
            let shot_power = speed_curve(
                ramp(
                    final_goal_speed,
                    self.stretch_shot_from,
                    self.stretch_shot_full,
                ),
                self.stretch_shot_curve,
            );
            if on_target {
                reward += FLICK_WEIGHT
                    * self.on_net_bonus
                    * elev
                    * (self.on_net_power_mix + (1.0 - self.on_net_power_mix) * shot_power);
            }
            let stretch = FLICK_WEIGHT
                * aim
                * elev
                * (self.stretch_gain_weight
                    * ramp(gain, self.stretch_gain_from, self.stretch_gain_full)
                    + if on_target {
                        self.stretch_shot_weight * shot_power
                    } else {
                        0.0
                    });
            reward += stretch;
            // The context and the scoop scale everything a musty earns, the training terms included
            reward *= context_mult * scoop_mult;
        }
        if final_goal_speed < 0.0 && flat(dv).length() >= MIN_PENALTY_IMPULSE && !on_target {
            reward -= BACKWARD_PENALTY * clamp01(-final_goal_speed / FULL_GAIN);
        }
        if outcome == Outcome::CarContact {
            return (0.0, None); // Judged, not credited
        }
        (reward, Some(launch))
    }

    /// One step of MustyFlickReward for the car: its pay and, when one was paid, the launch.
    fn musty_step(
        &self,
        mt: &mut MustyTrack,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        now: f32,
    ) -> (f32, Option<Launch>) {
        let (p, pp, ball) = (&cur.cars[car], &prev.cars[car], &cur.ball);
        let detail = events.detail(car);
        let since = |tick: f32| (now - tick).max(0.0);
        let sec_to_ticks = |sec: f32| (sec / TICK_SECONDS).round();
        if p.is_demoed {
            if mt.stage != IDLE {
                return self.finish(mt, p.team, ball, Outcome::Nothing);
            }
            return (0.0, None);
        }

        // ---- Where the flip comes from ----
        let surface = surface_of(p);
        if surface != AERIAL_OTHER {
            mt.surface = surface;
            mt.reset_since_surface = false;
        }
        if mt.stage != IDLE && (0..cur.cars.len()).any(|o| o != car && events.cars[o].touched_ball)
        {
            return self.finish(mt, p.team, ball, Outcome::Nothing); // Another car touched the ball
        }
        if cur.cars.len() > 1 && car_contact(cur, events, car) {
            mt.last_car_contact_tick = now;
            mt.any_car_contact = true;
            if mt.stage != IDLE {
                mt.car_contact = true;
            }
        }
        let touched = events.cars[car].touched_ball;
        let hit = last_hit(events, car, now).unwrap_or(Hit {
            tick: now - 1.0,
            ball_pos: ball.pos,
            rewind_ticks: 0.0,
        });

        // A flip reset off the ball: no flip left, then the jump and flip back away from every arena surface
        let had_no_flip = !pp.on_ground && !pp.can_flip;
        let reset_now = !p.has_jumped && !p.has_flipped && !p.has_double_jumped;
        if had_no_flip
            && reset_now
            && !near_arena_surface(p.body.pos, 120.0)
            && !near_arena_surface(pp.body.pos, 120.0)
            && p.body.pos.distance(ball.pos) <= RESET_MAX_BALL_DIST
        {
            mt.reset_since_surface = true;
        }

        let goal = cur.goal_scored.is_some();
        if mt.stage == IDLE {
            let dodge_started = p.has_flipped && !pp.has_flipped;
            if dodge_started && arm_pose_ok(p, detail.flip_rel_torque, touched, ball.pos) {
                mt.clear_attempt();
                mt.stage = ARMED;
                mt.arm_tick = now;
                mt.context = if mt.reset_since_surface {
                    FLIP_RESET
                } else {
                    mt.surface
                };
                mt.car_contact = mt.any_car_contact
                    && since(mt.last_car_contact_tick) <= sec_to_ticks(CAR_CONTACT_LOOKBACK);
            }
            return (0.0, None);
        }
        if mt.stage == ARMED {
            // Landing on (or bumping) any arena surface before the ball ends it
            let landed = surface != AERIAL_OTHER
                || (detail.world_contact_normal.is_some()
                    && near_arena_surface(p.body.pos, SURFACE_MARGIN));
            if landed {
                return self.finish(mt, p.team, ball, Outcome::Nothing);
            }
            if touched {
                let (local_hit, pitch) = hit_contact_local(&p.body, &hit);
                mt.contact_x = local_hit.x;
                mt.release_x = local_hit.x;
                mt.release_pitch_deg = pitch;
                mt.steps_without_touch = 0;
                if local_hit.z < TOP_MIN_UP {
                    return self.finish(mt, p.team, ball, Outcome::Nothing); // Not the top of the car
                }
                mt.stage = STRUCK;
                mt.strike_tick = now;
                mt.ball_vel0 = prev.ball.vel;
                mt.last_hit_tick = hit.tick + 1.0;
                record_touch(mt, ball, now);
            } else if since(mt.arm_tick) >= sec_to_ticks(STRIKE_WINDOW) {
                return self.finish(mt, p.team, ball, Outcome::Nothing); // No strike in time
            }
            if !goal || mt.stage != STRUCK {
                return (0.0, None);
            }
        }
        // STRUCK: follow the contact along the roof to where the ball leaves, then judge the launch
        if touched && now != mt.strike_tick {
            mt.last_hit_tick = hit.tick + 1.0;
            record_touch(mt, ball, now);
            let (local_hit, pitch) = hit_contact_local(&p.body, &hit);
            if local_hit.z >= TOP_MIN_UP {
                mt.release_x = local_hit.x;
                mt.release_pitch_deg = pitch;
                mt.ride_dist = mt.release_x - mt.contact_x;
            }
        }
        if now != mt.strike_tick {
            mt.steps_without_touch = if touched {
                0
            } else {
                mt.steps_without_touch + 1
            };
        }
        let (gap, away) = roof_gap(&p.body, ball);
        let on_roof = gap <= RIDE_HOVER_GAP
            && away < RIDE_LEAVE_SPEED
            && !near_arena_surface(ball.pos, RLGYM_BALL_RADIUS + 10.0);
        let contact_over = mt.steps_without_touch >= RELEASE_STEPS && !on_roof;
        let separated = p.body.pos.distance(ball.pos) >= SEPARATION_DIST;
        let time_up = since(mt.strike_tick) >= sec_to_ticks(RESOLVE_TIME);
        if contact_over || separated || time_up || goal {
            if musty_near_wall(mt.touch_ball_pos) && !goal {
                return self.finish(mt, p.team, ball, Outcome::Nothing); // Left the car next to a wall
            }
            let outcome = if CANCEL_ON_CAR_CONTACT && mt.car_contact {
                Outcome::CarContact
            } else {
                Outcome::Launch
            };
            return self.finish(mt, p.team, ball, outcome);
        }
        (0.0, None)
    }
}

impl ResetMusty {
    /// How many numbers of memory `step` takes.
    pub const MEMORY: usize = Clock::SIZE + FlipReset::MEMORY + MustyTrack::SIZE + Own::SIZE;

    /// One step of the reward with its memory: what `value_remembering` pays.
    pub fn step(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (clock_mem, rest) = memory.split_at_mut(Clock::SIZE);
        let (flip_mem, rest) = rest.split_at_mut(FlipReset::MEMORY);
        let (musty_mem, own_mem) = rest.split_at_mut(MustyTrack::SIZE);
        let mut clock = Clock::load(clock_mem);
        let first = clock.advance(events);
        let now = clock.now;
        // The flip reset evaluator, stepped before this reward in the same step
        resets(
            self.reset_min_ball_height,
            self.reset_min_air_time,
            self.reset_min_use_ball_height,
            self.start_reset_counts,
            self.two_step_resets,
            (
                self.reset_min_air_touches,
                self.reset_max_goal_dist,
                self.reset_max_fall_speed,
                self.reset_min_closing_speed,
            ),
        )
        .step(prev, cur, events, car, flip_mem);
        let (mut mt, mut own) = if first {
            // The episode's start: the musty labels its flip from the starting state
            let start = &prev.cars[car];
            let mt = MustyTrack {
                surface: start_surface_of(start),
                reset_since_surface: starts_after_flip_reset(
                    start,
                    &events.prev_detail(car),
                    prev.ball.pos,
                ),
                ..MustyTrack::default()
            };
            let own = Own {
                strike_ball_z: -1.0,
                off_walls: true,
                ..Own::default()
            };
            (mt, own)
        } else {
            (MustyTrack::load(musty_mem), Own::load(own_mem))
        };
        let (p, pp, ball) = (&cur.cars[car], &prev.cars[car], &cur.ball);

        // Each dodge's start, and whether the car held a counted reset then (before the outcome resolves)
        if p.has_flipped && !pp.has_flipped {
            own.has_dodge = true;
            own.dodge_tick = now;
            own.dodge_held_reset = FlipReset::holds_counted_reset(flip_mem);
            own.strike_ball_z = -1.0;
        } else if own.has_dodge && own.strike_ball_z < 0.0 && events.cars[car].touched_ball {
            own.strike_ball_z = ball.pos.z; // The strike
        }
        if self.wall_margin > 0.0 {
            if let Some(tick) = FlipReset::counted_reset_tick(flip_mem)
                && (!own.has_reset || tick != own.reset_tick)
            {
                own.has_reset = true; // A new counted reset
                own.reset_tick = tick;
                own.off_walls = true;
            }
            if own.has_reset && !self.ball_off_walls(ball.pos) {
                own.off_walls = false;
            }
        }

        let (mut out, launch) = self.musty_step(&mut mt, prev, cur, events, car, now);
        if out != 0.0 {
            let mut ok = !self.reset_only;
            if let (true, Some(l)) = (self.reset_only, launch) {
                ok = l.context == FLIP_RESET
                    && own.has_dodge
                    && own.dodge_tick == l.arm_tick
                    && own.dodge_held_reset
                    && own.strike_ball_z >= self.reset_min_use_ball_height; // Hit with the ball high enough
            }
            if ok && self.wall_margin > 0.0 && !own.off_walls {
                ok = false;
            }
            if !ok {
                out = 0.0;
            } else if let (true, Some(l)) = (out > 0.0, launch) {
                out *= self.sling_mult(l.ride_dist); // (a backward shot's penalty stays as it is)
                // The musty's own speed toward the net as it leaves the car, paid at once
                out += self.release_speed_weight
                    * speed_curve(
                        ramp(
                            l.goal_speed,
                            self.release_speed_zero,
                            self.release_speed_full,
                        ),
                        self.release_speed_curve,
                    );
            }
        }
        clock.store(clock_mem);
        mt.store(musty_mem);
        own.store(own_mem);
        finite_or_zero(out)
    }
}

impl Reward for ResetMusty {
    fn memory(&self) -> usize {
        Self::MEMORY
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        self.step(prev, cur, events, car, memory)
    }
}

impl ResetMustyPoseShaping {
    /// The pose's potential, 0-1: only while the car holds a counted reset it hasn't used.
    fn phi(&self, car: &CarSnapshot, detail: &CarDetail, ball: Vec3A, flip_mem: &[f32]) -> f32 {
        if !FlipReset::holds_counted_reset(flip_mem) || !has_dodge(car, detail) || car.is_demoed {
            return 0.0;
        }
        let to = ball - car.body.pos;
        let d = to.length();
        if d > self.max_dist || to.z < self.min_ball_above || d < 1.0 {
            return 0.0;
        }
        ramp(nose_down_deg(&car.body.rot, to), 0.0, self.nose_full_deg)
            * ramp(car.body.up().dot(to / d), self.roof_zero, self.roof_full)
            * ramp(d, self.max_dist, self.full_dist)
    }
}

impl Reward for ResetMustyPoseShaping {
    fn memory(&self) -> usize {
        Clock::SIZE + FlipReset::MEMORY + 1
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (clock_mem, rest) = memory.split_at_mut(Clock::SIZE);
        let (flip_mem, phi_mem) = rest.split_at_mut(FlipReset::MEMORY);
        let mut clock = Clock::load(clock_mem);
        if clock.advance(events) {
            phi_mem[0] = 0.0; // No reset held at the start
        }
        clock.store(clock_mem);
        resets(
            self.reset_min_ball_height,
            self.reset_min_air_time,
            self.reset_min_use_ball_height,
            self.start_reset_counts,
            self.two_step_resets,
            (
                self.reset_min_air_touches,
                self.reset_max_goal_dist,
                self.reset_max_fall_speed,
                self.reset_min_closing_speed,
            ),
        )
        .step(prev, cur, events, car, flip_mem);
        let (p, pp) = (&cur.cars[car], &prev.cars[car]);
        let detail = events.detail(car);
        let phi = phi_mem[0];
        let out = if p.has_flipped && !pp.has_flipped {
            // Banked when the dodge is one the musty arms on; paid back for any other
            let musty = arm_pose_ok(
                p,
                detail.flip_rel_torque,
                events.cars[car].touched_ball,
                cur.ball.pos,
            );
            phi_mem[0] = 0.0;
            if phi > 0.0 && musty { 0.0 } else { -phi }
        } else {
            let next = self.phi(p, &detail, cur.ball.pos, flip_mem);
            phi_mem[0] = next;
            self.gamma * next - phi
        };
        finite_or_zero(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    const BALL: Vec3A = Vec3A::new(0.0, 0.0, 1000.0);

    /// A car at `ball + rel` in the air with its axes (forward, right, up), all in Blue's view (mirrored for
    /// Orange); flags: (on_ground, has_jumped, has_flipped, can_flip).
    fn at(
        team: Team,
        rel: Vec3A,
        axes: [Vec3A; 3],
        flags: (bool, bool, bool, bool),
    ) -> GameSnapshot {
        let m = |v: Vec3A| as_blue(team, v);
        let mut g = game(m(BALL), &[(team, m(BALL + rel))]);
        let c = &mut g.cars[0];
        c.body.rot = Mat3A::from_cols(m(axes[0]), m(axes[1]), m(axes[2]));
        (c.on_ground, c.has_jumped, c.has_flipped, c.can_flip) = flags;
        g
    }

    fn upside_down() -> [Vec3A; 3] {
        [Vec3A::X, -Vec3A::Y, -Vec3A::Z]
    }

    /// Nose 100 deg down toward the ball (+y), roof toward it.
    fn musty_pose() -> [Vec3A; 3] {
        let a = 100f32.to_radians();
        let forward = Vec3A::new(0.0, a.cos(), -a.sin());
        let up = Vec3A::new(0.0, a.sin(), a.cos());
        [forward, up.cross(forward), up]
    }

    const USED: (bool, bool, bool, bool) = (false, true, true, false);
    const RESET: (bool, bool, bool, bool) = (true, false, false, true);
    const READY: (bool, bool, bool, bool) = (false, false, false, true);
    const DODGED: (bool, bool, bool, bool) = (false, false, true, false);

    /// A reset under the ball, the turn into the musty pose, the backflip, a strike on the roof, the ball away.
    pub(crate) fn reset_musty(
        team: Team,
        with_reset: bool,
    ) -> (Vec<GameSnapshot>, Vec<StepEvents>) {
        let m = |v: Vec3A| as_blue(team, v);
        let below = Vec3A::new(0.0, -150.0, -150.0);
        let pose = Vec3A::new(0.0, -150.0, -100.0);
        let (first, second) = if with_reset {
            (USED, RESET)
        } else {
            ((false, true, false, true), (false, true, false, true))
        };
        let mut games = vec![
            at(team, below, upside_down(), first),
            at(team, below, upside_down(), second),
            at(
                team,
                pose,
                musty_pose(),
                if with_reset { READY } else { second },
            ),
            at(team, pose, musty_pose(), DODGED),
        ];
        // The strike: the ball on the roof, 30 uu forward of the centre, leaving at (0, 2000, 500)
        let mut strike = at(team, pose, musty_pose(), DODGED);
        let [f, _, u] = musty_pose();
        let car = BALL + pose;
        strike.ball.pos = m(car + f * 30.0 + u * 131.0);
        strike.ball.vel = m(Vec3A::new(0.0, 2000.0, 500.0));
        games.push(strike.clone());
        let mut away = strike.clone();
        away.ball.pos = m(car + Vec3A::new(0.0, 400.0, 150.0));
        games.push(away);
        let mut events = Vec::new();
        for i in 0..games.len() - 1 {
            let mut ev = quiet(1);
            ev.step_seconds = 4.0 / 120.0;
            ev.tick = 1000 + 4 * i as u64;
            ev.details[0].flip_rel_torque = Vec3A::new(0.0, -1.0, 0.0);
            if i == 0 && with_reset {
                ev.cars[0].touched_ball = true; // The wheels on the ball
            }
            if i == 3 {
                ev.cars[0].touched_ball = true;
                ev.details[0].last_hit_tick = Some(ev.tick);
                ev.details[0].last_hit_ball_pos = games[4].ball.pos;
            }
            events.push(ev);
        }
        (games, events)
    }

    fn run(r: &impl Reward, games: &[GameSnapshot], events: &[StepEvents]) -> Vec<f32> {
        let mut memory = vec![0.0; r.memory()];
        games
            .windows(2)
            .zip(events)
            .map(|(pair, ev)| r.value_remembering(&pair[0], &pair[1], ev, 0, &mut memory))
            .collect()
    }

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        assert_listed("reset_musty", 28);
        assert_listed("reset_musty_pose_shaping", 16);
    }

    fn reward() -> ResetMusty {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    #[test]
    fn a_musty_off_a_counted_reset_pays_and_orange_the_same() {
        let r = reward();
        let (games, events) = reset_musty(Team::Blue, true);
        let blue = run(&r, &games, &events);
        assert!(
            blue[..4].iter().all(|&v| v == 0.0),
            "paid at the outcome only: {blue:?}"
        );
        // A roof push (the car still steep at the hit: scoop 0, x0.5), aerial (x1.5), no ride (x0.5 smack)
        assert!(blue[4] > 0.2 && blue[4] < 1.5, "{blue:?}");
        let (games, events) = reset_musty(Team::Orange, true);
        let orange = run(&r, &games, &events);
        assert!((blue[4] - orange[4]).abs() < 1e-4, "{blue:?} vs {orange:?}");
    }

    /// release_speed_weight adds its weight for the musty's goalward speed off the car: the scripted musty
    /// sends the ball at the net, so with the full point below its speed it pays exactly the weight more, the
    /// same for Orange; above its speed nothing more.
    #[test]
    fn release_speed_weight_pays_the_musty_speed_off_the_car() {
        let base = |team| {
            let (games, events) = reset_musty(team, true);
            run(&reward(), &games, &events)[4]
        };
        let with = |team, zero: f32, full: f32| {
            let r = ResetMusty {
                release_speed_weight: 0.7,
                release_speed_zero: zero,
                release_speed_full: full,
                ..reward()
            };
            let (games, events) = reset_musty(team, true);
            run(&r, &games, &events)[4]
        };
        for team in [Team::Blue, Team::Orange] {
            assert!(base(team) > 0.0);
            assert!(
                (with(team, 0.0, 1.0) - base(team) - 0.7).abs() < 1e-4,
                "full: +0.7"
            );
            assert!(
                (with(team, 9000.0, 9999.0) - base(team)).abs() < 1e-6,
                "too slow: nothing"
            );
        }
    }

    /// scoop_floor sets what the scripted roof push (scoop 0) is worth: its pay scales with the floor.
    #[test]
    fn scoop_floor_sets_what_a_roof_push_is_worth() {
        let (games, events) = reset_musty(Team::Blue, true);
        let half = run(&reward(), &games, &events)[4];
        let quarter = run(
            &ResetMusty {
                scoop_floor: 0.25,
                ..reward()
            },
            &games,
            &events,
        )[4];
        assert!(
            half > 0.0 && (quarter - 0.5 * half).abs() < 1e-4,
            "{half} vs {quarter}"
        );
    }

    #[test]
    fn a_musty_without_a_reset_pays_only_when_every_musty_is_wanted() {
        let (games, events) = reset_musty(Team::Blue, false);
        let only = run(&reward(), &games, &events);
        assert!(only.iter().all(|&v| v == 0.0), "{only:?}");
        let any = ResetMusty {
            reset_only: false,
            ..reward()
        };
        let every = run(&any, &games, &events);
        assert!(every[4] > 0.0, "{every:?}");
        let high = ResetMusty {
            reset_min_use_ball_height: 5000.0,
            ..reward()
        };
        let (games, events) = reset_musty(Team::Blue, true);
        assert_eq!(
            run(&high, &games, &events)[4],
            0.0,
            "the hit with the ball too low"
        );
    }

    /// A game that starts on the reset itself (the wheels on the ball): its musty pays, and the turn into the pose
    /// is shaped, only with start_reset_counts.
    #[test]
    fn a_musty_off_a_reset_held_from_the_start_pays_with_start_reset_counts() {
        let (games, events) = reset_musty(Team::Blue, true);
        let (games, events) = (&games[1..], &events[1..]);
        assert!(run(&reward(), games, events).iter().all(|&v| v == 0.0));
        let counts = ResetMusty {
            start_reset_counts: true,
            ..reward()
        };
        let paid = run(&counts, games, events);
        assert!(
            paid[3] > 0.2 && paid[..3].iter().all(|&v| v == 0.0),
            "{paid:?}"
        );
        let pose = |start_reset_counts| {
            let r = ResetMustyPoseShaping {
                start_reset_counts,
                ..serde_json::from_value(serde_json::json!({})).unwrap()
            };
            run(&r, games, events)
        };
        assert_eq!(pose(false)[0], 0.0);
        assert!(pose(true)[0] > 0.0, "{:?}", pose(true));
    }

    #[test]
    fn the_pose_shaping_rises_into_the_pose_and_is_kept_on_the_musty_dodge() {
        let r: ResetMustyPoseShaping = serde_json::from_value(serde_json::json!({})).unwrap();
        let (games, events) = reset_musty(Team::Blue, true);
        let out = run(&r, &games, &events);
        assert_eq!(out[0], 0.0, "no reset held yet: {out:?}");
        assert!(out[1] > 0.0, "the turn into the pose: {out:?}");
        assert_eq!(out[2], 0.0, "banked on the musty dodge: {out:?}");
        // The same dodge as a front flip pays the shaping back
        let mut front = events.clone();
        front[2].details[0].flip_rel_torque = Vec3A::new(0.0, 1.0, 0.0);
        let back = run(&r, &games, &front);
        assert!((back[2] + back[1] / r.gamma).abs() < 1e-4, "{back:?}");
        let (plain, events) = reset_musty(Team::Blue, false);
        assert!(run(&r, &plain, &events).iter().all(|&v| v == 0.0));
    }

    /// Every reward of the trick-move files (this one, flip_reset, air_dribble, mawkzy_flick,
    /// speed_flip) and the air sequence rewards in real 1v1 RocketSim games with random actions: no panic, every
    /// reward finite.
    #[test]
    fn random_play_with_every_trick_reward_stays_finite() {
        use crate::actions::NUM_ACTIONS;
        use crate::driver::PoolDriver;
        use crate::testing::{meshes_loaded, pool_config};
        if !meshes_loaded() {
            return;
        }
        let names = [
            "air_dribble",
            "flip_reset",
            "reset_musty",
            "reset_musty_pose_shaping",
            "mawkzy_flick",
            "speed_flip",
        ];
        let mut config = pool_config(4, 2, 2, 11);
        config.rewards = names
            .iter()
            .map(|name| {
                serde_json::from_value(serde_json::json!({
                    "kind": { "name": "custom", "kind": name, "settings": {} },
                    "weight": 1.0,
                    "zero_sum": null,
                }))
                .unwrap()
            })
            .collect();
        let mut driver = PoolDriver::new(&config).unwrap();
        let mut rng = fastrand::Rng::with_seed(3);
        for _ in 0..3000 {
            driver
                .step(|actions| actions.fill_with(|| rng.i32(0..NUM_ACTIONS as i32)))
                .unwrap();
            for owned in &driver.buffers {
                assert!(owned.reward.iter().all(|v| v.is_finite()));
            }
        }
        let stats = driver.pool.take_stats();
        for (name, paid) in names.iter().zip(&stats.rewards.raw) {
            assert!(paid.is_finite(), "{name}");
            eprintln!(
                "{name}: {paid:.3} over {} agent steps",
                stats.rewards.agent_steps
            );
        }
    }
}
