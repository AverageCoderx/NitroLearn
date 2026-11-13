//! Helpers shared by the flick rewards (musty, Raze and their shaping): the geometry they share. No
//! reward here.

use crate::custom::prelude::*;

/// RLGym's ball radius, which those rewards were written with (the kit's is 91.25).
pub const CV_BALL_RADIUS: f32 = 92.75;
/// The game's gravity (uu/s², along z).
pub const GRAVITY_Z: f32 = -650.0;
const DEG: f32 = 180.0 / std::f32::consts::PI;

pub fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// 0 at `lo`, 1 at `hi`, clamped (either way round); a step at `hi` when the two are equal.
pub fn ramp(v: f32, lo: f32, hi: f32) -> f32 {
    if hi == lo {
        return if v >= hi { 1.0 } else { 0.0 };
    }
    clamp01((v - lo) / (hi - lo))
}

/// `a / b`, 0 when `b` is 0.
pub fn div(a: f32, b: f32) -> f32 {
    if b == 0.0 { 0.0 } else { a / b }
}

pub fn degrees(radians: f32) -> f32 {
    radians * DEG
}

/// The horizontal direction of `v` (0 when it has none).
pub fn dir2d(v: Vec3A) -> Vec3A {
    let flat = Vec3A::new(v.x, v.y, 0.0);
    let len = flat.length();
    if len > 1e-4 { flat / len } else { Vec3A::ZERO }
}

pub fn flag(b: bool) -> f32 {
    f32::from(u8::from(b))
}

/// Seconds as whole arena ticks (as the headers' `lroundf(sec / TICK_TIME)`).
pub fn secs_to_ticks(seconds: f32) -> f32 {
    (seconds / TICK_SECONDS).round()
}

/// The ticks this step lasted.
pub fn step_ticks(events: &StepEvents) -> f32 {
    secs_to_ticks(events.step_seconds)
}

/// The goal mouth's centre that `team` attacks.
pub fn target_goal(team: Team) -> Vec3A {
    attack_goal(team)
}

/// The back of the net that `team` attacks.
pub fn goal_aim_point(team: Team) -> Vec3A {
    attack_net_back(team)
}

/// Within `m` of the floor, ceiling, side walls or back walls (flat planes).
pub fn near_any_surface(p: Vec3A, m: f32) -> bool {
    p.z < m || p.z > CEILING_Z - m || p.x.abs() > SIDE_WALL_X - m || p.y.abs() > BACK_WALL_Y - m
}

/// The ball resting on the roof, as the headers judge it.
#[derive(Clone, Copy)]
pub struct Dribble {
    pub min_up: f32,
    pub max_up: f32,
    pub target_forward: f32,
    pub max_planar_dist: f32,
    pub max_rel_speed: f32,
}

/// How well `car` controls the ball on its roof: 0 to 1 (0 off the ground or demolished).
pub fn control_quality(car: &CarSnapshot, ball: &Body, d: Dribble) -> f32 {
    if car.is_demoed || !car.on_ground {
        return 0.0;
    }
    let local = car.body.rot.transpose() * (ball.pos - car.body.pos);
    if local.z < d.min_up || local.z > d.max_up {
        return 0.0;
    }
    let (dx, dy) = (local.x - d.target_forward, local.y);
    let planar = div((dx * dx + dy * dy).sqrt(), d.max_planar_dist);
    let rel_speed = div((ball.vel - car.body.vel).length(), d.max_rel_speed);
    let (pos_term, vel_term) = (1.0 - planar * planar, 1.0 - rel_speed * rel_speed);
    if pos_term <= 0.0 || vel_term <= 0.0 {
        0.0
    } else {
        clamp01(pos_term * vel_term)
    }
}

/// A car's hitbox: its size and its centre's offset from the centre of mass (car frame).
#[derive(Clone, Copy)]
pub struct Hitbox {
    pub size: Vec3A,
    pub offset: Vec3A,
}

/// Do two boxes (centre, axes as columns, half sizes) overlap? The separating-axis test over 15 axes.
pub fn boxes_overlap(ca: Vec3A, ra: Mat3A, ha: Vec3A, cb: Vec3A, rb: Mat3A, hb: Vec3A) -> bool {
    let a_axes = [ra.x_axis, ra.y_axis, ra.z_axis];
    let b_axes = [rb.x_axis, rb.y_axis, rb.z_axis];
    let (a, b) = (ha.to_array(), hb.to_array());
    let mut r = [[0.0f32; 3]; 3];
    let mut abs_r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a_axes[i].dot(b_axes[j]);
            abs_r[i][j] = r[i][j].abs() + 1e-6;
        }
    }
    let d = cb - ca;
    let t = [d.dot(a_axes[0]), d.dot(a_axes[1]), d.dot(a_axes[2])];
    for i in 0..3 {
        let reach = b[0] * abs_r[i][0] + b[1] * abs_r[i][1] + b[2] * abs_r[i][2];
        if t[i].abs() > a[i] + reach {
            return false;
        }
    }
    for j in 0..3 {
        let along = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        let reach = a[0] * abs_r[0][j] + a[1] * abs_r[1][j] + a[2] * abs_r[2][j];
        if along.abs() > reach + b[j] {
            return false;
        }
    }
    for i in 0..3 {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        for j in 0..3 {
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let reach_a = a[i1] * abs_r[i2][j] + a[i2] * abs_r[i1][j];
            let reach_b = b[j1] * abs_r[i][j2] + b[j2] * abs_r[i][j1];
            if (t[i2] * r[i1][j] - t[i1] * r[i2][j]).abs() > reach_a + reach_b {
                return false;
            }
        }
    }
    true
}

/// Another car (not demolished) within `margin` of this car's hitbox (both taken as `hitbox`), or a bump
/// between them whose cooldown still runs.
pub fn car_contact(
    cur: &GameSnapshot,
    events: &StepEvents,
    car: usize,
    hitbox: Hitbox,
    margin: f32,
) -> bool {
    let half = hitbox.size * 0.5 + Vec3A::splat(margin * 0.5);
    let centre = |q: usize| cur.cars[q].body.pos + cur.cars[q].body.rot * hitbox.offset;
    let mine = events.detail(car);
    (0..cur.cars.len())
        .filter(|&q| q != car && !cur.cars[q].is_demoed)
        .any(|q| {
            let theirs = events.detail(q);
            let bumped = (theirs.last_contact_car == Some(car) && theirs.bump_cooldown > 0.0)
                || (mine.last_contact_car == Some(q) && mine.bump_cooldown > 0.0);
            bumped
                || boxes_overlap(
                    centre(car),
                    cur.cars[car].body.rot,
                    half,
                    centre(q),
                    cur.cars[q].body.rot,
                    half,
                )
        })
}

/// `v` turned by the rotation vector `w` (axis times angle, radians).
pub fn rotate(v: Vec3A, w: Vec3A) -> Vec3A {
    let angle = w.length();
    if angle <= 1e-6 {
        return v;
    }
    let k = w / angle;
    let (s, c) = angle.sin_cos();
    v * c + k.cross(v) * s + k * (k.dot(v) * (1.0 - c))
}

/// The orientation `rot` turned by the world rotation vector `w`.
pub fn turned(rot: Mat3A, w: Vec3A) -> Mat3A {
    Mat3A::from_cols(
        rotate(rot.x_axis, w),
        rotate(rot.y_axis, w),
        rotate(rot.z_axis, w),
    )
}

/// Ticks from the start of the car's latest ball-hit tick to now (at least 1). Nitro's hit tick is the
/// count after that tick, RocketSim's `tickCountWhenHit` the count during it.
pub fn hit_ticks_back(detail: &CarDetail, now: u64) -> f32 {
    let hit = detail.last_hit_tick.unwrap_or(now);
    (now.saturating_sub(hit) + 1) as f32
}

/// The ball's velocity just after this car's latest touch: now, with the gravity since then undone.
pub fn launch_velocity(detail: &CarDetail, ball: &Body, now: u64) -> Vec3A {
    let since = now.saturating_sub(detail.last_hit_tick.unwrap_or(now)) as f32;
    ball.vel - Vec3A::new(0.0, 0.0, GRAVITY_Z * since * TICK_SECONDS)
}

/// Where the ball's centre was when the latest hit tick began (RocketSim's `BallHitInfo::ballPos`):
/// Nitro keeps it at the tick's end, so it is wound back one tick at the velocity it left with.
pub fn ball_at_hit(detail: &CarDetail, ball: &Body, now: u64) -> Vec3A {
    if detail.last_hit_tick.is_none() {
        return ball.pos;
    }
    detail.last_hit_ball_pos - launch_velocity(detail, ball, now) * TICK_SECONDS
}

/// The car's pose `ticks_back` ticks ago, wound back at its current velocity and spin.
pub fn pose_back(body: &Body, ticks_back: f32) -> (Vec3A, Mat3A) {
    let back = ticks_back * TICK_SECONDS;
    (
        body.pos - body.vel * back,
        turned(body.rot, body.ang_vel * -back),
    )
}

/// The car-local (forward, right, up) contact point of the car's latest ball hit: the hitbox point closest
/// to the ball's centre then, with the car's pose wound back to the hit; and the nose's pitch then
/// (degrees, + up).
pub fn hit_contact_local(
    body: &Body,
    detail: &CarDetail,
    ball: &Body,
    now: u64,
    hitbox: Hitbox,
) -> (Vec3A, f32) {
    let (pos, rot) = pose_back(body, hit_ticks_back(detail, now));
    let pitch = degrees(rot.x_axis.z.clamp(-1.0, 1.0).asin());
    let d = ball_at_hit(detail, ball, now) - pos;
    let local = rot.transpose() * d - hitbox.offset;
    let half = hitbox.size * 0.5;
    (local.max(-half).min(half) + hitbox.offset, pitch)
}

/// Where a free ball's flight meets the goal line `team` attacks.
#[derive(Clone, Copy, Default)]
pub struct Shot {
    pub reaches: bool,
    pub on_target: bool,
    /// The ball centre's x there, as seen by `team` (Blue's view).
    pub x: f32,
    pub z: f32,
}

/// The ball's flight from `pos` at `vel`, tick by tick for `max_time` seconds: gravity, drag, bounces off
/// the floor, ceiling and side walls (no spin), until it reaches the goal line `team` attacks.
pub fn project_to_goal_line(mut pos: Vec3A, mut vel: Vec3A, team: Team, max_time: f32) -> Shot {
    const R: f32 = 91.25;
    const REST: f32 = 0.6;
    let dt = TICK_SECONDS;
    let drag = (1.0f32 - 0.03).powf(dt);
    let sign = attack_sign(team);
    let steps = (max_time / dt) as i32;
    for _ in 0..steps {
        vel.z += GRAVITY_Z * dt;
        vel *= drag;
        pos += vel * dt;
        if pos.z < R && vel.z < 0.0 {
            pos.z = R;
            vel.z = -vel.z * REST;
        }
        if pos.z > CEILING_Z - R && vel.z > 0.0 {
            pos.z = CEILING_Z - R;
            vel.z = -vel.z * REST;
        }
        if pos.x.abs() > SIDE_WALL_X - R && pos.x * vel.x > 0.0 {
            pos.x = (SIDE_WALL_X - R) * pos.x.signum();
            vel.x = -vel.x * REST;
        }
        if pos.y * sign >= BACK_WALL_Y {
            return Shot {
                reaches: true,
                on_target: pos.x.abs() <= GOAL_HALF_WIDTH - R && pos.z <= GOAL_HEIGHT - R,
                x: pos.x * sign,
                z: pos.z,
            };
        }
        if pos.y * sign < -BACK_WALL_Y {
            break;
        }
    }
    Shot::default()
}

/// A free ball's flight toward the goal line `team` attacks with gravity only: no drag, no bounces (the
/// ceiling and side walls ignored), as if the floor were not there.
#[derive(Clone, Copy, Default)]
pub struct AirShot {
    /// Heading for that goal line at all.
    pub toward: bool,
    /// The ball centre's x (as seen by `team`) and height where the flight crosses the goal line.
    pub x: f32,
    pub z: f32,
    /// How far short of the goal line (uu) the ball's flight comes down to the floor (its centre a ball
    /// radius up); 0 when it gets there first.
    pub short: f32,
}

pub fn air_path_to_goal_line(pos: Vec3A, vel: Vec3A, team: Team) -> AirShot {
    let sign = attack_sign(team);
    let (ahead, vy) = (BACK_WALL_Y - pos.y * sign, vel.y * sign);
    if vy <= 0.0 {
        return AirShot::default();
    }
    let g = -GRAVITY_Z;
    let t_line = ahead.max(0.0) / vy;
    // The later root of z(t) = BALL_RADIUS
    let h = (pos.z - BALL_RADIUS).max(0.0);
    let t_floor = (vel.z + (vel.z * vel.z + 2.0 * g * h).sqrt()) / g;
    AirShot {
        toward: true,
        x: (pos.x + vel.x * t_line) * sign,
        z: pos.z + vel.z * t_line - 0.5 * g * t_line * t_line,
        short: (ahead - vy * t_floor).max(0.0),
    }
}

/// The nose's angle below horizontal, measured toward the horizontal direction of `d`: 0 level pointing at
/// it, 90 straight down, 180 level pointing away (degrees; negative = nose up).
pub fn nose_down_deg(rot: Mat3A, d: Vec3A) -> f32 {
    degrees((-rot.x_axis.z).atan2(rot.x_axis.dot(dir2d(d))))
}

/// The musty's dodge pose (MustyFlickReward's ARM test, shared by its pose shaping).
#[derive(Clone, Copy)]
pub struct ArmTest {
    pub max_torque_y: f32,
    pub min_nose_down_deg: f32,
    pub max_nose_down_deg: f32,
    pub max_ball_dist: f32,
    pub min_ball_above: f32,
    pub min_up_dot_ball: f32,
    pub max_lateral: f32,
}

/// At a dodge start: a backflip or back-diagonal in the air, no touch this step, the nose far enough down
/// toward the ball, the ball close, not below the car and on the roof side.
pub fn arm_pose_ok(
    test: ArmTest,
    car: &CarSnapshot,
    detail: &CarDetail,
    touched: bool,
    ball: Vec3A,
) -> bool {
    if car.on_ground || touched || detail.flip_rel_torque.y > test.max_torque_y {
        return false;
    }
    let to_ball = ball - car.body.pos;
    let dist = to_ball.length();
    let nose_down = nose_down_deg(car.body.rot, to_ball);
    let (up_dot_ball, lateral) = if dist > 1.0 {
        let dir = to_ball / dist;
        (
            car.body.rot.z_axis.dot(dir),
            car.body.rot.y_axis.dot(dir).abs(),
        )
    } else {
        (0.0, 0.0)
    };
    nose_down >= test.min_nose_down_deg
        && nose_down <= test.max_nose_down_deg
        && dist <= test.max_ball_dist
        && to_ball.z >= test.min_ball_above
        && up_dot_ball >= test.min_up_dot_ball
        && lateral <= test.max_lateral
}

/// A dodge is available: in the air, no flip or double jump used, and the flip window still open.
pub fn has_dodge(car: &CarSnapshot, detail: &CarDetail) -> bool {
    !car.on_ground
        && !car.has_flipped
        && !car.has_double_jumped
        && (!car.has_jumped || detail.air_time_since_jump < 1.25)
}

/// On something: 3+ wheels down (an arena surface, or the ball), or a wheel or the body touching
/// something within `margin` of an arena surface.
pub fn supported(car: &CarSnapshot, detail: &CarDetail, margin: f32) -> bool {
    if car.on_ground {
        return true;
    }
    let touching =
        detail.wheel_contacts.iter().any(|&w| w) || detail.world_contact_normal.is_some();
    touching && near_any_surface(car.body.pos, margin)
}

/// The rotation from orientation `a` to `b` as a world axis times its angle (radians); exact below 180°.
pub fn rotation_between(a: Mat3A, b: Mat3A) -> Vec3A {
    let (a_axes, b_axes) = (
        [a.x_axis, a.y_axis, a.z_axis],
        [b.x_axis, b.y_axis, b.z_axis],
    );
    let mut q = [[0.0f32; 3]; 3];
    for (i, row) in q.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| b_axes[k][i] * a_axes[k][j]).sum();
        }
    }
    let axis = Vec3A::new(q[2][1] - q[1][2], q[0][2] - q[2][0], q[1][0] - q[0][1]);
    let s = axis.length();
    if s < 1e-7 {
        return Vec3A::ZERO;
    }
    let angle = (0.5 * s).atan2(0.5 * (q[0][0] + q[1][1] + q[2][2] - 1.0));
    axis * (angle / s)
}
