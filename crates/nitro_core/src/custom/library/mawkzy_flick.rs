//! The Mawkzy flick: a jump off a hood dribble, the turn into the Mawkzy's pose, then a strong,
//! goal-directed reverse or diagonal dodge flick, judged on the ball's launch.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::aerial_kit::*;

/// The Mawkzy flick: 0.1 for jumping off a controlled dribble, up to 0.3 for turning into the Mawkzy's pose
/// after the pop (kept only by a Mawkzy dodge), up to 0.15 for the dodge's shape, then once per possession the
/// flick's launch, from -0.5 (backward) to 1.5 (a full-power Mawkzy). No dribble pay unless dribble_rate is set.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MawkzyFlick {
    /// Dribble: lowest ball centre height above the car origin, along the car's up (resting on the roof ~131-136).
    #[serde(default = "dribble_min_up")]
    pub dribble_min_up: f32,
    /// Dribble: highest ball centre height above the car origin.
    #[serde(default = "dribble_max_up")]
    pub dribble_max_up: f32,
    /// Where a stable carry sits: the ball centre this far forward of the car origin.
    #[serde(default = "dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// The farthest the ball may be from that spot in the car's forward/right plane.
    #[serde(default = "dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// Ball-car relative speed at which control quality reaches 0.
    #[serde(default = "dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality above which the car is in control.
    #[serde(default = "control_threshold")]
    pub control_threshold: f32,
    /// Paid per second of perfect control (before the decay); 0 = no dribble pay (0.1 pays the dribble too).
    #[serde(default = "dribble_rate")]
    pub dribble_rate: f32,
    /// More per second while in control at full speed toward the target goal (0.1 pays the set-up too).
    #[serde(default = "setup_rate")]
    pub setup_rate: f32,
    /// Car speed toward goal (uu/s) for the full setup credit.
    #[serde(default = "setup_full_speed")]
    pub setup_full_speed: f32,
    /// Once per possession: leaving the ground by jumping while in control.
    #[serde(default = "jump_setup_bonus")]
    pub jump_setup_bonus: f32,
    /// Dense pay decays as exp(-control time / this) within one possession.
    #[serde(default = "possession_decay_time")]
    pub possession_decay_time: f32,
    /// The most dense (dribble + setup) pay per episode.
    #[serde(default = "max_dense_per_episode")]
    pub max_dense_per_episode: f32,
    /// Seconds of control before a dodge counts as a flick attempt.
    #[serde(default = "min_possession_time")]
    pub min_possession_time: f32,
    /// A dodge must start within this many seconds of the last controlled step.
    #[serde(default = "possession_memory")]
    pub possession_memory: f32,
    /// The ball-car distance (uu) when the dodge starts.
    #[serde(default = "arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Seconds after the dodge starts in which the launch is judged.
    #[serde(default = "window_time")]
    pub window_time: f32,
    /// Judged early once the ball is this far from the car after contact.
    #[serde(default = "separation_dist")]
    pub separation_dist: f32,
    /// Goal-directed ball speed gain (uu/s) needed for any credit.
    #[serde(default = "min_gain")]
    pub min_gain: f32,
    /// Gain for the full credit.
    #[serde(default = "full_gain")]
    pub full_gain: f32,
    /// Final goal-directed ball speed for the full credit (~115 km/h).
    #[serde(default = "full_launch_speed")]
    pub full_launch_speed: f32,
    /// Horizontal cosine to the goal at which aim credit is 0...
    #[serde(default = "aim_cos_zero")]
    pub aim_cos_zero: f32,
    /// ...and 1.
    #[serde(default = "aim_cos_full")]
    pub aim_cos_full: f32,
    /// The flick's weight within this reward.
    #[serde(default = "flick_weight")]
    pub flick_weight: f32,
    /// Strong-flick bonus on top of power, x this at strong_gain_full (0 = no bonus).
    #[serde(default = "power_bonus")]
    pub power_bonus: f32,
    /// The strong-flick bonus starts at this gain...
    #[serde(default = "strong_gain_start")]
    pub strong_gain_start: f32,
    /// ...and is full at this one.
    #[serde(default = "strong_gain_full")]
    pub strong_gain_full: f32,
    /// The most penalty for launching the ball away from the target goal.
    #[serde(default = "backward_penalty")]
    pub backward_penalty: f32,
    /// Horizontal ball speed change (uu/s) before a backward launch is penalised.
    #[serde(default = "min_penalty_impulse")]
    pub min_penalty_impulse: f32,
    /// Launches judged this close to a wall or the ceiling are ignored.
    #[serde(default = "wall_margin")]
    pub wall_margin: f32,
    /// 1: only exact 45-degree diagonal dodges get full credit (0 = pure outcome).
    #[serde(default = "style_weight")]
    pub style_weight: f32,
    /// 1: only backward (diagonal) dodges get full credit (0 = pure outcome).
    #[serde(default = "reverse_weight")]
    pub reverse_weight: f32,
    /// Require the Mawkzy's shape (false = the outcome-only reward).
    #[serde(default = "require_shape")]
    pub require_shape: bool,
    /// The nose's angle from the goal direction at the dodge press: full credit from this (deg)...
    #[serde(default = "heading_min")]
    pub heading_min: f32,
    /// ...to this...
    #[serde(default = "heading_max")]
    pub heading_max: f32,
    /// ...fading to 0 over this much either side.
    #[serde(default = "heading_edge")]
    pub heading_edge: f32,
    /// The nose's angle below horizontal: full credit from this (deg)...
    #[serde(default = "nose_down_min")]
    pub nose_down_min: f32,
    /// ...to this...
    #[serde(default = "nose_down_max")]
    pub nose_down_max: f32,
    /// ...fading to 0 over this much either side.
    #[serde(default = "nose_down_edge")]
    pub nose_down_edge: f32,
    /// The roof's tilt from vertical (the air roll): no credit below this (deg)...
    #[serde(default = "tilt_zero")]
    pub tilt_zero: f32,
    /// ...full credit from this.
    #[serde(default = "tilt_full")]
    pub tilt_full: f32,
    /// The roof toward the goal (car up . horizontal goal direction): no credit at this...
    #[serde(default = "roof_to_goal_zero")]
    pub roof_to_goal_zero: f32,
    /// ...full credit at this.
    #[serde(default = "roof_to_goal_full")]
    pub roof_to_goal_full: f32,
    /// The roof toward the ball (car up . direction to the ball): no credit at this...
    #[serde(default = "roof_to_ball_zero")]
    pub roof_to_ball_zero: f32,
    /// ...full credit at this.
    #[serde(default = "roof_to_ball_full")]
    pub roof_to_ball_full: f32,
    /// Credit for a straight backflip instead of the diagonal.
    #[serde(default = "backflip_credit")]
    pub backflip_credit: f32,
    /// Credit for a diagonal toward the turn side (not a Mawkzy).
    #[serde(default = "wrong_side_credit")]
    pub wrong_side_credit: f32,
    /// Paid once at the dodge press, times the shape's quality.
    #[serde(default = "setup_shape_bonus")]
    pub setup_shape_bonus: f32,
    /// Pose shaping: from the pop to the dodge, pays for getting closer to the Mawkzy pose and takes it back for
    /// moving away; kept only when the dodge arms a Mawkzy attempt, at most this per attempt (0 = off).
    #[serde(default = "pose_shaping_weight")]
    pub pose_shaping_weight: f32,
    /// Pose shaping's discount per step: set it to the training's (`ppo` discount per step).
    #[serde(default = "pose_gamma")]
    pub pose_gamma: f32,
    /// Pose shaping: no credit with the ball farther than this from the car (full at arm_max_ball_dist).
    #[serde(default = "pose_max_dist")]
    pub pose_max_dist: f32,
    /// Scale by the ball's launch angle.
    #[serde(default = "use_elevation")]
    pub use_elevation: bool,
    /// Launch angle with full credit from this (deg)...
    #[serde(default = "elev_full_lo")]
    pub elev_full_lo: f32,
    /// ...to this.
    #[serde(default = "elev_full_hi")]
    pub elev_full_hi: f32,
    /// No credit below this launch angle (deg)...
    #[serde(default = "elev_zero_lo")]
    pub elev_zero_lo: f32,
    /// ...or above this one.
    #[serde(default = "elev_zero_hi")]
    pub elev_zero_hi: f32,
    /// The ball must be hit by the front of the car during the flick.
    #[serde(default = "require_front_contact")]
    pub require_front_contact: bool,
    /// The contact point's forward coordinate from the car's centre (uu) that counts as the front.
    #[serde(default = "front_contact_min")]
    pub front_contact_min: f32,
    /// No credit when the car crashes into the floor before the contact (the Raze flick, floor crashes).
    #[serde(default = "reject_floor_crash")]
    pub reject_floor_crash: bool,
    /// The lowest hitbox corner below this with the car's up pointing down: a floor crash.
    #[serde(default = "floor_crash_height")]
    pub floor_crash_height: f32,
    /// Training only: a car starting with the ball close above its roof is already in possession (true
    /// suits starts part-way through the move).
    #[serde(default = "reset_grants_possession")]
    pub reset_grants_possession: bool,
    /// ...within this distance of the ball.
    #[serde(default = "reset_possession_dist")]
    pub reset_possession_dist: f32,
}

defaults! { MawkzyFlick {
    dribble_min_up: f32 = 105.0,
    dribble_max_up: f32 = 185.0,
    dribble_target_forward: f32 = 20.0,
    dribble_max_planar_dist: f32 = 90.0,
    dribble_max_rel_speed: f32 = 600.0,
    control_threshold: f32 = 0.25,
    dribble_rate: f32 = 0.0,
    setup_rate: f32 = 0.0,
    setup_full_speed: f32 = 1400.0,
    jump_setup_bonus: f32 = 0.10,
    possession_decay_time: f32 = 2.0,
    max_dense_per_episode: f32 = 1.0,
    min_possession_time: f32 = 0.25,
    possession_memory: f32 = 1.0,
    arm_max_ball_dist: f32 = 280.0,
    window_time: f32 = 0.5,
    separation_dist: f32 = 300.0,
    min_gain: f32 = 250.0,
    full_gain: f32 = 1800.0,
    full_launch_speed: f32 = 3200.0,
    aim_cos_zero: f32 = 0.0,
    aim_cos_full: f32 = 0.85,
    flick_weight: f32 = 1.0,
    power_bonus: f32 = 0.5,
    strong_gain_start: f32 = 800.0,
    strong_gain_full: f32 = 1600.0,
    backward_penalty: f32 = 0.5,
    min_penalty_impulse: f32 = 500.0,
    wall_margin: f32 = 200.0,
    style_weight: f32 = 0.0,
    reverse_weight: f32 = 0.0,
    require_shape: bool = true,
    heading_min: f32 = 25.0,
    heading_max: f32 = 95.0,
    heading_edge: f32 = 10.0,
    nose_down_min: f32 = -10.0,
    nose_down_max: f32 = 30.0,
    nose_down_edge: f32 = 10.0,
    tilt_zero: f32 = 25.0,
    tilt_full: f32 = 40.0,
    roof_to_goal_zero: f32 = 0.0,
    roof_to_goal_full: f32 = 0.3,
    roof_to_ball_zero: f32 = 0.3,
    roof_to_ball_full: f32 = 0.6,
    backflip_credit: f32 = 0.6,
    wrong_side_credit: f32 = 0.0,
    setup_shape_bonus: f32 = 0.15,
    pose_shaping_weight: f32 = 0.3,
    pose_gamma: f32 = 0.99,
    pose_max_dist: f32 = 450.0,
    use_elevation: bool = true,
    elev_full_lo: f32 = 5.0,
    elev_full_hi: f32 = 10.0,
    elev_zero_lo: f32 = -5.0,
    elev_zero_hi: f32 = 25.0,
    require_front_contact: bool = true,
    front_contact_min: f32 = 20.0,
    reject_floor_crash: bool = true,
    floor_crash_height: f32 = 15.0,
    reset_grants_possession: bool = false,
    reset_possession_dist: f32 = 280.0,
} }

kept! {
    /// Each car's own state between steps (ticks on the episode clock).
    struct Track {
        /// Control quality this step.
        quality: f32,
        in_control: bool,
        /// Seconds of control in the current possession.
        control_time: f32,
        last_control_tick: f32,
        possession: bool,
        jump_bonus_paid: bool,
        /// One flick attempt per possession.
        attempt_used: bool,
        /// Dense pay this episode.
        dense_paid: f32,
        window_active: bool,
        window_start_tick: f32,
        ball_vel0: Vec3A,
        touched: bool,
        dodge_rel_torque: Vec3A,
        /// The Mawkzy shape's quality at the dodge press.
        shape: f32,
        floor_crash: bool,
        have_contact: bool,
        contact_forward: f32,
        /// The tick the dodge started (from the flip time).
        dodge_tick: f32,
        /// The pose shaping's potential at the last step.
        pose_phi: f32,
        /// Pose shaping kept by the current attempt (taken back for a Raze or a floor crash).
        pose_banked: f32,
    }
}

impl Track {
    fn end_possession(&mut self) {
        self.possession = false;
        self.control_time = 0.0;
        self.jump_bonus_paid = false;
        self.attempt_used = false;
    }
}

/// The car's attitude toward the target goal and the ball (at the dodge press).
struct Attitude {
    heading_deg: f32,
    nose_down_deg: f32,
    tilt_deg: f32,
    roof_to_goal: f32,
    roof_to_ball: f32,
}

/// How a flick window ends.
#[derive(PartialEq)]
enum Ending {
    Launch,
    Cancelled,
}

/// 1 inside lo..hi, falling linearly to 0 at lo - edge and hi + edge.
fn band(v: f32, lo: f32, hi: f32, edge: f32) -> f32 {
    clamp01((v - lo + edge).min(hi + edge - v) / edge)
}

/// The Octane hitbox's point closest to a point in the car's frame.
fn closest_on_hitbox(l: Vec3A) -> Vec3A {
    let half = HITBOX_SIZE * 0.5;
    l.clamp(HITBOX_OFFSET - half, HITBOX_OFFSET + half)
}

/// The height of the hitbox's lowest corner.
fn lowest_corner_z(body: &Body) -> f32 {
    let mut lowest = f32::MAX;
    for fx in [-1.0, 1.0] {
        for ry in [-1.0, 1.0] {
            for uz in [-1.0, 1.0] {
                let l = HITBOX_OFFSET + Vec3A::new(fx, ry, uz) * HITBOX_SIZE * 0.5;
                lowest = lowest.min((body.pos + body.rot * l).z);
            }
        }
    }
    lowest
}

/// The car-frame point of its latest ball contact: the car's pose wound back to the hit with its velocity and
/// angular velocity, and the hitbox point closest to the ball's centre then.
fn estimate_contact_local(car: &Body, hit: &Hit) -> Vec3A {
    let dt = hit.rewind_ticks * TICK_SECONDS;
    let pos = car.pos - car.vel * dt;
    let back = |axis: Vec3A| (axis - car.ang_vel.cross(axis) * dt).normalize_or_zero();
    let (f, r, u) = (
        back(car.rot.x_axis),
        back(car.rot.y_axis),
        back(car.rot.z_axis),
    );
    let d = hit.ball_pos - pos;
    closest_on_hitbox(Vec3A::new(d.dot(f), d.dot(r), d.dot(u)))
}

impl MawkzyFlick {
    /// Control quality, 0-1 (left/right symmetric).
    fn control_quality(&self, car: &CarSnapshot, ball: &Body) -> f32 {
        if car.is_demoed || !car.on_ground {
            return 0.0;
        }
        let l = local(&car.body, ball.pos - car.body.pos);
        if l.z < self.dribble_min_up || l.z > self.dribble_max_up {
            return 0.0;
        }
        let (dx, dy) = (l.x - self.dribble_target_forward, l.y);
        let planar = (dx * dx + dy * dy).sqrt() / self.dribble_max_planar_dist;
        let rel_speed = (ball.vel - car.body.vel).length() / self.dribble_max_rel_speed;
        let pos_term = 1.0 - planar * planar;
        let vel_term = 1.0 - rel_speed * rel_speed;
        if pos_term <= 0.0 || vel_term <= 0.0 {
            return 0.0;
        }
        (pos_term * vel_term).clamp(0.0, 1.0)
    }

    /// heading > 0: the nose turned to the car's right of the goal direction.
    fn attitude(car: &Body, team: Team, ball_pos: Vec3A) -> Attitude {
        let goal = dir2d(attack_goal(team) - car.pos);
        let goal_right = Vec3A::new(-goal.y, goal.x, 0.0);
        let fh = dir2d(car.forward());
        Attitude {
            heading_deg: deg(fh.dot(goal_right).atan2(fh.dot(goal))),
            nose_down_deg: deg((-car.forward().z).clamp(-1.0, 1.0).asin()),
            tilt_deg: deg(car.up().z.clamp(-1.0, 1.0).acos()),
            roof_to_goal: car.up().dot(goal),
            roof_to_ball: car.up().dot((ball_pos - car.pos).normalize_or_zero()),
        }
    }

    /// How well the attitude at the dodge press and the dodge's direction match the Mawkzy, 0-1. Mirror
    /// symmetric: turning right needs the stick down-left (torque x > 0), turning left down-right.
    fn shape_quality(&self, a: &Attitude, torque: Vec3A) -> f32 {
        if torque.y > -0.3 {
            return 0.0; // Not a backward dodge
        }
        let side = if torque.x.abs() < 0.3 {
            self.backflip_credit
        } else if (torque.x > 0.0) == (a.heading_deg > 0.0) {
            1.0
        } else {
            self.wrong_side_credit
        };
        let heading = band(
            a.heading_deg.abs(),
            self.heading_min,
            self.heading_max,
            self.heading_edge,
        );
        let nose = band(
            a.nose_down_deg,
            self.nose_down_min,
            self.nose_down_max,
            self.nose_down_edge,
        );
        let tilt = ramp(a.tilt_deg, self.tilt_zero, self.tilt_full);
        let to_goal = ramp(
            a.roof_to_goal,
            self.roof_to_goal_zero,
            self.roof_to_goal_full,
        );
        let to_ball = ramp(
            a.roof_to_ball,
            self.roof_to_ball_zero,
            self.roof_to_ball_full,
        );
        side * heading * nose * tilt * to_goal * to_ball
    }

    /// The pose shaping's potential, 0-1: how far the car has turned into the Mawkzy pose (the mean of the nose
    /// turned off the goal line, the roof rolled over and the roof facing the goal, each 0 for a level car after
    /// the pop), gated softly by the nose's pitch and the roof facing the ball, times how close the ball is. 0
    /// without a recent possession, on the ground, or with no dodge left.
    fn pose_potential(
        &self,
        car: &CarSnapshot,
        detail: &CarDetail,
        ball: &Body,
        recent_control: bool,
    ) -> f32 {
        if detail.air_time_since_jump >= 1.25 {
            return 0.0; // The dodge is gone
        }
        if !recent_control
            || car.on_ground
            || !car.has_jumped
            || car.has_flipped
            || car.has_double_jumped
            || car.is_demoed
        {
            return 0.0;
        }
        let d = car.body.pos.distance(ball.pos);
        if d >= self.pose_max_dist {
            return 0.0;
        }
        let a = Self::attitude(&car.body, car.team, ball.pos);
        let heading = band(
            a.heading_deg.abs(),
            self.heading_min,
            self.heading_max,
            self.heading_min,
        );
        let nose = band(
            a.nose_down_deg,
            self.nose_down_min,
            self.nose_down_max,
            30.0,
        );
        let tilt = clamp01(a.tilt_deg / self.tilt_full);
        let to_goal = clamp01(a.roof_to_goal / self.roof_to_goal_full);
        let to_ball = clamp01((a.roof_to_ball + 0.2) / (self.roof_to_ball_full + 0.2));
        let close =
            clamp01((self.pose_max_dist - d) / (self.pose_max_dist - self.arm_max_ball_dist));
        close * nose * to_ball * (heading + tilt + to_goal) / 3.0
    }

    fn elevation_term(&self, e: f32) -> f32 {
        if e >= self.elev_full_lo && e <= self.elev_full_hi {
            1.0
        } else if e < self.elev_full_lo {
            ramp(e, self.elev_zero_lo, self.elev_full_lo)
        } else {
            clamp01((self.elev_zero_hi - e) / (self.elev_zero_hi - self.elev_full_hi))
        }
    }

    fn near_wall(&self, p: Vec3A) -> bool {
        let m = self.wall_margin + RLGYM_BALL_RADIUS;
        p.x.abs() > SIDE_WALL_X - m
            || p.z > CEILING_Z - m
            // The back walls, except inside the goal mouth
            || (p.y.abs() > BACK_WALL_Y - m && (p.x.abs() > GOAL_HALF_WIDTH || p.z > GOAL_HEIGHT))
    }

    /// Ends the flick window; a launch is judged on the ball's velocity change since the dodge.
    fn resolve(&self, tr: &mut Track, team: Team, ball: &Body, ending: Ending) -> f32 {
        tr.window_active = false;
        if ending == Ending::Cancelled {
            return 0.0;
        }
        let goal_dir = dir2d(attack_goal(team) - ball.pos);
        let v1 = ball.vel;
        let mut dv = v1 - tr.ball_vel0;
        dv.z = 0.0; // Horizontal only: gravity and floor bounces act on z
        let gain = dv.dot(goal_dir);
        let final_goal_speed = Vec3A::new(v1.x, v1.y, 0.0).dot(goal_dir);
        let horiz_impulse = dv.length();
        let aim_cos = dir2d(v1).dot(goal_dir);
        let gain_term = ramp(gain, self.min_gain, self.full_gain);
        let final_term = clamp01(final_goal_speed / self.full_launch_speed);
        let power = (gain_term * final_term).sqrt();
        let strong = if power > 0.0 {
            ramp(gain, self.strong_gain_start, self.strong_gain_full)
        } else {
            0.0
        };
        let aim = ramp(aim_cos, self.aim_cos_zero, self.aim_cos_full);
        let t = tr.dodge_rel_torque;
        let t_len_sq = t.x * t.x + t.y * t.y;
        let diag = if t_len_sq > 1e-6 {
            2.0 * (t.x * t.y).abs() / t_len_sq
        } else {
            0.0
        }; // 1 at 45 degrees
        let reverse = if t_len_sq > 1e-6 && t.y < 0.0 {
            1.0
        } else {
            0.0
        }; // A backward dodge
        let style = (1.0 - self.style_weight + self.style_weight * diag)
            * (1.0 - self.reverse_weight + self.reverse_weight * reverse);
        let mut reward = self.flick_weight * (power + self.power_bonus * strong) * aim * style;
        // The Mawkzy's shape: the attitude at the dodge press, a front-of-car contact, no floor crash
        if self.require_shape {
            let front_contact = tr.have_contact && tr.contact_forward >= self.front_contact_min;
            let mut shape = tr.shape;
            if self.require_front_contact && !front_contact {
                shape = 0.0;
            }
            if self.reject_floor_crash && tr.floor_crash {
                shape = 0.0;
            }
            reward *= shape;
        }
        let elevation = deg(v1.z.atan2(hspeed(v1)));
        if self.use_elevation {
            reward *= self.elevation_term(elevation);
        }
        if final_goal_speed < 0.0 && horiz_impulse >= self.min_penalty_impulse {
            reward -= self.backward_penalty * clamp01(-final_goal_speed / self.full_gain);
        }
        // A Raze (contact with the back of the car) or a floor crash looked like a Mawkzy at the dodge press: the
        // pose shaping and the shape bonus it got there are taken back
        if (tr.have_contact && tr.contact_forward < self.front_contact_min) || tr.floor_crash {
            reward -= tr.pose_banked;
            if self.require_shape {
                reward -= self.setup_shape_bonus * tr.shape;
            }
        }
        tr.pose_banked = 0.0;
        reward
    }

    fn update(
        &self,
        tr: &mut Track,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        now: f32,
    ) -> f32 {
        let (p, pp, ball) = (&cur.cars[car], &prev.cars[car], &cur.ball);
        let detail = events.detail(car);
        let dt = events.step_seconds.clamp(0.0, 0.5);
        let step_ticks = (dt / TICK_SECONDS).round();
        let sec_to_ticks = |sec: f32| (sec / TICK_SECONDS).round();
        let mut ticks_since_control = if tr.possession && now >= tr.last_control_tick {
            now - tr.last_control_tick
        } else {
            f32::INFINITY
        };
        let mut reward = 0.0;

        if p.is_demoed {
            if tr.window_active {
                self.resolve(tr, p.team, ball, Ending::Cancelled);
            }
            tr.end_possession();
            tr.in_control = false;
            tr.quality = 0.0;
            let pay_back = -self.pose_shaping_weight * tr.pose_phi;
            tr.pose_phi = 0.0;
            return pay_back;
        }

        // ---- The controlled dribble (paid only with dribble_rate or setup_rate set) ----
        let was_in_control = tr.in_control;
        tr.quality = self.control_quality(p, ball);
        tr.in_control = tr.quality >= self.control_threshold;
        if tr.in_control {
            if !tr.possession {
                tr.possession = true;
                tr.control_time = 0.0;
            }
            let decay = (-tr.control_time / self.possession_decay_time).exp();
            let goal_dir = dir2d(attack_goal(p.team) - p.body.pos);
            let speed_term = clamp01(p.body.vel.dot(goal_dir) / self.setup_full_speed);
            let dense =
                ((self.dribble_rate + self.setup_rate * speed_term) * tr.quality * decay * dt)
                    .min((self.max_dense_per_episode - tr.dense_paid).max(0.0));
            tr.dense_paid += dense;
            reward += dense;
            tr.control_time += dt;
            tr.last_control_tick = now;
            ticks_since_control = 0.0;
        } else if tr.possession
            && !tr.window_active
            && ticks_since_control > sec_to_ticks(self.possession_memory)
        {
            tr.end_possession();
        }
        let recent_control = tr.possession
            && ticks_since_control <= sec_to_ticks(self.possession_memory)
            && tr.control_time >= self.min_possession_time - 1e-4;

        // ---- The jump off the dribble (with the ball under control the step before) ----
        if was_in_control
            && recent_control
            && !tr.jump_bonus_paid
            && pp.on_ground
            && !p.on_ground
            && p.has_jumped
        {
            tr.jump_bonus_paid = true;
            reward += self.jump_setup_bonus;
        }

        // ---- A flick window opens on the dodge's start ----
        let dodge_started = p.has_flipped && !pp.has_flipped;
        let mut mawkzy_armed = false;
        if dodge_started && !tr.window_active && recent_control && !tr.attempt_used {
            let dist_now = p.body.pos.distance(ball.pos);
            let dist_prev = pp.body.pos.distance(prev.ball.pos);
            if dist_now.min(dist_prev) <= self.arm_max_ball_dist {
                tr.window_active = true;
                tr.attempt_used = true;
                tr.window_start_tick = now - step_ticks.min(now); // The dodge began inside this step
                tr.ball_vel0 = prev.ball.vel; // The ball's velocity before the dodge step
                tr.touched = false;
                tr.dodge_rel_torque = detail.flip_rel_torque;
                // The Mawkzy's shape at the dodge press: the state at the end of the step before
                let att = Self::attitude(&pp.body, p.team, prev.ball.pos);
                tr.shape = self.shape_quality(&att, detail.flip_rel_torque);
                tr.floor_crash = false;
                tr.have_contact = false;
                tr.contact_forward = 0.0;
                let flip_ticks = (detail.flip_time.max(0.0) / TICK_SECONDS).round();
                tr.dodge_tick = now - flip_ticks.min(now);
                if self.require_shape {
                    reward += self.setup_shape_bonus * tr.shape;
                }
                mawkzy_armed = tr.shape > 0.0;
            }
        }

        // ---- Pose shaping (potential-based; kept only by a dodge that arms a Mawkzy attempt) ----
        if self.pose_shaping_weight > 0.0 {
            if dodge_started {
                // Kept times the dodge's shape quality (0: all paid back)
                let keep = if mawkzy_armed { tr.shape.min(1.0) } else { 0.0 };
                reward -= self.pose_shaping_weight * tr.pose_phi * (1.0 - keep);
                tr.pose_banked = self.pose_shaping_weight * tr.pose_phi * keep;
                tr.pose_phi = 0.0;
            } else {
                let phi = if cur.goal_scored.is_some() {
                    0.0
                } else {
                    self.pose_potential(p, &detail, ball, recent_control)
                };
                reward += self.pose_shaping_weight * (self.pose_gamma * phi - tr.pose_phi);
                tr.pose_phi = phi;
            }
        }

        // ---- The contact, and the window's end ----
        if tr.window_active {
            let touched = events.cars[car].touched_ball;
            if (0..cur.cars.len()).any(|o| o != car && events.cars[o].touched_ball) {
                self.resolve(tr, p.team, ball, Ending::Cancelled);
            } else {
                if touched {
                    tr.touched = true;
                }
                // Upside down near the floor before the flick contact: a crash or a Raze, not a Mawkzy
                if !tr.have_contact
                    && p.body.up().z < 0.0
                    && lowest_corner_z(&p.body) < self.floor_crash_height
                {
                    tr.floor_crash = true;
                }
                // The first contact after the dodge started: where on the car did it hit?
                if !tr.have_contact
                    && touched
                    && let Some(hit) =
                        last_hit(events, car, now).filter(|hit| hit.tick >= tr.dodge_tick)
                {
                    tr.have_contact = true;
                    tr.contact_forward = estimate_contact_local(&p.body, &hit).x;
                }
                let elapsed = now - tr.window_start_tick;
                let separated = p.body.pos.distance(ball.pos) >= self.separation_dist;
                let time_up = elapsed >= sec_to_ticks(self.window_time);
                let goal = cur.goal_scored.is_some();
                if tr.touched && (separated || time_up || goal) {
                    let ending = if self.near_wall(ball.pos) && !goal {
                        Ending::Cancelled
                    } else {
                        Ending::Launch
                    };
                    reward += self.resolve(tr, p.team, ball, ending);
                } else if !tr.touched && (time_up || goal) {
                    self.resolve(tr, p.team, ball, Ending::Cancelled); // A miss
                }
            }
        }
        reward
    }
}

impl Reward for MawkzyFlick {
    fn memory(&self) -> usize {
        Clock::SIZE + Track::SIZE
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (clock_mem, track_mem) = memory.split_at_mut(Clock::SIZE);
        let mut clock = Clock::load(clock_mem);
        let mut tr = if clock.advance(events) {
            // The episode's start
            let start = &prev.cars[car];
            let mut tr = Track::default();
            if self.reset_grants_possession
                && start.body.pos.distance(prev.ball.pos) <= self.reset_possession_dist
                && local(&start.body, prev.ball.pos - start.body.pos).z > 60.0
            {
                tr.possession = true;
                tr.control_time = self.min_possession_time;
                tr.last_control_tick = 0.0;
            }
            tr
        } else {
            Track::load(track_mem)
        };
        let reward = self.update(&mut tr, prev, cur, events, car, clock.now);
        clock.store(clock_mem);
        tr.store(track_mem);
        finite_or_zero(reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        assert_listed("mawkzy_flick", 61);
    }

    fn reward() -> MawkzyFlick {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    /// A car on the floor at `car` facing +y (Blue's view, mirrored for Orange) with the ball on its roof.
    fn dribbling(team: Team, car: Vec3A, speed: f32) -> GameSnapshot {
        let m = |v: Vec3A| as_blue(team, v);
        let mut g = game(m(car + Vec3A::new(0.0, 20.0, 133.0)), &[(team, m(car))]);
        let yaw = if team == Team::Blue { 0.5 } else { -0.5 };
        g.cars[0].body.rot = Mat3A::from_rotation_z(yaw * std::f32::consts::PI);
        g.cars[0].body.vel = m(Vec3A::new(0.0, speed, 0.0));
        g.ball.vel = g.cars[0].body.vel;
        g
    }

    fn step(
        r: &MawkzyFlick,
        a: &GameSnapshot,
        b: &GameSnapshot,
        ev: &StepEvents,
        memory: &mut [f32],
    ) -> f32 {
        r.value_remembering(a, b, ev, 0, memory)
    }

    /// With dribble pay (dribble_rate and setup_rate 0.1).
    fn dribble_paid() -> MawkzyFlick {
        MawkzyFlick {
            dribble_rate: 0.1,
            setup_rate: 0.1,
            ..reward()
        }
    }

    #[test]
    fn dribbling_pays_nothing_unless_the_dribble_pay_is_set() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let g = dribbling(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0), 1000.0);
        for _ in 0..30 {
            assert_eq!(step(&r, &g, &g, &quiet(1), &mut memory), 0.0);
        }
        assert!(Track::load(&memory[Clock::SIZE..]).in_control);
    }

    #[test]
    fn dribbling_pays_a_little_and_orange_the_same() {
        let r = dribble_paid();
        let pay = |team: Team| {
            let mut memory = vec![0.0; r.memory()];
            let g = dribbling(team, Vec3A::new(0.0, -2000.0, 17.0), 1000.0);
            let first = step(&r, &g, &g, &quiet(1), &mut memory);
            let second = step(&r, &g, &g, &quiet(1), &mut memory);
            (first, second)
        };
        let (first, second) = pay(Team::Blue);
        // Full control, 1000 of 1400 uu/s toward goal: (0.1 + 0.1 x 0.71) x 8/120 at first, then decaying
        assert!((first - 0.1714 * 8.0 / 120.0).abs() < 1e-3, "{first}");
        assert!(second < first && second > 0.0);
        let (o_first, o_second) = pay(Team::Orange);
        assert!((first - o_first).abs() < 1e-5 && (second - o_second).abs() < 1e-5);
    }

    #[test]
    fn no_pay_without_the_ball_on_the_roof_and_memory_starts_again() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let mut g = dribbling(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0), 1000.0);
        g.ball.pos.z = 400.0;
        assert_eq!(step(&r, &g, &g, &quiet(1), &mut memory), 0.0);
        memory.fill(0.0);
        assert_eq!(Clock::load(&memory).now, 0.0);
        assert_eq!(step(&r, &g, &g, &quiet(1), &mut memory), 0.0);
    }

    #[test]
    fn a_forward_flick_toward_goal_off_a_dribble_pays_the_outcome_only_without_the_shape() {
        let r = MawkzyFlick {
            require_shape: false,
            use_elevation: false,
            ..reward()
        };
        let mut memory = vec![0.0; r.memory()];
        let g = dribbling(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0), 1000.0);
        for _ in 0..5 {
            step(&r, &g, &g, &quiet(1), &mut memory); // 0.33 s of control
        }
        // Jump: off the floor
        let mut jumped = g.clone();
        jumped.cars[0].on_ground = false;
        jumped.cars[0].has_jumped = true;
        jumped.cars[0].body.pos.z = 60.0;
        let jump = step(&r, &g, &jumped, &quiet(1), &mut memory);
        assert!(
            jump >= r.jump_setup_bonus,
            "the jump off the dribble: {jump}"
        );
        // Dodge (the window opens), touching the ball, which leaves at 2500 uu/s toward goal
        let mut dodged = jumped.clone();
        dodged.cars[0].has_flipped = true;
        let mut ev = touched(quiet(1), 0);
        ev.details[0].flip_rel_torque = Vec3A::new(0.7, -0.7, 0.0);
        dodged.ball.vel = Vec3A::new(0.0, 2500.0, 200.0);
        step(&r, &jumped, &dodged, &ev, &mut memory);
        // The ball away from the car: the launch is judged
        let mut away = dodged.clone();
        away.ball.pos.y += 400.0;
        let flick = step(&r, &dodged, &away, &quiet(1), &mut memory);
        // gain 1500 of 1800 (above 250), final 2500 of 3200: power 0.8, strong 0.875 x 0.5
        assert!(flick > 1.0 && flick < 1.5, "{flick}");
        assert!(!Track::load(&memory[Clock::SIZE..]).window_active);
    }

    /// After 0.33 s of dribbling and the pop, `steps` steps in the air turned into the Mawkzy pose (nose 38 deg
    /// off the goal line, level, roof rolled 49 deg toward the goal) with the ball on the roof. Returns the pay so
    /// far, the last state and the car's heading.
    fn popped_into_the_pose(
        r: &MawkzyFlick,
        memory: &mut [f32],
        steps: usize,
    ) -> (f32, GameSnapshot, f32) {
        let mut g = dribbling(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0), 1000.0);
        let mut total = 0.0;
        for _ in 0..5 {
            total += step(r, &g, &g, &quiet(1), memory);
        }
        let mut jumped = g.clone();
        jumped.cars[0].on_ground = false;
        jumped.cars[0].has_jumped = true;
        jumped.cars[0].body.pos.z = 60.0;
        total += step(r, &g, &jumped, &quiet(1), memory);
        g = jumped;
        let yaw = 0.5 * std::f32::consts::PI - 38f32.to_radians();
        let mut heading = 0.0;
        for i in 0..steps {
            let mut next = g.clone();
            let car = &mut next.cars[0];
            // Both roll directions: the one with the roof toward the goal
            let rot = [1.0f32, -1.0]
                .map(|s| {
                    Mat3A::from_rotation_z(yaw) * Mat3A::from_rotation_x(s * 49f32.to_radians())
                })
                .into_iter()
                .max_by(|a, b| a.z_axis.y.total_cmp(&b.z_axis.y))
                .unwrap();
            car.body.rot = rot;
            car.body.pos = Vec3A::new(0.0, -1900.0 + 20.0 * i as f32, 150.0);
            next.ball.pos = car.body.pos + rot.z_axis * 140.0 + rot.x_axis * 40.0;
            heading = MawkzyFlick::attitude(&car.body, Team::Blue, next.ball.pos).heading_deg;
            total += step(r, &g, &next, &quiet(1), memory);
            g = next;
        }
        (total, g, heading)
    }

    #[test]
    fn the_pose_shaping_is_paid_back_on_landing_so_hovering_in_the_pose_earns_nothing() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let (total, posed, _) = popped_into_the_pose(&r, &mut memory, 6);
        let phi = Track::load(&memory[Clock::SIZE..]).pose_phi;
        assert!(phi > 0.9, "in the pose with the ball on the roof: {phi}");
        assert!(total > r.jump_setup_bonus + 0.2, "{total}");
        let mut landed = dribbling(Team::Blue, Vec3A::new(0.0, -1700.0, 17.0), 1000.0);
        landed.ball.pos.z = 400.0;
        let back = step(&r, &posed, &landed, &quiet(1), &mut memory);
        assert!((back + r.pose_shaping_weight * phi).abs() < 1e-4, "{back}");
        let kept = total + back - r.jump_setup_bonus;
        assert!(
            kept <= 0.0 && kept > -0.05,
            "only the discount's cost: {kept}"
        );
    }

    #[test]
    fn a_mawkzy_dodge_keeps_the_pose_shaping_and_a_front_flip_pays_it_back() {
        let r = reward();
        let dodge = |torque_x: f32, torque_y: f32| {
            let mut memory = vec![0.0; r.memory()];
            let (_, posed, heading) = popped_into_the_pose(&r, &mut memory, 6);
            let phi = Track::load(&memory[Clock::SIZE..]).pose_phi;
            let mut dodged = posed.clone();
            dodged.cars[0].has_flipped = true;
            let mut ev = quiet(1);
            ev.details[0].flip_rel_torque = Vec3A::new(torque_x * heading.signum(), torque_y, 0.0);
            let pay = step(&r, &posed, &dodged, &ev, &mut memory);
            (pay, phi, Track::load(&memory[Clock::SIZE..]))
        };
        let (pay, phi, tr) = dodge(0.7, -0.7); // The diagonal away from the turn side, backward
        assert!(
            tr.window_active && tr.shape > 0.5,
            "a Mawkzy attempt: {}",
            tr.shape
        );
        let kept = r.pose_shaping_weight * phi * tr.shape.min(1.0);
        assert!((tr.pose_banked - kept).abs() < 1e-5, "kept times the shape");
        let back = r.pose_shaping_weight * phi - kept;
        assert!(
            (pay - (r.setup_shape_bonus * tr.shape - back)).abs() < 1e-4,
            "{pay}"
        );
        let (pay, phi, tr) = dodge(0.0, 1.0); // A front flip
        assert!(tr.window_active && tr.shape == 0.0);
        assert!(
            (pay + r.pose_shaping_weight * phi).abs() < 1e-4,
            "paid back: {pay}"
        );
    }

    #[test]
    fn a_raze_or_a_floor_crash_gives_back_the_pose_shaping_and_the_shape_bonus() {
        let r = reward();
        let ball = Body {
            vel: Vec3A::new(0.0, 2500.0, 300.0),
            ..Body::default()
        };
        for (rear_contact, crash) in [(true, false), (false, true)] {
            let mut tr = Track {
                pose_banked: 0.2,
                shape: 0.8,
                have_contact: true,
                contact_forward: if rear_contact { -50.0 } else { 50.0 },
                floor_crash: crash,
                ..Track::default()
            };
            let paid = r.resolve(&mut tr, Team::Blue, &ball, Ending::Launch);
            assert!(
                (paid + 0.2 + r.setup_shape_bonus * 0.8).abs() < 1e-5,
                "{paid}"
            );
            assert_eq!(tr.pose_banked, 0.0);
        }
    }
}
