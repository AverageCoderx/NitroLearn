//! The Raze flick: from a hood dribble, a pop, a dodge with the roof tilted, the car's top bouncing off
//! the floor, then its rear popping the ball up toward the goal.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::flick_kit::*;

/// A Raze flick, paid once when the ball has left the car after the whole move (dribble, dodge with the
/// roof tilted, floor rebound, rear hit): up to flick_weight for the pop, the pogo, the shot and its power;
/// down to -backward_penalty for one sent toward its own goal. Optional small credits for the stages.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RazeFlick {
    /// Ball under control on the roof: its centre at least this far above the car along its roof direction (uu). Default 105.
    #[serde(default = "rd::dribble_min_up")]
    pub dribble_min_up: f32,
    /// ... and at most this far (uu). Default 185.
    #[serde(default = "rd::dribble_max_up")]
    pub dribble_max_up: f32,
    /// ... best this far forward of the car's centre (uu). Default 20.
    #[serde(default = "rd::dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// ... no control this far or further from that spot, sideways (uu). Default 90.
    #[serde(default = "rd::dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// ... nor at this speed relative to the car (uu/s). Default 600.
    #[serde(default = "rd::dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality (0-1) that counts as the ball under control. Default 0.25.
    #[serde(default = "rd::control_threshold")]
    pub control_threshold: f32,
    /// Seconds of control before the jump (the Raze set: 0.1). Default 0.25.
    #[serde(default = "rd::min_possession_time")]
    pub min_possession_time: f32,
    /// The dodge must start within this many seconds of the last controlled step. Default 1.
    #[serde(default = "rd::possession_memory")]
    pub possession_memory: f32,
    /// Car-ball distance at the dodge, at most (uu). Default 400.
    #[serde(default = "rd::arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Ball centre above the car centre at the dodge, at least (uu): the ball was popped. Default 60.
    #[serde(default = "rd::arm_min_ball_above")]
    pub arm_min_ball_above: f32,
    /// Car centre height at the dodge, at most (uu). Default 250.
    #[serde(default = "rd::arm_max_car_height")]
    pub arm_max_car_height: f32,
    /// Roof tilted at the dodge: its upward component at most this. Default 0.5.
    #[serde(default = "rd::arm_max_up_z")]
    pub arm_max_up_z: f32,
    /// Seconds after the dodge for the floor rebound. Default 0.75.
    #[serde(default = "rd::rebound_window")]
    pub rebound_window: f32,
    /// Roof facing down at the rebound: its upward component at most this. Default -0.2.
    #[serde(default = "rd::rebound_max_up_z")]
    pub rebound_max_up_z: f32,
    /// Car centre near the floor at the rebound: at most this high (uu). Default 140.
    #[serde(default = "rd::rebound_max_car_z")]
    pub rebound_max_car_z: f32,
    /// Upward speed after the bounce, at least (uu/s). Default 80.
    #[serde(default = "rd::rebound_min_vz")]
    pub rebound_min_vz: f32,
    /// ... gained since the lowest point after the dodge, at least (uu/s). Default 150.
    #[serde(default = "rd::rebound_min_dvz")]
    pub rebound_min_dvz: f32,
    /// Rebound and hit in one step: the ball centre at least this far above the car's centre at the hit (uu). Default 50.
    #[serde(default = "rd::same_step_min_ball_above")]
    pub same_step_min_ball_above: f32,
    /// Seconds after the rebound for the hit. Default 0.5.
    #[serde(default = "rd::strike_window")]
    pub strike_window: f32,
    /// The hit must be with the rear: contact point car-local forward at most this (uu). Default -20.
    #[serde(default = "rd::rear_max_forward")]
    pub rear_max_forward: f32,
    /// The car's hitbox length (uu; the Octane's). Default 120.507.
    #[serde(default = "rd::hitbox_length")]
    pub hitbox_length: f32,
    /// The car's hitbox width (uu). Default 86.6994.
    #[serde(default = "rd::hitbox_width")]
    pub hitbox_width: f32,
    /// The car's hitbox height (uu). Default 38.6591.
    #[serde(default = "rd::hitbox_height")]
    pub hitbox_height: f32,
    /// The hitbox centre's offset from the car's centre of mass, forward (uu). Default 13.87566.
    #[serde(default = "rd::hitbox_offset_forward")]
    pub hitbox_offset_forward: f32,
    /// ... to the right (uu). Default 0.
    #[serde(default = "rd::hitbox_offset_right")]
    pub hitbox_offset_right: f32,
    /// ... up (uu). Default 20.755.
    #[serde(default = "rd::hitbox_offset_up")]
    pub hitbox_offset_up: f32,
    /// Optional: the nose must be turned away from the goal at the dodge (the 'fake'). Default false.
    #[serde(default = "rd::require_turned_setup")]
    pub require_turned_setup: bool,
    /// ... by at least this much (degrees). Default 25.
    #[serde(default = "rd::turned_min_deg")]
    pub turned_min_deg: f32,
    /// ... and at most this much (degrees). Default 110.
    #[serde(default = "rd::turned_max_deg")]
    pub turned_max_deg: f32,
    /// The ball's velocity change (uu/s) needed for any credit. Default 300.
    #[serde(default = "rd::min_impulse")]
    pub min_impulse: f32,
    /// The velocity change for a full impulse term (uu/s). Default 2000.
    #[serde(default = "rd::full_impulse")]
    pub full_impulse: f32,
    /// The ball's upward velocity change needed for any credit, where the pop term starts (uu/s). Default 200.
    #[serde(default = "rd::min_pop")]
    pub min_pop: f32,
    /// ... and where it is full (uu/s). Default 1500.
    #[serde(default = "rd::full_pop")]
    pub full_pop: f32,
    /// The car's upward speed after the floor bounce where the pogo term starts (uu/s). Default 100.
    #[serde(default = "rd::min_pogo")]
    pub min_pogo: f32,
    /// ... and where it is full (uu/s). Default 600.
    #[serde(default = "rd::full_pogo")]
    pub full_pogo: f32,
    /// Goal-directed velocity change where the power term starts (uu/s). Default 0.
    #[serde(default = "rd::min_gain")]
    pub min_gain: f32,
    /// ... and where it is full (uu/s). Default 1000.
    #[serde(default = "rd::full_gain")]
    pub full_gain: f32,
    /// Goal-directed launch speed for full power (uu/s). Default 2400.
    #[serde(default = "rd::full_launch_speed")]
    pub full_launch_speed: f32,
    /// The shot term for any shot on target; the rest grows toward the posts. Default 0.5.
    #[serde(default = "rd::shot_on_target")]
    pub shot_on_target: f32,
    /// |x| at the goal line where the corner part of the shot term starts (uu). Default 250.
    #[serde(default = "rd::corner_from_x")]
    pub corner_from_x: f32,
    /// ... and where it is full (uu). Default 760.
    #[serde(default = "rd::corner_full_x")]
    pub corner_full_x: f32,
    /// Seconds of ball flight projected toward the goal. Default 5.
    #[serde(default = "rd::shot_max_time")]
    pub shot_max_time: f32,
    /// Quality mix: weight of goal-directed power. Default 0.1.
    #[serde(default = "rd::w_power")]
    pub w_power: f32,
    /// Quality mix: weight of the hit's strength. Default 0.1.
    #[serde(default = "rd::w_impulse")]
    pub w_impulse: f32,
    /// Quality mix: weight of the pop. Default 0.3.
    #[serde(default = "rd::w_pop")]
    pub w_pop: f32,
    /// Quality mix: weight of the pogo. Default 0.3.
    #[serde(default = "rd::w_pogo")]
    pub w_pogo: f32,
    /// Quality mix: weight of the shot on net. Default 0.2.
    #[serde(default = "rd::w_shot")]
    pub w_shot: f32,
    /// Launch direction vs the goal (cosine): no credit at or below this. Default 0.
    #[serde(default = "rd::aim_cos_zero")]
    pub aim_cos_zero: f32,
    /// ... full credit at or above this. Default 0.85.
    #[serde(default = "rd::aim_cos_full")]
    pub aim_cos_full: f32,
    /// Launch elevation (degrees): no credit at or below this. Default 0.
    #[serde(default = "rd::elev_zero_low")]
    pub elev_zero_low: f32,
    /// ... full credit from this. Default 8.
    #[serde(default = "rd::elev_full_low")]
    pub elev_full_low: f32,
    /// ... full credit up to this. Default 45.
    #[serde(default = "rd::elev_full_high")]
    pub elev_full_high: f32,
    /// ... none from this (a ball popped straight up). Default 70.
    #[serde(default = "rd::elev_zero_high")]
    pub elev_zero_high: f32,
    /// Share of flick_weight for any successful Raze. Default 0.25.
    #[serde(default = "rd::base_credit")]
    pub base_credit: f32,
    /// Training option: extra for a shot on target (the Raze set: 0.6). Default 0.
    #[serde(default = "rd::on_net_bonus")]
    pub on_net_bonus: f32,
    /// Share of on_net_bonus paid flat; the rest grows with the shot's goal speed (Raze bot: 0.5). Default 1.
    #[serde(default = "rd::on_net_power_mix")]
    pub on_net_power_mix: f32,
    /// Training option: extra for a harder pogo (Raze bot: 0.35). Default 0.
    #[serde(default = "rd::stretch_pogo_weight")]
    pub stretch_pogo_weight: f32,
    /// ... starting at this car bounce speed (uu/s; Raze bot: 300). Default 600.
    #[serde(default = "rd::stretch_pogo_from")]
    pub stretch_pogo_from: f32,
    /// ... full at this (uu/s; Raze bot: 450). Default 1200.
    #[serde(default = "rd::stretch_pogo_full")]
    pub stretch_pogo_full: f32,
    /// Training option: extra for a stronger pop (Raze bot: 0.3). Default 0.
    #[serde(default = "rd::stretch_pop_weight")]
    pub stretch_pop_weight: f32,
    /// ... starting at this upward velocity change (uu/s; Raze bot: 750). Default 1500.
    #[serde(default = "rd::stretch_pop_from")]
    pub stretch_pop_from: f32,
    /// ... full at this (uu/s; Raze bot: 1200). Default 2500.
    #[serde(default = "rd::stretch_pop_full")]
    pub stretch_pop_full: f32,
    /// Training option: extra for a stronger hit (Raze bot: 0.1). Default 0.
    #[serde(default = "rd::stretch_impulse_weight")]
    pub stretch_impulse_weight: f32,
    /// ... starting at this velocity change (uu/s; Raze bot: 1000). Default 2000.
    #[serde(default = "rd::stretch_impulse_from")]
    pub stretch_impulse_from: f32,
    /// ... full at this (uu/s; Raze bot: 1500). Default 3500.
    #[serde(default = "rd::stretch_impulse_full")]
    pub stretch_impulse_full: f32,
    /// Training option: extra for a bigger goal-directed velocity change (Raze bot: 0.15). Default 0.
    #[serde(default = "rd::stretch_gain_weight")]
    pub stretch_gain_weight: f32,
    /// ... starting at this (uu/s; Raze bot: 350). Default 1000.
    #[serde(default = "rd::stretch_gain_from")]
    pub stretch_gain_from: f32,
    /// ... full at this (uu/s; Raze bot: 800). Default 2500.
    #[serde(default = "rd::stretch_gain_full")]
    pub stretch_gain_full: f32,
    /// Training option: extra for a powerful shot on target (Raze bot: 0.25). Default 0.
    #[serde(default = "rd::stretch_shot_weight")]
    pub stretch_shot_weight: f32,
    /// ... starting at this goal-directed speed (uu/s). Default 1600.
    #[serde(default = "rd::stretch_shot_from")]
    pub stretch_shot_from: f32,
    /// ... full at this (uu/s; Raze bot: 3000). Default 4000.
    #[serde(default = "rd::stretch_shot_full")]
    pub stretch_shot_full: f32,
    /// Another car near or bumping this one around the attempt cancels it. Default true.
    #[serde(default = "rd::cancel_on_car_contact")]
    pub cancel_on_car_contact: bool,
    /// How close another car's hitbox counts as contact (uu). Default 40.
    #[serde(default = "rd::car_contact_margin")]
    pub car_contact_margin: f32,
    /// Seconds before the dodge that contact still counts. Default 1.
    #[serde(default = "rd::car_contact_lookback")]
    pub car_contact_lookback: f32,
    /// What a full Raze pays. Default 1.
    #[serde(default = "rd::flick_weight")]
    pub flick_weight: f32,
    /// Training option: paid once when an attempt arms (Raze bot: 0.1). Default 0.
    #[serde(default = "rd::arm_credit")]
    pub arm_credit: f32,
    /// Training option: paid once for the floor rebound with the ball close above (Raze bot: 0.2). Default 0.
    #[serde(default = "rd::rebound_credit")]
    pub rebound_credit: f32,
    /// ... the ball within this distance of the car (uu). Default 450.
    #[serde(default = "rd::rebound_credit_max_dist")]
    pub rebound_credit_max_dist: f32,
    /// Training option: an episode starting with the ball near and above the car counts as coming from a dribble (Raze bot: true). Default false.
    #[serde(default = "rd::possession_at_reset")]
    pub possession_at_reset: bool,
    /// ... the ball within this distance (uu). Default 500.
    #[serde(default = "rd::possession_at_reset_max_dist")]
    pub possession_at_reset_max_dist: f32,
    /// ... and at least this high (uu; Raze bot: 140). Default 0.
    #[serde(default = "rd::possession_at_reset_min_ball_z")]
    pub possession_at_reset_min_ball_z: f32,
    /// Largest penalty for a Raze that sends the ball toward its own goal. Default 0.5.
    #[serde(default = "rd::backward_penalty")]
    pub backward_penalty: f32,
    /// Horizontal velocity change before that penalty applies (uu/s). Default 500.
    #[serde(default = "rd::min_penalty_impulse")]
    pub min_penalty_impulse: f32,
    /// The outcome is judged once the ball is this far from the car (uu) .... Default 300.
    #[serde(default = "rd::separation_dist")]
    pub separation_dist: f32,
    /// ... or this many seconds after the hit. Default 0.3.
    #[serde(default = "rd::resolve_time")]
    pub resolve_time: f32,
    /// Launches this close to a wall or the ceiling are cancelled (uu). Default 200.
    #[serde(default = "rd::wall_margin")]
    pub wall_margin: f32,
}

/// RazeFlick's defaults.
mod rd {
    pub fn dribble_min_up() -> f32 {
        105.0
    }
    pub fn dribble_max_up() -> f32 {
        185.0
    }
    pub fn dribble_target_forward() -> f32 {
        20.0
    }
    pub fn dribble_max_planar_dist() -> f32 {
        90.0
    }
    pub fn dribble_max_rel_speed() -> f32 {
        600.0
    }
    pub fn control_threshold() -> f32 {
        0.25
    }
    pub fn min_possession_time() -> f32 {
        0.25
    }
    pub fn possession_memory() -> f32 {
        1.0
    }
    pub fn arm_max_ball_dist() -> f32 {
        400.0
    }
    pub fn arm_min_ball_above() -> f32 {
        60.0
    }
    pub fn arm_max_car_height() -> f32 {
        250.0
    }
    pub fn arm_max_up_z() -> f32 {
        0.5
    }
    pub fn rebound_window() -> f32 {
        0.75
    }
    pub fn rebound_max_up_z() -> f32 {
        -0.2
    }
    pub fn rebound_max_car_z() -> f32 {
        140.0
    }
    pub fn rebound_min_vz() -> f32 {
        80.0
    }
    pub fn rebound_min_dvz() -> f32 {
        150.0
    }
    pub fn same_step_min_ball_above() -> f32 {
        50.0
    }
    pub fn strike_window() -> f32 {
        0.5
    }
    pub fn rear_max_forward() -> f32 {
        -20.0
    }
    pub fn hitbox_length() -> f32 {
        120.507
    }
    pub fn hitbox_width() -> f32 {
        86.6994
    }
    pub fn hitbox_height() -> f32 {
        38.6591
    }
    pub fn hitbox_offset_forward() -> f32 {
        13.87566
    }
    pub fn hitbox_offset_right() -> f32 {
        0.0
    }
    pub fn hitbox_offset_up() -> f32 {
        20.755
    }
    pub fn require_turned_setup() -> bool {
        false
    }
    pub fn turned_min_deg() -> f32 {
        25.0
    }
    pub fn turned_max_deg() -> f32 {
        110.0
    }
    pub fn min_impulse() -> f32 {
        300.0
    }
    pub fn full_impulse() -> f32 {
        2000.0
    }
    pub fn min_pop() -> f32 {
        200.0
    }
    pub fn full_pop() -> f32 {
        1500.0
    }
    pub fn min_pogo() -> f32 {
        100.0
    }
    pub fn full_pogo() -> f32 {
        600.0
    }
    pub fn min_gain() -> f32 {
        0.0
    }
    pub fn full_gain() -> f32 {
        1000.0
    }
    pub fn full_launch_speed() -> f32 {
        2400.0
    }
    pub fn shot_on_target() -> f32 {
        0.5
    }
    pub fn corner_from_x() -> f32 {
        250.0
    }
    pub fn corner_full_x() -> f32 {
        760.0
    }
    pub fn shot_max_time() -> f32 {
        5.0
    }
    pub fn w_power() -> f32 {
        0.1
    }
    pub fn w_impulse() -> f32 {
        0.1
    }
    pub fn w_pop() -> f32 {
        0.3
    }
    pub fn w_pogo() -> f32 {
        0.3
    }
    pub fn w_shot() -> f32 {
        0.2
    }
    pub fn aim_cos_zero() -> f32 {
        0.0
    }
    pub fn aim_cos_full() -> f32 {
        0.85
    }
    pub fn elev_zero_low() -> f32 {
        0.0
    }
    pub fn elev_full_low() -> f32 {
        8.0
    }
    pub fn elev_full_high() -> f32 {
        45.0
    }
    pub fn elev_zero_high() -> f32 {
        70.0
    }
    pub fn base_credit() -> f32 {
        0.25
    }
    pub fn on_net_bonus() -> f32 {
        0.0
    }
    pub fn on_net_power_mix() -> f32 {
        1.0
    }
    pub fn stretch_pogo_weight() -> f32 {
        0.0
    }
    pub fn stretch_pogo_from() -> f32 {
        600.0
    }
    pub fn stretch_pogo_full() -> f32 {
        1200.0
    }
    pub fn stretch_pop_weight() -> f32 {
        0.0
    }
    pub fn stretch_pop_from() -> f32 {
        1500.0
    }
    pub fn stretch_pop_full() -> f32 {
        2500.0
    }
    pub fn stretch_impulse_weight() -> f32 {
        0.0
    }
    pub fn stretch_impulse_from() -> f32 {
        2000.0
    }
    pub fn stretch_impulse_full() -> f32 {
        3500.0
    }
    pub fn stretch_gain_weight() -> f32 {
        0.0
    }
    pub fn stretch_gain_from() -> f32 {
        1000.0
    }
    pub fn stretch_gain_full() -> f32 {
        2500.0
    }
    pub fn stretch_shot_weight() -> f32 {
        0.0
    }
    pub fn stretch_shot_from() -> f32 {
        1600.0
    }
    pub fn stretch_shot_full() -> f32 {
        4000.0
    }
    pub fn cancel_on_car_contact() -> bool {
        true
    }
    pub fn car_contact_margin() -> f32 {
        40.0
    }
    pub fn car_contact_lookback() -> f32 {
        1.0
    }
    pub fn flick_weight() -> f32 {
        1.0
    }
    pub fn arm_credit() -> f32 {
        0.0
    }
    pub fn rebound_credit() -> f32 {
        0.0
    }
    pub fn rebound_credit_max_dist() -> f32 {
        450.0
    }
    pub fn possession_at_reset() -> bool {
        false
    }
    pub fn possession_at_reset_max_dist() -> f32 {
        500.0
    }
    pub fn possession_at_reset_min_ball_z() -> f32 {
        0.0
    }
    pub fn backward_penalty() -> f32 {
        0.5
    }
    pub fn min_penalty_impulse() -> f32 {
        500.0
    }
    pub fn separation_dist() -> f32 {
        300.0
    }
    pub fn resolve_time() -> f32 {
        0.3
    }
    pub fn wall_margin() -> f32 {
        200.0
    }
}

// Memory, per car.
const INIT: usize = 0;
const POSSESSION: usize = 1;
const CONTROL_TIME: usize = 2;
const CONTROL_AGE: usize = 3;
const ATTEMPT_USED: usize = 4;
const STAGE: usize = 5;
const ARM_AGE: usize = 6;
const REBOUND_AGE: usize = 7;
const STRIKE_AGE: usize = 8;
const POGO_VZ: usize = 9;
const AUTO_FLIP: usize = 10;
const CAR_CONTACT: usize = 11;
const BALL_VEL0: usize = 12;
/// Ticks since the last car-ball contact ended.
const SINCE_HIT: usize = 15;
const MIN_VZ_ARMED: usize = 16;
const ANY_CAR_CONTACT: usize = 17;
const CAR_CONTACT_AGE: usize = 18;
const MEMORY: usize = 19;

const IDLE: f32 = 0.0;
const ARMED: f32 = 1.0;
const REBOUNDED: f32 = 2.0;
const STRUCK: f32 = 3.0;

/// How an attempt ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ending {
    Launch,
    Other,
}

impl RazeFlick {
    fn hitbox(&self) -> Hitbox {
        Hitbox {
            size: Vec3A::new(self.hitbox_length, self.hitbox_width, self.hitbox_height),
            offset: Vec3A::new(
                self.hitbox_offset_forward,
                self.hitbox_offset_right,
                self.hitbox_offset_up,
            ),
        }
    }

    fn dribble(&self) -> Dribble {
        Dribble {
            min_up: self.dribble_min_up,
            max_up: self.dribble_max_up,
            target_forward: self.dribble_target_forward,
            max_planar_dist: self.dribble_max_planar_dist,
            max_rel_speed: self.dribble_max_rel_speed,
        }
    }

    /// Next to a wall or the ceiling (the back walls outside the goal mouth).
    fn near_wall(&self, p: Vec3A) -> bool {
        let m = self.wall_margin + CV_BALL_RADIUS;
        p.x.abs() > SIDE_WALL_X - m
            || p.z > CEILING_Z - m
            || (p.y.abs() > BACK_WALL_Y - m && (p.x.abs() > GOAL_HALF_WIDTH || p.z > GOAL_HEIGHT))
    }

    fn end_possession(m: &mut [f32]) {
        m[POSSESSION] = 0.0;
        m[CONTROL_TIME] = 0.0;
        m[ATTEMPT_USED] = 0.0;
    }

    /// What a launched Raze earns.
    fn outcome(&self, m: &[f32], cur: &GameSnapshot, team: Team) -> f32 {
        let ball = &cur.ball;
        let goal_dir = dir2d(target_goal(team) - ball.pos);
        let v1 = ball.vel;
        let v0 = Vec3A::new(m[BALL_VEL0], m[BALL_VEL0 + 1], m[BALL_VEL0 + 2]);
        // The gravity since the contact ended undone, so the elevation is the launch angle
        let vz1 = v1.z - GRAVITY_Z * m[SINCE_HIT].max(0.0) * TICK_SECONDS;
        let flat_dv = Vec3A::new(v1.x - v0.x, v1.y - v0.y, 0.0);
        let flat_v1 = Vec3A::new(v1.x, v1.y, 0.0);
        let horiz_speed = flat_v1.length();
        let gain = flat_dv.dot(goal_dir);
        let final_goal_speed = flat_v1.dot(goal_dir);
        let aim_cos = dir2d(v1).dot(goal_dir);
        let elevation = degrees(vz1.atan2(horiz_speed));
        let pop = vz1 - v0.z;
        let impulse = Vec3A::new(flat_dv.x, flat_dv.y, pop).length();
        let pogo = m[POGO_VZ];
        let power = (ramp(gain, self.min_gain, self.full_gain)
            * clamp01(div(final_goal_speed, self.full_launch_speed)))
        .sqrt();
        let impulse_term = ramp(impulse, self.min_impulse, self.full_impulse);
        let pop_term = ramp(pop, self.min_pop, self.full_pop);
        let pogo_term = ramp(pogo, self.min_pogo, self.full_pogo);
        let shot = project_to_goal_line(
            ball.pos,
            Vec3A::new(v1.x, v1.y, vz1),
            team,
            self.shot_max_time,
        );
        let shot_term = if shot.on_target {
            self.shot_on_target
                + (1.0 - self.shot_on_target)
                    * ramp(shot.x.abs(), self.corner_from_x, self.corner_full_x)
        } else {
            0.0
        };
        let quality = self.w_power * power
            + self.w_impulse * impulse_term
            + self.w_pop * pop_term
            + self.w_pogo * pogo_term
            + self.w_shot * shot_term;
        let aim = ramp(aim_cos, self.aim_cos_zero, self.aim_cos_full);
        let elev = if elevation < self.elev_full_low {
            ramp(elevation, self.elev_zero_low, self.elev_full_low)
        } else if self.elev_zero_high == self.elev_full_high {
            if elevation < self.elev_zero_high {
                1.0
            } else {
                0.0
            }
        } else {
            clamp01((self.elev_zero_high - elevation) / (self.elev_zero_high - self.elev_full_high))
        };
        // A real hit that pops the ball up and sends it goalward
        let success = impulse >= self.min_impulse
            && pop >= self.min_pop
            && final_goal_speed > 0.0
            && gain > -self.min_impulse;
        let mut reward = 0.0;
        if success {
            let fast_shot = ramp(
                final_goal_speed,
                self.stretch_shot_from,
                self.stretch_shot_full,
            );
            reward = self.flick_weight
                * (self.base_credit + (1.0 - self.base_credit) * quality)
                * aim
                * elev;
            if shot.on_target {
                reward += self.flick_weight
                    * self.on_net_bonus
                    * elev
                    * (self.on_net_power_mix + (1.0 - self.on_net_power_mix) * fast_shot);
            }
            let shot_part = if shot.on_target {
                self.stretch_shot_weight * fast_shot
            } else {
                0.0
            };
            reward += self.flick_weight
                * aim
                * elev
                * (self.stretch_pogo_weight
                    * ramp(pogo, self.stretch_pogo_from, self.stretch_pogo_full)
                    + self.stretch_pop_weight
                        * ramp(pop, self.stretch_pop_from, self.stretch_pop_full)
                    + self.stretch_impulse_weight
                        * ramp(
                            impulse,
                            self.stretch_impulse_from,
                            self.stretch_impulse_full,
                        )
                    + self.stretch_gain_weight
                        * ramp(gain, self.stretch_gain_from, self.stretch_gain_full)
                    + shot_part);
        }
        if final_goal_speed < 0.0 && flat_dv.length() >= self.min_penalty_impulse {
            reward -= self.backward_penalty * clamp01(div(-final_goal_speed, self.full_gain));
        }
        if self.cancel_on_car_contact && m[CAR_CONTACT] != 0.0 {
            reward = 0.0; // judged, but another car took part
        }
        reward
    }

    /// Ends the attempt in progress; what it pays (only a launch pays).
    fn finish(&self, m: &mut [f32], cur: &GameSnapshot, car: usize, ending: Ending) -> f32 {
        let reward = if ending == Ending::Launch {
            self.outcome(m, cur, cur.cars[car].team)
        } else {
            0.0
        };
        m[STAGE] = IDLE;
        m[ARM_AGE..=CAR_CONTACT].fill(0.0);
        reward
    }

    fn step(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        let before = &prev.cars[car];
        if m[INIT] == 0.0 {
            m[INIT] = 1.0;
            // Training option: an episode starting mid-Raze counts as coming from a dribble
            let ball = prev.ball.pos;
            if self.possession_at_reset
                && before.body.pos.distance(ball) <= self.possession_at_reset_max_dist
                && ball.z > before.body.pos.z
                && ball.z >= self.possession_at_reset_min_ball_z
            {
                m[POSSESSION] = 1.0;
                m[CONTROL_TIME] = self.min_possession_time;
                m[CONTROL_AGE] = 0.0;
            }
        }
        let ticks = step_ticks(events);
        for age in [
            CONTROL_AGE,
            ARM_AGE,
            REBOUND_AGE,
            STRIKE_AGE,
            SINCE_HIT,
            CAR_CONTACT_AGE,
        ] {
            m[age] += ticks;
        }
        let me = &cur.cars[car];
        let detail = events.detail(car);
        let now = events.tick;
        let dt = events.step_seconds.clamp(0.0, 0.5);
        // A goal ends the episode: the one last step Nitro lets a reward recognise
        let last = cur.goal_scored.is_some();
        if me.is_demoed {
            if m[STAGE] != IDLE {
                self.finish(m, cur, car, Ending::Other);
            }
            Self::end_possession(m);
            return 0.0;
        }

        // Possession
        let in_control = control_quality(me, &cur.ball, self.dribble()) >= self.control_threshold;
        let memory_ticks = secs_to_ticks(self.possession_memory);
        let mut since_control = if m[POSSESSION] != 0.0 {
            m[CONTROL_AGE]
        } else {
            f32::INFINITY
        };
        if in_control {
            if m[POSSESSION] == 0.0 {
                m[POSSESSION] = 1.0;
                m[CONTROL_TIME] = 0.0;
            }
            m[CONTROL_TIME] += dt;
            m[CONTROL_AGE] = 0.0;
            since_control = 0.0;
        } else if m[POSSESSION] != 0.0 && m[STAGE] == IDLE && since_control > memory_ticks {
            Self::end_possession(m);
        }
        let recent_control = m[POSSESSION] != 0.0
            && since_control <= memory_ticks
            && m[CONTROL_TIME] >= self.min_possession_time - 1e-4;

        let other_touch = events
            .cars
            .iter()
            .enumerate()
            .any(|(other, e)| other != car && e.touched_ball);
        if m[STAGE] != IDLE && other_touch {
            return self.finish(m, cur, car, Ending::Other);
        }
        // Car contact: remembered while idle (for the lookback before the dodge), marked on an attempt
        if cur.cars.len() > 1
            && car_contact(cur, events, car, self.hitbox(), self.car_contact_margin)
        {
            m[CAR_CONTACT_AGE] = 0.0;
            m[ANY_CAR_CONTACT] = 1.0;
            if m[STAGE] != IDLE {
                m[CAR_CONTACT] = 1.0;
            }
        }
        let touched = events.cars.get(car).is_some_and(|e| e.touched_ball);
        let mut reward = 0.0;

        if m[STAGE] == IDLE {
            // Arm on the dodge's start
            let dodge_started = me.has_flipped && !before.has_flipped;
            if !(dodge_started
                && recent_control
                && m[ATTEMPT_USED] == 0.0
                && !me.on_ground
                && !touched)
            {
                return 0.0;
            }
            let dist = me.body.pos.distance(cur.ball.pos);
            let above = cur.ball.pos.z - me.body.pos.z;
            // The attitude at the dodge's press: this step or the one before
            let up_z = me.body.rot.z_axis.z.min(before.body.rot.z_axis.z);
            let nose = dir2d(before.body.rot.x_axis);
            let to_goal = dir2d(target_goal(me.team) - before.body.pos);
            let heading = degrees(nose.dot(to_goal).clamp(-1.0, 1.0).acos());
            let turned_ok = !self.require_turned_setup
                || (heading >= self.turned_min_deg && heading <= self.turned_max_deg);
            if dist <= self.arm_max_ball_dist
                && above >= self.arm_min_ball_above
                && me.body.pos.z <= self.arm_max_car_height
                && up_z <= self.arm_max_up_z
                && turned_ok
            {
                m[ATTEMPT_USED] = 1.0;
                m[STAGE] = ARMED;
                m[ARM_AGE..=CAR_CONTACT].fill(0.0);
                if m[ANY_CAR_CONTACT] != 0.0
                    && m[CAR_CONTACT_AGE] <= secs_to_ticks(self.car_contact_lookback)
                {
                    m[CAR_CONTACT] = 1.0;
                }
                m[MIN_VZ_ARMED] = me.body.vel.z;
                return self.arm_credit;
            }
            return 0.0;
        }

        if m[STAGE] == ARMED {
            // Upward speed gained since the lowest point after the dodge (a bounce spreads over steps)
            m[MIN_VZ_ARMED] = m[MIN_VZ_ARMED].min(before.body.vel.z);
            let dvz = me.body.vel.z - m[MIN_VZ_ARMED];
            let inverted =
                me.body.rot.z_axis.z.min(before.body.rot.z_axis.z) <= self.rebound_max_up_z;
            let near_floor = me.body.pos.z.min(before.body.pos.z) <= self.rebound_max_car_z;
            // An auto-flip also pushes an inverted car up: not a rebound
            let auto_flip = detail.is_auto_flipping || events.prev_detail(car).is_auto_flipping;
            let rebound = inverted
                && near_floor
                && !auto_flip
                && !me.on_ground
                && me.body.vel.z >= self.rebound_min_vz
                && dvz >= self.rebound_min_dvz;
            if touched {
                // Rebound and hit in one step count only with the ball above the car at the hit
                let ball_above = ball_at_hit(&detail, &cur.ball, now).z
                    >= me.body.pos.z + self.same_step_min_ball_above;
                if !(rebound && ball_above) {
                    return self.finish(m, cur, car, Ending::Other); // an ordinary flick
                }
            }
            if !rebound {
                if m[ARM_AGE] >= secs_to_ticks(self.rebound_window) || last {
                    return self.finish(m, cur, car, Ending::Other);
                }
                return 0.0;
            }
            m[STAGE] = REBOUNDED;
            m[REBOUND_AGE] = 0.0;
            m[POGO_VZ] = me.body.vel.z;
            let ball_close = me.body.pos.distance(cur.ball.pos) <= self.rebound_credit_max_dist
                && cur.ball.pos.z > me.body.pos.z;
            if ball_close && !(self.cancel_on_car_contact && m[CAR_CONTACT] != 0.0) {
                reward += self.rebound_credit;
            }
        }

        if m[STAGE] == REBOUNDED {
            // The bounce keeps lifting the car for a few ticks: keep its peak, frozen once an auto-flip starts
            if detail.is_auto_flipping {
                m[AUTO_FLIP] = 1.0;
            }
            if !touched && m[AUTO_FLIP] == 0.0 {
                m[POGO_VZ] = m[POGO_VZ].max(me.body.vel.z);
            }
            if touched {
                let (local, _) =
                    hit_contact_local(&me.body, &detail, &cur.ball, now, self.hitbox());
                if local.x > self.rear_max_forward {
                    return reward + self.finish(m, cur, car, Ending::Other);
                }
                m[STAGE] = STRUCK;
                m[STRIKE_AGE] = 0.0;
                m[BALL_VEL0..BALL_VEL0 + 3].copy_from_slice(&prev.ball.vel.to_array());
                m[SINCE_HIT] = now.saturating_sub(detail.last_hit_tick.unwrap_or(now)) as f32;
            } else if m[REBOUND_AGE] >= secs_to_ticks(self.strike_window) || last {
                return reward + self.finish(m, cur, car, Ending::Other);
            }
            if !last {
                return reward;
            }
        }

        // Struck: judged once the contact is over
        let strike_step = m[STRIKE_AGE] == 0.0;
        if touched && !strike_step {
            m[SINCE_HIT] = now.saturating_sub(detail.last_hit_tick.unwrap_or(now)) as f32;
        }
        let contact_over = !touched && !strike_step;
        let separated = me.body.pos.distance(cur.ball.pos) >= self.separation_dist;
        let time_up = m[STRIKE_AGE] >= secs_to_ticks(self.resolve_time);
        if contact_over || separated || time_up || last {
            if self.near_wall(cur.ball.pos) && !last {
                return reward + self.finish(m, cur, car, Ending::Other);
            }
            reward += self.finish(m, cur, car, Ending::Launch);
        }
        reward
    }
}

impl Reward for RazeFlick {
    fn memory(&self) -> usize {
        MEMORY
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// Facing +y on its wheels: forward, right, up.
    fn level() -> Mat3A {
        Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z)
    }

    /// `level()` rolled by `deg` about the nose.
    fn rolled(deg: f32) -> Mat3A {
        turned(level(), Vec3A::Y * deg.to_radians())
    }

    fn mirrored(g: &GameSnapshot) -> GameSnapshot {
        let turn = |b: Body| Body {
            pos: as_blue(Team::Orange, b.pos),
            vel: as_blue(Team::Orange, b.vel),
            ang_vel: as_blue(Team::Orange, b.ang_vel),
            rot: Mat3A::from_cols(
                as_blue(Team::Orange, b.rot.x_axis),
                as_blue(Team::Orange, b.rot.y_axis),
                as_blue(Team::Orange, b.rot.z_axis),
            ),
        };
        let mut out = g.clone();
        out.ball = turn(g.ball);
        for c in &mut out.cars {
            c.body = turn(c.body);
            c.team = c.team.opposite();
        }
        out
    }

    /// Car 0's rewards over `games`, each step with the matching `events`, from fresh memory.
    fn run(reward: &RazeFlick, games: &[GameSnapshot], events: &[StepEvents]) -> Vec<f32> {
        let mut memory = vec![0.0; reward.memory()];
        (1..games.len())
            .map(|i| {
                reward.value_remembering(&games[i - 1], &games[i], &events[i - 1], 0, &mut memory)
            })
            .collect()
    }

    /// A Raze, step by step: five steps of dribbling, the pop jump, the dodge rolled onto the side, the
    /// floor rebound upside down, the rear hit, the ball away.
    fn raze() -> (Vec<GameSnapshot>, Vec<StepEvents>) {
        let opponent = (Team::Orange, Vec3A::new(0.0, 4500.0, 17.0));
        let car = |pos: Vec3A, rot: Mat3A, vz: f32, ball: Vec3A| {
            let mut g = game(ball, &[(Team::Blue, pos), opponent]);
            let me = &mut g.cars[0];
            me.body.rot = rot;
            me.body.vel.z = vz;
            me.on_ground = pos.z < 20.0;
            me.has_jumped = !me.on_ground;
            g
        };
        let dribble = car(
            Vec3A::new(0.0, 0.0, 17.0),
            level(),
            0.0,
            Vec3A::new(0.0, 20.0, 150.0),
        );
        let mut games = vec![dribble.clone(); 6];
        games.push(car(
            Vec3A::new(0.0, 0.0, 80.0),
            level(),
            300.0,
            Vec3A::new(0.0, 40.0, 300.0),
        ));
        let mut dodge = car(
            Vec3A::new(0.0, 0.0, 120.0),
            rolled(100.0),
            -100.0,
            Vec3A::new(0.0, 60.0, 300.0),
        );
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        dodge.ball.vel.z = -200.0;
        games.push(dodge.clone());
        let mut rebound = dodge.clone();
        rebound.cars[0].body = Body {
            pos: Vec3A::new(0.0, 0.0, 60.0),
            rot: rolled(180.0),
            vel: Vec3A::new(0.0, 0.0, 250.0),
            ..Body::RESTING
        };
        rebound.ball.pos = Vec3A::new(0.0, -140.0, 70.0);
        games.push(rebound.clone());
        let mut hit = rebound.clone();
        hit.ball.vel = Vec3A::new(0.0, 1500.0, 900.0);
        games.push(hit.clone());
        let mut away = hit;
        away.ball.pos = Vec3A::new(0.0, 0.0, 200.0);
        games.push(away);
        let mut events: Vec<StepEvents> = (0..games.len() - 1).map(|_| quiet(2)).collect();
        for (i, e) in events.iter_mut().enumerate() {
            e.tick = 8 * (i as u64 + 1);
        }
        let strike = games.len() - 3;
        let tick = events[strike].tick;
        events[strike] = touched(events[strike].clone(), 0);
        events[strike].details[0].last_hit_tick = Some(tick);
        events[strike].details[0].last_hit_ball_pos = games[strike + 1].ball.pos;
        (games, events)
    }

    #[test]
    fn a_raze_pays_once_at_its_end() {
        let reward: RazeFlick = serde_json::from_value(serde_json::json!({})).expect("defaults");
        assert_eq!(reward.resolve_time, 0.3);
        let (games, events) = raze();
        let paid = run(&reward, &games, &events);
        let (end, before) = paid.split_last().expect("steps");
        assert!(before.iter().all(|&r| r == 0.0), "{paid:?}");
        assert!(*end > 0.25 && *end <= 1.0, "{paid:?}");
        assert_eq!(
            run(&reward, &games, &events),
            paid,
            "fresh memory starts again"
        );
        // The training credits, once each
        let credited: RazeFlick =
            serde_json::from_value(serde_json::json!({ "arm_credit": 0.1, "rebound_credit": 0.2 }))
                .unwrap();
        let with = run(&credited, &games, &events);
        assert!(
            (with[6] - 0.1).abs() < 1e-6 && (with[7] - 0.2).abs() < 1e-6,
            "{with:?}"
        );
    }

    #[test]
    fn no_dribble_no_rebound_or_a_front_hit_earn_nothing() {
        let reward: RazeFlick = serde_json::from_value(serde_json::json!({})).unwrap();
        let (mut games, events) = raze();
        for g in &mut games[..6] {
            g.ball.pos = Vec3A::new(0.0, 400.0, 93.0); // no ball on the roof
        }
        assert!(run(&reward, &games, &events).iter().all(|&r| r == 0.0));
        let (mut games, events) = raze();
        games[8].cars[0].body.vel.z = -50.0; // no bounce
        assert!(run(&reward, &games, &events).iter().all(|&r| r == 0.0));
        let (mut games, mut events) = raze();
        for g in &mut games[8..] {
            g.ball.pos.y = 140.0; // in front of the car
        }
        events[8].details[0].last_hit_ball_pos.y = 140.0;
        assert!(run(&reward, &games, &events).iter().all(|&r| r == 0.0));
    }

    #[test]
    fn orange_is_paid_as_blue() {
        let reward: RazeFlick = serde_json::from_value(serde_json::json!({})).unwrap();
        let (games, mut events) = raze();
        let blue = run(&reward, &games, &events);
        let orange_games: Vec<GameSnapshot> = games.iter().map(mirrored).collect();
        for e in &mut events {
            for d in &mut e.details {
                d.last_hit_ball_pos = as_blue(Team::Orange, d.last_hit_ball_pos);
            }
        }
        let orange = run(&reward, &orange_games, &events);
        assert!(
            blue.iter().zip(&orange).all(|(b, o)| (b - o).abs() < 1e-4),
            "{blue:?} {orange:?}"
        );
    }
}
