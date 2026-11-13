//! Musty flicks (musty_flick), and the Breezi: a musty out of a tornado spin after a pop (breezi_flick).
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::flick_kit::*;

/// A musty flick, paid once when the ball has left the car: up to flick_weight for goal-directed power,
/// lift, aim and a shot on net, x 1.5 off a wall, the ceiling or a flip reset, half for a roof push;
/// down to -backward_penalty for one sent toward its own goal.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MustyFlick {
    /// The dodge must be a backflip or back-diagonal: the flip's torque y (car frame) at most this (backflip -1, back-diagonal -0.71). Default -0.4.
    #[serde(default = "md::arm_max_torque_y")]
    pub arm_max_torque_y: f32,
    /// At the dodge the nose is at least this far below horizontal, measured toward the ball (90 = straight down). Default 88.
    #[serde(default = "md::arm_min_nose_down_deg")]
    pub arm_min_nose_down_deg: f32,
    /// ... and at most this far (past 140 it is a flat roof ram). Default 140.
    #[serde(default = "md::arm_max_nose_down_deg")]
    pub arm_max_nose_down_deg: f32,
    /// Car centre to ball centre at the dodge, at most (uu). Default 350.
    #[serde(default = "md::arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Ball centre height above the car centre at the dodge, at least (uu). Default -100.
    #[serde(default = "md::arm_min_ball_above")]
    pub arm_min_ball_above: f32,
    /// Roof facing the ball at the dodge: roof direction . direction to the ball, at least. Default -0.35.
    #[serde(default = "md::arm_min_up_dot_ball")]
    pub arm_min_up_dot_ball: f32,
    /// |right . direction to the ball| at the dodge, at most (1 = no limit). Default 1.
    #[serde(default = "md::arm_max_lateral")]
    pub arm_max_lateral: f32,
    /// Seconds after the dodge for the first ball contact. Default 0.7.
    #[serde(default = "md::strike_window")]
    pub strike_window: f32,
    /// The first contact must be at least this high on the car (car-local up from the centre of mass; the roof is at +40). Default 28.
    #[serde(default = "md::top_min_up")]
    pub top_min_up: f32,
    /// The car's hitbox length (uu; the Octane's). Default 120.507.
    #[serde(default = "md::hitbox_length")]
    pub hitbox_length: f32,
    /// The car's hitbox width (uu). Default 86.6994.
    #[serde(default = "md::hitbox_width")]
    pub hitbox_width: f32,
    /// The car's hitbox height (uu). Default 38.6591.
    #[serde(default = "md::hitbox_height")]
    pub hitbox_height: f32,
    /// The hitbox centre's offset from the car's centre of mass, forward (uu). Default 13.87566.
    #[serde(default = "md::hitbox_offset_forward")]
    pub hitbox_offset_forward: f32,
    /// ... to the right (uu). Default 0.
    #[serde(default = "md::hitbox_offset_right")]
    pub hitbox_offset_right: f32,
    /// ... up (uu). Default 20.755.
    #[serde(default = "md::hitbox_offset_up")]
    pub hitbox_offset_up: f32,
    /// The ball's velocity change (uu/s) needed for any credit. Default 300.
    #[serde(default = "md::min_impulse")]
    pub min_impulse: f32,
    /// The velocity change for a full impulse term (uu/s). Default 2000.
    #[serde(default = "md::full_impulse")]
    pub full_impulse: f32,
    /// The ball's upward velocity change where the lift term starts (uu/s). Default 0.
    #[serde(default = "md::min_lift")]
    pub min_lift: f32,
    /// ... and where it is full (uu/s). Default 600.
    #[serde(default = "md::full_lift")]
    pub full_lift: f32,
    /// Goal-directed velocity change for a full power term (uu/s). Default 1000.
    #[serde(default = "md::full_gain")]
    pub full_gain: f32,
    /// Goal-directed launch speed for a full power term (uu/s). Default 2400.
    #[serde(default = "md::full_launch_speed")]
    pub full_launch_speed: f32,
    /// Where the ball leaves the roof, car-local forward: no scoop credit at or behind this (uu). Default -30.
    #[serde(default = "md::scoop_rear_x")]
    pub scoop_rear_x: f32,
    /// ... full scoop credit at or in front of this (uu). Default 40.
    #[serde(default = "md::scoop_front_x")]
    pub scoop_front_x: f32,
    /// Car pitch at the release: full scoop credit within this of level (degrees). Default 30.
    #[serde(default = "md::scoop_level_deg")]
    pub scoop_level_deg: f32,
    /// ... none beyond this (degrees). Default 75.
    #[serde(default = "md::scoop_zero_deg")]
    pub scoop_zero_deg: f32,
    /// What a roof push (no scoop) is worth, as a share of a full scoop. Default 0.5.
    #[serde(default = "md::scoop_floor")]
    pub scoop_floor: f32,
    /// The shot term for any shot on target; the rest grows toward the posts. Default 0.5.
    #[serde(default = "md::shot_on_target")]
    pub shot_on_target: f32,
    /// |x| at the goal line where the corner part of the shot term starts (uu). Default 250.
    #[serde(default = "md::corner_from_x")]
    pub corner_from_x: f32,
    /// ... and where it is full (uu). Default 760.
    #[serde(default = "md::corner_full_x")]
    pub corner_full_x: f32,
    /// Seconds of ball flight projected toward the goal. Default 5.
    #[serde(default = "md::shot_max_time")]
    pub shot_max_time: f32,
    /// Quality mix: weight of goal-directed power. Default 0.35.
    #[serde(default = "md::w_power")]
    pub w_power: f32,
    /// Quality mix: weight of the hit's strength. Default 0.1.
    #[serde(default = "md::w_impulse")]
    pub w_impulse: f32,
    /// Quality mix: weight of the lift. Default 0.2.
    #[serde(default = "md::w_lift")]
    pub w_lift: f32,
    /// Quality mix: weight of the shot on net. Default 0.35.
    #[serde(default = "md::w_shot")]
    pub w_shot: f32,
    /// Launch direction vs the goal (cosine): no credit at or below this. Default 0.
    #[serde(default = "md::aim_cos_zero")]
    pub aim_cos_zero: f32,
    /// ... full credit at or above this. Default 0.85.
    #[serde(default = "md::aim_cos_full")]
    pub aim_cos_full: f32,
    /// Launch elevation (degrees) for ground mustys: no credit at or below this. Default 0.
    #[serde(default = "md::elev_zero_low")]
    pub elev_zero_low: f32,
    /// ... full credit from this. Default 8.
    #[serde(default = "md::elev_full_low")]
    pub elev_full_low: f32,
    /// ... full credit up to this (any musty). Default 45.
    #[serde(default = "md::elev_full_high")]
    pub elev_full_high: f32,
    /// ... none from this (any musty). Default 70.
    #[serde(default = "md::elev_zero_high")]
    pub elev_zero_high: f32,
    /// Launch elevation for ceiling, flip-reset and other air mustys: no credit at or below this. Default -45.
    #[serde(default = "md::aerial_elev_zero_low")]
    pub aerial_elev_zero_low: f32,
    /// ... full credit from this. Default -20.
    #[serde(default = "md::aerial_elev_full_low")]
    pub aerial_elev_full_low: f32,
    /// Share of flick_weight for any successful musty. Default 0.25.
    #[serde(default = "md::base_credit")]
    pub base_credit: f32,
    /// Wall, ceiling, flip-reset and other air mustys earn x (1 + this). Default 0.5.
    #[serde(default = "md::aerial_bonus")]
    pub aerial_bonus: f32,
    /// Training option: extra for a shot on target (the musty set: 0.4). Default 0.
    #[serde(default = "md::on_net_bonus")]
    pub on_net_bonus: f32,
    /// Share of on_net_bonus paid flat; the rest grows with the shot's goal speed (Musty bot: 0.5). Default 1.
    #[serde(default = "md::on_net_power_mix")]
    pub on_net_power_mix: f32,
    /// Training option: extra for a powerful shot on target (Musty bot: 0.25). Default 0.
    #[serde(default = "md::stretch_shot_weight")]
    pub stretch_shot_weight: f32,
    /// Goal-directed speed where the powerful-shot extra starts (uu/s). Default 1600.
    #[serde(default = "md::stretch_shot_from")]
    pub stretch_shot_from: f32,
    /// ... and where it is full (uu/s). Default 3000.
    #[serde(default = "md::stretch_shot_full")]
    pub stretch_shot_full: f32,
    /// Training option: extra for a big goal-directed velocity change (Musty bot: 0.15). Default 0.
    #[serde(default = "md::stretch_gain_weight")]
    pub stretch_gain_weight: f32,
    /// Velocity change where that extra starts (uu/s). Default 500.
    #[serde(default = "md::stretch_gain_from")]
    pub stretch_gain_from: f32,
    /// ... and where it is full (uu/s). Default 1500.
    #[serde(default = "md::stretch_gain_full")]
    pub stretch_gain_full: f32,
    /// Another car near or bumping this one around the attempt cancels it. Default true.
    #[serde(default = "md::cancel_on_car_contact")]
    pub cancel_on_car_contact: bool,
    /// How close another car's hitbox counts as contact (uu). Default 40.
    #[serde(default = "md::car_contact_margin")]
    pub car_contact_margin: f32,
    /// Seconds before the dodge that contact still counts. Default 1.
    #[serde(default = "md::car_contact_lookback")]
    pub car_contact_lookback: f32,
    /// What a full musty pays. Default 1.
    #[serde(default = "md::flick_weight")]
    pub flick_weight: f32,
    /// Training option: paid when a musty arms (the pose and dodge were right). Default 0.
    #[serde(default = "md::arm_credit")]
    pub arm_credit: f32,
    /// Largest penalty for a musty that sends the ball toward its own goal. Default 0.5.
    #[serde(default = "md::backward_penalty")]
    pub backward_penalty: f32,
    /// Horizontal velocity change before that penalty applies (uu/s). Default 500.
    #[serde(default = "md::min_penalty_impulse")]
    pub min_penalty_impulse: f32,
    /// The outcome is judged once the ball is this far from the car (uu) .... Default 300.
    #[serde(default = "md::separation_dist")]
    pub separation_dist: f32,
    /// ... or this many seconds after the first contact. Default 0.6.
    #[serde(default = "md::resolve_time")]
    pub resolve_time: f32,
    /// ... or after this many steps without a touch, the ball no longer riding on the roof. Default 2.
    #[serde(default = "md::release_steps")]
    pub release_steps: u32,
    /// A ball this close above the roof (uu) .... Default 45.
    #[serde(default = "md::ride_hover_gap")]
    pub ride_hover_gap: f32,
    /// ... and leaving it slower than this (uu/s) can still land on it again. Default 500.
    #[serde(default = "md::ride_leave_speed")]
    pub ride_leave_speed: f32,
    /// Outcomes this close to a side or back wall are cancelled (uu). Default 200.
    #[serde(default = "md::wall_margin")]
    pub wall_margin: f32,
    /// Car centre this close to the floor or ceiling (with its wheels down) counts as on it (uu). Default 150.
    #[serde(default = "md::surface_margin")]
    pub surface_margin: f32,
    /// An episode starting mid-jump this close to a surface jumped off it (uu). Default 450.
    #[serde(default = "md::start_jump_margin")]
    pub start_jump_margin: f32,
    /// A flip reset needs the car within this of the ball (uu). Default 350.
    #[serde(default = "md::reset_max_ball_dist")]
    pub reset_max_ball_dist: f32,
    /// An episode starting as if after a flip reset needs the car within this of the ball (uu). Default 600.
    #[serde(default = "md::start_reset_max_ball_dist")]
    pub start_reset_max_ball_dist: f32,
}

/// The Breezi: a musty (judged as musty_flick, with the same settings) after a pop off the roof and a
/// tornado spin, the ball in the air throughout. Pays breezi_scale x the musty's credit x how full the
/// spin was, on the step the musty is judged (plus musty_share x the musty itself).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BreeziFlick {
    /// Share of the musty's own reward paid here too (0: only the Breezi bonus; 1: both). Default 0.
    #[serde(default = "bd::musty_share")]
    pub musty_share: f32,
    /// Bonus = this x spin x roll x yaw x pitch terms x the musty's credit. Default 1.
    #[serde(default = "bd::breezi_scale")]
    pub breezi_scale: f32,
    /// Total turn from the lift-off to the dodge: no bonus at or below this (degrees). Default 200.
    #[serde(default = "bd::spin_zero_deg")]
    pub spin_zero_deg: f32,
    /// ... full from this (degrees). Default 290.
    #[serde(default = "bd::spin_full_deg")]
    pub spin_full_deg: f32,
    /// Turn about the nose: no bonus at or below this (degrees). Default 30.
    #[serde(default = "bd::roll_zero_deg")]
    pub roll_zero_deg: f32,
    /// ... full from this (degrees). Default 80.
    #[serde(default = "bd::roll_full_deg")]
    pub roll_full_deg: f32,
    /// Turn about the roof: no bonus at or below this (degrees). Default 60.
    #[serde(default = "bd::yaw_zero_deg")]
    pub yaw_zero_deg: f32,
    /// ... full from this (degrees). Default 150.
    #[serde(default = "bd::yaw_full_deg")]
    pub yaw_full_deg: f32,
    /// Turn about the car's side: full bonus up to this (degrees). Default 180.
    #[serde(default = "bd::pitch_free_deg")]
    pub pitch_free_deg: f32,
    /// ... none from this (degrees). Default 300.
    #[serde(default = "bd::pitch_zero_deg")]
    pub pitch_zero_deg: f32,
    /// Nose-up turn (over the top): full bonus up to this (degrees). Default 30.
    #[serde(default = "bd::nose_up_free_deg")]
    pub nose_up_free_deg: f32,
    /// ... none from this (degrees). Default 90.
    #[serde(default = "bd::nose_up_zero_deg")]
    pub nose_up_zero_deg: f32,
    /// Roll and yaw must turn the same way (air roll one way, the stick the other). Default true.
    #[serde(default = "bd::tornado_sign")]
    pub tornado_sign: bool,
    /// The ball was under control on the roof this many seconds or less before the lift-off. Default 0.25.
    #[serde(default = "bd::pop_window")]
    pub pop_window: f32,
    /// The musty must be a ground one. Default true.
    #[serde(default = "bd::ground_only")]
    pub ground_only: bool,
    /// Ball centre within its radius + this of an arena surface: on it (uu). Default 8.
    #[serde(default = "bd::ball_rest_margin")]
    pub ball_rest_margin: f32,
    /// ... or within its radius + this (uu) .... Default 80.
    #[serde(default = "bd::bounce_margin")]
    pub bounce_margin: f32,
    /// ... with its velocity changed by more than this beyond gravity and no car touching it: a bounce (uu/s). Default 150.
    #[serde(default = "bd::bounce_dv")]
    pub bounce_dv: f32,
    /// A hitbox corner this close to an arena surface at a step's end: the car touched it (uu). Default 4.
    #[serde(default = "bd::car_clear_margin")]
    pub car_clear_margin: f32,
    /// Paid once more when a paid Breezi's ball goes into the goal this car attacks with no bounce and no
    /// other touch after the dodge (0: off). Default 0.
    #[serde(default = "bd::air_goal_bonus")]
    pub air_goal_bonus: f32,
    /// The dodge must be a backflip or back-diagonal: the flip's torque y (car frame) at most this (backflip -1, back-diagonal -0.71). Default -0.4.
    #[serde(default = "md::arm_max_torque_y")]
    pub arm_max_torque_y: f32,
    /// At the dodge the nose is at least this far below horizontal, measured toward the ball (90 = straight down). Default 88.
    #[serde(default = "md::arm_min_nose_down_deg")]
    pub arm_min_nose_down_deg: f32,
    /// ... and at most this far (past 140 it is a flat roof ram). Default 140.
    #[serde(default = "md::arm_max_nose_down_deg")]
    pub arm_max_nose_down_deg: f32,
    /// Car centre to ball centre at the dodge, at most (uu). Default 350.
    #[serde(default = "md::arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Ball centre height above the car centre at the dodge, at least (uu). Default -100.
    #[serde(default = "md::arm_min_ball_above")]
    pub arm_min_ball_above: f32,
    /// Roof facing the ball at the dodge: roof direction . direction to the ball, at least. Default -0.35.
    #[serde(default = "md::arm_min_up_dot_ball")]
    pub arm_min_up_dot_ball: f32,
    /// |right . direction to the ball| at the dodge, at most (1 = no limit). Default 1.
    #[serde(default = "md::arm_max_lateral")]
    pub arm_max_lateral: f32,
    /// Seconds after the dodge for the first ball contact. Default 0.7.
    #[serde(default = "md::strike_window")]
    pub strike_window: f32,
    /// The first contact must be at least this high on the car (car-local up from the centre of mass; the roof is at +40). Default 28.
    #[serde(default = "md::top_min_up")]
    pub top_min_up: f32,
    /// The car's hitbox length (uu; the Octane's). Default 120.507.
    #[serde(default = "md::hitbox_length")]
    pub hitbox_length: f32,
    /// The car's hitbox width (uu). Default 86.6994.
    #[serde(default = "md::hitbox_width")]
    pub hitbox_width: f32,
    /// The car's hitbox height (uu). Default 38.6591.
    #[serde(default = "md::hitbox_height")]
    pub hitbox_height: f32,
    /// The hitbox centre's offset from the car's centre of mass, forward (uu). Default 13.87566.
    #[serde(default = "md::hitbox_offset_forward")]
    pub hitbox_offset_forward: f32,
    /// ... to the right (uu). Default 0.
    #[serde(default = "md::hitbox_offset_right")]
    pub hitbox_offset_right: f32,
    /// ... up (uu). Default 20.755.
    #[serde(default = "md::hitbox_offset_up")]
    pub hitbox_offset_up: f32,
    /// The ball's velocity change (uu/s) needed for any credit. Default 300.
    #[serde(default = "md::min_impulse")]
    pub min_impulse: f32,
    /// The velocity change for a full impulse term (uu/s). Default 2000.
    #[serde(default = "md::full_impulse")]
    pub full_impulse: f32,
    /// The ball's upward velocity change where the lift term starts (uu/s). Default 0.
    #[serde(default = "md::min_lift")]
    pub min_lift: f32,
    /// ... and where it is full (uu/s). Default 600.
    #[serde(default = "md::full_lift")]
    pub full_lift: f32,
    /// Goal-directed velocity change for a full power term (uu/s). Default 1000.
    #[serde(default = "md::full_gain")]
    pub full_gain: f32,
    /// Goal-directed launch speed for a full power term (uu/s). Default 2400.
    #[serde(default = "md::full_launch_speed")]
    pub full_launch_speed: f32,
    /// Where the ball leaves the roof, car-local forward: no scoop credit at or behind this (uu). Default -30.
    #[serde(default = "md::scoop_rear_x")]
    pub scoop_rear_x: f32,
    /// ... full scoop credit at or in front of this (uu). Default 40.
    #[serde(default = "md::scoop_front_x")]
    pub scoop_front_x: f32,
    /// Car pitch at the release: full scoop credit within this of level (degrees). Default 30.
    #[serde(default = "md::scoop_level_deg")]
    pub scoop_level_deg: f32,
    /// ... none beyond this (degrees). Default 75.
    #[serde(default = "md::scoop_zero_deg")]
    pub scoop_zero_deg: f32,
    /// What a roof push (no scoop) is worth, as a share of a full scoop. Default 0.5.
    #[serde(default = "md::scoop_floor")]
    pub scoop_floor: f32,
    /// The shot term for any shot on target; the rest grows toward the posts. Default 0.5.
    #[serde(default = "md::shot_on_target")]
    pub shot_on_target: f32,
    /// |x| at the goal line where the corner part of the shot term starts (uu). Default 250.
    #[serde(default = "md::corner_from_x")]
    pub corner_from_x: f32,
    /// ... and where it is full (uu). Default 760.
    #[serde(default = "md::corner_full_x")]
    pub corner_full_x: f32,
    /// Seconds of ball flight projected toward the goal. Default 5.
    #[serde(default = "md::shot_max_time")]
    pub shot_max_time: f32,
    /// Quality mix: weight of goal-directed power. Default 0.35.
    #[serde(default = "md::w_power")]
    pub w_power: f32,
    /// Quality mix: weight of the hit's strength. Default 0.1.
    #[serde(default = "md::w_impulse")]
    pub w_impulse: f32,
    /// Quality mix: weight of the lift. Default 0.2.
    #[serde(default = "md::w_lift")]
    pub w_lift: f32,
    /// Quality mix: weight of the shot on net. Default 0.35.
    #[serde(default = "md::w_shot")]
    pub w_shot: f32,
    /// Launch direction vs the goal (cosine): no credit at or below this. Default 0.
    #[serde(default = "md::aim_cos_zero")]
    pub aim_cos_zero: f32,
    /// ... full credit at or above this. Default 0.85.
    #[serde(default = "md::aim_cos_full")]
    pub aim_cos_full: f32,
    /// Launch elevation (degrees) for ground mustys: no credit at or below this. Default 0.
    #[serde(default = "md::elev_zero_low")]
    pub elev_zero_low: f32,
    /// ... full credit from this. Default 8.
    #[serde(default = "md::elev_full_low")]
    pub elev_full_low: f32,
    /// ... full credit up to this (any musty). Default 45.
    #[serde(default = "md::elev_full_high")]
    pub elev_full_high: f32,
    /// ... none from this (any musty). Default 70.
    #[serde(default = "md::elev_zero_high")]
    pub elev_zero_high: f32,
    /// Launch elevation for ceiling, flip-reset and other air mustys: no credit at or below this. Default -45.
    #[serde(default = "md::aerial_elev_zero_low")]
    pub aerial_elev_zero_low: f32,
    /// ... full credit from this. Default -20.
    #[serde(default = "md::aerial_elev_full_low")]
    pub aerial_elev_full_low: f32,
    /// Share of flick_weight for any successful musty. Default 0.25.
    #[serde(default = "md::base_credit")]
    pub base_credit: f32,
    /// Wall, ceiling, flip-reset and other air mustys earn x (1 + this). Default 0.5.
    #[serde(default = "md::aerial_bonus")]
    pub aerial_bonus: f32,
    /// Training option: extra for a shot on target (the musty set: 0.4). Default 0.
    #[serde(default = "md::on_net_bonus")]
    pub on_net_bonus: f32,
    /// Share of on_net_bonus paid flat; the rest grows with the shot's goal speed (Musty bot: 0.5). Default 1.
    #[serde(default = "md::on_net_power_mix")]
    pub on_net_power_mix: f32,
    /// Training option: extra for a powerful shot on target (Musty bot: 0.25). Default 0.
    #[serde(default = "md::stretch_shot_weight")]
    pub stretch_shot_weight: f32,
    /// Goal-directed speed where the powerful-shot extra starts (uu/s). Default 1600.
    #[serde(default = "md::stretch_shot_from")]
    pub stretch_shot_from: f32,
    /// ... and where it is full (uu/s). Default 3000.
    #[serde(default = "md::stretch_shot_full")]
    pub stretch_shot_full: f32,
    /// Training option: extra for a big goal-directed velocity change (Musty bot: 0.15). Default 0.
    #[serde(default = "md::stretch_gain_weight")]
    pub stretch_gain_weight: f32,
    /// Velocity change where that extra starts (uu/s). Default 500.
    #[serde(default = "md::stretch_gain_from")]
    pub stretch_gain_from: f32,
    /// ... and where it is full (uu/s). Default 1500.
    #[serde(default = "md::stretch_gain_full")]
    pub stretch_gain_full: f32,
    /// Another car near or bumping this one around the attempt cancels it. Default true.
    #[serde(default = "md::cancel_on_car_contact")]
    pub cancel_on_car_contact: bool,
    /// How close another car's hitbox counts as contact (uu). Default 40.
    #[serde(default = "md::car_contact_margin")]
    pub car_contact_margin: f32,
    /// Seconds before the dodge that contact still counts. Default 1.
    #[serde(default = "md::car_contact_lookback")]
    pub car_contact_lookback: f32,
    /// What a full musty pays. Default 1.
    #[serde(default = "md::flick_weight")]
    pub flick_weight: f32,
    /// Training option: paid when a musty arms (the pose and dodge were right). Default 0.
    #[serde(default = "md::arm_credit")]
    pub arm_credit: f32,
    /// Largest penalty for a musty that sends the ball toward its own goal. Default 0.5.
    #[serde(default = "md::backward_penalty")]
    pub backward_penalty: f32,
    /// Horizontal velocity change before that penalty applies (uu/s). Default 500.
    #[serde(default = "md::min_penalty_impulse")]
    pub min_penalty_impulse: f32,
    /// The outcome is judged once the ball is this far from the car (uu) .... Default 300.
    #[serde(default = "md::separation_dist")]
    pub separation_dist: f32,
    /// ... or this many seconds after the first contact. Default 0.6.
    #[serde(default = "md::resolve_time")]
    pub resolve_time: f32,
    /// ... or after this many steps without a touch, the ball no longer riding on the roof. Default 2.
    #[serde(default = "md::release_steps")]
    pub release_steps: u32,
    /// A ball this close above the roof (uu) .... Default 45.
    #[serde(default = "md::ride_hover_gap")]
    pub ride_hover_gap: f32,
    /// ... and leaving it slower than this (uu/s) can still land on it again. Default 500.
    #[serde(default = "md::ride_leave_speed")]
    pub ride_leave_speed: f32,
    /// Outcomes this close to a side or back wall are cancelled (uu). Default 200.
    #[serde(default = "md::wall_margin")]
    pub wall_margin: f32,
    /// Car centre this close to the floor or ceiling (with its wheels down) counts as on it (uu). Default 150.
    #[serde(default = "md::surface_margin")]
    pub surface_margin: f32,
    /// An episode starting mid-jump this close to a surface jumped off it (uu). Default 450.
    #[serde(default = "md::start_jump_margin")]
    pub start_jump_margin: f32,
    /// A flip reset needs the car within this of the ball (uu). Default 350.
    #[serde(default = "md::reset_max_ball_dist")]
    pub reset_max_ball_dist: f32,
    /// An episode starting as if after a flip reset needs the car within this of the ball (uu). Default 600.
    #[serde(default = "md::start_reset_max_ball_dist")]
    pub start_reset_max_ball_dist: f32,
    /// Ball under control on the roof: its centre at least this far above the car along its roof direction (uu). Default 105.
    #[serde(default = "md::dribble_min_up")]
    pub dribble_min_up: f32,
    /// ... and at most this far (uu). Default 185.
    #[serde(default = "md::dribble_max_up")]
    pub dribble_max_up: f32,
    /// ... best this far forward of the car's centre (uu). Default 20.
    #[serde(default = "md::dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// ... no control this far or further from that spot, sideways (uu). Default 90.
    #[serde(default = "md::dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// ... nor at this speed relative to the car (uu/s). Default 600.
    #[serde(default = "md::dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality (0-1) that counts as the ball under control. Default 0.25.
    #[serde(default = "md::control_threshold")]
    pub control_threshold: f32,
}

/// The musty's defaults (shared by MustyFlick and BreeziFlick).
mod md {
    pub fn arm_max_torque_y() -> f32 {
        -0.4
    }
    pub fn arm_min_nose_down_deg() -> f32 {
        88.0
    }
    pub fn arm_max_nose_down_deg() -> f32 {
        140.0
    }
    pub fn arm_max_ball_dist() -> f32 {
        350.0
    }
    pub fn arm_min_ball_above() -> f32 {
        -100.0
    }
    pub fn arm_min_up_dot_ball() -> f32 {
        -0.35
    }
    pub fn arm_max_lateral() -> f32 {
        1.0
    }
    pub fn strike_window() -> f32 {
        0.7
    }
    pub fn top_min_up() -> f32 {
        28.0
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
    pub fn min_impulse() -> f32 {
        300.0
    }
    pub fn full_impulse() -> f32 {
        2000.0
    }
    pub fn min_lift() -> f32 {
        0.0
    }
    pub fn full_lift() -> f32 {
        600.0
    }
    pub fn full_gain() -> f32 {
        1000.0
    }
    pub fn full_launch_speed() -> f32 {
        2400.0
    }
    pub fn scoop_rear_x() -> f32 {
        -30.0
    }
    pub fn scoop_front_x() -> f32 {
        40.0
    }
    pub fn scoop_level_deg() -> f32 {
        30.0
    }
    pub fn scoop_zero_deg() -> f32 {
        75.0
    }
    pub fn scoop_floor() -> f32 {
        0.5
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
        0.35
    }
    pub fn w_impulse() -> f32 {
        0.1
    }
    pub fn w_lift() -> f32 {
        0.2
    }
    pub fn w_shot() -> f32 {
        0.35
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
    pub fn aerial_elev_zero_low() -> f32 {
        -45.0
    }
    pub fn aerial_elev_full_low() -> f32 {
        -20.0
    }
    pub fn base_credit() -> f32 {
        0.25
    }
    pub fn aerial_bonus() -> f32 {
        0.5
    }
    pub fn on_net_bonus() -> f32 {
        0.0
    }
    pub fn on_net_power_mix() -> f32 {
        1.0
    }
    pub fn stretch_shot_weight() -> f32 {
        0.0
    }
    pub fn stretch_shot_from() -> f32 {
        1600.0
    }
    pub fn stretch_shot_full() -> f32 {
        3000.0
    }
    pub fn stretch_gain_weight() -> f32 {
        0.0
    }
    pub fn stretch_gain_from() -> f32 {
        500.0
    }
    pub fn stretch_gain_full() -> f32 {
        1500.0
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
        0.6
    }
    pub fn release_steps() -> u32 {
        2
    }
    pub fn ride_hover_gap() -> f32 {
        45.0
    }
    pub fn ride_leave_speed() -> f32 {
        500.0
    }
    pub fn wall_margin() -> f32 {
        200.0
    }
    pub fn surface_margin() -> f32 {
        150.0
    }
    pub fn start_jump_margin() -> f32 {
        450.0
    }
    pub fn reset_max_ball_dist() -> f32 {
        350.0
    }
    pub fn start_reset_max_ball_dist() -> f32 {
        600.0
    }
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
}

/// BreeziFlick's own defaults.
mod bd {
    pub fn musty_share() -> f32 {
        0.0
    }
    pub fn breezi_scale() -> f32 {
        1.0
    }
    pub fn spin_zero_deg() -> f32 {
        200.0
    }
    pub fn spin_full_deg() -> f32 {
        290.0
    }
    pub fn roll_zero_deg() -> f32 {
        30.0
    }
    pub fn roll_full_deg() -> f32 {
        80.0
    }
    pub fn yaw_zero_deg() -> f32 {
        60.0
    }
    pub fn yaw_full_deg() -> f32 {
        150.0
    }
    pub fn pitch_free_deg() -> f32 {
        180.0
    }
    pub fn pitch_zero_deg() -> f32 {
        300.0
    }
    pub fn nose_up_free_deg() -> f32 {
        30.0
    }
    pub fn nose_up_zero_deg() -> f32 {
        90.0
    }
    pub fn tornado_sign() -> bool {
        true
    }
    pub fn pop_window() -> f32 {
        0.25
    }
    pub fn ground_only() -> bool {
        true
    }
    pub fn ball_rest_margin() -> f32 {
        8.0
    }
    pub fn bounce_margin() -> f32 {
        80.0
    }
    pub fn bounce_dv() -> f32 {
        150.0
    }
    pub fn car_clear_margin() -> f32 {
        4.0
    }
    pub fn air_goal_bonus() -> f32 {
        0.0
    }
}

/// Where the car's flip came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Context {
    Ground,
    Wall,
    Ceiling,
    FlipReset,
    AerialOther,
}

impl Context {
    fn of(v: f32) -> Self {
        match v as u8 {
            0 => Self::Ground,
            1 => Self::Wall,
            2 => Self::Ceiling,
            3 => Self::FlipReset,
            _ => Self::AerialOther,
        }
    }

    fn number(self) -> f32 {
        f32::from(self as u8)
    }

    /// Earns the aerial bonus.
    fn aerial(self) -> bool {
        self != Self::Ground
    }

    /// Judged with the downward elevation band.
    fn in_air(self) -> bool {
        matches!(self, Self::Ceiling | Self::FlipReset | Self::AerialOther)
    }
}

/// How a musty attempt ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ending {
    Launch,
    NoStrike,
    Landed,
    NotTop,
    OtherTouch,
    Wall,
    Demo,
    CarContact,
}

/// A musty attempt that ended this step.
#[derive(Clone, Copy, Debug)]
struct Attempt {
    ending: Ending,
    context: Context,
    reward: f32,
    /// Ticks since it armed (its dodge), and since its first contact if it had one.
    arm_age: f32,
    strike_age: Option<f32>,
}

// MustyFlick's memory, per car.
const INIT: usize = 0;
const SURFACE: usize = 1;
const RESET_SINCE_SURFACE: usize = 2;
const STAGE: usize = 3;
const CONTEXT: usize = 4;
const ARM_AGE: usize = 5;
const STRUCK: usize = 6;
const STRIKE_AGE: usize = 7;
const CAR_CONTACT: usize = 8;
const RELEASE_X: usize = 9;
const RELEASE_PITCH: usize = 10;
const BALL_VEL0: usize = 11;
const LAUNCH_VEL: usize = 14;
const TOUCH_POS: usize = 17;
const STEPS_WITHOUT_TOUCH: usize = 20;
const ANY_CAR_CONTACT: usize = 21;
const CAR_CONTACT_AGE: usize = 22;
const MUSTY_MEMORY: usize = 23;

const IDLE: f32 = 0.0;
const ARMED: f32 = 1.0;
const STRIKING: f32 = 2.0;

fn get3(m: &[f32], at: usize) -> Vec3A {
    Vec3A::new(m[at], m[at + 1], m[at + 2])
}

fn set3(m: &mut [f32], at: usize, v: Vec3A) {
    m[at..at + 3].copy_from_slice(&v.to_array());
}

/// 1 up to `full`, 0 from `zero` (the headers' `clamp((zero - v) / (zero - full))`).
fn falling(v: f32, full: f32, zero: f32) -> f32 {
    if zero == full {
        return if v < zero { 1.0 } else { 0.0 };
    }
    clamp01((zero - v) / (zero - full))
}

impl MustyFlick {
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

    fn arm_test(&self) -> ArmTest {
        ArmTest {
            max_torque_y: self.arm_max_torque_y,
            min_nose_down_deg: self.arm_min_nose_down_deg,
            max_nose_down_deg: self.arm_max_nose_down_deg,
            max_ball_dist: self.arm_max_ball_dist,
            min_ball_above: self.arm_min_ball_above,
            min_up_dot_ball: self.arm_min_up_dot_ball,
            max_lateral: self.arm_max_lateral,
        }
    }

    /// The arena surface the car's wheels are on (3+ down and close to it), or AerialOther.
    fn surface_of(&self, car: &CarSnapshot) -> Context {
        if !car.on_ground {
            return Context::AerialOther;
        }
        self.surface_near(car.body.pos, self.surface_margin)
    }

    fn surface_near(&self, p: Vec3A, m: f32) -> Context {
        if p.z < m {
            Context::Ground
        } else if p.z > CEILING_Z - m {
            Context::Ceiling
        } else if p.x.abs() > SIDE_WALL_X - m || p.y.abs() > BACK_WALL_Y - m {
            Context::Wall
        } else {
            Context::AerialOther
        }
    }

    /// The surface to credit for a car as an episode starts: the one it is on, or, during its first jump,
    /// the one it is still within start_jump_margin of.
    fn start_surface_of(&self, car: &CarSnapshot) -> Context {
        let on = self.surface_of(car);
        if on != Context::AerialOther
            || car.on_ground
            || !car.has_jumped
            || car.has_flipped
            || car.has_double_jumped
        {
            return on;
        }
        self.surface_near(car.body.pos, self.start_jump_margin)
    }

    /// A car starting in the air with its flip and no jump used, away from the surfaces and near the ball:
    /// its flip came back off the ball.
    fn starts_after_flip_reset(&self, car: &CarSnapshot, ball: Vec3A) -> bool {
        self.surface_of(car) == Context::AerialOther
            && !car.has_jumped
            && !car.has_flipped
            && !car.has_double_jumped
            && !near_any_surface(car.body.pos, 120.0)
            && car.body.pos.distance(ball) <= self.start_reset_max_ball_dist
    }

    /// Next to a side or back wall (outside the goal mouth).
    fn near_wall(&self, p: Vec3A) -> bool {
        let m = self.wall_margin + CV_BALL_RADIUS;
        p.x.abs() > SIDE_WALL_X - m
            || (p.y.abs() > BACK_WALL_Y - m && (p.x.abs() > GOAL_HALF_WIDTH || p.z > GOAL_HEIGHT))
    }

    /// The gap between the ball and the car's roof now (f32::MAX when the ball is not over the roof), and
    /// how fast the ball moves away from the roof's nearest point.
    fn roof_gap(&self, body: &Body, ball: &Body) -> (f32, f32) {
        let hitbox = self.hitbox();
        let local = body.rot.transpose() * (ball.pos - body.pos) - hitbox.offset;
        let half = hitbox.size * 0.5;
        let closest = local.max(-half).min(half);
        if closest.z + hitbox.offset.z < self.top_min_up {
            return (f32::MAX, 0.0);
        }
        let n = local - closest;
        let len = n.length();
        let n = if len > 1e-3 { n / len } else { Vec3A::Z };
        let point_vel = body.vel + body.ang_vel.cross(body.rot * (closest + hitbox.offset));
        let away = (ball.vel - point_vel).dot(body.rot * n);
        (len - CV_BALL_RADIUS, away)
    }

    /// The ball just after a touch this step: its velocity with the gravity since undone, its position.
    fn record_touch(m: &mut [f32], detail: &CarDetail, ball: &Body, now: u64) {
        set3(m, LAUNCH_VEL, launch_velocity(detail, ball, now));
        set3(m, TOUCH_POS, ball.pos);
    }

    /// What a launched musty earns, judged where the ball left the car.
    fn outcome(&self, m: &[f32], cur: &GameSnapshot, team: Team, context: Context) -> f32 {
        let v1 = get3(m, LAUNCH_VEL);
        let dv = v1 - get3(m, BALL_VEL0);
        let goal_dir = dir2d(goal_aim_point(team) - get3(m, TOUCH_POS));
        let flat_v1 = Vec3A::new(v1.x, v1.y, 0.0);
        let flat_dv = Vec3A::new(dv.x, dv.y, 0.0);
        let horiz_speed = flat_v1.length();
        let impulse = dv.length();
        let lift = dv.z;
        let gain = flat_dv.dot(goal_dir);
        let final_goal_speed = flat_v1.dot(goal_dir);
        let elevation = degrees(v1.z.atan2(horiz_speed));
        let aim_cos = if horiz_speed > 1.0 {
            final_goal_speed / horiz_speed
        } else {
            0.0
        };
        let power = (ramp(gain, 0.0, self.full_gain)
            * clamp01(div(final_goal_speed, self.full_launch_speed)))
        .sqrt();
        let impulse_term = ramp(impulse, self.min_impulse, self.full_impulse);
        let lift_term = ramp(lift, self.min_lift, self.full_lift);
        let scoop = ramp(m[RELEASE_X], self.scoop_rear_x, self.scoop_front_x)
            * (1.0
                - ramp(
                    m[RELEASE_PITCH].abs(),
                    self.scoop_level_deg,
                    self.scoop_zero_deg,
                ));
        // The shot from the ball as it is now
        let shot = project_to_goal_line(cur.ball.pos, cur.ball.vel, team, self.shot_max_time);
        let shot_term = if shot.on_target {
            self.shot_on_target
                + (1.0 - self.shot_on_target)
                    * ramp(shot.x.abs(), self.corner_from_x, self.corner_full_x)
        } else {
            0.0
        };
        let quality = self.w_power * power
            + self.w_impulse * impulse_term
            + self.w_lift * lift_term
            + self.w_shot * shot_term;
        let aim = ramp(aim_cos, self.aim_cos_zero, self.aim_cos_full);
        let (zero_low, full_low) = if context.in_air() {
            (self.aerial_elev_zero_low, self.aerial_elev_full_low)
        } else {
            (self.elev_zero_low, self.elev_full_low)
        };
        let elev = if elevation < full_low {
            ramp(elevation, zero_low, full_low)
        } else {
            falling(elevation, self.elev_full_high, self.elev_zero_high)
        };
        let context_mult = if context.aerial() {
            1.0 + self.aerial_bonus
        } else {
            1.0
        };
        let scoop_mult = self.scoop_floor + (1.0 - self.scoop_floor) * scoop;
        let mut reward = 0.0;
        if impulse >= self.min_impulse && final_goal_speed > 0.0 && gain > -self.min_impulse {
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
                * (self.stretch_gain_weight
                    * ramp(gain, self.stretch_gain_from, self.stretch_gain_full)
                    + shot_part);
            // Context and scoop scale everything a musty earns
            reward *= context_mult * scoop_mult;
        }
        if final_goal_speed < 0.0 && flat_dv.length() >= self.min_penalty_impulse && !shot.on_target
        {
            reward -= self.backward_penalty * clamp01(div(-final_goal_speed, self.full_gain));
        }
        reward
    }

    /// Ends the attempt in progress: what it pays (only a launch pays) and its record.
    fn finish(&self, m: &mut [f32], cur: &GameSnapshot, car: usize, ending: Ending) -> Attempt {
        let context = Context::of(m[CONTEXT]);
        let mut reward = 0.0;
        if matches!(ending, Ending::Launch | Ending::CarContact) {
            reward = self.outcome(m, cur, cur.cars[car].team, context);
            if ending == Ending::CarContact {
                reward = 0.0; // judged, but not credited
            }
        }
        let attempt = Attempt {
            ending,
            context,
            reward,
            arm_age: m[ARM_AGE],
            strike_age: (m[STRUCK] != 0.0).then_some(m[STRIKE_AGE]),
        };
        m[STAGE] = IDLE;
        m[CONTEXT..=RELEASE_PITCH].fill(0.0);
        attempt
    }

    /// One step of the musty detector for `car`: what it pays now, and the attempt that ended, if any.
    fn step(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> (f32, Option<Attempt>) {
        let before = &prev.cars[car];
        if m[INIT] == 0.0 {
            // The episode's start: credit the surface the car starts on (or jumped off)
            m[INIT] = 1.0;
            m[SURFACE] = self.start_surface_of(before).number();
            m[RESET_SINCE_SURFACE] = flag(self.starts_after_flip_reset(before, prev.ball.pos));
        }
        let ticks = step_ticks(events);
        for age in [ARM_AGE, STRIKE_AGE, CAR_CONTACT_AGE] {
            m[age] += ticks;
        }
        let me = &cur.cars[car];
        let detail = events.detail(car);
        let now = events.tick;
        // A goal ends the episode: the one last step Nitro lets a reward recognise
        let last = cur.goal_scored.is_some();
        if me.is_demoed {
            let ended = (m[STAGE] != IDLE).then(|| self.finish(m, cur, car, Ending::Demo));
            return (0.0, ended);
        }
        let surface = self.surface_of(me);
        if surface != Context::AerialOther {
            m[SURFACE] = surface.number();
            m[RESET_SINCE_SURFACE] = 0.0;
        }
        let other_touch = events
            .cars
            .iter()
            .enumerate()
            .any(|(other, e)| other != car && e.touched_ball);
        if m[STAGE] != IDLE && other_touch {
            return (0.0, Some(self.finish(m, cur, car, Ending::OtherTouch)));
        }
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
        // A flip reset off the ball: no flip left, then the jump and flip back away from every surface
        let had_no_flip = !before.on_ground && !before.can_flip;
        let reset_now = !me.has_jumped && !me.has_flipped && !me.has_double_jumped;
        if had_no_flip
            && reset_now
            && !near_any_surface(me.body.pos, 120.0)
            && !near_any_surface(before.body.pos, 120.0)
            && me.body.pos.distance(cur.ball.pos) <= self.reset_max_ball_dist
        {
            m[RESET_SINCE_SURFACE] = 1.0;
        }

        if m[STAGE] == IDLE {
            let dodge_started = me.has_flipped && !before.has_flipped;
            if !(dodge_started && arm_pose_ok(self.arm_test(), me, &detail, touched, cur.ball.pos))
            {
                return (0.0, None);
            }
            m[STAGE] = ARMED;
            m[CONTEXT..=RELEASE_PITCH].fill(0.0);
            m[CONTEXT] = if m[RESET_SINCE_SURFACE] != 0.0 {
                Context::FlipReset.number()
            } else {
                m[SURFACE]
            };
            if m[ANY_CAR_CONTACT] != 0.0
                && m[CAR_CONTACT_AGE] <= secs_to_ticks(self.car_contact_lookback)
            {
                m[CAR_CONTACT] = 1.0;
            }
            return (self.arm_credit, None);
        }

        if m[STAGE] == ARMED {
            // Landing on (or bumping) an arena surface before the ball ends it
            let landed = self.surface_of(me) != Context::AerialOther
                || (detail.world_contact_normal.is_some()
                    && near_any_surface(me.body.pos, self.surface_margin));
            if landed {
                return (0.0, Some(self.finish(m, cur, car, Ending::Landed)));
            }
            if touched {
                let (local, pitch) =
                    hit_contact_local(&me.body, &detail, &cur.ball, now, self.hitbox());
                m[RELEASE_X] = local.x;
                m[RELEASE_PITCH] = pitch;
                m[STEPS_WITHOUT_TOUCH] = 0.0;
                if local.z < self.top_min_up {
                    return (0.0, Some(self.finish(m, cur, car, Ending::NotTop)));
                }
                m[STAGE] = STRIKING;
                m[STRUCK] = 1.0;
                m[STRIKE_AGE] = 0.0;
                set3(m, BALL_VEL0, prev.ball.vel);
                Self::record_touch(m, &detail, &cur.ball, now);
            } else if m[ARM_AGE] >= secs_to_ticks(self.strike_window) || last {
                return (0.0, Some(self.finish(m, cur, car, Ending::NoStrike)));
            }
            if !last {
                return (0.0, None);
            }
        }

        // Struck: follow the ball along the roof until it has left the car
        let strike_step = m[STRIKE_AGE] == 0.0;
        if touched && !strike_step {
            Self::record_touch(m, &detail, &cur.ball, now);
            let (local, pitch) =
                hit_contact_local(&me.body, &detail, &cur.ball, now, self.hitbox());
            if local.z >= self.top_min_up {
                m[RELEASE_X] = local.x;
                m[RELEASE_PITCH] = pitch;
            }
        }
        if !strike_step {
            m[STEPS_WITHOUT_TOUCH] = if touched {
                0.0
            } else {
                m[STEPS_WITHOUT_TOUCH] + 1.0
            };
        }
        let (gap, away) = self.roof_gap(&me.body, &cur.ball);
        let on_roof = gap <= self.ride_hover_gap
            && away < self.ride_leave_speed
            && !near_any_surface(cur.ball.pos, CV_BALL_RADIUS + 10.0);
        let contact_over = m[STEPS_WITHOUT_TOUCH] >= self.release_steps as f32 && !on_roof;
        let separated = me.body.pos.distance(cur.ball.pos) >= self.separation_dist;
        let time_up = m[STRIKE_AGE] >= secs_to_ticks(self.resolve_time);
        if !(contact_over || separated || time_up || last) {
            return (0.0, None);
        }
        if self.near_wall(get3(m, TOUCH_POS)) && !last {
            return (0.0, Some(self.finish(m, cur, car, Ending::Wall)));
        }
        let ending = if self.cancel_on_car_contact && m[CAR_CONTACT] != 0.0 {
            Ending::CarContact
        } else {
            Ending::Launch
        };
        let attempt = self.finish(m, cur, car, ending);
        (attempt.reward, Some(attempt))
    }
}

impl Reward for MustyFlick {
    fn memory(&self) -> usize {
        MUSTY_MEMORY
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        self.step(prev, cur, events, car, memory).0
    }
}

// BreeziFlick's memory, per car, before the musty's own.
const B_INIT: usize = 0;
const B_AIRBORNE: usize = 1;
const B_EVER_CONTROL: usize = 2;
const B_CONTROL_AGE: usize = 3;
const B_OPEN: usize = 4;
const B_OPEN_FROM_LIFTOFF: usize = 5;
const B_DODGE_AGE: usize = 6;
/// A paid Breezi's ball still flying (no bounce or other touch since the dodge): its goal pays the bonus.
const B_FLIGHT: usize = 7;
/// The turn since the lift-off, and the same as it stood at the dodge.
const B_WIN: usize = 8;
const B_AT_DODGE: usize = B_WIN + SPIN_STATE;
const B_MUSTY: usize = B_AT_DODGE + SPIN_STATE;

// A turn and what happened since the lift-off: degrees in total and about the car's forward, up and
// right axes, the nose-up part, whether it followed a pop, and the ages of the first steps the ball was on
// a surface, another car touched it, the car came close to a surface.
const S_SPIN: usize = 0;
const S_ROLL: usize = 1;
const S_YAW: usize = 2;
const S_PITCH: usize = 3;
const S_NOSE_UP: usize = 4;
const S_POPPED: usize = 5;
const S_BALL_SURFACE: usize = 6;
const S_OTHER_TOUCH: usize = 8;
const S_CAR_SURFACE: usize = 10;
const SPIN_STATE: usize = 12;

/// Notes the first step `what` happened (a flag, then its age).
fn mark_first(s: &mut [f32], at: usize, what: bool) {
    if what && s[at] == 0.0 {
        s[at] = 1.0;
        s[at + 1] = 0.0;
    }
}

/// Adds a turn `w` (world axis x radians) to `s`, split along the axes of the pose it started from.
fn add_turn(s: &mut [f32], w: Vec3A, from: Mat3A) {
    let pitch = degrees(w.dot(from.y_axis));
    s[S_SPIN] += degrees(w.length());
    s[S_ROLL] += degrees(w.dot(from.x_axis));
    s[S_YAW] += degrees(w.dot(from.z_axis));
    s[S_PITCH] += pitch;
    s[S_NOSE_UP] += (-pitch).max(0.0); // a nose-down pitch is positive about the right axis
}

/// Within `m` of an arena surface, the goal mouths open (the back wall counts only outside them).
fn near_arena_surface(p: Vec3A, m: f32) -> bool {
    let mouth = p.x.abs() < GOAL_HALF_WIDTH - m && p.z < GOAL_HEIGHT - m;
    p.z < m
        || p.z > CEILING_Z - m
        || p.x.abs() > SIDE_WALL_X - m
        || (!mouth && p.y.abs() > BACK_WALL_Y - m)
}

/// How close the car's hitbox comes to the floor, ceiling, side walls and back walls outside the goal
/// mouths: the nearest of its 8 corners.
fn hitbox_clearance(body: &Body, hitbox: Hitbox) -> f32 {
    let (h, o) = (hitbox.size * 0.5, hitbox.offset);
    let pick = |bit: bool, v: f32| if bit { v } else { -v };
    (0..8)
        .map(|i| {
            let l = Vec3A::new(
                o.x + pick(i & 1 != 0, h.x),
                o.y + pick(i & 2 != 0, h.y),
                o.z + pick(i & 4 != 0, h.z),
            );
            let c = body.pos + body.rot * l;
            let mouth = c.x.abs() < GOAL_HALF_WIDTH && c.z < GOAL_HEIGHT;
            let back = if mouth {
                f32::MAX
            } else {
                BACK_WALL_Y - c.y.abs()
            };
            c.z.min(CEILING_Z - c.z)
                .min(SIDE_WALL_X - c.x.abs())
                .min(back)
        })
        .fold(f32::MAX, f32::min)
}

impl BreeziFlick {
    /// The musty that finishes a Breezi, with this reward's musty settings.
    fn musty(&self) -> MustyFlick {
        MustyFlick {
            arm_max_torque_y: self.arm_max_torque_y,
            arm_min_nose_down_deg: self.arm_min_nose_down_deg,
            arm_max_nose_down_deg: self.arm_max_nose_down_deg,
            arm_max_ball_dist: self.arm_max_ball_dist,
            arm_min_ball_above: self.arm_min_ball_above,
            arm_min_up_dot_ball: self.arm_min_up_dot_ball,
            arm_max_lateral: self.arm_max_lateral,
            strike_window: self.strike_window,
            top_min_up: self.top_min_up,
            hitbox_length: self.hitbox_length,
            hitbox_width: self.hitbox_width,
            hitbox_height: self.hitbox_height,
            hitbox_offset_forward: self.hitbox_offset_forward,
            hitbox_offset_right: self.hitbox_offset_right,
            hitbox_offset_up: self.hitbox_offset_up,
            min_impulse: self.min_impulse,
            full_impulse: self.full_impulse,
            min_lift: self.min_lift,
            full_lift: self.full_lift,
            full_gain: self.full_gain,
            full_launch_speed: self.full_launch_speed,
            scoop_rear_x: self.scoop_rear_x,
            scoop_front_x: self.scoop_front_x,
            scoop_level_deg: self.scoop_level_deg,
            scoop_zero_deg: self.scoop_zero_deg,
            scoop_floor: self.scoop_floor,
            shot_on_target: self.shot_on_target,
            corner_from_x: self.corner_from_x,
            corner_full_x: self.corner_full_x,
            shot_max_time: self.shot_max_time,
            w_power: self.w_power,
            w_impulse: self.w_impulse,
            w_lift: self.w_lift,
            w_shot: self.w_shot,
            aim_cos_zero: self.aim_cos_zero,
            aim_cos_full: self.aim_cos_full,
            elev_zero_low: self.elev_zero_low,
            elev_full_low: self.elev_full_low,
            elev_full_high: self.elev_full_high,
            elev_zero_high: self.elev_zero_high,
            aerial_elev_zero_low: self.aerial_elev_zero_low,
            aerial_elev_full_low: self.aerial_elev_full_low,
            base_credit: self.base_credit,
            aerial_bonus: self.aerial_bonus,
            on_net_bonus: self.on_net_bonus,
            on_net_power_mix: self.on_net_power_mix,
            stretch_shot_weight: self.stretch_shot_weight,
            stretch_shot_from: self.stretch_shot_from,
            stretch_shot_full: self.stretch_shot_full,
            stretch_gain_weight: self.stretch_gain_weight,
            stretch_gain_from: self.stretch_gain_from,
            stretch_gain_full: self.stretch_gain_full,
            cancel_on_car_contact: self.cancel_on_car_contact,
            car_contact_margin: self.car_contact_margin,
            car_contact_lookback: self.car_contact_lookback,
            flick_weight: self.flick_weight,
            arm_credit: self.arm_credit,
            backward_penalty: self.backward_penalty,
            min_penalty_impulse: self.min_penalty_impulse,
            separation_dist: self.separation_dist,
            resolve_time: self.resolve_time,
            release_steps: self.release_steps,
            ride_hover_gap: self.ride_hover_gap,
            ride_leave_speed: self.ride_leave_speed,
            wall_margin: self.wall_margin,
            surface_margin: self.surface_margin,
            start_jump_margin: self.start_jump_margin,
            reset_max_ball_dist: self.reset_max_ball_dist,
            start_reset_max_ball_dist: self.start_reset_max_ball_dist,
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

    /// The ball on an arena surface at the step's end, or bounced off one during the step.
    fn ball_on_surface(&self, prev: &Body, now: &Body, car_touched: bool, dt: f32) -> bool {
        if near_arena_surface(now.pos, CV_BALL_RADIUS + self.ball_rest_margin) {
            return true;
        }
        if car_touched {
            return false;
        }
        let expected = prev.vel + Vec3A::new(0.0, 0.0, GRAVITY_Z * dt);
        if (now.vel - expected).length() < self.bounce_dv {
            return false;
        }
        let m = CV_BALL_RADIUS + self.bounce_margin;
        near_arena_surface(now.pos, m) || near_arena_surface(prev.pos, m)
    }

    /// The spin, roll, yaw and pitch terms of a turn, and whether roll and yaw go the same way.
    fn terms(&self, s: &[f32]) -> ([f32; 4], bool) {
        let pitch = (1.0 - ramp(s[S_PITCH].abs(), self.pitch_free_deg, self.pitch_zero_deg))
            * (1.0 - ramp(s[S_NOSE_UP], self.nose_up_free_deg, self.nose_up_zero_deg));
        let terms = [
            ramp(s[S_SPIN], self.spin_zero_deg, self.spin_full_deg),
            ramp(s[S_ROLL].abs(), self.roll_zero_deg, self.roll_full_deg),
            ramp(s[S_YAW].abs(), self.yaw_zero_deg, self.yaw_full_deg),
            pitch,
        ];
        (terms, s[S_ROLL] * s[S_YAW] > 0.0)
    }

    /// The bonus for the musty attempt that ended this step, if it came from the open dodge.
    fn judge(&self, m: &mut [f32], attempt: Option<Attempt>) -> f32 {
        let Some(attempt) = attempt else {
            return 0.0;
        };
        if m[B_OPEN] == 0.0 || attempt.arm_age != m[B_DODGE_AGE] {
            return 0.0;
        }
        m[B_OPEN] = 0.0;
        let s = &m[B_AT_DODGE..B_AT_DODGE + SPIN_STATE];
        let (terms, same_sense) = self.terms(s);
        // Up to the strike (an attempt that never struck: up to its end)
        let until = attempt.strike_age.unwrap_or(0.0);
        let happened = |at: usize| s[at] != 0.0 && s[at + 1] >= until;
        let gates = attempt.ending == Ending::Launch
            && m[B_OPEN_FROM_LIFTOFF] != 0.0
            && s[S_POPPED] != 0.0
            && (!self.ground_only || attempt.context == Context::Ground)
            && !happened(S_BALL_SURFACE)
            && !happened(S_CAR_SURFACE)
            && !happened(S_OTHER_TOUCH)
            && (same_sense || !self.tornado_sign)
            && terms.iter().all(|&t| t > 0.0);
        if !gates {
            return 0.0;
        }
        let bonus = self.breezi_scale * terms.iter().product::<f32>() * attempt.reward.max(0.0);
        // Since the dodge, not only up to the strike: the flight on to the net must be clean too
        let clean = s[S_BALL_SURFACE] == 0.0 && s[S_OTHER_TOUCH] == 0.0;
        m[B_FLIGHT] = flag(bonus > 0.0 && clean);
        bonus
    }

    fn update(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
        attempt: Option<Attempt>,
    ) -> f32 {
        let before = &prev.cars[car];
        if m[B_INIT] == 0.0 {
            // Starting in the air: the air phase starts here, with no pop
            m[B_INIT] = 1.0;
            let on = supported(before, &events.prev_detail(car), self.surface_margin);
            m[B_AIRBORNE] = flag(!on);
        }
        let ticks = step_ticks(events);
        m[B_CONTROL_AGE] += ticks;
        m[B_DODGE_AGE] += ticks;
        for base in [B_WIN, B_AT_DODGE] {
            for at in [S_BALL_SURFACE, S_OTHER_TOUCH, S_CAR_SURFACE] {
                m[base + at + 1] += ticks;
            }
        }
        let me = &cur.cars[car];
        if me.is_demoed {
            m[B_AIRBORNE..].fill(0.0);
            return 0.0;
        }
        let any_touch = events.cars.iter().any(|e| e.touched_ball);
        let other_touch = events
            .cars
            .iter()
            .enumerate()
            .any(|(other, e)| other != car && e.touched_ball);
        let ball_surface =
            self.ball_on_surface(&prev.ball, &cur.ball, any_touch, events.step_seconds);
        let hitbox = Hitbox {
            size: Vec3A::new(self.hitbox_length, self.hitbox_width, self.hitbox_height),
            offset: Vec3A::new(
                self.hitbox_offset_forward,
                self.hitbox_offset_right,
                self.hitbox_offset_up,
            ),
        };
        let car_near = hitbox_clearance(&me.body, hitbox) < self.car_clear_margin;
        if control_quality(me, &cur.ball, self.dribble()) >= self.control_threshold {
            m[B_CONTROL_AGE] = 0.0;
            m[B_EVER_CONTROL] = 1.0;
        }
        // The dodge: the turn so far, plus the step's first ticks, which still ran the step before's
        // inputs (the action delay)
        let airborne = m[B_AIRBORNE] != 0.0;
        if me.has_flipped && !before.has_flipped {
            m[B_OPEN] = 1.0;
            m[B_OPEN_FROM_LIFTOFF] = flag(airborne);
            m[B_DODGE_AGE] = 0.0;
            if airborne {
                m.copy_within(B_WIN..B_WIN + SPIN_STATE, B_AT_DODGE);
                if ticks > 1.0 {
                    let w = before.body.ang_vel * ((ticks - 1.0) * TICK_SECONDS);
                    add_turn(&mut m[B_AT_DODGE..B_MUSTY], w, before.body.rot);
                }
            } else {
                m[B_AT_DODGE..B_MUSTY].fill(0.0);
            }
        }
        if supported(me, &events.detail(car), self.surface_margin) {
            m[B_AIRBORNE] = 0.0;
            m[B_WIN..B_AT_DODGE].fill(0.0);
        } else {
            if !airborne {
                m[B_AIRBORNE] = 1.0;
                m[B_WIN..B_AT_DODGE].fill(0.0);
                m[B_WIN + S_POPPED] = flag(
                    m[B_EVER_CONTROL] != 0.0 && m[B_CONTROL_AGE] <= secs_to_ticks(self.pop_window),
                );
            }
            let win = &mut m[B_WIN..B_AT_DODGE];
            add_turn(
                win,
                rotation_between(before.body.rot, me.body.rot),
                before.body.rot,
            );
            mark_first(win, S_BALL_SURFACE, ball_surface);
            mark_first(win, S_OTHER_TOUCH, other_touch);
            mark_first(win, S_CAR_SURFACE, car_near);
        }
        if m[B_OPEN] != 0.0 {
            let at_dodge = &mut m[B_AT_DODGE..B_MUSTY];
            mark_first(at_dodge, S_BALL_SURFACE, ball_surface);
            mark_first(at_dodge, S_OTHER_TOUCH, other_touch);
            mark_first(at_dodge, S_CAR_SURFACE, car_near);
        }
        if m[B_FLIGHT] != 0.0 && m[B_OPEN] == 0.0 && (ball_surface || any_touch) {
            m[B_FLIGHT] = 0.0;
        }
        let paid = self.judge(m, attempt);
        if m[B_FLIGHT] != 0.0 && cur.goal_scored.is_some() {
            m[B_FLIGHT] = 0.0;
            if cur.goal_scored == Some(me.team) {
                return paid + self.air_goal_bonus;
            }
        }
        paid
    }
}

impl Reward for BreeziFlick {
    fn memory(&self) -> usize {
        B_MUSTY + MUSTY_MEMORY
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (own, musty_memory) = memory.split_at_mut(B_MUSTY);
        let musty = self.musty();
        let (musty_reward, attempt) = musty.step(prev, cur, events, car, musty_memory);
        self.musty_share * musty_reward + self.update(prev, cur, events, car, own, attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn rotations_are_measured_and_wound_back() {
        let w = Vec3A::new(0.3, -0.2, 0.5);
        let a = Mat3A::IDENTITY;
        let b = turned(a, w);
        assert!((rotation_between(a, b) - w).length() < 1e-4);
        let body = Body {
            rot: b,
            ang_vel: w * 120.0,
            vel: Vec3A::new(120.0, 0.0, 0.0),
            ..Body::RESTING
        };
        let (pos, rot) = pose_back(&body, 1.0);
        assert!((pos.x + 1.0).abs() < 1e-4 && (rot.x_axis - a.x_axis).length() < 1e-4);
    }

    #[test]
    fn boxes_and_shots() {
        let h = Vec3A::splat(50.0);
        let r = Mat3A::IDENTITY;
        assert!(boxes_overlap(
            Vec3A::ZERO,
            r,
            h,
            Vec3A::new(90.0, 0.0, 0.0),
            r,
            h
        ));
        assert!(!boxes_overlap(
            Vec3A::ZERO,
            r,
            h,
            Vec3A::new(110.0, 0.0, 0.0),
            r,
            h
        ));
        let turned = Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!(!boxes_overlap(
            Vec3A::ZERO,
            r,
            h,
            Vec3A::new(125.0, 0.0, 0.0),
            turned,
            h
        ));
        assert!(boxes_overlap(
            Vec3A::ZERO,
            r,
            h,
            Vec3A::new(115.0, 0.0, 0.0),
            turned,
            h
        ));
        let shot = project_to_goal_line(
            Vec3A::new(0.0, 3000.0, 300.0),
            Vec3A::new(0.0, 2000.0, 100.0),
            Team::Blue,
            5.0,
        );
        assert!(shot.reaches && shot.on_target);
        let away = project_to_goal_line(
            Vec3A::new(0.0, 3000.0, 300.0),
            Vec3A::new(0.0, 2000.0, 100.0),
            Team::Orange,
            5.0,
        );
        assert!(!away.reaches);
        assert_eq!(ramp(5.0, 5.0, 5.0), 1.0);
        assert!((ramp(250.0, 450.0, 250.0) - 1.0).abs() < 1e-6);
    }

    /// Facing +y on its wheels: forward, right, up.
    fn level() -> Mat3A {
        Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z)
    }

    /// `rot` pitched nose-down by `deg` about its own right axis.
    fn nose_down(rot: Mat3A, deg: f32) -> Mat3A {
        turned(rot, rot.y_axis * deg.to_radians())
    }

    /// The same moment seen with the teams swapped: everything turned half a turn about the vertical.
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
        out.goal_scored = g.goal_scored.map(Team::opposite);
        out
    }

    fn mirrored_events(e: &StepEvents) -> StepEvents {
        let mut out = e.clone();
        for d in out.details.iter_mut().chain(out.prev_details.iter_mut()) {
            d.last_hit_ball_pos = as_blue(Team::Orange, d.last_hit_ball_pos);
        }
        out
    }

    /// Steps of a game: each snapshot after the first, with what happened on the way to it.
    struct Play {
        games: Vec<GameSnapshot>,
        events: Vec<StepEvents>,
    }

    impl Play {
        fn new(start: GameSnapshot) -> Self {
            Self {
                games: vec![start],
                events: Vec::new(),
            }
        }

        fn push(&mut self, g: GameSnapshot, mut e: StepEvents) {
            e.tick = 8 * self.games.len() as u64;
            self.games.push(g);
            self.events.push(e);
        }

        fn mirrored(&self) -> Self {
            Self {
                games: self.games.iter().map(mirrored).collect(),
                events: self.events.iter().map(mirrored_events).collect(),
            }
        }

        /// Car 0's reward each step, from fresh memory.
        fn run(&self, reward: &dyn Reward) -> Vec<f32> {
            let mut memory = vec![0.0; reward.memory()];
            (1..self.games.len())
                .map(|i| {
                    let (prev, cur) = (&self.games[i - 1], &self.games[i]);
                    reward.value_remembering(prev, cur, &self.events[i - 1], 0, &mut memory)
                })
                .collect()
        }
    }

    const CAR: Vec3A = Vec3A::new(0.0, 0.0, 300.0);

    /// A car in the air after a jump off the floor, and an opponent far away.
    fn in_the_air(rot: Mat3A, ball: Vec3A) -> GameSnapshot {
        let mut g = game(
            ball,
            &[
                (Team::Blue, CAR),
                (Team::Orange, Vec3A::new(0.0, 4500.0, 17.0)),
            ],
        );
        let me = &mut g.cars[0];
        me.body.rot = rot;
        me.on_ground = false;
        me.has_jumped = true;
        g
    }

    /// From the musty pose to the end: the dodge (a backflip if `torque_y` < 0), the roof scooping the
    /// ball goalward, the ball flying off.
    fn finish_musty(play: &mut Play, torque_y: f32) {
        let pose = nose_down(level(), 100.0);
        let ball = CAR + Vec3A::new(0.0, 150.0, 100.0);
        let mut dodge = in_the_air(pose, ball);
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        let mut e = quiet(2);
        e.details[0].flip_rel_torque = Vec3A::new(0.0, torque_y, 0.0);
        play.push(dodge.clone(), e);
        // The strike: the ball on top of the roof, leaving goalward and rising
        let mut strike = dodge.clone();
        strike.ball.pos = CAR + pose * Vec3A::new(13.9, 0.0, 134.0);
        strike.ball.vel = Vec3A::new(0.0, 2000.0, 600.0);
        let mut e = touched(quiet(2), 0);
        let tick = 8 * play.games.len() as u64;
        e.details[0].last_hit_tick = Some(tick);
        e.details[0].last_hit_ball_pos = strike.ball.pos;
        e.details[0].flip_rel_torque = Vec3A::new(0.0, torque_y, 0.0);
        play.push(strike.clone(), e);
        let mut away = strike;
        away.ball.pos = Vec3A::new(0.0, 600.0, 700.0);
        away.ball.vel = Vec3A::new(0.0, 2000.0, 580.0);
        play.push(away, quiet(2));
    }

    fn musty_play(torque_y: f32) -> Play {
        let mut play = Play::new(in_the_air(
            nose_down(level(), 60.0),
            CAR + Vec3A::new(0.0, 150.0, 100.0),
        ));
        finish_musty(&mut play, torque_y);
        play
    }

    fn defaults<T: for<'de> Deserialize<'de>>() -> T {
        serde_json::from_value(serde_json::json!({})).expect("every setting has a default")
    }

    #[test]
    fn a_musty_pays_once_the_ball_has_left_the_roof() {
        let musty: MustyFlick = defaults();
        assert_eq!((musty.arm_max_torque_y, musty.release_steps), (-0.4, 2));
        let paid = musty_play(-1.0).run(&musty);
        assert_eq!(paid[..2], [0.0, 0.0], "nothing at the dodge or the hit");
        // A ground musty, the ball leaving steeply nose-down (half credit): (0.25 + 0.75 x quality 0.7945) / 2
        assert!((paid[2] - 0.4229).abs() < 1e-3, "{paid:?}");
        assert_eq!(
            musty_play(-1.0).run(&musty),
            paid,
            "fresh memory starts again"
        );
    }

    #[test]
    fn a_front_flip_or_another_touch_earns_nothing() {
        let musty: MustyFlick = defaults();
        assert!(musty_play(1.0).run(&musty).iter().all(|&r| r == 0.0));
        let mut play = musty_play(-1.0);
        play.events[2] = touched(quiet(2), 1);
        assert!(play.run(&musty).iter().all(|&r| r == 0.0));
    }

    #[test]
    fn orange_is_paid_as_blue() {
        let musty: MustyFlick = defaults();
        let play = musty_play(-1.0);
        let (blue, orange) = (play.run(&musty), play.mirrored().run(&musty));
        assert!(
            blue.iter().zip(&orange).all(|(b, o)| (b - o).abs() < 1e-4),
            "{blue:?} {orange:?}"
        );
    }

    /// A Breezi: a dribble, the pop, a tornado (a full turn about the nose and roof together, `tornado`
    /// degrees), the nose down past vertical, then the musty.
    fn breezi_play(tornado: f32) -> Play {
        let ball_on_roof = Vec3A::new(0.0, 20.0, 150.0);
        let mut dribble = game(
            ball_on_roof,
            &[
                (Team::Blue, Vec3A::new(0.0, 0.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 4500.0, 17.0)),
            ],
        );
        dribble.cars[0].body.rot = level();
        let mut play = Play::new(dribble.clone());
        play.push(dribble, quiet(2));
        let ball = CAR + Vec3A::new(0.0, 150.0, 100.0);
        let axis = (Vec3A::Y + Vec3A::Z).normalize();
        let steps = 8;
        for k in 0..=steps {
            let rot = turned(
                level(),
                axis * (tornado * k as f32 / steps as f32).to_radians(),
            );
            play.push(in_the_air(rot, ball), quiet(2));
        }
        for k in 1..=3 {
            let rot = nose_down(
                turned(level(), axis * tornado.to_radians()),
                25.0 * k as f32,
            );
            play.push(in_the_air(rot, ball), quiet(2));
        }
        finish_musty(&mut play, -1.0);
        play
    }

    #[test]
    fn a_breezi_pays_the_musty_again_and_a_plain_musty_nothing() {
        let breezi: BreeziFlick = defaults();
        let musty: MustyFlick = defaults();
        let play = breezi_play(360.0);
        let (bonus, plain) = (play.run(&breezi), play.run(&musty));
        let (bonus, plain) = (bonus.iter().sum::<f32>(), plain.iter().sum::<f32>());
        assert!(
            plain > 0.3 && (bonus - plain).abs() < 1e-5,
            "{bonus} {plain}"
        );
        // No tornado: the same musty, no Breezi
        let no_spin = breezi_play(0.0);
        assert!(no_spin.run(&musty).iter().sum::<f32>() > 0.3);
        assert!(no_spin.run(&breezi).iter().all(|&r| r == 0.0));
        // musty_share 1 pays both
        let both: BreeziFlick =
            serde_json::from_value(serde_json::json!({ "musty_share": 1.0 })).unwrap();
        assert!((play.run(&both).iter().sum::<f32>() - 2.0 * plain).abs() < 1e-5);
    }

    /// A Breezi whose ball flies on (at `low_z` for one step: below 101 that is a bounce) and goes into
    /// the goal `scorer` attacks.
    fn breezi_goal_play(low_z: f32, scorer: Team) -> Play {
        let mut play = breezi_play(360.0);
        let last = play.games.last().unwrap().clone();
        for (k, z) in [(1.0, 600.0), (2.0, low_z), (3.0, 500.0)] {
            let mut g = last.clone();
            g.ball.pos = Vec3A::new(0.0, 600.0 + 1200.0 * k, z);
            g.ball.vel = Vec3A::new(0.0, 2000.0, 0.0);
            play.push(g, quiet(2));
        }
        let mut goal = last;
        goal.ball.pos = Vec3A::new(0.0, 5250.0, 400.0);
        goal.ball.vel = Vec3A::new(0.0, 2000.0, 0.0);
        goal.goal_scored = Some(scorer);
        play.push(goal, quiet(2));
        play
    }

    #[test]
    fn the_air_goal_bonus_pays_a_breezi_that_scores_before_it_bounces() {
        let plain: BreeziFlick = defaults();
        let with_bonus: BreeziFlick =
            serde_json::from_value(serde_json::json!({ "air_goal_bonus": 2.0 })).unwrap();
        let sum = |p: &Play, r: &BreeziFlick| p.run(r).iter().sum::<f32>();
        let clean = breezi_goal_play(700.0, Team::Blue);
        let breezi = sum(&clean, &plain);
        assert!(breezi > 0.3, "{breezi}");
        assert!((sum(&clean, &with_bonus) - breezi - 2.0).abs() < 1e-4);
        assert!(
            (sum(&clean.mirrored(), &with_bonus) - breezi - 2.0).abs() < 1e-4,
            "orange too"
        );
        let bounced = breezi_goal_play(95.0, Team::Blue);
        assert!(
            (sum(&bounced, &with_bonus) - breezi).abs() < 1e-4,
            "a bounce first: no bonus"
        );
        let own_goal = breezi_goal_play(700.0, Team::Orange);
        assert!(
            (sum(&own_goal, &with_bonus) - breezi).abs() < 1e-4,
            "the other net: no bonus"
        );
        let mut touched_again = breezi_goal_play(700.0, Team::Blue);
        let at = touched_again.events.len() - 2;
        touched_again.events[at] = touched(quiet(2), 0);
        assert!(
            (sum(&touched_again, &with_bonus) - breezi).abs() < 1e-4,
            "a touch first: no bonus"
        );
        // No Breezi, no bonus: the same flight after a plain musty
        let mut no_spin = breezi_play(0.0);
        let tail: Vec<_> = clean.games[clean.games.len() - 4..].to_vec();
        for g in tail {
            no_spin.push(g, quiet(2));
        }
        assert!(no_spin.run(&with_bonus).iter().all(|&r| r == 0.0));
    }

    #[test]
    fn a_breezi_needs_the_ball_in_the_air_and_works_for_orange() {
        let breezi: BreeziFlick = defaults();
        let play = breezi_play(360.0);
        let blue: f32 = play.run(&breezi).iter().sum();
        let orange: f32 = play.mirrored().run(&breezi).iter().sum();
        assert!(
            blue > 0.3 && (blue - orange).abs() < 1e-4,
            "{blue} {orange}"
        );
        let mut bounced = breezi_play(360.0);
        bounced.games[5].ball.pos.z = 95.0;
        assert!(bounced.run(&breezi).iter().all(|&r| r == 0.0));
    }
}
