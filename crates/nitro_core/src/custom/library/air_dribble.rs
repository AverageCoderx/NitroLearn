//! Air dribbles: chains of separate, useful air touches by one car on a ball kept off the floor, a goal
//! at the end paying the most.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::aerial_kit::*;

/// Air dribbles: a goal from a valid chain of air touches pays goal_reward (2) times its bonuses (extra touches,
/// air roll, flip resets, distance carried); shaping for touches, control and carrying the ball toward goal is
/// capped at 1.5 an episode; an own goal off the chain costs 0.5.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirDribble {
    /// Touches for a valid air dribble popped off the floor (GROUND origin).
    #[serde(default = "min_touches_ground")]
    pub min_touches_ground: i32,
    /// Touches for one started from a roof dribble (DRIBBLE origin).
    #[serde(default = "min_touches_dribble")]
    pub min_touches_dribble: i32,
    /// Touches for one popped off a wall or the ceiling (WALL origin).
    #[serde(default = "min_touches_wall")]
    pub min_touches_wall: i32,
    /// Touches for an aerial catch of a high ball (AERIAL origin).
    #[serde(default = "min_touches_aerial")]
    pub min_touches_aerial: i32,
    /// Seconds allowed between consecutive chain touches.
    #[serde(default = "max_touch_gap")]
    pub max_touch_gap: f32,
    /// Physics ticks without contact before a touch counts as a new one (0.05 s).
    #[serde(default = "min_separation_ticks")]
    pub min_separation_ticks: i32,
    /// ...and how far (uu, hitbox to ball surface) the ball must have left the car since the last contact.
    #[serde(default = "min_separation_gap")]
    pub min_separation_gap: f32,
    /// A touch counts as a new one only this long (s) after the last new one, so quick repeated touches cannot
    /// be farmed; sooner, it is the same touch going on (0: off).
    #[serde(default = "min_touch_interval")]
    pub min_touch_interval: f32,
    /// Each paid chain touch also pays this times the speed it added to the ball relative to the car (uu/s, the
    /// ball's velocity less the car's, after the step against before), along the line halfway between straight up
    /// and toward the goal, over touch_push_away_full, 0 to 1: a touch that sends the ball up and on away from the
    /// car makes room for the next touch or a reset (0: off). Part of the shaping.
    #[serde(default = "touch_push_away_weight")]
    pub touch_push_away_weight: f32,
    /// ...the relative speed added (uu/s) at which touch_push_away_weight pays in full.
    #[serde(default = "touch_push_away_full")]
    pub touch_push_away_full: f32,
    /// The ball's radius for that gap (RocketSim's 91.25).
    #[serde(default = "ball_radius")]
    pub ball_radius: f32,
    /// Seconds from a surface (setup) touch to the first air touch.
    #[serde(default = "setup_window")]
    pub setup_window: f32,
    /// Ball height at the first air touch for an AERIAL catch.
    #[serde(default = "aerial_catch_height")]
    pub aerial_catch_height: f32,
    /// ...with the car airborne at least this long (s).
    #[serde(default = "aerial_min_air_time")]
    pub aerial_min_air_time: f32,
    /// On a surface with the roof's up.z below this: a wall or the ceiling.
    #[serde(default = "wall_up_z")]
    pub wall_up_z: f32,
    /// Ball centre below this: touching the floor (RLGym's ball radius + 20).
    #[serde(default = "floor_contact_height")]
    pub floor_contact_height: f32,
    /// A down-to-up bounce below this height that no car caused: a floor bounce.
    #[serde(default = "floor_bounce_height")]
    pub floor_bounce_height: f32,
    /// Roof dribble: lowest ball height above the car, along its up.
    #[serde(default = "dribble_min_up")]
    pub dribble_min_up: f32,
    /// Roof dribble: highest ball height above the car.
    #[serde(default = "dribble_max_up")]
    pub dribble_max_up: f32,
    /// Roof dribble: farthest ball from the car centre in the car's forward/right plane.
    #[serde(default = "dribble_max_planar")]
    pub dribble_max_planar: f32,
    /// Seconds on the roof before a surface touch counts as a DRIBBLE origin.
    #[serde(default = "dribble_min_time")]
    pub dribble_min_time: f32,
    /// The goal at the end of a valid air dribble, before its bonuses.
    #[serde(default = "goal_reward")]
    pub goal_reward: f32,
    /// Goal bonus per touch beyond the minimum...
    #[serde(default = "per_extra_touch")]
    pub per_extra_touch: f32,
    /// ...up to this many touches.
    #[serde(default = "max_extra_touches")]
    pub max_extra_touches: i32,
    /// Seconds from the chain's last touch to the goal.
    #[serde(default = "goal_window")]
    pub goal_window: f32,
    /// The ball must be inside the goal mouth when the goal is scored.
    #[serde(default = "require_goal_mouth")]
    pub require_goal_mouth: bool,
    /// The goal mouth's half width.
    #[serde(default = "goal_half_width")]
    pub goal_half_width: f32,
    /// The goal mouth's height.
    #[serde(default = "goal_height")]
    pub goal_height: f32,
    /// Paid (negative) when a chain the car touched recently goes into its own net.
    #[serde(default = "own_goal_penalty")]
    pub own_goal_penalty: f32,
    /// false: the ball must not touch the floor between the last touch and the goal (a shot may bounce).
    #[serde(default = "allow_floor_before_goal")]
    pub allow_floor_before_goal: bool,
    /// An air touch extends a chain only with the ball centre at least this high.
    #[serde(default = "min_air_touch_height")]
    pub min_air_touch_height: f32,
    /// In the air, the car farther than this from the ball ends the chain.
    #[serde(default = "max_car_ball_distance")]
    pub max_car_ball_distance: f32,
    /// ...or more than this above the ball (control lost: the goal no longer counts either).
    #[serde(default = "above_margin")]
    pub above_margin: f32,
    /// ...or more than this goal-side of the ball (control lost).
    #[serde(default = "ahead_margin")]
    pub ahead_margin: f32,
    /// A chain touch must add this much upward ball speed (uu/s, gravity added back)...
    #[serde(default = "min_touch_lift")]
    pub min_touch_lift: f32,
    /// ...or this much toward the opponent goal...
    #[serde(default = "min_touch_push")]
    pub min_touch_push: f32,
    /// ...without driving the ball down by more than this (uu/s).
    #[serde(default = "max_push_touch_drop")]
    pub max_push_touch_drop: f32,
    /// Boost (0-100) when the car went for it; below this the chain earns nothing.
    #[serde(default = "min_boost_at_start")]
    pub min_boost_at_start: f32,
    /// Shaping per second of air roll during an open chain's air phase.
    #[serde(default = "air_roll_rate")]
    pub air_roll_rate: f32,
    /// Air roll counts with the car rolling at least this fast about its nose (rad/s).
    #[serde(default = "air_roll_min_rate")]
    pub air_roll_min_rate: f32,
    /// The most air roll shaping per chain.
    #[serde(default = "max_air_roll_per_chain")]
    pub max_air_roll_per_chain: f32,
    /// Goal payout x (1 + this x the share of the chain's air time spent air rolling; full with a flip reset).
    #[serde(default = "air_roll_goal_bonus")]
    pub air_roll_goal_bonus: f32,
    /// Goal payout x (1 + this x the flip resets during the air dribble)...
    #[serde(default = "reset_goal_bonus")]
    pub reset_goal_bonus: f32,
    /// ...counting at most this many.
    #[serde(default = "max_resets_for_bonus")]
    pub max_resets_for_bonus: i32,
    /// ...and none at all below this many (2 = a goal off one reset pays as a plain air dribble
    /// goal; 0 = every reset counts).
    #[serde(default = "min_resets_for_bonus")]
    pub min_resets_for_bonus: i32,
    /// A chain touch this soon after a dodge started that pushes the ball goalward is a shot (s).
    #[serde(default = "shot_dodge_window")]
    pub shot_dodge_window: f32,
    /// Seconds since the last reset contact for a flip reset to count again.
    #[serde(default = "min_reset_interval")]
    pub min_reset_interval: f32,
    /// Goal payout x (1 + this x the ramp of the ball's progress toward goal over the chain).
    #[serde(default = "cross_map_bonus")]
    pub cross_map_bonus: f32,
    /// That ramp starts at this much progress (uu)...
    #[serde(default = "cross_map_start")]
    pub cross_map_start: f32,
    /// ...and is full at this much.
    #[serde(default = "cross_map_full")]
    pub cross_map_full: f32,
    /// Shaping per chain-extending air touch (touch 2 on), times the touch's quality.
    #[serde(default = "touch_reward")]
    pub touch_reward: f32,
    /// The most touches paid that shaping per chain.
    #[serde(default = "max_paid_touches_per_chain")]
    pub max_paid_touches_per_chain: i32,
    /// Shaping per second in the air phase for keeping the ball up, high and close.
    #[serde(default = "control_rate")]
    pub control_rate: f32,
    /// Ball height for the full control shaping.
    #[serde(default = "control_full_height")]
    pub control_full_height: f32,
    /// The most control shaping per chain.
    #[serde(default = "max_control_per_chain")]
    pub max_control_per_chain: f32,
    /// Room between car and ball to set up a reset: the control shaping is full with the car's centre this far (uu)
    /// from the ball's, rising from nothing at 200 (the ball against the car) and falling to nothing at
    /// max_car_ball_distance, so a bouncing ball with room under it pays more than one carried glued to the car
    /// (0: off, full anywhere within 250).
    #[serde(default = "control_space_dist")]
    pub control_space_dist: f32,
    /// Shaping per second in the air phase for carrying the ball toward the net.
    #[serde(default = "progress_rate")]
    pub progress_rate: f32,
    /// Paid once when a chain reaches its minimum touches.
    #[serde(default = "valid_chain_bonus")]
    pub valid_chain_bonus: f32,
    /// An AERIAL catch's first touch, times its softness.
    #[serde(default = "catch_bonus")]
    pub catch_bonus: f32,
    /// Ball-car speed difference after the catch at which softness reaches 0.
    #[serde(default = "catch_soft_speed")]
    pub catch_soft_speed: f32,
    /// All shaping together per episode (below goal_reward: scoring stays the biggest motivation).
    #[serde(default = "max_shaping_per_episode")]
    pub max_shaping_per_episode: f32,
    /// Positioning: while a chain is open and the car in the air, pays for getting placed for the next touch
    /// (close, under, behind the ball, matching its speed, facing it) and takes it back for drifting out; only a
    /// touch that extends the chain keeps it, at most this per touch (0 = off).
    #[serde(default = "position_weight")]
    pub position_weight: f32,
    /// Positioning's discount per step: set it to the training's (`ppo` discount per step).
    #[serde(default = "position_gamma")]
    pub position_gamma: f32,
    /// Positioning: full closeness within this distance of the ball (none at max_car_ball_distance).
    #[serde(default = "position_full_dist")]
    pub position_full_dist: f32,
    /// Positioning: full credit this far below the ball's centre (none level with it).
    #[serde(default = "position_below_full")]
    pub position_below_full: f32,
    /// Positioning: full credit this far behind the ball, toward the own goal.
    #[serde(default = "position_behind_full")]
    pub position_behind_full: f32,
    /// Positioning: full credit at this ball-car speed difference or less...
    #[serde(default = "position_match_full")]
    pub position_match_full: f32,
    /// ...none at this one.
    #[serde(default = "position_match_zero")]
    pub position_match_zero: f32,
    /// Positioning: none once the car has been falling at least this share of gravity (not boosting against it)...
    #[serde(default = "position_fall_fraction")]
    pub position_fall_fraction: f32,
    /// ...for longer than this (s): riding the ball down pays nothing; a short coast into a touch still counts.
    #[serde(default = "position_fall_grace")]
    pub position_fall_grace: f32,
    /// "Toward goal" in the pay (progress, touch push and quality, shots, positioning behind the ball) is
    /// horizontally from the ball to the opponent goal's centre, and the cross-map bonus measures how much closer
    /// to it the ball got (false: straight downfield and y progress). The car in front of the ball,
    /// which voids the chain, stays downfield.
    #[serde(default = "aim_at_goal")]
    pub aim_at_goal: bool,
    /// Each paid chain touch (as touch_reward) also pays this times the upward ball speed it added (gravity added
    /// back) over touch_lift_full, 0 to 1: bouncy touches that pop the ball up pay more than ones that only carry
    /// or push it (0: off). Part of the shaping, under max_shaping_per_episode.
    #[serde(default = "touch_lift_weight")]
    pub touch_lift_weight: f32,
    /// ...the upward speed added (uu/s) at which touch_lift_weight pays in full.
    #[serde(default = "touch_lift_full")]
    pub touch_lift_full: f32,
    /// Each paid chain touch also pays this times how directly under the ball the car is at the touch, 0 to 1 by
    /// the car-to-ball line's angle from straight up (0: off). Part of the shaping, under max_shaping_per_episode.
    #[serde(default = "touch_under_weight")]
    pub touch_under_weight: f32,
    /// ...in full with that line's cosine from straight up at this or more (0.9: within about 26 degrees)...
    #[serde(default = "touch_under_full")]
    pub touch_under_full: f32,
    /// ...and nothing at this or less (0.6: about 53 degrees).
    #[serde(default = "touch_under_zero")]
    pub touch_under_zero: f32,
    /// A flip reset also counts when RocketSim gives the flip back a step after the wheels reach the ball, while
    /// they are still on it (the one-step test misses about a third of real resets at tick skip 4); false: the
    /// one-step test only.
    #[serde(default)]
    pub two_step_resets: bool,
}

defaults! { AirDribble {
    control_space_dist: f32 = 0.0,
    two_step_resets: bool = false,
    min_touches_ground: i32 = 2,
    min_touches_dribble: i32 = 3,
    min_touches_wall: i32 = 3,
    min_touches_aerial: i32 = 3,
    max_touch_gap: f32 = 1.5,
    min_separation_ticks: i32 = 6,
    min_separation_gap: f32 = 25.0,
    min_touch_interval: f32 = 0.0,
    touch_push_away_weight: f32 = 0.0,
    touch_push_away_full: f32 = 600.0,
    ball_radius: f32 = 91.25,
    setup_window: f32 = 1.2,
    aerial_catch_height: f32 = 450.0,
    aerial_min_air_time: f32 = 0.35,
    wall_up_z: f32 = 0.7,
    floor_contact_height: f32 = RLGYM_BALL_RADIUS + 20.0,
    floor_bounce_height: f32 = 300.0,
    dribble_min_up: f32 = 105.0,
    dribble_max_up: f32 = 185.0,
    dribble_max_planar: f32 = 110.0,
    dribble_min_time: f32 = 0.25,
    goal_reward: f32 = 2.0,
    per_extra_touch: f32 = 0.15,
    max_extra_touches: i32 = 6,
    goal_window: f32 = 2.5,
    require_goal_mouth: bool = true,
    goal_half_width: f32 = 892.755,
    goal_height: f32 = 642.775,
    own_goal_penalty: f32 = 0.5,
    allow_floor_before_goal: bool = false,
    min_air_touch_height: f32 = 300.0,
    max_car_ball_distance: f32 = 550.0,
    above_margin: f32 = 50.0,
    ahead_margin: f32 = 50.0,
    min_touch_lift: f32 = 100.0,
    min_touch_push: f32 = 125.0,
    max_push_touch_drop: f32 = 25.0,
    min_boost_at_start: f32 = 50.0,
    air_roll_rate: f32 = 0.15,
    air_roll_min_rate: f32 = 3.0,
    max_air_roll_per_chain: f32 = 0.2,
    air_roll_goal_bonus: f32 = 0.25,
    reset_goal_bonus: f32 = 0.5,
    max_resets_for_bonus: i32 = 4,
    min_resets_for_bonus: i32 = 0,
    shot_dodge_window: f32 = 0.5,
    min_reset_interval: f32 = 0.5,
    cross_map_bonus: f32 = 1.0,
    cross_map_start: f32 = 1500.0,
    cross_map_full: f32 = 6000.0,
    touch_reward: f32 = 0.15,
    max_paid_touches_per_chain: i32 = 10,
    control_rate: f32 = 0.25,
    control_full_height: f32 = 1000.0,
    max_control_per_chain: f32 = 0.6,
    progress_rate: f32 = 0.3,
    valid_chain_bonus: f32 = 0.25,
    catch_bonus: f32 = 0.15,
    catch_soft_speed: f32 = 1200.0,
    max_shaping_per_episode: f32 = 1.5,
    position_weight: f32 = 0.1,
    position_gamma: f32 = 0.99,
    position_full_dist: f32 = 200.0,
    position_below_full: f32 = 120.0,
    position_behind_full: f32 = 100.0,
    position_match_full: f32 = 300.0,
    position_match_zero: f32 = 1500.0,
    position_fall_fraction: f32 = 0.7,
    position_fall_grace: f32 = 0.5,
    aim_at_goal: bool = false,
    touch_lift_weight: f32 = 0.0,
    touch_lift_full: f32 = 600.0,
    touch_under_weight: f32 = 0.0,
    touch_under_full: f32 = 0.9,
    touch_under_zero: f32 = 0.6,
} }

// How a chain started.
const GROUND: i32 = 1;
const DRIBBLE: i32 = 2;
const WALL: i32 = 3;
const AERIAL: i32 = 4;

kept! {
    /// Each car's own state between steps (ticks on the episode clock).
    struct Track {
        /// Last physics tick this car was in contact with the ball.
        last_contact_tick: f32,
        had_contact: bool,
        /// Largest ball-hitbox gap seen since the last contact.
        max_gap_since_contact: f32,
        /// The surface the car last stood on was a wall or the ceiling.
        on_wall_last: bool,
        left_surface_tick: f32,
        was_on_surface: bool,
        /// Continuous time with the ball resting on the roof.
        dribble_time: f32,
        /// Boost when the car last left a surface.
        boost_at_takeoff: f32,
        shaping_paid: f32,
        /// The positioning potential at the last step.
        position_phi: f32,
        /// Seconds the car has been falling at position_fall_fraction of gravity or more.
        fall_time: f32,
        /// This step counted a touch that extends a chain (touches 2 to max_paid_touches_per_chain + 1).
        extended: bool,
        /// The last new touch's tick (min_touch_interval), and whether there was one.
        last_new_touch_tick: f32,
        had_new_touch: bool,
        /// No flip left in the air, waiting for a flip reset (`flip_reset_over_steps`).
        reset_waiting: bool,
    }
}

kept! {
    /// The car's current chain of touches.
    struct Chain {
        /// Still accepting touches.
        open: bool,
        /// Can still be paid for a goal.
        eligible: bool,
        origin: i32,
        touches: i32,
        air_touches: i32,
        /// Started by a surface touch; waiting for the first air touch.
        setup_pending: bool,
        valid: bool,
        goal_paid: bool,
        /// The ball touched the floor after the chain's last touch.
        floor_since_touch: bool,
        last_touch_tick: f32,
        paid_touches: i32,
        boost_ok: bool,
        start_ball_pos: Vec3A,
        air_time: f32,
        roll_time: f32,
        roll_paid: f32,
        control_paid: f32,
        resets: i32,
        /// The last chain touch was a shot (a dodge into the ball toward goal).
        shot_taken: bool,
        had_reset_contact: bool,
        last_reset_contact_tick: f32,
    }
}

impl Chain {
    fn fresh() -> Self {
        Self {
            boost_ok: true,
            ..Self::default()
        }
    }

    /// A new chain at touch 1, open and eligible.
    fn start(origin: i32, setup_pending: bool, boost_ok: bool) -> Self {
        Self {
            open: true,
            eligible: true,
            setup_pending,
            origin,
            boost_ok,
            ..Self::default()
        }
    }

    fn close(&mut self) {
        self.open = false;
        self.setup_pending = false;
    }
}

/// What a step gives the reward to look at.
struct Ctx<'a> {
    prev: &'a GameSnapshot,
    cur: &'a GameSnapshot,
    car: usize,
    dt: f32,
    detail: CarDetail,
    /// "Toward goal" for the pay terms (downfield, or toward the goal's centre with aim_at_goal).
    goal_dir: Vec3A,
    /// Straight downfield, for the checks that void a chain (a car beside the ball near the net is not in front).
    downfield: Vec3A,
}

impl AirDribble {
    fn min_touches(&self, origin: i32) -> i32 {
        match origin {
            GROUND => self.min_touches_ground,
            DRIBBLE => self.min_touches_dribble,
            WALL => self.min_touches_wall,
            AERIAL => self.min_touches_aerial,
            _ => 1 << 20,
        }
    }

    /// The horizontal unit direction "toward goal" for a ball at `ball`: to the opponent goal's centre with
    /// aim_at_goal (a ball past the goal line aims along it, at x = 0), else straight downfield.
    fn goal_dir(&self, team: Team, ball: Vec3A) -> Vec3A {
        let sign = attack_sign(team);
        let down = Vec3A::new(0.0, sign, 0.0);
        if !self.aim_at_goal {
            return down;
        }
        let to = Vec3A::new(-ball.x, (GOAL_LINE_Y - ball.y * sign).max(0.0) * sign, 0.0);
        let len = to.length();
        if len > 1.0 { to / len } else { down }
    }

    /// How far the ball got toward the goal from `from` to `to`: with aim_at_goal, how much closer horizontally
    /// to the opponent goal's centre (past the goal line counts as on it); else its progress downfield.
    fn progress(&self, team: Team, from: Vec3A, to: Vec3A) -> f32 {
        let sign = attack_sign(team);
        if !self.aim_at_goal {
            return (to.y - from.y) * sign;
        }
        let distance = |p: Vec3A| p.x.hypot((GOAL_LINE_Y - p.y * sign).max(0.0));
        distance(from) - distance(to)
    }

    fn in_goal_mouth(&self, ball: Vec3A) -> bool {
        ball.x.abs() < self.goal_half_width && ball.z < self.goal_height
    }

    /// The ball touched the floor during the step: low, or a down-to-up bounce near the floor no car caused.
    fn floor_contact(&self, s: &Ctx, any_car_touch: bool) -> bool {
        let (ball, before) = (&s.cur.ball, &s.prev.ball);
        if ball.pos.z < self.floor_contact_height {
            return true;
        }
        !any_car_touch
            && before.vel.z < -1.0
            && ball.vel.z > 1.0
            && before.pos.z.min(ball.pos.z) < self.floor_bounce_height
    }

    /// Distance from the ball's surface to the car's hitbox (0 when touching).
    fn ball_gap(&self, car: &Body, ball: Vec3A) -> f32 {
        let l = local(car, ball - car.pos) - HITBOX_OFFSET;
        let d = (l.abs() - HITBOX_SIZE * 0.5).max(Vec3A::ZERO);
        (d.length() - self.ball_radius).max(0.0)
    }

    fn ball_on_roof(&self, car: &CarSnapshot, ball: Vec3A) -> bool {
        if !car.on_ground || car.body.up().z < self.wall_up_z {
            return false;
        }
        let l = local(&car.body, ball - car.body.pos);
        l.z >= self.dribble_min_up
            && l.z <= self.dribble_max_up
            && (l.x * l.x + l.y * l.y).sqrt() <= self.dribble_max_planar
    }

    fn pay_shaping(&self, tr: &mut Track, amount: f32) -> f32 {
        let room = (self.max_shaping_per_episode - tr.shaping_paid).max(0.0);
        let paid = amount.clamp(0.0, room);
        tr.shaping_paid += paid;
        paid
    }

    /// What a touch did to the ball over the step: upward speed added (gravity added back) and horizontal speed
    /// added toward the opponent goal.
    fn touch_effect(s: &Ctx) -> (f32, f32) {
        let dv = s.cur.ball.vel - s.prev.ball.vel;
        let lift = dv.z + 650.0 * s.dt;
        let push = Vec3A::new(dv.x, dv.y, 0.0).dot(s.goal_dir);
        (lift, push)
    }

    /// The speed the step added to the ball relative to the car (uu/s), along the line halfway between straight up
    /// and the direction toward the goal.
    fn push_away(&self, s: &Ctx) -> f32 {
        let (car, pcar) = (&s.cur.cars[s.car], &s.prev.cars[s.car]);
        let rel = (s.cur.ball.vel - car.body.vel) - (s.prev.ball.vel - pcar.body.vel);
        let dir = (Vec3A::new(s.goal_dir.x, s.goal_dir.y, 0.0).normalize_or_zero() + Vec3A::Z)
            .normalize_or_zero();
        rel.dot(dir)
    }

    fn useful_touch(&self, s: &Ctx) -> bool {
        let (lift, push) = Self::touch_effect(s);
        lift >= self.min_touch_lift
            || (push >= self.min_touch_push && lift >= -self.max_push_touch_drop)
    }

    /// How directly under the ball the car is, 0-1 (touch_under_zero to touch_under_full).
    fn under(&self, car: &CarSnapshot, ball: Vec3A) -> f32 {
        let rel = ball - car.body.pos;
        let cos = rel.z / rel.length().max(1.0);
        clamp01((cos - self.touch_under_zero) / (self.touch_under_full - self.touch_under_zero))
    }

    /// Ball heading toward the opponent goal (horizontally), 0 when heading away.
    fn touch_quality(s: &Ctx) -> f32 {
        let v = s.cur.ball.vel;
        clamp01((Vec3A::new(v.x, v.y, 0.0).dot(s.goal_dir) + 500.0) / 1500.0)
    }

    fn count_touch(&self, tr: &mut Track, c: &mut Chain, s: &Ctx, touch_tick: f32) -> f32 {
        let car = &s.cur.cars[s.car];
        c.touches += 1;
        if c.touches >= 2 && c.touches <= self.max_paid_touches_per_chain + 1 {
            tr.extended = true;
        }
        c.last_touch_tick = touch_tick;
        c.floor_since_touch = false;
        let (_, push) = Self::touch_effect(s);
        let dodging = s.detail.is_flipping
            || (car.has_flipped
                && s.detail.flip_time > 0.0
                && s.detail.flip_time < self.shot_dodge_window);
        c.shot_taken = dodging && push >= self.min_touch_push;
        let mut r = 0.0;
        if c.touches == 1 {
            c.start_ball_pos = s.cur.ball.pos;
        }
        if c.boost_ok && c.touches >= 2 && c.paid_touches < self.max_paid_touches_per_chain {
            c.paid_touches += 1;
            let (lift, _) = Self::touch_effect(s);
            let lift_pay = self.touch_lift_weight * clamp01(lift / self.touch_lift_full)
                + self.touch_push_away_weight
                    * clamp01(self.push_away(s) / self.touch_push_away_full);
            let under_pay = self.touch_under_weight * self.under(car, s.cur.ball.pos);
            r += self.pay_shaping(
                tr,
                self.touch_reward * Self::touch_quality(s) + lift_pay + under_pay,
            );
        }
        if !c.valid && c.touches >= self.min_touches(c.origin) {
            c.valid = true;
            if c.boost_ok {
                r += self.pay_shaping(tr, self.valid_chain_bonus);
            }
        }
        r
    }

    /// How well the car is placed for its open chain's next touch, 0-1: 0 without an open chain, on a surface, or
    /// for a chain that went for it with too little boost.
    fn position_potential(&self, c: &Chain, s: &Ctx) -> f32 {
        let (p, ball) = (&s.cur.cars[s.car], &s.cur.ball);
        if !c.open || !c.boost_ok || c.touches == 0 || p.is_demoed || on_arena_surface(p) {
            return 0.0;
        }
        let rel = ball.pos - p.body.pos;
        let d = rel.length();
        if d >= self.max_car_ball_distance || d < 1.0 {
            return 0.0;
        }
        let close = clamp01(
            (self.max_car_ball_distance - d)
                / (self.max_car_ball_distance - self.position_full_dist),
        );
        let below = clamp01(rel.z / self.position_below_full);
        let behind = clamp01(rel.dot(s.goal_dir) / self.position_behind_full);
        let matching = clamp01(
            (self.position_match_zero - (ball.vel - p.body.vel).length())
                / (self.position_match_zero - self.position_match_full),
        );
        let facing = clamp01((p.body.forward().dot(rel / d) + 0.2) / 0.9);
        // Under the ball is what an air dribble touch needs: it gates; the rest scale
        close * below * (0.5 + 0.5 * behind) * (0.5 + 0.25 * matching + 0.25 * facing)
    }

    /// Positioning (potential-based): a touch that extends the chain keeps the approach so far, within the
    /// episode's shaping cap (the rest is paid back); any other chain end pays it back through the potential.
    fn positioning(&self, tr: &mut Track, c: &Chain, s: &Ctx) -> f32 {
        if self.position_weight <= 0.0 {
            return 0.0;
        }
        let mut reward = 0.0;
        if tr.extended {
            let kept = self.position_weight * tr.position_phi;
            let room = (self.max_shaping_per_episode - tr.shaping_paid).max(0.0);
            reward -= (kept - room).max(0.0);
            tr.shaping_paid += kept.min(room);
            tr.position_phi = 0.0;
        }
        let (p, pp) = (&s.cur.cars[s.car], &s.prev.cars[s.car]);
        let falling = !on_arena_surface(p)
            && (p.body.vel.z - pp.body.vel.z) / s.dt <= self.position_fall_fraction * GRAVITY_Z;
        tr.fall_time = if falling { tr.fall_time + s.dt } else { 0.0 };
        // Riding the ball down holds no position
        let phi = if s.cur.goal_scored.is_some() || tr.fall_time > self.position_fall_grace {
            0.0
        } else {
            self.position_potential(c, s)
        };
        reward += self.position_weight * (self.position_gamma * phi - tr.position_phi);
        tr.position_phi = phi;
        reward
    }

    fn update(&self, tr: &mut Track, c: &mut Chain, s: &Ctx, events: &StepEvents, now: f32) -> f32 {
        let mut reward = 0.0;
        let (p, pp, ball) = (&s.cur.cars[s.car], &s.prev.cars[s.car], &s.cur.ball);
        let since = |tick: f32| (now - tick) / 120.0;

        // Surface bookkeeping: wheels on the arena, not on the ball
        let on_surface = on_arena_surface(p) && !p.is_demoed;
        let on_wall = on_surface && p.body.up().z < self.wall_up_z;
        if on_surface {
            tr.on_wall_last = on_wall;
        }
        if tr.was_on_surface && !on_surface {
            tr.left_surface_tick = now;
            tr.boost_at_takeoff = pp.boost; // Boost just before leaving the surface
        }
        tr.was_on_surface = on_surface;
        tr.dribble_time = if self.ball_on_roof(p, ball.pos) {
            tr.dribble_time + s.dt
        } else {
            0.0
        };

        let touched = events.cars[s.car].touched_ball && !p.is_demoed;
        // A new touch: no contact for min_separation_ticks before this step, and the ball has left the hitbox
        let step_start = now - step_ticks(events).min(now);
        let gap_now = self.ball_gap(&p.body, ball.pos);
        if !touched {
            tr.max_gap_since_contact = tr.max_gap_since_contact.max(gap_now);
        }
        let new_touch = touched
            && (!tr.had_contact
                || (step_start >= tr.last_contact_tick + self.min_separation_ticks as f32
                    && tr.max_gap_since_contact >= self.min_separation_gap))
            && (!tr.had_new_touch
                || (now - tr.last_new_touch_tick) / 120.0 >= self.min_touch_interval);
        if new_touch {
            (tr.last_new_touch_tick, tr.had_new_touch) = (now, true);
        }
        if touched {
            tr.max_gap_since_contact = 0.0;
            let hit_tick = last_hit(events, s.car, now)
                .map(|hit| hit.tick)
                .filter(|&tick| tick <= now)
                .unwrap_or(now);
            tr.last_contact_tick = hit_tick.max(step_start + 1.0);
            tr.had_contact = true;
        }

        let other_touch = (0..s.cur.cars.len()).any(|o| o != s.car && events.cars[o].touched_ball);
        let floor = self.floor_contact(s, touched || other_touch);

        // ---- Things that end or void the chain ----
        if p.is_demoed {
            c.close();
        }
        if other_touch {
            c.close();
            c.eligible = false;
        }
        // A ball still low but rising right after a pop (setup touch) is not a floor contact
        let popped_rising =
            c.open && c.setup_pending && ball.vel.z > 0.0 && ball.pos.z < self.floor_bounce_height;
        if c.open && floor && !touched && !popped_rising {
            c.close();
        }
        if floor && !touched && !popped_rising && c.touches > 0 {
            c.floor_since_touch = true;
        }
        if c.open && c.air_touches > 0 && on_surface && !touched {
            c.close();
        }
        // Timeouts are measured to this step's contact tick when touching (else to the step's end)
        let touch_tick = if touched { tr.last_contact_tick } else { now };
        let gap_to = |tick: f32| (touch_tick - tick).max(0.0) / 120.0;
        if c.open && c.setup_pending && new_touch && gap_to(c.last_touch_tick) > self.setup_window {
            c.close();
        }
        if c.open && c.setup_pending && !touched && since(c.last_touch_tick) > self.setup_window {
            c.close();
        }
        if c.open
            && !c.setup_pending
            && (!touched || new_touch)
            && gap_to(c.last_touch_tick) > self.max_touch_gap
        {
            c.close();
        }
        if c.eligible && since(c.last_touch_tick) > self.goal_window {
            c.eligible = false;
        }

        // Air phase of an open chain: the car must stay close to the ball, below it and behind it (goal-wise)
        if c.open && !on_surface && c.touches > 0 {
            let rel = p.body.pos - ball.pos;
            if rel.z > self.above_margin || rel.dot(s.downfield) > self.ahead_margin {
                // Above or in front of the ball: control lost (after a shot, the shot's goal still counts)
                c.close();
                if !c.shot_taken {
                    c.eligible = false;
                }
            } else if rel.length() > self.max_car_ball_distance {
                c.close();
            }
        }
        let air_phase = c.open && !on_surface && c.touches > 0;
        let (two_step, waiting) = flip_reset_over_steps(tr.reset_waiting, pp, p, ball.pos, 350.0);
        tr.reset_waiting = waiting;
        let reset_now = if self.two_step_resets {
            two_step
        } else {
            flip_reset_happened(pp, p, ball.pos, 350.0)
        };
        if air_phase && reset_now {
            if !c.had_reset_contact
                || (now - c.last_reset_contact_tick) / 120.0 >= self.min_reset_interval
            {
                c.resets += 1;
            }
            c.had_reset_contact = true;
            c.last_reset_contact_tick = now;
            c.last_touch_tick = now; // A reset keeps the air dribble alive like a touch
        }
        if air_phase {
            c.air_time += s.dt;
            if c.boost_ok && c.control_paid < self.max_control_per_chain {
                // Keeping the ball up, high and close
                let height = ramp(
                    ball.pos.z,
                    self.min_air_touch_height,
                    self.control_full_height,
                );
                let dist = (p.body.pos - ball.pos).length();
                let close = if self.control_space_dist > 0.0 && dist < self.control_space_dist {
                    ramp(dist, 200.0, self.control_space_dist)
                } else {
                    let full = self.control_space_dist.max(250.0);
                    clamp01(1.0 - (dist - full) / (self.max_car_ball_distance - full).max(1.0))
                };
                let r = (self.control_rate * height * close * s.dt)
                    .min(self.max_control_per_chain - c.control_paid);
                c.control_paid += r;
                reward += self.pay_shaping(tr, r);
            }
            if c.boost_ok && self.progress_rate > 0.0 {
                // Carrying the ball toward the net
                let toward = (ball.vel.dot(s.goal_dir) / 2300.0).clamp(-0.5, 1.0);
                let r = self.progress_rate * toward * s.dt;
                reward += if r > 0.0 { self.pay_shaping(tr, r) } else { r };
            }
            if air_rolling(p, &s.detail, self.air_roll_min_rate) {
                c.roll_time += s.dt;
                if c.boost_ok && c.roll_paid < self.max_air_roll_per_chain {
                    let r =
                        (self.air_roll_rate * s.dt).min(self.max_air_roll_per_chain - c.roll_paid);
                    c.roll_paid += r;
                    reward += self.pay_shaping(tr, r);
                }
            }
        }

        // ---- Touches ----
        if touched && !other_touch && !p.is_demoed {
            let takeoff_ok = tr.boost_at_takeoff >= self.min_boost_at_start;
            if on_surface {
                // Surface (setup) touch: starts a new chain at touch 1, or refreshes the current setup contact
                let origin = if on_wall {
                    WALL
                } else if tr.dribble_time >= self.dribble_min_time {
                    DRIBBLE
                } else {
                    GROUND
                };
                if new_touch || !(c.open && c.setup_pending) {
                    *c = Chain::start(origin, true, p.boost >= self.min_boost_at_start);
                    reward += self.count_touch(tr, c, s, touch_tick);
                } else {
                    c.origin = origin; // The same contact: the origin may upgrade (GROUND -> DRIBBLE)
                    c.last_touch_tick = touch_tick;
                }
            } else if floor {
                // An airborne car hitting a ball at the floor (jump pop): a GROUND setup touch
                if new_touch {
                    *c = Chain::start(GROUND, true, takeoff_ok);
                    reward += self.count_touch(tr, c, s, touch_tick);
                } else if c.open {
                    c.last_touch_tick = touch_tick;
                }
            } else if new_touch {
                let extends = c.open && (c.setup_pending || c.air_touches > 0);
                let low = ball.pos.z < self.min_air_touch_height;
                if extends && low {
                    // A low touch is a fresh pop: GROUND touch 1
                    *c = Chain::start(GROUND, true, takeoff_ok);
                    reward += self.count_touch(tr, c, s, touch_tick);
                } else if extends && !self.useful_touch(s) {
                    c.close(); // E.g. riding it down: not an air dribble touch
                } else if extends {
                    if c.air_touches == 0 {
                        c.boost_ok = c.boost_ok && takeoff_ok; // Went for it
                    }
                    c.setup_pending = false;
                    c.air_touches += 1;
                    reward += self.count_touch(tr, c, s, touch_tick);
                } else if low {
                    // A low air touch with no chain: a pop (needs a real air touch next)
                    *c = Chain::start(GROUND, true, takeoff_ok);
                    reward += self.count_touch(tr, c, s, touch_tick);
                } else {
                    // The first touch of a chain that starts in the air
                    let origin = if tr.on_wall_last {
                        WALL
                    } else if ball.pos.z >= self.aerial_catch_height
                        && since(tr.left_surface_tick) >= self.aerial_min_air_time
                    {
                        AERIAL
                    } else {
                        GROUND
                    };
                    *c = Chain::start(origin, false, takeoff_ok);
                    c.air_touches = 1;
                    reward += self.count_touch(tr, c, s, touch_tick);
                    if origin == AERIAL && c.boost_ok {
                        let soft =
                            clamp01(1.0 - (ball.vel - p.body.vel).length() / self.catch_soft_speed);
                        reward += self.pay_shaping(tr, self.catch_bonus * soft);
                    }
                }
            } else if c.open {
                c.last_touch_tick = touch_tick; // A continuing contact (sticky carry): the same touch
            }
        }

        // ---- Goal ----
        if s.cur.goal_scored.is_some()
            && s.prev.goal_scored.is_none()
            && c.touches > 0
            && !c.goal_paid
        {
            let into_opponent = ball.pos.y * attack_sign(p.team) > 0.0;
            c.goal_paid = true;
            if !into_opponent {
                // Own net: penalised whenever the chain touched it recently, even if control was already lost
                if since(c.last_touch_tick) <= self.goal_window && self.own_goal_penalty > 0.0 {
                    reward -= self.own_goal_penalty;
                }
            } else if c.eligible
                && (self.allow_floor_before_goal || !c.floor_since_touch || c.shot_taken)
                && c.valid
                && c.boost_ok
                && (!self.require_goal_mouth || self.in_goal_mouth(ball.pos))
            {
                let extra = (c.touches - self.min_touches(c.origin)).min(self.max_extra_touches);
                let mut g = self.goal_reward * (1.0 + self.per_extra_touch * extra.max(0) as f32);
                let rolled = if c.resets > 0 {
                    1.0
                } else if c.air_time > 0.0 {
                    clamp01(c.roll_time / c.air_time)
                } else {
                    0.0
                };
                g *= 1.0 + self.air_roll_goal_bonus * rolled;
                let progress = self.progress(p.team, c.start_ball_pos, ball.pos);
                g *= 1.0
                    + self.cross_map_bonus
                        * ramp(progress, self.cross_map_start, self.cross_map_full);
                let counted = if c.resets < self.min_resets_for_bonus {
                    0
                } else {
                    c.resets.min(self.max_resets_for_bonus)
                };
                g *= 1.0 + self.reset_goal_bonus * counted as f32;
                reward += g;
            }
            c.close();
            c.eligible = false;
        }
        reward
    }
}

impl Reward for AirDribble {
    fn memory(&self) -> usize {
        Clock::SIZE + Track::SIZE + Chain::SIZE
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
        let (track_mem, chain_mem) = rest.split_at_mut(Track::SIZE);
        let mut clock = Clock::load(clock_mem);
        let (mut tr, mut chain) = if clock.advance(events) {
            // The episode's start: from the starting state
            let start = &prev.cars[car];
            let track = Track {
                max_gap_since_contact: 1e9,
                was_on_surface: start.on_ground,
                on_wall_last: start.on_ground && start.body.up().z < self.wall_up_z,
                left_surface_tick: 0.0,
                boost_at_takeoff: start.boost,
                ..Track::default()
            };
            (track, Chain::fresh())
        } else {
            (Track::load(track_mem), Chain::load(chain_mem))
        };
        let ctx = Ctx {
            prev,
            cur,
            car,
            dt: step_seconds(events),
            detail: events.detail(car),
            goal_dir: self.goal_dir(cur.cars[car].team, cur.ball.pos),
            downfield: Vec3A::new(0.0, attack_sign(cur.cars[car].team), 0.0),
        };
        tr.extended = false;
        let reward = self.update(&mut tr, &mut chain, &ctx, events, clock.now)
            + self.positioning(&mut tr, &chain, &ctx);
        clock.store(clock_mem);
        tr.store(track_mem);
        chain.store(chain_mem);
        finite_or_zero(reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        assert_listed("air_dribble", 77);
    }

    fn reward() -> AirDribble {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    /// A car (Blue, or Orange mirrored) in the air under the ball; the ball moving `ball_vel` (Blue's view).
    fn air(team: Team, car_pos: Vec3A, ball_pos: Vec3A, ball_vel: Vec3A) -> GameSnapshot {
        let m = |v: Vec3A| as_blue(team, v);
        let mut g = game(m(ball_pos), &[(team, m(car_pos))]);
        g.ball.vel = m(ball_vel);
        let c = &mut g.cars[0];
        c.on_ground = false;
        c.has_jumped = true;
        c.boost = 80.0;
        c.body.rot = if team == Team::Blue {
            Mat3A::IDENTITY
        } else {
            Mat3A::from_rotation_z(std::f32::consts::PI)
        };
        g
    }

    /// Steps a sequence of games through the reward, the car touching where `touch` says; the rewards.
    fn run(r: &AirDribble, games: &[GameSnapshot], touch: &[bool], memory: &mut [f32]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut tick = 1000;
        for (i, pair) in games.windows(2).enumerate() {
            tick += 8;
            let mut ev = quiet(1);
            ev.tick = tick;
            if touch[i] {
                ev = touched(ev, 0);
                ev.details[0].last_hit_tick = Some(tick - 3);
            }
            out.push(r.value_remembering(&pair[0], &pair[1], &ev, 0, memory));
        }
        out
    }

    /// Ground pop, two lifting air touches, then the ball into the net: a valid GROUND chain scores.
    fn dribble_and_score(team: Team) -> (Vec<f32>, Vec<f32>) {
        dribble_and_score_with(&reward(), team)
    }

    fn dribble_and_score_with(r: &AirDribble, team: Team) -> (Vec<f32>, Vec<f32>) {
        let mut memory = vec![0.0; r.memory()];
        let mut games = Vec::new();
        // On the floor with the ball just above (a pop)
        let mut start = air(
            team,
            Vec3A::new(0.0, -2000.0, 17.0),
            Vec3A::new(0.0, -1950.0, 200.0),
            Vec3A::ZERO,
        );
        start.cars[0].on_ground = true;
        start.cars[0].has_jumped = false;
        games.push(start.clone());
        let mut pop = start.clone();
        pop.ball.vel = as_blue(team, Vec3A::new(0.0, 300.0, 900.0));
        games.push(pop);
        // In the air: a quiet step, then a lift touch, quiet, another lift touch
        let mut ball_vz = 900.0;
        for (i, z) in [400.0, 550.0, 650.0, 800.0, 900.0].iter().enumerate() {
            let lift = i == 1 || i == 3;
            if lift {
                ball_vz += 400.0;
            } else {
                ball_vz -= 43.0;
            }
            let y = -1800.0 + 150.0 * i as f32;
            games.push(air(
                team,
                Vec3A::new(0.0, y - 120.0, z - 130.0),
                Vec3A::new(0.0, y, *z),
                Vec3A::new(0.0, 900.0, ball_vz),
            ));
        }
        let mut goal = air(
            team,
            Vec3A::new(0.0, 4800.0, 200.0),
            Vec3A::new(0.0, 5200.0, 300.0),
            Vec3A::new(0.0, 2000.0, 0.0),
        );
        goal.goal_scored = Some(team);
        games.push(goal);
        // pop touch (on the floor), quiet, lift touch, quiet, lift touch, quiet, goal
        let touches = [true, false, true, false, true, false, false];
        let out = run(r, &games, &touches, &mut memory);
        (out, memory)
    }

    /// `dribble_and_score_with` with a flip reset on the quiet step between the two lift touches; the goal's pay.
    fn reset_goal_pay(r: &AirDribble) -> f32 {
        let (plain, _) = dribble_and_score_with(r, Team::Blue);
        let mut memory = vec![0.0; r.memory()];
        let mut games = Vec::new();
        let mut start = air(
            Team::Blue,
            Vec3A::new(0.0, -2000.0, 17.0),
            Vec3A::new(0.0, -1950.0, 200.0),
            Vec3A::ZERO,
        );
        start.cars[0].on_ground = true;
        start.cars[0].has_jumped = false;
        games.push(start.clone());
        let mut pop = start.clone();
        pop.ball.vel = Vec3A::new(0.0, 300.0, 900.0);
        games.push(pop);
        let mut ball_vz = 900.0;
        for (i, z) in [400.0, 550.0, 650.0, 800.0, 900.0].iter().enumerate() {
            let lift = i == 1 || i == 3;
            ball_vz += if lift { 400.0 } else { -43.0 };
            let y = -1800.0 + 150.0 * i as f32;
            let mut g = air(
                Team::Blue,
                Vec3A::new(0.0, y - 120.0, z - 130.0),
                Vec3A::new(0.0, y, *z),
                Vec3A::new(0.0, 900.0, ball_vz),
            );
            let c = &mut g.cars[0];
            c.can_flip = i > 2; // flip used until the reset on the third step gives it back
            if i == 2 {
                (c.has_jumped, c.has_flipped, c.has_double_jumped) = (false, false, false);
            }
            games.push(g);
        }
        let mut goal = air(
            Team::Blue,
            Vec3A::new(0.0, 4800.0, 200.0),
            Vec3A::new(0.0, 5200.0, 300.0),
            Vec3A::new(0.0, 2000.0, 0.0),
        );
        goal.goal_scored = Some(Team::Blue);
        games.push(goal);
        let touches = [true, false, true, false, true, false, false];
        let out = run(r, &games, &touches, &mut memory);
        assert!(
            *plain.last().unwrap() > 0.0,
            "the plain play scores: {plain:?}"
        );
        *out.last().unwrap()
    }

    /// min_resets_for_bonus 2 leaves the reset out of reset_goal_bonus for a goal off one reset (default 0:
    /// it adds reset_goal_bonus, 1.5 times the pay at the default 0.5).
    #[test]
    fn min_resets_for_bonus_leaves_a_one_reset_goal_plain() {
        let with = reset_goal_pay(&reward());
        let two = AirDribble {
            min_resets_for_bonus: 2,
            ..reward()
        };
        let off = reset_goal_pay(&two);
        // The goal step also carries a little shaping, so about 1.5 times
        let ratio = with / off;
        assert!(
            off > 0.0 && (1.45..1.55).contains(&ratio),
            "{with} vs {off}"
        );
    }

    /// touch_lift_weight pays each paid chain touch for the upward speed it added: the two lift touches of
    /// `dribble_and_score` added 443 uu/s each (400 plus the step's gravity), so 0.3 x 443 / 600 more each; off by
    /// default.
    #[test]
    fn touch_lift_weight_pays_a_touch_that_pops_the_ball_up() {
        let roomy = |w: f32| AirDribble {
            touch_lift_weight: w,
            max_shaping_per_episode: 10.0,
            ..reward()
        };
        let shaping = |r: &AirDribble| {
            let (out, _) = dribble_and_score_with(r, Team::Blue);
            out[..out.len() - 1].iter().sum::<f32>()
        };
        let extra = shaping(&roomy(0.3)) - shaping(&roomy(0.0));
        let lift = 400.0 + 650.0 * 8.0 / 120.0;
        assert!((extra - 2.0 * 0.3 * lift / 600.0).abs() < 1e-3, "{extra}");
        assert_eq!(reward().touch_lift_weight, 0.0);
    }

    /// min_touch_interval: `dribble_and_score`'s touches come every 2 steps (16 ticks, 0.13 s); with 0.3 s
    /// between new touches some count as the same touch and pay nothing, with 0.1 s all count (off by default).
    #[test]
    fn min_touch_interval_counts_quick_touches_as_one() {
        let with = |interval: f32| AirDribble {
            min_touch_interval: interval,
            max_shaping_per_episode: 10.0,
            ..reward()
        };
        let shaping = |r: &AirDribble| {
            let (out, _) = dribble_and_score_with(r, Team::Blue);
            out[..out.len() - 1].iter().sum::<f32>()
        };
        let (off, quick, slow) = (
            shaping(&with(0.0)),
            shaping(&with(0.1)),
            shaping(&with(0.3)),
        );
        assert!((quick - off).abs() < 1e-6, "{quick} vs {off}");
        assert!(
            slow < off - 0.5 * reward().touch_reward * 0.5,
            "{slow} vs {off}"
        );
        assert_eq!(reward().min_touch_interval, 0.0);
    }

    /// touch_push_away_weight pays a paid chain touch for the ball's speed gained relative to the car, up and
    /// toward the goal: `dribble_and_score`'s two lift touches each add 400 uu/s straight up (the car's speed
    /// unchanged), 400 x cos 45 = 283 along that line; the plain touches lose height and pay nothing. Off by default.
    #[test]
    fn touch_push_away_weight_pays_the_ball_sent_up_and_on_away_from_the_car() {
        let roomy = |w: f32| AirDribble {
            touch_push_away_weight: w,
            max_shaping_per_episode: 10.0,
            ..reward()
        };
        let shaping = |r: &AirDribble| {
            let (out, _) = dribble_and_score_with(r, Team::Blue);
            out[..out.len() - 1].iter().sum::<f32>()
        };
        let extra = shaping(&roomy(0.5)) - shaping(&roomy(0.0));
        let along = 400.0 * std::f32::consts::FRAC_1_SQRT_2;
        assert!((extra - 2.0 * 0.5 * along / 600.0).abs() < 1e-3, "{extra}");
        let (orange, _) = dribble_and_score_with(&roomy(0.5), Team::Orange);
        let (blue, _) = dribble_and_score_with(&roomy(0.5), Team::Blue);
        assert!(
            (orange.iter().sum::<f32>() - blue.iter().sum::<f32>()).abs() < 1e-3,
            "mirrored"
        );
        assert_eq!(reward().touch_push_away_weight, 0.0);
    }

    /// touch_under_weight pays for how far under the ball the car is: the two paid touches of
    /// `dribble_and_score` have the car 120 behind and 130 below the ball (cosine 0.735 from straight up), so
    /// 0.45 of the weight each; directly under pays in full; off by default.
    #[test]
    fn touch_under_weight_pays_a_touch_from_under_the_ball() {
        let roomy = |w: f32| AirDribble {
            touch_under_weight: w,
            max_shaping_per_episode: 10.0,
            ..reward()
        };
        let shaping = |r: &AirDribble| {
            let (out, _) = dribble_and_score_with(r, Team::Blue);
            out[..out.len() - 1].iter().sum::<f32>()
        };
        let extra = shaping(&roomy(0.5)) - shaping(&roomy(0.0));
        let cos = 130.0 / 120.0f32.hypot(130.0);
        assert!(
            (extra - 2.0 * 0.5 * (cos - 0.6) / 0.3).abs() < 1e-3,
            "{extra}"
        );
        let car = |pos: Vec3A| CarSnapshot {
            body: Body {
                pos,
                ..Body::RESTING
            },
            ..CarSnapshot::default()
        };
        let r = roomy(1.0);
        let ball = Vec3A::new(0.0, 0.0, 1000.0);
        assert_eq!(r.under(&car(Vec3A::new(0.0, 0.0, 850.0)), ball), 1.0);
        assert_eq!(r.under(&car(Vec3A::new(0.0, -150.0, 950.0)), ball), 0.0);
        assert_eq!(reward().touch_under_weight, 0.0);
    }

    #[test]
    fn a_valid_air_dribble_that_scores_pays_the_goal_and_orange_the_same() {
        let (blue, _) = dribble_and_score(Team::Blue);
        let goal = *blue.last().unwrap();
        assert!(goal >= 2.0, "the goal pays at least goal_reward: {blue:?}");
        let shaping: f32 = blue[..blue.len() - 1].iter().sum();
        assert!(shaping > 0.0 && shaping <= 1.5, "capped shaping: {blue:?}");
        let (orange, _) = dribble_and_score(Team::Orange);
        for (b, o) in blue.iter().zip(&orange) {
            assert!((b - o).abs() < 1e-4, "mirrored: {blue:?} vs {orange:?}");
        }
    }

    #[test]
    fn nothing_without_touches_and_a_fresh_episode_starts_again() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let a = air(
            Team::Blue,
            Vec3A::new(0.0, 0.0, 500.0),
            Vec3A::new(0.0, 100.0, 600.0),
            Vec3A::ZERO,
        );
        let mut b = a.clone();
        b.goal_scored = Some(Team::Blue);
        b.ball.pos.y = 5200.0;
        let out = run(&r, &[a.clone(), a.clone(), b], &[false, false], &mut memory);
        assert_eq!(out, vec![0.0, 0.0], "a goal without a chain pays nothing");
        let (_, mut used) = dribble_and_score(Team::Blue);
        assert!(used.iter().any(|&v| v != 0.0));
        used.fill(0.0);
        let again = run(&r, &[a.clone(), a], &[false], &mut used);
        assert_eq!(again, vec![0.0]);
        assert_eq!(Clock::load(&used).now, 8.0, "the clock starts again");
        assert_eq!(Chain::load(&used[Clock::SIZE + Track::SIZE..]).touches, 0);
    }

    #[test]
    fn a_touch_that_rides_the_ball_down_ends_the_chain_and_defaults_load() {
        let r = reward();
        assert_eq!(r.min_touches_ground, 2);
        assert!((r.floor_contact_height - 112.75).abs() < 1e-4);
        let mut memory = vec![0.0; r.memory()];
        let first = air(
            Team::Blue,
            Vec3A::new(0.0, 0.0, 500.0),
            Vec3A::new(0.0, 100.0, 650.0),
            Vec3A::new(0.0, 0.0, 100.0),
        );
        let mut down = first.clone();
        down.ball.vel.z = -600.0; // touched from above: the ball driven down
        let mut quiet_step = first.clone();
        quiet_step.ball.vel.z = -650.0;
        let out = run(
            &r,
            &[first.clone(), first.clone(), quiet_step, down],
            &[true, false, true],
            &mut memory,
        );
        assert!(
            out[2] < 0.5 * r.touch_reward,
            "no touch pay for a ride-down: {out:?}"
        );
        let chain = Chain::load(&memory[Clock::SIZE + Track::SIZE..]);
        assert!(!chain.open && chain.touches == 1 && chain.origin == GROUND);
    }

    /// Positioning: a car falling with the ball (not boosting against gravity) for longer than the grace holds no
    /// position, however well placed under the ball; boosting against gravity brings it back (C++
    /// positioning_zero_while_falling_with_the_ball).
    #[test]
    fn positioning_is_zero_once_the_car_has_fallen_for_longer_than_the_grace() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let dt = 8.0 / 120.0;
        let fall = Vec3A::new(0.0, 0.0, -400.0);
        let mut cur = air(
            Team::Blue,
            Vec3A::new(0.0, 0.0, 800.0),
            Vec3A::new(0.0, 100.0, 950.0),
            fall,
        );
        cur.cars[0].body.vel = fall;
        r.value_remembering(&cur, &cur, &quiet(1), 0, &mut memory); // The episode's first step
        let falling_at = |g: f32| {
            let mut prev = cur.clone();
            prev.cars[0].body.vel.z = fall.z + g * 650.0 * dt;
            prev
        };
        let step = |prev: &GameSnapshot, memory: &mut [f32]| {
            // A chain kept open (synthetic), its last touch just now
            let now = Clock::load(memory).now;
            let chain = Chain {
                last_touch_tick: now,
                touches: 2,
                ..Chain::start(GROUND, false, true)
            };
            chain.store(&mut memory[Clock::SIZE + Track::SIZE..]);
            r.value_remembering(prev, &cur, &quiet(1), 0, memory);
            Track::load(&memory[Clock::SIZE..]).position_phi
        };
        let mut steps = 0;
        let mut phi = step(&falling_at(0.9), &mut memory);
        assert!(phi > 0.5, "placed under the ball: {phi}");
        while phi > 0.0 && steps < 30 {
            steps += 1;
            phi = step(&falling_at(0.9), &mut memory);
        }
        let fell = (steps + 1) as f32 * dt;
        assert!(
            phi == 0.0 && fell > r.position_fall_grace && fell < r.position_fall_grace + 0.2,
            "phi {phi} after {fell} s of falling at 0.9 g"
        );
        let boosting = step(&falling_at(0.3), &mut memory);
        assert!(boosting > 0.5, "only 0.3 g down: {boosting}");
        assert_eq!(Track::load(&memory[Clock::SIZE..]).fall_time, 0.0);
    }

    /// One step of a chain set into memory after the episode's first step (its last touch just now): the reward.
    fn chain_step(r: &AirDribble, chain: Chain, prev: &GameSnapshot, cur: &GameSnapshot) -> f32 {
        chain_step_to(r, chain, prev, cur).0
    }

    /// `chain_step`, and the chain after it.
    fn chain_step_to(
        r: &AirDribble,
        chain: Chain,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
    ) -> (f32, Chain) {
        let mut memory = vec![0.0; r.memory()];
        r.value_remembering(prev, prev, &quiet(1), 0, &mut memory);
        let now = Clock::load(&memory).now;
        let chain = Chain {
            last_touch_tick: now,
            ..chain
        };
        chain.store(&mut memory[Clock::SIZE + Track::SIZE..]);
        let reward = r.value_remembering(prev, cur, &quiet(1), 0, &mut memory);
        (reward, Chain::load(&memory[Clock::SIZE + Track::SIZE..]))
    }

    /// aim_at_goal pays going towards the goal, not just downfield: carrying the ball at the same speed
    /// along the side wall or diagonally toward the net. aim_at_goal pays the diagonal more; off, the progress
    /// pay is the downfield one.
    #[test]
    fn aim_at_goal_pays_a_carry_toward_the_net_more_than_one_along_the_side_wall() {
        let speed = 1500.0;
        let ball_pos = Vec3A::new(3300.0, 0.0, 800.0);
        let to_goal = Vec3A::new(-ball_pos.x, GOAL_LINE_Y - ball_pos.y, 0.0).normalize();
        let along = Vec3A::new(0.0, speed, 0.0);
        let diagonal = to_goal * speed;
        let carry = |r: &AirDribble, team: Team, vel: Vec3A| {
            let mut g = air(team, ball_pos - Vec3A::new(0.0, 0.0, 150.0), ball_pos, vel);
            g.cars[0].body.vel = as_blue(team, vel);
            let chain = Chain {
                touches: 2,
                air_touches: 2,
                ..Chain::start(GROUND, false, true)
            };
            chain_step(r, chain, &g, &g)
        };
        let aimed = AirDribble {
            aim_at_goal: true,
            ..reward()
        };
        let downfield = reward();
        assert!(!downfield.aim_at_goal, "off by default");
        for team in [Team::Blue, Team::Orange] {
            let (a, d) = (carry(&aimed, team, along), carry(&aimed, team, diagonal));
            assert!(d > a, "{team:?} aimed: along {a}, diagonal {d}");
            // Only the progress pay differs: the ball's speed toward the goal centre, over 2300, a second
            let dt = 8.0 / 120.0;
            let pay = |v: Vec3A| aimed.progress_rate * v.dot(to_goal) / 2300.0 * dt;
            assert!(((d - a) - (pay(diagonal) - pay(along))).abs() < 1e-5);
            // Off: downfield speed
            let (a, d) = (
                carry(&downfield, team, along),
                carry(&downfield, team, diagonal),
            );
            let pay = |v: Vec3A| downfield.progress_rate * v.y / 2300.0 * dt;
            assert!(a > d, "{team:?} downfield: along {a}, diagonal {d}");
            assert!(((a - d) - (pay(along) - pay(diagonal))).abs() < 1e-5);
        }
        // Past the goal line it aims along the line, and at the goal's centre straight downfield
        let past = aimed.goal_dir(Team::Blue, Vec3A::new(500.0, 5300.0, 300.0));
        assert_eq!(past, Vec3A::new(-1.0, 0.0, 0.0));
        let centre = aimed.goal_dir(Team::Orange, Vec3A::new(0.0, -5120.0, 300.0));
        assert_eq!(centre, Vec3A::new(0.0, -1.0, 0.0));
        // On the centre line the two agree
        let (on, _) = dribble_and_score_with(&aimed, Team::Blue);
        let (off, _) = dribble_and_score(Team::Blue);
        for (a, b) in on.iter().zip(&off) {
            assert!((a - b).abs() < 1e-4, "{on:?} vs {off:?}");
        }
    }

    /// aim_at_goal changes only the pay terms. Whether the car is in front of the ball (which voids the
    /// chain) stays downfield: near the net the direction to the goal's centre turns sideways, and a car level
    /// with the ball on its middle side would otherwise count as in front of it.
    #[test]
    fn aim_at_goal_does_not_void_a_chain_for_a_car_beside_the_ball_near_the_net() {
        let aimed = AirDribble {
            aim_at_goal: true,
            ..reward()
        };
        let beside = |team: Team, car_x: f32, car_y: f32| {
            let ball = Vec3A::new(600.0, 4700.0, 400.0);
            let car = Vec3A::new(car_x, car_y, 300.0);
            let g = air(team, car, ball, Vec3A::new(0.0, 300.0, 0.0));
            let chain = Chain {
                touches: 2,
                air_touches: 2,
                ..Chain::start(GROUND, false, true)
            };
            chain_step_to(&aimed, chain, &g, &g).1
        };
        for team in [Team::Blue, Team::Orange] {
            // 100 to the middle side, level downfield: 82 "toward the goal centre", over the 50 margin
            let level = beside(team, 500.0, 4700.0);
            assert!(level.open && level.eligible, "{team:?}: beside the ball");
            // 100 goal-side of it downfield: in front
            let ahead = beside(team, 600.0, 4800.0);
            assert!(!ahead.open && !ahead.eligible, "{team:?}: in front");
        }
    }

    /// With aim_at_goal the cross-map bonus is how much closer to the goal's centre the ball got over the
    /// chain (a ball in the net counts as on its line); off, its progress downfield.
    #[test]
    fn aim_at_goal_measures_the_cross_map_bonus_toward_the_goal_centre() {
        let goal_pay = |r: &AirDribble, team: Team, start: Vec3A| {
            let m = |v: Vec3A| as_blue(team, v);
            let mut g = air(
                team,
                Vec3A::new(0.0, 4800.0, 200.0),
                Vec3A::new(0.0, 5200.0, 300.0),
                Vec3A::new(0.0, 2000.0, 0.0),
            );
            let prev = g.clone();
            g.goal_scored = Some(team);
            let chain = Chain {
                open: false,
                valid: true,
                touches: r.min_touches_ground,
                air_touches: 2,
                start_ball_pos: m(start),
                ..Chain::start(GROUND, false, true)
            };
            chain_step(r, chain, &prev, &g)
        };
        let aimed = AirDribble {
            aim_at_goal: true,
            ..reward()
        };
        let downfield = reward();
        let paid = |r: &AirDribble, progress: f32| {
            r.goal_reward * (1.0 + r.cross_map_bonus * ramp(progress, 1500.0, 6000.0))
        };
        // From the side wall's middle: 6,103 closer to the goal centre, 5,200 downfield
        let side = Vec3A::new(3500.0, 0.0, 600.0);
        let closer = 3500.0f32.hypot(GOAL_LINE_Y);
        for team in [Team::Blue, Team::Orange] {
            let (on, off) = (
                goal_pay(&aimed, team, side),
                goal_pay(&downfield, team, side),
            );
            assert!((on - paid(&aimed, closer)).abs() < 1e-3, "{team:?}: {on}");
            assert!(
                (off - paid(&downfield, 5200.0)).abs() < 1e-3,
                "{team:?}: {off}"
            );
            assert!(on > off);
        }
    }

    #[test]
    fn the_positioning_is_kept_only_by_touches_that_extend_the_chain() {
        let on = reward();
        let off = AirDribble {
            position_weight: 0.0,
            ..reward()
        };
        // The scoring air dribble: its two lifting touches keep the approach to them
        let total = |r: &AirDribble| {
            let (out, _) = dribble_and_score_with(r, Team::Blue);
            out.iter().sum::<f32>()
        };
        let kept = total(&on) - total(&off);
        assert!(
            kept > 0.0 && kept <= 2.0 * on.position_weight + 1e-4,
            "{kept}"
        );
        // A pop, then hovering under the ball and landing without another touch: paid back (only the
        // discount's cost is left)
        let hover = |r: &AirDribble| {
            let mut memory = vec![0.0; r.memory()];
            let mut start = air(
                Team::Blue,
                Vec3A::new(0.0, -2000.0, 17.0),
                Vec3A::new(0.0, -1950.0, 200.0),
                Vec3A::ZERO,
            );
            start.cars[0].on_ground = true;
            let mut games = vec![start.clone()];
            for i in 0..4 {
                let z = 400.0 + 50.0 * i as f32;
                games.push(air(
                    Team::Blue,
                    Vec3A::new(0.0, -1990.0, z - 130.0),
                    Vec3A::new(0.0, -1900.0, z),
                    Vec3A::new(0.0, 0.0, 300.0),
                ));
            }
            let mut landed = start.clone();
            landed.ball.pos = Vec3A::new(0.0, -1000.0, 800.0);
            games.push(landed);
            let touches = [true, false, false, false, false, false];
            let out = run(r, &games, &touches, &mut memory);
            (
                out,
                Track::load(&memory[Clock::SIZE..Clock::SIZE + Track::SIZE]),
            )
        };
        let (with, tr) = hover(&on);
        let (without, _) = hover(&off);
        let left: f32 = with.iter().sum::<f32>() - without.iter().sum::<f32>();
        assert!(
            with.iter().zip(&without).any(|(a, b)| a > b),
            "paid on the way: {with:?}"
        );
        assert!(left <= 1e-5 && left > -0.02, "paid back on landing: {left}");
        assert_eq!(tr.position_phi, 0.0);
    }
}
