//! Flip resets off the ball in the air: the approach, the reset itself, staying with the ball, using it,
//! and a goal off a used reset.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::aerial_kit::*;
use crate::custom::library::flick_kit::{air_path_to_goal_line, project_to_goal_line};

/// Flip resets: a reset pays reset_reward (10) times its style (flip or spin into it, multi-reset, need), up to
/// follow_reward (3) for staying with the ball, use_reward (8) for using it well, goal_reward (20) for scoring off
/// it; the approach pays at most 4.5 an episode, for progress only.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlipReset {
    /// Paid for raising the setup potential from 0 to 1 (a whole approach).
    #[serde(default = "setup_scale")]
    pub setup_scale: f32,
    /// Car-ball centre distance beyond which the setup potential is 0...
    #[serde(default = "setup_max_dist")]
    pub setup_max_dist: f32,
    /// ...and below which closeness is full.
    #[serde(default = "setup_full_dist")]
    pub setup_full_dist: f32,
    /// The closeness curve's power: x0.02 at 1500 uu, x0.17 at 1000, x0.5 at 600, x0.85 at 300.
    #[serde(default = "setup_close_exponent")]
    pub setup_close_exponent: f32,
    /// Up close with the flip still unused (a wheel touch now wouldn't be a reset): potential times this.
    #[serde(default = "setup_flip_ready_factor")]
    pub setup_flip_ready_factor: f32,
    /// Up close, the car centre this far above the ball centre: potential 0...
    #[serde(default = "max_above_ball")]
    pub max_above_ball: f32,
    /// ...plus this much more allowed far away.
    #[serde(default = "setup_above_looseness")]
    pub setup_above_looseness: f32,
    /// The most setup pay per episode (below one reset).
    #[serde(default = "max_setup_per_episode")]
    pub max_setup_per_episode: f32,
    /// Ball centre height for the setup and resets to count.
    #[serde(default = "min_ball_height")]
    pub min_ball_height: f32,
    /// Seconds off the floor, walls and ceiling for a reset to count, 0 = off.
    #[serde(default = "min_air_time")]
    pub min_air_time: f32,
    /// Paid for a reset, times its style.
    #[serde(default = "reset_reward")]
    pub reset_reward: f32,
    /// Car centre to ball centre right after the reset.
    #[serde(default = "max_reset_ball_dist")]
    pub max_reset_ball_dist: f32,
    /// The most resets paid per episode.
    #[serde(default = "max_resets_per_episode")]
    pub max_resets_per_episode: i32,
    /// Seconds since the last reset contact (counted or not) for a reset to count.
    #[serde(default = "min_reset_interval")]
    pub min_reset_interval: f32,
    /// Ball progress toward the opponent goal since the previous reset for full credit on a later one (uu): a later
    /// reset's style, and so its own pay, follow-through and use, is times this progress (0 to 1).
    #[serde(default = "multi_reset_full_progress")]
    pub multi_reset_full_progress: f32,
    /// ...and below which (uu) a later reset gets no credit at all: a reset in place pays nothing (0: credit from
    /// the first unit of progress).
    #[serde(default = "multi_reset_min_progress")]
    pub multi_reset_min_progress: f32,
    /// Style: the wheels hit the ball during a front or side dodge, x (1 + this).
    #[serde(default = "flip_into_reset_bonus")]
    pub flip_into_reset_bonus: f32,
    /// Style: ...during a backflip, x (1 + this).
    #[serde(default = "backflip_into_reset_bonus")]
    pub backflip_into_reset_bonus: f32,
    /// Seconds since the dodge started for it to count as a flip into the reset.
    #[serde(default = "flip_into_reset_window")]
    pub flip_into_reset_window: f32,
    /// Style: spinning about the roll axis at the reset, x (1 + this) at spin_full_rate.
    #[serde(default = "spin_reset_bonus")]
    pub spin_reset_bonus: f32,
    /// Roll rate (rad/s) where the spin bonus starts...
    #[serde(default = "spin_min_rate")]
    pub spin_min_rate: f32,
    /// ...and where it is full.
    #[serde(default = "spin_full_rate")]
    pub spin_full_rate: f32,
    /// The n-th reset of the episode: x (1 + this x (n - 1) x (0.5 + need))...
    #[serde(default = "multi_reset_step")]
    pub multi_reset_step: f32,
    /// ...up to this.
    #[serde(default = "max_multi_reset_mult")]
    pub max_multi_reset_mult: f32,
    /// x (1 + this x need): low boost or far from goal makes a reset worth more.
    #[serde(default = "need_bonus")]
    pub need_bonus: f32,
    /// Distance to the opponent goal (uu) at which the distance half of need is full.
    #[serde(default = "need_full_dist")]
    pub need_full_dist: f32,
    /// Paid for following the reset through, times its style.
    #[serde(default = "follow_reward")]
    pub follow_reward: f32,
    /// Seconds after the reset the car must stay useful.
    #[serde(default = "follow_time")]
    pub follow_time: f32,
    /// ...within this distance of the ball.
    #[serde(default = "max_follow_distance")]
    pub max_follow_distance: f32,
    /// Car centre above the ball centre by more than this: lost it.
    #[serde(default = "above_margin")]
    pub above_margin: f32,
    /// Car this far goal-side of the ball: lost it.
    #[serde(default = "ahead_margin")]
    pub ahead_margin: f32,
    /// Paid for using the reset, times the use's quality and the style.
    #[serde(default = "use_reward")]
    pub use_reward: f32,
    /// Seconds after the reset to use it.
    #[serde(default = "use_window")]
    pub use_window: f32,
    /// A dodge into the ball at all is worth this much of a full use...
    #[serde(default = "use_flip_base")]
    pub use_flip_base: f32,
    /// ...full when it adds this much ball speed toward goal (uu/s).
    #[serde(default = "use_full_gain")]
    pub use_full_gain: f32,
    /// Another reset is a use worth this much.
    #[serde(default = "use_reset_quality")]
    pub use_reset_quality: f32,
    /// A touch without the dodge is worth this much, times how much the ball heads to goal.
    #[serde(default = "use_touch_quality")]
    pub use_touch_quality: f32,
    /// Seconds after the dodge starts in which it must touch the ball (the whole flip).
    #[serde(default = "dodge_hit_window")]
    pub dodge_hit_window: f32,
    /// Ball centre at least this high when the reset is used, 0 = off.
    #[serde(default = "min_use_ball_height")]
    pub min_use_ball_height: f32,
    /// A delayed reset: seconds from the reset to the dodge...
    #[serde(default = "delayed_setup_time")]
    pub delayed_setup_time: f32,
    /// ...and from the dodge to the ball contact.
    #[serde(default = "delayed_hit_time")]
    pub delayed_hit_time: f32,
    /// A delayed reset's use quality x (1 + this).
    #[serde(default = "delayed_use_bonus")]
    pub delayed_use_bonus: f32,
    /// Car-ball centre distance that means the car came off the ball after the reset.
    #[serde(default = "separation_dist")]
    pub separation_dist: f32,
    /// Paid for a goal off a used reset.
    #[serde(default = "goal_reward")]
    pub goal_reward: f32,
    /// Seconds from the car's last touch after the use to the goal.
    #[serde(default = "goal_window")]
    pub goal_window: f32,
    /// Training only: a car that starts the game in the air holding a flip reset (no jump or flip used, away
    /// from every surface, within 600 units of a ball at least min_ball_height high, at most 2 s after its last
    /// wheel contact) has a counted reset, so a start part-way through a move can pay its use (false: such a start
    /// holds nothing).
    #[serde(default = "start_reset_counts")]
    pub start_reset_counts: bool,
    /// A flip reset also counts when RocketSim gives the flip back a step after the wheels reach the ball, while
    /// they are still on it (the one-step test misses about a third of real resets at tick skip 4); false: the
    /// one-step test only.
    #[serde(default)]
    pub two_step_resets: bool,
    /// A dodge into the ball with the reset pays its use times how well the hit sends the ball at the opponents'
    /// net: 1 on target, 0 going sideways or away (false: aim does not matter). Plain touches and a second reset
    /// as a use are paid as without it.
    #[serde(default = "use_aim")]
    pub use_aim: bool,
    /// With use_aim: the hit counts as on target only if the ball's flight (gravity only) reaches the goal line
    /// inside the goal mouth before it comes down to the floor; one landing short earns less the shorter it falls,
    /// none 400 units short (false: a ball rolling or bouncing into the net counts too). For plays where the ball
    /// must not touch the floor.
    #[serde(default = "use_aim_reach")]
    pub use_aim_reach: bool,
    /// With use_aim_reach: a ball coming down this far short of the goal line (uu) aims at nothing, one nearer
    /// linearly more, up to 1 at the line (400; above it, a shot landing a little short still earns
    /// part of the use, so dodging stays worth trying while the bot learns to get in range).
    #[serde(default = "use_reach_short_zero")]
    pub use_reach_short_zero: f32,
    /// A dodge into the ball started sooner than delayed_setup_time after the reset pays its use times this
    /// (1: as any other; below 1 pays dodging at once less than waiting).
    #[serde(default = "quick_use_factor")]
    pub quick_use_factor: f32,
    /// A dodge with the reset that misses the ball costs this much (times the weight), and with use_aim on one
    /// that hits it off target this times (1 - the aim) (0: free).
    #[serde(default = "wasted_dodge_cost")]
    pub wasted_dodge_cost: f32,
    /// Multi resets: a dodge with the reset whose own contact is a new reset (the wheels brought onto the
    /// ball by the flip, a flip into the next reset) is that reset's use, paid as a multi-reset use when it counts,
    /// not judged as a shot (no wasted_dodge_cost for it) (false: judged as a shot like any dodge).
    #[serde(default = "flip_into_reset_free")]
    pub flip_into_reset_free: bool,
    /// The reset's own pay times (1 + this x how high the ball's centre is: 0 at min_ball_height, full at
    /// reset_height_full), for higher resets (0: off). The style, and so the follow and use pay, is unchanged.
    #[serde(default = "reset_height_bonus")]
    pub reset_height_bonus: f32,
    /// ...the ball's centre height (uu) where reset_height_bonus is full.
    #[serde(default = "reset_height_full")]
    pub reset_height_full: f32,
    /// A counted reset with the ball within this distance (uu, along the field's length) of the opponents' goal
    /// line is close and pays as without it; one farther out is far, and pays its own reward and its follow-through
    /// only with the ball's centre at least far_reset_min_height up, as does the approach to it (0: off, a reset
    /// pays the same anywhere). A far low reset still counts, for its use, a goal off it and the success rewards. A
    /// fifth of the field is 2048.
    #[serde(default = "close_reset_dist")]
    pub close_reset_dist: f32,
    /// With close_reset_dist: the lowest ball centre (uu) at which a far reset pays; for a far reset
    /// reset_height_bonus ramps from here (not min_ball_height) to reset_height_full.
    #[serde(default = "far_reset_min_height")]
    pub far_reset_min_height: f32,
    /// With close_reset_dist: delayed_use_bonus pays only for a dodge use that hits the ball within
    /// close_reset_dist of the goal line; farther out a delayed dodge pays as a plain one (false: anywhere).
    #[serde(default = "delayed_bonus_close_only")]
    pub delayed_bonus_close_only: bool,
    /// With close_reset_dist: a far reset high enough to pay is not paid when made but held, and paid when the next
    /// counted reset comes, times that one's credit for carrying the ball closer (multi_reset_min_progress to
    /// multi_reset_full_progress); lost if none follows. Its follow-through and the approach to a far ball pay
    /// nothing (false: a far high reset pays at once).
    #[serde(default = "far_reset_needs_next")]
    pub far_reset_needs_next: bool,
    /// A reset counts only after the car has touched the ball this many separate times (0.15 s apart, so a carry
    /// counts once) since it last left the floor, the touch off a wall included; a game starting with the car in
    /// the air counts 1, for the touch that put it there. A reset that does not count pays nothing, nor does the
    /// approach to it, and holds nothing to use (0: off). A reset held from the start is not affected.
    #[serde(default = "reset_min_air_touches")]
    pub reset_min_air_touches: i32,
    /// A reset counts only with the ball within this distance (uu, along the field's length) of the opponents'
    /// goal line; one farther out pays nothing, nor does the approach to it, and holds nothing to use (0: off). A
    /// third of the field is about 3,400.
    #[serde(default = "reset_max_goal_dist")]
    pub reset_max_goal_dist: f32,
    /// A reset counts only with the ball and the car each falling slower than this (uu/s; their vertical speed
    /// above minus this); falling faster, a shot or a musty off it hardly works. One that does not count pays
    /// nothing, nor does the approach to it, and holds nothing to use (0: off). A reset held from the start is not
    /// affected.
    #[serde(default = "reset_max_fall_speed")]
    pub reset_max_fall_speed: f32,
    /// The approach pays for closing on the ball too: the setup potential times the car's speed toward the ball
    /// (relative to the ball's, uu/s) over this, 0 when not closing, full from this (0: off). A reset comes easier
    /// with the car moving into the ball.
    #[serde(default = "setup_closing_full")]
    pub setup_closing_full: f32,
    /// A reset counts only as a firm touch: on the step before it the car was moving toward the ball at least this
    /// fast relative to the ball (uu/s), not grazing it (0: off). One that does not count pays nothing and holds
    /// nothing to use.
    #[serde(default = "reset_min_closing_speed")]
    pub reset_min_closing_speed: f32,
    /// The reset's own pay times (1 + this x how near the top of its arc the ball is: full with the ball rising or
    /// still, falling to 0 with it falling at reset_apex_zero_fall), for resets high in the ball's flight, where a
    /// shot or a musty still has time (0: off). The style, and so the follow and use pay, is unchanged.
    #[serde(default = "reset_apex_bonus")]
    pub reset_apex_bonus: f32,
    /// ...the ball's fall speed (uu/s) at which reset_apex_bonus is 0.
    #[serde(default = "reset_apex_zero_fall")]
    pub reset_apex_zero_fall: f32,
    /// A game's first counted reset far from the net (close_reset_dist) with the ball at least
    /// far_reset_min_height up pays up to this at once (not held by far_reset_needs_next), for a firm upward hit:
    /// times how fast the car moves into the ball (relative to it, on the step before) from far_firm_closing_zero
    /// (nothing) to far_firm_closing_full (all), times how little the car is falling, from far_firm_car_fall_zero
    /// (uu/s down: nothing) to rising or still (all) (0: off). That is the first reset a double reset is built on;
    /// a soft catch with car and ball falling together pays next to nothing.
    #[serde(default = "far_firm_reward")]
    pub far_firm_reward: f32,
    /// ...the car's fall speed (uu/s) at which far_firm_reward pays nothing (all with it rising or still).
    #[serde(default = "far_firm_car_fall_zero")]
    pub far_firm_car_fall_zero: f32,
    /// ...the car's speed into the ball (uu/s, relative to it) at which far_firm_reward pays nothing...
    #[serde(default = "far_firm_closing_zero")]
    pub far_firm_closing_zero: f32,
    /// ...and all of it.
    #[serde(default = "far_firm_closing_full")]
    pub far_firm_closing_full: f32,
    /// The approach to a game's first reset far out pays too (with far_reset_needs_next it pays nothing
    /// there; the ball at least far_reset_min_height up), times how little the car is falling (as far_firm_reward:
    /// nothing at far_firm_car_fall_zero, all rising or still), so it leads to the firm upward first reset that
    /// far_firm_reward pays (false: off).
    #[serde(default = "far_firm_setup")]
    pub far_firm_setup: bool,
}

defaults! { FlipReset {
    setup_scale: f32 = 3.0,
    setup_max_dist: f32 = 2000.0,
    setup_full_dist: f32 = 200.0,
    setup_close_exponent: f32 = 3.0,
    setup_flip_ready_factor: f32 = 0.4,
    max_above_ball: f32 = 100.0,
    setup_above_looseness: f32 = 600.0,
    max_setup_per_episode: f32 = 4.5,
    min_ball_height: f32 = 300.0,
    min_air_time: f32 = 0.0,
    reset_reward: f32 = 10.0,
    max_reset_ball_dist: f32 = 350.0,
    max_resets_per_episode: i32 = 5,
    min_reset_interval: f32 = 0.5,
    multi_reset_full_progress: f32 = 800.0,
    multi_reset_min_progress: f32 = 0.0,
    flip_into_reset_bonus: f32 = 0.4,
    backflip_into_reset_bonus: f32 = 0.75,
    flip_into_reset_window: f32 = 1.0,
    spin_reset_bonus: f32 = 0.3,
    spin_min_rate: f32 = 3.5,
    spin_full_rate: f32 = 5.25,
    multi_reset_step: f32 = 0.25,
    max_multi_reset_mult: f32 = 2.0,
    need_bonus: f32 = 0.5,
    need_full_dist: f32 = 8000.0,
    follow_reward: f32 = 3.0,
    follow_time: f32 = 0.75,
    max_follow_distance: f32 = 550.0,
    above_margin: f32 = 0.0,
    ahead_margin: f32 = 50.0,
    use_reward: f32 = 8.0,
    use_window: f32 = 2.0,
    use_flip_base: f32 = 0.3,
    use_full_gain: f32 = 1500.0,
    use_reset_quality: f32 = 0.6,
    use_touch_quality: f32 = 0.4,
    dodge_hit_window: f32 = 0.65,
    min_use_ball_height: f32 = 0.0,
    delayed_setup_time: f32 = 0.25,
    delayed_hit_time: f32 = 0.3,
    delayed_use_bonus: f32 = 1.0,
    separation_dist: f32 = 180.0,
    goal_reward: f32 = 20.0,
    goal_window: f32 = 3.0,
    start_reset_counts: bool = false,
    two_step_resets: bool = false,
    use_aim: bool = false,
    use_aim_reach: bool = false,
    use_reach_short_zero: f32 = 400.0,
    quick_use_factor: f32 = 1.0,
    wasted_dodge_cost: f32 = 0.0,
    flip_into_reset_free: bool = false,
    reset_height_bonus: f32 = 0.0,
    reset_height_full: f32 = 1500.0,
    close_reset_dist: f32 = 0.0,
    far_reset_min_height: f32 = 1100.0,
    delayed_bonus_close_only: bool = false,
    far_reset_needs_next: bool = false,
    reset_min_air_touches: i32 = 0,
    reset_max_goal_dist: f32 = 0.0,
    reset_max_fall_speed: f32 = 0.0,
    setup_closing_full: f32 = 0.0,
    reset_min_closing_speed: f32 = 0.0,
    reset_apex_bonus: f32 = 0.0,
    reset_apex_zero_fall: f32 = 800.0,
    far_firm_reward: f32 = 0.0,
    far_firm_car_fall_zero: f32 = 600.0,
    far_firm_closing_zero: f32 = 150.0,
    far_firm_closing_full: f32 = 600.0,
    far_firm_setup: bool = false,
} }

kept! {
    /// Each car's own state between steps (ticks on the episode clock).
    struct Track {
        /// Resets counted this episode.
        resets: i32,
        setup_paid: f32,
        /// Any reset contact this episode (counted or not).
        had_reset_contact: bool,
        last_reset_contact_tick: f32,
        /// The ball at the last counted reset.
        last_reset_ball_pos: Vec3A,
        /// The best setup potential on the current approach (only progress above it pays).
        setup_best: f32,
        /// Watching the follow-through of the last reset.
        following: bool,
        reset_tick: f32,
        /// The last reset's use hasn't been judged yet.
        use_pending: bool,
        /// A dodge started after the reset, waiting for it to hit the ball.
        dodging: bool,
        dodge_tick: f32,
        /// The last reset's style multiplier.
        style: f32,
        ball_goal_speed_at_dodge: f32,
        /// The car came off the ball since the last reset.
        separated: bool,
        /// A reset was used: a goal from this play pays goal_reward.
        shot_armed: bool,
        /// The car's last touch since the use.
        shot_touch_tick: f32,
        /// Touched the floor, a wall or the ceiling this episode (else it started in the air).
        landed_this_episode: bool,
        /// The last tick on the floor, a wall or the ceiling.
        surface_tick: f32,
        /// The last counted reset was used by a dodge into the ball, at this tick.
        dodge_used: bool,
        dodge_use_tick: f32,
        /// The last counted reset was far and low, or far and held (close_reset_dist): its follow-through pays nothing.
        unpaid: bool,
        /// A far reset's pay held until the next counted reset (far_reset_needs_next).
        held_pay: f32,
        /// Separate touches since the car last left the floor, and its latest touch (reset_min_air_touches).
        air_touches: i32,
        last_touch_tick: f32,
        /// No flip left in the air, waiting for a flip reset (`flip_reset_over_steps`).
        reset_waiting: bool,
    }
}

/// The farthest a car starting with its flip unused may be from the ball for that flip to count as a reset off it.
pub const START_RESET_MAX_BALL_DIST: f32 = 600.0;
/// The longest since its last 3+ wheel contact (RocketSim's air time) a car starting with its flip unused may have
/// been in the air for that flip to count as a reset off the ball: the default use window, so a moment of the
/// start part-way through a move still holding a usable reset keeps it (held resets in such starts last up to
/// 1.24 s).
pub const START_RESET_MAX_AIR_TIME: f32 = 2.0;

/// A car first seen in the air with its jump and flip unused, away from every arena surface, near the ball and
/// not long off its last wheel contact: its flip came back off the ball (a start part-way through a move).
/// `detail` is the car's at that moment.
pub fn holds_reset_at_start(car: &CarSnapshot, detail: &CarDetail, ball: Vec3A) -> bool {
    !car.is_demoed
        && !car.has_jumped
        && !car.has_flipped
        && !car.has_double_jumped
        && !near_arena_surface(car.body.pos, 120.0)
        && car.body.pos.distance(ball) <= START_RESET_MAX_BALL_DIST
        && detail.air_time <= START_RESET_MAX_AIR_TIME
}

/// The longest a shot's flight is followed to the goal line (s).
const SHOT_MAX_TIME: f32 = 6.0;
/// A shot over the bar by this much (uu, the ball's centre) aims at nothing.
const OVER_BAR_ZERO: f32 = 400.0;

/// A shot that must reach the net in the air (`air_shot_aim`) landing this far short of the goal line (uu, where
/// the ball's centre comes down to a ball radius up) aims at nothing.
const SHORT_ZERO: f32 = 400.0;

/// How well the ball, just hit, heads into the net `team` attacks, 0-1. 1 when its flight (gravity, drag and
/// bounces, `SHOT_MAX_TIME` at most) crosses the goal line inside the goal mouth. Otherwise the angle between its
/// horizontal direction and the ball-to-goal-centre direction, from 0 at 90 degrees (sideways) or more (away) to 1
/// at the edge of the mouth as seen from the ball, squared (so a near miss rates near 1, a wide one low), times the
/// height where it crosses: 1 under the bar, 0 `OVER_BAR_ZERO` over it (0 if it does not get there).
pub fn shot_aim(team: Team, ball: &Body) -> f32 {
    let shot = project_to_goal_line(ball.pos, ball.vel, team, SHOT_MAX_TIME);
    if shot.on_target {
        return 1.0;
    }
    if !shot.reaches {
        return 0.0;
    }
    across_aim(team, ball) * ramp(shot.z, BAR + OVER_BAR_ZERO, BAR)
}

/// `shot_aim` for a ball that must reach the net in the air (flip_reset's use_aim_reach): its flight with gravity
/// only (no drag or bounces). The width: 1 when it crosses the goal line between the posts (a ball radius inside),
/// else `shot_aim`'s angle term. The height: where it crosses, 1 under the bar falling to 0 `OVER_BAR_ZERO` over;
/// or, when it comes down to the floor first, 1 at the goal line falling to 0 `SHORT_ZERO` short of it. 0 heading
/// away. So a shot from far out, or a ball hit low and falling, that lands before the net rates 0.
pub fn air_shot_aim(team: Team, ball: &Body) -> f32 {
    air_shot_aim_short(team, ball, SHORT_ZERO)
}

/// `air_shot_aim` with a ball landing `short_zero` (uu) short of the goal line rating 0 instead of `SHORT_ZERO`.
pub fn air_shot_aim_short(team: Team, ball: &Body, short_zero: f32) -> f32 {
    let shot = air_path_to_goal_line(ball.pos, ball.vel, team);
    if !shot.toward {
        return 0.0;
    }
    let across = if shot.x.abs() <= GOAL_HALF_WIDTH - BALL_RADIUS {
        1.0
    } else {
        across_aim(team, ball)
    };
    let height = if shot.short > 0.0 {
        ramp(shot.short, short_zero, 0.0)
    } else {
        ramp(shot.z, BAR + OVER_BAR_ZERO, BAR)
    };
    across * height
}

/// The highest a ball's centre may cross the goal line and go in.
const BAR: f32 = GOAL_HEIGHT - BALL_RADIUS;

/// The ball's horizontal heading against the ball-to-goal-centre direction: 0 at 90 degrees or more, 1 at the edge
/// of the mouth as seen from the ball, squared.
fn across_aim(team: Team, ball: &Body) -> f32 {
    let to_goal = attack_goal(team) - ball.pos;
    let dist = hspeed(to_goal).max(1.0);
    let edge = (GOAL_HALF_WIDTH - BALL_RADIUS).atan2(dist);
    let off = dir2d(ball.vel).dot(dir2d(to_goal)).clamp(-1.0, 1.0).acos();
    ramp(off, std::f32::consts::FRAC_PI_2, edge).powi(2)
}

/// Touches at least this far apart (s) are separate.
const AIR_TOUCH_GAP: f32 = 0.15;

/// The car's wheels on the floor (not a wall or the ceiling).
fn on_floor(car: &CarSnapshot) -> bool {
    on_arena_surface(car) && car.body.pos.z < 120.0
}

const NO_FLIP: i32 = 0;
const FLIP_INTO: i32 = 1;
const BACKFLIP_INTO: i32 = 2;

impl FlipReset {
    /// How many numbers of memory `step` takes.
    pub const MEMORY: usize = Clock::SIZE + Track::SIZE;

    /// Holds a reset it counted and hasn't used yet (its use is pending), by the memory `step` keeps.
    pub fn holds_counted_reset(memory: &[f32]) -> bool {
        let tr = Track::load(&memory[Clock::SIZE..]);
        tr.resets > 0 && tr.use_pending
    }

    /// The episode tick of the last counted reset (None before the first; 0 for one held at the start).
    pub fn counted_reset_tick(memory: &[f32]) -> Option<f32> {
        let tr = Track::load(&memory[Clock::SIZE..]);
        (tr.resets > 0).then_some(tr.reset_tick)
    }

    /// The episode tick at which the last counted reset was used by a dodge into the ball (None if it wasn't).
    pub fn dodge_use_tick(memory: &[f32]) -> Option<f32> {
        let tr = Track::load(&memory[Clock::SIZE..]);
        tr.dodge_used.then_some(tr.dodge_use_tick)
    }

    /// Was the car dodging when it got the reset (its state at the start of the reset step)? RocketSim keeps
    /// counting the flip time after the dodge torque ends, so a dodge whose wheels reach the ball a little later
    /// still counts.
    fn flip_into_reset(&self, prev: &CarSnapshot, prev_detail: &CarDetail) -> i32 {
        let dodging = prev_detail.is_flipping
            || (prev.has_flipped
                && prev_detail.flip_time > 0.0
                && prev_detail.flip_time < self.flip_into_reset_window);
        if !dodging {
            NO_FLIP
        } else if prev_detail.flip_rel_torque.y < -0.5 {
            BACKFLIP_INTO // A backward dodge: torque y < 0
        } else {
            FLIP_INTO
        }
    }

    fn spin_term(&self, prev: &CarSnapshot) -> f32 {
        let rate = prev.body.ang_vel.dot(prev.body.forward()).abs(); // The roll rate
        clamp01((rate - self.spin_min_rate) / (self.spin_full_rate - self.spin_min_rate))
    }

    /// The ball at `pos` is beyond close_reset_dist of the goal line `team` attacks (never with it 0).
    fn is_far(&self, team: Team, pos: Vec3A) -> bool {
        self.close_reset_dist > 0.0
            && BACK_WALL_Y - pos.y * attack_sign(team) > self.close_reset_dist
    }

    /// A reset by `car` off `ball` after `air_touches` touches counts (reset_min_air_touches, reset_max_goal_dist,
    /// reset_max_fall_speed).
    fn reset_allowed(&self, car: &CarSnapshot, ball: &Body, air_touches: i32) -> bool {
        let falling = |vz: f32| self.reset_max_fall_speed > 0.0 && vz < -self.reset_max_fall_speed;
        air_touches >= self.reset_min_air_touches
            && (self.reset_max_goal_dist <= 0.0
                || BACK_WALL_Y - ball.pos.y * attack_sign(car.team) <= self.reset_max_goal_dist)
            && !falling(car.body.vel.z)
            && !falling(ball.vel.z)
    }

    /// The car moving into the ball firmly enough for a reset (reset_min_closing_speed), from their states before it.
    fn firm(&self, car: &CarSnapshot, ball: &Body) -> bool {
        let to_ball = ball.pos - car.body.pos;
        self.reset_min_closing_speed <= 0.0
            || (car.body.vel - ball.vel).dot(to_ball.normalize_or_zero())
                >= self.reset_min_closing_speed
    }

    /// far_firm_reward's share for a first reset by `car` (its state before) off `ball` (before) with the ball now at
    /// `now`: 0 unless far and high enough; then the closing speed's ramp times the car's fall's.
    fn far_firm(&self, car: &CarSnapshot, ball: &Body, now: Vec3A) -> f32 {
        if self.far_firm_reward <= 0.0
            || !self.is_far(car.team, now)
            || now.z < self.far_reset_min_height
        {
            return 0.0;
        }
        let closing = (car.body.vel - ball.vel).dot((ball.pos - car.body.pos).normalize_or_zero());
        ramp(
            closing,
            self.far_firm_closing_zero,
            self.far_firm_closing_full,
        ) * ramp(car.body.vel.z, -self.far_firm_car_fall_zero, 0.0)
    }

    /// How much a reset is needed now, 0-1: half from low boost, half from distance to the opponent goal.
    fn need(&self, car: &CarSnapshot) -> f32 {
        let goal = Vec3A::new(0.0, attack_sign(car.team) * 5120.0, 0.0);
        let mut d = goal - car.body.pos;
        d.z = 0.0;
        clamp01(0.5 * (1.0 - car.boost / 100.0) + 0.5 * clamp01(d.length() / self.need_full_dist))
    }

    /// The setup potential, 0-1: close x wheels toward the ball x not above it x flip; 0 where a reset after
    /// `air_touches` touches would not count.
    fn setup_potential(
        &self,
        car: &CarSnapshot,
        ball_body: &Body,
        air_touches: i32,
        first: bool,
    ) -> f32 {
        let ball = ball_body.pos;
        if ball.z < self.min_ball_height
            || car.is_demoed
            || near_arena_surface(car.body.pos, 120.0)
            || !self.reset_allowed(car, ball_body, air_touches)
        {
            return 0.0;
        }
        let far = self.is_far(car.team, ball);
        let firm_far = far && first && self.far_firm_setup && ball.z >= self.far_reset_min_height;
        if far && !firm_far && (ball.z < self.far_reset_min_height || self.far_reset_needs_next) {
            return 0.0; // A reset there would not be paid at once
        }
        let rising = if firm_far {
            ramp(car.body.vel.z, -self.far_firm_car_fall_zero, 0.0)
        } else {
            1.0
        };
        let d = ball - car.body.pos;
        let dist = d.length();
        if dist < 1.0 {
            return 0.0;
        }
        let dir = d / dist;
        let near =
            clamp01((self.setup_max_dist - dist) / (self.setup_max_dist - self.setup_full_dist));
        let strictness = near * near; // Loose until fairly close, then strict
        let close = near.powf(self.setup_close_exponent);
        let facing = (-dir.dot(car.body.up())).clamp(-1.0, 1.0); // 1 = wheels straight at the ball
        let loose = 0.5 * (facing + 1.0);
        let strict = facing.max(0.0) * facing.max(0.0);
        let wheels = loose + (strict - loose) * strictness;
        let allowed_above = self.max_above_ball + (1.0 - strictness) * self.setup_above_looseness;
        let not_above = clamp01(1.0 - (car.body.pos.z - ball.z) / allowed_above);
        let flip = if needs_reset(car) {
            1.0
        } else {
            1.0 - (1.0 - self.setup_flip_ready_factor) * strictness
        };
        let closing = if self.setup_closing_full > 0.0 {
            clamp01((car.body.vel - ball_body.vel).dot(dir) / self.setup_closing_full)
        } else {
            1.0
        };
        close * wheels * not_above * flip * closing * rising
    }

    /// One step of the reward with its memory: what `value_remembering` pays (ResetMusty steps it too).
    pub fn step(
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
            // The episode's start: only progress from the starting position pays
            let start = &prev.cars[car];
            // A start in the air: the touch that put it there
            let air_touches = i32::from(!on_arena_surface(start));
            let mut tr = Track {
                setup_best: self.setup_potential(start, &prev.ball, air_touches, true),
                landed_this_episode: on_arena_surface(start),
                surface_tick: 0.0,
                style: 1.0,
                air_touches,
                last_touch_tick: -1000.0,
                ..Track::default()
            };
            if self.start_reset_counts
                && prev.ball.pos.z >= self.min_ball_height
                && holds_reset_at_start(start, &events.prev_detail(car), prev.ball.pos)
            {
                // Held from the start: counted at tick 0, unused, its contact just made
                tr.resets = 1;
                tr.use_pending = true;
                tr.last_reset_ball_pos = prev.ball.pos;
                tr.had_reset_contact = true;
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

    fn update(
        &self,
        tr: &mut Track,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        now: f32,
    ) -> f32 {
        let mut reward = 0.0;
        let (p, pp, ball) = (&cur.cars[car], &prev.cars[car], &cur.ball);
        let touched = events.cars[car].touched_ball;
        let goal = Vec3A::new(0.0, attack_sign(p.team), 0.0);
        let rel = p.body.pos - ball.pos;

        // The flip back off the ball, also over two steps
        let (two_step, waiting) =
            flip_reset_over_steps(tr.reset_waiting, pp, p, ball.pos, self.max_reset_ball_dist);
        tr.reset_waiting = waiting;
        let reset_now = if self.two_step_resets {
            two_step
        } else {
            flip_reset_happened(pp, p, ball.pos, self.max_reset_ball_dist)
        };

        if (tr.following || tr.use_pending) && rel.length() >= self.separation_dist {
            tr.separated = true;
        }
        if on_arena_surface(p) {
            tr.landed_this_episode = true;
            tr.surface_tick = now;
        }
        let flying =
            !tr.landed_this_episode || (now - tr.surface_tick) / 120.0 >= self.min_air_time;

        // ---- The use of the last reset (judged before the follow-through: using it well ends that) ----
        let mut used_now = false;
        if tr.use_pending {
            let since = (now - tr.reset_tick) / 120.0;
            let dodge_started = p.has_flipped && !pp.has_flipped;
            let mut quality = -1.0;
            let mut pay_mult = 1.0; // Changes what the use pays, not whether it counts as one
            let mut aim = 1.0;
            if dodge_started && !tr.dodging {
                tr.dodging = true;
                tr.dodge_tick = now;
                tr.ball_goal_speed_at_dodge = prev.ball.vel.dot(goal);
            }
            let into_reset =
                self.flip_into_reset_free && ball.pos.z >= self.min_ball_height && reset_now;
            if tr.dodging && into_reset {
                // A flip into the next reset: left pending for the reset below to pay as a multi-reset use
            } else if tr.dodging {
                if touched {
                    let gain = ball.vel.dot(goal) - tr.ball_goal_speed_at_dodge;
                    quality = self.use_flip_base
                        + (1.0 - self.use_flip_base) * clamp01(gain / self.use_full_gain);
                    let delayed = (tr.dodge_tick - tr.reset_tick) / 120.0
                        >= self.delayed_setup_time
                        && (now - tr.dodge_tick) / 120.0 >= self.delayed_hit_time;
                    if delayed {
                        if !(self.delayed_bonus_close_only && self.is_far(p.team, ball.pos)) {
                            quality *= 1.0 + self.delayed_use_bonus;
                        }
                    } else if (tr.dodge_tick - tr.reset_tick) / 120.0 < self.delayed_setup_time {
                        pay_mult = self.quick_use_factor;
                    }
                    if self.use_aim {
                        aim = if self.use_aim_reach {
                            air_shot_aim_short(p.team, ball, self.use_reach_short_zero)
                        } else {
                            shot_aim(p.team, ball)
                        };
                        pay_mult *= aim;
                    }
                } else if (now - tr.dodge_tick) / 120.0 > self.dodge_hit_window {
                    quality = 0.0; // The dodge missed the ball
                }
            } else if touched && tr.separated {
                // Played the ball without the dodge
                quality = self.use_touch_quality * clamp01((ball.vel.dot(goal) + 500.0) / 1500.0);
            }
            if quality > 0.0 && ball.pos.z < self.min_use_ball_height {
                quality = 0.0; // Used with the ball too low
            }
            if quality < 0.0
                && (since > self.use_window
                    || p.is_demoed
                    || on_arena_surface(p)
                    || rel.length() > self.max_follow_distance * 1.5)
            {
                quality = 0.0; // Not used in time, or lost the ball
            }
            if quality >= 0.0 {
                used_now = quality > 0.0;
                if used_now && tr.dodging {
                    tr.dodge_used = true;
                    tr.dodge_use_tick = now;
                }
                if tr.dodging {
                    // A dodge that missed the ball (or hit it with the ball too low), or with use_aim off target
                    let wasted = if !used_now {
                        1.0
                    } else if self.use_aim {
                        1.0 - aim
                    } else {
                        0.0
                    };
                    reward -= self.wasted_dodge_cost * wasted;
                }
                tr.use_pending = false;
                tr.dodging = false;
                reward += self.use_reward * quality * pay_mult * tr.style;
                if used_now {
                    tr.shot_armed = true;
                    tr.shot_touch_tick = now;
                }
            }
        }

        // ---- The follow-through of the last reset: kept if the car stays close, below and behind the ball for
        // follow_time, plays the ball again or uses the reset; not judged while a dodge is on its way ----
        let follow_pay = if tr.unpaid {
            0.0
        } else {
            self.follow_reward * tr.style
        };
        if tr.following && used_now {
            tr.following = false;
            reward += follow_pay;
        } else if tr.following && !tr.dodging {
            let bad = p.is_demoed
                || on_arena_surface(p)
                || rel.length() > self.max_follow_distance
                || rel.z > self.above_margin
                || rel.dot(goal) > self.ahead_margin;
            let used = touched && tr.separated; // Played the ball again (not the reset contact itself)
            if bad {
                tr.following = false;
            } else if used || (now - tr.reset_tick) / 120.0 >= self.follow_time {
                tr.following = false;
                reward += follow_pay;
            }
        }

        // ---- The reset ----
        let ball_up = ball.pos.z >= self.min_ball_height;
        let reset_contact = ball_up && reset_now;
        let too_soon = reset_contact
            && tr.had_reset_contact
            && (now - tr.last_reset_contact_tick) / 120.0 < self.min_reset_interval;
        if reset_contact {
            tr.had_reset_contact = true;
            tr.last_reset_contact_tick = now;
        }
        if reset_contact
            && !too_soon
            && flying
            && tr.resets < self.max_resets_per_episode
            && self.reset_allowed(p, ball, tr.air_touches)
            && self.firm(pp, &prev.ball)
        {
            // A second or later reset counts for as much as the ball has moved toward the opponent goal since
            let progress = if tr.resets == 0 {
                1.0
            } else {
                clamp01(
                    ((ball.pos - tr.last_reset_ball_pos).dot(goal) - self.multi_reset_min_progress)
                        / (self.multi_reset_full_progress - self.multi_reset_min_progress).max(1.0),
                )
            };
            if tr.use_pending {
                // The previous reset was used to get this one: a multi-reset
                tr.use_pending = false;
                tr.dodging = false;
                let q = self.use_reset_quality * progress;
                reward += self.use_reward * q * tr.style;
                if q > 0.0 {
                    tr.shot_armed = true;
                    tr.shot_touch_tick = now;
                }
            }
            if tr.resets == 0 {
                reward += self.far_firm_reward * self.far_firm(pp, &prev.ball, ball.pos);
            }
            if self.far_firm_setup {
                // The first approach's progress does not count against the next one's
                tr.setup_best = 0.0;
            }
            tr.resets += 1;
            tr.dodge_used = false;
            tr.last_reset_ball_pos = ball.pos;
            tr.following = true;
            tr.use_pending = true;
            tr.reset_tick = now;
            tr.separated = false;
            let flip = self.flip_into_reset(pp, &events.prev_detail(car));
            let spin = if flip == NO_FLIP {
                self.spin_term(pp)
            } else {
                0.0
            };
            let flip_mult = match flip {
                BACKFLIP_INTO => 1.0 + self.backflip_into_reset_bonus,
                FLIP_INTO => 1.0 + self.flip_into_reset_bonus,
                _ => 1.0,
            };
            let need = self.need(p);
            let multi = (1.0 + self.multi_reset_step * (tr.resets - 1) as f32 * (0.5 + need))
                .min(self.max_multi_reset_mult);
            tr.style = flip_mult
                * (1.0 + self.spin_reset_bonus * spin)
                * multi
                * (1.0 + self.need_bonus * need)
                * progress;
            // A held far reset is paid for as much as this one carried the ball closer
            reward += tr.held_pay * progress;
            tr.held_pay = 0.0;
            // A far reset pays only high (its height bonus from far_reset_min_height), maybe held for the next
            let far = self.is_far(p.team, ball.pos);
            let low = far && ball.pos.z < self.far_reset_min_height;
            let held = far && !low && self.far_reset_needs_next;
            tr.unpaid = low || held;
            let height = if !far {
                ramp(ball.pos.z, self.min_ball_height, self.reset_height_full)
            } else if self.reset_height_full > self.far_reset_min_height {
                ramp(
                    ball.pos.z,
                    self.far_reset_min_height,
                    self.reset_height_full,
                )
            } else {
                1.0
            };
            let apex = ramp(ball.vel.z, -self.reset_apex_zero_fall, 0.0);
            let own = self.reset_reward
                * tr.style
                * (1.0 + self.reset_height_bonus * height)
                * (1.0 + self.reset_apex_bonus * apex);
            if held {
                tr.held_pay = own;
            } else if !low {
                reward += own;
            }
        }

        // ---- The setup (progress only, capped per episode): the approach ends when the car lands or is
        // demolished or the ball drops; off while the car holds a reset it hasn't used ----
        if on_arena_surface(p) || p.is_demoed || !ball_up {
            tr.setup_best = 0.0;
        } else if tr.resets < self.max_resets_per_episode && !tr.use_pending {
            let pot = self.setup_potential(p, ball, tr.air_touches, tr.resets == 0);
            if pot > tr.setup_best {
                let r = (self.setup_scale * (pot - tr.setup_best))
                    .min(self.max_setup_per_episode - tr.setup_paid);
                tr.setup_best = pot;
                if r > 0.0 {
                    tr.setup_paid += r;
                    reward += r;
                }
            }
        }

        // ---- Touches since leaving the floor, after the reset (judged on the touches before this step) ----
        if on_floor(p) {
            tr.air_touches = 0;
        }
        if touched {
            if (now - tr.last_touch_tick) / 120.0 >= AIR_TOUCH_GAP {
                tr.air_touches += 1;
            }
            tr.last_touch_tick = now;
        }

        // ---- A goal off a used reset ----
        if tr.shot_armed {
            if (0..cur.cars.len()).any(|o| o != car && events.cars[o].touched_ball) {
                tr.shot_armed = false; // Someone else played it
            }
            if touched {
                tr.shot_touch_tick = now;
            }
            if (now - tr.shot_touch_tick) / 120.0 > self.goal_window {
                tr.shot_armed = false;
            }
        }
        if cur.goal_scored.is_some() && prev.goal_scored.is_none() {
            if tr.shot_armed && ball.pos.y * goal.y > 0.0 {
                reward += self.goal_reward;
            }
            tr.shot_armed = false;
        }
        reward
    }
}

impl Reward for FlipReset {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn listed_with_every_setting_optional_and_described() {
        assert_listed("flip_reset", 71);
    }

    fn reward() -> FlipReset {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    /// A car (Blue's view, mirrored for Orange) in the air below the ball, its jump and flip used, wheels up.
    fn under_ball(team: Team, car_z: f32) -> GameSnapshot {
        let m = |v: Vec3A| as_blue(team, v);
        let mut g = game(
            m(Vec3A::new(0.0, -1000.0, 900.0)),
            &[(team, m(Vec3A::new(0.0, -1100.0, car_z)))],
        );
        let c = &mut g.cars[0];
        c.on_ground = false;
        c.has_jumped = true;
        c.has_flipped = true;
        c.can_flip = false;
        c.boost = 50.0;
        let upside_down = Mat3A::from_rotation_x(std::f32::consts::PI);
        c.body.rot = if team == Team::Blue {
            upside_down
        } else {
            Mat3A::from_rotation_z(std::f32::consts::PI) * upside_down
        };
        g
    }

    /// The car got its flip back off the ball (wheels on it).
    fn reset_on(mut g: GameSnapshot) -> GameSnapshot {
        let c = &mut g.cars[0];
        c.has_jumped = false;
        c.has_flipped = false;
        c.can_flip = true;
        c.on_ground = true;
        g
    }

    fn step(
        r: &FlipReset,
        a: &GameSnapshot,
        b: &GameSnapshot,
        touch: bool,
        memory: &mut [f32],
    ) -> f32 {
        let ev = if touch {
            touched(quiet(1), 0)
        } else {
            quiet(1)
        };
        r.value_remembering(a, b, &ev, 0, memory)
    }

    #[test]
    fn a_reset_pays_its_reward_times_need_and_orange_the_same() {
        let r = reward();
        let pay = |team: Team| {
            let mut memory = vec![0.0; r.memory()];
            let far = under_ball(team, 300.0);
            let near = under_ball(team, 800.0);
            let setup = step(&r, &far, &near, false, &mut memory);
            let got = reset_on(near.clone());
            let reset = step(&r, &near, &got, true, &mut memory);
            assert!(FlipReset::holds_counted_reset(&memory));
            (setup, reset)
        };
        let (setup, reset) = pay(Team::Blue);
        assert!(
            setup > 0.0 && setup <= r.max_setup_per_episode,
            "the approach pays: {setup}"
        );
        // need: half boost (0.25) + distance 6120 / 8000 (0.38) = 0.63 -> x1.32
        assert!(
            reset > r.reset_reward && reset < r.reset_reward * 1.5,
            "{reset}"
        );
        let (o_setup, o_reset) = pay(Team::Orange);
        assert!((setup - o_setup).abs() < 1e-4 && (reset - o_reset).abs() < 1e-4);
    }

    /// reset_height_bonus pays a higher reset more (its own pay only); off by default.
    #[test]
    fn a_higher_reset_pays_more_with_reset_height_bonus() {
        let pay = |r: &FlipReset, ball_z: f32| {
            let mut memory = vec![0.0; r.memory()];
            let mut near = under_ball(Team::Blue, 800.0);
            near.ball.pos.z = ball_z;
            near.cars[0].body.pos.z = ball_z - 100.0;
            step(r, &near, &reset_on(near.clone()), true, &mut memory)
        };
        let base = reward();
        assert_eq!(pay(&base, 600.0), pay(&base, 1500.0), "off: the same");
        let high = FlipReset {
            reset_height_bonus: 0.5,
            ..reward()
        };
        let (low, mid, top) = (pay(&high, 300.0), pay(&high, 900.0), pay(&high, 1500.0));
        assert!((low - pay(&base, 300.0)).abs() < 1e-4, "{low}");
        assert!(
            (mid / low - 1.25).abs() < 1e-3 && (top / low - 1.5).abs() < 1e-3,
            "{low} {mid} {top}"
        );
    }

    #[test]
    fn no_reset_with_the_ball_low_or_too_soon_after_another_and_memory_starts_again() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let mut low = under_ball(Team::Blue, 150.0);
        low.ball.pos.z = 250.0; // below min_ball_height
        low.cars[0].body.pos.z = 200.0;
        let got = reset_on(low.clone());
        assert_eq!(step(&r, &low, &got, true, &mut memory), 0.0);
        assert!(!FlipReset::holds_counted_reset(&memory));
        memory.fill(0.0);
        let near = under_ball(Team::Blue, 800.0);
        let got = reset_on(near.clone());
        assert!(step(&r, &near, &got, true, &mut memory) > r.reset_reward);
        // Its flip used again at once and another reset 1 step later: too soon (min_reset_interval)
        let used = under_ball(Team::Blue, 800.0);
        step(&r, &got, &used, false, &mut memory);
        let again = step(&r, &used, &reset_on(used.clone()), true, &mut memory);
        assert!(again < r.reset_reward, "too soon: {again}");
        assert_eq!(Track::load(&memory[Clock::SIZE..]).resets, 1);
    }

    #[test]
    fn a_dodge_into_the_ball_after_the_reset_is_a_use() {
        let r = reward();
        let mut memory = vec![0.0; r.memory()];
        let near = under_ball(Team::Blue, 800.0);
        let got = reset_on(near.clone());
        step(&r, &near, &got, true, &mut memory);
        // Off the ball, then a dodge that hits it toward the goal
        let mut off = got.clone();
        off.cars[0].on_ground = false;
        off.cars[0].body.pos.y = -1250.0;
        let off_pay = step(&r, &got, &off, false, &mut memory);
        let mut dodge = off.clone();
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        dodge.ball.vel = Vec3A::new(0.0, 1500.0, 0.0);
        let used = step(&r, &off, &dodge, true, &mut memory);
        // Use quality 1 (gain 1500) x use_reward, plus the follow-through
        assert!(
            used >= r.use_reward + r.follow_reward,
            "{off_pay} then {used}"
        );
        assert!(!FlipReset::holds_counted_reset(&memory));
        assert_eq!(FlipReset::counted_reset_tick(&memory), Some(8.0));
        assert_eq!(FlipReset::dodge_use_tick(&memory), Some(24.0));
    }

    /// The reset under the ball, off it, then a dodge (hitting the ball to `vel` when given, else missing it,
    /// judged a miss after dodge_hit_window): every step's pay, and the episode tick of the dodge use if any.
    fn dodge_after_reset(r: &FlipReset, vel: Option<Vec3A>) -> (Vec<f32>, Option<f32>) {
        let mut memory = vec![0.0; r.memory()];
        let near = under_ball(Team::Blue, 800.0);
        let got = reset_on(near.clone());
        let mut off = got.clone();
        off.cars[0].on_ground = false;
        off.cars[0].body.pos.y = -1250.0;
        let mut dodge = off.clone();
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        let mut pays = vec![
            step(r, &near, &got, true, &mut memory),
            step(r, &got, &off, false, &mut memory),
        ];
        match vel {
            Some(vel) => {
                dodge.ball.vel = vel;
                pays.push(step(r, &off, &dodge, true, &mut memory));
            }
            None => {
                pays.push(step(r, &off, &dodge, false, &mut memory));
                for _ in 0..12 {
                    pays.push(step(r, &dodge, &dodge, false, &mut memory));
                }
            }
        }
        (pays, FlipReset::dodge_use_tick(&memory))
    }

    /// use_aim scales a dodge use's pay by the shot's aim, quick_use_factor scales a dodge right after the reset,
    /// wasted_dodge_cost charges a miss (and with use_aim an off-target hit): none of them changes what counts as
    /// a use.
    #[test]
    fn aim_waiting_and_wasted_dodges_change_the_use_pay_only() {
        let base = reward();
        let with = |f: fn(&mut FlipReset)| {
            let mut r = reward();
            f(&mut r);
            r
        };
        let aimed = with(|r| r.use_aim = true);
        let goalward = Vec3A::new(0.0, 3000.0, 300.0);
        let mut flying = under_ball(Team::Blue, 800.0).ball;
        flying.vel = goalward;
        assert_eq!(shot_aim(Team::Blue, &flying), 1.0, "on target");
        let away = Vec3A::new(0.0, -1500.0, 0.0);
        let (plain_on, tick) = dodge_after_reset(&base, Some(goalward));
        assert_eq!(tick, Some(24.0));
        let (aimed_on, tick) = dodge_after_reset(&aimed, Some(goalward));
        assert!(tick == Some(24.0) && (plain_on[2] - aimed_on[2]).abs() < 1e-4);
        let (plain_away, _) = dodge_after_reset(&base, Some(away));
        let (aimed_away, tick) = dodge_after_reset(&aimed, Some(away));
        assert_eq!(tick, Some(24.0), "still a use, so a goal off it still pays");
        // Backward: the base 0.3 of a use, all of it taken off; the follow-through stays
        let style = plain_away[0] / base.reset_reward;
        let use_pay = base.use_reward * base.use_flip_base * style;
        assert!(
            (plain_away[2] - aimed_away[2] - use_pay).abs() < 1e-3,
            "{plain_away:?} {aimed_away:?}"
        );
        // The dodge came 16 ticks (0.13 s) after the reset: quick, its use pay times quick_use_factor
        let quick = with(|r| r.quick_use_factor = 0.25);
        let (quick_away, tick) = dodge_after_reset(&quick, Some(away));
        assert_eq!(tick, Some(24.0));
        assert!((plain_away[2] - quick_away[2] - 0.75 * use_pay).abs() < 1e-3);
        // A miss costs wasted_dodge_cost; with use_aim, a backward hit (aim 0) costs it all too
        let costly = with(|r| r.wasted_dodge_cost = 2.0);
        let (missed, tick) = dodge_after_reset(&costly, None);
        let (missed_free, _) = dodge_after_reset(&base, None);
        assert!(tick.is_none());
        let diff: f32 = missed.iter().zip(&missed_free).map(|(a, b)| a - b).sum();
        assert!((diff + 2.0).abs() < 1e-4, "{missed:?}");
        let costly_aimed = with(|r| {
            r.wasted_dodge_cost = 2.0;
            r.use_aim = true;
        });
        let (costly_away, _) = dodge_after_reset(&costly_aimed, Some(away));
        assert!((aimed_away[2] - costly_away[2] - 2.0).abs() < 1e-4);
        let (costly_on, _) = dodge_after_reset(&costly_aimed, Some(goalward));
        assert!(
            (aimed_on[2] - costly_on[2]).abs() < 1e-4,
            "on target costs nothing"
        );
        // Without use_aim a backward hit is a use like any other: no cost
        let (costly_plain, _) = dodge_after_reset(&costly, Some(away));
        assert!((plain_away[2] - costly_plain[2]).abs() < 1e-4);
    }

    #[test]
    fn the_shot_aim_is_1_on_target_0_sideways_or_away_and_falls_off_wide_or_high() {
        let at = Vec3A::new(0.0, 3000.0, 600.0);
        for team in [Team::Blue, Team::Orange] {
            let aim = |pos: Vec3A, vel: Vec3A| {
                shot_aim(team, &ball(as_blue(team, pos), as_blue(team, vel)))
            };
            assert_eq!(aim(at, Vec3A::new(0.0, 2500.0, 0.0)), 1.0);
            assert_eq!(aim(at, Vec3A::new(2500.0, 0.0, 0.0)), 0.0);
            assert_eq!(aim(at, Vec3A::new(0.0, -2500.0, 0.0)), 0.0);
            assert_eq!(aim(at, Vec3A::new(-2000.0, -1500.0, 0.0)), 0.0);
            let wide = aim(at, Vec3A::new(1500.0, 2500.0, 0.0)); // 31 deg off, the post 21: wide
            assert!(wide > 0.3 && wide < 0.8, "{wide}");
            let high = aim(at, Vec3A::new(0.0, 2500.0, 600.0)); // Over the bar, on line
            assert!(high > 0.0 && high < 1.0, "{high}");
            let way_high = aim(at, Vec3A::new(0.0, 2500.0, 1500.0));
            assert_eq!(way_high, 0.0, "{way_high}");
        }
    }

    fn ball(pos: Vec3A, vel: Vec3A) -> Body {
        Body {
            pos,
            vel,
            ..Body::RESTING
        }
    }

    /// air_shot_aim is 1 only for a flight that reaches the goal line inside the mouth before the floor, falls
    /// off smoothly for one landing just short, and is 0 far short, where shot_aim (bounces count) still says 1.
    #[test]
    fn the_air_shot_aim_needs_the_flight_to_reach_the_net_before_the_floor() {
        for team in [Team::Blue, Team::Orange] {
            let both = |pos: Vec3A, vel: Vec3A| {
                let b = ball(as_blue(team, pos), as_blue(team, vel));
                (air_shot_aim(team, &b), shot_aim(team, &b))
            };
            // 2,000 uu out, 500 up, at 2,200 uu/s: crosses the line 0.91 s later at 460 up
            let near = Vec3A::new(0.0, BACK_WALL_Y - 2000.0, 500.0);
            assert_eq!(both(near, Vec3A::new(0.0, 2200.0, 250.0)), (1.0, 1.0));
            // 5,000 uu out, 300 up: down on the floor 1.17 s later, 2,670 uu short; rolls in with bounces
            let far = Vec3A::new(0.0, BACK_WALL_Y - 5000.0, 300.0);
            assert_eq!(both(far, Vec3A::new(0.0, 2000.0, 200.0)), (0.0, 1.0));
            // Hit low and falling near the net: short by 1,490 uu
            let low = Vec3A::new(0.0, BACK_WALL_Y - 2120.0, 600.0);
            assert_eq!(both(low, Vec3A::new(0.0, 1200.0, -800.0)).0, 0.0);
            // Wide in the air: the angle term; over the bar: partial; far over or away: 0
            let (wide, old_wide) = both(near, Vec3A::new(1500.0, 2200.0, 250.0));
            assert!(
                wide > 0.0 && wide < 1.0 && (wide - old_wide).abs() < 1e-5,
                "{wide}"
            );
            let over = both(near, Vec3A::new(0.0, 2200.0, 700.0)).0; // 867 up at the line
            assert!(over > 0.1 && over < 0.4, "{over}");
            assert_eq!(both(near, Vec3A::new(0.0, 2200.0, 1500.0)).0, 0.0);
            assert_eq!(both(near, Vec3A::new(0.0, -2200.0, 250.0)).0, 0.0);
            // From 1,000 out, 200 up and flat: 0 when slow, rising smoothly to 1 once it carries to the line
            let close = Vec3A::new(0.0, BACK_WALL_Y - 1000.0, 200.0);
            let aims: Vec<f32> = (0..=20)
                .map(|i| both(close, Vec3A::new(0.0, 600.0 + 100.0 * i as f32, 0.0)).0)
                .collect();
            assert!(aims[0] == 0.0 && aims[20] == 1.0, "{aims:?}");
            assert!(
                aims.windows(2).all(|w| w[1] >= w[0] && w[1] - w[0] < 0.35),
                "{aims:?}"
            );
            assert!(aims.iter().any(|&a| a > 0.2 && a < 0.8), "{aims:?}");
        }
    }

    /// use_aim_reach (with use_aim) pays a dodge use by air_shot_aim: a hit low and falling toward the net, which
    /// shot_aim calls on target, pays nothing and with wasted_dodge_cost costs it; one carrying to the net pays as
    /// before.
    #[test]
    fn use_aim_reach_pays_only_a_shot_that_reaches_the_net_in_the_air() {
        let aimed = FlipReset {
            use_aim: true,
            wasted_dodge_cost: 2.0,
            ..reward()
        };
        let reach = FlipReset {
            use_aim: true,
            use_aim_reach: true,
            wasted_dodge_cost: 2.0,
            ..reward()
        };
        // The hit ball sits at (0, -1000, 900) in dodge_after_reset
        let carries = Vec3A::new(0.0, 3000.0, 300.0); // crosses the line 2.04 s later 160 up
        let falling = Vec3A::new(0.0, 1500.0, -300.0); // down 4,350 uu short
        let at = Vec3A::new(0.0, -1000.0, 900.0);
        assert_eq!(air_shot_aim(Team::Blue, &ball(at, carries)), 1.0);
        assert_eq!(air_shot_aim(Team::Blue, &ball(at, falling)), 0.0);
        assert_eq!(shot_aim(Team::Blue, &ball(at, falling)), 1.0);
        let (old_on, _) = dodge_after_reset(&aimed, Some(carries));
        let (new_on, tick) = dodge_after_reset(&reach, Some(carries));
        assert!(tick.is_some() && (old_on[2] - new_on[2]).abs() < 1e-4);
        let (old_low, _) = dodge_after_reset(&aimed, Some(falling));
        let (new_low, tick) = dodge_after_reset(&reach, Some(falling));
        assert_eq!(tick, Some(24.0), "still a use");
        assert!((old_low[2] - old_on[2]).abs() < 1e-4, "on target before");
        // Without the use pay (0) and with the whole wasted_dodge_cost; the follow-through stays
        let style = old_on[0] / aimed.reset_reward;
        let gain = (1500.0f32 / aimed.use_full_gain).min(1.0);
        let quality = aimed.use_flip_base + (1.0 - aimed.use_flip_base) * gain;
        let use_pay = aimed.use_reward * quality * style;
        assert!(
            (old_low[2] - new_low[2] - use_pay - 2.0).abs() < 1e-3,
            "{old_low:?} {new_low:?}"
        );
        // Off by default, and nothing without use_aim
        assert!(!reward().use_aim_reach);
        let reach_only = FlipReset {
            use_aim_reach: true,
            ..reward()
        };
        let (plain, _) = dodge_after_reset(&reward(), Some(falling));
        let (alone, _) = dodge_after_reset(&reach_only, Some(falling));
        assert_eq!(plain, alone);
    }

    /// reset_min_air_touches counts separate touches since the car left the floor (a start in the air 1,
    /// a carry once, the floor starting again); a reset before enough pays nothing, holds nothing, and its approach
    /// pays nothing either.
    #[test]
    fn a_reset_counts_only_after_its_air_touches() {
        let r = FlipReset {
            reset_min_air_touches: 3,
            ..reward()
        };
        let air = under_ball(Team::Blue, 800.0);
        let got = reset_on(air.clone());
        let mut floor = air.clone();
        floor.cars[0].body.pos.z = 17.0;
        floor.cars[0].on_ground = true;
        // touches: which steps touch the ball (8 ticks a step; 0.15 s apart is 3 steps), then the reset
        let reset_after = |r: &FlipReset, touches: &[bool], land_first: bool| {
            let mut memory = vec![0.0; r.memory()];
            step(r, &air, &air, false, &mut memory);
            if land_first {
                step(r, &air, &floor, false, &mut memory);
                step(r, &floor, &air, false, &mut memory);
            }
            for &t in touches {
                step(r, &air, &air, t, &mut memory);
            }
            let pay = step(r, &air, &got, true, &mut memory);
            (pay, FlipReset::holds_counted_reset(&memory))
        };
        let (f, t) = (false, true);
        // Granted 1 + two separate touches: counts
        let (pay, holds) = reset_after(&r, &[t, f, f, f, t, f, f, f], false);
        assert!(pay > 0.0 && holds, "{pay}");
        // One touch, or a carry (touching every step) that counts once: not enough
        for touches in [&[t, f, f, f][..], &[t, t, t, t, t, t, f, f]] {
            assert_eq!(reset_after(&r, touches, false), (0.0, false), "{touches:?}");
        }
        // Landing starts the count again: the two touches after it alone
        assert_eq!(
            reset_after(&r, &[t, f, f, f, t, f, f, f], true),
            (0.0, false)
        );
        let two = FlipReset {
            reset_min_air_touches: 2,
            ..reward()
        };
        assert!(reset_after(&two, &[t, f, f, f, t, f, f, f], true).1);
        // Off by default
        assert_eq!(reward().reset_min_air_touches, 0);
        assert!(reset_after(&reward(), &[], true).1);
        // The approach to a reset that would not count pays nothing
        let mut memory = vec![0.0; r.memory()];
        let low = under_ball(Team::Blue, 300.0);
        assert_eq!(step(&r, &low, &air, false, &mut memory), 0.0);
        let mut memory = vec![0.0; r.memory()];
        assert!(step(&reward(), &low, &air, false, &mut memory) > 0.0);
    }

    /// reset_max_fall_speed counts a reset only with car and ball falling slower than it; the approach to a
    /// reset that would not count pays nothing. setup_closing_full pays the approach for closing on the ball.
    #[test]
    fn a_reset_counts_only_not_falling_fast_and_the_approach_pays_closing_in() {
        let r = FlipReset {
            reset_max_fall_speed: 500.0,
            ..reward()
        };
        let reset_falling = |r: &FlipReset, car_vz: f32, ball_vz: f32| {
            let mut memory = vec![0.0; r.memory()];
            let mut near = under_ball(Team::Blue, 800.0);
            near.cars[0].body.vel.z = car_vz;
            near.ball.vel.z = ball_vz;
            let mut got = reset_on(near.clone());
            got.cars[0].body.vel.z = car_vz;
            step(r, &near, &got, true, &mut memory);
            FlipReset::holds_counted_reset(&memory)
        };
        assert!(reset_falling(&r, -400.0, -400.0));
        assert!(!reset_falling(&r, -600.0, 0.0), "the car falling fast");
        assert!(!reset_falling(&r, 0.0, -600.0), "the ball falling fast");
        assert!(reset_falling(&reward(), -900.0, -900.0), "off by default");
        // The approach toward a ball falling fast pays nothing
        let mut low = under_ball(Team::Blue, 300.0);
        let mut near = under_ball(Team::Blue, 800.0);
        low.ball.vel.z = -800.0;
        near.ball.vel.z = -800.0;
        let mut memory = vec![0.0; r.memory()];
        assert_eq!(step(&r, &low, &near, false, &mut memory), 0.0);
        // Closing on the ball: none still, half at half setup_closing_full, full from it
        let approach = |r: &FlipReset, closing: f32| {
            let mut memory = vec![0.0; r.memory()];
            let low = under_ball(Team::Blue, 300.0);
            let mut near = under_ball(Team::Blue, 800.0);
            near.cars[0].body.vel = (near.ball.pos - near.cars[0].body.pos).normalize() * closing;
            step(r, &low, &near, false, &mut memory)
        };
        let closing = FlipReset {
            setup_closing_full: 800.0,
            ..reward()
        };
        // (the start, the car still, has potential 0 with the setting: the whole potential pays)
        let full = approach(&closing, 1200.0);
        assert!(full > 0.0 && approach(&reward(), 0.0) > 0.0);
        assert_eq!(approach(&closing, 0.0), 0.0);
        assert!((approach(&closing, 400.0) - full * 0.5).abs() < 1e-4);
        assert!((approach(&closing, 800.0) - full).abs() < 1e-4);
        assert_eq!(reward().setup_closing_full, 0.0);
    }

    /// reset_apex_bonus pays a reset near the top of the ball's arc more: full with the ball rising or still,
    /// none with it falling at reset_apex_zero_fall; off by default.
    #[test]
    fn a_reset_near_the_top_of_the_arc_pays_more_with_reset_apex_bonus() {
        let pay = |r: &FlipReset, ball_vz: f32| {
            let mut memory = vec![0.0; r.memory()];
            let mut near = under_ball(Team::Blue, 800.0);
            near.ball.vel.z = ball_vz;
            let mut got = reset_on(near.clone());
            got.ball.vel.z = ball_vz;
            step(r, &near, &got, true, &mut memory)
        };
        let base = reward();
        assert_eq!(pay(&base, 300.0), pay(&base, -700.0), "off: the same");
        let apex = FlipReset {
            reset_apex_bonus: 2.0,
            ..reward()
        };
        let plain = pay(&base, 0.0);
        assert!((pay(&apex, 300.0) / plain - 3.0).abs() < 1e-3);
        assert!((pay(&apex, 0.0) / plain - 3.0).abs() < 1e-3);
        assert!((pay(&apex, -400.0) / plain - 2.0).abs() < 1e-3);
        assert!((pay(&apex, -900.0) / plain - 1.0).abs() < 1e-3);
    }

    /// far_firm_reward pays a game's first reset far out and high at once for a firm upward hit (the car
    /// moving into the ball at far_firm_closing_full or more and not falling: all; part between), on top of the
    /// held reset pay; next to nothing for a soft falling catch; nothing close to the net, low, for a later reset,
    /// or with it off (the default).
    #[test]
    fn a_firm_upward_first_reset_far_out_pays_at_once_with_far_firm_reward() {
        let held = FlipReset {
            close_reset_dist: 3000.0,
            far_reset_min_height: 800.0,
            far_reset_needs_next: true,
            ..reward()
        };
        let firm = FlipReset {
            far_firm_reward: 4.0,
            ..held
        };
        assert_eq!(reward().far_firm_reward, 0.0);
        for team in [Team::Blue, Team::Orange] {
            // under_ball: the ball 900 up, 6,120 from the goal line (far), the car 100 behind and below it
            let reset = |r: &FlipReset, vel: Vec3A| {
                let mut memory = vec![0.0; r.memory()];
                let mut near = under_ball(team, 800.0);
                near.cars[0].body.vel = as_blue(team, vel);
                let pay = step(r, &near, &reset_on(near.clone()), true, &mut memory);
                (pay, Track::load(&memory[Clock::SIZE..]).resets)
            };
            let rising = Vec3A::new(0.0, 500.0, 500.0); // into the ball at about 707 uu/s
            assert_eq!(
                reset(&held, rising),
                (0.0, 1),
                "the far reset itself is held"
            );
            let (pay, resets) = reset(&firm, rising);
            assert!(
                (pay - firm.far_firm_reward).abs() < 1e-4 && resets == 1,
                "{pay}"
            );
            // A soft touch, or the car falling fast: nothing
            assert_eq!(reset(&firm, Vec3A::new(0.0, 100.0, 50.0)).0, 0.0, "soft");
            assert_eq!(
                reset(&firm, Vec3A::new(0.0, 1800.0, -700.0)).0,
                0.0,
                "falling"
            );
        }
        let blue_car = |pos: Vec3A, vel: Vec3A| {
            let mut c = under_ball(Team::Blue, 0.0).cars[0];
            c.body.pos = pos;
            c.body.vel = vel;
            c
        };
        let ball_at = |pos: Vec3A| Body {
            pos,
            ..Body::RESTING
        };
        let far_ball = Vec3A::new(0.0, -1000.0, 900.0);
        let below = Vec3A::new(0.0, -1000.0, 800.0);
        // Straight up into it at 375: half between 150 and 600
        let half = firm.far_firm(
            &blue_car(below, Vec3A::Z * 375.0),
            &ball_at(far_ball),
            far_ball,
        );
        assert!((half - 0.5).abs() < 1e-4, "{half}");
        // Into it at 600 from beside (level) with the car falling at 300: half; a soft falling catch: a little
        let beside = Vec3A::new(0.0, -1100.0, 900.0);
        let fall = |closing: f32, vz: f32| {
            firm.far_firm(
                &blue_car(beside, Vec3A::new(0.0, closing, vz)),
                &ball_at(far_ball),
                far_ball,
            )
        };
        assert!((fall(600.0, -300.0) - 0.5).abs() < 1e-4);
        assert!(
            (0.0..0.1).contains(&fall(200.0, -400.0)),
            "{}",
            fall(200.0, -400.0)
        );
        // Close to the net, or low: nothing
        let close = Vec3A::new(0.0, 3000.0, 900.0);
        let up = Vec3A::Z * 800.0;
        assert_eq!(
            firm.far_firm(
                &blue_car(close - Vec3A::Z * 100.0, up),
                &ball_at(close),
                close
            ),
            0.0
        );
        let low = Vec3A::new(0.0, -1000.0, 700.0);
        assert_eq!(
            firm.far_firm(&blue_car(low - Vec3A::Z * 100.0, up), &ball_at(low), low),
            0.0
        );
        // A later reset: nothing more than without the setting (only the first of a game)
        let second_reset = |r: &FlipReset| {
            let mut memory = vec![0.0; r.memory()];
            let mut near = under_ball(Team::Blue, 800.0);
            near.cars[0].body.vel = Vec3A::new(0.0, 500.0, 500.0);
            step(r, &near, &reset_on(near.clone()), true, &mut memory);
            let mut again = near.clone();
            again.cars[0].has_flipped = true;
            again.cars[0].has_jumped = true;
            again.cars[0].can_flip = false;
            again.cars[0].on_ground = false;
            for _ in 0..60 {
                step(r, &again, &again, false, &mut memory);
            }
            let mut second = again.clone();
            second.ball.pos.y += 600.0;
            second.cars[0].body.pos.y += 600.0;
            let pay = step(r, &second, &reset_on(second.clone()), true, &mut memory);
            (pay, Track::load(&memory[Clock::SIZE..]).resets)
        };
        let (with, resets) = second_reset(&firm);
        assert_eq!(resets, 2);
        assert!((with - second_reset(&held).0).abs() < 1e-4, "{with}");
    }

    /// far_firm_setup pays the approach to a game's first reset far out (it pays nothing there with
    /// far_reset_needs_next), less the faster the car falls; not for a later reset, and off by default.
    #[test]
    fn the_approach_to_a_far_first_reset_pays_with_far_firm_setup() {
        let held = FlipReset {
            close_reset_dist: 3000.0,
            far_reset_min_height: 800.0,
            far_reset_needs_next: true,
            setup_closing_full: 600.0,
            ..reward()
        };
        let setup = FlipReset {
            far_firm_setup: true,
            ..held
        };
        assert!(!reward().far_firm_setup);
        // under_ball's ball is 900 up, 6,120 from the goal line (far); the car flies in from behind and below
        let approach = |r: &FlipReset, team: Team, car_vz: f32| {
            let mut memory = vec![0.0; r.memory()];
            let low = under_ball(team, 300.0);
            let mut near = under_ball(team, 700.0);
            near.cars[0].body.pos = near.ball.pos - as_blue(team, Vec3A::new(0.0, 400.0, 200.0));
            let to_ball = (near.ball.pos - near.cars[0].body.pos).normalize();
            near.cars[0].body.vel = to_ball * 800.0;
            near.cars[0].body.vel.z = car_vz;
            step(r, &low, &near, false, &mut memory)
        };
        for team in [Team::Blue, Team::Orange] {
            assert_eq!(
                approach(&held, team, 400.0),
                0.0,
                "far: nothing without the setting"
            );
            let rising = approach(&setup, team, 400.0);
            assert!(rising > 0.0, "{rising}");
            let falling = approach(&setup, team, -300.0);
            assert!(falling > 0.0 && falling < rising, "{falling} {rising}");
            assert_eq!(approach(&setup, team, -700.0), 0.0, "falling fast");
        }
        // A later reset's approach: nothing far out, as before
        let mut memory = vec![0.0; setup.memory()];
        let near = under_ball(Team::Blue, 800.0);
        step(&setup, &near, &reset_on(near.clone()), true, &mut memory);
        assert_eq!(Track::load(&memory[Clock::SIZE..]).resets, 1);
        assert_eq!(
            setup.setup_potential(&near.cars[0], &near.ball, 1, false),
            0.0,
            "not the first"
        );
    }

    /// reset_min_closing_speed counts a reset only as a firm touch: the car moving into the ball on the step
    /// before.
    #[test]
    fn a_reset_counts_only_as_a_firm_touch_with_reset_min_closing_speed() {
        let r = FlipReset {
            reset_min_closing_speed: 300.0,
            ..reward()
        };
        let reset_closing = |r: &FlipReset, closing: f32| {
            let mut memory = vec![0.0; r.memory()];
            let mut near = under_ball(Team::Orange, 800.0);
            near.cars[0].body.vel = (near.ball.pos - near.cars[0].body.pos).normalize() * closing;
            step(r, &near, &reset_on(near.clone()), true, &mut memory);
            FlipReset::holds_counted_reset(&memory)
        };
        assert!(reset_closing(&r, 400.0));
        assert!(!reset_closing(&r, 100.0), "a graze");
        assert!(reset_closing(&reward(), 0.0), "off by default");
    }

    /// reset_max_goal_dist counts a reset only with the ball that near the opponents' goal line, for both teams;
    /// the approach farther out pays nothing.
    #[test]
    fn a_reset_counts_only_near_the_net_with_reset_max_goal_dist() {
        let r = FlipReset {
            reset_max_goal_dist: 3000.0,
            ..reward()
        };
        for team in [Team::Blue, Team::Orange] {
            // 3,620 and 2,620 from the goal line
            let (_, _, far) = resets_at(&r, team, &[Vec3A::new(0.0, 1500.0, 900.0)]);
            let (_, _, near) = resets_at(&r, team, &[Vec3A::new(0.0, 2500.0, 900.0)]);
            assert_eq!((far, near), (0, 1), "{team:?}");
            let (pay, _, _) = resets_at(&r, team, &[Vec3A::new(0.0, 1500.0, 900.0)]);
            assert_eq!(pay, vec![0.0]);
            let (_, _, anywhere) = resets_at(&reward(), team, &[Vec3A::new(0.0, 1500.0, 900.0)]);
            assert_eq!(anywhere, 1);
            let (low, air) = (under_ball(team, 300.0), under_ball(team, 800.0));
            let mut memory = vec![0.0; r.memory()];
            assert_eq!(step(&r, &low, &air, false, &mut memory), 0.0, "6,120 out");
        }
    }

    /// The reset step under the ball at `spot` (Blue's view, mirrored for Orange): before it, and after it with the
    /// flip back.
    fn reset_spot(team: Team, spot: Vec3A) -> (GameSnapshot, GameSnapshot) {
        let mut pre = under_ball(team, 800.0);
        let by = as_blue(team, spot - Vec3A::new(0.0, -1000.0, 900.0));
        pre.ball.pos += by;
        pre.cars[0].body.pos += by;
        let got = reset_on(pre.clone());
        (pre, got)
    }

    /// A reset with the ball at each of `spots`, 14 steps apart, each followed by a dodge that misses the ball: every
    /// reset step's pay, the whole episode's, and how many resets counted.
    fn resets_at(r: &FlipReset, team: Team, spots: &[Vec3A]) -> (Vec<f32>, f32, i32) {
        let mut memory = vec![0.0; r.memory()];
        let (mut each, mut total) = (Vec::new(), 0.0);
        let mut last: Option<GameSnapshot> = None;
        for &spot in spots {
            let (pre, got) = reset_spot(team, spot);
            if let Some(last) = &last {
                total += step(r, last, &pre, false, &mut memory);
            }
            let pay = step(r, &pre, &got, true, &mut memory);
            each.push(pay);
            total += pay;
            total += step(r, &got, &pre, false, &mut memory); // The flip used again: a dodge
            for _ in 0..12 {
                total += step(r, &pre, &pre, false, &mut memory);
            }
            last = Some(pre);
        }
        (each, total, Track::load(&memory[Clock::SIZE..]).resets)
    }

    fn placed() -> FlipReset {
        FlipReset {
            close_reset_dist: 2048.0,
            far_reset_min_height: 1100.0,
            reset_height_bonus: 1.0,
            reset_height_full: 1800.0,
            ..reward()
        }
    }

    /// With close_reset_dist a reset beyond it from the goal line pays (its own pay and its follow-through)
    /// only with the ball at least far_reset_min_height up, more higher; a far low one pays nothing but still counts;
    /// a close one pays as without it. Off by default: the same pay anywhere.
    #[test]
    fn a_far_reset_pays_only_high_and_a_close_one_as_before() {
        let far = |z: f32| Vec3A::new(0.0, -1000.0, z); // 6,120 from the goal line
        let close = |z: f32| Vec3A::new(0.0, 3500.0, z); // 1,620
        for team in [Team::Blue, Team::Orange] {
            let r = placed();
            let one = |r: &FlipReset, spot: Vec3A| resets_at(r, team, &[spot]);
            // Far and low: nothing at all, yet counted
            let (each, total, resets) = one(&r, far(900.0));
            assert_eq!((each[0], total, resets), (0.0, 0.0, 1), "{team:?}");
            // Far and high: from x1 at 1,100 to x2 at 1,800 (reset_height_bonus 1), and the follow-through
            let (at_min, _, _) = one(&r, far(1100.0));
            let (mid, _, _) = one(&r, far(1450.0));
            let (top, top_total, _) = one(&r, far(1800.0));
            let base = at_min[0];
            assert!(at_min[0] > r.reset_reward, "{at_min:?}");
            assert!((mid[0] / base - 1.5).abs() < 1e-3 && (top[0] / base - 2.0).abs() < 1e-3);
            assert!(top_total > top[0] + 0.9 * r.follow_reward, "{top_total}");
            // Close: as without the setting (the height bonus from min_ball_height), low or high
            let plain = FlipReset {
                close_reset_dist: 0.0,
                ..placed()
            };
            for z in [400.0, 900.0, 1800.0] {
                let (a, ta, _) = one(&r, close(z));
                let (b, tb, _) = one(&plain, close(z));
                assert!(a[0] > 0.0 && a[0] == b[0] && ta == tb, "{z} {a:?} {b:?}");
            }
            // Defaults: off, a far low reset pays as a close one would
            let d = reward();
            assert_eq!(d.close_reset_dist, 0.0);
            let (far_low, far_total, _) = one(&d, far(900.0));
            assert!(far_low[0] > d.reset_reward && far_total > far_low[0]);
            let others = FlipReset {
                far_reset_min_height: 5000.0,
                delayed_bonus_close_only: true,
                ..reward()
            };
            assert_eq!(one(&others, far(900.0)), (far_low, far_total, 1));
        }
    }

    /// A later reset pays (its own pay and follow-through) only for carrying the ball toward the net since the
    /// previous one: none below multi_reset_min_progress, full at multi_reset_full_progress; at most
    /// max_resets_per_episode count. Off by default (credit from the first unit).
    #[test]
    fn a_later_reset_pays_only_after_carrying_the_ball_closer() {
        let r = FlipReset {
            max_resets_per_episode: 3,
            multi_reset_min_progress: 300.0,
            multi_reset_full_progress: 1000.0,
            ..reward()
        };
        let at = |y: f32| Vec3A::new(0.0, y, 900.0);
        for team in [Team::Blue, Team::Orange] {
            // Toward the net 1,000 at a time: each pays (the later ones more: multi-reset style)
            let (chain, _, n) =
                resets_at(&r, team, &[at(-1000.0), at(0.0), at(1000.0), at(2000.0)]);
            assert_eq!(n, 3, "the fourth is over the limit");
            assert!(
                chain[1] > chain[0] && chain[2] > chain[0] && chain[3] == 0.0,
                "{chain:?}"
            );
            // In place (200 forward, then back): nothing after the first
            let (still, still_total, n) =
                resets_at(&r, team, &[at(-1000.0), at(-800.0), at(-1000.0)]);
            assert_eq!(n, 3);
            assert!(
                still[0] > 0.0 && still[1] == 0.0 && still[2] == 0.0,
                "{still:?}"
            );
            let (first, first_total, _) = resets_at(&r, team, &[at(-1000.0)]);
            assert_eq!(
                (first[0], first_total),
                (still[0], still_total),
                "no follow-through paid either"
            );
            // Halfway between 300 and 1,000 (650): half the pay of a full-credit reset at the same spot
            let full = FlipReset {
                multi_reset_min_progress: 0.0,
                multi_reset_full_progress: 650.0,
                ..r
            };
            let half = resets_at(&r, team, &[at(-1000.0), at(-350.0)]).0;
            let whole = resets_at(&full, team, &[at(-1000.0), at(-350.0)]).0;
            assert!(
                (half[1] / whole[1] - 0.5).abs() < 1e-4,
                "{half:?} {whole:?}"
            );
            // The default: 200 of progress pays 200 / 800 of a reset
            let d = FlipReset {
                max_resets_per_episode: 3,
                ..reward()
            };
            assert_eq!(d.multi_reset_min_progress, 0.0);
            let small = resets_at(&d, team, &[at(-1000.0), at(-800.0)]).0;
            let big = resets_at(&d, team, &[at(-1000.0), at(-200.0)]).0;
            assert!(
                small[1] > 0.0 && (small[1] / big[1] - 0.25).abs() < 0.02,
                "{small:?} {big:?}"
            );
        }
    }

    /// With far_reset_needs_next a far high reset is paid only when the next counted reset carries the ball
    /// closer (times that one's progress credit); one far reset alone, or one followed by a reset in place, pays
    /// nothing; its follow-through pays nothing; a close reset pays at once.
    #[test]
    fn a_far_reset_can_be_paid_only_when_a_closer_reset_follows() {
        let at_once = FlipReset {
            max_resets_per_episode: 3,
            multi_reset_min_progress: 300.0,
            multi_reset_full_progress: 1000.0,
            ..placed()
        };
        let r = FlipReset {
            far_reset_needs_next: true,
            ..at_once
        };
        let high = |y: f32| Vec3A::new(0.0, y, 1500.0);
        for team in [Team::Blue, Team::Orange] {
            // Alone: nothing at all (paid at once, the reset and its follow-through)
            assert_eq!(resets_at(&r, team, &[high(-1000.0)]).1, 0.0, "{team:?}");
            let (alone, alone_total, _) = resets_at(&at_once, team, &[high(-1000.0)]);
            assert!(alone[0] > 0.0 && alone_total > alone[0]);
            // Far, far 1,000 closer, then close: each far one paid at the next reset, the close one at once
            let spots = [high(-1000.0), high(0.0), Vec3A::new(0.0, 3500.0, 900.0)];
            let (chain, _, n) = resets_at(&r, team, &spots);
            let (paid, _, _) = resets_at(&at_once, team, &spots);
            assert_eq!(n, 3);
            assert_eq!(chain[0], 0.0);
            assert!((chain[1] - paid[0]).abs() < 1e-4, "{chain:?} {paid:?}");
            assert!(
                (chain[2] - paid[1] - paid[2]).abs() < 1e-4,
                "{chain:?} {paid:?}"
            );
            // Far, then a reset in place (200 closer): nothing
            let (still, still_total, _) = resets_at(&r, team, &[high(-1000.0), high(-800.0)]);
            assert_eq!((still, still_total), (vec![0.0, 0.0], 0.0));
            // Far, then 650 closer: half the held pay
            let (half, _, _) = resets_at(&r, team, &[high(-1000.0), high(-350.0)]);
            assert!(
                (half[1] - 0.5 * alone[0]).abs() < 1e-4,
                "{half:?} {alone:?}"
            );
        }
    }

    /// With close_reset_dist the approach to a far ball pays only where a reset would be paid at once (high
    /// enough, and not held by far_reset_needs_next); close, it pays as without it.
    #[test]
    fn the_approach_to_a_far_reset_pays_only_where_the_reset_would() {
        let approach = |r: &FlipReset, ball: Vec3A| {
            let mut memory = vec![0.0; r.memory()];
            let mut from = under_ball(Team::Blue, 300.0);
            let mut to = under_ball(Team::Blue, 800.0);
            for g in [&mut from, &mut to] {
                let by = ball - g.ball.pos;
                g.ball.pos += by;
                g.cars[0].body.pos += by;
            }
            step(r, &from, &to, false, &mut memory)
        };
        let far_low = Vec3A::new(0.0, -1000.0, 900.0);
        let far_high = Vec3A::new(0.0, -1000.0, 1500.0);
        let close = Vec3A::new(0.0, 3500.0, 900.0);
        let held = FlipReset {
            far_reset_needs_next: true,
            ..placed()
        };
        for spot in [far_low, far_high, close] {
            assert!(approach(&reward(), spot) > 0.0, "{spot}");
        }
        assert_eq!(approach(&placed(), far_low), 0.0);
        assert!(approach(&placed(), far_high) > 0.0);
        assert_eq!(approach(&held, far_high), 0.0);
        assert_eq!(approach(&held, close), approach(&reward(), close));
    }

    /// The reset at `spot`, held 4 steps off the ball, a dodge, and 6 steps later the hit sending the ball goalward at
    /// 1,500 (a delayed use: 0.33 s to the dodge, 0.4 s to the hit): the hit step's pay and the reset's.
    fn delayed_use_at(r: &FlipReset, team: Team, spot: Vec3A) -> (f32, f32) {
        let mut memory = vec![0.0; r.memory()];
        let (pre, got) = reset_spot(team, spot);
        let reset = step(r, &pre, &got, true, &mut memory);
        let mut off = got.clone();
        off.cars[0].on_ground = false;
        off.cars[0].body.pos -= as_blue(team, Vec3A::new(0.0, 150.0, 0.0));
        step(r, &got, &off, false, &mut memory);
        for _ in 0..3 {
            step(r, &off, &off, false, &mut memory);
        }
        let mut dodge = off.clone();
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        step(r, &off, &dodge, false, &mut memory);
        for _ in 0..5 {
            step(r, &dodge, &dodge, false, &mut memory);
        }
        let mut hit = dodge.clone();
        hit.ball.vel = as_blue(team, Vec3A::new(0.0, 1500.0, 0.0));
        let used = step(r, &dodge, &hit, true, &mut memory);
        assert!(FlipReset::dodge_use_tick(&memory).is_some());
        (used, reset)
    }

    /// delayed_bonus_close_only (with close_reset_dist) pays the delayed-dodge bonus only for a hit within
    /// close_reset_dist of the goal line; farther out a delayed dodge pays as a plain one.
    #[test]
    fn the_delayed_dodge_bonus_can_be_paid_only_close_to_the_net() {
        let r = FlipReset {
            close_reset_dist: 2048.0,
            far_reset_min_height: 0.0,
            ..reward()
        };
        let only_close = FlipReset {
            delayed_bonus_close_only: true,
            ..r
        };
        for team in [Team::Blue, Team::Orange] {
            let far = Vec3A::new(0.0, -1000.0, 900.0);
            let (with, reset) = delayed_use_at(&r, team, far);
            let (without, _) = delayed_use_at(&only_close, team, far);
            // The use: quality 1 (1,500 goalward) x (1 + delayed_use_bonus 1), now x1: use_reward x style less
            let style = reset / r.reset_reward;
            assert!(
                (with - without - r.use_reward * style).abs() < 1e-3,
                "{with} {without} {style}"
            );
            let close = Vec3A::new(0.0, 3500.0, 900.0);
            assert_eq!(
                delayed_use_at(&r, team, close),
                delayed_use_at(&only_close, team, close)
            );
        }
    }

    /// use_reach_short_zero pays a shot landing short part of its use (linearly less, 0 that far short) and
    /// charges the wasted dodge only for the rest; 400 (the default) is air_shot_aim.
    #[test]
    fn use_reach_short_zero_pays_a_shot_landing_a_little_short_in_part() {
        let with = |short_zero: f32| FlipReset {
            use_aim: true,
            use_aim_reach: true,
            use_reach_short_zero: short_zero,
            wasted_dodge_cost: 2.0,
            ..reward()
        };
        assert_eq!(reward().use_reach_short_zero, 400.0);
        // The hit ball at (0, -1000, 900) in dodge_after_reset, at 2,600 uu/s: down about 1,150 short
        let at = Vec3A::new(0.0, -1000.0, 900.0);
        let short = Vec3A::new(0.0, 2600.0, 200.0);
        let path = air_path_to_goal_line(at, short, Team::Blue);
        assert!(path.short > 1000.0 && path.short < 1300.0, "{}", path.short);
        let aim = air_shot_aim_short(Team::Blue, &ball(at, short), 2000.0);
        assert!((aim - (1.0 - path.short / 2000.0)).abs() < 1e-4, "{aim}");
        assert_eq!(air_shot_aim(Team::Blue, &ball(at, short)), 0.0);
        for team in [Team::Blue, Team::Orange] {
            let b = ball(as_blue(team, at), as_blue(team, short));
            assert_eq!(air_shot_aim_short(team, &b, 400.0), air_shot_aim(team, &b));
            assert!((air_shot_aim_short(team, &b, 2000.0) - aim).abs() < 1e-4);
        }
        let (strict, _) = dodge_after_reset(&with(400.0), Some(short));
        let (soft, tick) = dodge_after_reset(&with(2000.0), Some(short));
        assert_eq!(tick, Some(24.0));
        // Quality 1 (2,600 goalward) times the aim, and the wasted dodge 2 x (1 - aim) instead of 2
        let style = strict[0] / reward().reset_reward;
        let use_pay = reward().use_reward * style;
        assert!(
            (soft[2] - strict[2] - (use_pay + 2.0) * aim).abs() < 1e-3,
            "{strict:?} {soft:?} {aim}"
        );
        // In the goal mouth in the air: 1 either way
        let carries = Vec3A::new(0.0, 3000.0, 300.0);
        assert_eq!(
            dodge_after_reset(&with(400.0), Some(carries)),
            dodge_after_reset(&with(2000.0), Some(carries))
        );
    }

    /// In RocketSim, a ball hit at the net from 2,000 uu at a good speed reaches the goal line inside the mouth
    /// in the air, where air_shot_aim puts it (no drag: within a few uu); one from 5,000 uu lands short first, and
    /// air_shot_aim gives it 0.
    #[test]
    fn the_air_shot_aim_matches_rocketsim_near_and_far() {
        use crate::sim::Sim;
        use crate::testing::meshes_loaded;

        if !meshes_loaded() {
            return;
        }
        let mut sim = Sim::new(0, false, 1);
        let (mut snap, mut events) = (GameSnapshot::default(), StepEvents::default());
        // (where, how fast, air_shot_aim, reaches the line in the air)
        let shots = [
            (
                Vec3A::new(-300.0, BACK_WALL_Y - 2000.0, 500.0),
                Vec3A::new(100.0, 2200.0, 250.0),
                1.0,
                true,
            ),
            (
                Vec3A::new(300.0, BACK_WALL_Y - 5000.0, 300.0),
                Vec3A::new(0.0, 2000.0, 200.0),
                0.0,
                false,
            ),
        ];
        for (pos, vel, aim, reaches) in shots {
            for team in [Team::Blue, Team::Orange] {
                let start = ball(as_blue(team, pos), as_blue(team, vel));
                assert_eq!(air_shot_aim(team, &start), aim);
                let path = air_path_to_goal_line(start.pos, start.vel, team);
                sim.place(start, &[]);
                let mut crossed = None;
                for _ in 0..600 {
                    events.clear(0);
                    sim.step(1, &mut events);
                    sim.read(&mut snap);
                    if events.ball_on_field_floor {
                        break;
                    }
                    if snap.ball.pos.y * attack_sign(team) >= BACK_WALL_Y {
                        crossed = Some(snap.ball.pos);
                        break;
                    }
                }
                assert_eq!(crossed.is_some(), reaches, "{team:?} {pos}");
                if let Some(at) = crossed {
                    let x = at.x * attack_sign(team);
                    assert!(
                        x.abs() < GOAL_HALF_WIDTH - BALL_RADIUS && at.z < GOAL_HEIGHT - BALL_RADIUS,
                        "{at}"
                    );
                    assert!(
                        (x - path.x).abs() < 15.0 && (at.z - path.z).abs() < 30.0,
                        "sim {at}, projected {} {}",
                        path.x,
                        path.z
                    );
                }
            }
        }
    }

    /// A start in the air with the flip unused under a high ball (a start part-way through a reset): a counted
    /// reset with start_reset_counts, used by a dodge into the ball; nothing without it, on the floor or far away.
    #[test]
    fn a_start_holding_its_flip_under_the_ball_holds_a_counted_reset_with_start_reset_counts() {
        let counts = FlipReset {
            start_reset_counts: true,
            ..reward()
        };
        let mut start = reset_on(under_ball(Team::Orange, 800.0));
        start.cars[0].on_ground = false;
        let first = |r: &FlipReset, start: &GameSnapshot| {
            let mut memory = vec![0.0; r.memory()];
            let pay = step(r, start, start, false, &mut memory);
            (pay, memory)
        };
        let (pay, memory) = first(&reward(), &start);
        assert!(pay == 0.0 && FlipReset::counted_reset_tick(&memory).is_none());
        let (pay, mut memory) = first(&counts, &start);
        assert_eq!(pay, 0.0, "a reset held from the start is not paid itself");
        assert!(FlipReset::holds_counted_reset(&memory));
        assert_eq!(FlipReset::counted_reset_tick(&memory), Some(0.0));
        // Off the ball, then a dodge into it toward the goal: a use
        let mut off = start.clone();
        off.cars[0].body.pos.y += 150.0; // Orange attacks -y: away from the ball, toward its own goal
        step(&counts, &start, &off, false, &mut memory);
        let mut dodge = off.clone();
        dodge.cars[0].has_flipped = true;
        dodge.cars[0].can_flip = false;
        dodge.ball.vel = Vec3A::new(0.0, -1500.0, 0.0);
        let used = step(&counts, &off, &dodge, true, &mut memory);
        assert!(used >= counts.use_reward, "{used}");
        assert_eq!(FlipReset::dodge_use_tick(&memory), Some(24.0));
        // Its flip used, on the floor, or far from the ball: nothing held
        let changes: [fn(&mut GameSnapshot); 4] = [
            |g: &mut GameSnapshot| g.cars[0].has_jumped = true,
            |g: &mut GameSnapshot| g.cars[0].body.pos.z = 17.0,
            |g: &mut GameSnapshot| g.cars[0].body.pos.y += 700.0,
            |g: &mut GameSnapshot| g.ball.pos.z = 299.0,
        ];
        for change in changes {
            let mut other = start.clone();
            change(&mut other);
            let (_, memory) = first(&counts, &other);
            assert!(FlipReset::counted_reset_tick(&memory).is_none());
        }
        // Long since its last wheel contact (a drive off a wall keeps the flip too): nothing held
        for (air_time, held) in [(START_RESET_MAX_AIR_TIME, true), (2.5, false)] {
            let mut ev = quiet(1);
            ev.prev_details[0].air_time = air_time;
            let mut memory = vec![0.0; counts.memory()];
            counts.value_remembering(&start, &start, &ev, 0, &mut memory);
            assert_eq!(FlipReset::holds_counted_reset(&memory), held, "{air_time}");
        }
    }
}
