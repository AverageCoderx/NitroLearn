//! A few well-known mechanics, detected from the state step by step, for the gameplay statistics:
//! aerial touches, dribbles, flicks, double taps, flip resets, shots, saves and
//! demolitions. Approximate by design, and each rule is written out below: it fires on the situations it
//! is built for, from the state at the end of each step and the events latched during it.

use rocketsim::Vec3A;

use crate::custom::library::aerial_kit::flip_reset_over_steps;
use crate::env::PlayStats;
use crate::snapshot::{Body, GameSnapshot, StepEvents, Team};

/// A touch with the car off the ground and the ball's centre above this (uu) is aerial: higher than a
/// car reaches without leaving the ground.
pub const AERIAL_BALL_Z: f32 = 300.0;
/// The ball is "on the car" with its centre this much above the car's centre (uu) and, sideways, within
/// `ON_CAR_REACH` of it: resting on the roof or the hood.
const ON_CAR_HEIGHT: (f32, f32) = (90.0, 230.0);
const ON_CAR_REACH: f32 = 110.0;
/// A dribble: the ball on the car, the car on the ground, for this long without a break.
const DRIBBLE_SECONDS: f32 = 0.75;
/// A flick: within this long of the ball leaving the car after at least `FLICK_CARRY_SECONDS` on it, the
/// car's own touch during a fresh dodge speeds the ball up by `FLICK_SPEED_GAIN` uu/s or more.
const FLICK_WINDOW_SECONDS: f32 = 0.5;
const FLICK_CARRY_SECONDS: f32 = 0.5;
const FLICK_SPEED_GAIN: f32 = 500.0;
/// A dodge counts as fresh this long after it started.
const FRESH_DODGE_SECONDS: f32 = 0.25;
/// Shots and saves: whether the ball, flying freely under gravity (no bounces), crosses a goal line inside
/// the goal within this long.
const SHOT_HORIZON_SECONDS: f32 = 1.5;
const GOAL_LINE_Y: f32 = 5120.0;
/// A flip reset: the car this close to the ball (uu, centres), as the flip_reset reward's default.
const FLIP_RESET_BALL_DIST: f32 = 350.0;
/// A reset musty goal (a counter): the car's team scores within this long (s) of the car's flip reset, with
/// the car's back flip (a dodge with backward torque) between them.
const RESET_MUSTY_GOAL_SECONDS: f32 = 6.0;
const GOAL_HALF_WIDTH: f32 = 892.755;
const GOAL_HEIGHT: f32 = 642.775;
/// The ball bounced off a backboard: it was moving toward a back wall and now moves away from it, with
/// its centre this close to the wall's line and above the goal.
const BACKBOARD_Y: f32 = 4800.0;
const GRAVITY_Z: f32 = -650.0;

/// What a car did in a step, as bits (`Mechanics::marks`; the viewer's timeline marks, `capture`).
pub const MARK_AERIAL_TOUCH: u32 = 1;
pub const MARK_DRIBBLE: u32 = 2;
pub const MARK_FLICK: u32 = 4;
pub const MARK_DOUBLE_TAP: u32 = 8;
pub const MARK_FLIP_RESET: u32 = 16;
pub const MARK_SHOT: u32 = 32;
pub const MARK_SAVE: u32 = 64;
pub const MARK_DEMO: u32 = 128;

/// Whether the ball, flying freely, crosses the goal line at `goal_y` inside the goal soon.
pub fn heading_into_goal(ball: &Body, goal_y: f32) -> bool {
    let (dy, vy) = (goal_y - ball.pos.y, ball.vel.y);
    if vy == 0.0 || dy.signum() != vy.signum() {
        return false;
    }
    let t = dy / vy;
    if t > SHOT_HORIZON_SECONDS {
        return false;
    }
    let x = ball.pos.x + ball.vel.x * t;
    let z = ball.pos.z + ball.vel.z * t + 0.5 * GRAVITY_Z * t * t;
    x.abs() < GOAL_HALF_WIDTH && z < GOAL_HEIGHT
}

/// The y of the goal `team` defends (Blue defends −y).
fn own_goal_y(team: Team) -> f32 {
    match team {
        Team::Blue => -GOAL_LINE_Y,
        Team::Orange => GOAL_LINE_Y,
    }
}

fn ball_on_car(ball: Vec3A, car: &Body) -> bool {
    let offset = ball - car.pos;
    let up = car.up();
    let height = offset.dot(up);
    let sideways = (offset - up * height).length();
    (ON_CAR_HEIGHT.0..=ON_CAR_HEIGHT.1).contains(&height) && sideways <= ON_CAR_REACH
}

#[derive(Clone, Copy, Debug, Default)]
struct CarTrack {
    /// Steps the ball has been on this car without a break, and whether that stretch already counted.
    on_car: u32,
    counted: bool,
    /// The last stretch on the car, and the steps since it ended (none: no stretch yet).
    carried: u32,
    since_carry: Option<u32>,
    /// Steps since this car's current dodge started (none: no dodge since it last landed).
    since_dodge: Option<u32>,
    /// No flip left in the air, waiting for a flip reset (`flip_reset_over_steps`).
    reset_waiting: bool,
    /// Steps since this car's last flip reset, and whether it has back-flipped since (musty goals).
    since_reset: Option<u32>,
    back_flip_after_reset: bool,
}

/// One arena's mechanic detector; `reset` at every kickoff.
#[derive(Clone, Debug)]
pub struct Mechanics {
    step_seconds: f32,
    cars: Vec<CarTrack>,
    /// A double tap in progress: the car whose aerial touch started it, and whether the ball has since
    /// come off the backboard it was sent at.
    double_tap: Option<(usize, bool)>,
    /// Per car, the `MARK_*` bits of what it did in the last step, counted or not (`nitro watch`'s marks).
    marks: Vec<u32>,
}

impl Mechanics {
    pub fn new(tick_skip: u32) -> Self {
        Self {
            step_seconds: tick_skip as f32 / 120.0,
            cars: Vec::new(),
            double_tap: None,
            marks: Vec::new(),
        }
    }

    /// Per car, what it did in the last step (`MARK_*` bits).
    pub fn marks(&self) -> &[u32] {
        &self.marks
    }

    pub fn reset(&mut self) {
        self.cars.clear();
        self.double_tap = None;
    }

    fn steps(&self, seconds: f32) -> u32 {
        (seconds / self.step_seconds).ceil() as u32
    }

    /// Adds what the counted cars did in the step from `prev` to `snap` to `stats`.
    pub fn update(
        &mut self,
        prev: &GameSnapshot,
        snap: &GameSnapshot,
        events: &StepEvents,
        counted: &[bool],
        stats: &mut PlayStats,
    ) {
        let cars = snap.cars.len();
        if self.cars.len() != cars || prev.cars.len() != cars {
            self.cars = vec![CarTrack::default(); cars];
        }
        self.marks.clear();
        self.marks.resize(cars, 0);
        let (dribble, carry, window, fresh) = (
            self.steps(DRIBBLE_SECONDS),
            self.steps(FLICK_CARRY_SECONDS),
            self.steps(FLICK_WINDOW_SECONDS),
            self.steps(FRESH_DODGE_SECONDS),
        );
        let musty_window = self.steps(RESET_MUSTY_GOAL_SECONDS);
        let speed_gain = snap.ball.vel.length() - prev.ball.vel.length();
        let touched: Vec<bool> = (0..cars)
            .map(|c| events.cars.get(c).is_some_and(|e| e.touched_ball))
            .collect();
        self.follow_double_tap(prev, snap, events, &touched, counted, stats);
        for (c, (before, now)) in prev.cars.iter().zip(&snap.cars).enumerate() {
            let counts = u64::from(counted.get(c).copied().unwrap_or(true));
            let track = &mut self.cars[c];
            // Dodges: a fresh one started this step, or not long ago; a landing ends it.
            track.since_dodge = if now.on_ground {
                None
            } else if now.has_flipped && !before.has_flipped {
                Some(0)
            } else {
                track.since_dodge.map(|s| s + 1)
            };
            // Dribbles: the ball on the car, the car on the ground.
            if now.on_ground && !now.is_demoed && ball_on_car(snap.ball.pos, &now.body) {
                track.on_car += 1;
                stats.ball_on_car += counts;
                if track.on_car >= dribble && !track.counted {
                    stats.dribbles += counts;
                    track.counted = true;
                    self.marks[c] |= MARK_DRIBBLE;
                }
            } else if track.on_car > 0 {
                (track.carried, track.since_carry) = (track.on_car, Some(0));
                (track.on_car, track.counted) = (0, false);
            } else {
                track.since_carry = track.since_carry.map(|s| s + 1);
            }
            // A flip reset: in the air with no flip left, and the flip back off the ball. Not only on
            // touch steps: wheels on the ball are no touch in RocketSim (rays, not the hitbox); it counts them as
            // on the ground, where the car often ends the step, and the dodge flag can clear a step later.
            let reset;
            (reset, track.reset_waiting) = flip_reset_over_steps(
                track.reset_waiting,
                before,
                now,
                snap.ball.pos,
                FLIP_RESET_BALL_DIST,
            );
            // Reset musty goals and the ball's speed into the net (a counter, looser than the rewards)
            track.since_reset = if reset {
                track.back_flip_after_reset = false;
                Some(0)
            } else {
                track.since_reset.map(|s| s + 1)
            };
            if now.has_flipped
                && !before.has_flipped
                && events.detail(c).flip_rel_torque.y < -0.5
                && track.since_reset.is_some()
            {
                track.back_flip_after_reset = true;
            }
            if snap.goal_scored == Some(now.team)
                && prev.goal_scored.is_none()
                && track.back_flip_after_reset
                && track.since_reset.is_some_and(|s| s <= musty_window)
            {
                stats.reset_musty_goals += counts;
                stats.reset_musty_goal_speed += counts as f64 * f64::from(snap.ball.vel.length());
                track.since_reset = None;
            }
            if reset {
                stats.flip_resets += counts;
                let goal_dist = (-own_goal_y(now.team) - snap.ball.pos.y).abs();
                stats.flip_reset_height += counts as f64 * f64::from(snap.ball.pos.z);
                stats.flip_reset_goal_dist += counts as f64 * f64::from(goal_dist);
                self.marks[c] |= MARK_FLIP_RESET;
            }
            if !touched[c] {
                continue;
            }
            let in_air = !now.on_ground;
            if in_air && snap.ball.pos.z > AERIAL_BALL_Z {
                stats.aerial_touches += counts;
                self.marks[c] |= MARK_AERIAL_TOUCH;
            }
            let carried = track.on_car.max(track.carried);
            let just_carried = track.on_car > 0 || track.since_carry.is_some_and(|s| s <= window);
            if carried >= carry
                && just_carried
                && track.since_dodge.is_some_and(|s| s <= fresh)
                && speed_gain >= FLICK_SPEED_GAIN
            {
                stats.flicks += counts;
                track.since_carry = None;
                self.marks[c] |= MARK_FLICK;
            }
            let (own, theirs) = (own_goal_y(now.team), -own_goal_y(now.team));
            // At the end of a goal's step the ball sits in the net, no longer "heading" anywhere: a goal
            // there counts as going in.
            let conceded = snap.goal_scored == Some(now.team.opposite());
            let scored = snap.goal_scored == Some(now.team);
            if heading_into_goal(&prev.ball, own)
                && !heading_into_goal(&snap.ball, own)
                && !conceded
            {
                stats.saves += counts;
                self.marks[c] |= MARK_SAVE;
            }
            if (scored || heading_into_goal(&snap.ball, theirs))
                && !heading_into_goal(&prev.ball, theirs)
            {
                stats.shots += counts;
                self.marks[c] |= MARK_SHOT;
            }
        }
        for (c, car_events) in events.cars.iter().enumerate() {
            if car_events.demoed_opponent {
                stats.demos += u64::from(counted.get(c).copied().unwrap_or(true));
                self.marks[c] |= MARK_DEMO;
            }
        }
    }

    /// A double tap: an aerial touch sends the ball at the opponents' backboard, it comes off it, and the
    /// same car touches it again in the air before anyone else does or it reaches the floor.
    fn follow_double_tap(
        &mut self,
        prev: &GameSnapshot,
        snap: &GameSnapshot,
        events: &StepEvents,
        touched: &[bool],
        counted: &[bool],
        stats: &mut PlayStats,
    ) {
        if let Some((car, false)) = self.double_tap {
            let wall = -own_goal_y(snap.cars[car].team);
            let toward = prev.ball.vel.y * wall.signum() > 0.0;
            let away = snap.ball.vel.y * wall.signum() <= 0.0;
            if toward
                && away
                && snap.ball.pos.y * wall.signum() > BACKBOARD_Y
                && snap.ball.pos.z > GOAL_HEIGHT
            {
                self.double_tap = Some((car, true));
            }
        }
        if events.ball_on_floor {
            self.double_tap = None;
        }
        for (c, &touch) in touched.iter().enumerate() {
            if !touch {
                continue;
            }
            let aerial = !snap.cars[c].on_ground && snap.ball.pos.z > AERIAL_BALL_Z;
            self.double_tap = match self.double_tap {
                Some((car, true)) if car == c && aerial => {
                    stats.double_taps += u64::from(counted.get(c).copied().unwrap_or(true));
                    self.marks[c] |= MARK_DOUBLE_TAP;
                    None
                }
                _ if aerial => Some((c, false)),
                _ => None,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use rocketsim::Mat3A;

    use super::*;
    use crate::snapshot::{CarDetail, CarEvents, CarSnapshot};

    const TICK_SKIP: u32 = 8; // 15 steps a second

    fn car(pos: Vec3A, on_ground: bool) -> CarSnapshot {
        CarSnapshot {
            team: Team::Blue,
            body: Body {
                pos,
                rot: Mat3A::IDENTITY,
                vel: Vec3A::ZERO,
                ang_vel: Vec3A::ZERO,
            },
            boost: 33.0,
            on_ground,
            has_jumped: !on_ground,
            has_double_jumped: false,
            has_flipped: false,
            can_flip: true,
            is_demoed: false,
            is_supersonic: false,
            demo_timer: 0.0,
            flip_time_left: 1.25,
        }
    }

    fn snap(ball: Vec3A, ball_vel: Vec3A, cars: Vec<CarSnapshot>) -> GameSnapshot {
        GameSnapshot {
            ball: Body {
                pos: ball,
                vel: ball_vel,
                ..Body::RESTING
            },
            cars,
            ..GameSnapshot::default()
        }
    }

    fn events(cars: usize, touches: &[usize]) -> StepEvents {
        let mut events = StepEvents::default();
        events.clear(cars);
        for &c in touches {
            events.cars[c] = CarEvents {
                touched_ball: true,
                ..CarEvents::default()
            };
        }
        events
    }

    /// Runs the steps and returns the statistics of every car.
    fn play(steps: &[(GameSnapshot, StepEvents)]) -> PlayStats {
        let mut mechanics = Mechanics::new(TICK_SKIP);
        let mut stats = PlayStats::default();
        for pair in steps.windows(2) {
            let ((prev, _), (now, events)) = (&pair[0], &pair[1]);
            mechanics.update(prev, now, events, &[true, true], &mut stats);
        }
        stats
    }

    /// Runs the steps and returns, per car, every mark it got.
    fn marked(steps: &[(GameSnapshot, StepEvents)]) -> Vec<u32> {
        let mut mechanics = Mechanics::new(TICK_SKIP);
        let mut all = Vec::new();
        for pair in steps.windows(2) {
            let ((prev, _), (now, events)) = (&pair[0], &pair[1]);
            mechanics.update(prev, now, events, &[true, true], &mut PlayStats::default());
            all.resize(mechanics.marks().len(), 0);
            all.iter_mut()
                .zip(mechanics.marks())
                .for_each(|(a, m)| *a |= m);
        }
        all
    }

    #[test]
    fn a_ball_carried_on_the_roof_for_a_second_is_one_dribble() {
        let roof = |t: usize| {
            let at = Vec3A::new(100.0 * t as f32, 0.0, 17.0);
            (
                snap(
                    at + Vec3A::new(10.0, 0.0, 140.0),
                    Vec3A::new(1500.0, 0.0, 0.0),
                    vec![car(at, true)],
                ),
                events(1, &[]),
            )
        };
        let stats = play(&(0..16).map(roof).collect::<Vec<_>>());
        assert_eq!((stats.dribbles, stats.ball_on_car), (1, 15));
        assert_eq!(
            marked(&(0..16).map(roof).collect::<Vec<_>>())[0],
            MARK_DRIBBLE
        );
        let short = play(&(0..8).map(roof).collect::<Vec<_>>());
        assert_eq!(
            short.dribbles, 0,
            "half a second on the roof is not a dribble"
        );
        let beside = |t: usize| {
            let at = Vec3A::new(100.0 * t as f32, 0.0, 17.0);
            (
                snap(
                    at + Vec3A::new(0.0, 200.0, 140.0),
                    Vec3A::ZERO,
                    vec![car(at, true)],
                ),
                events(1, &[]),
            )
        };
        assert_eq!(
            play(&(0..16).map(beside).collect::<Vec<_>>()).ball_on_car,
            0,
            "the ball beside the car"
        );
    }

    #[test]
    fn a_dodge_that_fires_a_carried_ball_away_is_a_flick_and_a_plain_touch_is_not() {
        let at = Vec3A::new(0.0, 0.0, 17.0);
        let mut steps: Vec<_> = (0..10)
            .map(|_| {
                (
                    snap(
                        at + Vec3A::new(0.0, 0.0, 140.0),
                        Vec3A::new(0.0, 500.0, 0.0),
                        vec![car(at, true)],
                    ),
                    events(1, &[]),
                )
            })
            .collect();
        let mut dodging = car(at + Vec3A::new(0.0, 0.0, 40.0), false);
        dodging.has_flipped = true;
        let fired = snap(
            at + Vec3A::new(0.0, 200.0, 300.0),
            Vec3A::new(0.0, 1800.0, 900.0),
            vec![dodging],
        );
        steps.push((fired.clone(), events(1, &[0])));
        assert_eq!(play(&steps).flicks, 1);
        assert_ne!(marked(&steps)[0] & MARK_FLICK, 0);
        let mut no_dodge = steps.clone();
        no_dodge.last_mut().unwrap().0.cars[0].has_flipped = false;
        assert_eq!(play(&no_dodge).flicks, 0, "no dodge, no flick");
        let mut soft = steps.clone();
        soft.last_mut().unwrap().0.ball.vel = Vec3A::new(0.0, 700.0, 0.0);
        assert_eq!(play(&soft).flicks, 0, "the ball barely sped up");
    }

    #[test]
    fn regaining_the_flip_by_touching_the_ball_in_the_air_is_a_flip_reset() {
        let high = Vec3A::new(0.0, 0.0, 900.0);
        let mut spent = car(high, false);
        spent.can_flip = false;
        let mut reset = spent;
        (reset.can_flip, reset.has_jumped) = (true, false);
        let ball = high + Vec3A::new(0.0, 0.0, 120.0);
        let steps = [
            (snap(ball, Vec3A::ZERO, vec![spent]), events(1, &[])),
            (snap(ball, Vec3A::ZERO, vec![reset]), events(1, &[0])),
        ];
        let stats = play(&steps);
        assert_eq!((stats.flip_resets, stats.aerial_touches), (1, 1));
        assert_eq!(marked(&steps)[0], MARK_FLIP_RESET | MARK_AERIAL_TOUCH);
        // The step ending with the wheels still on the ball (RocketSim: on the ground, and no touch) is
        // a reset too, counted once while they stay there.
        let mut on_ball = reset;
        on_ball.on_ground = true;
        let held = [
            steps[0].clone(),
            (snap(ball, Vec3A::ZERO, vec![on_ball]), events(1, &[])),
            (snap(ball, Vec3A::ZERO, vec![on_ball]), events(1, &[])),
            (snap(ball, Vec3A::ZERO, vec![reset]), events(1, &[])),
        ];
        assert_eq!(play(&held).flip_resets, 1);
        let held_stats = play(&held);
        assert_eq!(held_stats.flip_reset_height, f64::from(ball.z));
        assert_eq!(
            held_stats.flip_reset_goal_dist,
            f64::from(GOAL_LINE_Y - ball.y)
        );
        // ...and when the dodge flag clears a step after the wheels reached the ball, still on it.
        let mut landing = on_ball;
        landing.has_flipped = true;
        let late = [
            steps[0].clone(),
            (snap(ball, Vec3A::ZERO, vec![landing]), events(1, &[])),
            (snap(ball, Vec3A::ZERO, vec![on_ball]), events(1, &[])),
            (snap(ball, Vec3A::ZERO, vec![reset]), events(1, &[])),
        ];
        assert_eq!(play(&late).flip_resets, 1);
        let landed = [
            steps[0].clone(),
            (
                snap(
                    ball,
                    Vec3A::ZERO,
                    vec![car(Vec3A::new(0.0, 0.0, 17.0), true)],
                ),
                events(1, &[0]),
            ),
        ];
        assert_eq!(
            play(&landed).flip_resets,
            0,
            "landing gives the flip back too, but is no reset"
        );
    }

    /// A goal within 6 s of the car's flip reset, with its back flip between, is a reset musty goal, and the
    /// ball's speed into the net is summed; without the back flip (a front flip) it is not.
    #[test]
    fn a_goal_after_a_flip_reset_and_a_back_flip_is_a_reset_musty_goal() {
        let high = Vec3A::new(0.0, 0.0, 900.0);
        let mut spent = car(high, false);
        spent.can_flip = false;
        let mut reset = spent;
        (reset.can_flip, reset.has_jumped) = (true, false);
        let ball = high + Vec3A::new(0.0, 0.0, 120.0);
        let mut flipped = reset;
        (flipped.has_flipped, flipped.can_flip) = (true, false);
        let dodge = |torque_y: f32| {
            let mut ev = events(1, &[]);
            ev.details = vec![CarDetail::default()];
            ev.details[0].flip_rel_torque = Vec3A::new(0.0, torque_y, 0.0);
            ev
        };
        let play_with = |torque_y: f32| {
            let mut goal = snap(ball, Vec3A::new(0.0, 3000.0, 0.0), vec![flipped]);
            goal.goal_scored = Some(Team::Blue);
            play(&[
                (snap(ball, Vec3A::ZERO, vec![spent]), events(1, &[])),
                (snap(ball, Vec3A::ZERO, vec![reset]), events(1, &[])),
                (snap(ball, Vec3A::ZERO, vec![flipped]), dodge(torque_y)),
                (goal, events(1, &[])),
            ])
        };
        let musty = play_with(-1.0);
        assert_eq!(
            (musty.reset_musty_goals, musty.reset_musty_goal_speed),
            (1, 3000.0)
        );
        assert_eq!(play_with(1.0).reset_musty_goals, 0, "a front flip");
    }

    #[test]
    fn shots_and_saves_follow_where_the_ball_is_heading() {
        let blue = car(Vec3A::new(0.0, 3000.0, 17.0), true);
        let mut orange = car(Vec3A::new(0.0, 4500.0, 17.0), true);
        orange.team = Team::Orange;
        let rolling = snap(
            Vec3A::new(0.0, 3100.0, 93.0),
            Vec3A::new(0.0, 100.0, 0.0),
            vec![blue, orange],
        );
        let shot = snap(
            Vec3A::new(0.0, 3300.0, 150.0),
            Vec3A::new(0.0, 2500.0, 200.0),
            vec![blue, orange],
        );
        let cleared = snap(
            Vec3A::new(0.0, 4400.0, 150.0),
            Vec3A::new(1500.0, -2000.0, 300.0),
            vec![blue, orange],
        );
        let stats = play(&[
            (rolling.clone(), events(2, &[])),
            (shot.clone(), events(2, &[0])),
            (cleared, events(2, &[1])),
        ]);
        assert_eq!((stats.shots, stats.saves), (1, 1));
        assert!(
            heading_into_goal(&shot.ball, GOAL_LINE_Y)
                && !heading_into_goal(&shot.ball, -GOAL_LINE_Y)
        );
        let wide = Body {
            pos: Vec3A::new(0.0, 3300.0, 150.0),
            vel: Vec3A::new(1500.0, 2500.0, 0.0),
            ..Body::RESTING
        };
        assert!(!heading_into_goal(&wide, GOAL_LINE_Y), "it passes the post");
        // A touch in the step the ball goes in: the ball sits in the net at the step's end.
        let mut netted = snap(
            Vec3A::new(0.0, 5300.0, 150.0),
            Vec3A::new(0.0, 300.0, 0.0),
            vec![blue, orange],
        );
        netted.goal_scored = Some(Team::Blue);
        let stats = play(&[
            (rolling, events(2, &[])),
            (netted.clone(), events(2, &[0, 1])),
        ]);
        assert_eq!(
            (stats.shots, stats.saves),
            (1, 0),
            "Blue's touch scored; Orange's touch did not save it"
        );
        let stats = play(&[(shot, events(2, &[])), (netted, events(2, &[1]))]);
        assert_eq!(stats.saves, 0, "a goal conceded is no save");
    }

    #[test]
    fn an_aerial_off_the_backboard_and_a_second_touch_is_a_double_tap() {
        let air = |y: f32| car(Vec3A::new(0.0, y, 800.0), false);
        let steps = [
            (
                snap(
                    Vec3A::new(0.0, 3800.0, 1000.0),
                    Vec3A::new(0.0, 1000.0, 300.0),
                    vec![air(3600.0)],
                ),
                events(1, &[]),
            ),
            (
                snap(
                    Vec3A::new(0.0, 4200.0, 1100.0),
                    Vec3A::new(0.0, 2500.0, 300.0),
                    vec![air(3900.0)],
                ),
                events(1, &[0]),
            ),
            (
                snap(
                    Vec3A::new(0.0, 4950.0, 1200.0),
                    Vec3A::new(0.0, -1500.0, 100.0),
                    vec![air(4300.0)],
                ),
                events(1, &[]),
            ),
            (
                snap(
                    Vec3A::new(0.0, 4700.0, 1100.0),
                    Vec3A::new(0.0, 2000.0, -200.0),
                    vec![air(4600.0)],
                ),
                events(1, &[0]),
            ),
        ];
        assert_eq!(play(&steps).double_taps, 1);
        assert_ne!(marked(&steps)[0] & MARK_DOUBLE_TAP, 0);
        let mut floored = steps.clone();
        floored[2].1.ball_on_floor = true;
        assert_eq!(
            play(&floored).double_taps,
            0,
            "the ball reached the floor in between"
        );
        let mut no_wall = steps.clone();
        no_wall[2].0.ball.vel = Vec3A::new(0.0, 1500.0, 100.0);
        assert_eq!(
            play(&no_wall).double_taps,
            0,
            "it never came off the backboard"
        );
    }

    #[test]
    fn only_counted_cars_are_counted_and_demolitions_come_from_the_events() {
        let at = Vec3A::new(0.0, 0.0, 17.0);
        let mut events = events(2, &[]);
        events.cars[1].demoed_opponent = true;
        let mut mechanics = Mechanics::new(TICK_SKIP);
        let mut stats = PlayStats::default();
        let s = snap(
            Vec3A::new(0.0, 0.0, 93.0),
            Vec3A::ZERO,
            vec![car(at, true), car(at, true)],
        );
        mechanics.update(&s, &s, &events, &[true, false], &mut stats);
        assert_eq!(stats.demos, 0, "the demolishing car is not counted");
        assert_eq!(
            mechanics.marks(),
            [0, MARK_DEMO],
            "but marked: the viewer shows every car"
        );
        mechanics.update(&s, &s, &events, &[true, true], &mut stats);
        assert_eq!(stats.demos, 1);
        mechanics.update(&s, &s, &StepEvents::default(), &[true, true], &mut stats);
        assert_eq!(mechanics.marks(), [0, 0], "a mark lasts one step");
    }
}
