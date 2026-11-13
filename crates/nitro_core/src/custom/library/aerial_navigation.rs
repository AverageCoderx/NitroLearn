//! Aerial navigation: flying up at a high ball and facing it (aerial_navigation), and a stricter
//! version paid more the nearer the ball is (aerial_navigation_strict).

use crate::custom::prelude::*;

/// The cosine of the angle between the car's flight and the line to the ball (0 when either is zero).
fn flight_alignment(me: &CarSnapshot, ball: Vec3A) -> f32 {
    cosine(ball - me.body.pos, me.body.vel)
}

/// The cosine of the angle between the car's nose and the line to the ball.
fn facing(me: &CarSnapshot, ball: Vec3A) -> f32 {
    me.body
        .forward()
        .dot((ball - me.body.pos).normalize_or_zero())
}

/// Half the alignment, never below 0 (beginner), or the alignment times the car's speed over 2300.
fn flight_part(beginner: bool, me: &CarSnapshot, ball: Vec3A) -> f32 {
    let alignment = flight_alignment(me, ball);
    if beginner {
        (alignment * 0.5).max(0.0)
    } else {
        alignment * me.body.vel.length() / CAR_MAX_SPEED
    }
}

/// Every step the car flies upwards towards a ball above 400 uu (car above 200 uu, at most 3 times the
/// ball's height away across the field): up to 0.5 for flying straight at the ball, up to 0.25 for facing
/// it and up to 0.1 more for boosting then. 0 to 0.85 as a beginner; never negative.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialNavigation {
    /// The ball must be higher than this (uu).
    #[serde(default = "ball_height_400")]
    pub ball_height_min: f32,
    /// The car must be higher than this (uu).
    #[serde(default = "car_height_200")]
    pub player_height_min: f32,
    /// true = half the cosine of the angle between the car's flight and the ball, never below 0; false =
    /// the cosine times the car's speed over 2300, which can be negative and take from the facing part.
    #[serde(default = "yes")]
    pub beginner: bool,
}

fn ball_height_400() -> f32 {
    400.0
}

fn car_height_200() -> f32 {
    200.0
}

fn yes() -> bool {
    true
}

impl Reward for AerialNavigation {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let (me, ball) = (&cur.cars[car], cur.ball.pos);
        let pos = me.body.pos;
        let rising = !me.on_ground
            && ball.z > self.ball_height_min
            && pos.z > self.player_height_min
            && me.body.vel.z > 0.0;
        if !rising || (pos.x - ball.x).hypot(pos.y - ball.y) >= ball.z * 3.0 {
            return 0.0;
        }
        let mut reward = flight_part(self.beginner, me, ball);
        let face = facing(me, ball);
        if face > 0.0 {
            reward += face * 0.25;
            if events.detail(car).controls.boost && me.boost > 0.0 {
                reward += face * 0.1;
            }
        }
        reward.max(0.0)
    }
}

/// Every step the car flies upwards with over 20% boost, below a ball higher than 600 uu (car above
/// 450 uu): up to 0.5 for flying straight at the ball and up to 0.15 for facing it, both times how near the
/// ball is (1 at it, 0 from `max_distance` away), plus 0.85 for touching it then. 0 to 1.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialNavigationStrict {
    /// The ball must be higher than this (uu).
    #[serde(default = "ball_height_600")]
    pub ball_height_min: f32,
    /// The car must be higher than this (uu).
    #[serde(default = "car_height_450")]
    pub player_height_min: f32,
    /// true = half the cosine of the angle between the car's flight and the ball, never below 0; false =
    /// the cosine times the car's speed over 2300, which can be negative.
    #[serde(default = "yes")]
    pub beginner: bool,
    /// From this far from the ball (uu, centre to centre) flying and facing pay nothing; nearer, they pay
    /// more in a straight line up to full at the ball. The default is the field's height (the ceiling).
    #[serde(default = "field_height")]
    pub max_distance: f32,
}

fn ball_height_600() -> f32 {
    600.0
}

fn car_height_450() -> f32 {
    450.0
}

fn field_height() -> f32 {
    2044.0
}

/// 1 at the ball, falling in a straight line to 0 at `max_distance` (0 for a `max_distance` of 0 or less).
fn nearness(distance: f32, max_distance: f32) -> f32 {
    if max_distance > 0.0 {
        (1.0 - distance / max_distance).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl Reward for AerialNavigationStrict {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let (me, ball) = (&cur.cars[car], cur.ball.pos);
        let pos = me.body.pos;
        let in_position = !me.on_ground
            && ball.z > self.ball_height_min
            && pos.z < ball.z
            && pos.z > self.player_height_min
            && me.body.vel.z > 0.0
            && me.boost / 100.0 > 0.2;
        if !in_position {
            return 0.0;
        }
        let mut reward = flight_part(self.beginner, me, ball);
        let face = facing(me, ball);
        if face > 0.0 {
            reward += face * 0.15;
        }
        reward *= nearness(pos.distance(ball), self.max_distance);
        if events.cars[car].touched_ball {
            reward += 0.85;
        }
        reward.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// A car in the air at (0, 0, 500) flying at `vel`, nose along its flight, boosting; the ball at `ball`.
    fn flying(ball: Vec3A, vel: Vec3A) -> (GameSnapshot, StepEvents) {
        let mut cur = game(ball, &[(Team::Blue, Vec3A::new(0.0, 0.0, 500.0))]);
        let me = &mut cur.cars[0];
        me.on_ground = false;
        me.boost = 50.0;
        me.body.vel = vel;
        let forward = vel.normalize();
        let right = forward.cross(Vec3A::Z).normalize();
        me.body.rot = Mat3A::from_cols(forward, right, right.cross(forward));
        let mut events = quiet(1);
        events.details[0].controls.boost = true;
        (cur, events)
    }

    fn defaults<T: for<'de> Deserialize<'de>>() -> T {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    #[test]
    fn navigation_pays_flight_facing_and_boost_when_straight_at_the_ball() {
        let reward: AerialNavigation = defaults();
        assert_eq!(
            (
                reward.ball_height_min,
                reward.player_height_min,
                reward.beginner
            ),
            (400.0, 200.0, true)
        );
        let ball = Vec3A::new(0.0, 600.0, 1300.0);
        let (cur, events) = flying(
            ball,
            (ball - Vec3A::new(0.0, 0.0, 500.0)).normalize() * 1500.0,
        );
        assert!((reward.value(&cur, &cur, &events, 0) - 0.85).abs() < 1e-5);
        let (cur, mut coasting) = flying(ball, (ball - Vec3A::new(0.0, 0.0, 500.0)) * 2.0);
        coasting.details[0].controls.boost = false;
        assert!((reward.value(&cur, &cur, &coasting, 0) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn navigation_pays_by_speed_when_not_a_beginner() {
        let reward = AerialNavigation {
            beginner: false,
            ..defaults()
        };
        let ball = Vec3A::new(0.0, 600.0, 1300.0);
        let (cur, events) = flying(
            ball,
            (ball - Vec3A::new(0.0, 0.0, 500.0)).normalize() * 1150.0,
        );
        assert!((reward.value(&cur, &cur, &events, 0) - 0.85).abs() < 1e-5);
    }

    #[test]
    fn navigation_pays_nothing_out_of_position() {
        let reward: AerialNavigation = defaults();
        let towards = |ball: Vec3A| (ball - Vec3A::new(0.0, 0.0, 500.0)).normalize() * 1000.0;
        let low_ball = Vec3A::new(0.0, 300.0, 390.0);
        let (cur, events) = flying(low_ball, Vec3A::new(0.0, 1000.0, 10.0));
        assert_eq!(reward.value(&cur, &cur, &events, 0), 0.0);
        let far = Vec3A::new(0.0, 4000.0, 1000.0);
        let (cur, events) = flying(far, towards(far));
        assert_eq!(reward.value(&cur, &cur, &events, 0), 0.0);
        let ball = Vec3A::new(0.0, 600.0, 1300.0);
        let (mut cur, events) = flying(ball, towards(ball));
        cur.cars[0].on_ground = true;
        assert_eq!(reward.value(&cur, &cur, &events, 0), 0.0);
        let (cur, events) = flying(ball, Vec3A::new(0.0, 1000.0, -100.0));
        assert_eq!(reward.value(&cur, &cur, &events, 0), 0.0);
    }

    #[test]
    fn navigation_2_pays_a_touch_in_position_capped_at_one() {
        let reward: AerialNavigationStrict = defaults();
        assert_eq!(
            (
                reward.ball_height_min,
                reward.player_height_min,
                reward.beginner,
                reward.max_distance
            ),
            (600.0, 450.0, true, 2044.0)
        );
        // 1000 uu from the ball: flying (0.5) and facing (0.15) straight at it, times 1 - 1000 / 2044.
        let ball = Vec3A::new(0.0, 600.0, 1300.0);
        let to_ball = (ball - Vec3A::new(0.0, 0.0, 500.0)).normalize();
        let (cur, events) = flying(ball, to_ball * 1500.0);
        let near = 1.0 - 1000.0 / 2044.0;
        assert!((reward.value(&cur, &cur, &events, 0) - 0.65 * near).abs() < 1e-5);
        assert_eq!(reward.value(&cur, &cur, &touched(events, 0), 0), 1.0);
    }

    #[test]
    fn navigation_2_pays_more_nearer_and_nothing_from_max_distance() {
        let reward: AerialNavigationStrict = defaults();
        let car = Vec3A::new(0.0, 0.0, 500.0);
        let pay = |reward: &AerialNavigationStrict, ball: Vec3A| {
            let (cur, events) = flying(ball, (ball - car).normalize() * 1500.0);
            reward.value(&cur, &cur, &events, 0)
        };
        let near = pay(&reward, Vec3A::new(0.0, 300.0, 900.0));
        let farther = pay(&reward, Vec3A::new(0.0, 900.0, 1700.0));
        assert!(near > farther && farther > 0.0);
        let far = Vec3A::new(0.0, 3000.0, 1300.0); // about 3104 uu away
        assert_eq!(pay(&reward, far), 0.0);
        let wider = AerialNavigationStrict {
            max_distance: 4000.0,
            ..defaults()
        };
        assert!((pay(&wider, far) - 0.65 * (1.0 - far.distance(car) / 4000.0)).abs() < 1e-5);
    }

    #[test]
    fn navigation_2_pays_nothing_without_boost_or_above_the_ball() {
        let reward: AerialNavigationStrict = defaults();
        let ball = Vec3A::new(0.0, 600.0, 1300.0);
        let to_ball = (ball - Vec3A::new(0.0, 0.0, 500.0)).normalize();
        let (mut cur, events) = flying(ball, to_ball * 1500.0);
        cur.cars[0].boost = 20.0;
        assert_eq!(reward.value(&cur, &cur, &events, 0), 0.0);
        let below = Vec3A::new(0.0, 100.0, 480.0 + 300.0);
        let (mut cur, events) = flying(below, Vec3A::new(0.0, 300.0, 1000.0));
        cur.cars[0].body.pos.z = 900.0;
        assert_eq!(reward.value(&cur, &cur, &touched(events, 0), 0), 0.0);
    }
}
