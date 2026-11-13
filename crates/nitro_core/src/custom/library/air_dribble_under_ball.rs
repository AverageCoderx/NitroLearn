//! Air dribble shaping: the car in the air close under the ball, nose up, moving with it.

use crate::custom::prelude::*;

/// The ball must be nearer the car than this (uu, centre to centre).
const MAX_BALL_DISTANCE: f32 = 300.0;
/// When car and ball both fall, both must fall slower than this (uu/s).
const FALL_SPEED_LIMIT: f32 = 650.0;

/// Every step in the air: 1 when the ball is within 300 uu of the car, above it, and the car's nose points
/// up (when car and ball both fall, only while both fall slower than 650 uu/s); plus 0.5 whenever car and
/// ball move the same general way, near or not. 0 to 1.5; 0 on the ground.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirDribbleUnderBall {
    // No settings.
}

impl Reward for AirDribbleUnderBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        if me.on_ground {
            return 0.0;
        }
        let (pos, vel, ball) = (me.body.pos, me.body.vel, &cur.ball);
        let mut reward = 0.0;
        if ball.pos.distance(pos) < MAX_BALL_DISTANCE
            && pos.z < ball.pos.z
            && me.body.forward().z > 0.0
        {
            let both_falling = vel.z < 0.0 && ball.vel.z < 0.0;
            if !both_falling
                || (vel.z.abs() < FALL_SPEED_LIMIT && ball.vel.z.abs() < FALL_SPEED_LIMIT)
            {
                reward += 1.0;
            }
        }
        if vel.dot(ball.vel) > 0.0 {
            reward += 0.5;
        }
        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// A car in the air at 500 uu, nose up, under a ball at `ball`; car and ball moving at the speeds given.
    fn paid(ball: Vec3A, car_vel: Vec3A, ball_vel: Vec3A, nose_up: bool) -> f32 {
        let mut cur = game(ball, &[(Team::Blue, Vec3A::new(0.0, 0.0, 500.0))]);
        let me = &mut cur.cars[0];
        me.on_ground = false;
        me.body.vel = car_vel;
        me.body.rot = Mat3A::from_rotation_y(if nose_up { -0.5 } else { 0.5 });
        cur.ball.vel = ball_vel;
        AirDribbleUnderBall {}.value(&cur, &cur, &quiet(1), 0)
    }

    const ABOVE: Vec3A = Vec3A::new(0.0, 0.0, 700.0);

    #[test]
    fn pays_most_close_under_the_ball_nose_up_moving_with_it() {
        let up = Vec3A::new(0.0, 0.0, 300.0);
        assert_eq!(paid(ABOVE, up, up * 0.5, true), 1.5);
        let slow_fall = Vec3A::new(0.0, 0.0, -300.0);
        assert_eq!(paid(ABOVE, slow_fall, slow_fall, true), 1.5);
    }

    #[test]
    fn only_the_half_when_falling_fast_nose_down_or_far() {
        let fast_fall = Vec3A::new(0.0, 0.0, -800.0);
        assert_eq!(paid(ABOVE, fast_fall, fast_fall, true), 0.5);
        let up = Vec3A::new(0.0, 0.0, 300.0);
        assert_eq!(paid(ABOVE, up, up, false), 0.5);
        assert_eq!(paid(Vec3A::new(0.0, 2000.0, 700.0), up, up, true), 0.5);
        assert_eq!(paid(ABOVE, up, -up, true), 1.0);
    }

    #[test]
    fn nothing_on_the_ground() {
        let mut cur = game(ABOVE, &[(Team::Blue, Vec3A::new(0.0, 0.0, 600.0))]);
        cur.cars[0].body.vel = Vec3A::new(0.0, 0.0, 100.0);
        cur.ball.vel = cur.cars[0].body.vel;
        assert_eq!(AirDribbleUnderBall {}.value(&cur, &cur, &quiet(1), 0), 0.0);
    }
}
