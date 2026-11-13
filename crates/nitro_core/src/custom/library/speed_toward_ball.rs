//! How fast the car drives towards the ball, never negative.
//! Based on the SpeedTowardBallReward of the RLGym-PPO guide (github.com/ZealanL/RLGym-PPO-Guide).

use crate::custom::prelude::*;

/// Every step: the car's speed towards the ball over its top speed (2300 uu/s), 0 to 1. 0 when it moves
/// away from the ball (moving away is not punished).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedTowardBall {}

impl Reward for SpeedTowardBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let body = &cur.cars[car].body;
        let towards = (cur.ball.pos - body.pos).normalize_or_zero();
        (body.vel.dot(towards) / CAR_MAX_SPEED).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_speed_towards_the_ball_and_nothing_for_moving_away() {
        let reward: SpeedTowardBall = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(0.0, 1000.0, 93.15),
            &[(Team::Blue, Vec3A::new(0.0, 0.0, 17.0))],
        );
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 0.0, "standing still");
        g.cars[0].body.vel = Vec3A::new(0.0, 1150.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(1), 0) - 0.5).abs() < 0.01);
        g.cars[0].body.vel = Vec3A::new(0.0, -1150.0, 0.0);
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 0.0, "moving away");
        g.cars[0].body.pos = g.ball.pos;
        assert_eq!(
            reward.value(&g, &g, &quiet(1), 0),
            0.0,
            "no NaN at the ball"
        );
    }
}
