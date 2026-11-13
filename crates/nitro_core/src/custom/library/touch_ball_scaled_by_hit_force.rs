//! Hitting the ball hard: a touch paid by how much it changed the ball's velocity.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

/// A change of the ball's velocity (uu/s) worth 1.
const MAX_HIT_SPEED: f32 = 3000.0;
/// A smaller change pays nothing.
const MIN_HIT_SPEED: f32 = 500.0;

/// On a touch that changes the ball's velocity by more than 500 uu/s over the step: that change over
/// 3000 uu/s, from about 0.17 (with no cap: a 6000 uu/s change pays 2). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TouchBallScaledByHitForce {}

impl Reward for TouchBallScaledByHitForce {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        let hit = cur.ball.vel.distance(prev.ball.vel);
        if hit > MIN_HIT_SPEED {
            hit / MAX_HIT_SPEED
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn pays_the_change_of_the_ball_velocity_on_a_touch() {
        let reward: TouchBallScaledByHitForce =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let mut prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -200.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 200.0, 17.0)),
            ],
        );
        prev.ball.vel = Vec3A::new(0.0, -1500.0, 0.0);
        let mut cur = prev.clone();
        cur.ball.vel = Vec3A::new(0.0, 1500.0, 0.0);
        assert!((reward.value(&prev, &cur, &touched(quiet(2), 0), 0) - 1.0).abs() < 1e-6);
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0, "no touch");
        cur.ball.vel = Vec3A::new(0.0, -1500.0, 6000.0);
        assert!(
            (reward.value(&prev, &cur, &touched(quiet(2), 1), 1) - 2.0).abs() < 1e-6,
            "Orange alike, no cap"
        );
        cur.ball.vel = Vec3A::new(0.0, -1000.0, 0.0);
        assert_eq!(
            reward.value(&prev, &cur, &touched(quiet(2), 0), 0),
            0.0,
            "500 uu/s is not more than 500"
        );
    }
}
