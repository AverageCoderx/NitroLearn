//! Being between the ball and one's own goal (behind the ball, facing the attack).

use crate::custom::prelude::*;

/// Every step: 1 while the car is nearer its own back wall than the ball is (behind the ball), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehindBall {}

impl Reward for BehindBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let team = cur.cars[car].team;
        let behind = as_blue(team, cur.cars[car].body.pos).y < as_blue(team, cur.ball.pos).y;
        f32::from(u8::from(behind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn one_behind_the_ball_for_either_team() {
        let reward: BehindBall = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = game(
            Vec3A::new(300.0, 500.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(-200.0, -1000.0, 17.0)),
                (Team::Orange, Vec3A::new(200.0, 2000.0, 17.0)),
            ],
        );
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 1.0, "Orange alike");
        let ahead = game(
            Vec3A::new(300.0, 500.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(-200.0, 1000.0, 17.0)),
                (Team::Orange, Vec3A::new(200.0, 0.0, 17.0)),
            ],
        );
        assert_eq!(reward.value(&ahead, &ahead, &quiet(2), 0), 0.0);
        assert_eq!(reward.value(&ahead, &ahead, &quiet(2), 1), 0.0);
    }
}
