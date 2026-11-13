//! Going fast: the car's speed on a square-root curve, full from 1800 uu/s.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

/// The speed (uu/s) that earns the full reward.
const FULL_SPEED: f32 = 1800.0;

/// Every step: the square root of the car's speed over 1800 uu/s, 0 to 1 (1 from 1800 uu/s up).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedSqrt {}

impl Reward for SpeedSqrt {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        (cur.cars[car].body.vel.length() / FULL_SPEED)
            .clamp(0.0, 1.0)
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_speed_on_a_square_root_curve() {
        let reward: SpeedSqrt = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        );
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 0.0, "standing still");
        g.cars[0].body.vel = Vec3A::new(0.0, 450.0, 0.0);
        g.cars[1].body.vel = Vec3A::new(0.0, -450.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-6);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 0.5).abs() < 1e-6,
            "Orange alike"
        );
        g.cars[0].body.vel = Vec3A::new(2300.0, 0.0, 0.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0, "capped");
    }
}
