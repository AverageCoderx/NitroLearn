//! Going fast: the car's speed on a square-root curve, full at top speed.

use crate::custom::prelude::*;

/// Every step: √(speed / 2300 uu/s), 0 standing still to 1 at top speed; pays most for the first bit of
/// speed (a quarter of top speed already pays 0.5).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedSqrtV2 {}

impl Reward for SpeedSqrtV2 {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        // The same number as sqrt(speed / 2300 * 100) / 10.
        (cur.cars[car].body.vel.length() / 2300.0 * 100.0).sqrt() / 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn square_root_of_the_speed_fraction() {
        let reward: SpeedSqrtV2 = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(Vec3A::new(0.0, 0.0, 93.15), &[(Team::Orange, Vec3A::ZERO)]);
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 0.0, "standing still");
        g.cars[0].body.vel = Vec3A::new(0.0, -575.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(1), 0) - 0.5).abs() < 1e-5);
        g.cars[0].body.vel = Vec3A::new(2300.0, 0.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(1), 0) - 1.0).abs() < 1e-5);
    }
}
