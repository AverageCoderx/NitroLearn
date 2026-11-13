//! Boost pickup: the square root of the boost a car gains in a step.

use crate::custom::prelude::*;

/// When the car gains boost in a step: the square root of the gain as a share of a full tank (a big pad
/// from empty pays 1, a small pad about 0.35). 0 when its boost stays the same or goes down.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostPickupSqrt {
    // No settings.
}

impl Reward for BoostPickupSqrt {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        let gained = (cur.cars[car].boost - prev.cars[car].boost) / 100.0;
        if gained > 0.0 { gained.sqrt() } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn with_boost(before: f32, after: f32) -> f32 {
        let mut prev = game(Vec3A::new(0.0, 0.0, 93.0), &[(Team::Blue, Vec3A::ZERO)]);
        let mut cur = prev.clone();
        prev.cars[0].boost = before;
        cur.cars[0].boost = after;
        let reward: BoostPickupSqrt = serde_json::from_value(serde_json::json!({})).unwrap();
        reward.value(&prev, &cur, &quiet(1), 0)
    }

    #[test]
    fn pays_the_square_root_of_the_gain() {
        assert!((with_boost(0.0, 100.0) - 1.0).abs() < 1e-6);
        assert!((with_boost(33.0, 45.0) - 0.12f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn pays_nothing_without_a_gain() {
        assert_eq!(with_boost(50.0, 50.0), 0.0);
        assert_eq!(with_boost(50.0, 20.0), 0.0);
    }
}
