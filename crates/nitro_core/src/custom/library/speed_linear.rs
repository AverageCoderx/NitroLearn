//! Speed: how fast the car goes, not capped.

use crate::custom::prelude::*;

/// Every step: the car's speed over 1000 uu/s, so 0 at rest to 2.3 at top speed (not capped at 1).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedLinear {
    // No settings.
}

impl Reward for SpeedLinear {
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
        cur.cars[car].body.vel.length() / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_the_speed_over_a_thousand() {
        let mut cur = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
        assert_eq!(SpeedLinear {}.value(&cur, &cur, &quiet(1), 0), 0.0);
        cur.cars[0].body.vel = Vec3A::new(1380.0, 1840.0, 0.0);
        assert!((SpeedLinear {}.value(&cur, &cur, &quiet(1), 0) - 2.3).abs() < 1e-5);
    }
}
