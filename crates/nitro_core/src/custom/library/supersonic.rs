//! Supersonic: 1 while the car is supersonic.

use crate::custom::prelude::*;

/// Every step: 1 while the car is supersonic (the game's own flag: on from 2200 uu/s, kept a moment while
/// slowing), otherwise 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Supersonic {
    // No settings.
}

impl Reward for Supersonic {
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
        f32::from(u8::from(cur.cars[car].is_supersonic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn one_while_supersonic_else_nothing() {
        let mut cur = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
        assert_eq!(Supersonic {}.value(&cur, &cur, &quiet(1), 0), 0.0);
        cur.cars[0].is_supersonic = true;
        assert_eq!(Supersonic {}.value(&cur, &cur, &quiet(1), 0), 1.0);
    }
}
