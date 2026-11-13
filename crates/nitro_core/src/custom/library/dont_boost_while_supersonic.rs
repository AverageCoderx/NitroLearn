//! A penalty for spending boost while already supersonic.

use crate::custom::prelude::*;

/// −1 on each step the car's boost went down while it moves faster than 2200 uu/s (supersonic), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DontBoostWhileSupersonic {}

impl Reward for DontBoostWhileSupersonic {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let boosted = me.boost < prev.cars[car].boost;
        if boosted && me.body.vel.length() > SUPERSONIC_SPEED {
            -1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn minus_one_for_boost_spent_while_supersonic() {
        let reward: DontBoostWhileSupersonic =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -3000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        );
        let mut cur = prev.clone();
        cur.cars[0].body.vel = Vec3A::new(0.0, 2250.0, 0.0);
        cur.cars[1].body.vel = Vec3A::new(0.0, -2250.0, 0.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 0),
            0.0,
            "no boost spent"
        );
        cur.cars[0].boost = 30.0;
        cur.cars[1].boost = 30.0;
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), -1.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 1),
            -1.0,
            "Orange alike"
        );
        cur.cars[0].body.vel = Vec3A::new(0.0, 2000.0, 0.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 0),
            0.0,
            "not supersonic"
        );
        let mut other = prev.clone();
        other.cars[1].boost = 80.0;
        let mut after = other.clone();
        after.cars[0].body.vel = Vec3A::new(0.0, 2250.0, 0.0);
        after.cars[1].boost = 50.0;
        assert_eq!(
            reward.value(&other, &after, &quiet(2), 0),
            0.0,
            "another car's boost does not count"
        );
    }
}
