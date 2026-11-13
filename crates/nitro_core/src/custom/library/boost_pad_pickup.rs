//! Boost pickups, a big pad worth more than a small one.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

/// When the car's boost goes up: the gain (as a share of a full tank) × `big`, plus the gain × `small` when the
/// boost ends below 98 having started below 88 (a small pad): 1.56 for a small pad, 10 for a big pad from
/// empty with the defaults. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostPadPickup {
    /// The extra pay per full tank for a gain that leaves the boost below 98 from below 88.
    #[serde(default = "three")]
    pub small: f32,
    /// The pay per full tank of any gain.
    #[serde(default = "ten")]
    pub big: f32,
}

fn three() -> f32 {
    3.0
}

fn ten() -> f32 {
    10.0
}

impl Reward for BoostPadPickup {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let before = prev.cars[car].boost / 100.0;
        let now = cur.cars[car].boost / 100.0;
        let gain = now - before;
        if gain <= 0.0 {
            return 0.0;
        }
        let mut reward = gain * self.big;
        if now < 0.98 && before < 0.88 {
            reward += gain * self.small;
        }
        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn boosts(before: f32, after: f32) -> (GameSnapshot, GameSnapshot) {
        let mut prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        );
        for car in &mut prev.cars {
            car.boost = before;
        }
        let mut cur = prev.clone();
        for car in &mut cur.cars {
            car.boost = after;
        }
        (prev, cur)
    }

    #[test]
    fn pays_the_gain_more_for_a_small_pad_per_unit() {
        let reward: BoostPadPickup = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, cur) = boosts(50.0, 62.0);
        assert!((reward.value(&prev, &cur, &quiet(2), 0) - 1.56).abs() < 1e-5);
        assert!(
            (reward.value(&prev, &cur, &quiet(2), 1) - 1.56).abs() < 1e-5,
            "Orange alike"
        );
        let (prev, cur) = boosts(0.0, 100.0);
        assert!(
            (reward.value(&prev, &cur, &quiet(2), 0) - 10.0).abs() < 1e-5,
            "big pad"
        );
        let (prev, cur) = boosts(90.0, 100.0);
        assert!(
            (reward.value(&prev, &cur, &quiet(2), 0) - 1.0).abs() < 1e-5,
            "topped up"
        );
        let (prev, cur) = boosts(50.0, 40.0);
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0, "spent");
        let (prev, cur) = boosts(50.0, 50.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 0),
            0.0,
            "held, not gained"
        );
    }
}
