//! Spending boost: charged near the floor, less higher up, and paid above the goal's height.
//! Adapted from Necto (github.com/Rolv-Arild/Necto), licensed CC BY-NC-SA 4.0
//! (https://creativecommons.org/licenses/by-nc-sa/4.0/), and changed from the original. This file is shared
//! under the same licence.

use crate::custom::prelude::*;

/// When the car's boost goes down: `boost_lose_w` × the fall in √(boost / 100) × (1 − its height / 642.775),
/// so a penalty below the goal's height, 0 at it and a reward above it. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostLose {
    /// How much the fall in √(boost / 100) counts.
    #[serde(default = "one")]
    pub boost_lose_w: f32,
}

fn one() -> f32 {
    1.0
}

impl Reward for BoostLose {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let root = |boost: f32| (boost / 100.0).clamp(0.0, 1.0).sqrt();
        let change = root(cur.cars[car].boost) - root(prev.cars[car].boost);
        if change < 0.0 {
            self.boost_lose_w * change * (1.0 - cur.cars[car].body.pos.z / GOAL_HEIGHT)
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
    fn charges_boost_spent_low_and_pays_it_high() {
        let reward: BoostLose = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 0.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 0.0)),
            ],
        );
        prev.cars[0].boost = 100.0;
        prev.cars[1].boost = 100.0;
        let mut cur = prev.clone();
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 0),
            0.0,
            "nothing spent"
        );
        cur.cars[0].boost = 64.0;
        cur.cars[1].boost = 64.0;
        assert!((reward.value(&prev, &cur, &quiet(2), 0) + 0.2).abs() < 1e-5);
        assert!(
            (reward.value(&prev, &cur, &quiet(2), 1) + 0.2).abs() < 1e-5,
            "Orange alike"
        );
        cur.cars[0].body.pos.z = GOAL_HEIGHT / 2.0;
        assert!((reward.value(&prev, &cur, &quiet(2), 0) + 0.1).abs() < 1e-5);
        cur.cars[0].body.pos.z = GOAL_HEIGHT * 2.0;
        assert!(
            (reward.value(&prev, &cur, &quiet(2), 0) - 0.2).abs() < 1e-5,
            "high up"
        );
        assert_eq!(reward.value(&cur, &prev, &quiet(2), 1), 0.0, "boost gained");
    }
}
