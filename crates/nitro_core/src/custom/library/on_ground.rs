//! Being on the ground (the opposite of the built-in `air`).
//! Based on the InAirReward of the RLGym-PPO guide (github.com/ZealanL/RLGym-PPO-Guide), with its two
//! answers swapped.

use crate::custom::prelude::*;

/// Every step: 1 while the car is on the ground (its wheels on the floor, a wall or the ceiling), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnGround {}

impl Reward for OnGround {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        f32::from(u8::from(cur.cars[car].on_ground))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn one_on_the_ground_zero_in_the_air() {
        let reward: OnGround = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(Vec3A::new(0.0, 0.0, 93.15), &[(Team::Blue, Vec3A::ZERO)]);
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 1.0);
        g.cars[0].on_ground = false;
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 0.0);
    }
}
