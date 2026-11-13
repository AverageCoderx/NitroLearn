//! Being on the wall: on the car's wheels, high up.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

/// Above this height (uu) a car on its wheels is on a wall (or the ceiling).
const WALL_HEIGHT: f32 = 300.0;

/// Every step: 1 while the car is on its wheels higher than 300 uu (on a wall, or the ceiling), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerOnWall {}

impl Reward for PlayerOnWall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        f32::from(u8::from(me.on_ground && me.body.pos.z > WALL_HEIGHT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn one_on_the_wall_zero_elsewhere() {
        let reward: PlayerOnWall = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(-4080.0, 0.0, 600.0)),
                (Team::Orange, Vec3A::new(4080.0, 0.0, 600.0)),
            ],
        );
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 1.0, "Orange alike");
        g.cars[0].on_ground = false;
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 0.0, "in the air");
        g.cars[1].body.pos = Vec3A::new(0.0, 2000.0, 17.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 0.0, "on the floor");
    }
}
