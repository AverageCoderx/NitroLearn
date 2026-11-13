//! The car's distance to the ball: being far away (distance_to_ball), or close while in the air
//! (aerial_distance_to_ball).

use crate::custom::prelude::*;

/// Every step: the car's distance to the ball's centre over 2000 uu, from about 0.05 touching it to about 6.6
/// across the field. It pays being far away; give it a negative weight to pay being near.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistanceToBall {}

impl Reward for DistanceToBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        cur.ball.pos.distance(cur.cars[car].body.pos) / 2000.0
    }
}

/// Every step in the air: 1 − distance to the ball / 600 uu, 0 from 600 uu away (0 to 1). 0 on the ground.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialDistanceToBall {}

impl Reward for AerialDistanceToBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        if me.on_ground {
            return 0.0;
        }
        (1.0 - cur.ball.pos.distance(me.body.pos) / 600.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn settings<T: for<'de> Deserialize<'de>>() -> T {
        serde_json::from_value(serde_json::json!({})).expect("no settings needed")
    }

    #[test]
    fn distance_grows_with_distance() {
        let reward: DistanceToBall = settings();
        let g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[(Team::Blue, Vec3A::new(0.0, -2000.0, 93.15))],
        );
        assert!((reward.value(&g, &g, &quiet(1), 0) - 1.0).abs() < 1e-5);
        let near = game(Vec3A::new(0.0, 0.0, 93.15), &[(Team::Blue, Vec3A::ZERO)]);
        assert!(reward.value(&near, &near, &quiet(1), 0) < 0.05);
    }

    #[test]
    fn aerial_distance_pays_near_the_ball_in_the_air_only() {
        let reward: AerialDistanceToBall = settings();
        let mut g = game(
            Vec3A::new(0.0, 0.0, 600.0),
            &[(Team::Orange, Vec3A::new(0.0, 0.0, 300.0))],
        );
        assert_eq!(reward.value(&g, &g, &quiet(1), 0), 0.0, "on the ground");
        g.cars[0].on_ground = false;
        assert!((reward.value(&g, &g, &quiet(1), 0) - 0.5).abs() < 1e-5);
        g.cars[0].body.pos.z = -500.0;
        assert_eq!(
            reward.value(&g, &g, &quiet(1), 0),
            0.0,
            "600 uu or more away"
        );
    }
}
