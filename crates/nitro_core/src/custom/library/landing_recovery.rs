//! Coming down from high up ready to land: nose along the way the car travels, wheels down.
//! Based on a reward in rishieissocool's reward collection.

use crate::custom::prelude::*;

/// Every step the car falls in the air higher than 250 uu: half of (the cosine between its nose and its
/// flight, both seen from above, plus how far its roof points up), −1 to 1. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LandingRecovery {}

impl Reward for LandingRecovery {
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
        let me = &cur.cars[car];
        let body = &me.body;
        if me.on_ground || body.vel.z >= 0.0 || body.pos.z <= 250.0 {
            return 0.0;
        }
        // As the source: a flat vector of length 0 stays 0, so standing still sideways adds nothing.
        let flat = |v: Vec3A| Vec3A::new(v.x, v.y, 0.0).normalize_or_zero();
        (flat(body.vel).dot(flat(body.forward())) + body.up().z) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// A Blue car and an Orange car 600 uu up, falling and flying forward (each along its own nose: the
    /// Orange one turned half a turn).
    fn falling() -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 600.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 600.0)),
            ],
        );
        g.cars[0].body.vel = Vec3A::new(800.0, 0.0, -300.0);
        g.cars[1].body.rot = Mat3A::from_rotation_z(std::f32::consts::PI);
        g.cars[1].body.vel = Vec3A::new(-800.0, 0.0, -300.0);
        for c in &mut g.cars {
            c.on_ground = false;
        }
        g
    }

    #[test]
    fn pays_nose_along_the_flight_and_wheels_down() {
        let reward: LandingRecovery = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = falling();
        assert!((reward.value(&g, &g, &quiet(2), 0) - 1.0).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 1.0).abs() < 1e-5,
            "Orange alike"
        );
        let mut upside_down = g.clone();
        upside_down.cars[0].body.rot = Mat3A::from_rotation_x(std::f32::consts::PI);
        assert!(
            reward.value(&upside_down, &upside_down, &quiet(2), 0).abs() < 1e-5,
            "nose along the flight (+1), roof down (−1)"
        );
        let mut backwards = g.clone();
        backwards.cars[0].body.vel.x = -800.0;
        assert!((reward.value(&backwards, &backwards, &quiet(2), 0)).abs() < 1e-5);
        let mut straight_down = g.clone();
        straight_down.cars[0].body.vel = Vec3A::new(0.0, 0.0, -300.0);
        assert!(
            (reward.value(&straight_down, &straight_down, &quiet(2), 0) - 0.5).abs() < 1e-5,
            "no flight seen from above: only the roof counts"
        );
    }

    #[test]
    fn nothing_rising_low_or_on_the_ground() {
        let reward = LandingRecovery {};
        let mut rising = falling();
        rising.cars[0].body.vel.z = 300.0;
        assert_eq!(reward.value(&rising, &rising, &quiet(2), 0), 0.0);
        let mut low = falling();
        low.cars[0].body.pos.z = 240.0;
        assert_eq!(reward.value(&low, &low, &quiet(2), 0), 0.0);
        let mut grounded = falling();
        grounded.cars[0].on_ground = true;
        assert_eq!(reward.value(&grounded, &grounded, &quiet(2), 0), 0.0);
    }
}
