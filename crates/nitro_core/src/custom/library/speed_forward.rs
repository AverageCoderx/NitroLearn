//! Driving fast nose first: the car's speed, negative in reverse.
//! Based on a reward in rishieissocool's reward collection.

use crate::custom::prelude::*;

/// Every step: the car's speed over 2300 uu/s, positive moving nose first and negative in reverse, −1 to 1.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedForward {}

impl Reward for SpeedForward {
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
        let body = &cur.cars[car].body;
        let speed = body.vel.length() / 2300.0;
        // As the source: np.sign(nose · velocity) < 0 is reversing; sideways (0) counts as forward.
        let signed = if body.forward().dot(body.vel) < 0.0 {
            -speed
        } else {
            speed
        };
        signed.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_speed_forward_and_charges_it_in_reverse() {
        let reward: SpeedForward = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        );
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 0.0, "standing still");
        g.cars[0].body.vel = Vec3A::new(1150.0, 0.0, 0.0);
        g.cars[1].body.rot = Mat3A::from_rotation_z(std::f32::consts::PI);
        g.cars[1].body.vel = Vec3A::new(-1150.0, 0.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-6);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 0.5).abs() < 1e-6,
            "Orange alike"
        );
        g.cars[0].body.vel = Vec3A::new(-1150.0, 0.0, 0.0);
        assert!(
            (reward.value(&g, &g, &quiet(2), 0) + 0.5).abs() < 1e-6,
            "reverse"
        );
        g.cars[0].body.vel = Vec3A::new(0.0, 1150.0, 0.0);
        assert!(
            (reward.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-6,
            "sideways counts as forward"
        );
        g.cars[0].body.vel = Vec3A::new(2400.0, 0.0, 0.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0, "at most 1");
    }
}
