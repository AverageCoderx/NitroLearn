//! Flying up at a high ball from below: towards it, fast, and closer each step.

use crate::custom::prelude::*;

/// Every step in the air, rising, below `ball_height_min` with the ball above it and nearer than 3 × the
/// ball's height: the cosine between the car's flight and the ball × speed / 2300 (+ half the cosine, not
/// below 0, with `beginner`), +0.5 when nearer than on its last such step, × 1.5 on a touch: 0 to 3.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialApproach {
    /// The ball must be higher than this and the car lower (uu).
    #[serde(default = "four_hundred")]
    pub ball_height_min: f32,
    /// Also pays half the cosine between the car's flight and the ball (not below 0).
    #[serde(default = "yes")]
    pub beginner: bool,
}

fn four_hundred() -> f32 {
    400.0
}

fn yes() -> bool {
    true
}

impl Reward for AerialApproach {
    fn every_step(&self) -> bool {
        true
    }

    fn memory(&self) -> usize {
        2
    }

    fn value_remembering(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        // memory[0]: 1 once this car has a previous distance; memory[1]: that distance (uu).
        let me = &cur.cars[car];
        let (ball, body) = (cur.ball.pos, &me.body);
        let distance = ball.distance(body.pos);
        let navigating = !me.on_ground
            && ball.z > self.ball_height_min
            && self.ball_height_min > body.pos.z
            && body.vel.z > 0.0
            && distance < ball.z * 3.0;
        if !navigating {
            return 0.0;
        }
        let alignment = (ball - body.pos)
            .normalize_or_zero()
            .dot(body.vel.normalize_or_zero());
        let mut reward = 0.0;
        if self.beginner {
            reward += (alignment * 0.5).max(0.0);
        }
        reward += alignment * (body.vel.length() / 2300.0);
        if memory[0] == 1.0 && memory[1] - distance > 0.0 {
            reward += 0.5;
        }
        memory[0] = 1.0;
        memory[1] = distance;
        if events.cars[car].touched_ball {
            reward *= 1.5;
        }
        reward.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// A Blue and an Orange car at `car_z` on the centre spot (the same spot mirrored), flying straight up
    /// at 1150 uu/s to a ball at 800 uu.
    fn flying(car_z: f32) -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 800.0),
            &[
                (Team::Blue, Vec3A::new(0.0, 0.0, car_z)),
                (Team::Orange, Vec3A::new(0.0, 0.0, car_z)),
            ],
        );
        for car in &mut g.cars {
            car.on_ground = false;
            car.body.vel = Vec3A::new(0.0, 0.0, 1150.0);
        }
        g
    }

    #[test]
    fn pays_flying_at_the_ball_and_getting_closer() {
        let reward: AerialApproach = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut memory = [[0.0f32; 2]; 2];
        let first = flying(200.0);
        let mut step = |g: &GameSnapshot, events: &StepEvents, car: usize| {
            reward.value_remembering(g, g, events, car, &mut memory[car])
        };
        assert!(
            (step(&first, &quiet(2), 0) - 1.0).abs() < 1e-5,
            "0.5 + 0.5, no distance yet"
        );
        assert!(
            (step(&first, &quiet(2), 1) - 1.0).abs() < 1e-5,
            "Orange alike"
        );
        let closer = flying(250.0);
        assert!(
            (step(&closer, &quiet(2), 0) - 1.5).abs() < 1e-5,
            "+0.5 for getting closer"
        );
        let higher = flying(300.0);
        let touch = step(&higher, &touched(quiet(2), 0), 0);
        assert!((touch - 2.25).abs() < 1e-5, "× 1.5 on a touch");
        assert!(
            (step(&first, &quiet(2), 1) - 1.0).abs() < 1e-5,
            "Orange no closer: its own distance"
        );
    }

    #[test]
    fn nothing_outside_the_gate_and_never_negative() {
        let reward = AerialApproach {
            ball_height_min: 400.0,
            beginner: false,
        };
        let mut memory = [0.0f32; 2];
        let mut too_high = flying(450.0);
        assert_eq!(
            reward.value_remembering(&too_high, &too_high, &quiet(2), 0, &mut memory),
            0.0
        );
        too_high.cars[0].body.pos.z = 200.0;
        too_high.cars[0].on_ground = true;
        assert_eq!(
            reward.value_remembering(&too_high, &too_high, &quiet(2), 0, &mut memory),
            0.0
        );
        assert_eq!(memory, [0.0, 0.0], "outside the gate nothing is remembered");
        let mut away = flying(200.0);
        away.cars[0].body.vel = Vec3A::new(0.0, 1000.0, 100.0);
        away.ball.pos.y = -2000.0;
        assert_eq!(
            reward.value_remembering(&away, &away, &quiet(2), 0, &mut memory),
            0.0,
            "flying away pays nothing, not a penalty"
        );
    }
}
