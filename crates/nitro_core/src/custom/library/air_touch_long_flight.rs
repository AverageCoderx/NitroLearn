//! Touches after a long time in the air with the ball high, paid the smaller of the two fractions.
//! Based on the air-touch reward of the RLGym-PPO guide (github.com/ZealanL/RLGym-PPO-Guide).

use crate::custom::prelude::*;

/// The longest time in the air that counts (s): "a rough estimate of the maximum reasonable aerial time".
const MAX_TIME_IN_AIR: f32 = 1.75;

/// On a touch: the smaller of the car's time in the air over 1.75 s (at most 1) and the ball's height over
/// the ceiling's, 0 to 1. 0 without a touch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirTouchLongFlight {}

impl Reward for AirTouchLongFlight {
    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        let air_time = events.detail(car).air_time.min(MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR;
        air_time.min(cur.ball.pos.z / CEILING_Z)
    }
}

/// On a touch: the smaller of the car's time in the air before this step over 1.75 s (at most 1) and the
/// ball's height over 2044 uu, 0 to 1. The time counts the steps that ended with the car in the air and
/// starts again on the ground. 0 without a touch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialTouchAirtime {}

impl Reward for AerialTouchAirtime {
    fn memory(&self) -> usize {
        1
    }

    fn value_remembering(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        // memory[0]: this car's time in the air (s), as the source's AirTime.
        let air_time = memory[0].min(MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR;
        let height = cur.ball.pos.z / 2044.0;
        let reward = if events.cars[car].touched_ball {
            air_time.min(height)
        } else {
            0.0
        };
        if cur.cars[car].on_ground {
            memory[0] = 0.0;
        } else {
            memory[0] += events.step_seconds;
        }
        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn high_ball() -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 1022.0),
            &[
                (Team::Blue, Vec3A::new(0.0, -100.0, 900.0)),
                (Team::Orange, Vec3A::new(0.0, 100.0, 900.0)),
            ],
        );
        for car in &mut g.cars {
            car.on_ground = false;
        }
        g
    }

    #[test]
    fn air_touch_pays_the_smaller_fraction_on_a_touch() {
        let reward: AirTouchLongFlight = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = high_ball();
        let mut events = quiet(2);
        events.details[0].air_time = 0.875;
        events.details[1].air_time = 3.0;
        assert_eq!(reward.value(&g, &g, &events, 0), 0.0, "no touch");
        let events = touched(touched(events, 0), 1);
        assert!((reward.value(&g, &g, &events, 0) - 0.5).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &events, 1) - 0.5).abs() < 1e-5,
            "the height's half is the smaller for Orange"
        );
    }

    #[test]
    fn aerial_counts_each_cars_time_in_the_air() {
        let reward: AerialTouchAirtime = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = high_ball();
        let mut grounded = g.clone();
        grounded.cars[1].on_ground = true;
        let mut memory = [[0.0f32; 1]; 2];
        let mut step = |cur: &GameSnapshot, events: &StepEvents, car: usize| {
            reward.value_remembering(&g, cur, events, car, &mut memory[car])
        };
        // Car 0 is in the air for 7 steps, car 1 on the ground; then both touch.
        for _ in 0..7 {
            assert_eq!(step(&grounded, &quiet(2), 0), 0.0);
            assert_eq!(step(&grounded, &quiet(2), 1), 0.0);
        }
        let both = touched(touched(quiet(2), 0), 1);
        let air = 7.0 * (8.0 / 120.0) / 1.75;
        assert!((step(&grounded, &both, 0) - air).abs() < 1e-5);
        assert_eq!(step(&grounded, &both, 1), 0.0, "on the ground all along");
        let mut landed = g.clone();
        landed.cars[0].on_ground = true;
        assert_eq!(step(&landed, &quiet(2), 0), 0.0);
        assert_eq!(step(&g, &both, 0), 0.0, "the count starts again on landing");
        let mut fresh = [0.0f32; 1];
        assert_eq!(
            reward.value_remembering(&g, &g, &both, 0, &mut fresh),
            0.0,
            "fresh memory: no time in the air yet"
        );
    }
}
