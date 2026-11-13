//! High touch: a touch high up after a long time in the air.
//! Based on the air-touch reward of the RLGym-PPO guide (github.com/ZealanL/RLGym-PPO-Guide).

use crate::custom::prelude::*;

/// The time in the air that counts in full (s; the original's "rough estimate of the maximum reasonable
/// aerial time").
const MAX_TIME_IN_AIR: f32 = 1.75;

/// When the car touches the ball: the smaller of its time in the air (full at 1.75 s) and the ball's height
/// over the ceiling's (2044 uu). 0 to 1; 0 for a touch from the ground or without a touch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HighTouch {
    // No settings.
}

impl Reward for HighTouch {
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
        let height = cur.ball.pos.z / CEILING_Z;
        air_time.min(height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn paid(air_time: f32, ball_z: f32, touch: bool) -> f32 {
        let cur = game(Vec3A::new(0.0, 0.0, ball_z), &[(Team::Blue, Vec3A::ZERO)]);
        let mut events = if touch {
            touched(quiet(1), 0)
        } else {
            quiet(1)
        };
        events.details[0].air_time = air_time;
        HighTouch {}.value(&cur, &cur, &events, 0)
    }

    #[test]
    fn pays_the_smaller_of_air_time_and_height() {
        assert!((paid(0.875, 1533.0, true) - 0.5).abs() < 1e-6);
        assert!((paid(3.0, 1022.0, true) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn nothing_from_the_ground_or_without_a_touch() {
        assert_eq!(paid(0.0, 1533.0, true), 0.0);
        assert_eq!(paid(1.75, 1533.0, false), 0.0);
    }
}
