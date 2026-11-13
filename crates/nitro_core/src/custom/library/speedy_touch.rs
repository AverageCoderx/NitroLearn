//! Speedy touch: a touch that changes the ball's speed a lot, up or down.

use crate::custom::prelude::*;

/// The change of speed that earns the full reward: 100 km/h (uu/s).
const FULL_CHANGE: f32 = 2777.0;

/// When the car touches the ball: how much the ball's speed changed over the step, up or down, over 2777
/// uu/s (100 km/h), at most 1. 0 without a touch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedyTouch {
    // No settings.
}

impl Reward for SpeedyTouch {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        let change = (cur.ball.vel.length() - prev.ball.vel.length()).abs();
        (change / FULL_CHANGE).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn ball_speeds(before: f32, after: f32) -> (GameSnapshot, GameSnapshot) {
        let mut prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -200.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 200.0, 17.0)),
            ],
        );
        let mut cur = prev.clone();
        prev.ball.vel = Vec3A::new(0.0, before, 0.0);
        cur.ball.vel = Vec3A::new(0.0, after, 0.0);
        (prev, cur)
    }

    #[test]
    fn pays_the_change_of_ball_speed_on_a_touch_at_most_one() {
        let (prev, cur) = ball_speeds(0.0, 1388.5);
        assert!((SpeedyTouch {}.value(&prev, &cur, &touched(quiet(2), 0), 0) - 0.5).abs() < 1e-6);
        let (prev, cur) = ball_speeds(2000.0, 500.0);
        let slowed = SpeedyTouch {}.value(&prev, &cur, &touched(quiet(2), 0), 0);
        assert!((slowed - 1500.0 / 2777.0).abs() < 1e-6);
        let (prev, cur) = ball_speeds(0.0, 4000.0);
        assert_eq!(
            SpeedyTouch {}.value(&prev, &cur, &touched(quiet(2), 0), 0),
            1.0
        );
    }

    #[test]
    fn nothing_without_a_touch() {
        let (prev, cur) = ball_speeds(0.0, 1388.5);
        assert_eq!(SpeedyTouch {}.value(&prev, &cur, &quiet(2), 0), 0.0);
    }

    #[test]
    fn every_car_that_touched_sees_the_same_change() {
        let (prev, cur) = ball_speeds(0.0, 1388.5);
        let both = touched(touched(quiet(2), 0), 1);
        let first = SpeedyTouch {}.value(&prev, &cur, &both, 0);
        assert_eq!(SpeedyTouch {}.value(&prev, &cur, &both, 1), first);
    }
}
