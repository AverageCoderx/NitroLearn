//! Flip reset: the car regains its flip while touching the ball in the air, more the higher it is.

use crate::custom::prelude::*;

/// The car's height that earns the full reward (uu).
const FULL_HEIGHT: f32 = 1500.0;

/// When the car, in the air, touches the ball and has a flip it did not have the step before (a flip
/// reset): its height over 1500 uu, at most 1. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlipResetHeight {
    // No settings.
}

impl Reward for FlipResetHeight {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let (before, now) = (&prev.cars[car], &cur.cars[car]);
        if !now.on_ground && events.cars[car].touched_ball && !before.can_flip && now.can_flip {
            (now.body.pos.z / FULL_HEIGHT).min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// Two cars in the air at `height`, neither with a flip; then car 0 has one again (`regained`).
    fn steps(height: f32, regained: bool) -> (GameSnapshot, GameSnapshot) {
        let ball = Vec3A::new(0.0, 0.0, height + 100.0);
        let mut prev = game(
            ball,
            &[
                (Team::Blue, Vec3A::new(0.0, 0.0, height)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, height)),
            ],
        );
        for car in &mut prev.cars {
            car.on_ground = false;
            car.can_flip = false;
        }
        let mut cur = prev.clone();
        cur.cars[0].can_flip = regained;
        (prev, cur)
    }

    #[test]
    fn pays_by_height_on_a_reset_at_most_one() {
        let (prev, cur) = steps(750.0, true);
        let events = touched(quiet(2), 0);
        assert!((FlipResetHeight {}.value(&prev, &cur, &events, 0) - 0.5).abs() < 1e-6);
        let (prev, cur) = steps(1900.0, true);
        assert_eq!(FlipResetHeight {}.value(&prev, &cur, &events, 0), 1.0);
    }

    #[test]
    fn nothing_without_a_touch_a_new_flip_or_in_the_air() {
        let (prev, cur) = steps(750.0, true);
        assert_eq!(FlipResetHeight {}.value(&prev, &cur, &quiet(2), 0), 0.0);
        let events = touched(quiet(2), 0);
        let (prev, still) = steps(750.0, false);
        assert_eq!(FlipResetHeight {}.value(&prev, &still, &events, 0), 0.0);
        let (mut had, cur) = steps(750.0, true);
        had.cars[0].can_flip = true;
        assert_eq!(FlipResetHeight {}.value(&had, &cur, &events, 0), 0.0);
        let (prev, mut landed) = steps(750.0, true);
        landed.cars[0].on_ground = true;
        assert_eq!(FlipResetHeight {}.value(&prev, &landed, &events, 0), 0.0);
    }

    #[test]
    fn each_car_is_judged_by_its_own_flip() {
        let (mut prev, mut cur) = steps(750.0, true);
        prev.cars[1].can_flip = true;
        cur.cars[1].can_flip = true;
        let events = touched(quiet(2), 0);
        assert!((FlipResetHeight {}.value(&prev, &cur, &events, 0) - 0.5).abs() < 1e-6);
        let events = touched(quiet(2), 1);
        assert_eq!(FlipResetHeight {}.value(&prev, &cur, &events, 1), 0.0);
    }
}
