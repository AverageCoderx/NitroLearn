//! Kickoff race: first to the ball on a kickoff.

use crate::custom::prelude::*;

/// Every step while the ball sits on the centre spot (x and y exactly 0, until it is first hit): 1 if this
/// car is nearer the ball than every opponent, otherwise -0.8 (also with no opponents). 0 once the ball has
/// moved.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kickoff {
    // No settings.
}

impl Reward for Kickoff {
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
        let ball = cur.ball.pos;
        if ball.x != 0.0 || ball.y != 0.0 {
            return 0.0;
        }
        let mine = cur.cars[car].body.pos.distance(ball);
        let nearest = opponents(cur, car)
            .map(|other| cur.cars[other].body.pos.distance(ball))
            .reduce(f32::min);
        match nearest {
            Some(theirs) if mine < theirs => 1.0,
            _ => -0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    const SPOT: Vec3A = Vec3A::new(0.0, 0.0, 93.15);

    fn both(cur: &GameSnapshot) -> [f32; 2] {
        let events = quiet(cur.cars.len());
        [0, 1].map(|car| Kickoff {}.value(cur, cur, &events, car))
    }

    #[test]
    fn the_nearer_car_gets_one_the_other_minus_0_8_for_either_team() {
        let blue_nearer = game(
            SPOT,
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        );
        assert_eq!(both(&blue_nearer), [1.0, -0.8]);
        let orange_nearer = game(
            SPOT,
            &[
                (Team::Blue, Vec3A::new(0.0, -3000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        );
        assert_eq!(both(&orange_nearer), [-0.8, 1.0]);
        let tie = game(
            SPOT,
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        );
        assert_eq!(both(&tie), [-0.8, -0.8]);
    }

    #[test]
    fn nothing_once_the_ball_has_moved_and_minus_0_8_alone() {
        let moved = game(
            Vec3A::new(10.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        );
        assert_eq!(both(&moved), [0.0, 0.0]);
        let alone = game(SPOT, &[(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0))]);
        assert_eq!(Kickoff {}.value(&alone, &alone, &quiet(1), 0), -0.8);
    }
}
