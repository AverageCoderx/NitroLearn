//! Kickoffs: being nearer the ball than every opponent while it waits on the centre spot.

use crate::custom::prelude::*;

/// A kickoff: the ball at x = y = 0 exactly (before the first touch).
fn ball_on_centre_spot(game: &GameSnapshot) -> bool {
    game.ball.pos.x == 0.0 && game.ball.pos.y == 0.0
}

/// This car's distance to the ball, and whether it is nearer than every opponent (false without one).
fn nearest(game: &GameSnapshot, car: usize) -> (f32, bool) {
    let distance = |other: usize| game.cars[other].body.pos.distance(game.ball.pos);
    let mine = distance(car);
    let mut opponents = opponents(game, car).map(distance).peekable();
    let has_one = opponents.peek().is_some();
    (mine, has_one && opponents.all(|theirs| mine < theirs))
}

/// Every step of a kickoff: 1 while the car is nearer the ball than every opponent, else −0.8 (also with no
/// opponent). 0 once the ball has moved. Teammates do not count.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffNearest {}

impl Reward for KickoffNearest {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        if !ball_on_centre_spot(cur) {
            return 0.0;
        }
        if nearest(cur, car).1 { 1.0 } else { -0.8 }
    }
}

/// Every step of a kickoff while the car is more than `min_distance` (1200 uu) from the ball: 1 when nearer
/// it than every opponent, else −1. 0 within `min_distance` and once the ball has moved.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffNearestV2 {
    /// Nearer the ball than this (uu) the reward pays nothing.
    #[serde(default = "twelve_hundred")]
    pub min_distance: f32,
}

fn twelve_hundred() -> f32 {
    1200.0
}

impl Reward for KickoffNearestV2 {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        if !ball_on_centre_spot(cur) {
            return 0.0;
        }
        match nearest(cur, car) {
            (distance, _) if distance <= self.min_distance => 0.0,
            (_, true) => 1.0,
            (_, false) => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn kickoff(blue_y: f32, orange_y: f32) -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, blue_y, 17.0)),
                (Team::Orange, Vec3A::new(0.0, orange_y, 17.0)),
            ],
        )
    }

    #[test]
    fn proximity_pays_the_nearer_car() {
        let reward: KickoffNearest = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = kickoff(-2000.0, 3000.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), -0.8);
        let mirrored = kickoff(-3000.0, 2000.0);
        assert_eq!(
            reward.value(&mirrored, &mirrored, &quiet(2), 1),
            1.0,
            "Orange alike"
        );
        let alone = game(Vec3A::new(0.0, 0.0, 93.15), &[(Team::Blue, Vec3A::ZERO)]);
        assert_eq!(
            reward.value(&alone, &alone, &quiet(1), 0),
            -0.8,
            "no opponent"
        );
        let mut played = g.clone();
        played.ball.pos.y = 5.0;
        assert_eq!(reward.value(&played, &played, &quiet(2), 0), 0.0);
    }

    #[test]
    fn modified_proximity_only_far_from_the_ball() {
        let reward: KickoffNearestV2 = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = kickoff(-2000.0, 3000.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), -1.0);
        let mirrored = kickoff(-3000.0, 2000.0);
        assert_eq!(
            reward.value(&mirrored, &mirrored, &quiet(2), 1),
            1.0,
            "Orange alike"
        );
        let close = kickoff(-1000.0, 3000.0);
        assert_eq!(
            reward.value(&close, &close, &quiet(2), 0),
            0.0,
            "within 1200 uu"
        );
        let mut played = g.clone();
        played.ball.pos.x = 5.0;
        assert_eq!(reward.value(&played, &played, &quiet(2), 0), 0.0);
        let nearer = KickoffNearestV2 {
            min_distance: 500.0,
        };
        assert_eq!(reward.min_distance, 1200.0);
        assert_eq!(
            nearer.value(&close, &close, &quiet(2), 0),
            1.0,
            "pays from 500 uu"
        );
    }
}
