//! How far up the field the ball is, from the car's team's view (the ball's y, not its height).
//! Adapted from BallYCoordinateReward in rlgym-sim (github.com/AechPro/rocket-league-gym-sim,
//! rlgym_sim/utils/reward_functions/common_rewards/ball_goal_rewards.py), the same as in RLGym 1.2.2
//! (Copyright 2021 Lucas Emery).
//! Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; the
//! full text is in LICENSES/Apache-2.0.txt).
//! Modified: written in Rust; `exponent` is a whole number (a fraction of a negative y has no
//! answer).

use crate::custom::prelude::*;

/// rlgym-sim's back wall (5120) plus its ball radius (92.75), which the source divides by.
const FIELD_HALF_LENGTH: f32 = 5120.0 + 92.75;

/// Every step: the ball's y over 5212.75 uu as the car's team sees it (attacking +y), to the power
/// `exponent`: about −1 at its own back wall, 0 at midfield, about 1 at the opponent's.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallYCoordinate {
    /// Bends the curve; keep it odd (1, 3, …) so the ball in the car's own half pays negative.
    #[serde(default = "one")]
    pub exponent: u32,
}

fn one() -> u32 {
    1
}

impl Reward for BallYCoordinate {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let y = as_blue(cur.cars[car].team, cur.ball.pos).y;
        let power = i32::try_from(self.exponent).unwrap_or(i32::MAX);
        (y / FIELD_HALF_LENGTH).powi(power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn the_balls_y_from_each_teams_view() {
        let reward: BallYCoordinate = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = game(
            Vec3A::new(0.0, FIELD_HALF_LENGTH / 2.0, 93.15),
            &[(Team::Blue, Vec3A::ZERO), (Team::Orange, Vec3A::ZERO)],
        );
        assert!((reward.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-6);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) + 0.5).abs() < 1e-6,
            "Orange's own half"
        );
        let mut mirrored = g.clone();
        mirrored.ball.pos.y = -FIELD_HALF_LENGTH / 2.0;
        assert!(
            (reward.value(&mirrored, &mirrored, &quiet(2), 1) - 0.5).abs() < 1e-6,
            "Orange alike"
        );
        let cubed = BallYCoordinate { exponent: 3 };
        assert!((cubed.value(&mirrored, &mirrored, &quiet(2), 0) + 0.125).abs() < 1e-6);
        let mut centre = g.clone();
        centre.ball.pos.y = 0.0;
        assert_eq!(reward.value(&centre, &centre, &quiet(2), 0), 0.0);
    }
}
