//! Being lined up with the ball and the goals: between the ball and one's own net, and with the ball
//! between oneself and the opponent's net.
//! Adapted from AlignBallGoal in rlgym-sim (github.com/AechPro/rocket-league-gym-sim,
//! rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py), the same as in RLGym 1.2.2
//! (Copyright 2021 Lucas Emery).
//! Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; the
//! full text is in LICENSES/Apache-2.0.txt).
//! Modified: written in Rust; `defense` defaults to 0.1 (1 in the original); 0, not NaN, for a car
//! exactly at the ball's centre or at a net's back.

use crate::custom::prelude::*;

/// Every step: `defense` × the cosine between car→ball and own net→car, plus `offense` × the cosine between
/// car→ball and car→opponent's net (each net's back, half the goal's height up). −1.1 to 1.1 by default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlignBallGoal {
    /// Weight of being between the ball and one's own net (rlgym-sim's own default is 1).
    #[serde(default = "tenth")]
    pub defense: f32,
    /// Weight of having the ball between oneself and the opponent's net.
    #[serde(default = "one")]
    pub offense: f32,
}

fn tenth() -> f32 {
    0.1
}

fn one() -> f32 {
    1.0
}

impl Reward for AlignBallGoal {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let (ball, pos) = (cur.ball.pos, me.body.pos);
        let defensive = self.defense * cosine(ball - pos, pos - own_net_back(me.team));
        let offensive = self.offense * cosine(ball - pos, attack_net_back(me.team) - pos);
        defensive + offensive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn lined_up_between_own_net_and_ball_pays_both_parts() {
        let reward: AlignBallGoal = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = game(
            Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, GOAL_HEIGHT / 2.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, GOAL_HEIGHT / 2.0)),
            ],
        );
        assert!((reward.value(&g, &g, &quiet(2), 0) - 1.1).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 1.1).abs() < 1e-5,
            "Orange alike"
        );
        let wrong_side = game(
            Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0),
            &[(Team::Blue, Vec3A::new(0.0, 2000.0, GOAL_HEIGHT / 2.0))],
        );
        assert!((reward.value(&wrong_side, &wrong_side, &quiet(1), 0) + 1.1).abs() < 1e-5);
        let library = AlignBallGoal {
            defense: 1.0,
            offense: 1.0,
        };
        assert!((library.value(&g, &g, &quiet(2), 0) - 2.0).abs() < 1e-5);
        let mut at_ball = g.clone();
        at_ball.cars[0].body.pos = at_ball.ball.pos;
        assert_eq!(
            reward.value(&at_ball, &at_ball, &quiet(2), 0),
            0.0,
            "no NaN"
        );
    }
}
