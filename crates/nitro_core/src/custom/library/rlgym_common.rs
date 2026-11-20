//! RLGym's common rewards that NitroLearn's own list lacks, so a bot brought over with `nitro import-rlgym`
//! keeps its rewards: `EventReward`'s team goal and concede, touch and boost pickup parts, and the Liu
//! distance rewards. The others it uses (velocity to ball and to goal, face ball, touch ball, save boost,
//! demolitions) are NitroLearn's built-in or `gigalearn_*` rewards already.
//! Adapted from EventReward (misc_rewards.py), LiuDistancePlayerToBallReward (player_ball_rewards.py) and
//! LiuDistanceBallToGoalReward (ball_goal_rewards.py) in rlgym-sim (github.com/AechPro/rocket-league-gym-sim,
//! rlgym_sim/utils/reward_functions/common_rewards/), the same as in RLGym 1.2.2 (Copyright 2021 Lucas
//! Emery).
//! Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; the
//! full text is in LICENSES/Apache-2.0.txt).
//! Modified: written in Rust; EventReward split into one reward per part (each with its own weight).

use crate::custom::prelude::*;

/// `team_goal` on the step the car's team scores, `concede` on the step the other team does (EventReward's
/// team_goal and concede: the change of each team's score).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RlgymGoal {
    #[serde(default)]
    pub team_goal: f32,
    #[serde(default)]
    pub concede: f32,
}

impl Reward for RlgymGoal {
    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        match cur.goal_scored {
            Some(team) if team == cur.cars[car].team => self.team_goal,
            Some(_) => self.concede,
            None => 0.0,
        }
    }
}

/// 1 on a step the car touches the ball after a step it did not (EventReward's touch: the rise of
/// `ball_touched`, so a run of touching steps pays once).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RlgymTouchStart {}

impl Reward for RlgymTouchStart {
    fn memory(&self) -> usize {
        1
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        _cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        f32::from(u8::from(events.cars[car].touched_ball))
    }

    fn value_remembering(
        &self,
        _prev: &GameSnapshot,
        _cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let now = f32::from(u8::from(events.cars[car].touched_ball));
        let paid = (now - memory[0]).max(0.0);
        memory[0] = now;
        paid
    }
}

/// The boost the car gained this step as a share of a full tank, 0 when it did not gain (EventReward's
/// boost_pickup: a big pad up to 1, a small one 0.12).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RlgymBoostGain {}

impl Reward for RlgymBoostGain {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        ((cur.cars[car].boost - prev.cars[car].boost) / 100.0).max(0.0)
    }
}

/// Every step: exp(-0.5 × (distance to the ball less its radius) / 2300), 1 at the ball's surface
/// (LiuDistancePlayerToBallReward).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RlgymLiuDistancePlayerToBall {}

impl Reward for RlgymLiuDistancePlayerToBall {
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
        // Less the ball's radius (RLGym's 92.75), as RLGym: the inside of the ball cannot be reached.
        let distance = (cur.ball.pos - cur.cars[car].body.pos).length() - 92.75;
        (-0.5 * distance / CAR_MAX_SPEED).exp()
    }
}

/// Every step: exp(-0.5 × the ball's distance to the back of the net / 6000), the distance less the net's
/// depth and the ball's radius (as RLGym); the net the car attacks, or with `own_goal` the one it defends
/// (LiuDistanceBallToGoalReward).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RlgymLiuDistanceBallToGoal {
    #[serde(default)]
    pub own_goal: bool,
}

impl Reward for RlgymLiuDistanceBallToGoal {
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
        // RLGym's constants: back of the net 6000, back wall 5120, ball radius 92.75.
        const DEPTH: f32 = 6000.0 - 5120.0 + 92.75;
        let team = cur.cars[car].team;
        let target = if self.own_goal {
            own_net_back(team)
        } else {
            attack_net_back(team)
        };
        let distance = (cur.ball.pos - target).length() - DEPTH;
        (-0.5 * distance / BALL_MAX_SPEED).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn two_cars() -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        )
    }

    #[test]
    fn goal_pays_team_goal_or_concede() {
        let mut g = two_cars();
        let reward = RlgymGoal {
            team_goal: 1.0,
            concede: -0.5,
        };
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 0.0);
        g.goal_scored = Some(Team::Blue);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), -0.5);
    }

    #[test]
    fn touch_start_pays_the_first_of_a_run_of_touching_steps() {
        let g = two_cars();
        let reward = RlgymTouchStart {};
        let mut memory = [0.0];
        let touch = touched(quiet(2), 0);
        let paid: Vec<f32> = [&touch, &touch, &quiet(2), &touch]
            .iter()
            .map(|events| reward.value_remembering(&g, &g, events, 0, &mut memory))
            .collect();
        assert_eq!(paid, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn boost_gain_is_a_share_of_the_tank() {
        let prev = two_cars();
        let mut cur = two_cars();
        cur.cars[0].boost = 45.0; // from 33
        let reward = RlgymBoostGain {};
        assert!((reward.value(&prev, &cur, &quiet(2), 0) - 0.12).abs() < 1e-6);
        assert_eq!(
            reward.value(&cur, &prev, &quiet(2), 0),
            0.0,
            "spending pays nothing"
        );
    }

    #[test]
    fn liu_distances_are_one_at_the_target_and_fall_off() {
        let mut g = two_cars();
        let to_ball = RlgymLiuDistancePlayerToBall {};
        // RLGym takes the ball's radius off the distance: 1 with the car at the ball's surface.
        g.cars[0].body.pos = g.ball.pos - Vec3A::new(0.0, 92.75, 0.0);
        assert!((to_ball.value(&g, &g, &quiet(2), 0) - 1.0).abs() < 1e-6);
        g.cars[0].body.pos.y -= 2300.0;
        assert!((to_ball.value(&g, &g, &quiet(2), 0) - (-0.5f32).exp()).abs() < 1e-6);
        let to_goal = RlgymLiuDistanceBallToGoal { own_goal: false };
        g.ball.pos = attack_net_back(Team::Blue) - Vec3A::new(0.0, 6000.0 - 5120.0 + 92.75, 0.0);
        assert!((to_goal.value(&g, &g, &quiet(2), 0) - 1.0).abs() < 1e-5);
        let own = RlgymLiuDistanceBallToGoal { own_goal: true };
        assert!(own.value(&g, &g, &quiet(2), 0) < to_goal.value(&g, &g, &quiet(2), 0));
    }
}
