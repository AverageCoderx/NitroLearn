//! Goals scored fast: the ball's speed just before it went in, paid to the scorers and charged to the
//! conceders, and (goal_speed_top_corner) more in a top corner.
//! Adapted from Necto (github.com/Rolv-Arild/Necto), licensed CC BY-NC-SA 4.0
//! (https://creativecommons.org/licenses/by-nc-sa/4.0/), and changed from the original. This file is shared
//! under the same licence.

use crate::custom::prelude::*;

/// The speed bonus for `car` on a step with a goal: `weight` × the ball's speed on the step before over
/// 6000 uu/s, + for the scoring team and − for the other; 0 without a goal.
fn speed_bonus(prev: &GameSnapshot, cur: &GameSnapshot, car: usize, weight: f32) -> f32 {
    let Some(scorer) = cur.goal_scored else {
        return 0.0;
    };
    let bonus = weight * prev.ball.vel.length() / BALL_MAX_SPEED;
    if scorer == cur.cars[car].team {
        bonus
    } else {
        -bonus
    }
}

/// On a goal: the ball's speed on the step before over 6000 uu/s, times `goal_speed_bonus_w`; paid to the
/// scoring team, charged to the other (−1 to 1 with the default). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalSpeedBonus {
    /// What a goal at the ball's top speed is worth.
    #[serde(default = "one")]
    pub goal_speed_bonus_w: f32,
}

fn one() -> f32 {
    1.0
}

impl Reward for GoalSpeedBonus {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        speed_bonus(prev, cur, car, self.goal_speed_bonus_w)
    }
}

/// GoalSpeedBonus times |ball height / 640| × |ball x / 640| where the ball went in: about 0 for a goal
/// along the floor or through the middle, up to about 1.4 in a top corner.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalSpeedTopCorner {
    /// What a goal at the ball's top speed is worth, before the placement.
    #[serde(default = "one_placed")]
    pub goal_speed_bonus_w: f32,
}

fn one_placed() -> f32 {
    1.0
}

impl Reward for GoalSpeedTopCorner {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        // 640 is about the goal's height; the goal is about twice as wide, so |x| / 640 is about 0 to 1.
        let ball = cur.ball.pos;
        speed_bonus(prev, cur, car, self.goal_speed_bonus_w)
            * (ball.z / 640.0).abs()
            * (ball.x / 640.0).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// Two cars a side; the ball, at 3000 uu/s on the step before, ends the step at `ball` in the net of
    /// the team that did not score.
    fn goal(scorer: Team, ball: Vec3A) -> (GameSnapshot, GameSnapshot) {
        let cars = [
            (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
            (Team::Blue, Vec3A::new(500.0, -2000.0, 17.0)),
            (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            (Team::Orange, Vec3A::new(500.0, 2000.0, 17.0)),
        ];
        let mut prev = game(ball, &cars);
        prev.ball.vel = Vec3A::new(0.0, 3000.0 * attack_sign(scorer), 0.0);
        let mut cur = prev.clone();
        cur.ball.pos = Vec3A::new(ball.x, ball.y * attack_sign(scorer), ball.z);
        cur.goal_scored = Some(scorer);
        (prev, cur)
    }

    #[test]
    fn goal_speed_bonus_pays_every_scorer_and_charges_every_conceder() {
        let reward: GoalSpeedBonus = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, cur) = goal(Team::Blue, Vec3A::new(0.0, 5300.0, 93.15));
        let paid: Vec<f32> = (0..4)
            .map(|car| reward.value(&prev, &cur, &quiet(4), car))
            .collect();
        assert_eq!(paid, [0.5, 0.5, -0.5, -0.5]);
        let (prev, cur) = goal(Team::Orange, Vec3A::new(0.0, 5300.0, 93.15));
        assert_eq!(reward.value(&prev, &cur, &quiet(4), 2), 0.5, "Orange alike");
        assert_eq!(reward.value(&prev, &cur, &quiet(4), 0), -0.5);
        let mut no_goal = cur.clone();
        no_goal.goal_scored = None;
        assert_eq!(reward.value(&prev, &no_goal, &quiet(4), 2), 0.0);
        let double = GoalSpeedBonus {
            goal_speed_bonus_w: 2.0,
        };
        assert_eq!(double.value(&prev, &cur, &quiet(4), 2), 1.0);
    }

    #[test]
    fn goal_speed_and_placement_pays_the_top_corners() {
        let reward: GoalSpeedTopCorner = serde_json::from_value(serde_json::json!({})).unwrap();
        for scorer in [Team::Blue, Team::Orange] {
            let (prev, corner) = goal(scorer, Vec3A::new(-800.0, 5300.0, 560.0));
            let blue = if scorer == Team::Blue { 0 } else { 2 };
            let expected = 0.5 * (560.0 / 640.0) * (800.0 / 640.0);
            let at_corner = reward.value(&prev, &corner, &quiet(4), blue);
            assert!((at_corner - expected).abs() < 1e-5, "{scorer:?}");
            let conceded = reward.value(&prev, &corner, &quiet(4), 2 - blue);
            assert!((conceded + expected).abs() < 1e-5);
            let (prev, middle) = goal(scorer, Vec3A::new(0.0, 5300.0, 300.0));
            assert_eq!(reward.value(&prev, &middle, &quiet(4), blue), 0.0, "x = 0");
        }
    }
}
