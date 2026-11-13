//! Goals paid by how fast the ball went in, and (goal_speed_and_placement) more for one off the floor.
//! Based on rewards from apollo-bot by redd-rl; goal_speed also builds on one by fredrik.

use crate::custom::prelude::*;

/// The source's ball radius (RLGymSim's 92.75, not the game's 91.25).
const RADIUS: f32 = 92.75;
/// GoalSpeedAndPlacement: a ball higher than this before going in was off the floor.
const MIN_HEIGHT: f32 = RADIUS + 10.0;
/// GoalSpeedAndPlacement: the bonus factor for such a goal.
const HEIGHT_REWARD: f32 = 1.75;

/// On the step the car's team scores: the ball's speed then (uu/s) × `reward_factor` × 2, so 600 for a
/// 3000 uu/s goal with the default: give it a low weight. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalSpeed {
    /// The pay per uu/s of the ball's speed, before it is doubled.
    #[serde(default = "tenth")]
    pub reward_factor: f32,
}

fn tenth() -> f32 {
    0.1
}

impl Reward for GoalSpeed {
    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        // The source's "the team's score is above 0": its episodes ended at the goal.
        if cur.goal_scored == Some(cur.cars[car].team) {
            cur.ball.vel.length() * self.reward_factor * 2.0
        } else {
            0.0
        }
    }
}

/// On the step the car's team scores: the ball's speed on the step before over 6000 uu/s, × 1.75 when the
/// ball was then higher than 102.75 uu (off the floor): 0 to 1.75. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalSpeedAndPlacement {}

impl Reward for GoalSpeedAndPlacement {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        if cur.goal_scored != Some(cur.cars[car].team) {
            return 0.0;
        }
        let speed = prev.ball.vel.length() / BALL_MAX_SPEED;
        if prev.ball.pos.z > MIN_HEIGHT {
            HEIGHT_REWARD * speed
        } else {
            speed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// The step before Blue's goal (the ball flying at `speed` uu/s, `height` uu up) and the goal step.
    fn blue_goal(speed: f32, height: f32) -> (GameSnapshot, GameSnapshot) {
        let mut prev = game(
            Vec3A::new(0.0, 5000.0, height),
            &[
                (Team::Blue, Vec3A::new(0.0, 3000.0, 17.0)),
                (Team::Blue, Vec3A::new(500.0, 2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 4000.0, 17.0)),
            ],
        );
        prev.ball.vel = Vec3A::new(0.0, speed, 0.0);
        let mut cur = prev.clone();
        cur.ball.pos.y = 5300.0;
        cur.goal_scored = Some(Team::Blue);
        (prev, cur)
    }

    fn mirrored(g: &GameSnapshot) -> GameSnapshot {
        let mut m = g.clone();
        m.ball.pos = as_blue(Team::Orange, m.ball.pos);
        m.ball.vel = as_blue(Team::Orange, m.ball.vel);
        for car in &mut m.cars {
            car.team = car.team.opposite();
            car.body.pos = as_blue(Team::Orange, car.body.pos);
        }
        m.goal_scored = m.goal_scored.map(Team::opposite);
        m
    }

    #[test]
    fn goal_speed_pays_the_scoring_team_the_ball_speed() {
        let reward: GoalSpeed = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, cur) = blue_goal(3000.0, 93.15);
        for car in [0, 1] {
            assert!((reward.value(&prev, &cur, &quiet(3), car) - 600.0).abs() < 1e-3);
        }
        assert_eq!(reward.value(&prev, &cur, &quiet(3), 2), 0.0, "conceded");
        assert_eq!(reward.value(&prev, &prev, &quiet(3), 0), 0.0, "no goal");
        // Mirrored, the two scorers are Orange.
        let (prev, cur) = (mirrored(&prev), mirrored(&cur));
        assert!(
            (reward.value(&prev, &cur, &quiet(3), 0) - 600.0).abs() < 1e-3,
            "Orange alike"
        );
        assert_eq!(reward.value(&prev, &cur, &quiet(3), 2), 0.0);
    }

    #[test]
    fn goal_speed_and_placement_pays_more_off_the_floor() {
        let reward: GoalSpeedAndPlacement = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, mut cur) = blue_goal(3000.0, 93.15);
        cur.ball.vel = Vec3A::ZERO;
        for car in [0, 1] {
            assert!(
                (reward.value(&prev, &cur, &quiet(3), car) - 0.5).abs() < 1e-6,
                "the speed before the goal, every car of the team"
            );
        }
        assert_eq!(reward.value(&prev, &cur, &quiet(3), 2), 0.0, "conceded");
        assert_eq!(reward.value(&prev, &prev, &quiet(3), 0), 0.0, "no goal");
        let (prev, cur) = blue_goal(3000.0, 500.0);
        assert!((reward.value(&prev, &cur, &quiet(3), 0) - 0.875).abs() < 1e-6);
        // Mirrored, the two scorers are Orange.
        let (prev, cur) = (mirrored(&prev), mirrored(&cur));
        assert!(
            (reward.value(&prev, &cur, &quiet(3), 0) - 0.875).abs() < 1e-6,
            "Orange alike"
        );
        assert_eq!(reward.value(&prev, &cur, &quiet(3), 2), 0.0);
    }
}
