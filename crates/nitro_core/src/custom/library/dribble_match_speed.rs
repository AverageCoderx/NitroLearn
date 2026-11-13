//! Dribbling on the ground while matching the ball's speed.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

const MIN_BALL_HEIGHT: f32 = 109.0;
const MAX_BALL_HEIGHT: f32 = 180.0;
const MAX_DISTANCE: f32 = 197.0;
const SPEED_MATCH_FACTOR: f32 = 2.0;

/// Every step the car is on the ground within 197 uu of a ball 109 to 180 uu high: its speed over 2300 uu/s
/// plus 2 × how closely its speed matches the ball's (1 − difference / sum): 0 to 3. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleMatchSpeed {}

impl Reward for DribbleMatchSpeed {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let ball = &cur.ball;
        if !(me.on_ground
            && (MIN_BALL_HEIGHT..=MAX_BALL_HEIGHT).contains(&ball.pos.z)
            && me.body.pos.distance(ball.pos) < MAX_DISTANCE)
        {
            return 0.0;
        }
        let car_speed = me.body.vel.length();
        let ball_speed = ball.vel.length();
        let sum = car_speed + ball_speed;
        let mismatch = if sum > 0.0 {
            (car_speed - ball_speed).abs() / sum
        } else {
            0.0
        };
        car_speed / CAR_MAX_SPEED + SPEED_MATCH_FACTOR * (1.0 - mismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_a_ground_dribble_more_when_the_speeds_match() {
        let reward: DribbleMatchSpeed = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(0.0, 0.0, 150.0),
            &[
                (Team::Blue, Vec3A::new(0.0, 0.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        );
        assert_eq!(
            reward.value(&g, &g, &quiet(2), 0),
            2.0,
            "both still: matched"
        );
        g.cars[0].body.vel = Vec3A::new(0.0, 1150.0, 0.0);
        g.ball.vel = Vec3A::new(0.0, 1150.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(2), 0) - 2.5).abs() < 1e-6);
        g.ball.vel = Vec3A::new(0.0, 3450.0, 0.0);
        assert!(
            (reward.value(&g, &g, &quiet(2), 0) - 1.5).abs() < 1e-6,
            "ball faster"
        );
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 0.0, "far from the ball");
        let mut orange = g.clone();
        orange.cars[1].body.pos = Vec3A::new(0.0, 0.0, 17.0);
        orange.cars[1].body.vel = Vec3A::new(0.0, 1150.0, 0.0);
        assert!(
            (reward.value(&orange, &orange, &quiet(2), 1) - 1.5).abs() < 1e-6,
            "Orange alike"
        );
        let mut air = g.clone();
        air.cars[0].on_ground = false;
        assert_eq!(reward.value(&air, &air, &quiet(2), 0), 0.0, "in the air");
        let mut high = g.clone();
        high.ball.pos.z = 181.0;
        assert_eq!(
            reward.value(&high, &high, &quiet(2), 0),
            0.0,
            "ball too high"
        );
    }
}
