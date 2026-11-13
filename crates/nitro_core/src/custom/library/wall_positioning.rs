//! Waiting on the wall under the ball with boost, ready to start an air dribble.

use crate::custom::prelude::*;

/// The ball radius this reward measures from (rlgym-sim's 92.75 uu, not the game's 91.25).
const RLGYM_BALL_RADIUS: f32 = 92.75;

/// Every step the car is on a wall above 200 uu with the ball above it and within 463.75 uu (5 ball
/// radii), holding more than 50 boost: 1, or 4 with a touch. Else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WallPositioning {}

impl Reward for WallPositioning {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        let (ball, pos) = (cur.ball.pos, me.body.pos);
        let in_position = me.on_ground
            && pos.z > 200.0
            && ball.z > pos.z
            && ball.distance(pos) < RLGYM_BALL_RADIUS * 5.0
            && me.boost > 50.0;
        if !in_position {
            0.0
        } else if events.cars[car].touched_ball {
            4.0
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn pays_waiting_under_the_ball_on_the_wall() {
        let reward: WallPositioning = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::new(4000.0, 0.0, 900.0),
            &[
                (Team::Blue, Vec3A::new(4079.0, 0.0, 600.0)),
                (Team::Orange, Vec3A::new(-4079.0, 0.0, 600.0)),
            ],
        );
        for car in &mut g.cars {
            car.boost = 60.0;
        }
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &touched(quiet(2), 0), 0), 4.0);
        let mut mirrored = g.clone();
        mirrored.ball.pos.x = -4000.0;
        assert_eq!(
            reward.value(&mirrored, &mirrored, &quiet(2), 1),
            1.0,
            "Orange alike"
        );
        let mut low_boost = g.clone();
        low_boost.cars[0].boost = 40.0;
        assert_eq!(reward.value(&low_boost, &low_boost, &quiet(2), 0), 0.0);
        let mut below = g.clone();
        below.ball.pos.z = 500.0;
        assert_eq!(
            reward.value(&below, &below, &quiet(2), 0),
            0.0,
            "ball below the car"
        );
        let mut jumped = g.clone();
        jumped.cars[0].on_ground = false;
        assert_eq!(
            reward.value(&jumped, &jumped, &quiet(2), 0),
            0.0,
            "off the wall"
        );
    }
}
