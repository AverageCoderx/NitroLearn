//! Dribbling: the ball just above the car's roof and the car under it, away from the walls.

use crate::custom::prelude::*;

/// The ball radius this reward measures from (rlgym-sim's 92.75 uu, not the game's 91.25).
const RLGYM_BALL_RADIUS: f32 = 92.75;

/// Every step the ball is between 112.75 and 292.75 uu high and within 150 uu of the car along x and y
/// (compared as distances from the centre lines), away from the walls: 0.2, or 1 with a touch. Else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleRoof {}

impl Reward for DribbleRoof {
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
        let (ball, pos) = (cur.ball.pos, cur.cars[car].body.pos);
        // |car x| against |ball x| (and y): mirrored spots across a centre line pass too.
        // Its first test, ball z > 100, is inside the next one.
        let dribbling = RLGYM_BALL_RADIUS + 20.0 < ball.z
            && ball.z < RLGYM_BALL_RADIUS + 200.0
            && (pos.x.abs() - ball.x.abs()).abs() < 150.0
            && (pos.y.abs() - ball.y.abs()).abs() < 150.0
            && ball.x.abs() < 3900.0
            && ball.y.abs() < 5000.0;
        if !dribbling {
            0.0
        } else if events.cars[car].touched_ball {
            1.0
        } else {
            0.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    #[test]
    fn pays_a_ball_on_the_roof_more_with_a_touch() {
        let reward: DribbleRoof = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = game(
            Vec3A::new(500.0, 1000.0, 180.0),
            &[
                (Team::Blue, Vec3A::new(520.0, 980.0, 17.0)),
                (Team::Orange, Vec3A::new(520.0, 3000.0, 17.0)),
            ],
        );
        assert!((reward.value(&g, &g, &quiet(2), 0) - 0.2).abs() < 1e-6);
        assert_eq!(reward.value(&g, &g, &touched(quiet(2), 0), 0), 1.0);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 0.0, "far from the ball");
        let mirrored = game(
            Vec3A::new(-500.0, -1000.0, 180.0),
            &[
                (Team::Blue, Vec3A::new(-520.0, -3000.0, 17.0)),
                (Team::Orange, Vec3A::new(-520.0, -980.0, 17.0)),
            ],
        );
        assert!(
            (reward.value(&mirrored, &mirrored, &quiet(2), 1) - 0.2).abs() < 1e-6,
            "Orange alike"
        );
        let mut opposite = g.clone();
        opposite.cars[1].body.pos = Vec3A::new(-520.0, -980.0, 17.0);
        assert!(
            (reward.value(&opposite, &opposite, &quiet(2), 1) - 0.2).abs() < 1e-6,
            "the |x|, |y| test passes the spot mirrored across the centre lines too"
        );
        let mut on_floor = g.clone();
        on_floor.ball.pos.z = 93.15;
        assert_eq!(reward.value(&on_floor, &on_floor, &quiet(2), 0), 0.0);
        let mut far = g.clone();
        far.cars[0].body.pos.y = 700.0;
        assert_eq!(reward.value(&far, &far, &quiet(2), 0), 0.0);
        let mut by_wall = g.clone();
        by_wall.ball.pos.x = 3950.0;
        by_wall.cars[0].body.pos.x = 3950.0;
        assert_eq!(reward.value(&by_wall, &by_wall, &quiet(2), 0), 0.0);
    }
}
