//! Touches scaled by height: of the ball (any touch, or in the air) or of the car (touching on a wall).

use crate::custom::prelude::*;

/// On a touch: the ball's height over 2000 uu (about 0.05 on the ground, 1 at the ceiling). With
/// `high_touch_penalty`, only above 200 uu and always negative instead: 1 − √(height / 20), −2.2 to about −8.9.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallTouchScaledByHeight {
    /// false: height / 2000 on every touch; true: 1 − √(height / 2000 × 100) on touches above 200 uu, a
    /// penalty growing with the height.
    #[serde(default)]
    pub high_touch_penalty: bool,
}

impl Reward for BallTouchScaledByHeight {
    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let height = cur.ball.pos.z;
        if !events.cars[car].touched_ball {
            0.0
        } else if !self.high_touch_penalty {
            height / 2000.0
        } else if height > 200.0 {
            1.0 - (height / 2000.0 * 100.0).sqrt()
        } else {
            0.0
        }
    }
}

/// On a touch while the car is in the air: the ball's height over 2000 uu (0 to about 1), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialTouchScaledByHeight {}

impl Reward for AerialTouchScaledByHeight {
    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if events.cars[car].touched_ball && !cur.cars[car].on_ground {
            cur.ball.pos.z / 2000.0
        } else {
            0.0
        }
    }
}

/// On a touch while the car's wheels are on a wall above 200 uu: the car's height over 2044 uu (0.1 to 1),
/// else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TouchBallOnWallScaledByHeight {}

impl Reward for TouchBallOnWallScaledByHeight {
    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        if events.cars[car].touched_ball && me.on_ground && me.body.pos.z > 200.0 {
            me.body.pos.z / 2044.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn ball_at(z: f32) -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, z),
            &[
                (Team::Blue, Vec3A::new(0.0, -150.0, z)),
                (Team::Orange, Vec3A::new(0.0, 150.0, z)),
            ],
        )
    }

    #[test]
    fn ball_touch_by_height_both_formulas() {
        let testing: BallTouchScaledByHeight =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let penalty = BallTouchScaledByHeight {
            high_touch_penalty: true,
        };
        let g = ball_at(1000.0);
        let both = touched(touched(quiet(2), 0), 1);
        assert_eq!(testing.value(&g, &g, &quiet(2), 0), 0.0, "no touch");
        assert!((testing.value(&g, &g, &both, 0) - 0.5).abs() < 1e-6);
        assert_eq!(testing.value(&g, &g, &both, 1), 0.5, "Orange alike");
        assert!((penalty.value(&g, &g, &both, 0) - (1.0 - 50f32.sqrt())).abs() < 1e-5);
        assert!(penalty.value(&g, &g, &both, 0) < -2.0, "a penalty");
        let low = ball_at(150.0);
        assert_eq!(penalty.value(&low, &low, &both, 0), 0.0, "not above 200 uu");
        assert!(testing.value(&low, &low, &both, 0) > 0.0);
    }

    #[test]
    fn aerial_touch_needs_the_car_in_the_air() {
        let reward: AerialTouchScaledByHeight =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = ball_at(1000.0);
        let events = touched(quiet(2), 1);
        assert_eq!(reward.value(&g, &g, &events, 1), 0.0, "on the ground");
        g.cars[1].on_ground = false;
        assert_eq!(reward.value(&g, &g, &events, 1), 0.5);
        assert_eq!(reward.value(&g, &g, &quiet(2), 1), 0.0, "no touch");
    }

    #[test]
    fn wall_touch_by_the_cars_height() {
        let reward: TouchBallOnWallScaledByHeight =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = ball_at(1022.0);
        let events = touched(quiet(2), 0);
        assert!((reward.value(&g, &g, &events, 0) - 0.5).abs() < 1e-6);
        assert_eq!(reward.value(&g, &g, &quiet(2), 0), 0.0, "no touch");
        g.cars[0].on_ground = false;
        assert_eq!(reward.value(&g, &g, &events, 0), 0.0, "in the air");
        let floor = ball_at(100.0);
        assert_eq!(
            reward.value(&floor, &floor, &events, 0),
            0.0,
            "on the floor"
        );
    }
}
