//! Touching the ball: 1 a touch for a beginner, otherwise the ball speed gained since the car's last touch.

use crate::custom::prelude::*;

/// On a touch: 1 with `beginner`; otherwise how much faster the ball is than after this car's previous
/// touch this episode, in uu/s (0 to 6000, never negative; the first touch pays the ball's whole speed).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TouchBallSpeedGain {
    /// true = 1 for every touch; false = the ball speed gained (uu/s, so give it a small weight).
    #[serde(default = "yes")]
    pub beginner: bool,
}

fn yes() -> bool {
    true
}

impl Reward for TouchBallSpeedGain {
    fn memory(&self) -> usize {
        1
    }

    fn value_remembering(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        if self.beginner {
            return 1.0;
        }
        // memory[0]: the ball's speed after this car's last touch (0 before its first).
        let speed = cur.ball.vel.length();
        let gain = (speed - memory[0]).max(0.0);
        memory[0] = speed;
        gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn moving_ball(speed: f32) -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -150.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 150.0, 17.0)),
            ],
        );
        g.ball.vel = Vec3A::new(0.0, speed, 0.0);
        g
    }

    #[test]
    fn beginner_pays_one_a_touch() {
        let reward: TouchBallSpeedGain = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(reward.beginner);
        let g = moving_ball(1000.0);
        let mut memory = [0.0];
        let events = touched(quiet(2), 1);
        assert_eq!(
            reward.value_remembering(&g, &g, &events, 1, &mut memory),
            1.0
        );
        assert_eq!(
            reward.value_remembering(&g, &g, &quiet(2), 1, &mut memory),
            0.0
        );
    }

    #[test]
    fn otherwise_the_speed_gained_since_the_cars_own_last_touch() {
        let reward = TouchBallSpeedGain { beginner: false };
        let mut memory = [[0.0f32; 1]; 2];
        let blue = touched(quiet(2), 0);
        let orange = touched(quiet(2), 1);
        let fast = moving_ball(1500.0);
        let slow = moving_ball(800.0);
        let faster = moving_ball(2000.0);
        assert_eq!(
            reward.value_remembering(&fast, &fast, &blue, 0, &mut memory[0]),
            1500.0,
            "the first touch pays the whole speed"
        );
        assert_eq!(
            reward.value_remembering(&slow, &slow, &orange, 1, &mut memory[1]),
            800.0,
            "Orange's first touch: Blue's does not count"
        );
        assert_eq!(
            reward.value_remembering(&slow, &slow, &blue, 0, &mut memory[0]),
            0.0,
            "slower than after Blue's last touch"
        );
        assert_eq!(
            reward.value_remembering(&faster, &faster, &blue, 0, &mut memory[0]),
            1200.0
        );
        assert_eq!(
            reward.value_remembering(&faster, &faster, &quiet(2), 0, &mut memory[0]),
            0.0,
            "no touch"
        );
        let mut fresh = [0.0];
        assert_eq!(
            reward.value_remembering(&slow, &slow, &blue, 0, &mut fresh),
            800.0,
            "a new episode starts from 0"
        );
    }
}
