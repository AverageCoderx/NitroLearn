//! Kickoffs: speed towards the ball once the car has used up its boost.

use crate::custom::prelude::*;

/// Every step of a kickoff (the ball on the centre spot) while the car's boost is below `max_boost`:
/// `goal_speed` × its speed towards the ball over 2300 uu/s, at least `goal_speed` × 0.025 (0.0125 to
/// 0.5 with the defaults, even when moving away). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffSpeedNoBoostV2 {
    /// The reward at top speed straight at the ball.
    #[serde(default = "half")]
    pub goal_speed: f32,
    /// Pays only while the car's boost (0-100) is below this; 101 = at any boost.
    #[serde(default = "two")]
    pub max_boost: f32,
}

fn half() -> f32 {
    0.5
}

fn two() -> f32 {
    2.0
}

impl Reward for KickoffSpeedNoBoostV2 {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        // A kickoff: the ball at x = y = 0 exactly.
        let kickoff = cur.ball.pos.x == 0.0 && cur.ball.pos.y == 0.0;
        if !kickoff || me.boost >= self.max_boost {
            return 0.0;
        }
        let towards = (cur.ball.pos - me.body.pos).normalize_or_zero();
        self.goal_speed * towards.dot(me.body.vel / CAR_MAX_SPEED).max(0.025)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn kickoff(boost: f32) -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -3000.0, 93.15)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 93.15)),
            ],
        );
        g.cars[0].body.vel = Vec3A::new(0.0, 2300.0, 0.0);
        g.cars[1].body.vel = Vec3A::new(0.0, -2300.0, 0.0);
        for car in &mut g.cars {
            car.boost = boost;
        }
        g
    }

    #[test]
    fn pays_speed_at_the_ball_once_the_boost_is_gone() {
        let reward: KickoffSpeedNoBoostV2 = serde_json::from_value(serde_json::json!({})).unwrap();
        let empty = kickoff(1.0);
        assert!((reward.value(&empty, &empty, &quiet(2), 0) - 0.5).abs() < 1e-5);
        assert!(
            (reward.value(&empty, &empty, &quiet(2), 1) - 0.5).abs() < 1e-5,
            "Orange alike"
        );
        let full = kickoff(33.0);
        assert_eq!(reward.value(&full, &full, &quiet(2), 0), 0.0, "boost left");
        let any = KickoffSpeedNoBoostV2 {
            goal_speed: 0.5,
            max_boost: 101.0,
        };
        assert!(
            (any.value(&full, &full, &quiet(2), 0) - 0.5).abs() < 1e-5,
            "max_boost above a full tank pays with any boost"
        );
        let mut away = empty.clone();
        away.cars[0].body.vel = Vec3A::new(0.0, -1000.0, 0.0);
        assert!(
            (reward.value(&away, &away, &quiet(2), 0) - 0.0125).abs() < 1e-6,
            "the floor"
        );
        let mut played = empty.clone();
        played.ball.pos.x = 1.0;
        assert_eq!(
            reward.value(&played, &played, &quiet(2), 0),
            0.0,
            "not a kickoff"
        );
    }
}
