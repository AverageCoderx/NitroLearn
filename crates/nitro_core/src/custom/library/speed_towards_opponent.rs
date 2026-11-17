//! Driving at the opponent, from a community GigaLearn bot's rewards (SpeedTowardsOpponentReward in its
//! CommonRewards.h), whose author let it be used (CREDITS.md).
//! Changed: nothing (its car-velocity-toward-a-point maths is written fresh).

use crate::custom::prelude::*;

/// Every step: this car's speed toward the first opponent (in the car order) as a share of its top speed,
/// −1 to 1, times `scale`. 0 with no opponent. Meant for 1v1.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedTowardsOpponent {
    /// Multiplies the value (the original's default: 1).
    #[serde(default = "one")]
    pub scale: f32,
}

fn one() -> f32 {
    1.0
}

impl Reward for SpeedTowardsOpponent {
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
        let Some(other) = opponents(cur, car).next() else {
            return 0.0;
        };
        let me = &cur.cars[car].body;
        let toward = (cur.cars[other].body.pos - me.pos).normalize_or_zero();
        (toward.dot(me.vel) / CAR_MAX_SPEED).clamp(-1.0, 1.0) * self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_speed_toward_the_opponent_for_either_team() {
        let r: SpeedTowardsOpponent = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = game(
            Vec3A::ZERO,
            &[
                (Team::Blue, Vec3A::new(0.0, -1000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 1000.0, 17.0)),
            ],
        );
        g.cars[0].body.vel = Vec3A::new(0.0, 1150.0, 0.0);
        g.cars[1].body.vel = Vec3A::new(0.0, -1150.0, 0.0);
        let events = quiet(2);
        assert!((r.value(&g, &g, &events, 0) - 0.5).abs() < 1e-6);
        assert!(
            (r.value(&g, &g, &events, 1) - 0.5).abs() < 1e-6,
            "Orange the same"
        );
        g.cars[0].body.vel = Vec3A::new(0.0, -3000.0, 0.0);
        assert_eq!(r.value(&g, &g, &events, 0), -1.0, "away, clamped");
        let doubled = SpeedTowardsOpponent { scale: 2.0 };
        assert_eq!(doubled.value(&g, &g, &events, 0), -2.0);
        g.cars[0].body.vel = Vec3A::new(1000.0, 0.0, 0.0);
        assert!(r.value(&g, &g, &events, 0).abs() < 1e-6, "across");
        let alone = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
        assert_eq!(r.value(&alone, &alone, &quiet(1), 0), 0.0);
    }
}
