//! Getting a flip back high up (a flip reset), paid by the car's height.

use crate::custom::prelude::*;

/// When the car gets its flip back while higher than `minimum_height`: its height × `height_scaling`
/// (250 at 1000 uu with the defaults). Landing on a wall or the ceiling above that height counts too.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlipResetHeightV2 {
    /// What each uu of the car's height is worth.
    #[serde(default = "quarter")]
    pub height_scaling: f32,
    /// Only higher than this (uu).
    #[serde(default = "hundred")]
    pub minimum_height: f32,
}

fn quarter() -> f32 {
    0.25
}

fn hundred() -> f32 {
    100.0
}

impl Reward for FlipResetHeightV2 {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let regained = !prev.cars[car].can_flip && me.can_flip;
        if regained && me.body.pos.z > self.minimum_height {
            me.body.pos.z * self.height_scaling
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn high() -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 1100.0),
            &[
                (Team::Blue, Vec3A::new(0.0, -50.0, 1000.0)),
                (Team::Orange, Vec3A::new(0.0, 50.0, 1000.0)),
            ],
        );
        for car in &mut g.cars {
            car.on_ground = false;
            car.can_flip = false;
        }
        g
    }

    #[test]
    fn pays_the_height_when_the_flip_comes_back() {
        let reward: FlipResetHeightV2 = serde_json::from_value(serde_json::json!({})).unwrap();
        let prev = high();
        let mut cur = prev.clone();
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0, "no flip back");
        cur.cars[0].can_flip = true;
        cur.cars[1].can_flip = true;
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 250.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 1),
            250.0,
            "Orange alike"
        );
        assert_eq!(
            reward.value(&cur, &cur, &quiet(2), 0),
            0.0,
            "it already had its flip"
        );
        let mut low = cur.clone();
        low.cars[0].body.pos.z = 90.0;
        assert_eq!(reward.value(&prev, &low, &quiet(2), 0), 0.0, "too low");
    }

    #[test]
    fn each_car_against_itself() {
        let reward = FlipResetHeightV2 {
            height_scaling: 1.0,
            minimum_height: 100.0,
        };
        let mut prev = high();
        prev.cars[1].can_flip = true;
        let mut cur = prev.clone();
        cur.cars[0].can_flip = true;
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 1000.0);
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 1),
            0.0,
            "car 1 had it already"
        );
    }
}
