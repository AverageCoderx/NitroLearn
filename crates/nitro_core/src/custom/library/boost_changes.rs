//! Changes in the car's boost: boost gained (boost_gain), small pads (small_boost_pickup) and boost spent
//! (boost_use).

use crate::custom::prelude::*;

/// When the car's boost goes up: the boost gained, as a fraction of a full tank (0 to 1), or its square
/// root (`sqrt`, so a small pad's 12 pays 0.35). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostGain {
    /// true = the square root of the boost gained; false = the boost gained itself.
    #[serde(default = "yes")]
    pub sqrt: bool,
}

fn yes() -> bool {
    true
}

impl Reward for BoostGain {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let gained = (cur.cars[car].boost - prev.cars[car].boost) / 100.0;
        if gained <= 0.0 {
            0.0
        } else if self.sqrt {
            // The same number as sqrt(gained * 100) / 10.
            gained.sqrt()
        } else {
            gained
        }
    }
}

/// 1 when the car's boost goes up by less than 15 (a small pad, or a big pad when nearly full), else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmallBoostPickup {}

impl Reward for SmallBoostPickup {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let gained = (cur.cars[car].boost - prev.cars[car].boost) / 100.0;
        f32::from(u8::from(gained > 0.0 && gained < 0.15))
    }
}

/// When the car's boost goes down: the square root of the boost spent as a fraction of a full tank (a
/// positive reward for using boost; a step of full boosting, about 2.2, pays 0.15). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostUse {}

impl Reward for BoostUse {
    fn value(&self, prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let spent = (prev.cars[car].boost - cur.cars[car].boost) / 100.0;
        if spent > 0.0 { spent.sqrt() } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn two_cars() -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -3000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        )
    }

    #[test]
    fn boost_pickup_pays_the_gain_or_its_square_root() {
        let sqrt: BoostGain = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(sqrt.sqrt, "the square root by default");
        let linear = BoostGain { sqrt: false };
        let prev = two_cars();
        let mut cur = prev.clone();
        assert_eq!(sqrt.value(&prev, &cur, &quiet(2), 0), 0.0, "no change");
        cur.cars[0].boost = 45.0;
        cur.cars[1].boost = 45.0;
        assert!((sqrt.value(&prev, &cur, &quiet(2), 0) - 0.12f32.sqrt()).abs() < 1e-5);
        assert!((linear.value(&prev, &cur, &quiet(2), 1) - 0.12).abs() < 1e-5);
        assert_eq!(
            sqrt.value(&prev, &cur, &quiet(2), 0),
            sqrt.value(&prev, &cur, &quiet(2), 1),
            "Orange alike"
        );
        cur.cars[0].boost = 20.0;
        assert_eq!(sqrt.value(&prev, &cur, &quiet(2), 0), 0.0, "boost spent");
    }

    #[test]
    fn boost_pickup_compares_each_car_with_itself() {
        let reward = BoostGain { sqrt: false };
        let mut prev = two_cars();
        prev.cars[1].boost = 100.0;
        let cur = prev.clone();
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0);
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 1), 0.0);
    }

    #[test]
    fn small_boost_pickup_counts_gains_below_15() {
        let reward: SmallBoostPickup = serde_json::from_value(serde_json::json!({})).unwrap();
        let prev = two_cars();
        let mut cur = prev.clone();
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0);
        cur.cars[0].boost = 45.0;
        cur.cars[1].boost = 100.0;
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 1.0, "a small pad");
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 1), 0.0, "a big pad");
    }

    #[test]
    fn boost_use_pays_the_square_root_of_boost_spent() {
        let reward: BoostUse = serde_json::from_value(serde_json::json!({})).unwrap();
        let prev = two_cars();
        let mut cur = prev.clone();
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 1), 0.0);
        cur.cars[1].boost = 29.0;
        assert!((reward.value(&prev, &cur, &quiet(2), 1) - 0.2).abs() < 1e-5);
        cur.cars[1].boost = 50.0;
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 1), 0.0, "boost gained");
    }
}
