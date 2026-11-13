//! Aerial chains: the first air touch paid by how high car and ball are, each further one by how far
//! they travelled since the last.
//! Based on a reward by Opti.

use crate::custom::prelude::*;

/// Below this height (uu) a car counts as on the ground (rlgym-sim's RAMP_HEIGHT).
const RAMP_HEIGHT: f32 = 256.0;

/// Car `car` touched the ball this step at least RAMP_HEIGHT up, which makes it the chain's holder.
fn air_touch(cur: &GameSnapshot, events: &StepEvents, car: usize) -> bool {
    events.cars[car].touched_ball && cur.cars[car].body.pos.z >= RAMP_HEIGHT
}

/// On the car's first touch 256+ uu up: `height_scale` × (car height + ball height − 512 uu, at least 0);
/// on each further touch before it comes down or another car touches in the air: `distance_scale` × how far
/// the car and the ball travelled since. Both over 10240 uu (about 0.03 to 0.15). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialChainDistanceV2 {
    /// What each uu of the first touch's car and ball height (above 256 uu each) is worth, over 10240.
    #[serde(default = "half")]
    pub height_scale: f32,
    /// What each uu that car and ball travelled between touches is worth, over 10240.
    #[serde(default = "half_distance")]
    pub distance_scale: f32,
}

fn half() -> f32 {
    0.5
}

fn half_distance() -> f32 {
    0.5
}

impl Reward for AerialChainDistanceV2 {
    fn memory(&self) -> usize {
        3
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        // memory[0]: 1 while this car holds the chain; [1], [2]: how far it and the ball travelled since its
        // last touch (uu). The source looks at the cars in order, so a car earlier in the list that touches
        // in the air this step takes the chain before this car's turn, and one later in the list after it.
        let mut holding = memory[0] == 1.0 && !(0..car).any(|other| air_touch(cur, events, other));
        let me = &cur.cars[car];
        let touched = events.cars[car].touched_ball;
        let mut reward = 0.0;
        if me.body.pos.z < RAMP_HEIGHT {
            holding = false;
        } else if touched && !holding {
            holding = true;
            memory[1] = 0.0;
            memory[2] = 0.0;
            reward =
                self.height_scale * (me.body.pos.z + cur.ball.pos.z - 2.0 * RAMP_HEIGHT).max(0.0);
        } else if holding {
            memory[1] += me.body.pos.distance(prev.cars[car].body.pos);
            memory[2] += cur.ball.pos.distance(prev.ball.pos);
            if touched {
                reward = self.distance_scale * (memory[1] + memory[2]);
                memory[1] = 0.0;
                memory[2] = 0.0;
            }
        }
        if (car + 1..cur.cars.len()).any(|other| air_touch(cur, events, other)) {
            holding = false;
        }
        memory[0] = f32::from(u8::from(holding));
        reward / (2.0 * BACK_WALL_Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// A Blue car at `blue`, an Orange car at the mirrored spot, and the ball at `ball`.
    fn at(blue: Vec3A, ball: Vec3A) -> GameSnapshot {
        let mut g = game(
            ball,
            &[
                (Team::Blue, blue),
                (Team::Orange, Vec3A::new(-blue.x, -blue.y, blue.z)),
            ],
        );
        for c in &mut g.cars {
            c.on_ground = false;
        }
        g
    }

    fn both_cars(
        reward: &AerialChainDistanceV2,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        memory: &mut [[f32; 3]; 2],
    ) -> [f32; 2] {
        [0, 1].map(|car| reward.value_remembering(prev, cur, events, car, &mut memory[car]))
    }

    #[test]
    fn pays_the_first_air_touch_by_height_and_later_ones_by_distance() {
        let reward: AerialChainDistanceV2 = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut memory = [[0.0f32; 3]; 2];
        let s0 = at(
            Vec3A::new(0.0, -1000.0, 500.0),
            Vec3A::new(0.0, -900.0, 600.0),
        );
        let first = both_cars(&reward, &s0, &s0, &touched(quiet(2), 0), &mut memory);
        assert!((first[0] - 0.5 * 588.0 / 10240.0).abs() < 1e-6 && first[1] == 0.0);
        let s1 = at(
            Vec3A::new(100.0, -1000.0, 500.0),
            Vec3A::new(0.0, -700.0, 600.0),
        );
        assert_eq!(
            both_cars(&reward, &s0, &s1, &quiet(2), &mut memory),
            [0.0; 2]
        );
        let s2 = at(
            Vec3A::new(200.0, -1000.0, 500.0),
            Vec3A::new(0.0, -500.0, 600.0),
        );
        let again = both_cars(&reward, &s1, &s2, &touched(quiet(2), 0), &mut memory);
        assert!((again[0] - 0.5 * 600.0 / 10240.0).abs() < 1e-6, "{again:?}");
        let s3 = at(
            Vec3A::new(300.0, -1000.0, 500.0),
            Vec3A::new(0.0, -300.0, 600.0),
        );
        let next = both_cars(&reward, &s2, &s3, &touched(quiet(2), 0), &mut memory);
        assert!((next[0] - 0.5 * 300.0 / 10240.0).abs() < 1e-6, "{next:?}");
        let mut landed = s3.clone();
        landed.cars[0].body.pos.z = 17.0;
        assert_eq!(
            both_cars(&reward, &s3, &landed, &quiet(2), &mut memory),
            [0.0; 2]
        );
        let back_up = both_cars(&reward, &landed, &s3, &touched(quiet(2), 0), &mut memory);
        assert!(
            (back_up[0] - 0.5 * 588.0 / 10240.0).abs() < 1e-6,
            "a first touch again after coming down"
        );
        let low = at(
            Vec3A::new(0.0, -1000.0, 200.0),
            Vec3A::new(0.0, -900.0, 300.0),
        );
        let mut fresh = [[0.0f32; 3]; 2];
        assert_eq!(
            both_cars(&reward, &low, &low, &touched(quiet(2), 0), &mut fresh),
            [0.0; 2],
            "a touch below 256 uu"
        );
        let mut fresh = [[0.0f32; 3]; 2];
        let orange = both_cars(&reward, &s0, &s0, &touched(quiet(2), 1), &mut fresh);
        assert_eq!(orange, [0.0, first[0]], "Orange alike");
    }

    #[test]
    fn another_cars_air_touch_takes_the_chain() {
        let reward = AerialChainDistanceV2 {
            height_scale: 0.5,
            distance_scale: 0.5,
        };
        let mut memory = [[0.0f32; 3]; 2];
        let s0 = at(Vec3A::new(0.0, -1000.0, 500.0), Vec3A::new(0.0, 0.0, 600.0));
        both_cars(&reward, &s0, &s0, &touched(quiet(2), 0), &mut memory);
        let taken = both_cars(&reward, &s0, &s0, &touched(quiet(2), 1), &mut memory);
        assert!(taken[0] == 0.0 && taken[1] > 0.0, "{taken:?}");
        let retaken = both_cars(&reward, &s0, &s0, &touched(quiet(2), 0), &mut memory);
        assert!(
            (retaken[0] - 0.5 * 588.0 / 10240.0).abs() < 1e-6,
            "a first touch again: {retaken:?}"
        );
        let mut both = touched(quiet(2), 0);
        both.cars[1].touched_ball = true;
        let together = both_cars(&reward, &s0, &s0, &both, &mut memory);
        assert!(together[1] > 0.0, "the later car takes it: {together:?}");
        let after = both_cars(&reward, &s0, &s0, &touched(quiet(2), 1), &mut memory);
        assert_eq!(after[1], 0.0, "car 1 holds it: no move since, no pay");
        assert_eq!(memory[0][0], 0.0);
    }
}
