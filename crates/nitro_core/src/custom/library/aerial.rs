//! Aerials: a touch paid by the ball's height (jump_touch), flying fast at a high ball
//! (aerial_fly_at_ball), and aerial touches paid by the distance travelled (aerial_chain_distance).
//! Based on rewards from apollo-bot by redd-rl. aerial_chain_distance also follows AerialDistanceReward of
//! rlgym-tools (github.com/RLGym/rlgym-tools, by Rolv-Arild Braaten), licensed under the Apache License,
//! Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; the full text is in LICENSES/Apache-2.0.txt).
//! Modified: written in Rust; one leading car at a time; its travel and the ball's are added up step by
//! step and paid on its next touch; distances over 10240 uu.

use crate::custom::prelude::*;

/// The source's ball radius (RLGymSim's 92.75, not the game's 91.25).
const RADIUS: f32 = 92.75;
/// JumpTouch: ball heights count up to half the ceiling.
const HALF_CEILING: f32 = CEILING_Z / 2.0;
/// JumpTouch: after a paid touch, the car's next this many steps pay nothing.
const WAIT_STEPS: f32 = 149.0;
/// Aerial: the ball's lowest height (uu).
const BALL_MIN_HEIGHT: f32 = 300.0;
/// AerialChainDistance: below this height (uu) the car is not in an aerial.
const RAMP_HEIGHT: f32 = 256.0;

/// On a touch with the car in the air and the ball at least `min_height` up: (the ball's height above its
/// radius, counted up to 1022 uu, over 929.25) to the power `exp`: up to 1, paid only above 0.05.
/// After a paid touch the car's touches pay nothing for 149 steps.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JumpTouch {
    /// The ball's lowest height (uu) for a touch to count.
    #[serde(default = "two_hundred")]
    pub min_height: f32,
    /// The curve: 1 = in proportion to the ball's height; below 1 pays low touches more.
    #[serde(default = "one")]
    pub exp: f32,
    /// true: touches count with the car on its wheels (the floor, a wall) instead.
    #[serde(default)]
    pub car_on_ground: bool,
}

fn two_hundred() -> f32 {
    200.0
}

fn one() -> f32 {
    1.0
}

impl Reward for JumpTouch {
    /// Steps left before the car's touches pay again.
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
        let me = &cur.cars[car];
        let ball_z = cur.ball.pos.z;
        if events.cars[car].touched_ball
            && me.on_ground == self.car_on_ground
            && ball_z >= self.min_height
            && memory[0] <= 0.0
        {
            memory[0] = WAIT_STEPS;
            let reward = (ball_z.min(HALF_CEILING) - RADIUS).powf(self.exp)
                / (HALF_CEILING - RADIUS).powf(self.exp);
            // A NaN failed this test in the source too; an infinity (odd settings only) is held back with it.
            if reward.is_finite() && reward > 0.05 {
                return reward;
            }
        }
        memory[0] = (memory[0] - 1.0).max(0.0);
        0.0
    }
}

/// Every step the car is in the air moving towards a ball at least 300 uu up: its speed towards the ball
/// over 2300 uu/s × e^(−0.5 × distance / 2300), the distance to the ball's surface, plus 0.1 once it has
/// double jumped: about 0 to 1.1. 0 on the ground, moving away, or with the ball lower.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialFlyAtBall {}

impl Reward for AerialFlyAtBall {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let to_ball = (cur.ball.pos - me.body.pos).normalize_or_zero();
        let distance = me.body.pos.distance(cur.ball.pos) - RADIUS;
        let near = (-0.5 * distance / CAR_MAX_SPEED).exp();
        let speed = me.body.vel.dot(to_ball);
        if speed < 0.0 || me.on_ground || cur.ball.pos.z < BALL_MIN_HEIGHT {
            return 0.0;
        }
        let reward = speed / CAR_MAX_SPEED * near;
        if me.has_double_jumped {
            reward + 0.1
        } else {
            reward
        }
    }
}

/// Touches in an aerial, led by one car at a time: a car's first touch above 256 uu pays `height_scale` × how
/// far the car and the ball together are above 512 uu; each later touch of the leader while it stays above
/// 256 uu pays `distance_scale` × how far it and the ball moved since its last; both over 10240. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AerialChainDistance {
    /// The first touch's pay per uu of height.
    #[serde(default = "one")]
    pub height_scale: f32,
    /// A later touch's pay per uu the car and ball travelled.
    #[serde(default = "one")]
    pub distance_scale: f32,
}

impl Reward for AerialChainDistance {
    /// The source's state, shared by all cars, which every car works out alike: the leader (its index + 1,
    /// 0 for none), and how far the leader and the ball have moved since the leader's last touch.
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
        let (mut leader, mut car_moved, mut ball_moved) = (memory[0], memory[1], memory[2]);
        let ball_step = cur.ball.pos.distance(prev.ball.pos);
        let mut mine = 0.0;
        // The source's cars one after the other, as it was asked for each in turn.
        for (index, other) in cur.cars.iter().enumerate() {
            let touched = events.cars[index].touched_ball;
            let mut leading = leader == (index + 1) as f32;
            let mut reward = 0.0;
            if other.body.pos.z < RAMP_HEIGHT {
                if leading {
                    leading = false;
                    leader = 0.0;
                }
            } else if touched && !leading {
                leading = true;
                car_moved = 0.0;
                ball_moved = 0.0;
                let height = other.body.pos.z + cur.ball.pos.z - 2.0 * RAMP_HEIGHT;
                reward = self.height_scale * height.max(0.0);
            } else if leading {
                car_moved += other.body.pos.distance(prev.cars[index].body.pos);
                ball_moved += ball_step;
                if touched {
                    reward = self.distance_scale * (car_moved + ball_moved);
                    car_moved = 0.0;
                    ball_moved = 0.0;
                }
            }
            if leading {
                leader = (index + 1) as f32;
            }
            if index == car {
                mine = reward / (2.0 * BACK_WALL_Y);
            }
        }
        memory.copy_from_slice(&[leader, car_moved, ball_moved]);
        mine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// Blue and Orange in the air at `z`, 100 uu either side of a ball at `ball_z`.
    fn up(z: f32, ball_z: f32) -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, ball_z),
            &[
                (Team::Blue, Vec3A::new(0.0, -100.0, z)),
                (Team::Orange, Vec3A::new(0.0, 100.0, z)),
            ],
        );
        for car in &mut g.cars {
            car.on_ground = false;
        }
        g
    }

    #[test]
    fn jump_touch_pays_air_touches_by_height_then_waits() {
        let reward: JumpTouch = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = up(1000.0, 1022.0);
        let mut memory = vec![0.0; reward.memory()];
        let hit = touched(quiet(2), 0);
        let paid = reward.value_remembering(&g, &g, &hit, 0, &mut memory);
        assert!((paid - 1.0).abs() < 1e-5, "{paid}");
        for _ in 0..149 {
            assert_eq!(
                reward.value_remembering(&g, &g, &hit, 0, &mut memory),
                0.0,
                "waiting"
            );
        }
        assert!((reward.value_remembering(&g, &g, &hit, 0, &mut memory) - 1.0).abs() < 1e-5);
        let mut fresh = [0.0];
        let orange = touched(quiet(2), 1);
        assert!(
            (reward.value_remembering(&g, &g, &orange, 1, &mut fresh) - 1.0).abs() < 1e-5,
            "Orange alike, and each car waits on its own"
        );
        let half = up(500.0, (HALF_CEILING + RADIUS) / 2.0);
        let paid = reward.value_remembering(&half, &half, &hit, 0, &mut [0.0]);
        assert!((paid - 0.5).abs() < 1e-5, "{paid}");
        let mut ground = g.clone();
        ground.cars[0].on_ground = true;
        assert_eq!(
            reward.value_remembering(&ground, &ground, &hit, 0, &mut [0.0]),
            0.0,
            "car on its wheels"
        );
        let as_source = JumpTouch {
            min_height: 200.0,
            exp: 1.0,
            car_on_ground: true,
        };
        assert!(
            (as_source.value_remembering(&ground, &ground, &hit, 0, &mut [0.0]) - 1.0).abs() < 1e-5
        );
        let bent = JumpTouch {
            exp: 0.8,
            ..serde_json::from_value(serde_json::json!({})).unwrap()
        };
        let low_touch = up(500.0, 300.0);
        let straight = reward.value_remembering(&low_touch, &low_touch, &hit, 0, &mut [0.0]);
        let curved = bent.value_remembering(&low_touch, &low_touch, &hit, 0, &mut [0.0]);
        let share = (300.0 - RADIUS) / (HALF_CEILING - RADIUS);
        assert!((straight - share).abs() < 1e-5 && (curved - share.powf(0.8)).abs() < 1e-5);
        assert!(curved > straight, "exp below 1 pays a low touch more");
        let low = up(100.0, 150.0);
        assert_eq!(
            reward.value_remembering(&low, &low, &hit, 0, &mut [0.0]),
            0.0,
            "ball too low"
        );
        assert_eq!(
            reward.value_remembering(&g, &g, &quiet(2), 0, &mut [0.0]),
            0.0,
            "no touch"
        );
    }

    #[test]
    fn aerial_pays_flying_at_a_high_ball() {
        let reward: AerialFlyAtBall = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut g = up(1000.0, 1000.0);
        g.cars[0].body.pos = Vec3A::new(0.0, -RADIUS, 1000.0);
        g.cars[0].body.vel = Vec3A::new(0.0, 2300.0, 0.0);
        g.cars[1].body.pos = Vec3A::new(0.0, RADIUS, 1000.0);
        g.cars[1].body.vel = Vec3A::new(0.0, -2300.0, 0.0);
        assert!((reward.value(&g, &g, &quiet(2), 0) - 1.0).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 1.0).abs() < 1e-5,
            "Orange alike"
        );
        g.cars[0].has_double_jumped = true;
        assert!((reward.value(&g, &g, &quiet(2), 0) - 1.1).abs() < 1e-5);
        let mut far = g.clone();
        far.cars[0].body.pos.y = -RADIUS - 4600.0;
        let expected = (-1.0f32).exp() + 0.1;
        assert!((reward.value(&far, &far, &quiet(2), 0) - expected).abs() < 1e-5);
        let mut ground = g.clone();
        ground.cars[0].on_ground = true;
        assert_eq!(
            reward.value(&ground, &ground, &quiet(2), 0),
            0.0,
            "on the ground"
        );
        let mut away = g.clone();
        away.cars[0].body.vel.y = -100.0;
        assert_eq!(reward.value(&away, &away, &quiet(2), 0), 0.0, "moving away");
        let mut low = g.clone();
        low.ball.pos.z = 299.0;
        assert_eq!(reward.value(&low, &low, &quiet(2), 0), 0.0, "ball too low");
    }

    #[test]
    fn aerial_distance_pays_the_first_touch_by_height_then_the_distance_carried() {
        let reward: AerialChainDistance = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut memories = [vec![0.0; reward.memory()], vec![0.0; reward.memory()]];
        let mut step = |prev: &GameSnapshot, cur: &GameSnapshot, events: &StepEvents| {
            let [a, b] = &mut memories;
            [
                reward.value_remembering(prev, cur, events, 0, a),
                reward.value_remembering(prev, cur, events, 1, b),
            ]
        };
        let start = up(600.0, 700.0);
        let paid = step(&start, &start, &touched(quiet(2), 0));
        assert!(
            (paid[0] - 788.0 / 10240.0).abs() < 1e-6 && paid[1] == 0.0,
            "{paid:?}"
        );
        let mut moved = start.clone();
        moved.cars[0].body.pos.y += 300.0;
        moved.ball.pos.y += 400.0;
        assert_eq!(
            step(&start, &moved, &quiet(2)),
            [0.0, 0.0],
            "carried, no touch"
        );
        let mut again = moved.clone();
        again.cars[0].body.pos.y += 100.0;
        again.ball.pos.y += 200.0;
        let paid = step(&moved, &again, &touched(quiet(2), 0));
        assert!((paid[0] - 1000.0 / 10240.0).abs() < 1e-6, "{paid:?}");
        let paid = step(&again, &again, &touched(quiet(2), 1));
        assert!(
            paid[0] == 0.0 && (paid[1] - 788.0 / 10240.0).abs() < 1e-6,
            "Orange takes the lead, paid alike: {paid:?}"
        );
        let paid = step(&again, &again, &touched(quiet(2), 0));
        assert!(
            (paid[0] - 788.0 / 10240.0).abs() < 1e-6,
            "Blue's touch is a first one again: {paid:?}"
        );
        let mut landed = again.clone();
        landed.cars[0].body.pos.z = 17.0;
        assert_eq!(
            step(&again, &landed, &touched(quiet(2), 0)),
            [0.0, 0.0],
            "too low"
        );
        let mut fresh = [0.0; 3];
        assert!(
            (reward.value_remembering(&start, &start, &touched(quiet(2), 0), 0, &mut fresh)
                - 788.0 / 10240.0)
                .abs()
                < 1e-6,
            "a new episode starts again"
        );
    }
}
