//! Possession: being nearest the ball after one's team touched it last, paid more the longer it lasts.
//! Based on a reward from apollo-bot by redd-rl.

use crate::custom::prelude::*;

/// Every step the car's team touched the ball last and the car is within `min_dist` of it with no opponent
/// nearer: the possession's count over 10, the count going up by 1 each such step of each such car and
/// starting again when the other team touches the ball (0.1, 0.2, 0.3 …, without end). 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Possession {
    /// How near the ball (uu) the car must be.
    #[serde(default = "two_hundred")]
    pub min_dist: f32,
}

fn two_hundred() -> f32 {
    200.0
}

/// A team as the memory keeps it: 0 for none yet.
fn code(team: Team) -> f32 {
    team.index() as f32 + 1.0
}

impl Reward for Possession {
    /// The source's state, shared by all cars, which every car works out alike: the team that touched the
    /// ball last, and the count.
    fn memory(&self) -> usize {
        2
    }

    fn every_step(&self) -> bool {
        true
    }

    fn value_remembering(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let (mut holder, mut count) = (memory[0], memory[1]);
        for (other, happened) in cur.cars.iter().zip(&events.cars) {
            if happened.touched_ball && code(other.team) != holder {
                holder = code(other.team);
                count = 0.0;
            }
        }
        let distance = |other: usize| cur.cars[other].body.pos.distance(cur.ball.pos);
        let mut mine = 0.0;
        // The source's cars one after the other, as it was asked for each in turn.
        for (index, other) in cur.cars.iter().enumerate() {
            let near = distance(index);
            let holds = code(other.team) == holder
                && near < self.min_dist
                && opponents(cur, index).all(|them| distance(them) >= near);
            if holds {
                count += 1.0;
                if index == car {
                    mine = count / 10.0;
                }
            }
        }
        memory.copy_from_slice(&[holder, count]);
        mine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// The ball at the centre; Blue at `blue` uu from it, Orange 1000 uu away on its side.
    fn field(blue: f32) -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -blue, 93.15)),
                (Team::Orange, Vec3A::new(0.0, 1000.0, 93.15)),
            ],
        )
    }

    fn step(
        reward: &Possession,
        g: &GameSnapshot,
        events: &StepEvents,
        memory: &mut [Vec<f32>],
    ) -> [f32; 2] {
        [0, 1].map(|car| reward.value_remembering(g, g, events, car, &mut memory[car]))
    }

    #[test]
    fn pays_more_the_longer_the_team_keeps_it() {
        let reward: Possession = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut memory = vec![vec![0.0; reward.memory()]; 2];
        let near = field(150.0);
        assert_eq!(
            step(&reward, &near, &quiet(2), &mut memory),
            [0.0, 0.0],
            "nobody touched it"
        );
        let paid = step(&reward, &near, &touched(quiet(2), 0), &mut memory);
        assert!((paid[0] - 0.1).abs() < 1e-6 && paid[1] == 0.0, "{paid:?}");
        let paid = step(&reward, &near, &quiet(2), &mut memory);
        assert!((paid[0] - 0.2).abs() < 1e-6, "{paid:?}");
        let far = field(300.0);
        assert_eq!(
            step(&reward, &far, &quiet(2), &mut memory),
            [0.0, 0.0],
            "too far"
        );
        let paid = step(&reward, &near, &quiet(2), &mut memory);
        assert!((paid[0] - 0.3).abs() < 1e-6, "the count kept: {paid:?}");
        let mut beaten = near.clone();
        beaten.cars[1].body.pos.y = 100.0;
        assert_eq!(
            step(&reward, &beaten, &quiet(2), &mut memory),
            [0.0, 0.0],
            "opponent nearer"
        );

        let mut orange = near.clone();
        orange.cars[0].body.pos.y = -1000.0;
        orange.cars[1].body.pos.y = 150.0;
        let paid = step(&reward, &orange, &touched(quiet(2), 1), &mut memory);
        assert!(
            paid[0] == 0.0 && (paid[1] - 0.1).abs() < 1e-6,
            "Orange alike, count anew: {paid:?}"
        );

        let mut fresh = vec![vec![0.0; 2]; 2];
        assert_eq!(
            step(&reward, &near, &quiet(2), &mut fresh),
            [0.0, 0.0],
            "a new episode starts again"
        );
    }

    #[test]
    fn teammates_share_one_count_in_car_order() {
        let reward = Possession { min_dist: 200.0 };
        let g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -150.0, 93.15)),
                (Team::Blue, Vec3A::new(0.0, -180.0, 93.15)),
                (Team::Orange, Vec3A::new(0.0, 1000.0, 93.15)),
            ],
        );
        let mut memory = vec![vec![0.0; 2]; 3];
        let events = touched(quiet(3), 0);
        let paid: Vec<f32> = (0..3)
            .map(|car| reward.value_remembering(&g, &g, &events, car, &mut memory[car]))
            .collect();
        assert!(
            (paid[0] - 0.1).abs() < 1e-6 && (paid[1] - 0.2).abs() < 1e-6 && paid[2] == 0.0,
            "{paid:?}"
        );
    }
}
