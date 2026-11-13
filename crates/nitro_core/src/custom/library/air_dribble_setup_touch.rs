//! Air touches that push the ball upfield, each touch of an aerial worth more than the one before.

use crate::custom::prelude::*;

/// On a touch in the air with the ball above 250 uu, above the car and farther upfield than it:
/// (1 − |ball x| / 2200) × the number of such touches since the car left the ground (negative when the
/// ball is more than 2200 uu off centre). 0 otherwise; the count starts again on the ground.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirDribbleSetupTouch {}

impl Reward for AirDribbleSetupTouch {
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
        // memory[0]: this car's touches since it left the ground.
        let me = &cur.cars[car];
        let (ball, pos) = (cur.ball.pos, me.body.pos);
        let upfield = as_blue(me.team, ball).y > as_blue(me.team, pos).y;
        let mut reward = 0.0;
        if events.cars[car].touched_ball
            && ball.z > 250.0
            && ball.z > pos.z
            && upfield
            && !me.on_ground
        {
            memory[0] += 1.0;
            reward = (1.0 - (ball.x / 2200.0).abs()) * memory[0];
        }
        if me.on_ground {
            memory[0] = 0.0;
        }
        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// The ball 1100 uu off centre, above and upfield of a Blue and an Orange car, both in the air.
    fn setup() -> GameSnapshot {
        let mut g = game(
            Vec3A::new(1100.0, 0.0, 800.0),
            &[
                (Team::Blue, Vec3A::new(1100.0, -100.0, 650.0)),
                (Team::Orange, Vec3A::new(1100.0, 100.0, 650.0)),
            ],
        );
        for car in &mut g.cars {
            car.on_ground = false;
        }
        g
    }

    #[test]
    fn each_touch_of_an_aerial_counts_more() {
        let reward: AirDribbleSetupTouch = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = setup();
        let both = touched(touched(quiet(2), 0), 1);
        let mut memory = [[0.0f32; 1]; 2];
        let mut step = |cur: &GameSnapshot, events: &StepEvents, car: usize| {
            reward.value_remembering(cur, cur, events, car, &mut memory[car])
        };
        assert_eq!(step(&g, &both, 0), 0.5);
        assert_eq!(step(&g, &both, 1), 0.5, "Orange alike");
        assert_eq!(step(&g, &quiet(2), 0), 0.0, "no touch");
        assert_eq!(step(&g, &both, 0), 1.0, "the second touch counts twice");
        assert_eq!(step(&g, &touched(quiet(2), 0), 0), 1.5);
        assert_eq!(step(&g, &both, 1), 1.0, "Orange's own count");
        let mut landed = g.clone();
        landed.cars[0].on_ground = true;
        assert_eq!(step(&landed, &both, 0), 0.0, "on the ground");
        assert_eq!(step(&g, &both, 0), 0.5, "a new aerial counts from 1");
    }

    #[test]
    fn needs_the_ball_upfield_above_and_high() {
        let reward: AirDribbleSetupTouch = serde_json::from_value(serde_json::json!({})).unwrap();
        let both = touched(touched(quiet(2), 0), 1);
        let mut behind = setup();
        behind.cars[0].body.pos.y = 100.0;
        behind.cars[1].body.pos.y = -100.0;
        assert_eq!(
            reward.value_remembering(&behind, &behind, &both, 0, &mut [0.0]),
            0.0
        );
        assert_eq!(
            reward.value_remembering(&behind, &behind, &both, 1, &mut [0.0]),
            0.0
        );
        let mut low = setup();
        low.ball.pos.z = 240.0;
        low.cars[0].body.pos.z = 100.0;
        assert_eq!(
            reward.value_remembering(&low, &low, &both, 0, &mut [0.0]),
            0.0
        );
    }
}
