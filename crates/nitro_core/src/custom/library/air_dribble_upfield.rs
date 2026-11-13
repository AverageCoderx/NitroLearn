//! Air dribbling upfield: close under the ball in the opponent's half and facing it, more with boost and
//! touches.

use crate::custom::prelude::*;

/// Every step in the air under a ball above 250 uu in the opponent's half, within 400 uu and facing it
/// (cosine above 0.74): up to 53.5 with more than 15 boost (40 for a touch, 5 for being behind the ball),
/// half as much with less; halved when the car or ball is not rising; then × 2 × (1 − |car x| / 4096).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirDribbleUpfield {}

impl Reward for AirDribbleUpfield {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let me = &cur.cars[car];
        let (ball, pos) = (
            as_blue(me.team, cur.ball.pos),
            as_blue(me.team, me.body.pos),
        );
        let distance = ball.distance(pos);
        let facing = me
            .body
            .forward()
            .dot((cur.ball.pos - me.body.pos).normalize_or_zero());
        let mut reward = 0.0;
        if !me.on_ground && ball.z > 250.0 && pos.z < ball.z && ball.y > 0.0 {
            let behind = pos.y < ball.y;
            let touched = events.cars[car].touched_ball;
            let near = distance < 400.0 && facing > 0.74;
            // Two cases: with more than 15 boost, and half of it all without.
            let full = if near && me.boost > 15.0 {
                Some(1.0)
            } else if near {
                Some(0.5)
            } else {
                None
            };
            if let Some(share) = full {
                if behind {
                    if touched {
                        reward += 40.0 * share;
                    }
                    reward += 5.0 * share;
                }
                reward += facing * share;
                reward += (1.0 - distance / 400.0) * 7.5 * share;
            }
            let rising =
                me.body.pos.z > prev.cars[car].body.pos.z && cur.ball.pos.z > prev.ball.pos.z;
            if !rising {
                reward /= 2.0;
            }
        }
        reward * (1.0 - (me.body.pos.x / SIDE_WALL_X).abs()) * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// A Blue car facing upfield 200 uu behind and 150 uu below a ball at y = 2000 (250 uu away, facing it
    /// at a cosine of 0.8), both 30 uu higher than on the step before (or lower); an Orange car far off.
    fn dribble(rising: bool) -> (GameSnapshot, GameSnapshot) {
        let mut cur = game(
            Vec3A::new(0.0, 2000.0, 800.0),
            &[
                (Team::Blue, Vec3A::new(0.0, 1800.0, 650.0)),
                (Team::Orange, Vec3A::new(0.0, 200.0, 650.0)),
            ],
        );
        cur.cars[0].body.rot = Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2);
        cur.cars[1].body.rot = Mat3A::from_rotation_z(-std::f32::consts::FRAC_PI_2);
        for car in &mut cur.cars {
            car.on_ground = false;
            car.boost = 50.0;
        }
        let mut prev = cur.clone();
        let climb = if rising { 30.0 } else { -30.0 };
        prev.ball.pos.z -= climb;
        for car in &mut prev.cars {
            car.body.pos.z -= climb;
        }
        (prev, cur)
    }

    /// The sum with more than 15 boost, behind the ball, at x = 0: (5 + facing + closeness) × 2.
    const FULL: f32 = (5.0 + 0.8 + (1.0 - 250.0 / 400.0) * 7.5) * 2.0;

    #[test]
    fn pays_close_facing_rising_with_more_for_boost_and_touches() {
        let reward: AirDribbleUpfield = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, cur) = dribble(true);
        assert!((reward.value(&prev, &cur, &quiet(2), 0) - FULL).abs() < 1e-4);
        let touch = reward.value(&prev, &cur, &touched(quiet(2), 0), 0);
        assert!(
            (touch - (FULL + 80.0)).abs() < 1e-4,
            "40 more for a touch, doubled"
        );
        let mut low = cur.clone();
        low.cars[0].boost = 10.0;
        let half = reward.value(&prev, &low, &quiet(2), 0);
        assert!((half - FULL / 2.0).abs() < 1e-4, "half without boost");
        let (fall_prev, fall) = dribble(false);
        let falling = reward.value(&fall_prev, &fall, &quiet(2), 0);
        assert!((falling - FULL / 2.0).abs() < 1e-4, "half when not rising");
        let mut aside = cur.clone();
        aside.ball.pos.x = 2048.0;
        aside.cars[0].body.pos.x = 2048.0;
        let mut aside_prev = prev.clone();
        aside_prev.ball.pos.x = 2048.0;
        aside_prev.cars[0].body.pos.x = 2048.0;
        let off_centre = reward.value(&aside_prev, &aside, &quiet(2), 0);
        assert!(
            (off_centre - FULL / 2.0).abs() < 1e-4,
            "half way to the side wall"
        );
    }

    #[test]
    fn orange_alike_and_nothing_outside_the_gate() {
        let reward: AirDribbleUpfield = serde_json::from_value(serde_json::json!({})).unwrap();
        let (prev, cur) = dribble(true);
        let mut mirrored = cur.clone();
        mirrored.ball.pos = Vec3A::new(0.0, -2000.0, 800.0);
        mirrored.cars[1].body.pos = Vec3A::new(0.0, -1800.0, 650.0);
        let mut mirrored_prev = prev.clone();
        mirrored_prev.ball.pos = Vec3A::new(0.0, -2000.0, 770.0);
        mirrored_prev.cars[1].body.pos = Vec3A::new(0.0, -1800.0, 620.0);
        let orange = reward.value(&mirrored_prev, &mirrored, &quiet(2), 1);
        assert!((orange - FULL).abs() < 1e-4, "Orange alike");
        assert_eq!(
            reward.value(&prev, &cur, &quiet(2), 1),
            0.0,
            "the ball in Orange's own half"
        );
        let mut grounded = cur.clone();
        grounded.cars[0].on_ground = true;
        assert_eq!(reward.value(&prev, &grounded, &quiet(2), 0), 0.0);
        let mut own_half = cur.clone();
        own_half.ball.pos.y = -100.0;
        own_half.cars[0].body.pos.y = -300.0;
        assert_eq!(reward.value(&prev, &own_half, &quiet(2), 0), 0.0);
        let mut far = cur.clone();
        far.cars[0].body.pos.y = 1500.0;
        assert_eq!(reward.value(&prev, &far, &quiet(2), 0), 0.0, "500 uu away");
    }
}
