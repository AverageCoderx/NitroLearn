//! Air roll: how fast the car spins about its own nose, anywhere (air_roll) or on kickoffs
//! (air_roll_on_kickoff).

use crate::custom::prelude::*;

/// The car's angular velocity in its own axes: x = roll (about the nose), y = pitch, z = yaw (rad/s).
fn spin(body: &Body) -> Vec3A {
    body.rot.transpose() * body.ang_vel
}

/// Every step: how fast the car air rolls (rad/s, 0 to 5.5) while above 200 uu. With `in_air_with_flip`,
/// while in the air with a flip left instead, and a fifth of its pitching and yawing taken off (−2.2 to 5.5).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirRoll {
    /// false: |roll| whenever the car is higher than 200 uu; true: |roll| − |pitch| / 5 − |yaw| / 5 while
    /// the car is in the air and can still flip.
    #[serde(default)]
    pub in_air_with_flip: bool,
}

impl Reward for AirRoll {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        let spin = spin(&me.body);
        if !self.in_air_with_flip {
            if me.body.pos.z > 200.0 {
                spin.x.abs()
            } else {
                0.0
            }
        } else if me.can_flip && !me.on_ground {
            spin.x.abs() - spin.y.abs() / 5.0 - spin.z.abs() / 5.0
        } else {
            0.0
        }
    }
}

/// Every step of a kickoff (the ball on the centre spot): the car's roll speed over 5.5 rad/s, −1 to 1,
/// positive rolling one way and negative rolling the other. 0 once the ball has moved.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirRollOnKickoff {}

impl Reward for AirRollOnKickoff {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        // A kickoff: the ball at x = y = 0 exactly.
        if cur.ball.pos.x == 0.0 && cur.ball.pos.y == 0.0 {
            spin(&cur.cars[car].body).x / CAR_MAX_ANG_VEL
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// A Blue car facing +y and an Orange car facing −y, both 300 uu up, spinning `world` (rad/s) each.
    fn spinning(world_blue: Vec3A, world_orange: Vec3A) -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 300.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 300.0)),
            ],
        );
        g.cars[0].body.rot = Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2);
        g.cars[1].body.rot = Mat3A::from_rotation_z(-std::f32::consts::FRAC_PI_2);
        g.cars[0].body.ang_vel = world_blue;
        g.cars[1].body.ang_vel = world_orange;
        for car in &mut g.cars {
            car.on_ground = false;
        }
        g
    }

    #[test]
    fn air_roll_is_spin_about_the_cars_nose() {
        let reward: AirRoll = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = spinning(Vec3A::new(0.0, 4.0, 0.0), Vec3A::new(0.0, -4.0, 0.0));
        assert!((reward.value(&g, &g, &quiet(2), 0) - 4.0).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 4.0).abs() < 1e-5,
            "Orange alike"
        );
        let pitching = spinning(Vec3A::new(4.0, 0.0, 0.0), Vec3A::new(4.0, 0.0, 0.0));
        assert!(
            reward.value(&pitching, &pitching, &quiet(2), 0).abs() < 1e-5,
            "pitch is not roll"
        );
        let mut low = g.clone();
        low.cars[0].body.pos.z = 150.0;
        assert_eq!(reward.value(&low, &low, &quiet(2), 0), 0.0, "below 200 uu");
    }

    #[test]
    fn in_air_with_flip_takes_off_pitch_and_yaw_and_needs_a_flip() {
        let reward = AirRoll {
            in_air_with_flip: true,
        };
        let g = spinning(Vec3A::new(1.0, 4.0, 2.0), Vec3A::new(-1.0, -4.0, 2.0));
        let expected = 4.0 - 1.0 / 5.0 - 2.0 / 5.0;
        assert!((reward.value(&g, &g, &quiet(2), 0) - expected).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - expected).abs() < 1e-5,
            "Orange alike"
        );
        let mut no_flip = g.clone();
        no_flip.cars[0].can_flip = false;
        assert_eq!(reward.value(&no_flip, &no_flip, &quiet(2), 0), 0.0);
        let mut grounded = g.clone();
        grounded.cars[0].on_ground = true;
        assert_eq!(reward.value(&grounded, &grounded, &quiet(2), 0), 0.0);
    }

    #[test]
    fn air_roll_on_kickoff_is_signed_and_the_same_for_both_teams() {
        let reward: AirRollOnKickoff = serde_json::from_value(serde_json::json!({})).unwrap();
        // The same roll as each driver sees it: Orange's world spin is Blue's turned half a turn.
        let g = spinning(Vec3A::new(0.0, 2.75, 0.0), Vec3A::new(0.0, -2.75, 0.0));
        assert!((reward.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-5);
        assert!(
            (reward.value(&g, &g, &quiet(2), 1) - 0.5).abs() < 1e-5,
            "Orange alike"
        );
        let other_way = spinning(Vec3A::new(0.0, -2.75, 0.0), Vec3A::ZERO);
        assert!((reward.value(&other_way, &other_way, &quiet(2), 0) + 0.5).abs() < 1e-5);
        let mut played = g.clone();
        played.ball.pos.x = 10.0;
        assert_eq!(
            reward.value(&played, &played, &quiet(2), 0),
            0.0,
            "not a kickoff"
        );
    }
}
