//! Shaping toward the musty's pose before its dodge (musty_pose_shaping) and toward the Breezi's tornado
//! spin (tornado_shaping).
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::flick_kit::*;

/// Each step in the air with a dodge left and the ball close: gamma x the potential now minus before, the
/// potential being how far the nose is down toward the ball, the roof facing it and the ball close (0-1).
/// Kept at a dodge musty_flick arms on, paid back at any other; at most 1 per dodge.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MustyPoseShaping {
    /// No potential with the ball this far or further (uu). Default 450.
    #[serde(default = "pd::max_dist")]
    pub max_dist: f32,
    /// Full potential with the ball this close or closer (uu). Default 250.
    #[serde(default = "pd::full_dist")]
    pub full_dist: f32,
    /// No potential with the ball centre lower than this below the car's (uu). Default -100.
    #[serde(default = "pd::min_ball_above")]
    pub min_ball_above: f32,
    /// Full potential with the nose this far below horizontal toward the ball (degrees). Default 100.
    #[serde(default = "pd::nose_full_deg")]
    pub nose_full_deg: f32,
    /// The training's discount per step (0.995 suits tick skip 4). Default 0.995.
    #[serde(default = "pd::gamma")]
    pub gamma: f32,
    /// The dodge must be a backflip or back-diagonal: the flip's torque y (car frame) at most this (backflip -1, back-diagonal -0.71). Default -0.4.
    #[serde(default = "pd::arm_max_torque_y")]
    pub arm_max_torque_y: f32,
    /// At the dodge the nose is at least this far below horizontal, measured toward the ball (90 = straight down). Default 88.
    #[serde(default = "pd::arm_min_nose_down_deg")]
    pub arm_min_nose_down_deg: f32,
    /// ... and at most this far (past 140 it is a flat roof ram). Default 140.
    #[serde(default = "pd::arm_max_nose_down_deg")]
    pub arm_max_nose_down_deg: f32,
    /// Car centre to ball centre at the dodge, at most (uu). Default 350.
    #[serde(default = "pd::arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Ball centre height above the car centre at the dodge, at least (uu). Default -100.
    #[serde(default = "pd::arm_min_ball_above")]
    pub arm_min_ball_above: f32,
    /// Roof facing the ball at the dodge: roof direction . direction to the ball, at least. Default -0.35.
    #[serde(default = "pd::arm_min_up_dot_ball")]
    pub arm_min_up_dot_ball: f32,
    /// |right . direction to the ball| at the dodge, at most (1 = no limit). Default 1.
    #[serde(default = "pd::arm_max_lateral")]
    pub arm_max_lateral: f32,
}

/// As musty_pose_shaping, the potential being the tornado turn since a pop (|roll| + |yaw| since the
/// lift-off, the two the same way) over full_deg, while the ball is close and in the air. At most 1 per
/// dodge, kept only at a dodge musty_flick arms on.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TornadoShaping {
    /// Full potential at this much tornado turn (|roll| + |yaw|, degrees). Default 290.
    #[serde(default = "td::full_deg")]
    pub full_deg: f32,
    /// No potential with the ball this far or further (uu). Default 700.
    #[serde(default = "td::max_dist")]
    pub max_dist: f32,
    /// No potential with the ball lower than this (uu). Default 140.
    #[serde(default = "td::min_ball_z")]
    pub min_ball_z: f32,
    /// No potential with the ball centre lower than this below the car's (uu). Default -150.
    #[serde(default = "td::min_ball_above")]
    pub min_ball_above: f32,
    /// The training's discount per step (0.995 suits tick skip 4). Default 0.995.
    #[serde(default = "td::gamma")]
    pub gamma: f32,
    /// The ball was under control on the roof this many seconds or less before the lift-off. Default 0.25.
    #[serde(default = "td::pop_window")]
    pub pop_window: f32,
    /// A car whose wheels or body touch something within this of an arena surface is on it (uu). Default 150.
    #[serde(default = "td::surface_margin")]
    pub surface_margin: f32,
    /// Ball under control on the roof: its centre at least this far above the car along its roof direction (uu). Default 105.
    #[serde(default = "td::dribble_min_up")]
    pub dribble_min_up: f32,
    /// ... and at most this far (uu). Default 185.
    #[serde(default = "td::dribble_max_up")]
    pub dribble_max_up: f32,
    /// ... best this far forward of the car's centre (uu). Default 20.
    #[serde(default = "td::dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// ... no control this far or further from that spot, sideways (uu). Default 90.
    #[serde(default = "td::dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// ... nor at this speed relative to the car (uu/s). Default 600.
    #[serde(default = "td::dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality (0-1) that counts as the ball under control. Default 0.25.
    #[serde(default = "td::control_threshold")]
    pub control_threshold: f32,
    /// The dodge must be a backflip or back-diagonal: the flip's torque y (car frame) at most this (backflip -1, back-diagonal -0.71). Default -0.4.
    #[serde(default = "td::arm_max_torque_y")]
    pub arm_max_torque_y: f32,
    /// At the dodge the nose is at least this far below horizontal, measured toward the ball (90 = straight down). Default 88.
    #[serde(default = "td::arm_min_nose_down_deg")]
    pub arm_min_nose_down_deg: f32,
    /// ... and at most this far (past 140 it is a flat roof ram). Default 140.
    #[serde(default = "td::arm_max_nose_down_deg")]
    pub arm_max_nose_down_deg: f32,
    /// Car centre to ball centre at the dodge, at most (uu). Default 350.
    #[serde(default = "td::arm_max_ball_dist")]
    pub arm_max_ball_dist: f32,
    /// Ball centre height above the car centre at the dodge, at least (uu). Default -100.
    #[serde(default = "td::arm_min_ball_above")]
    pub arm_min_ball_above: f32,
    /// Roof facing the ball at the dodge: roof direction . direction to the ball, at least. Default -0.35.
    #[serde(default = "td::arm_min_up_dot_ball")]
    pub arm_min_up_dot_ball: f32,
    /// |right . direction to the ball| at the dodge, at most (1 = no limit). Default 1.
    #[serde(default = "td::arm_max_lateral")]
    pub arm_max_lateral: f32,
}

/// MustyPoseShaping's defaults.
mod pd {
    pub fn max_dist() -> f32 {
        450.0
    }
    pub fn full_dist() -> f32 {
        250.0
    }
    pub fn min_ball_above() -> f32 {
        -100.0
    }
    pub fn nose_full_deg() -> f32 {
        100.0
    }
    pub fn gamma() -> f32 {
        0.995
    }
    pub fn arm_max_torque_y() -> f32 {
        -0.4
    }
    pub fn arm_min_nose_down_deg() -> f32 {
        88.0
    }
    pub fn arm_max_nose_down_deg() -> f32 {
        140.0
    }
    pub fn arm_max_ball_dist() -> f32 {
        350.0
    }
    pub fn arm_min_ball_above() -> f32 {
        -100.0
    }
    pub fn arm_min_up_dot_ball() -> f32 {
        -0.35
    }
    pub fn arm_max_lateral() -> f32 {
        1.0
    }
}

/// TornadoShaping's defaults.
mod td {
    pub fn full_deg() -> f32 {
        290.0
    }
    pub fn max_dist() -> f32 {
        700.0
    }
    pub fn min_ball_z() -> f32 {
        140.0
    }
    pub fn min_ball_above() -> f32 {
        -150.0
    }
    pub fn gamma() -> f32 {
        0.995
    }
    pub fn pop_window() -> f32 {
        0.25
    }
    pub fn surface_margin() -> f32 {
        150.0
    }
    pub fn dribble_min_up() -> f32 {
        105.0
    }
    pub fn dribble_max_up() -> f32 {
        185.0
    }
    pub fn dribble_target_forward() -> f32 {
        20.0
    }
    pub fn dribble_max_planar_dist() -> f32 {
        90.0
    }
    pub fn dribble_max_rel_speed() -> f32 {
        600.0
    }
    pub fn control_threshold() -> f32 {
        0.25
    }
    pub fn arm_max_torque_y() -> f32 {
        -0.4
    }
    pub fn arm_min_nose_down_deg() -> f32 {
        88.0
    }
    pub fn arm_max_nose_down_deg() -> f32 {
        140.0
    }
    pub fn arm_max_ball_dist() -> f32 {
        350.0
    }
    pub fn arm_min_ball_above() -> f32 {
        -100.0
    }
    pub fn arm_min_up_dot_ball() -> f32 {
        -0.35
    }
    pub fn arm_max_lateral() -> f32 {
        1.0
    }
}

// MustyPoseShaping's memory, per car.
const INIT: usize = 0;
const PHI: usize = 1;

// TornadoShaping's memory, per car (after INIT and PHI).
const ROLL: usize = 2;
const YAW: usize = 3;
const AIRBORNE: usize = 4;
const POPPED: usize = 5;
const EVER_CONTROL: usize = 6;
const CONTROL_AGE: usize = 7;

impl MustyPoseShaping {
    fn arm_test(&self) -> ArmTest {
        ArmTest {
            max_torque_y: self.arm_max_torque_y,
            min_nose_down_deg: self.arm_min_nose_down_deg,
            max_nose_down_deg: self.arm_max_nose_down_deg,
            max_ball_dist: self.arm_max_ball_dist,
            min_ball_above: self.arm_min_ball_above,
            min_up_dot_ball: self.arm_min_up_dot_ball,
            max_lateral: self.arm_max_lateral,
        }
    }

    fn phi(&self, car: &CarSnapshot, detail: &CarDetail, ball: Vec3A) -> f32 {
        if !has_dodge(car, detail) || car.is_demoed {
            return 0.0;
        }
        let to = ball - car.body.pos;
        let d = to.length();
        if d > self.max_dist || to.z < self.min_ball_above || d < 1.0 {
            return 0.0;
        }
        let nose = nose_down_deg(car.body.rot, to);
        ramp(nose, 0.0, self.nose_full_deg)
            * ramp(car.body.rot.z_axis.dot(to / d), -0.2, 0.6)
            * ramp(d, self.max_dist, self.full_dist)
    }
}

impl Reward for MustyPoseShaping {
    fn memory(&self) -> usize {
        2
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        if m[INIT] == 0.0 {
            m[INIT] = 1.0;
            m[PHI] = self.phi(&prev.cars[car], &events.prev_detail(car), prev.ball.pos);
        }
        let me = &cur.cars[car];
        let detail = events.detail(car);
        if me.has_flipped && !prev.cars[car].has_flipped {
            // Bank the potential if musty_flick arms on this dodge, else pay it back
            let touched = events.cars.get(car).is_some_and(|e| e.touched_ball);
            let armed = arm_pose_ok(self.arm_test(), me, &detail, touched, cur.ball.pos);
            let paid = if armed { 0.0 } else { -m[PHI] };
            m[PHI] = 0.0;
            return paid;
        }
        if cur.goal_scored.is_some() {
            // The episode's end: its potential is 0
            let paid = -m[PHI];
            m[PHI] = 0.0;
            return paid;
        }
        let phi = self.phi(me, &detail, cur.ball.pos);
        let paid = self.gamma * phi - m[PHI];
        m[PHI] = phi;
        paid
    }
}

impl TornadoShaping {
    fn arm_test(&self) -> ArmTest {
        ArmTest {
            max_torque_y: self.arm_max_torque_y,
            min_nose_down_deg: self.arm_min_nose_down_deg,
            max_nose_down_deg: self.arm_max_nose_down_deg,
            max_ball_dist: self.arm_max_ball_dist,
            min_ball_above: self.arm_min_ball_above,
            min_up_dot_ball: self.arm_min_up_dot_ball,
            max_lateral: self.arm_max_lateral,
        }
    }

    fn dribble(&self) -> Dribble {
        Dribble {
            min_up: self.dribble_min_up,
            max_up: self.dribble_max_up,
            target_forward: self.dribble_target_forward,
            max_planar_dist: self.dribble_max_planar_dist,
            max_rel_speed: self.dribble_max_rel_speed,
        }
    }

    fn phi(&self, m: &[f32], car: &CarSnapshot, detail: &CarDetail, ball: Vec3A) -> f32 {
        let (roll, yaw) = (m[ROLL], m[YAW]);
        if m[POPPED] == 0.0 || !has_dodge(car, detail) || car.is_demoed || roll * yaw <= 0.0 {
            return 0.0;
        }
        let to = ball - car.body.pos;
        if to.length() > self.max_dist || ball.z < self.min_ball_z || to.z < self.min_ball_above {
            return 0.0;
        }
        clamp01(div(roll.abs() + yaw.abs(), self.full_deg))
    }
}

impl Reward for TornadoShaping {
    fn memory(&self) -> usize {
        8
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        let before = &prev.cars[car];
        if m[INIT] == 0.0 {
            m[INIT] = 1.0;
            let on = supported(before, &events.prev_detail(car), self.surface_margin);
            m[AIRBORNE] = flag(!on);
        }
        m[CONTROL_AGE] += step_ticks(events);
        let me = &cur.cars[car];
        let detail = events.detail(car);
        if me.has_flipped && !before.has_flipped {
            let touched = events.cars.get(car).is_some_and(|e| e.touched_ball);
            let armed = arm_pose_ok(self.arm_test(), me, &detail, touched, cur.ball.pos);
            let paid = if armed { 0.0 } else { -m[PHI] };
            m[PHI] = 0.0;
            return paid;
        }
        if control_quality(me, &cur.ball, self.dribble()) >= self.control_threshold {
            m[CONTROL_AGE] = 0.0;
            m[EVER_CONTROL] = 1.0;
        }
        if supported(me, &detail, self.surface_margin) {
            m[ROLL] = 0.0;
            m[YAW] = 0.0;
            m[AIRBORNE] = 0.0;
            m[POPPED] = 0.0;
        } else {
            if m[AIRBORNE] == 0.0 {
                m[AIRBORNE] = 1.0;
                m[POPPED] = flag(
                    m[EVER_CONTROL] != 0.0 && m[CONTROL_AGE] <= secs_to_ticks(self.pop_window),
                );
            }
            let w = rotation_between(before.body.rot, me.body.rot);
            m[ROLL] += degrees(w.dot(before.body.rot.x_axis));
            m[YAW] += degrees(w.dot(before.body.rot.z_axis));
        }
        if cur.goal_scored.is_some() {
            let paid = -m[PHI];
            m[PHI] = 0.0;
            return paid;
        }
        let phi = self.phi(m, me, &detail, cur.ball.pos);
        let paid = self.gamma * phi - m[PHI];
        m[PHI] = phi;
        paid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn level() -> Mat3A {
        Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z)
    }

    fn nose_down(rot: Mat3A, deg: f32) -> Mat3A {
        turned(rot, rot.y_axis * deg.to_radians())
    }

    const CAR: Vec3A = Vec3A::new(0.0, 0.0, 300.0);

    fn in_the_air(rot: Mat3A, flipped: bool) -> GameSnapshot {
        let mut g = game(CAR + Vec3A::new(0.0, 150.0, 100.0), &[(Team::Blue, CAR)]);
        let me = &mut g.cars[0];
        me.body.rot = rot;
        me.on_ground = false;
        me.has_jumped = true;
        me.has_flipped = flipped;
        g
    }

    fn run(reward: &dyn Reward, games: &[GameSnapshot], torque_y: f32) -> Vec<f32> {
        let mut events = quiet(1);
        events.details[0].flip_rel_torque = Vec3A::new(0.0, torque_y, 0.0);
        events.details[0].air_time_since_jump = 0.3;
        let mut memory = vec![0.0; reward.memory()];
        (1..games.len())
            .map(|i| reward.value_remembering(&games[i - 1], &games[i], &events, 0, &mut memory))
            .collect()
    }

    #[test]
    fn the_pose_is_kept_at_a_musty_dodge_and_paid_back_at_another() {
        let reward: MustyPoseShaping =
            serde_json::from_value(serde_json::json!({})).expect("defaults");
        let games: Vec<GameSnapshot> = [30.0, 60.0, 90.0, 100.0]
            .iter()
            .map(|&d| in_the_air(nose_down(level(), d), false))
            .chain([in_the_air(nose_down(level(), 100.0), true)])
            .collect();
        let backflip = run(&reward, &games, -1.0);
        assert!(
            backflip[..3].iter().all(|&r| r > -0.01) && backflip[3] == 0.0,
            "{backflip:?}"
        );
        assert!(backflip.iter().sum::<f32>() > 0.3, "{backflip:?}");
        let front_flip = run(&reward, &games, 1.0);
        assert!(front_flip.iter().sum::<f32>() <= 0.0, "{front_flip:?}");
        assert!(
            front_flip[3] < -0.3,
            "paid back at the dodge: {front_flip:?}"
        );
    }

    #[test]
    fn the_tornado_after_a_pop_is_kept_at_a_musty_dodge() {
        let reward: TornadoShaping =
            serde_json::from_value(serde_json::json!({})).expect("defaults");
        let mut dribble = game(
            Vec3A::new(0.0, 20.0, 150.0),
            &[(Team::Blue, Vec3A::new(0.0, 0.0, 17.0))],
        );
        dribble.cars[0].body.rot = level();
        let axis = (Vec3A::Y + Vec3A::Z).normalize();
        let mut games = vec![dribble.clone(), dribble];
        for k in 0..=8 {
            games.push(in_the_air(
                turned(level(), axis * (45.0 * k as f32).to_radians()),
                false,
            ));
        }
        games.push(in_the_air(nose_down(level(), 100.0), true));
        let musty = run(&reward, &games, -1.0);
        assert!(musty.iter().sum::<f32>() > 0.5, "{musty:?}");
        let other = run(&reward, &games, 1.0);
        assert!(other.iter().sum::<f32>() <= 0.0, "{other:?}");
        // No pop: no potential
        let mut no_pop = games.clone();
        no_pop[0].ball.pos.z = 400.0;
        no_pop[1].ball.pos.z = 400.0;
        assert!(run(&reward, &no_pop, -1.0).iter().all(|&r| r == 0.0));
    }
}
