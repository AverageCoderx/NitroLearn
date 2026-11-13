//! The wizard flick: a late forward-diagonal dodge flick off the roof that rotates onto the ball and meets
//! it with a bottom front corner, underside up.
//! group: moves

use crate::custom::prelude::*;

// Octane hitbox in the car's own frame (RocketSim CarConfig): x forward, z up, from the car's centre.
const HB_X0: f32 = 13.875_66 - 60.2535;
const HB_X1: f32 = 13.875_66 + 60.2535;
const HB_Y: f32 = 43.3497;
const HB_Z0: f32 = 20.755 - 19.329_55;
/// The ball's upward speed must rise this much (uu/s) from the step before the touch to the step after.
const MIN_POP: f32 = 200.0;
/// The car must have been off the floor this long before the touch (s).
const MIN_CLEAR_OF_FLOOR: f32 = 0.25;

/// Memory slots.
const ARMED: usize = 0; // a forward-diagonal dodge seen, no touch since
const FLIP_SEEN: usize = 1; // this airtime's dodge was looked at
const BALL_FLOORED: usize = 2; // the ball has been on the floor this episode
const CLEAR: usize = 3; // seconds since the car was on or at the floor
const PENDING: usize = 4; // the touch passed, waiting for the pop
const VZ_BEFORE: usize = 5;

/// The ball's centre in the car's frame (forward, side, up).
fn local(car: &Body, p: Vec3A) -> Vec3A {
    let d = p - car.pos;
    Vec3A::new(
        car.rot.x_axis.dot(d),
        car.rot.y_axis.dot(d),
        car.rot.z_axis.dot(d),
    )
}

fn lowest_corner_z(car: &Body) -> f32 {
    let mut low = f32::MAX;
    for x in [HB_X0, HB_X1] {
        for y in [-HB_Y, HB_Y] {
            for z in [HB_Z0, HB_Z0 + 2.0 * 19.329_55] {
                let p = car.pos + car.rot.x_axis * x + car.rot.y_axis * y + car.rot.z_axis * z;
                low = low.min(p.z);
            }
        }
    }
    low
}

/// The hitbox point nearest the ball's centre is a bottom front corner (left or right).
fn bottom_front_corner(car: &Body, ball: Vec3A) -> bool {
    let b = local(car, ball);
    let c = Vec3A::new(
        b.x.clamp(HB_X0, HB_X1),
        b.y.clamp(-HB_Y, HB_Y),
        b.z.clamp(HB_Z0, HB_Z0 + 38.6591),
    );
    c.z < HB_Z0 + 12.0 && c.x > HB_X1 - 15.0 && c.y.abs() > HB_Y - 12.0
}

/// A dodge 10° to 80° off straight forward (RocketSim's flip torque: y forward, -x to the right).
fn forward_diagonal(torque: Vec3A) -> bool {
    let angle = (-torque.x).abs().atan2(torque.y).to_degrees();
    torque.y > 0.0 && (10.0..=80.0).contains(&angle)
}

/// 1 for a wizard flick: the first touch after a forward-diagonal dodge hits the ball with a bottom front
/// corner, the car's roof tilted past 90° and its nose above level, the car well off the floor for 0.25 s
/// and the ball never on the floor this game, and the ball's upward speed rises 200+ uu/s. Else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WizardFlick {}

impl Reward for WizardFlick {
    fn memory(&self) -> usize {
        6
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        let me = &cur.cars[car];
        let detail = events.detail(car);
        let at_floor = me.on_ground || lowest_corner_z(&me.body) < 5.0;
        m[CLEAR] = if at_floor {
            0.0
        } else {
            m[CLEAR] + events.step_seconds
        };
        if events.ball_on_floor || cur.ball.pos.z < BALL_RADIUS + 9.0 {
            m[BALL_FLOORED] = 1.0;
        }
        if m[PENDING] > 0.0 {
            m[PENDING] = 0.0;
            return f32::from(u8::from(cur.ball.vel.z - m[VZ_BEFORE] >= MIN_POP));
        }
        if me.on_ground {
            m[ARMED] = 0.0;
            m[FLIP_SEEN] = 0.0;
        } else if detail.is_flipping && m[FLIP_SEEN] == 0.0 {
            m[FLIP_SEEN] = 1.0;
            m[ARMED] = f32::from(u8::from(forward_diagonal(detail.flip_rel_torque)));
        }
        if m[ARMED] > 0.0 && events.cars[car].touched_ball {
            m[ARMED] = 0.0;
            let body = &me.body;
            let ok = m[BALL_FLOORED] == 0.0
                && m[CLEAR] >= MIN_CLEAR_OF_FLOOR
                && body.up().z <= 0.0
                && body.forward().z >= 5f32.to_radians().sin()
                && bottom_front_corner(body, cur.ball.pos);
            if ok {
                m[PENDING] = 1.0;
                m[VZ_BEFORE] = prev.ball.vel.z;
            }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    /// The confirmed flick's pose at the touch: roof tilted 111°, nose 53° up (a rotation about the car's
    /// side axis), 300 uu up, the ball at its bottom front right corner.
    fn wizard_pose() -> GameSnapshot {
        let rot = Mat3A::from_rotation_y((-127f32).to_radians());
        let car = Body {
            pos: Vec3A::new(0.0, 0.0, 300.0),
            rot,
            ..Body::RESTING
        };
        let corner = Vec3A::new(HB_X1, HB_Y, HB_Z0);
        let out = Vec3A::new(1.0, 1.0, -1.0).normalize() * BALL_RADIUS;
        let ball = car.pos + rot * (corner + out);
        let mut g = game(ball, &[(Team::Blue, car.pos)]);
        g.cars[0].body = car;
        g.cars[0].on_ground = false;
        g
    }

    fn dodge_events(torque: Vec3A) -> StepEvents {
        let mut e = quiet(1);
        e.details[0].is_flipping = true;
        e.details[0].flip_rel_torque = torque;
        e
    }

    /// Runs: a diagonal dodge, 0.5 s in the air, the touch, then a step with the ball's `vz_after`.
    fn run(touch: &GameSnapshot, torque: Vec3A, vz_after: f32, m: &mut [f32]) -> f32 {
        let r = WizardFlick {};
        let mut air = touch.clone();
        air.ball.pos.z = 600.0;
        r.value_remembering(&air, &air, &dodge_events(torque), 0, m);
        for _ in 0..8 {
            r.value_remembering(&air, &air, &quiet(1), 0, m);
        }
        let mut before = touch.clone();
        before.ball.vel.z = -100.0;
        r.value_remembering(&before, touch, &touched(quiet(1), 0), 0, m) + {
            let mut after = touch.clone();
            after.ball.vel.z = vz_after;
            r.value_remembering(touch, &after, &quiet(1), 0, m)
        }
    }

    const DIAGONAL: Vec3A = Vec3A::new(
        -std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
    );

    #[test]
    fn pays_the_confirmed_pose() {
        assert!(bottom_front_corner(
            &wizard_pose().cars[0].body,
            wizard_pose().ball.pos
        ));
        assert_eq!(run(&wizard_pose(), DIAGONAL, 700.0, &mut [0.0; 6]), 1.0);
        let left = Vec3A::new(-DIAGONAL.x, DIAGONAL.y, 0.0);
        assert_eq!(
            run(&wizard_pose(), left, 700.0, &mut [0.0; 6]),
            1.0,
            "a left dodge too"
        );
    }

    #[test]
    fn nothing_without_each_part() {
        let pose = wizard_pose();
        assert_eq!(run(&pose, DIAGONAL, 50.0, &mut [0.0; 6]), 0.0, "no pop");
        assert_eq!(
            run(&pose, Vec3A::new(0.0, 1.0, 0.0), 700.0, &mut [0.0; 6]),
            0.0,
            "a front flip"
        );
        assert_eq!(
            run(&pose, Vec3A::new(-1.0, 0.0, 0.0), 700.0, &mut [0.0; 6]),
            0.0,
            "a side flip"
        );
        let mut m = [0.0; 6];
        m[BALL_FLOORED] = 1.0;
        assert_eq!(
            run(&pose, DIAGONAL, 700.0, &mut m),
            0.0,
            "the ball bounced first"
        );
        let mut level = pose.clone();
        level.cars[0].body.rot = Mat3A::IDENTITY;
        assert_eq!(
            run(&level, DIAGONAL, 700.0, &mut [0.0; 6]),
            0.0,
            "roof up: a normal flick"
        );
        let mut edge = pose.clone();
        let rot = edge.cars[0].body.rot;
        edge.ball.pos = edge.cars[0].body.pos + rot * Vec3A::new(HB_X1 + BALL_RADIUS, 0.0, HB_Z0);
        assert_eq!(
            run(&edge, DIAGONAL, 700.0, &mut [0.0; 6]),
            0.0,
            "the front face, not a corner"
        );
        let mut low = pose.clone();
        let lift = 300.0 - 40.0;
        low.cars[0].body.pos.z -= lift;
        low.ball.pos.z -= lift;
        assert_eq!(
            run(&low, DIAGONAL, 700.0, &mut [0.0; 6]),
            0.0,
            "a corner at the floor"
        );
    }

    #[test]
    fn only_the_first_touch_after_the_dodge_counts() {
        let pose = wizard_pose();
        let r = WizardFlick {};
        let mut m = [0.0; 6];
        let mut level = pose.clone();
        level.cars[0].body.rot = Mat3A::IDENTITY;
        r.value_remembering(&level, &level, &dodge_events(DIAGONAL), 0, &mut m);
        for _ in 0..8 {
            r.value_remembering(&level, &level, &quiet(1), 0, &mut m);
        }
        r.value_remembering(&level, &level, &touched(quiet(1), 0), 0, &mut m);
        r.value_remembering(&pose, &pose, &touched(quiet(1), 0), 0, &mut m);
        let mut after = pose.clone();
        after.ball.vel.z = 900.0;
        assert_eq!(
            r.value_remembering(&pose, &after, &quiet(1), 0, &mut m),
            0.0
        );
    }
}
