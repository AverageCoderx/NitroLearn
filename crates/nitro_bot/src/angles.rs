//! Rocket League's (pitch, yaw, roll) rotator and the rotation matrix it stands for.
//!
//! The matrix columns are the car's forward, right and up directions, as in `nitro_core::snapshot::Body`.
//! The formula is the one every RLBot bot library uses for the game's rotator (RLUtilities, RocketSim's
//! C++ `Angle::ToRotMat`): positive pitch is nose up, yaw turns from +X toward +Y, positive roll tilts
//! the car to its right.

use nitro_core::custom::prelude::{Mat3A, Vec3A};
use rlbot::flat::Vector3;

/// An RLBot vector as RocketSim's (RLBot's own conversion is to an older glam than RocketSim 0.3's).
pub fn vec3(v: Vector3) -> Vec3A {
    Vec3A::new(v.x, v.y, v.z)
}

/// The rotation matrix of a rotator, in radians.
pub fn rotation_matrix(pitch: f32, yaw: f32, roll: f32) -> Mat3A {
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    let (sr, cr) = roll.sin_cos();
    Mat3A::from_cols(
        Vec3A::new(cp * cy, cp * sy, sp),
        Vec3A::new(cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr),
        Vec3A::new(-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr),
    )
}

/// The (pitch, yaw, roll) rotator of a rotation matrix: the inverse of [`rotation_matrix`], used to
/// synthesise game packets from simulated states in the parity tests.
pub fn rotator(rot: Mat3A) -> (f32, f32, f32) {
    let forward = rot.x_axis;
    let pitch = forward.z.clamp(-1.0, 1.0).asin();
    let yaw = forward.y.atan2(forward.x);
    // Roll from the up and right vectors' vertical components: up.z = cp·cr, right.z = −cp·sr.
    let roll = (-rot.y_axis.z).atan2(rot.z_axis.z);
    (pitch, yaw, roll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, TAU};

    fn close(a: Vec3A, b: Vec3A) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn axis_cases_match_the_games_conventions() {
        let flat = rotation_matrix(0.0, 0.0, 0.0);
        assert!(
            close(flat.x_axis, Vec3A::X)
                && close(flat.y_axis, Vec3A::Y)
                && close(flat.z_axis, Vec3A::Z)
        );
        // Yaw a quarter turn: facing +Y.
        assert!(close(rotation_matrix(0.0, FRAC_PI_2, 0.0).x_axis, Vec3A::Y));
        // Nose straight up.
        assert!(close(rotation_matrix(FRAC_PI_2, 0.0, 0.0).x_axis, Vec3A::Z));
        // Rolled a quarter turn to the right: the roof points to the car's right (+Y when facing +X).
        assert!(close(rotation_matrix(0.0, 0.0, FRAC_PI_2).z_axis, Vec3A::Y));
    }

    #[test]
    fn rotator_inverts_rotation_matrix() {
        let mut rng = fastrand::Rng::with_seed(5);
        for _ in 0..1000 {
            let pitch = (rng.f32() - 0.5) * 3.0; // within (−π/2, π/2)
            let yaw = (rng.f32() - 0.5) * TAU;
            let roll = (rng.f32() - 0.5) * TAU;
            let rot = rotation_matrix(pitch, yaw, roll);
            let (p, y, r) = rotator(rot);
            let back = rotation_matrix(p, y, r);
            for col in 0..3 {
                assert!(close(rot.col(col), back.col(col)), "{pitch} {yaw} {roll}");
            }
        }
    }
}
