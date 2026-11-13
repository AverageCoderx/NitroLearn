//! Dense credit for setting up the Raze flick: from the pop jump off a dribble to the dodge, paid by how
//! closely the car and ball match reference Razes.
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::flick_kit::*;

/// From the pop jump off a dribble to the dodge, at most once per possession: setup_rate x (0-1, how
/// closely the setup matches the scripted Razes) per second; about 0.87 x setup_rate in all at best.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RazeFlickSetup {
    /// Ball under control on the roof: its centre at least this far above the car along its roof direction (uu). Default 105.
    #[serde(default = "sd::dribble_min_up")]
    pub dribble_min_up: f32,
    /// ... and at most this far (uu). Default 185.
    #[serde(default = "sd::dribble_max_up")]
    pub dribble_max_up: f32,
    /// ... best this far forward of the car's centre (uu). Default 20.
    #[serde(default = "sd::dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// ... no control this far or further from that spot, sideways (uu). Default 90.
    #[serde(default = "sd::dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// ... nor at this speed relative to the car (uu/s). Default 600.
    #[serde(default = "sd::dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality (0-1) that counts as the ball under control. Default 0.25.
    #[serde(default = "sd::control_threshold")]
    pub control_threshold: f32,
    /// Seconds of control before the pop jump (the Raze set: 0.05). Default 0.25.
    #[serde(default = "sd::min_possession_time")]
    pub min_possession_time: f32,
    /// The jump must come within this many seconds of the last controlled step. Default 1.
    #[serde(default = "sd::possession_memory")]
    pub possession_memory: f32,
    /// Reward per second at a perfect match. Default 1.
    #[serde(default = "sd::setup_rate")]
    pub setup_rate: f32,
    /// Training option: a ball on the roof when the episode starts counts as a dribble (Raze bot: true). Default false.
    #[serde(default = "sd::possession_at_reset")]
    pub possession_at_reset: bool,
    /// Ball contacts within this many ticks of the jump are the pop, not a hit. Default 24.
    #[serde(default = "sd::pop_ticks")]
    pub pop_ticks: f32,
    /// Back on the wheels after this many ticks: the attempt is over. Default 24.
    #[serde(default = "sd::land_ticks")]
    pub land_ticks: f32,
    /// Each feature's squared mismatch counts at most this much. Default 16.
    #[serde(default = "sd::max_z2")]
    pub max_z2: f32,
    /// Match = exp(-0.5 x this x mean squared mismatch). Default 1.
    #[serde(default = "sd::q_scale")]
    pub q_scale: f32,
    /// Smallest spread allowed for the car height (uu). Default 12.
    #[serde(default = "sd::min_spread_car_z")]
    pub min_spread_car_z: f32,
    /// Smallest spread allowed for the car vertical speed (uu/s). Default 60.
    #[serde(default = "sd::min_spread_car_vz")]
    pub min_spread_car_vz: f32,
    /// Smallest spread allowed for the heading away from the goal (degrees). Default 10.
    #[serde(default = "sd::min_spread_heading")]
    pub min_spread_heading: f32,
    /// Smallest spread allowed for the roof tilt (degrees). Default 10.
    #[serde(default = "sd::min_spread_roof_tilt")]
    pub min_spread_roof_tilt: f32,
    /// Smallest spread allowed for the nose pitch (forward z). Default 0.1.
    #[serde(default = "sd::min_spread_nose_pitch")]
    pub min_spread_nose_pitch: f32,
    /// Smallest spread allowed for the ball ahead along the goal direction (uu). Default 35.
    #[serde(default = "sd::min_spread_ball_along_goal")]
    pub min_spread_ball_along_goal: f32,
    /// Smallest spread allowed for the ball to the turn side (uu). Default 40.
    #[serde(default = "sd::min_spread_ball_lateral")]
    pub min_spread_ball_lateral: f32,
    /// Smallest spread allowed for the ball height above the car (uu). Default 35.
    #[serde(default = "sd::min_spread_ball_z")]
    pub min_spread_ball_z: f32,
    /// Smallest spread allowed for the ball speed along the goal direction (uu/s). Default 120.
    #[serde(default = "sd::min_spread_ball_vel_along_goal")]
    pub min_spread_ball_vel_along_goal: f32,
    /// Smallest spread allowed for the ball vertical speed relative to the car (uu/s). Default 100.
    #[serde(default = "sd::min_spread_ball_vz")]
    pub min_spread_ball_vz: f32,
    /// Smallest spread allowed for the car speed toward the goal (uu/s). Default 300.
    #[serde(default = "sd::min_spread_car_speed_along_goal")]
    pub min_spread_car_speed_along_goal: f32,
    /// Smallest spread allowed for the roof toward the turn side. Default 0.15.
    #[serde(default = "sd::min_spread_roof_lateral")]
    pub min_spread_roof_lateral: f32,
}

/// RazeFlickSetup's defaults.
mod sd {
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
    pub fn min_possession_time() -> f32 {
        0.25
    }
    pub fn possession_memory() -> f32 {
        1.0
    }
    pub fn setup_rate() -> f32 {
        1.0
    }
    pub fn possession_at_reset() -> bool {
        false
    }
    pub fn pop_ticks() -> f32 {
        24.0
    }
    pub fn land_ticks() -> f32 {
        24.0
    }
    pub fn max_z2() -> f32 {
        16.0
    }
    pub fn q_scale() -> f32 {
        1.0
    }
    pub fn min_spread_car_z() -> f32 {
        12.0
    }
    pub fn min_spread_car_vz() -> f32 {
        60.0
    }
    pub fn min_spread_heading() -> f32 {
        10.0
    }
    pub fn min_spread_roof_tilt() -> f32 {
        10.0
    }
    pub fn min_spread_nose_pitch() -> f32 {
        0.1
    }
    pub fn min_spread_ball_along_goal() -> f32 {
        35.0
    }
    pub fn min_spread_ball_lateral() -> f32 {
        40.0
    }
    pub fn min_spread_ball_z() -> f32 {
        35.0
    }
    pub fn min_spread_ball_vel_along_goal() -> f32 {
        120.0
    }
    pub fn min_spread_ball_vz() -> f32 {
        100.0
    }
    pub fn min_spread_car_speed_along_goal() -> f32 {
        300.0
    }
    pub fn min_spread_roof_lateral() -> f32 {
        0.15
    }
}

const FEATURES: usize = 12;
const REF_STEPS: usize = 26;
/// Ticks between the reference's steps.
const REF_STEP_TICKS: f32 = 4.0;

/// The scripted Razes' mean of each feature at each reference step (4 ticks apart from the pop jump; 267
/// paying scripted Razes, `raze_sim setupref`): car height, car vertical speed, |heading| from the goal
/// (degrees), roof tilt (degrees), nose pitch (forward z), ball relative along the goal direction, to the
/// turn side, up; ball relative velocity along the goal direction, up; car speed toward the goal; roof
/// toward the turn side.
const REF_MEAN: [[f32; FEATURES]; REF_STEPS] = [
    [
        19.413,
        297.03,
        6.888,
        0.76658,
        -0.012175,
        43.889,
        0.10899,
        130.89,
        36.022,
        -265.97,
        1609.1,
        -0.00086711,
    ],
    [
        28.32, 272.46, 8.6875, 3.0902, -0.047246, 48.509, -4.1261, 133.88, 226.11, 198.5, 1589.7,
        -0.01045,
    ],
    [
        37.803, 293.87, 10.561, 5.376, -0.081151, 56.065, -7.2238, 139.63, 224.79, 154.96, 1588.9,
        -0.017476,
    ],
    [
        48.155, 320.53, 12.37, 7.4348, -0.11143, 63.439, -10.274, 143.78, 218.23, 106.19, 1593.3,
        -0.022394,
    ],
    [
        59.279, 333.64, 14.11, 9.3278, -0.1393, 70.506, -13.283, 146.43, 203.77, 72.57, 1606.6,
        -0.025322,
    ],
    [
        69.816, 305.79, 15.825, 11.749, -0.17791, 76.712, -23.702, 149.24, 173.41, 88.218, 1641.1,
        -0.03739,
    ],
    [
        79.418, 277.2, 17.251, 14.757, -0.2283, 81.808, -34.953, 152.32, 138.83, 94.756, 1673.5,
        -0.042833,
    ],
    [
        88.05, 249.19, 19.07, 17.459, -0.27036, 85.845, -40.823, 155.62, 112.96, 100.72, 1697.2,
        -0.039413,
    ],
    [
        95.881, 224.67, 21.673, 21.325, -0.31583, 89.573, -46.762, 159.0, 103.16, 103.24, 1704.7,
        -0.061194,
    ],
    [
        102.66, 190.65, 24.923, 25.809, -0.34776, 92.384, -53.031, 162.69, 69.288, 115.26, 1736.3,
        -0.11157,
    ],
    [
        108.29, 155.67, 27.771, 29.879, -0.36513, 94.113, -59.758, 166.8, 36.628, 128.27, 1766.7,
        -0.16227,
    ],
    [
        112.77, 123.64, 30.128, 34.848, -0.3776, 94.867, -66.958, 171.33, 12.502, 138.36, 1788.5,
        -0.23074,
    ],
    [
        116.44, 101.97, 32.809, 40.865, -0.38021, 95.414, -74.245, 175.93, 11.262, 138.1, 1787.3,
        -0.31296,
    ],
    [
        119.36, 76.839, 35.536, 44.714, -0.36522, 95.887, -81.556, 180.56, 2.9623, 141.33, 1793.1,
        -0.36072,
    ],
    [
        121.19, 41.751, 38.08, 46.163, -0.34511, 95.603, -89.386, 185.55, -26.054, 154.54, 1819.6,
        -0.37105,
    ],
    [
        121.87, 7.6784, 40.83, 46.916, -0.31895, 94.428, -97.899, 190.96, -53.76, 166.75, 1844.8,
        -0.37103,
    ],
    [
        121.44, -25.084, 43.277, 46.109, -0.29441, 92.443, -107.12, 196.74, -79.668, 177.68,
        1868.0, -0.3515,
    ],
    [
        119.95, -54.807, 45.524, 46.396, -0.25747, 89.717, -117.06, 202.87, -99.421, 185.59,
        1884.9, -0.35623,
    ],
    [
        117.67, -76.473, 47.544, 46.802, -0.20447, 86.886, -127.15, 209.05, -99.807, 185.47,
        1882.4, -0.37337,
    ],
    [
        114.67, -98.14, 49.198, 47.318, -0.15409, 84.114, -137.24, 215.23, -100.12, 185.36, 1879.7,
        -0.39135,
    ],
    [
        110.95, -119.81, 50.525, 49.682, -0.11075, 81.408, -147.33, 221.41, -100.35, 185.28,
        1876.8, -0.4224,
    ],
    [
        106.5, -141.47, 52.36, 54.094, -0.068774, 78.779, -157.42, 227.58, -100.5, 185.23, 1873.7,
        -0.45695,
    ],
    [
        101.33, -163.52, 53.947, 60.177, -0.030174, 76.194, -167.57, 233.76, -106.04, 185.56,
        1875.9, -0.49266,
    ],
    [
        95.416, -185.69, 55.412, 67.659, 0.016827, 73.184, -178.5, 239.96, -126.92, 186.06, 1893.1,
        -0.52396,
    ],
    [
        88.798, -205.98, 56.13, 76.038, 0.077543, 69.716, -190.37, 246.14, -145.74, 184.69, 1908.0,
        -0.54931,
    ],
    [
        81.548, -224.09, 56.082, 82.023, 0.13302, 64.815, -195.71, 252.23, -216.03, 181.17, 1974.6,
        -0.56042,
    ],
];

/// ... and their spread.
const REF_STD: [[f32; FEATURES]; REF_STEPS] = [
    [
        0.062339, 8.2152, 5.218, 0.1446, 0.0022621, 1.7993, 0.34144, 0.48303, 20.427, 69.232,
        37.086, 0.0013479,
    ],
    [
        0.074878, 2.2039, 5.7331, 0.13722, 0.0018533, 2.3027, 3.8024, 0.65236, 41.375, 63.571,
        37.869, 0.0078815,
    ],
    [
        0.13589, 2.2055, 5.8123, 0.21976, 0.0032414, 3.2517, 5.8229, 2.7556, 41.434, 63.504,
        38.026, 0.013364,
    ],
    [
        0.20556, 2.2163, 5.8942, 0.31689, 0.0049158, 4.4372, 7.9176, 4.8691, 41.411, 63.438,
        38.137, 0.017461,
    ],
    [
        0.27769, 2.3004, 5.9793, 0.40781, 0.0064547, 5.7156, 10.022, 6.9648, 40.072, 59.981,
        38.364, 0.020292,
    ],
    [
        0.33301, 2.2039, 6.0696, 0.51011, 0.0072934, 6.6796, 11.944, 7.9651, 27.67, 25.406, 39.071,
        0.024532,
    ],
    [
        0.40009, 2.3942, 6.1655, 0.61688, 0.0082919, 7.5799, 8.9122, 8.7935, 28.387, 25.474,
        39.894, 0.028581,
    ],
    [
        0.47723, 2.5736, 6.267, 0.72842, 0.0093395, 8.5151, 9.4872, 9.6262, 29.043, 25.507, 40.703,
        0.034204,
    ],
    [
        0.55853, 2.6531, 6.3735, 0.86355, 0.01028, 9.4749, 10.051, 10.461, 29.433, 25.497, 41.306,
        0.041406,
    ],
    [
        0.64953, 2.9984, 6.486, 1.0154, 0.011114, 10.473, 10.536, 11.298, 30.613, 25.53, 42.688,
        0.048916,
    ],
    [
        0.75289, 3.3843, 6.6049, 1.1522, 0.011964, 11.521, 10.921, 12.136, 32.043, 25.565, 44.371,
        0.055323,
    ],
    [
        0.86888, 3.6993, 6.7311, 1.2723, 0.012732, 12.626, 11.212, 12.976, 33.37, 25.586, 46.04,
        0.062105,
    ],
    [
        0.98841, 3.6993, 6.8627, 1.3694, 0.013303, 13.767, 11.494, 13.816, 33.722, 25.56, 46.859,
        0.069125,
    ],
    [
        1.1097, 3.8183, 6.9987, 1.4481, 0.013841, 14.942, 11.782, 14.657, 34.539, 25.551, 48.179,
        0.073839,
    ],
    [
        1.2436, 4.3101, 7.1419, 1.5156, 0.014302, 16.192, 11.995, 15.498, 36.927, 25.597, 51.107,
        0.077033,
    ],
    [
        1.3937, 4.8145, 7.2932, 1.5738, 0.014622, 17.538, 12.113, 16.34, 39.673, 25.657, 54.518,
        0.079925,
    ],
    [
        1.5599, 5.3173, 7.454, 1.6257, 0.014902, 18.996, 12.144, 17.183, 42.765, 25.733, 58.428,
        0.081684,
    ],
    [
        1.7422, 5.721, 7.6237, 1.6676, 0.015245, 20.582, 12.09, 18.028, 45.675, 25.798, 62.356,
        0.083681,
    ],
    [
        1.9286, 5.721, 7.8013, 1.6981, 0.015651, 22.258, 12.03, 18.873, 46.585, 25.773, 64.514,
        0.085021,
    ],
    [
        2.1157, 5.721, 7.9893, 1.7184, 0.016038, 24.015, 11.985, 19.719, 47.556, 25.748, 66.881,
        0.086509,
    ],
    [
        2.3035, 5.721, 8.1886, 1.7288, 0.016397, 25.859, 11.958, 20.565, 48.591, 25.723, 69.479,
        0.090352,
    ],
    [
        2.4917, 5.721, 8.3999, 1.738, 0.016524, 27.796, 11.953, 21.411, 49.698, 25.699, 72.334,
        0.097531,
    ],
    [
        2.6814, 5.8695, 8.6241, 1.7454, 0.016609, 29.844, 11.969, 22.257, 51.903, 25.708, 76.489,
        0.10684,
    ],
    [
        2.8867, 6.4578, 8.8642, 1.7518, 0.01662, 32.106, 11.945, 23.104, 57.498, 25.826, 84.319,
        0.11719,
    ],
    [
        3.1103, 7.0247, 9.1225, 1.7532, 0.016749, 34.635, 11.881, 23.954, 63.455, 25.949, 92.926,
        0.12664,
    ],
    [
        3.3526, 7.6102, 9.3955, 1.7526, 0.01696, 36.433, 11.578, 24.805, 88.172, 26.079, 69.158,
        0.13317,
    ],
];

// Memory, per car.
const INIT: usize = 0;
const POSSESSION: usize = 1;
const CONTROL_TIME: usize = 2;
const CONTROL_AGE: usize = 3;
const ATTEMPT_USED: usize = 4;
const ACTIVE: usize = 5;
const JUMP_AGE: usize = 6;
const SIDE: usize = 7;
const MEMORY: usize = 8;

impl RazeFlickSetup {
    fn dribble(&self) -> Dribble {
        Dribble {
            min_up: self.dribble_min_up,
            max_up: self.dribble_max_up,
            target_forward: self.dribble_target_forward,
            max_planar_dist: self.dribble_max_planar_dist,
            max_rel_speed: self.dribble_max_rel_speed,
        }
    }

    fn min_spread(&self) -> [f32; FEATURES] {
        [
            self.min_spread_car_z,
            self.min_spread_car_vz,
            self.min_spread_heading,
            self.min_spread_roof_tilt,
            self.min_spread_nose_pitch,
            self.min_spread_ball_along_goal,
            self.min_spread_ball_lateral,
            self.min_spread_ball_z,
            self.min_spread_ball_vel_along_goal,
            self.min_spread_ball_vz,
            self.min_spread_car_speed_along_goal,
            self.min_spread_roof_lateral,
        ]
    }

    /// The match (0 to 1) of `f` with the reference at step `k`.
    fn quality(&self, f: &[f32; FEATURES], k: f32) -> f32 {
        if !(0.0..REF_STEPS as f32).contains(&k) {
            return 0.0;
        }
        let k = k as usize;
        let spread = self.min_spread();
        let sum: f32 = (0..FEATURES)
            .map(|i| {
                let z = div(f[i] - REF_MEAN[k][i], REF_STD[k][i].max(spread[i]));
                (z * z).min(self.max_z2)
            })
            .sum();
        (-0.5 * self.q_scale * sum / FEATURES as f32).exp()
    }
}

/// The horizontal direction from the car to the goal it attacks, and its left-handed lateral (the car's
/// right when facing the goal).
fn goal_frame(car: &CarSnapshot) -> (Vec3A, Vec3A) {
    let to_goal = target_goal(car.team) - car.body.pos;
    let flat = Vec3A::new(to_goal.x, to_goal.y, 0.0);
    let len = flat.length();
    let g = if len > 1e-3 { flat / len } else { Vec3A::Y };
    (g, Vec3A::new(-g.y, g.x, 0.0))
}

/// The features, lateral ones taken on the side the car turns to (`side`).
fn features(car: &CarSnapshot, ball: &Body, side: f32) -> [f32; FEATURES] {
    let (g, l) = goal_frame(car);
    let rot = car.body.rot;
    let nose = Vec3A::new(rot.x_axis.x, rot.x_axis.y, 0.0);
    let len = nose.length();
    let nose = if len > 1e-3 { nose / len } else { g };
    let heading = degrees(nose.dot(g).clamp(-1.0, 1.0).acos());
    let rel = ball.pos - car.body.pos;
    let rel_vel = ball.vel - car.body.vel;
    [
        car.body.pos.z,
        car.body.vel.z,
        heading,
        degrees(rot.z_axis.z.clamp(-1.0, 1.0).acos()),
        rot.x_axis.z,
        rel.dot(g),
        rel.dot(l) * side,
        rel.z,
        rel_vel.dot(g),
        rel_vel.z,
        car.body.vel.dot(g),
        rot.z_axis.dot(l) * side,
    ]
}

/// The side the car turns to: the sign of its nose's lateral part once clearly turned (8°), else 0.
fn turn_side(car: &CarSnapshot) -> f32 {
    let to_goal = target_goal(car.team) - car.body.pos;
    let g = Vec3A::new(to_goal.x, to_goal.y, 0.0).normalize_or_zero();
    let l = Vec3A::new(-g.y, g.x, 0.0);
    let f = Vec3A::new(car.body.rot.x_axis.x, car.body.rot.x_axis.y, 0.0).normalize_or_zero();
    let (lat, fwd) = (f.dot(l), f.dot(g));
    if degrees(lat.atan2(fwd).abs()) < 8.0 {
        0.0
    } else if lat > 0.0 {
        1.0
    } else {
        -1.0
    }
}

impl Reward for RazeFlickSetup {
    fn memory(&self) -> usize {
        MEMORY
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
            // Training option: a ball on the roof as the episode starts counts as a dribble
            if self.possession_at_reset
                && control_quality(before, &prev.ball, self.dribble()) >= self.control_threshold
            {
                m[POSSESSION] = 1.0;
                m[CONTROL_TIME] = self.min_possession_time;
                m[CONTROL_AGE] = 0.0;
            }
        }
        let ticks = step_ticks(events);
        m[CONTROL_AGE] += ticks;
        m[JUMP_AGE] += ticks;
        let me = &cur.cars[car];
        let dt = events.step_seconds.clamp(0.0, 0.5);
        if me.is_demoed {
            m[ACTIVE] = 0.0;
            m[POSSESSION] = 0.0;
            m[ATTEMPT_USED] = 0.0;
            return 0.0;
        }

        // Possession (as raze_flick)
        let in_control = control_quality(me, &cur.ball, self.dribble()) >= self.control_threshold;
        let memory_ticks = secs_to_ticks(self.possession_memory);
        let mut since = if m[POSSESSION] != 0.0 {
            m[CONTROL_AGE]
        } else {
            f32::INFINITY
        };
        if in_control {
            if m[POSSESSION] == 0.0 {
                m[POSSESSION] = 1.0;
                m[CONTROL_TIME] = 0.0;
                m[ATTEMPT_USED] = 0.0;
            }
            m[CONTROL_TIME] += dt;
            m[CONTROL_AGE] = 0.0;
            since = 0.0;
        } else if m[POSSESSION] != 0.0 && m[ACTIVE] == 0.0 && since > memory_ticks {
            m[POSSESSION] = 0.0;
            m[CONTROL_TIME] = 0.0;
            m[ATTEMPT_USED] = 0.0;
        }
        let recent_control = m[POSSESSION] != 0.0
            && since <= memory_ticks
            && m[CONTROL_TIME] >= self.min_possession_time - 1e-4;

        // The start: a jump off the ground with the ball under control (the pop)
        if m[ACTIVE] == 0.0
            && m[ATTEMPT_USED] == 0.0
            && recent_control
            && me.has_jumped
            && !before.has_jumped
            && before.on_ground
        {
            m[ACTIVE] = 1.0;
            m[ATTEMPT_USED] = 1.0;
            m[JUMP_AGE] = 0.0;
            m[SIDE] = 0.0;
        }
        if m[ACTIVE] == 0.0 {
            return 0.0;
        }
        let phase = m[JUMP_AGE];
        let k = (phase / REF_STEP_TICKS).round();
        let dodged = me.has_flipped && !before.has_flipped;
        let landed = me.on_ground && phase > self.land_ticks;
        let hit = events.cars.get(car).is_some_and(|e| e.touched_ball) && phase > self.pop_ticks;
        let past_reference = k >= REF_STEPS as f32;
        if landed || hit || past_reference || (me.has_flipped && !dodged) {
            m[ACTIVE] = 0.0;
            return 0.0;
        }
        if m[SIDE] == 0.0 {
            m[SIDE] = turn_side(me);
        }
        let q = self.quality(&features(me, &cur.ball, m[SIDE]), k);
        if dodged {
            m[ACTIVE] = 0.0; // the dodge step is scored, then raze_flick takes over
        }
        self.setup_rate * q * dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn mirrored(g: &GameSnapshot) -> GameSnapshot {
        let turn = |b: Body| Body {
            pos: as_blue(Team::Orange, b.pos),
            vel: as_blue(Team::Orange, b.vel),
            ang_vel: as_blue(Team::Orange, b.ang_vel),
            rot: Mat3A::from_cols(
                as_blue(Team::Orange, b.rot.x_axis),
                as_blue(Team::Orange, b.rot.y_axis),
                as_blue(Team::Orange, b.rot.z_axis),
            ),
        };
        let mut out = g.clone();
        out.ball = turn(g.ball);
        for c in &mut out.cars {
            c.body = turn(c.body);
            c.team = c.team.opposite();
        }
        out
    }

    /// Five steps of dribbling toward the goal, the pop jump, four steps in the air, the dodge, more air.
    fn setup() -> Vec<GameSnapshot> {
        let level = Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z);
        let mut dribble = game(
            Vec3A::new(0.0, 20.0, 150.0),
            &[(Team::Blue, Vec3A::new(0.0, 0.0, 17.0))],
        );
        dribble.cars[0].body.rot = level;
        dribble.cars[0].body.vel = Vec3A::new(0.0, 1500.0, 0.0);
        dribble.ball.vel = dribble.cars[0].body.vel;
        let mut games = vec![dribble.clone(); 6];
        for k in 0..7 {
            let mut air = dribble.clone();
            let me = &mut air.cars[0];
            me.on_ground = false;
            me.has_jumped = true;
            me.has_flipped = k >= 5;
            me.body.pos.z = 20.0 + 10.0 * k as f32;
            me.body.vel.z = 300.0;
            air.ball.pos = Vec3A::new(0.0, 50.0, 160.0 + 12.0 * k as f32);
            games.push(air);
        }
        games
    }

    fn run(reward: &RazeFlickSetup, games: &[GameSnapshot]) -> Vec<f32> {
        let mut memory = vec![0.0; reward.memory()];
        let events = quiet(games[0].cars.len());
        (1..games.len())
            .map(|i| reward.value_remembering(&games[i - 1], &games[i], &events, 0, &mut memory))
            .collect()
    }

    #[test]
    fn the_setup_pays_from_the_pop_jump_to_the_dodge() {
        let reward: RazeFlickSetup =
            serde_json::from_value(serde_json::json!({})).expect("defaults");
        let paid = run(&reward, &setup());
        let dt = 8.0 / 120.0;
        assert!(paid[..5].iter().all(|&r| r == 0.0), "{paid:?}");
        assert!(paid[5..11].iter().all(|&r| r > 0.0 && r <= dt), "{paid:?}");
        assert_eq!(paid[11], 0.0, "the attempt ended at the dodge");
        assert_eq!(run(&reward, &setup()), paid, "fresh memory starts again");
        let mirror: Vec<GameSnapshot> = setup().iter().map(mirrored).collect();
        let orange = run(&reward, &mirror);
        assert!(
            paid.iter().zip(&orange).all(|(b, o)| (b - o).abs() < 1e-6),
            "{orange:?}"
        );
    }

    #[test]
    fn no_dribble_no_credit_and_the_reference_itself_matches_fully() {
        let reward: RazeFlickSetup = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut games = setup();
        for g in &mut games[..6] {
            g.ball.pos.z = 400.0;
        }
        assert!(run(&reward, &games).iter().all(|&r| r == 0.0));
        assert!((reward.quality(&REF_MEAN[3], 3.0) - 1.0).abs() < 1e-6);
        assert!(reward.quality(&REF_MEAN[3], 26.0) == 0.0);
    }
}
