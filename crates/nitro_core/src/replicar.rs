//! A recording built straight from a replicar file's columns (the large replay mode): the
//! one `replays.recording` writes as RLPR bytes and `rlpr::read` parses back, without the bytes, so the
//! work after the conversion runs outside Python's lock. Python reads the file and lays its columns out as
//! below (`replays.core_columns`), cars Blue first; every value is computed as `replays.recording` does.

use rocketsim::{Mat3A, Vec3A};

use crate::replay::is_kickoff;
use crate::rlpr::{CarRecord, Recording, Tick, VERSION};
use crate::snapshot::{Body, Controls};

/// Per row and car, as little-endian f32: position 3, rotation quaternion (x, y, z, w) 4, velocity 3,
/// angular velocity 3, flip torque 3, flip time, air time since the jump, jump ticks, boost (0 to 100),
/// throttle, steer, pitch, yaw, roll.
pub const CAR_FLOATS: usize = 25;
/// Per row and car, replicar's flags as u8 (1 when replicar's is 1, else 0): on the ground, has jumped, has
/// flipped, has double jumped, jumping, flipping, jump held, boost held, handbrake held.
pub const CAR_FLAGS: usize = 9;
/// Per row, as little-endian f32: the ball's position 3, rotation quaternion 4, velocity 3, angular
/// velocity 3.
pub const BALL_FLOATS: usize = 13;

const DOUBLE_JUMP_MAX_DELAY: f32 = 1.25;
const TICK_SECONDS: f32 = 1.0 / 120.0;
/// The Octane's hitbox (uu) for the header, as `replays.recording` writes it.
const OCTANE_SIZE: [f64; 3] = [120.507, 86.6994, 38.6591];
const OCTANE_OFFSET: [f64; 3] = [13.8757, 0.0, 20.755];

/// A replicar file's columns, every row of it (`usable` marks the rows kept).
pub struct Columns<'a> {
    pub cars: usize,
    /// Per row: the simulation tick, little-endian i64.
    pub frames: &'a [u8],
    /// Per row: 1 when the ball is known and every car present, not demolished and placed.
    pub usable: &'a [u8],
    pub car_floats: &'a [u8],
    pub car_flags: &'a [u8],
    pub ball_floats: &'a [u8],
}

#[derive(Debug)]
pub struct Built {
    pub recording: Recording,
    /// Ticks that are kickoffs (`replay::is_kickoff`); with none the teams cannot be told apart.
    pub kickoffs: usize,
    pub rows_left_out: usize,
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().expect("4 bytes"))
}

fn vec_at(bytes: &[u8], index: usize) -> Vec3A {
    Vec3A::new(
        f32_at(bytes, index),
        f32_at(bytes, index + 1),
        f32_at(bytes, index + 2),
    )
}

/// The rotation of a unit quaternion (x, y, z, w), computed in f64 and stored as f32 row by row, as
/// `replays._rotation_rows` and the parser read it.
fn rotation(bytes: &[u8], index: usize) -> Mat3A {
    let [x, y, z, w] = [0, 1, 2, 3].map(|k| f64::from(f32_at(bytes, index + k)));
    #[rustfmt::skip]
    let rows = [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    ]
    .map(|v| v as f32);
    let column = |c: usize| Vec3A::new(rows[c], rows[3 + c], rows[6 + c]);
    Mat3A::from_cols(column(0), column(1), column(2))
}

fn body(bytes: &[u8], at: usize) -> Body {
    Body {
        pos: vec_at(bytes, at),
        rot: rotation(bytes, at + 3),
        vel: vec_at(bytes, at + 7),
        ang_vel: vec_at(bytes, at + 10),
    }
}

fn car(floats: &[u8], flags: &[u8], at: usize, flag_at: usize) -> CarRecord {
    let flag = |k: usize| flags[flag_at + k] == 1;
    let value = |k: usize| f32_at(floats, at + k);
    let (on_ground, has_jumped, flipped, double) = (flag(0), flag(1), flag(2), flag(3));
    // The port counts jump ticks on the ground too; the format's jump time is 0 until a jump.
    let jump_time = if has_jumped {
        (f64::from(value(18)) / 120.0) as f32
    } else {
        0.0
    };
    let boost = f64::from(value(19)) / 100.0;
    let boost = if boost < 0.0 { 0.0 } else { boost };
    CarRecord {
        body: body(floats, at),
        on_ground,
        is_jumping: flag(4),
        is_flipping: flag(5),
        jump_time,
        // The format's flip time is 0 unless the car flipped (demos tells a dodge from a double jump by it);
        // replicar 2.1+ counts it as RocketSim 0.3 does, also on the ground and after a double jump.
        flip_time: if flipped { value(16) } else { 0.0 },
        has_jumped,
        double_jumped_or_flipped: double || flipped,
        // RocketSim 0.3's has_flip_or_jump: the next tick's time counted first.
        has_flip: on_ground
            || (!flipped && !double && value(17) + TICK_SECONDS < DOUBLE_JUMP_MAX_DELAY),
        flip_rel_torque: vec_at(floats, at + 13),
        boost: (if boost > 1.0 { 1.0 } else { boost }) as f32,
        touching_ball: false,
        controls: Controls {
            throttle: value(20),
            steer: value(21),
            pitch: value(22),
            yaw: value(23),
            roll: value(24),
            jump: flag(6),
            boost: flag(7),
            handbrake: flag(8),
        },
        touching_car: false,
    }
}

/// The recording of `columns`' usable rows, one tick each numbered by its simulation tick.
pub fn recording(columns: &Columns) -> Result<Built, String> {
    let (rows, cars) = (columns.usable.len(), columns.cars);
    let sizes = [
        (columns.frames.len(), rows * 8, "tick numbers"),
        (
            columns.car_floats.len(),
            rows * cars * CAR_FLOATS * 4,
            "car values",
        ),
        (
            columns.car_flags.len(),
            rows * cars * CAR_FLAGS,
            "car flags",
        ),
        (
            columns.ball_floats.len(),
            rows * BALL_FLOATS * 4,
            "ball values",
        ),
    ];
    for (got, wanted, what) in sizes {
        if got != wanted {
            return Err(format!(
                "the {what} hold {got} bytes, not {wanted} ({rows} rows, {cars} cars)"
            ));
        }
    }
    // In f64 with Python's 0.02, as `replays.recording` packs them.
    let low = |k: usize| ((OCTANE_OFFSET[k] - OCTANE_SIZE[k] / 2.0) * 0.02) as f32;
    let high = |k: usize| ((OCTANE_OFFSET[k] + OCTANE_SIZE[k] / 2.0) * 0.02) as f32;
    let mut ticks = Vec::with_capacity(columns.usable.iter().filter(|&&u| u == 1).count());
    let mut kickoffs = 0;
    for row in (0..rows).filter(|&row| columns.usable[row] == 1) {
        let frame = i64::from_le_bytes(
            columns.frames[row * 8..row * 8 + 8]
                .try_into()
                .expect("8 bytes"),
        );
        let frame =
            u32::try_from(frame).map_err(|_| "a tick number outside what a recording can hold")?;
        let tick = Tick {
            frame,
            cars: (0..cars)
                .map(|c| {
                    let index = row * cars + c;
                    car(
                        columns.car_floats,
                        columns.car_flags,
                        index * CAR_FLOATS,
                        index * CAR_FLAGS,
                    )
                })
                .collect(),
            ball: body(columns.ball_floats, row * BALL_FLOATS),
            ball_world_contact: false,
        };
        kickoffs += usize::from(is_kickoff(&tick));
        ticks.push(tick);
    }
    let rows_left_out = rows - ticks.len();
    Ok(Built {
        recording: Recording {
            version: VERSION,
            num_cars: cars as u32,
            hitbox_min_bt: Vec3A::new(low(0), low(1), low(2)),
            hitbox_max_bt: Vec3A::new(high(0), high(1), high(2)),
            ticks,
        },
        kickoffs,
        rows_left_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rows {
        frames: Vec<u8>,
        usable: Vec<u8>,
        car_floats: Vec<u8>,
        car_flags: Vec<u8>,
        ball_floats: Vec<u8>,
    }

    impl Rows {
        fn columns(&self, cars: usize) -> Columns<'_> {
            Columns {
                cars,
                frames: &self.frames,
                usable: &self.usable,
                car_floats: &self.car_floats,
                car_flags: &self.car_flags,
                ball_floats: &self.ball_floats,
            }
        }
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Two rows of one car and the ball; the second row is left out.
    fn rows(car: [f32; CAR_FLOATS], flags: [u8; CAR_FLAGS], frame: i64) -> Rows {
        let mut ball = [0.0f32; BALL_FLOATS];
        ball[6] = 1.0; // w
        Rows {
            frames: [frame, frame + 1]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            usable: vec![1, 0],
            car_floats: floats(&[car, car].concat()),
            car_flags: [flags, flags].concat(),
            ball_floats: floats(&[ball, ball].concat()),
        }
    }

    fn a_car() -> [f32; CAR_FLOATS] {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let mut car = [0.0; CAR_FLOATS];
        car[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        car[3..7].copy_from_slice(&[0.0, 0.0, s, s]); // a quarter turn about up
        (car[16], car[17], car[18], car[19]) = (0.3, 1.0, 12.0, 150.0);
        car[20] = 0.5;
        car
    }

    #[test]
    fn a_row_becomes_the_tick_the_parser_would_read() {
        let flags = [0, 1, 0, 0, 1, 0, 1, 0, 1]; // jumped, jumping, jump and handbrake held
        let made = recording(&rows(a_car(), flags, 7000).columns(1)).unwrap();
        assert_eq!(
            (
                made.recording.ticks.len(),
                made.rows_left_out,
                made.kickoffs
            ),
            (1, 1, 0)
        );
        let (tick, car) = (&made.recording.ticks[0], made.recording.ticks[0].cars[0]);
        assert_eq!(tick.frame, 7000);
        assert_eq!(car.body.pos, Vec3A::new(1.0, 2.0, 3.0));
        assert!(
            (car.body.rot.x_axis - Vec3A::Y).length() < 1e-6,
            "forward turned to +y"
        );
        assert_eq!(tick.ball.rot, Mat3A::IDENTITY);
        assert!(car.has_jumped && car.is_jumping && !car.on_ground && !car.is_flipping);
        assert_eq!((car.jump_time, car.flip_time, car.boost), (0.1, 0.0, 1.0));
        assert!(
            car.has_flip,
            "in the air within the flip window, not flipped"
        );
        assert!(car.controls.jump && car.controls.handbrake && !car.controls.boost);
        assert_eq!(car.controls.throttle, 0.5);
        let mut late = a_car();
        (late[17], late[19]) = (1.3, -5.0);
        let made = recording(&rows(late, [0, 0, 0, 0, 0, 0, 0, 0, 0], 0).columns(1)).unwrap();
        let car = made.recording.ticks[0].cars[0];
        assert!(!car.has_flip && car.jump_time == 0.0 && car.boost == 0.0);
        let flipped = [0, 1, 1, 0, 0, 1, 0, 0, 0];
        let made = recording(&rows(a_car(), flipped, 0).columns(1)).unwrap();
        let car = made.recording.ticks[0].cars[0];
        assert!(
            car.double_jumped_or_flipped && car.flip_time == 0.3,
            "a dodge keeps its flip time"
        );
    }

    #[test]
    fn wrong_sizes_and_tick_numbers_are_refused() {
        let mut short = rows(a_car(), [0; CAR_FLAGS], 0);
        short.car_flags.pop();
        assert!(
            recording(&short.columns(1))
                .unwrap_err()
                .contains("car flags")
        );
        let negative = rows(a_car(), [0; CAR_FLAGS], -3);
        assert!(
            recording(&negative.columns(1))
                .unwrap_err()
                .contains("tick number")
        );
        let mut unused = rows(a_car(), [0; CAR_FLAGS], -3);
        unused.usable = vec![0, 1];
        unused.frames = [-3i64, 5].iter().flat_map(|f| f.to_le_bytes()).collect();
        assert!(
            recording(&unused.columns(1)).is_ok(),
            "a left-out row's number is not read"
        );
    }
}
