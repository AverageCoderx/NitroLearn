//! RLPR, RocketSim's recording format (`rocketsim_test/src/rlpr` in RocketSim's repository): the format the
//! replay converter writes (`nitro.replays`) and the replay start-state setter and the replay copy read,
//! for RocketSim's own recordings of real play as well as ours.
//!
//! A file is `"RLPR"`, one endianness byte (0 = little), the version, a sized `RecordingInfo` (car count
//! and the first car's hitbox as Bullet-unit bounds), the tick count, then per tick one sized `CarRecord`
//! per car and one sized `PhysRecord` for the ball. Records are plain C structs at fixed offsets; this
//! module reads them by hand. The replay converter writes version 6 (588-byte car records); this module
//! reads versions 2 to 9: every version shares the first 392 bytes of a car record (physics, jump and flip
//! state, boost, ball touch, controls), which is all a start state needs; the per-wheel data behind it
//! differs by version and is skipped.

use std::path::Path;

use rocketsim::{Mat3A, Vec3A};

use crate::snapshot::{Body, Controls};

pub const MAGIC: &[u8; 4] = b"RLPR";
/// The version written; versions 7–9 add fields (boost latch, handbrake value, raw touch frame) the game
/// does not expose through RLBot.
pub const VERSION: u32 = 6;
pub const PHYS_RECORD_SIZE: usize = 332;
pub const CAR_RECORD_SIZE: usize = 588;
pub const RECORDING_INFO_SIZE: usize = 28;
/// Magic 4 + endianness 1 + version 4 + sized info (4 + 28) + tick count 4.
pub const HEADER_SIZE: usize = 45;
/// Bullet units per unreal unit (RocketSim: 50 uu = 1 m).
pub const UU_TO_BT: f32 = 0.02;
pub const BT_TO_UU: f32 = 50.0;
/// Bytes of a car record before its wheels: the same in every version.
const CAR_PREFIX_SIZE: usize = 392;
const IMPULSE_RECORD_SIZE: usize = 28;
const MAX_IMPULSES: usize = 8;
const WHEEL_RECORD_SIZE: usize = 48;

/// The car record size of each version RocketSim's reader accepts.
fn car_record_size(version: u32) -> Option<usize> {
    Some(match version {
        2 => 584,
        3 => 744,
        4 => 864,
        5 => 872,
        6 => CAR_RECORD_SIZE,
        7 => 596,
        8 => 600,
        9 => 604,
        _ => return None,
    })
}

/// One car at one tick, as the format stores it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarRecord {
    pub body: Body,
    pub on_ground: bool,
    pub is_jumping: bool,
    pub is_flipping: bool,
    /// RocketSim's `jump_ticks` / 120: 1 tick when a jump starts and when its force stops, one more every
    /// tick otherwise (the game's own value in RocketSim's recordings).
    pub jump_time: f32,
    pub flip_time: f32,
    pub has_jumped: bool,
    pub double_jumped_or_flipped: bool,
    /// Whether a jump or flip is still available (RocketSim's `has_flip_or_jump`).
    pub has_flip: bool,
    pub flip_rel_torque: Vec3A,
    /// 0 to 1 (the format's scale; the simulator's is 0 to 100).
    pub boost: f32,
    pub touching_ball: bool,
    /// The controls that produced this tick.
    pub controls: Controls,
    pub touching_car: bool,
}

/// Every car and the ball at one physics frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Tick {
    pub frame: u32,
    pub cars: Vec<CarRecord>,
    pub ball: Body,
    pub ball_world_contact: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    pub version: u32,
    pub num_cars: u32,
    pub hitbox_min_bt: Vec3A,
    pub hitbox_max_bt: Vec3A,
    pub ticks: Vec<Tick>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], String> {
        if self.at + count > self.bytes.len() {
            return Err(format!(
                "the recording ends early: {} bytes needed at offset {}, {} left",
                count,
                self.at,
                self.bytes.len() - self.at
            ));
        }
        let slice = &self.bytes[self.at..self.at + count];
        self.at += count;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn bool(&mut self) -> Result<bool, String> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(format!(
                "a flag byte is {other}, not 0 or 1 (offset {})",
                self.at - 1
            )),
        }
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn vec(&mut self) -> Result<Vec3A, String> {
        Ok(Vec3A::new(self.f32()?, self.f32()?, self.f32()?))
    }

    fn skip(&mut self, count: usize) -> Result<(), String> {
        self.take(count).map(|_| ())
    }

    /// `PhysRecord` without the world-contact details and impulses.
    fn phys(&mut self) -> Result<(u32, Body, bool), String> {
        let frame = self.u32()?;
        let pos = self.vec()?;
        let mut rows = [[0.0f32; 3]; 3];
        for row in &mut rows {
            for value in row.iter_mut() {
                *value = self.f32()?;
            }
        }
        let column = |c: usize| Vec3A::new(rows[0][c], rows[1][c], rows[2][c]);
        let rot = Mat3A::from_cols(column(0), column(1), column(2));
        let vel = self.vec()?;
        let ang_vel = self.vec()?;
        let world_contact = self.bool()?;
        self.skip(3 + 24 + MAX_IMPULSES * IMPULSE_RECORD_SIZE + 4)?;
        let body = Body {
            pos,
            rot,
            vel,
            ang_vel,
        };
        Ok((frame, body, world_contact))
    }

    fn car(&mut self, version: u32) -> Result<(u32, CarRecord), String> {
        let size = car_record_size(version).expect("checked by the caller");
        let start = self.at;
        let (frame, body, _) = self.phys()?;
        let on_ground = self.bool()?;
        let is_jumping = self.bool()?;
        let is_flipping = self.bool()?;
        self.skip(1)?;
        let jump_time = self.f32()?;
        let flip_time = self.f32()?;
        let has_jumped = self.bool()?;
        let double_jumped_or_flipped = self.bool()?;
        let has_flip = self.bool()?;
        self.skip(1)?;
        let flip_rel_torque = self.vec()?;
        let boost = self.f32()?;
        let touching_ball = self.bool()?;
        self.skip(3)?;
        let (throttle, steer, pitch, yaw, roll) = (
            self.f32()?,
            self.f32()?,
            self.f32()?,
            self.f32()?,
            self.f32()?,
        );
        let (jump, boost_held, handbrake) = (self.bool()?, self.bool()?, self.bool()?);
        self.skip(1)?;
        debug_assert_eq!(self.at - start, CAR_PREFIX_SIZE);
        // Wheels (version-dependent stride), then the car–car contact flag from version 6 on.
        let touching_car = if version >= 6 {
            self.skip(4 * WHEEL_RECORD_SIZE)?;
            self.bool()?
        } else {
            false
        };
        self.at = start + size;
        if self.at > self.bytes.len() {
            return Err("the recording ends inside a car record".to_owned());
        }
        Ok((
            frame,
            CarRecord {
                body,
                on_ground,
                is_jumping,
                is_flipping,
                jump_time,
                flip_time,
                has_jumped,
                double_jumped_or_flipped,
                has_flip,
                flip_rel_torque,
                boost,
                touching_ball,
                controls: Controls {
                    throttle,
                    steer,
                    pitch,
                    yaw,
                    roll,
                    jump,
                    boost: boost_held,
                    handbrake,
                },
                touching_car,
            },
        ))
    }
}

/// Parses a recording (any version 2 to 9, little-endian).
pub fn read(bytes: &[u8]) -> Result<Recording, String> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != MAGIC {
        return Err("not an RLPR recording (wrong magic bytes)".to_owned());
    }
    if r.u8()? != 0 {
        return Err("a big-endian recording; only little-endian ones are read".to_owned());
    }
    let version = r.u32()?;
    let Some(car_size) = car_record_size(version) else {
        return Err(format!(
            "RLPR version {version} is not one RocketSim's reader knows (2 to 9)"
        ));
    };
    if r.u32()? as usize != RECORDING_INFO_SIZE {
        return Err("the recording info block has an unexpected size".to_owned());
    }
    let num_cars = r.u32()?;
    let hitbox_min_bt = r.vec()?;
    let hitbox_max_bt = r.vec()?;
    let num_ticks = r.u32()?;
    let mut ticks = Vec::with_capacity(num_ticks as usize);
    for _ in 0..num_ticks {
        let mut cars = Vec::with_capacity(num_cars as usize);
        let mut frame = None;
        loop {
            let size = r.u32()? as usize;
            if size == PHYS_RECORD_SIZE {
                let (ball_frame, ball, ball_world_contact) = r.phys()?;
                ticks.push(Tick {
                    frame: frame.unwrap_or(ball_frame),
                    cars,
                    ball,
                    ball_world_contact,
                });
                break;
            }
            if size != car_size {
                return Err(format!(
                    "a record of {size} bytes is neither a car ({car_size}) nor the ball ({PHYS_RECORD_SIZE})"
                ));
            }
            let (car_frame, car) = r.car(version)?;
            frame.get_or_insert(car_frame);
            cars.push(car);
        }
    }
    if r.at != bytes.len() {
        return Err(format!(
            "{} bytes follow the last tick (a cut-off recording? the tick count says {num_ticks})",
            bytes.len() - r.at
        ));
    }
    Ok(Recording {
        version,
        num_cars,
        hitbox_min_bt,
        hitbox_max_bt,
        ticks,
    })
}

/// Reads a plain `.rlpr` file or a zstd-compressed `.rlpr.zst` one (RocketSim's bundled recordings).
pub fn read_file(path: &Path) -> Result<Recording, String> {
    read(&read_bytes(path)?).map_err(|err| format!("{}: {err}", path.display()))
}

/// The bytes of a plain `.rlpr` file, or of a `.rlpr.zst` one decompressed.
pub fn read_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let raw =
        std::fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    Ok(if path.extension().is_some_and(|ext| ext == "zst") {
        // Allow the largest window (31 on 64-bit): recordings compressed with `zstd --long` need it, as
        // RocketSim's own reader allows.
        let decompress = || -> std::io::Result<Vec<u8>> {
            let mut decoder = zstd::stream::Decoder::new(raw.as_slice())?;
            decoder.window_log_max(if cfg!(target_pointer_width = "64") {
                31
            } else {
                30
            })?;
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut out)?;
            Ok(out)
        };
        decompress().map_err(|err| format!("cannot decompress {}: {err}", path.display()))?
    } else {
        raw
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the tick count sits, so a writer can patch it when the recording ends.
    const NUM_TICKS_OFFSET: usize = 41;

    // A version-6 writer, to check the reader against what RocketSim's own reader makes of the same bytes.

    fn put_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn put_f32(buf: &mut Vec<u8>, value: f32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn put_vec(buf: &mut Vec<u8>, v: Vec3A) {
        put_f32(buf, v.x);
        put_f32(buf, v.y);
        put_f32(buf, v.z);
    }

    fn put_bool(buf: &mut Vec<u8>, value: bool) {
        buf.push(u8::from(value));
    }

    fn put_zeros(buf: &mut Vec<u8>, count: usize) {
        buf.resize(buf.len() + count, 0);
    }

    /// The file header with a tick count of 0; patch `NUM_TICKS_OFFSET` when the recording ends.
    fn write_header(buf: &mut Vec<u8>, num_cars: u32, hitbox_min_bt: Vec3A, hitbox_max_bt: Vec3A) {
        let start = buf.len();
        buf.extend_from_slice(MAGIC);
        buf.push(0); // little-endian
        put_u32(buf, VERSION);
        put_u32(buf, RECORDING_INFO_SIZE as u32);
        put_u32(buf, num_cars);
        put_vec(buf, hitbox_min_bt);
        put_vec(buf, hitbox_max_bt);
        put_u32(buf, 0);
        debug_assert_eq!(buf.len() - start, HEADER_SIZE);
    }

    /// `PhysRecord`: frame, position, the rotation as three rows (column 0 forward, 1 right, 2 up), the
    /// velocities, a world-contact flag (no point or normal), no impulse records.
    fn put_phys(buf: &mut Vec<u8>, frame: u32, body: &Body, world_contact: bool) {
        let start = buf.len();
        put_u32(buf, frame);
        put_vec(buf, body.pos);
        for row in 0..3 {
            for col in 0..3 {
                put_f32(buf, body.rot.col(col)[row]);
            }
        }
        put_vec(buf, body.vel);
        put_vec(buf, body.ang_vel);
        put_bool(buf, world_contact);
        put_zeros(buf, 3);
        put_zeros(buf, 24); // contact point and normal: unknown
        put_zeros(buf, MAX_IMPULSES * IMPULSE_RECORD_SIZE);
        put_u32(buf, 0); // impulse record count
        debug_assert_eq!(buf.len() - start, PHYS_RECORD_SIZE);
    }

    fn put_car(buf: &mut Vec<u8>, frame: u32, car: &CarRecord) {
        let start = buf.len();
        put_phys(buf, frame, &car.body, false);
        put_bool(buf, car.on_ground);
        put_bool(buf, car.is_jumping);
        put_bool(buf, car.is_flipping);
        put_zeros(buf, 1);
        put_f32(buf, car.jump_time);
        put_f32(buf, car.flip_time);
        put_bool(buf, car.has_jumped);
        put_bool(buf, car.double_jumped_or_flipped);
        put_bool(buf, car.has_flip);
        put_zeros(buf, 1);
        put_vec(buf, car.flip_rel_torque);
        put_f32(buf, car.boost);
        put_bool(buf, car.touching_ball);
        put_zeros(buf, 3);
        let c = car.controls;
        for value in [c.throttle, c.steer, c.pitch, c.yaw, c.roll] {
            put_f32(buf, value);
        }
        put_bool(buf, c.jump);
        put_bool(buf, c.boost);
        put_bool(buf, c.handbrake);
        put_zeros(buf, 1);
        debug_assert_eq!(buf.len() - start, CAR_PREFIX_SIZE);
        // Wheels: contact follows the car, the normal is its up direction; suspension and forces unknown.
        for _ in 0..4 {
            put_zeros(buf, 8);
            put_bool(buf, car.on_ground);
            put_zeros(buf, 3);
            put_vec(
                buf,
                if car.on_ground {
                    car.body.up()
                } else {
                    Vec3A::ZERO
                },
            );
            put_zeros(buf, 24);
        }
        put_bool(buf, car.touching_car);
        put_zeros(buf, 3);
        debug_assert_eq!(buf.len() - start, CAR_RECORD_SIZE);
    }

    /// One tick: every car's sized record in order, then the ball's.
    fn write_tick(buf: &mut Vec<u8>, tick: &Tick) {
        for car in &tick.cars {
            put_u32(buf, CAR_RECORD_SIZE as u32);
            put_car(buf, tick.frame, car);
        }
        put_u32(buf, PHYS_RECORD_SIZE as u32);
        put_phys(buf, tick.frame, &tick.ball, tick.ball_world_contact);
    }

    /// Bytes per tick of a version-6 recording with `num_cars` cars.
    fn tick_size(num_cars: usize) -> usize {
        num_cars * (4 + CAR_RECORD_SIZE) + 4 + PHYS_RECORD_SIZE
    }

    fn car(seed: f32, on_ground: bool) -> CarRecord {
        CarRecord {
            body: Body {
                pos: Vec3A::new(seed, -2.0 * seed, 17.0 + seed),
                rot: Mat3A::from_rotation_z(0.3 * seed) * Mat3A::from_rotation_y(0.1),
                vel: Vec3A::new(100.0, 200.0, seed),
                ang_vel: Vec3A::new(0.1, -0.2, 0.3),
            },
            on_ground,
            is_jumping: !on_ground,
            is_flipping: false,
            jump_time: if on_ground { 0.0 } else { 3.0 / 120.0 },
            flip_time: 0.0,
            has_jumped: !on_ground,
            double_jumped_or_flipped: false,
            has_flip: true,
            flip_rel_torque: Vec3A::new(0.0, 0.6, 0.0),
            boost: 0.33,
            touching_ball: seed > 1.0,
            controls: Controls {
                throttle: 1.0,
                steer: -0.5,
                pitch: 0.0,
                yaw: 0.25,
                roll: 0.0,
                jump: true,
                boost: false,
                handbrake: true,
            },
            touching_car: false,
        }
    }

    fn recording_bytes(ticks: &[Tick]) -> Vec<u8> {
        let mut buf = Vec::new();
        write_header(
            &mut buf,
            ticks[0].cars.len() as u32,
            Vec3A::new(-0.927561, -0.866994, 0.028509),
            Vec3A::new(1.482587, 0.866994, 0.801690),
        );
        for tick in ticks {
            write_tick(&mut buf, tick);
        }
        let count = (ticks.len() as u32).to_le_bytes();
        buf[NUM_TICKS_OFFSET..NUM_TICKS_OFFSET + 4].copy_from_slice(&count);
        buf
    }

    #[test]
    fn a_written_recording_reads_back_exactly() {
        let ticks: Vec<Tick> = (0..5)
            .map(|i| Tick {
                frame: 100 + i,
                cars: vec![car(i as f32, i % 2 == 0), car(i as f32 + 0.5, true)],
                ball: Body {
                    pos: Vec3A::new(0.0, 10.0 * i as f32, 93.15),
                    rot: Mat3A::IDENTITY,
                    vel: Vec3A::new(0.0, 500.0, 0.0),
                    ang_vel: Vec3A::ZERO,
                },
                ball_world_contact: i == 0,
            })
            .collect();
        let bytes = recording_bytes(&ticks);
        assert_eq!(bytes.len(), HEADER_SIZE + 5 * tick_size(2));
        let back = read(&bytes).unwrap();
        assert_eq!(back.version, VERSION);
        assert_eq!(back.num_cars, 2);
        let size_uu = (back.hitbox_max_bt - back.hitbox_min_bt) * BT_TO_UU;
        assert!(
            (size_uu - Vec3A::new(120.507, 86.6994, 38.6591))
                .abs()
                .max_element()
                < 0.01
        );
        assert_eq!(back.ticks, ticks);
    }

    /// The recording behind the golden files: two cars and the ball over four ticks.
    fn golden_ticks() -> Vec<Tick> {
        (0..4)
            .map(|i| Tick {
                frame: 40 + i,
                cars: vec![car(i as f32, i % 2 == 1), car(i as f32 + 0.5, true)],
                ball: Body {
                    pos: Vec3A::new(5.0 * i as f32, 20.0, 93.15 + i as f32),
                    rot: Mat3A::IDENTITY,
                    vel: Vec3A::new(0.0, -300.0, 10.0),
                    ang_vel: Vec3A::new(0.5, 0.0, 0.0),
                },
                ball_world_contact: i == 2,
            })
            .collect()
    }

    fn vec_text(v: Vec3A) -> String {
        format!("{:?} {:?} {:?}", v.x, v.y, v.z)
    }

    /// What a reader got from a recording, one `name: value` line per value (the layout of
    /// `testdata/rlpr_written_v6_read.txt`). Floats print in Rust's shortest exact form, so equal text means
    /// equal values.
    fn read_back_text(recording: &Recording) -> String {
        let mut out = format!(
            "version: {}\ncars: {}\nhitbox_min_bt: {}\nhitbox_max_bt: {}\n",
            recording.version,
            recording.num_cars,
            vec_text(recording.hitbox_min_bt),
            vec_text(recording.hitbox_max_bt)
        );
        let body = |b: &Body| {
            vec![
                ("pos", vec_text(b.pos)),
                (
                    "rot columns",
                    [0, 1, 2].map(|c| vec_text(b.rot.col(c))).join(" | "),
                ),
                ("vel", vec_text(b.vel)),
                ("ang_vel", vec_text(b.ang_vel)),
            ]
        };
        for (i, tick) in recording.ticks.iter().enumerate() {
            for (j, car) in tick.cars.iter().enumerate() {
                let c = car.controls;
                let mut fields = vec![("frame", tick.frame.to_string())];
                fields.extend(body(&car.body));
                fields.extend([
                    ("on_ground", car.on_ground.to_string()),
                    ("is_jumping", car.is_jumping.to_string()),
                    ("is_flipping", car.is_flipping.to_string()),
                    ("jump_time", format!("{:?}", car.jump_time)),
                    ("flip_time", format!("{:?}", car.flip_time)),
                    ("has_jumped", car.has_jumped.to_string()),
                    (
                        "double_jumped_or_flipped",
                        car.double_jumped_or_flipped.to_string(),
                    ),
                    ("has_flip", car.has_flip.to_string()),
                    ("flip_rel_torque", vec_text(car.flip_rel_torque)),
                    ("boost", format!("{:?}", car.boost)),
                    ("touching_ball", car.touching_ball.to_string()),
                    (
                        "controls",
                        format!(
                            "{:?} {:?} {:?} {:?} {:?} {} {} {}",
                            c.throttle,
                            c.steer,
                            c.pitch,
                            c.yaw,
                            c.roll,
                            c.jump,
                            c.boost,
                            c.handbrake
                        ),
                    ),
                    ("touching_car", car.touching_car.to_string()),
                ]);
                for (name, value) in fields {
                    out += &format!("tick {i} car {j} {name}: {value}\n");
                }
            }
            let mut ball = vec![("frame", tick.frame.to_string())];
            ball.extend(body(&tick.ball));
            ball.push(("world_contact", tick.ball_world_contact.to_string()));
            for (name, value) in ball {
                out += &format!("tick {i} ball {name}: {value}\n");
            }
        }
        out
    }

    /// What this module writes is what RocketSim's own reader reads. `testdata/rlpr_written_v6.bin` holds the
    /// bytes this writer makes of `golden_ticks()`; RocketSim's reader (`rocketsim_test::rlpr::Recording::
    /// from_bytes` at RocketSim commit 45722876705adf5dc673aa764e80382cfe8c3312, the commit crates.io's
    /// rocketsim 0.3.3 was published from) accepted them and read back the values in
    /// `testdata/rlpr_written_v6_read.txt`. The writer must still make exactly those bytes, and this module's
    /// reader must read back exactly those values.
    #[test]
    fn written_recordings_read_as_rocketsims_reader_read_them() {
        let golden: &[u8] = include_bytes!("../testdata/rlpr_written_v6.bin");
        assert!(
            recording_bytes(&golden_ticks()) == golden,
            "the writer no longer makes the bytes of testdata/rlpr_written_v6.bin"
        );
        let recording = read(golden).unwrap();
        let expected = include_str!("../testdata/rlpr_written_v6_read.txt").replace("\r\n", "\n");
        assert_eq!(read_back_text(&recording), expected);
        assert_eq!(recording.ticks, golden_ticks());
    }

    #[test]
    fn damaged_recordings_are_refused_with_a_reason() {
        let tick = Tick {
            frame: 1,
            cars: vec![car(1.0, true)],
            ball: Body::RESTING,
            ball_world_contact: true,
        };
        let bytes = recording_bytes(&[tick]);
        assert!(
            read(&bytes[..bytes.len() - 10])
                .unwrap_err()
                .contains("ends early")
        );
        let mut extra = bytes.clone();
        extra.extend_from_slice(&[0; 7]);
        assert!(read(&extra).unwrap_err().contains("follow the last tick"));
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert!(read(&wrong).unwrap_err().contains("magic"));
        let mut future = bytes.clone();
        future[5..9].copy_from_slice(&42u32.to_le_bytes());
        assert!(read(&future).unwrap_err().contains("version 42"));
        let mut flag = bytes;
        flag[HEADER_SIZE + 4 + 332] = 7; // the first car's on_ground byte
        assert!(read(&flag).unwrap_err().contains("not 0 or 1"));
    }
}
