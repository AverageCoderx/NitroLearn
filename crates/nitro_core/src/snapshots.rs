//! Recorded moments to start games from: one car and the ball, with the car's jump and flip state, in a
//! CSV read by column name (`COLUMNS`), placed on one car with the others parked away from the ball, and
//! recorded back from a running game (`record`), as Blue sees the field.

use rocketsim::{Mat3A, Vec3A};

use crate::setters::{MIN_CAR_GAP, PLACEMENT_ATTEMPTS, parked, random_field_point};
use crate::sim::Sim;
use crate::snapshot::{Body, CarDetail, CarMoves, CarSnapshot};

/// One recorded moment: the car (body, boost, on its wheels, jump and flip state) and the ball.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    pub car: Body,
    pub boost: f32,
    pub on_ground: bool,
    pub moves: CarMoves,
    pub ball: Body,
}

/// A moment as a CSV line holds it, with its label (the `group` column, if any).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub group: String,
    pub snapshot: Snapshot,
}

const CAR_REST_Z: f32 = 17.0;
/// Other cars start at least this far from the ball, out of the way of the move.
pub const CLEAR_OF_BALL: f32 = 2000.0;

/// The columns of a moment, in the order `to_values` writes them.
pub const COLUMNS: [&str; 41] = [
    "cx",
    "cy",
    "cz",
    "cvx",
    "cvy",
    "cvz",
    "cwx",
    "cwy",
    "cwz",
    "fx",
    "fy",
    "fz",
    "rx",
    "ry",
    "rz",
    "ux",
    "uy",
    "uz",
    "onGround",
    "hasJumped",
    "hasDoubleJumped",
    "hasFlipped",
    "isJumping",
    "jumpTime",
    "isFlipping",
    "flipTime",
    "ftx",
    "fty",
    "ftz",
    "airTime",
    "airTimeSinceJump",
    "boost",
    "bx",
    "by",
    "bz",
    "bvx",
    "bvy",
    "bvz",
    "bwx",
    "bwy",
    "bwz",
];

/// Reads a snapshot CSV (header line first, columns by name, extra columns ignored).
pub fn parse_csv(text: &str) -> Result<Vec<Row>, String> {
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    let (_, header) = lines.next().ok_or("the file is empty")?;
    let names: Vec<&str> = header.split(',').map(str::trim).collect();
    let find = |name: &str| names.iter().position(|n| *n == name);
    let columns = COLUMNS
        .iter()
        .map(|&name| find(name).ok_or(format!("no column '{name}' in the header")))
        .collect::<Result<Vec<_>, _>>()?;
    let group = find("group");
    let mut rows = Vec::new();
    for (number, line) in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let cell = |at: usize| -> Result<&str, String> {
            cells
                .get(at)
                .copied()
                .ok_or(format!("line {}: too few columns", number + 1))
        };
        let number_at = |at: usize| -> Result<f32, String> {
            let text = cell(at)?;
            let value: f32 = text
                .parse()
                .map_err(|_| format!("line {}: '{text}' is not a number", number + 1))?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!(
                    "line {}: '{text}' is not a finite number",
                    number + 1
                ))
            }
        };
        let v = columns
            .iter()
            .map(|&at| number_at(at))
            .collect::<Result<Vec<f32>, _>>()?;
        rows.push(Row {
            group: group.map_or(Ok(""), cell)?.to_owned(),
            snapshot: from_values(&v),
        });
    }
    Ok(rows)
}

/// Room a moment may take beyond the field's walls and ceiling (uu): the goals' depth and a little more.
const FIELD_X: f32 = 4200.0;
const FIELD_Y: f32 = 6100.0;
const FIELD_Z: (f32, f32) = (-50.0, 2150.0);
/// Speeds (uu/s) and spins (rad/s) well over the game's caps (a car on the ground is read spinning past
/// its 5.5 rad/s cap): only numbers no game holds are refused.
const CAR_SPEED: f32 = 3000.0;
const BALL_SPEED: f32 = 7000.0;
const SPIN: f32 = 50.0;

/// What makes `s` a moment the game could not be in, if anything: a car or ball off the field or faster
/// than the game allows, an orientation that is not one (RocketSim's physics turns it into NaN or stops),
/// boost outside 0 to 100.
pub fn problem(s: &Snapshot) -> Option<String> {
    let on_field = |p: Vec3A| {
        p.x.abs() <= FIELD_X && p.y.abs() <= FIELD_Y && (FIELD_Z.0..=FIELD_Z.1).contains(&p.z)
    };
    let body = |what: &str, b: &Body, speed: f32| {
        if !on_field(b.pos) {
            Some(format!("the {what} is off the field at {}", b.pos))
        } else if b.vel.length() > speed || b.ang_vel.length() > SPIN {
            Some(format!("the {what} moves faster than the game allows"))
        } else {
            None
        }
    };
    let axes = [s.car.rot.x_axis, s.car.rot.y_axis, s.car.rot.z_axis];
    let unit = axes.iter().all(|a| (a.length() - 1.0).abs() < 0.01);
    let square = (0..3).all(|i| axes[i].dot(axes[(i + 1) % 3]).abs() < 0.01);
    if !(unit && square) {
        return Some(
            "the car's orientation (fx..uz) is not three unit directions at right angles"
                .to_owned(),
        );
    }
    if !(0.0..=100.0).contains(&s.boost) {
        return Some(format!("the car's boost is {}, not 0 to 100", s.boost));
    }
    body("car", &s.car, CAR_SPEED).or_else(|| body("ball", &s.ball, BALL_SPEED))
}

/// A moment from its 41 numbers in `COLUMNS` order.
pub fn from_values(v: &[f32]) -> Snapshot {
    let at = |i: usize| Vec3A::new(v[i], v[i + 1], v[i + 2]);
    Snapshot {
        car: Body {
            pos: at(0),
            vel: at(3),
            ang_vel: at(6),
            rot: Mat3A::from_cols(at(9), at(12), at(15)),
        },
        on_ground: v[18] != 0.0,
        moves: CarMoves {
            has_jumped: v[19] != 0.0,
            has_double_jumped: v[20] != 0.0,
            has_flipped: v[21] != 0.0,
            is_jumping: v[22] != 0.0,
            jump_time: v[23],
            is_flipping: v[24] != 0.0,
            flip_time: v[25],
            flip_rel_torque: at(26),
            air_time: v[29],
            air_time_since_jump: v[30],
        },
        boost: v[31],
        ball: Body {
            pos: at(32),
            vel: at(35),
            ang_vel: at(38),
            rot: Mat3A::IDENTITY,
        },
    }
}

/// The 41 numbers of a moment, in `COLUMNS` order.
pub fn to_values(s: &Snapshot) -> [f32; 41] {
    let flag = |b: bool| f32::from(u8::from(b));
    let (c, m, b) = (&s.car, &s.moves, &s.ball);
    let mut out = [0.0; 41];
    let mut put = |at: usize, values: &[f32]| out[at..at + values.len()].copy_from_slice(values);
    put(0, &c.pos.to_array());
    put(3, &c.vel.to_array());
    put(6, &c.ang_vel.to_array());
    put(9, &c.rot.x_axis.to_array());
    put(12, &c.rot.y_axis.to_array());
    put(15, &c.rot.z_axis.to_array());
    put(
        18,
        &[
            flag(s.on_ground),
            flag(m.has_jumped),
            flag(m.has_double_jumped),
            flag(m.has_flipped),
            flag(m.is_jumping),
            m.jump_time,
            flag(m.is_flipping),
            m.flip_time,
        ],
    );
    put(26, &m.flip_rel_torque.to_array());
    put(29, &[m.air_time, m.air_time_since_jump, s.boost]);
    put(32, &b.pos.to_array());
    put(35, &b.vel.to_array());
    put(38, &b.ang_vel.to_array());
    out
}

/// The moment car `car` is in now (as Blue sees it: an Orange car's moment is turned around).
pub fn record(car: &CarSnapshot, detail: &CarDetail, ball: Body) -> Snapshot {
    let snapshot = Snapshot {
        car: car.body,
        boost: car.boost,
        on_ground: car.on_ground,
        moves: CarMoves {
            has_jumped: car.has_jumped,
            has_double_jumped: car.has_double_jumped,
            has_flipped: car.has_flipped,
            is_jumping: detail.is_jumping,
            is_flipping: detail.is_flipping,
            jump_time: detail.jump_time,
            flip_time: detail.flip_time,
            flip_rel_torque: detail.flip_rel_torque,
            air_time: detail.air_time,
            air_time_since_jump: detail.air_time_since_jump,
        },
        ball,
    };
    if car.team == crate::snapshot::Team::Blue {
        snapshot
    } else {
        turned_around(&snapshot)
    }
}

/// The moment rotated 180° about the vertical axis (Orange's view of a Blue moment and back).
pub fn turned_around(s: &Snapshot) -> Snapshot {
    let flip = Vec3A::new(-1.0, -1.0, 1.0);
    let body = |b: &Body| Body {
        pos: b.pos * flip,
        vel: b.vel * flip,
        ang_vel: b.ang_vel * flip,
        rot: Mat3A::from_cols(
            b.rot.x_axis * flip,
            b.rot.y_axis * flip,
            b.rot.z_axis * flip,
        ),
    };
    Snapshot {
        car: body(&s.car),
        ball: body(&s.ball),
        ..*s
    }
}

/// Gives the moment to car `car` (its jump and flip state included) and parks every other car at rest,
/// at least `CLEAR_OF_BALL` from the ball where there is room.
pub fn place(sim: &mut Sim, rng: &mut fastrand::Rng, start: &Snapshot, car: usize) {
    place_with_room(sim, rng, start, car, MIN_CAR_GAP);
}

/// `place`, with the other cars also parked at least `room` from car `car` where there is room (a start
/// that drives over a boost pad keeps them off it).
pub fn place_with_room(
    sim: &mut Sim,
    rng: &mut fastrand::Rng,
    start: &Snapshot,
    car: usize,
    room: f32,
) {
    let cars = sim.num_cars();
    let mut placed: Vec<Option<(Body, f32, bool)>> = vec![None; cars];
    placed[car] = Some((start.car, start.boost, start.on_ground));
    for slot in 0..cars {
        if placed[slot].is_some() {
            continue;
        }
        let clear =
            |p: Vec3A, from_ball: f32, from_car: f32, placed: &[Option<(Body, f32, bool)>]| {
                p.distance(start.ball.pos) > from_ball
                    && p.distance(start.car.pos) > from_car
                    && placed
                        .iter()
                        .flatten()
                        .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
            };
        let pos = (0..PLACEMENT_ATTEMPTS)
            .map(|_| random_field_point(rng, CAR_REST_Z, CAR_REST_Z))
            .find(|&p| clear(p, CLEAR_OF_BALL, room, &placed))
            .or_else(|| {
                (0..PLACEMENT_ATTEMPTS)
                    .map(|_| random_field_point(rng, CAR_REST_Z, CAR_REST_Z))
                    .find(|&p| clear(p, MIN_CAR_GAP, MIN_CAR_GAP, &placed))
            })
            .expect("no free spot for a car (field too crowded)");
        placed[slot] = Some((parked(pos, rng), rng.f32() * 100.0, true));
    }
    let placed: Vec<(Body, f32, bool)> = placed.into_iter().flatten().collect();
    sim.place(start.ball, &placed);
    sim.set_moves(car, &start.moves);
}

/// A random Blue car (Blue cars are the even ones; Blue is the learning side against past versions).
pub fn random_blue_car(sim: &Sim, rng: &mut fastrand::Rng) -> usize {
    2 * rng.usize(..sim.num_cars().div_ceil(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moment_reads_back_from_its_numbers_and_turns_around_twice_to_itself() {
        let values: Vec<f32> = (0..41).map(|i| i as f32 * 0.5 - 3.0).collect();
        let moment = from_values(&values);
        let back = to_values(&moment);
        for (i, (&a, &b)) in values.iter().zip(&back).enumerate() {
            let flag = (18..=22).contains(&i) || i == 24;
            assert!(
                if flag {
                    (a != 0.0) == (b != 0.0)
                } else {
                    a == b
                },
                "{}",
                COLUMNS[i]
            );
        }
        assert_eq!(turned_around(&turned_around(&moment)), moment);
        let text = format!(
            "group,{}\npractice,{}\n",
            COLUMNS.join(","),
            back.map(|v| v.to_string()).join(",")
        );
        let rows = parse_csv(&text).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].group, "practice");
        assert_eq!(to_values(&rows[0].snapshot), back);
        assert!(parse_csv("cx,cy\n1,2\n").unwrap_err().contains("no column"));
        let bad = text.replace("practice,-3", "practice,NaN");
        assert!(parse_csv(&bad).unwrap_err().contains("not a"));
    }

    /// A car at rest on its wheels facing +Y, the ball resting ahead of it.
    fn resting() -> Snapshot {
        let mut v = [0.0f32; 41];
        v[1] = -1000.0;
        v[2] = 17.0;
        (v[9], v[13], v[17]) = (1.0, 1.0, 1.0);
        v[18] = 1.0;
        v[31] = 50.0;
        v[34] = 93.15;
        from_values(&v)
    }

    #[test]
    fn a_moment_the_game_could_not_be_in_is_named() {
        assert_eq!(problem(&resting()), None);
        let changed = |change: fn(&mut Snapshot)| {
            let mut moment = resting();
            change(&mut moment);
            problem(&moment).expect("refused")
        };
        // A car turned by hand-typed numbers (unit rows, not quite at right angles) is fine.
        let mut tilted = resting();
        tilted.car.rot = Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert_eq!(problem(&tilted), None);
        assert!(changed(|m| m.car.rot = Mat3A::ZERO).contains("orientation"));
        assert!(changed(|m| m.car.rot.x_axis = Vec3A::new(0.7, 0.7, 0.0)).contains("orientation"));
        assert!(changed(|m| m.car.pos.x = 1e20).contains("car"));
        assert!(changed(|m| m.ball.pos.y = -7000.0).contains("ball"));
        assert!(changed(|m| m.car.vel.y = 1e20).contains("car"));
        assert!(changed(|m| m.ball.ang_vel.z = 1e9).contains("ball"));
        assert!(changed(|m| m.boost = 250.0).contains("boost"));
    }

    #[test]
    fn a_moments_file_with_a_moment_the_game_could_not_be_in_is_refused_at_the_start() {
        let line = |s: &Snapshot| to_values(s).map(|v| v.to_string()).join(",");
        let mut far = resting();
        far.car.pos.x = 1e20;
        let file = std::env::temp_dir().join(format!("nitro_moments_{}.csv", std::process::id()));
        std::fs::write(
            &file,
            format!(
                "{}\n{}\n{}\n",
                COLUMNS.join(","),
                line(&resting()),
                line(&far)
            ),
        )
        .unwrap();
        let setters = vec![crate::setters::WeightedSetter {
            kind: crate::setters::SetterKind::Snapshots {
                file: file.to_string_lossy().into_owned(),
            },
            weight: 1.0,
        }];
        let made = crate::setters::SetterMix::new(setters, 1, true);
        std::fs::remove_file(&file).ok();
        let err = made.err().expect("refused");
        assert!(
            err.contains("moment 2") && err.contains("off the field"),
            "{err}"
        );
    }
}
