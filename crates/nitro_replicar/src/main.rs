//! `nitro-replicar convert <replay> -o <file> --meshes <folder>`: converts one replay as `replicar convert`
//! does (the same library, version and settings, so the same states and controls), but writes only what the
//! replay copy reads (`nitro.replays.read_lean`), as plain arrays: no Parquet file of every column at zstd
//! level 9, which took about a fifth of a conversion.
//!
//! The file: `NRPL`, a u32 version, a u32 length and that much JSON (`players` as replicar's header has them,
//! `rows`, `cars`, `status_names`), then little-endian arrays, row by row: the sim tick (i64), the ball (13
//! f32: position, velocity, angular velocity, rotation), each car's floats (`CAR_FLOATS`, NaN when replicar's
//! is unknown), its small values (`CAR_SMALL` as i8: -1 unknown; the status as its index in `status_names`)
//! and its jump ticks (i64, -1 unknown).

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use replicar_format::CarStatus;
use replicar_format::record::{Car, Frame};

const VERSION: u32 = 1;
/// Per car, in this order: position, velocity, angular velocity, rotation, flip relative torque (3, 3, 3, 4,
/// 3), flip time, air time since jump, boost, then the controls' throttle, steer, pitch, yaw and roll.
const CAR_FLOATS: usize = 24;
/// Per car: status, is_on_ground, has_jumped, has_double_jumped, has_flipped, is_flipping, is_jumping,
/// is_demoed, the four wheel contacts, then the controls' jump, boost and handbrake.
const CAR_SMALL: usize = 15;

fn floats(car: Option<&Car>, out: &mut Vec<u8>) {
    let Some(c) = car else {
        for _ in 0..CAR_FLOATS {
            out.extend_from_slice(&f32::NAN.to_le_bytes());
        }
        return;
    };
    let (b, i, k) = (&c.body, &c.internals, &c.controls);
    let values = b
        .position
        .iter()
        .chain(&b.velocity)
        .chain(&b.angular_velocity)
        .chain(&b.rotation);
    let values = values.chain(&i.flip_relative_torque).chain([
        &i.flip_time,
        &i.air_time_since_jump,
        &c.boost,
    ]);
    for v in values.chain([&k.throttle, &k.steer, &k.pitch, &k.yaw, &k.roll]) {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn small(status: Option<CarStatus>, car: Option<&Car>, out: &mut Vec<u8>) {
    let code = status.and_then(|s| CarStatus::ALL.iter().position(|&a| a == s));
    out.push(code.map_or(u8::MAX, |c| c as u8));
    let Some(c) = car else {
        out.extend_from_slice(&[u8::MAX; CAR_SMALL - 1]); // -1 as i8
        return;
    };
    let (i, k) = (&c.internals, &c.controls);
    let flags = [
        i.is_on_ground,
        i.has_jumped,
        i.has_double_jumped,
        i.has_flipped,
        i.is_flipping,
        i.is_jumping,
        i.is_demoed,
    ];
    let flags =
        flags
            .into_iter()
            .chain(i.wheels_with_contact)
            .chain([k.jump, k.boost, k.handbrake]);
    out.extend(flags.map(u8::from));
}

/// The rows `replicar convert` writes by default: every simulated tick in a play segment.
fn lean(players: serde_json::Value, frames: &[Frame]) -> Vec<u8> {
    let rows: Vec<&Frame> = frames.iter().filter(|f| f.segment.is_some()).collect();
    let cars = players.as_array().map_or(0, Vec::len);
    let names: Vec<&str> = CarStatus::ALL.iter().map(|s| s.name()).collect();
    let json = serde_json::json!({"players": players, "rows": rows.len(), "cars": cars, "status_names": names});
    let json = json.to_string();
    let mut out = Vec::with_capacity(
        json.len() + rows.len() * (21 + cars * (8 + 4 * CAR_FLOATS + CAR_SMALL)),
    );
    out.extend_from_slice(b"NRPL");
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    for r in &rows {
        out.extend_from_slice(&(r.sim_tick as i64).to_le_bytes());
    }
    for r in &rows {
        let b = &r.state.ball.body;
        for v in b
            .position
            .iter()
            .chain(&b.velocity)
            .chain(&b.angular_velocity)
            .chain(&b.rotation)
        {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    let car = |r: &Frame, p: usize| r.state.cars.get(p).copied().flatten();
    for r in &rows {
        for p in 0..cars {
            floats(car(r, p).as_ref(), &mut out);
        }
    }
    for r in &rows {
        for p in 0..cars {
            small(
                r.state.car_status.get(p).copied(),
                car(r, p).as_ref(),
                &mut out,
            );
        }
    }
    for r in &rows {
        for p in 0..cars {
            let ticks = car(r, p).map_or(-1, |c| i64::from(c.internals.jump_ticks));
            out.extend_from_slice(&ticks.to_le_bytes());
        }
    }
    out
}

fn convert(replay: &Path, output: &Path, meshes: &Path) -> Result<(), String> {
    let meshes = replicar::Meshes::load(meshes).map_err(|e| e.to_string())?;
    let converter = replicar::Converter::new(&meshes, replicar::Config::default());
    let bytes = std::fs::read(replay).map_err(|e| format!("{}: {e}", replay.display()))?;
    let conversion = converter.convert(&bytes).map_err(|e| e.to_string())?;
    let header = serde_json::to_value(&conversion.header).map_err(|e| e.to_string())?;
    let data = lean(header["players"].clone(), &conversion.frames);
    let mut partial = output.as_os_str().to_owned();
    partial.push(".partial"); // as replicar names a file it is writing
    let partial = PathBuf::from(partial);
    std::fs::write(&partial, data).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, output).map_err(|e| format!("{}: {e}", output.display()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
    };
    let (Some("convert"), Some(replay), Some(output), Some(meshes)) = (
        args.first().map(String::as_str),
        args.get(1).map(PathBuf::from),
        value("-o"),
        value("--meshes"),
    ) else {
        eprintln!("usage: nitro-replicar convert <replay> -o <file> --meshes <folder>");
        return ExitCode::from(2);
    };
    match convert(&replay, &output, &meshes) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
