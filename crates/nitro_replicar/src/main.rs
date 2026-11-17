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
//!
//! `--fit-timing` adds the timing fit (`fit`) after replicar's conversion; the header then
//! carries `timing_fit` (the fit's version and what it did). Without it the file is as before, byte for byte.
//!
//! `nitro-replicar measure <replay> --meshes <folder> [--plain <file>] [--fitted <file>]` converts once, fits a
//! copy, and prints JSON: the conversion's and the fit's seconds, the fit's report and the car error at 20 ticks
//! (`measure`) of each window before and after; `--plain` and `--fitted` also write both files.

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod car;
mod fit;
mod measure;
#[cfg(test)]
mod tests;
mod track;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

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
fn lean(
    players: serde_json::Value,
    frames: &[Frame],
    timing_fit: Option<serde_json::Value>,
) -> Vec<u8> {
    let rows: Vec<&Frame> = frames.iter().filter(|f| f.segment.is_some()).collect();
    let cars = players.as_array().map_or(0, Vec::len);
    let names: Vec<&str> = CarStatus::ALL.iter().map(|s| s.name()).collect();
    let mut json = serde_json::json!({"players": players, "rows": rows.len(), "cars": cars, "status_names": names});
    if let Some(report) = timing_fit {
        json["timing_fit"] = report;
    }
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

/// A replay converted by replicar: its header's players (as JSON and as replicar's) and its frames.
struct Converted {
    players_json: serde_json::Value,
    players: Vec<replicar_format::header::PlayerInfo>,
    frames: Vec<Frame>,
}

fn convert(replay: &Path, meshes: &Path) -> Result<Converted, String> {
    let meshes = replicar::Meshes::load(meshes).map_err(|e| e.to_string())?;
    let converter = replicar::Converter::new(&meshes, replicar::Config::default());
    let bytes = std::fs::read(replay).map_err(|e| format!("{}: {e}", replay.display()))?;
    let conversion = converter.convert(&bytes).map_err(|e| e.to_string())?;
    let header = serde_json::to_value(&conversion.header).map_err(|e| e.to_string())?;
    Ok(Converted {
        players_json: header["players"].clone(),
        players: conversion.header.players.clone(),
        frames: conversion.frames,
    })
}

fn write(output: &Path, data: &[u8]) -> Result<(), String> {
    let mut partial = output.as_os_str().to_owned();
    partial.push(".partial"); // as replicar names a file it is writing
    let partial = PathBuf::from(partial);
    std::fs::write(&partial, data).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, output).map_err(|e| format!("{}: {e}", output.display()))
}

fn run_convert(
    replay: &Path,
    output: &Path,
    meshes: &Path,
    fit_timing: bool,
) -> Result<(), String> {
    let mut c = convert(replay, meshes)?;
    let report = if fit_timing {
        Some(fit::fit(&c.players, &mut c.frames, fit::Settings::DEFAULT)?.json())
    } else {
        None
    };
    write(output, &lean(c.players_json, &c.frames, report))
}

fn run_measure(
    replay: &Path,
    meshes: &Path,
    plain: Option<&Path>,
    fitted: Option<&Path>,
) -> Result<(), String> {
    let began = Instant::now();
    let c = convert(replay, meshes)?;
    let convert_seconds = began.elapsed().as_secs_f64();
    let mut frames = c.frames.clone();
    let began = Instant::now();
    let report = fit::fit(&c.players, &mut frames, fit::Settings::DEFAULT)?;
    let fit_seconds = began.elapsed().as_secs_f64();
    let windows = measure::windows(&c.frames, c.players.len(), 20);
    let round = |e: Vec<Option<f32>>| -> Vec<Option<f64>> {
        e.into_iter()
            .map(|v| v.map(|v| (f64::from(v) * 1000.0).round() / 1000.0))
            .collect()
    };
    let before = round(measure::errors(&c.players, &c.frames, &windows)?);
    let after = round(measure::errors(&c.players, &frames, &windows)?);
    let teams = [0u8, 1].map(|t| c.players.iter().filter(|p| p.team == t).count());
    let out = serde_json::json!({
        "cars": c.players.len(),
        "teams": teams,
        "convert_seconds": convert_seconds,
        "fit_seconds": fit_seconds,
        "fit": report.json(),
        "before": before,
        "after": after,
    });
    if let Some(path) = plain {
        write(path, &lean(c.players_json.clone(), &c.frames, None))?;
    }
    if let Some(path) = fitted {
        write(path, &lean(c.players_json, &frames, Some(report.json())))?;
    }
    println!("{out}");
    Ok(())
}

const USAGE: &str = "usage: nitro-replicar convert <replay> -o <file> --meshes <folder> [--fit-timing]\n       \
                     nitro-replicar measure <replay> --meshes <folder> [--plain <file>] [--fitted <file>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
    };
    let fit_timing = args.iter().any(|a| a == "--fit-timing");
    let result = match (
        args.first().map(String::as_str),
        args.get(1).map(PathBuf::from),
        value("--meshes"),
    ) {
        (Some("convert"), Some(replay), Some(meshes)) => match value("-o") {
            Some(output) => run_convert(&replay, &output, &meshes, fit_timing),
            None => Err(USAGE.to_owned()),
        },
        (Some("measure"), Some(replay), Some(meshes)) => run_measure(
            &replay,
            &meshes,
            value("--plain").as_deref(),
            value("--fitted").as_deref(),
        ),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
