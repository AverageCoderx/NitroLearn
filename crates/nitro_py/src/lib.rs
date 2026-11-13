//! `nitro._core`: the Python face of the Rust core. Keep this layer thin — logic lives in `nitro_core`.

mod env_pool;
mod policy;

use std::path::PathBuf;

use nitro_core::meshes;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_core, MeshLoadError, PyException);

/// Result of loading a collision-mesh folder.
#[pyclass(frozen, get_all, name = "MeshReport")]
struct PyMeshReport {
    folder: String,
    soccar_files: usize,
    soccar_recognized: usize,
    soccar_expected: usize,
}

#[pymethods]
impl PyMeshReport {
    fn is_complete(&self) -> bool {
        self.soccar_recognized == self.soccar_expected
    }
}

/// Loads arena collision meshes into RocketSim (once per process).
#[pyfunction]
fn load_collision_meshes(py: Python<'_>, folder: PathBuf) -> PyResult<PyMeshReport> {
    let report = py
        .detach(|| meshes::load(&folder))
        .map_err(|err| MeshLoadError::new_err(err.to_string()))?;
    Ok(PyMeshReport {
        folder: report.folder.display().to_string(),
        soccar_files: report.soccar_files,
        soccar_recognized: report.soccar_recognized,
        soccar_expected: meshes::SOCCAR_MESH_COUNT,
    })
}

/// The eight controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) of action `index`.
#[pyfunction]
fn action_controls(index: usize) -> PyResult<[f32; 8]> {
    if index >= nitro_core::actions::NUM_ACTIONS {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "action {index} is out of range (the table has {} actions)",
            nitro_core::actions::NUM_ACTIONS
        )));
    }
    Ok(nitro_core::actions::ActionTable::new()
        .controls(index)
        .to_floats())
}

/// How many actions the table has.
#[pyfunction]
fn num_actions() -> usize {
    nitro_core::actions::NUM_ACTIONS
}

fn pool_config(settings_json: &str) -> PyResult<nitro_core::config::PoolConfig> {
    nitro_core::config::PoolConfig::from_json(settings_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

fn le_bytes<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}

/// Behaviour-cloning samples of the recording at `path` (`.rlpr` or `.rlpr.zst`) under the core
/// settings `settings_json`: a dict with `obs` (f32), `mask` (u8), `label` (f32), `car` and `team` (u8) as
/// little-endian bytes, `count`, `obs_size`, `skipped` (decisions left out at gaps) and `outside_window`
/// (samples outside every kickoff window with `kickoff_only_seconds`). With `compact`,
/// the large replay mode's form instead (`compact_dict`).
#[pyfunction]
#[pyo3(signature = (path, settings_json, softness = 0.1, compact = false))]
fn demonstrations<'py>(
    py: Python<'py>,
    path: PathBuf,
    settings_json: &str,
    softness: f32,
    compact: bool,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let name = path.display().to_string();
    let read = || nitro_core::rlpr::read_file(&path);
    if compact {
        let (rules, expected) = demo_rules(settings_json, softness)?;
        let made = py
            .detach(|| -> Result<_, String> {
                let recording = read().map_err(|err| format!("{name}: {err}"))?;
                Ok(CompactBytes::of(&checked(
                    &name, &recording, &rules, expected,
                )?))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return compact_dict(py, &made);
    }
    samples_dict(py, &name, settings_json, softness, read)
}

/// The same from a plain recording's bytes held in memory, `name` naming it in errors.
#[pyfunction]
#[pyo3(signature = (data, name, settings_json, softness = 0.1))]
fn demonstrations_bytes<'py>(
    py: Python<'py>,
    data: &[u8],
    name: &str,
    settings_json: &str,
    softness: f32,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    samples_dict(py, name, settings_json, softness, || {
        nitro_core::rlpr::read(data)
    })
}

/// The large replay mode's samples of one replay straight from its replicar columns (the layout in
/// `nitro_core::replicar`), all of it outside Python's lock: the compact dict (`compact_dict`) plus
/// `kickoffs`, `ticks` and `rows_left_out`. With no kickoff the teams cannot be told apart: `count` is 0.
#[pyfunction]
#[pyo3(signature = (cars, frames, usable, car_floats, car_flags, ball_floats, name, settings_json, softness = 0.1))]
#[allow(clippy::too_many_arguments)]
fn replay_samples<'py>(
    py: Python<'py>,
    cars: usize,
    frames: &[u8],
    usable: &[u8],
    car_floats: &[u8],
    car_flags: &[u8],
    ball_floats: &[u8],
    name: &str,
    settings_json: &str,
    softness: f32,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (mut rules, expected) = demo_rules(settings_json, softness)?;
    // Replicar's rows hold the controls applied next, whatever the settings say.
    rules.replicar_rows = true;
    let columns = nitro_core::replicar::Columns {
        cars,
        frames,
        usable,
        car_floats,
        car_flags,
        ball_floats,
    };
    let (made, built) = py
        .detach(|| -> Result<_, String> {
            let built = nitro_core::replicar::recording(&columns)
                .map_err(|err| format!("{name}: {err}"))?;
            let made = if built.kickoffs == 0 {
                CompactBytes {
                    obs_size: expected,
                    ..CompactBytes::default()
                }
            } else {
                CompactBytes::of(&checked(name, &built.recording, &rules, expected)?)
            };
            Ok((
                made,
                (
                    built.kickoffs,
                    built.recording.ticks.len(),
                    built.rows_left_out,
                ),
            ))
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = compact_dict(py, &made)?;
    out.set_item("kickoffs", built.0)?;
    out.set_item("ticks", built.1)?;
    out.set_item("rows_left_out", built.2)?;
    Ok(out)
}

/// The demonstration rules of the core settings, with the copy's own keys (`demos::split_copy_keys`), and
/// the observation size they give.
fn demo_rules(
    settings_json: &str,
    softness: f32,
) -> PyResult<(nitro_core::demos::DemoRules, usize)> {
    let (pool_json, keys) = nitro_core::demos::split_copy_keys(settings_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let config = pool_config(&pool_json)?;
    let expected = config
        .observation
        .layout(config.team_size, config.opponents, config.obs_team_size)
        .size;
    let rules = nitro_core::demos::DemoRules {
        obs: config.observation,
        obs_team_size: config.obs_team_size,
        tick_skip: config.tick_skip,
        action_delay: config.action_delay,
        masks: config.action_masks,
        softness,
        replicar_rows: keys.replicar_rows,
        kickoff_only: keys.kickoff_only,
    };
    Ok((rules, expected))
}

/// The demonstrations of `recording`, refused when its game's observations differ from the settings'.
fn checked(
    name: &str,
    recording: &nitro_core::rlpr::Recording,
    rules: &nitro_core::demos::DemoRules,
    expected: usize,
) -> Result<nitro_core::demos::Demos, String> {
    let demos = nitro_core::demos::demonstrations(recording, rules)
        .map_err(|err| format!("{name}: {err}"))?;
    if demos.obs_size != expected {
        return Err(format!(
            "{name}: its game gives observations of {} numbers, the settings' game {expected}; a padded \
             observation (env.obs_team_size) takes recordings of every game size up to its room",
            demos.obs_size
        ));
    }
    Ok(demos)
}

fn samples_dict<'py>(
    py: Python<'py>,
    name: &str,
    settings_json: &str,
    softness: f32,
    read: impl FnOnce() -> Result<nitro_core::rlpr::Recording, String> + Send,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (rules, expected) = demo_rules(settings_json, softness)?;
    let demos = py
        .detach(|| {
            let recording = read().map_err(|err| format!("{name}: {err}"))?;
            checked(name, &recording, &rules, expected)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("count", demos.len())?;
    out.set_item("obs_size", demos.obs_size)?;
    out.set_item("skipped", demos.skipped)?;
    out.set_item("outside_window", demos.outside_window)?;
    out.set_item(
        "obs",
        pyo3::types::PyBytes::new(py, &le_bytes(&demos.obs, f32::to_le_bytes)),
    )?;
    out.set_item(
        "label",
        pyo3::types::PyBytes::new(py, &le_bytes(&demos.label, f32::to_le_bytes)),
    )?;
    out.set_item("mask", pyo3::types::PyBytes::new(py, &demos.mask))?;
    out.set_item("car", pyo3::types::PyBytes::new(py, &demos.car))?;
    out.set_item("team", pyo3::types::PyBytes::new(py, &demos.team))?;
    Ok(out)
}

/// The compact form's bytes, made outside Python's lock.
#[derive(Default)]
struct CompactBytes {
    count: usize,
    obs_size: usize,
    skipped: usize,
    outside_window: usize,
    obs: Vec<u8>,
    mask: Vec<u8>,
    label_index: Vec<u8>,
    label_share: Vec<u8>,
}

impl CompactBytes {
    fn of(demos: &nitro_core::demos::Demos) -> Self {
        let compact = demos.compact();
        Self {
            count: demos.len(),
            obs_size: demos.obs_size,
            skipped: demos.skipped,
            outside_window: demos.outside_window,
            obs: le_bytes(&compact.obs, u16::to_le_bytes),
            mask: compact.mask,
            label_index: compact.label_index,
            label_share: le_bytes(&compact.label_share, u16::to_le_bytes),
        }
    }
}

/// `count`, `obs_size`, `skipped`, `outside_window` (samples outside every kickoff window) and as bytes `obs`
/// (f16), `mask` (the bits packed, lowest action first,
/// `nitro_core::demos::MASK_BYTES` a sample), `label_index` (u8) and `label_share` (f16), each label's
/// `LABEL_ENTRIES` largest shares, the largest first.
fn compact_dict<'py>(
    py: Python<'py>,
    made: &CompactBytes,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("count", made.count)?;
    out.set_item("obs_size", made.obs_size)?;
    out.set_item("skipped", made.skipped)?;
    out.set_item("outside_window", made.outside_window)?;
    out.set_item("obs", pyo3::types::PyBytes::new(py, &made.obs))?;
    out.set_item("mask", pyo3::types::PyBytes::new(py, &made.mask))?;
    out.set_item(
        "label_index",
        pyo3::types::PyBytes::new(py, &made.label_index),
    )?;
    out.set_item(
        "label_share",
        pyo3::types::PyBytes::new(py, &made.label_share),
    )?;
    Ok(out)
}

/// The observation's layout for settings: (size, own part, one other car's block, teammate
/// places, opponent places, room). No games are built.
#[pyfunction]
fn observation_layout(settings_json: &str) -> PyResult<(usize, usize, usize, usize, usize, usize)> {
    let config = pool_config(settings_json)?;
    let layout =
        config
            .observation
            .layout(config.team_size, config.opponents, config.obs_team_size);
    Ok((
        layout.size,
        layout.self_part,
        layout.other_car,
        layout.teammates,
        layout.opponents,
        layout.room,
    ))
}

/// The observation's layout from the observation settings alone: what rebuilding a saved network
/// needs, so the rest of a settings file (rewards, start states, endings) is not checked on the way.
#[pyfunction]
fn observation_layout_of(
    observation: &str,
    team_size: usize,
    opponents: bool,
    obs_team_size: usize,
) -> PyResult<(usize, usize, usize, usize, usize, usize)> {
    let kind = nitro_core::obs::ObsKind::from_name(observation).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown observation '{observation}'"))
    })?;
    let layout = kind
        .checked_layout(team_size, opponents, obs_team_size)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        layout.size,
        layout.self_part,
        layout.other_car,
        layout.teammates,
        layout.opponents,
        layout.room,
    ))
}

/// The rewards of one's own compiled into this build: (name, description, paid every step,
/// file, [(setting, Rust type, optional, description)]).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn custom_rewards() -> Vec<(
    &'static str,
    &'static str,
    bool,
    &'static str,
    Vec<(&'static str, &'static str, bool, &'static str, &'static str)>,
)> {
    nitro_core::custom::REWARDS
        .iter()
        .map(|info| {
            let params = info
                .params
                .iter()
                .map(|p| (p.name, p.kind, p.optional, p.doc, p.default))
                .collect();
            (info.name, info.doc, info.every_step, info.file, params)
        })
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", nitro_core::VERSION)?;
    m.add("SIMULATOR", nitro_core::SIMULATOR)?;
    m.add("SOURCE_HASH", env!("NITRO_SOURCE_HASH"))?;
    m.add("MeshLoadError", m.py().get_type::<MeshLoadError>())?;
    m.add_class::<PyMeshReport>()?;
    m.add_function(wrap_pyfunction!(load_collision_meshes, m)?)?;
    m.add_function(wrap_pyfunction!(custom_rewards, m)?)?;
    m.add_function(wrap_pyfunction!(action_controls, m)?)?;
    m.add_function(wrap_pyfunction!(num_actions, m)?)?;
    m.add_function(wrap_pyfunction!(demonstrations, m)?)?;
    m.add_function(wrap_pyfunction!(demonstrations_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(replay_samples, m)?)?;
    m.add_function(wrap_pyfunction!(observation_layout, m)?)?;
    m.add_function(wrap_pyfunction!(observation_layout_of, m)?)?;
    policy::register(m)?;
    env_pool::register(m)
}
