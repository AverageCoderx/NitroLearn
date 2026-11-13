//! `EnvPool` for Python: environments that write straight into PyTorch CPU tensors.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use nitro_core::config::PoolConfig;
use nitro_core::driver::PoolDriver;
use nitro_core::env::{EnvStats, EpisodeRecord, PlayStats};
use nitro_core::pool::{self, BufferShape, GroupBuffers, RawBuffers, Work};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

create_exception!(_core, EnvError, PyRuntimeError);

#[derive(Clone, Copy)]
enum Columns {
    None,
    Obs,
    Actions,
}

/// Attribute name, dtype and row width of every tensor a group needs, in `RawBuffers` order.
const TENSORS: [(&str, &str, Columns); 9] = [
    ("obs", "torch.float32", Columns::Obs),
    ("mask", "torch.bool", Columns::Actions),
    ("reward", "torch.float32", Columns::None),
    ("terminal", "torch.bool", Columns::None),
    ("truncated", "torch.bool", Columns::None),
    ("episode", "torch.int64", Columns::None),
    ("step", "torch.int32", Columns::None),
    ("final_obs", "torch.float32", Columns::Obs),
    ("actions", "torch.int32", Columns::None),
];

/// A group's tensors, held so they outlive any work that writes into them.
struct Attached {
    /// Tensor, address and element count when attached (re-checked before each use).
    tensors: Vec<(Py<PyAny>, usize, usize)>,
    buffers: GroupBuffers,
}

/// Reward and timing totals since the previous `take_stats` call.
#[pyclass(frozen, get_all, name = "EnvStats")]
struct PyEnvStats {
    agent_steps: u64,
    raw: Vec<f64>,
    weighted: Vec<f64>,
    weighted_abs: Vec<f64>,
    own_abs: Vec<f64>,
    /// Per reward: agent steps on which it paid something before its weight.
    fired: Vec<u64>,
    env_steps: u64,
    step_seconds: f64,
    physics_seconds: f64,
    episodes: Vec<PyEpisodeRecord>,
    /// Gameplay statistics of the counted cars (`nitro_core::env::PlayStats`) by name.
    play: std::collections::HashMap<&'static str, f64>,
    /// With mixed game sizes: the same per players a side; empty otherwise.
    play_by_size: std::collections::HashMap<usize, std::collections::HashMap<&'static str, f64>>,
    /// Steps the card started (`card_start`), and the seconds its host node spent starting them.
    card_submits: u64,
    card_submit_seconds: f64,
}

/// `PlayStats` by name, as `PyEnvStats.play` holds them.
fn play_by_name(play: &PlayStats) -> std::collections::HashMap<&'static str, f64> {
    [
        ("car_steps", play.car_steps as f64),
        ("in_air", play.in_air as f64),
        ("touches", play.touches as f64),
        ("demolished", play.demolished as f64),
        ("supersonic", play.supersonic as f64),
        ("speed", play.speed),
        ("toward_ball", play.toward_ball),
        ("boost", play.boost),
        ("aerial_touches", play.aerial_touches as f64),
        ("dribbles", play.dribbles as f64),
        ("ball_on_car", play.ball_on_car as f64),
        ("flicks", play.flicks as f64),
        ("double_taps", play.double_taps as f64),
        ("flip_resets", play.flip_resets as f64),
        ("flip_reset_height", play.flip_reset_height),
        ("flip_reset_goal_dist", play.flip_reset_goal_dist),
        ("reset_musty_goals", play.reset_musty_goals as f64),
        ("reset_musty_goal_speed", play.reset_musty_goal_speed),
        ("shots", play.shots as f64),
        ("saves", play.saves as f64),
        ("demos", play.demos as f64),
    ]
    .into_iter()
    .collect()
}

/// One car's practice-task measurements for one finished episode (`nitro_core::env::EpisodeRecord`).
#[pyclass(frozen, get_all, skip_from_py_object, name = "EpisodeRecord")]
#[derive(Clone)]
pub(crate) struct PyEpisodeRecord {
    env: u32,
    setter: u8,
    car: u32,
    seconds: f32,
    terminal: bool,
    first_touch: f32,
    landed: f32,
    boost_collected: f32,
    ball_y_after_touch: f32,
    goal: i8,
    ball_first_touch: f32,
    kickoff: i8,
}

impl From<EpisodeRecord> for PyEpisodeRecord {
    fn from(record: EpisodeRecord) -> Self {
        Self {
            env: record.env,
            setter: record.setter,
            car: record.car,
            seconds: record.seconds,
            terminal: record.terminal,
            first_touch: record.first_touch,
            landed: record.landed,
            boost_collected: record.boost_collected,
            ball_y_after_touch: record.ball_y_after_touch,
            goal: record.goal,
            ball_first_touch: record.ball_first_touch,
            kickoff: record.kickoff,
        }
    }
}

impl From<EnvStats> for PyEnvStats {
    fn from(stats: EnvStats) -> Self {
        Self {
            agent_steps: stats.rewards.agent_steps,
            raw: stats.rewards.raw,
            weighted: stats.rewards.weighted,
            weighted_abs: stats.rewards.weighted_abs,
            own_abs: stats.rewards.own_abs,
            fired: stats.rewards.fired,
            env_steps: stats.steps,
            step_seconds: stats.step_time.as_secs_f64(),
            physics_seconds: stats.physics_time.as_secs_f64(),
            episodes: stats.episodes.into_iter().map(Into::into).collect(),
            play: play_by_name(&stats.play),
            play_by_size: stats
                .play_by_size
                .iter()
                .enumerate()
                .filter(|(size, _)| *size > 0)
                .map(|(size, play)| (size, play_by_name(play)))
                .collect(),
            card_submits: 0,
            card_submit_seconds: 0.0,
        }
    }
}

#[pyclass(frozen, get_all, name = "ObsLayout")]
struct PyObsLayout {
    size: usize,
    self_part: usize,
    other_car: usize,
    teammates: usize,
    opponents: usize,
    /// Players a side the observation has places for (0 = the game's own).
    room: usize,
}

#[pyclass(name = "EnvPool")]
struct PyEnvPool {
    // Declared first so it is dropped first: its workers stop before the tensors are released.
    pool: pool::EnvPool,
    attached: Vec<Option<Attached>>,
    /// Every card start made (`card_start`), kept until the pool goes: a CUDA graph holds their addresses,
    /// so each is boxed to stay where it is as the list grows.
    #[allow(clippy::vec_box)]
    card_starts: Vec<Box<CardStart>>,
}

/// What a graphics card needs to start a group's step itself (`env.card_starts_games`): the pool, the
/// group and its buffers as attached; a start that fails leaves its reason for the group's next `wait`.
struct CardStart {
    pool: *const pool::EnvPool,
    group: usize,
    buffers: GroupBuffers,
    /// Set when the group's tensors are attached again: its buffers are then gone.
    stale: AtomicBool,
    failure: Mutex<Option<String>>,
    /// Steps started and the nanoseconds `submit` took, since `take_stats` last read them.
    submits: AtomicU64,
    submit_nanos: AtomicU64,
}

// SAFETY: `pool` points at the `EnvPool` of the `PyEnvPool` that keeps this start, which is `Sync` (`_sync`)
// and outlives it; it is only read through a shared reference.
unsafe impl Send for CardStart {}
unsafe impl Sync for CardStart {}

/// Starts a group's step from a CUDA host node, on the driver's thread, without Python: as `step` does.
extern "C" fn start_from_card(data: *mut c_void) {
    // SAFETY: `data` is a `CardStart` its `PyEnvPool` keeps until it is dropped, and the graphs that call this
    // keep that pool alive (`nitro.inference`); the pool itself is shared between threads (`_sync`).
    let start = unsafe { &*(data as *const CardStart) };
    let result = if start.stale.load(Ordering::Acquire) {
        Err(format!(
            "group {}'s tensors were attached again after the card was set to start its steps",
            start.group
        ))
    } else {
        // Timed: the card's stream waits for this node, so its length is collection time.
        let begun = Instant::now();
        let result = unsafe { &*start.pool }.submit(start.group, Work::Step, start.buffers);
        let nanos = u64::try_from(begun.elapsed().as_nanos()).unwrap_or(u64::MAX);
        start.submit_nanos.fetch_add(nanos, Ordering::Relaxed);
        start.submits.fetch_add(1, Ordering::Relaxed);
        result
    };
    if let Err(failure) = result {
        *start.failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(failure);
    }
}

/// The steps the card started and the seconds their `submit` took, since this was last called.
fn take_card_submits(starts: &[Box<CardStart>]) -> (u64, f64) {
    starts.iter().fold((0, 0.0), |(submits, seconds), start| {
        let nanos = start.submit_nanos.swap(0, Ordering::Relaxed);
        (
            submits + start.submits.swap(0, Ordering::Relaxed),
            seconds + nanos as f64 * 1e-9,
        )
    })
}

/// The first failure of a card start of `group` since it was last asked, if any (reported once).
fn card_failure(starts: &[Box<CardStart>], group: usize) -> Result<(), String> {
    for start in starts.iter().filter(|start| start.group == group) {
        if let Some(failure) = start
            .failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            return Err(failure);
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _sync() {
    fn sync<T: Sync>() {}
    sync::<pool::EnvPool>();
}

fn parse_settings(settings_json: &str) -> PyResult<PoolConfig> {
    PoolConfig::from_json(settings_json).map_err(PyValueError::new_err)
}

fn checked_tensor(
    tensor: &Bound<'_, PyAny>,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> PyResult<(usize, usize)> {
    let device: String = tensor.getattr("device")?.getattr("type")?.extract()?;
    let actual_dtype = tensor.getattr("dtype")?.str()?.to_string();
    let actual_shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
    let contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if device != "cpu" || actual_dtype != dtype || actual_shape != shape || !contiguous {
        return Err(PyValueError::new_err(format!(
            "'{name}' must be a contiguous CPU tensor of {dtype} with shape {shape:?}, \
             got {actual_dtype} with shape {actual_shape:?} on {device}{}",
            if contiguous { "" } else { " (not contiguous)" }
        )));
    }
    Ok((
        tensor.call_method0("data_ptr")?.extract()?,
        tensor.call_method0("numel")?.extract()?,
    ))
}

impl PyEnvPool {
    fn shape(&self, group: usize) -> PyResult<BufferShape> {
        if group >= self.pool.num_groups() {
            return Err(PyValueError::new_err(format!(
                "group {group} does not exist (there are {})",
                self.pool.num_groups()
            )));
        }
        Ok(self.pool.shape(group))
    }

    /// The group's attached buffers, its tensors checked unchanged since `attach`.
    fn checked_buffers(&self, py: Python<'_>, group: usize) -> PyResult<GroupBuffers> {
        self.shape(group)?;
        let Some(attached) = &self.attached[group] else {
            return Err(EnvError::new_err(format!(
                "attach tensors to group {group} first"
            )));
        };
        for ((tensor, ptr, numel), (name, ..)) in attached.tensors.iter().zip(TENSORS) {
            let tensor = tensor.bind(py);
            let now: (usize, usize) = (
                tensor.call_method0("data_ptr")?.extract()?,
                tensor.call_method0("numel")?.extract()?,
            );
            if now != (*ptr, *numel) {
                return Err(EnvError::new_err(format!(
                    "tensor '{name}' of group {group} was resized or replaced after attach"
                )));
            }
        }
        Ok(attached.buffers)
    }

    fn submit(&self, py: Python<'_>, group: usize, work: Work) -> PyResult<()> {
        let buffers = self.checked_buffers(py, group)?;
        self.pool
            .submit(group, work, buffers)
            .map_err(EnvError::new_err)
    }
}

#[pymethods]
impl PyEnvPool {
    /// Builds every arena from the settings JSON (see `nitro.envs`); loads collision meshes if needed.
    #[new]
    fn new(py: Python<'_>, settings_json: &str) -> PyResult<Self> {
        let config = parse_settings(settings_json)?;
        let pool = py
            .detach(|| pool::EnvPool::new(&config))
            .map_err(EnvError::new_err)?;
        let attached = (0..pool.num_groups()).map(|_| None).collect();
        Ok(Self {
            pool,
            attached,
            card_starts: Vec::new(),
        })
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.pool.num_groups()
    }

    #[getter]
    fn num_workers(&self) -> usize {
        self.pool.num_workers()
    }

    #[getter]
    fn obs_size(&self) -> usize {
        self.pool.obs_layout().size
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.pool.num_actions()
    }

    #[getter]
    fn obs_layout(&self) -> PyObsLayout {
        let layout = self.pool.obs_layout();
        PyObsLayout {
            size: layout.size,
            self_part: layout.self_part,
            other_car: layout.other_car,
            teammates: layout.teammates,
            opponents: layout.opponents,
            room: layout.room,
        }
    }

    #[getter]
    /// Every arena's players a side, in arena order (several sizes with env.team_sizes).
    fn arena_team_sizes(&self) -> Vec<usize> {
        self.pool.arena_team_sizes().to_vec()
    }

    #[getter]
    fn reward_labels(&self) -> Vec<&'static str> {
        self.pool.reward_labels().to_vec()
    }

    /// Number of agents (cars) in `group`: the row count of each of its tensors.
    fn group_agents(&self, group: usize) -> PyResult<usize> {
        Ok(self.shape(group)?.agents)
    }

    /// Lets `group` write into the tensors held as attributes of `tensors` (obs, mask, reward, terminal,
    /// truncated, episode, step, final_obs, actions). They must stay alive and unresized while attached.
    fn attach(&mut self, group: usize, tensors: &Bound<'_, PyAny>) -> PyResult<()> {
        let shape = self.shape(group)?;
        if self.pool.is_busy(group) {
            return Err(EnvError::new_err(format!(
                "group {group} is working; wait for it before attaching new tensors"
            )));
        }
        let mut held = Vec::with_capacity(TENSORS.len());
        for (name, dtype, columns) in TENSORS {
            let tensor = tensors.getattr(name)?;
            let dims = match columns {
                Columns::None => vec![shape.agents],
                Columns::Obs => vec![shape.agents, shape.obs_size],
                Columns::Actions => vec![shape.agents, shape.num_actions],
            };
            let (ptr, numel) = checked_tensor(&tensor, name, dtype, &dims)?;
            held.push((tensor.unbind(), ptr, numel));
        }
        let ptr = |i: usize| held[i].1;
        // SAFETY: each address belongs to a contiguous CPU tensor of exactly the size `shape` implies
        // (checked above; bool is one byte in PyTorch). `held` keeps the tensors alive while attached,
        // and `submit` re-checks that none was resized or replaced.
        let buffers = unsafe {
            GroupBuffers::from_raw(
                shape,
                RawBuffers {
                    obs: ptr(0) as *mut f32,
                    mask: ptr(1) as *mut u8,
                    reward: ptr(2) as *mut f32,
                    terminal: ptr(3) as *mut u8,
                    truncated: ptr(4) as *mut u8,
                    episode: ptr(5) as *mut i64,
                    step: ptr(6) as *mut i32,
                    final_obs: ptr(7) as *mut f32,
                    actions: ptr(8) as *mut i32,
                },
            )
        };
        for start in self.card_starts.iter().filter(|start| start.group == group) {
            start.stale.store(true, Ordering::Release);
        }
        self.attached[group] = Some(Attached {
            tensors: held,
            buffers,
        });
        Ok(())
    }

    /// Starts writing the current observations of `group` (needed once, before its first step).
    fn observe(&self, py: Python<'_>, group: usize) -> PyResult<()> {
        self.submit(py, group, Work::Observe)
    }

    /// Starts one step of `group` with the actions in its `actions` tensor; returns immediately.
    fn step(&self, py: Python<'_>, group: usize) -> PyResult<()> {
        self.submit(py, group, Work::Step)
    }

    /// Waits until `group` has finished; until then its tensors must not be read or written. With the card
    /// starting the group's steps, wait for the card first (its host node has then run).
    fn wait(&self, py: Python<'_>, group: usize) -> PyResult<()> {
        self.shape(group)?;
        let pool = &self.pool;
        py.detach(|| pool.wait(group)).map_err(EnvError::new_err)?;
        card_failure(&self.card_starts, group).map_err(EnvError::new_err)
    }

    /// The address of a C function and of its argument that start one step of `group` with the actions in
    /// its `actions` tensor, as `step` does, from any thread and without Python: for a CUDA host node at the
    /// end of the group's action choosing (`env.card_starts_games`). The tensors are checked now, not
    /// at each start; a start that fails is reported by the group's next `wait`. Valid while this pool
    /// lives and the group's tensors stay attached.
    fn card_start(&mut self, py: Python<'_>, group: usize) -> PyResult<(usize, usize)> {
        let buffers = self.checked_buffers(py, group)?;
        let start = Box::new(CardStart {
            pool: &raw const self.pool,
            group,
            buffers,
            stale: AtomicBool::new(false),
            failure: Mutex::new(None),
            submits: AtomicU64::new(0),
            submit_nanos: AtomicU64::new(0),
        });
        let data = &raw const *start as usize;
        self.card_starts.push(start);
        Ok((start_from_card as *const () as usize, data))
    }

    /// Totals since the last call, with the card's starts of the games (`card_start`) and their time.
    fn take_stats(&self, py: Python<'_>) -> PyEnvStats {
        let pool = &self.pool;
        let mut stats: PyEnvStats = py.detach(|| pool.take_stats()).into();
        (stats.card_submits, stats.card_submit_seconds) = take_card_submits(&self.card_starts);
        stats
    }

    /// Chooses which of arena `arena`'s cars count in `take_stats` (one flag per car, in the arena's car
    /// order: Blue, Orange, Blue, Orange...). Self-play uses it to leave the frozen opponents out.
    fn set_counted(&self, arena: usize, counted: Vec<bool>) -> PyResult<()> {
        self.pool
            .set_counted(arena, &counted)
            .map_err(PyValueError::new_err)
    }

    /// Starts writing viewer frames for `arena` every `every_ticks` physics ticks, beginning with
    /// the current state; returns the boost pads as (x, y, z, big). Call between steps.
    fn start_capture(
        &self,
        py: Python<'_>,
        arena: usize,
        every_ticks: u32,
    ) -> PyResult<Vec<[f32; 4]>> {
        let pool = &self.pool;
        py.detach(|| pool.start_capture(arena, every_ticks))
            .map_err(PyValueError::new_err)
    }

    /// The viewer frames `arena` wrote since the last call, flat (layout in `nitro_core::capture`).
    fn take_capture(&self, py: Python<'_>, arena: usize) -> PyResult<Vec<f32>> {
        let pool = &self.pool;
        py.detach(|| pool.take_capture(arena))
            .map_err(PyValueError::new_err)
    }

    /// Arena `arena`'s current state as floats: the ball's position, then for each car its team (0 Blue,
    /// 1 Orange), position, forward direction, up direction, boost (0-100), on the ground (0/1) and
    /// demolished (0/1). Positions in uu.
    fn frame(&self, py: Python<'_>, arena: usize) -> PyResult<Vec<f32>> {
        let pool = &self.pool;
        let snap = py
            .detach(|| pool.snapshot(arena))
            .map_err(PyValueError::new_err)?;
        let mut frame = snap.ball.pos.to_array().to_vec();
        for car in &snap.cars {
            frame.push(car.team.index() as f32);
            frame.extend(car.body.pos.to_array());
            frame.extend(car.body.forward().to_array());
            frame.extend(car.body.up().to_array());
            frame.extend([
                car.boost,
                f32::from(u8::from(car.on_ground)),
                f32::from(u8::from(car.is_demoed)),
            ]);
        }
        Ok(frame)
    }
}

/// Result of [`rust_driven_benchmark`].
#[pyclass(frozen, get_all, name = "BenchmarkResult")]
struct PyBenchmarkResult {
    agents: usize,
    steps: usize,
    seconds: f64,
    stats: Py<PyEnvStats>,
}

/// Plays a pool built from `settings_json` for `warmup_steps` + `steps` steps entirely in Rust (no Python
/// in the loop), the reference for measuring Python's overhead. Each step's actions are read in order
/// from `actions` (little-endian int32, all agents of group 0, then group 1, ...), so a Python-driven run
/// can replay exactly the same games. `pipelined` starts each group's next step as soon as it is free,
/// as the trainer does; otherwise every group steps and all are waited for.
#[pyfunction]
fn rust_driven_benchmark(
    py: Python<'_>,
    settings_json: &str,
    actions: &[u8],
    warmup_steps: usize,
    steps: usize,
    pipelined: bool,
) -> PyResult<PyBenchmarkResult> {
    let config = parse_settings(settings_json)?;
    let table: Vec<i32> = actions
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect();
    let (agents, seconds, stats) = py
        .detach(|| -> Result<_, String> {
            let mut driver = PoolDriver::new(&config)?;
            let agents = driver.buffers.iter().map(|b| b.reward.len()).sum::<usize>();
            if actions.len() != (warmup_steps + steps) * agents * 4 {
                return Err(format!(
                    "actions holds {} bytes; {} steps of {agents} agents need {}",
                    actions.len(),
                    warmup_steps + steps,
                    (warmup_steps + steps) * agents * 4
                ));
            }
            let mut next = 0;
            let mut run = |driver: &mut PoolDriver, steps: usize| -> Result<(), String> {
                let mut fill = |slot: &mut [i32]| {
                    slot.copy_from_slice(&table[next..next + slot.len()]);
                    next += slot.len();
                };
                for _ in 0..steps {
                    if pipelined {
                        driver.step_pipelined(&mut fill)?;
                    } else {
                        driver.step(&mut fill)?;
                    }
                }
                driver.finish()
            };
            run(&mut driver, warmup_steps)?;
            driver.pool.take_stats();
            let start = Instant::now();
            run(&mut driver, steps)?;
            let seconds = start.elapsed().as_secs_f64();
            Ok((agents, seconds, driver.pool.take_stats()))
        })
        .map_err(EnvError::new_err)?;
    Ok(PyBenchmarkResult {
        agents,
        steps,
        seconds,
        stats: Py::new(py, PyEnvStats::from(stats))?,
    })
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("EnvError", m.py().get_type::<EnvError>())?;
    m.add_class::<PyEnvPool>()?;
    m.add_class::<PyEnvStats>()?;
    m.add_class::<PyEpisodeRecord>()?;
    m.add_class::<PyObsLayout>()?;
    m.add_class::<PyBenchmarkResult>()?;
    m.add_function(wrap_pyfunction!(rust_driven_benchmark, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CardStart, start_from_card, take_card_submits};
    use nitro_core::pool::{OwnedBuffers, Work};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64};

    /// Each step the card starts (its host node calling `start_from_card`, here called directly) is
    /// counted and timed, and `take_stats` hands the totals over once.
    #[test]
    fn the_cards_starts_of_the_games_are_counted_and_timed() {
        let meshes = std::env::var_os("NITRO_TEST_MESHES").map_or_else(
            || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../collision_meshes"),
            std::path::PathBuf::from,
        );
        if !meshes.join("soccar").is_dir() {
            eprintln!("SKIPPED: no collision meshes at {}", meshes.display());
            return;
        }
        let settings = two_arena_settings(&meshes.display().to_string());
        let pool = nitro_core::pool::EnvPool::new(
            &nitro_core::config::PoolConfig::from_json(&settings).unwrap(),
        )
        .unwrap();
        let shape = pool.shape(0);
        let mut owned = OwnedBuffers::new(shape);
        pool.submit(0, Work::Observe, owned.view(shape)).unwrap();
        pool.wait(0).unwrap();
        let start = Box::new(CardStart {
            pool: &raw const pool,
            group: 0,
            buffers: owned.view(shape),
            stale: AtomicBool::new(false),
            failure: Mutex::new(None),
            submits: AtomicU64::new(0),
            submit_nanos: AtomicU64::new(0),
        });
        let starts = vec![start];
        for _ in 0..5 {
            start_from_card(&raw const *starts[0] as *mut std::ffi::c_void);
            pool.wait(0).unwrap();
        }
        assert!(starts[0].failure.lock().unwrap().is_none());
        let (submits, seconds) = take_card_submits(&starts);
        assert_eq!(submits, 5);
        assert!(seconds > 0.0 && seconds < 5.0, "{seconds}");
        assert_eq!(take_card_submits(&starts), (0, 0.0), "handed over once");
    }

    /// Settings of a pool of two 1v1 arenas in one group on one thread, meshes in `meshes`.
    fn two_arena_settings(meshes: &str) -> String {
        format!(
            r#"{{"collision_meshes": {meshes:?}, "seed": 1, "team_size": 1, "opponents": true,
            "num_arenas": 2, "groups": 1, "threads": 1, "tick_skip": 8, "action_delay": 1,
            "observation": "native", "action_masks": "standard",
            "terminals": {{"no_touch_seconds": 2.0, "max_episode_seconds": 30.0, "end_on_touch": false,
                           "end_on_landing": false}},
            "rewards": [], "state_setters": [{{"kind": {{"name": "kickoff", "fuzz_uu": 0.0}}, "weight": 1.0}}]}}"#
        )
    }
}
