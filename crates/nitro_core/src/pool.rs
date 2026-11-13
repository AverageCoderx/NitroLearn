//! Many environments stepped by a fixed set of worker threads, writing straight into shared buffers.
//!
//! Arenas are split into groups. A caller submits work for one group and gets control back at once,
//! so it can run inference for another group while the workers step this one.
//!
//! Work is handed out without a shared lock: each worker steps its own share of every group, the same
//! arenas each step so their memory stays in its caches, then takes what is left of the others' shares.

use std::cell::UnsafeCell;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::config::PoolConfig;
use crate::env::{AgentRows, Env, EnvRules, EnvStats, PlayStats};
use crate::meshes;
use crate::obs::ObsLayout;
use crate::snapshot::GameSnapshot;

/// A view of memory owned elsewhere (a Python tensor or [`OwnedBuffers`]).
#[derive(Clone, Copy, Debug)]
struct Raw<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: a `Raw` is only dereferenced through `GroupBuffers::rows`, whose contract gives each worker
// a disjoint range; the owner keeps the memory alive while a pool references it.
unsafe impl<T: Send> Send for Raw<T> {}
unsafe impl<T: Send> Sync for Raw<T> {}

impl<T> Raw<T> {
    /// # Safety
    /// The memory must be valid for the whole range and not be accessed concurrently elsewhere.
    unsafe fn slice<'a>(self, start: usize, len: usize) -> &'a mut [T] {
        assert!(start + len <= self.len, "buffer range out of bounds");
        if len == 0 {
            return &mut []; // an empty buffer's pointer may be null (a tensor with no elements)
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

/// Element counts of one group's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferShape {
    pub agents: usize,
    pub obs_size: usize,
    pub num_actions: usize,
}

impl BufferShape {
    pub fn obs_len(self) -> usize {
        self.agents * self.obs_size
    }

    pub fn mask_len(self) -> usize {
        self.agents * self.num_actions
    }
}

/// Where one group reads actions from and writes everything else to.
#[derive(Clone, Copy, Debug)]
pub struct GroupBuffers {
    shape: BufferShape,
    obs: Raw<f32>,
    mask: Raw<u8>,
    reward: Raw<f32>,
    terminal: Raw<u8>,
    truncated: Raw<u8>,
    episode: Raw<i64>,
    step: Raw<i32>,
    final_obs: Raw<f32>,
    actions: Raw<i32>,
}

pub struct RawBuffers {
    pub obs: *mut f32,
    pub mask: *mut u8,
    pub reward: *mut f32,
    pub terminal: *mut u8,
    pub truncated: *mut u8,
    pub episode: *mut i64,
    pub step: *mut i32,
    pub final_obs: *mut f32,
    pub actions: *mut i32,
}

impl GroupBuffers {
    /// # Safety
    /// Every pointer must reference memory of the length `shape` implies (`obs`/`final_obs`:
    /// `agents × obs_size`, `mask`: `agents × num_actions`, the rest: `agents`), which stays valid and
    /// is not touched by anyone else while the pool works on this group.
    pub unsafe fn from_raw(shape: BufferShape, raw: RawBuffers) -> Self {
        let n = shape.agents;
        Self {
            shape,
            obs: Raw {
                ptr: raw.obs,
                len: shape.obs_len(),
            },
            mask: Raw {
                ptr: raw.mask,
                len: shape.mask_len(),
            },
            reward: Raw {
                ptr: raw.reward,
                len: n,
            },
            terminal: Raw {
                ptr: raw.terminal,
                len: n,
            },
            truncated: Raw {
                ptr: raw.truncated,
                len: n,
            },
            episode: Raw {
                ptr: raw.episode,
                len: n,
            },
            step: Raw {
                ptr: raw.step,
                len: n,
            },
            final_obs: Raw {
                ptr: raw.final_obs,
                len: shape.obs_len(),
            },
            actions: Raw {
                ptr: raw.actions,
                len: n,
            },
        }
    }

    /// # Safety
    /// No two live calls may cover the same agents.
    unsafe fn rows<'a>(self, first_agent: usize, agents: usize) -> (AgentRows<'a>, &'a [i32]) {
        let size = self.shape.obs_size;
        unsafe {
            let rows = AgentRows {
                obs: self.obs.slice(first_agent * size, agents * size),
                mask: self.mask.slice(
                    first_agent * self.shape.num_actions,
                    agents * self.shape.num_actions,
                ),
                reward: self.reward.slice(first_agent, agents),
                terminal: self.terminal.slice(first_agent, agents),
                truncated: self.truncated.slice(first_agent, agents),
                episode: self.episode.slice(first_agent, agents),
                step: self.step.slice(first_agent, agents),
                final_obs: self.final_obs.slice(first_agent * size, agents * size),
            };
            (rows, self.actions.slice(first_agent, agents))
        }
    }
}

/// Buffers owned by Rust, for tests and Rust-only benchmarks.
pub struct OwnedBuffers {
    pub obs: Vec<f32>,
    pub mask: Vec<u8>,
    pub reward: Vec<f32>,
    pub terminal: Vec<u8>,
    pub truncated: Vec<u8>,
    pub episode: Vec<i64>,
    pub step: Vec<i32>,
    pub final_obs: Vec<f32>,
    pub actions: Vec<i32>,
}

impl OwnedBuffers {
    pub fn new(shape: BufferShape) -> Self {
        let n = shape.agents;
        Self {
            obs: vec![0.0; shape.obs_len()],
            mask: vec![0; shape.mask_len()],
            reward: vec![0.0; n],
            terminal: vec![0; n],
            truncated: vec![0; n],
            episode: vec![0; n],
            step: vec![0; n],
            final_obs: vec![0.0; shape.obs_len()],
            actions: vec![0; n],
        }
    }

    /// Views into these vectors; `self` must outlive the pool's use of them and not be resized.
    pub fn view(&mut self, shape: BufferShape) -> GroupBuffers {
        // SAFETY: the vectors were sized from `shape`; the caller keeps them alive and unmoved.
        unsafe {
            GroupBuffers::from_raw(
                shape,
                RawBuffers {
                    obs: self.obs.as_mut_ptr(),
                    mask: self.mask.as_mut_ptr(),
                    reward: self.reward.as_mut_ptr(),
                    terminal: self.terminal.as_mut_ptr(),
                    truncated: self.truncated.as_mut_ptr(),
                    episode: self.episode.as_mut_ptr(),
                    step: self.step.as_mut_ptr(),
                    final_obs: self.final_obs.as_mut_ptr(),
                    actions: self.actions.as_mut_ptr(),
                },
            )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Apply the actions in the buffer and step every environment of the group.
    Step,
    /// Write the current observations (needed once before the first step).
    Observe,
}

/// What a group's open phase does.
#[derive(Clone, Copy)]
struct Task {
    work: Work,
    buffers: GroupBuffers,
}

/// A counter on a cache line of its own, so workers claiming from their own shares never share a line.
#[repr(align(128))]
struct Counter(AtomicUsize);

/// `T` alone on its cache lines (128 bytes: neighbouring-line prefetchers fetch 64-byte lines in pairs), so
/// threads writing it never slow down the reads of what lies next to it.
#[repr(align(128))]
struct Padded<T>(T);

impl<T> std::ops::Deref for Padded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// A lock and the condvar that waits with it, together on cache lines of their own (as `Padded`).
#[repr(align(128))]
struct Locked<T> {
    lock: Mutex<T>,
    changed: Condvar,
}

impl<T> Locked<T> {
    fn new(value: T) -> Self {
        Self {
            lock: Mutex::new(value),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, T> {
        self.lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One group's work. Environments are claimed one at a time from the workers' shares (`share`): a worker
/// first from its own, then from the others'. A phase ends when the environments stepped, which each worker
/// adds up once it finds nothing left to claim, reach the group's size; the worker that adds the last ones
/// ends it.
struct Job {
    task: UnsafeCell<Option<Task>>,
    /// Per share: the next environment to claim; past the share's end when there is nothing to claim.
    next: Vec<Counter>,
    done: Padded<AtomicUsize>,
}

// SAFETY: `task` is written only by the thread that opens a phase, while every share's counter is past its
// end (no claim can succeed: the group is idle, or its last phase was completely stepped), and read only
// after a successful claim, whose acquire pairs with the release that reopened that counter after the write.
unsafe impl Sync for Job {}

/// The environments of share `s` of `shares` in a group of `envs`.
fn share(envs: usize, shares: usize, s: usize) -> Range<usize> {
    s * envs / shares..(s + 1) * envs / shares
}

struct Schedule {
    busy: Vec<bool>,
    /// Per group: the first error since its last `wait`.
    failures: Vec<Option<String>>,
    /// Set by any error: a failed environment may be half-stepped, so no further work is accepted.
    broken: Option<String>,
}

struct Shared {
    groups: Vec<Vec<Mutex<Env>>>,
    /// Per group: each environment's first row, then the group's row count (arenas of several game sizes
    /// have different car counts).
    rows: Vec<Vec<usize>>,
    jobs: Vec<Job>,
    shares: usize,
    /// How long a worker with nothing to do watches for work before it sleeps (`spin`).
    spin: Duration,
    /// Moves whenever a phase opens or the pool shuts down; a worker with nothing to do waits for it to move.
    epoch: Padded<AtomicU64>,
    sleepers: Padded<AtomicUsize>,
    shutdown: AtomicBool,
    /// Where workers with nothing to do sleep; `changed` when the epoch moves.
    sleep: Locked<()>,
    /// `changed` when a group's work is done.
    schedule: Locked<Schedule>,
    /// Tests only: an environment index that panics in its step.
    #[cfg(test)]
    fail_at: Mutex<Option<usize>>,
}

/// How long a worker with nothing to do watches for new work before it sleeps: a group's next step usually
/// comes within a network pass, and a sleeping worker takes a system call to wake. 20 µs, or
/// `NITRO_POOL_SPIN_US` microseconds (a knob for measuring; read once per process).
fn spin() -> Duration {
    static SPIN: OnceLock<Duration> = OnceLock::new();
    *SPIN.get_or_init(|| {
        const DEFAULT: u64 = 20;
        const MOST: u64 = 10_000; // longer would keep every idle worker busy through a learning update
        let Ok(text) = std::env::var("NITRO_POOL_SPIN_US") else {
            return Duration::from_micros(DEFAULT);
        };
        let micros = match text.trim().parse::<u64>() {
            Ok(micros) if micros > MOST => {
                eprintln!("NITRO_POOL_SPIN_US={micros} is more than {MOST} microseconds; using {MOST}");
                MOST
            }
            Ok(micros) => micros,
            Err(_) => {
                eprintln!("NITRO_POOL_SPIN_US={text:?} is not a whole number of microseconds; using {DEFAULT}");
                DEFAULT
            }
        };
        Duration::from_micros(micros)
    })
}

impl Shared {
    fn schedule(&self) -> MutexGuard<'_, Schedule> {
        self.schedule.lock()
    }

    /// Opens `task` on `group`, whose counters are all past their ends.
    fn open(&self, group: usize, task: Task) {
        let job = &self.jobs[group];
        // SAFETY: nothing can claim from this group now (see `Job`).
        unsafe { *job.task.get() = Some(task) };
        job.done.store(0, Ordering::Relaxed);
        let envs = self.groups[group].len();
        for (s, counter) in job.next.iter().enumerate() {
            counter
                .0
                .store(share(envs, self.shares, s).start, Ordering::Release);
        }
        self.wake();
    }

    fn wake(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = self.sleep.lock();
            self.sleep.changed.notify_all();
        }
    }

    /// Lets every worker stop once it has nothing to do (the pool is going).
    fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.wake();
        let guard = self.sleep.lock();
        self.sleep.changed.notify_all();
        drop(guard);
    }

    /// Returns once the epoch has moved past `seen` (or the pool shuts down): watching it for `spin`, then
    /// asleep.
    fn wait_for_work(&self, seen: u64) {
        let start = Instant::now();
        while start.elapsed() < self.spin {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            if self.epoch.load(Ordering::SeqCst) != seen {
                return;
            }
        }
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.sleep.lock();
        while self.epoch.load(Ordering::SeqCst) == seen && !self.shutdown.load(Ordering::SeqCst) {
            guard = self
                .sleep
                .changed
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(guard);
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct EnvPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    rules: Arc<EnvRules>,
    reward_labels: Vec<&'static str>,
    /// Every arena's players a side.
    arena_team_sizes: Vec<usize>,
    /// Arenas of more than one game size: the statistics are also kept per size.
    mixed: bool,
}

impl EnvPool {
    /// Loads the collision meshes (once per process) and builds every arena.
    pub fn new(config: &PoolConfig) -> Result<Self, String> {
        let report = meshes::load(&config.collision_meshes).map_err(|err| err.to_string())?;
        if !report.is_complete() {
            return Err(format!(
                "only {}/{} soccar collision meshes were recognised in {}",
                report.soccar_recognized,
                meshes::SOCCAR_MESH_COUNT,
                report.folder.display()
            ));
        }
        let rules = Arc::new(config.rules()?);
        let arena_team_sizes = config.arena_team_sizes();
        let mut groups: Vec<Vec<Mutex<Env>>> = (0..config.groups).map(|_| Vec::new()).collect();
        for arena in 0..config.num_arenas {
            let (rules, id, seed) = (
                Arc::clone(&rules),
                arena as u32,
                mix_seed(config.seed, arena as u64),
            );
            let env = Env::sized(rules, id, seed, arena_team_sizes[arena]);
            groups[arena % config.groups].push(Mutex::new(env));
        }
        let reward_labels = rules.rewards.iter().map(|term| term.kind.label()).collect();
        let shares = config.worker_threads();
        let jobs = groups
            .iter()
            .map(|envs| Job {
                task: UnsafeCell::new(None),
                next: (0..shares)
                    .map(|s| Counter(AtomicUsize::new(share(envs.len(), shares, s).end)))
                    .collect(),
                done: Padded(AtomicUsize::new(0)),
            })
            .collect();
        let sides = if config.opponents { 2 } else { 1 };
        let rows = (0..config.groups)
            .map(|group| {
                let mut firsts = vec![0];
                for arena in (group..config.num_arenas).step_by(config.groups) {
                    firsts.push(
                        firsts.last().copied().unwrap_or(0) + arena_team_sizes[arena] * sides,
                    );
                }
                firsts
            })
            .collect();
        let mixed = arena_team_sizes.iter().any(|&n| n != config.team_size);
        let shared = Arc::new(Shared {
            rows,
            groups,
            jobs,
            shares,
            spin: spin(),
            epoch: Padded(AtomicU64::new(0)),
            sleepers: Padded(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
            sleep: Locked::new(()),
            schedule: Locked::new(Schedule {
                busy: vec![false; config.groups],
                failures: vec![None; config.groups],
                broken: None,
            }),
            #[cfg(test)]
            fail_at: Mutex::new(None),
        });
        // Built before the workers so that, if one fails to start, dropping the pool stops the others.
        let mut pool = Self {
            shared,
            workers: Vec::new(),
            rules,
            reward_labels,
            arena_team_sizes,
            mixed,
        };
        for i in 0..shares {
            let shared = Arc::clone(&pool.shared);
            let handle = std::thread::Builder::new()
                .name(format!("nitro-env-{i}"))
                .spawn(move || worker(&shared, i))
                .map_err(|err| format!("could not start worker thread: {err}"))?;
            pool.workers.push(handle);
        }
        Ok(pool)
    }

    pub fn num_groups(&self) -> usize {
        self.shared.groups.len()
    }

    pub fn shape(&self, group: usize) -> BufferShape {
        BufferShape {
            agents: self.shared.rows[group].last().copied().unwrap_or(0),
            obs_size: self.rules.obs_size(),
            num_actions: self.rules.actions.len(),
        }
    }

    /// Every arena's players a side, in arena order (several sizes with `team_sizes`).
    pub fn arena_team_sizes(&self) -> &[usize] {
        &self.arena_team_sizes
    }

    pub fn obs_layout(&self) -> ObsLayout {
        self.rules.obs.layout(
            self.rules.team_size,
            self.rules.opponents,
            self.rules.obs_team_size,
        )
    }

    pub fn reward_labels(&self) -> &[&'static str] {
        &self.reward_labels
    }

    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Starts `work` on `group` and returns immediately. The buffers must match `shape(group)` and stay
    /// untouched until [`EnvPool::wait`] returns for this group.
    pub fn submit(&self, group: usize, work: Work, buffers: GroupBuffers) -> Result<(), String> {
        if buffers.shape != self.shape(group) {
            return Err(format!(
                "buffers have shape {:?}, group {group} needs {:?}",
                buffers.shape,
                self.shape(group)
            ));
        }
        let mut schedule = self.shared.schedule();
        if let Some(reason) = &schedule.broken {
            return Err(format!(
                "the environments stopped after an internal error: {reason}"
            ));
        }
        if schedule.busy[group] {
            return Err(format!("group {group} is still working; wait for it first"));
        }
        if work == Work::Step {
            // SAFETY: the group is idle, so nothing else touches its buffers.
            let actions = unsafe { buffers.actions.slice(0, buffers.shape.agents) };
            let count = self.rules.actions.len();
            if let Some(bad) = actions
                .iter()
                .find(|&&a| usize::try_from(a).map_or(true, |a| a >= count))
            {
                return Err(format!(
                    "group {group} has action {bad}; actions must be 0 to {}",
                    count - 1
                ));
            }
        }
        schedule.busy[group] = true;
        drop(schedule);
        self.shared.open(group, Task { work, buffers });
        Ok(())
    }

    /// Blocks until `group` has finished its submitted work.
    pub fn wait(&self, group: usize) -> Result<(), String> {
        let shared = &*self.shared;
        let mut schedule = shared.schedule();
        while schedule.busy[group] {
            schedule = shared
                .schedule
                .changed
                .wait(schedule)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        match schedule.failures[group].take() {
            Some(failure) => Err(failure),
            None => Ok(()),
        }
    }

    /// True between submit and the end of the work for group.
    pub fn is_busy(&self, group: usize) -> bool {
        self.shared.schedule().busy[group]
    }

    /// A copy of the current state of arena `arena` (arenas are numbered as when the pool was built).
    /// Chooses which of arena `arena`'s cars enter the reward statistics (self-play's frozen opponents do
    /// not). Call between steps, as any other access to an arena.
    pub fn set_counted(&self, arena: usize, counted: &[bool]) -> Result<(), String> {
        let groups = &self.shared.groups;
        let env = groups[arena % groups.len()]
            .get(arena / groups.len())
            .ok_or_else(|| format!("there is no arena {arena}"))?;
        env.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_counted(counted)
    }

    pub fn snapshot(&self, arena: usize) -> Result<GameSnapshot, String> {
        let groups = &self.shared.groups;
        let env = groups[arena % groups.len()]
            .get(arena / groups.len())
            .ok_or_else(|| format!("there is no arena {arena}"))?;
        Ok(env
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot()
            .clone())
    }

    fn with_env<T>(&self, arena: usize, f: impl FnOnce(&mut Env) -> T) -> Result<T, String> {
        let groups = &self.shared.groups;
        let env = groups[arena % groups.len()]
            .get(arena / groups.len())
            .ok_or_else(|| format!("there is no arena {arena}"))?;
        Ok(f(&mut env
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)))
    }

    /// Starts writing viewer frames for `arena` every `every_ticks` ticks; call between steps.
    pub fn start_capture(&self, arena: usize, every_ticks: u32) -> Result<Vec<[f32; 4]>, String> {
        self.with_env(arena, |env| {
            env.start_capture(every_ticks);
            env.pad_layout()
        })
    }

    /// The viewer frames `arena` wrote since the last call (`capture`).
    pub fn take_capture(&self, arena: usize) -> Result<Vec<f32>, String> {
        self.with_env(arena, Env::take_capture)
    }

    pub fn num_actions(&self) -> usize {
        self.rules.actions.len()
    }

    /// Statistics of every environment since the last call; steps still in progress count towards the next.
    pub fn take_stats(&self) -> EnvStats {
        let mut total = EnvStats::new(self.reward_labels.len());
        if self.mixed {
            let largest = self.arena_team_sizes.iter().copied().max().unwrap_or(0);
            total.play_by_size = vec![PlayStats::default(); largest + 1];
        }
        for env in self.shared.groups.iter().flatten() {
            let mut env = env
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.mixed {
                // Several game sizes: this env's play also goes to its size's.
                let stats = env.take_stats();
                total.play_by_size[env.team_size()].merge(&stats.play);
                total.merge(&stats);
            } else {
                env.add_stats_to(&mut total);
            }
        }
        total
    }
}

impl Drop for EnvPool {
    fn drop(&mut self) {
        self.shared.stop();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Independent, well-spread seeds for each arena (SplitMix64).
pub(crate) fn mix_seed(seed: u64, index: u64) -> u64 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn worker(shared: &Shared, me: usize) {
    loop {
        let seen = shared.epoch.load(Ordering::SeqCst);
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let mut worked = false;
        for group in 0..shared.jobs.len() {
            worked |= drain(shared, group, me);
        }
        if !worked {
            shared.wait_for_work(seen);
        }
    }
}

/// Steps every environment of `group`'s open phase this worker can claim, its own share first; false if it
/// claimed none. The worker that steps the phase's last environments ends the phase.
fn drain(shared: &Shared, group: usize, me: usize) -> bool {
    let job = &shared.jobs[group];
    let envs = shared.groups[group].len();
    let mut stepped = 0;
    for k in 0..shared.shares {
        let s = (me + k) % shared.shares;
        let end = share(envs, shared.shares, s).end;
        loop {
            // Read first: at many threads most shares a worker looks at are done, and a read leaves the
            // counter's cache line with the others where a claim takes it. A stale read only skips a share
            // the epoch then sends this worker back to (`wait_for_work`).
            if job.next[s].0.load(Ordering::Acquire) >= end {
                break;
            }
            let index = job.next[s].0.fetch_add(1, Ordering::AcqRel);
            if index >= end {
                break;
            }
            // SAFETY: a successful claim (see `Job`).
            let task = unsafe { *job.task.get() }.expect("an open phase has its task");
            step_env(shared, group, index, task);
            stepped += 1;
        }
    }
    if stepped == 0 {
        return false;
    }
    if job.done.fetch_add(stepped, Ordering::AcqRel) + stepped == envs {
        end_phase(shared, group);
    }
    true
}

fn step_env(shared: &Shared, group: usize, index: usize, task: Task) {
    let Task { work, buffers } = task;
    let result = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        if *shared.fail_at.lock().unwrap() == Some(index) {
            panic!("failing on purpose");
        }
        let (first, end) = (shared.rows[group][index], shared.rows[group][index + 1]);
        let mut env = shared.groups[group][index]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: environment `index` of this group owns agents [first, end), disjoint from the others'.
        let (mut rows, actions) = unsafe { buffers.rows(first, end - first) };
        match work {
            Work::Observe => env.write_observations(&mut rows),
            Work::Step => env.step(actions, &mut rows),
        }
    }));
    if let Err(panic) = result {
        let failure = format!(
            "environment {index} of group {group} failed: {}",
            panic_message(&*panic)
        );
        fail(&mut shared.schedule(), group, failure);
    }
}

/// After every environment of `group` stepped: the group is free again.
fn end_phase(shared: &Shared, group: usize) {
    let mut schedule = shared.schedule();
    schedule.busy[group] = false;
    drop(schedule);
    shared.schedule.changed.notify_all();
}

/// Records a failure: the group's first, and the pool is broken (a failed environment may be half-stepped).
fn fail(schedule: &mut Schedule, group: usize, failure: String) {
    schedule.broken.get_or_insert_with(|| failure.clone());
    schedule.failures[group].get_or_insert(failure);
}

/// A panic's message, or "unknown error".
pub(crate) fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("unknown error")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::NUM_ACTIONS;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use crate::driver::PoolDriver;
    use crate::testing::{meshes_loaded, pool_config};

    /// A pool stepped with seeded random actions.
    struct RandomPlay {
        driver: PoolDriver,
        rng: fastrand::Rng,
    }

    impl RandomPlay {
        fn step(&mut self) {
            let rng = &mut self.rng;
            self.driver.step(|actions| randomise(rng, actions)).unwrap();
        }

        fn step_pipelined(&mut self) {
            let rng = &mut self.rng;
            self.driver
                .step_pipelined(|actions| randomise(rng, actions))
                .unwrap();
        }
    }

    fn randomise(rng: &mut fastrand::Rng, actions: &mut [i32]) {
        actions.fill_with(|| rng.i32(0..NUM_ACTIONS as i32));
    }

    fn play(num_arenas: usize, groups: usize, threads: usize, seed: u64) -> RandomPlay {
        RandomPlay {
            driver: PoolDriver::new(&pool_config(num_arenas, groups, threads, seed)).unwrap(),
            rng: fastrand::Rng::with_seed(7),
        }
    }

    fn assert_same(a: &RandomPlay, b: &RandomPlay) {
        for (x, y) in a.driver.buffers.iter().zip(&b.driver.buffers) {
            assert_eq!(x.obs, y.obs);
            assert_eq!(x.mask, y.mask);
            assert_eq!(x.reward, y.reward);
            assert_eq!(x.terminal, y.terminal);
            assert_eq!(x.truncated, y.truncated);
            assert_eq!(x.episode, y.episode);
            assert_eq!(x.step, y.step);
            assert_eq!(x.final_obs, y.final_obs);
        }
    }

    #[test]
    fn same_seed_gives_identical_runs_regardless_of_thread_count() {
        if !meshes_loaded() {
            return;
        }
        let mut one_thread = play(6, 2, 1, 42);
        let mut three_threads = play(6, 2, 3, 42);
        assert_same(&one_thread, &three_threads);
        for _ in 0..300 {
            one_thread.step();
            three_threads.step();
            assert_same(&one_thread, &three_threads);
        }
        let (one, three) = (
            one_thread.driver.pool.take_stats(),
            three_threads.driver.pool.take_stats(),
        );
        assert_eq!(one.rewards, three.rewards);
        assert_eq!(one.steps, 300 * 6);
        assert!(one.physics_time > Duration::ZERO && one.physics_time < one.step_time);
    }

    #[test]
    fn pipelined_driving_gives_the_same_games_as_lockstep() {
        if !meshes_loaded() {
            return;
        }
        let mut lockstep = play(6, 3, 4, 8);
        let mut pipelined = play(6, 3, 4, 8);
        for _ in 0..200 {
            lockstep.step();
            pipelined.step_pipelined();
        }
        pipelined.driver.finish().unwrap();
        assert_same(&lockstep, &pipelined);
    }

    #[test]
    fn different_seeds_give_different_runs() {
        if !meshes_loaded() {
            return;
        }
        let mut a = play(2, 1, 2, 1);
        let mut b = play(2, 1, 2, 2);
        for _ in 0..20 {
            a.step();
            b.step();
        }
        assert_ne!(a.driver.buffers[0].obs, b.driver.buffers[0].obs);
    }

    #[test]
    fn every_agent_sees_consecutive_steps_of_one_episode_until_it_ends() {
        if !meshes_loaded() {
            return;
        }
        let mut game = play(8, 2, 4, 3);
        let mut endings = 0;
        for _ in 0..1500 {
            let before: Vec<(Vec<i64>, Vec<i32>)> = game
                .driver
                .buffers
                .iter()
                .map(|b| (b.episode.clone(), b.step.clone()))
                .collect();
            game.step();
            for (owned, (episodes, steps)) in game.driver.buffers.iter().zip(&before) {
                for agent in 0..owned.episode.len() {
                    assert!(owned.terminal[agent] + owned.truncated[agent] <= 1);
                    if owned.terminal[agent] == 1 || owned.truncated[agent] == 1 {
                        endings += 1;
                        assert_ne!(
                            owned.episode[agent], episodes[agent],
                            "a new episode must start"
                        );
                        assert_eq!(owned.step[agent], 0);
                    } else {
                        assert_eq!(
                            owned.episode[agent], episodes[agent],
                            "episode changed without an ending"
                        );
                        assert_eq!(
                            owned.step[agent],
                            steps[agent] + 1,
                            "skipped or repeated a step"
                        );
                    }
                }
            }
        }
        assert!(
            endings > 50,
            "only {endings} episode endings; the test needs many"
        );
    }

    #[test]
    fn observations_and_rewards_stay_finite_and_some_action_is_always_allowed() {
        if !meshes_loaded() {
            return;
        }
        let mut game = play(4, 2, 2, 9);
        for _ in 0..500 {
            game.step();
            for owned in &game.driver.buffers {
                assert!(owned.obs.iter().chain(&owned.reward).all(|v| v.is_finite()));
                assert!(owned.mask.chunks(NUM_ACTIONS).all(|m| m.contains(&1)));
            }
        }
    }

    /// A padded pool of `num_arenas` in 2 groups on 3 threads, each arena `sizes` (empty: all 3v3).
    fn sized_config(num_arenas: usize, sizes: &[usize], shuffle: bool) -> PoolConfig {
        let mut config = pool_config(num_arenas, 2, 3, 23);
        (config.team_size, config.obs_team_size) = (3, 3);
        config.team_sizes = sizes.to_vec();
        config.shuffle_slots = shuffle;
        config
    }

    /// Arena `arena`'s rows of every buffer (in group `arena % groups`), from a pool's per-arena sizes.
    fn arena_span(sizes: &[usize], groups: usize, arena: usize) -> (usize, Range<usize>) {
        let group = arena % groups;
        let before: usize = (group..arena).step_by(groups).map(|a| 2 * sizes[a]).sum();
        (group, before..before + 2 * sizes[arena])
    }

    /// Steps `driver` with an action per car that depends only on its arena, car and the step.
    fn step_by_arena(driver: &mut PoolDriver, step: u64) {
        let sizes = driver.pool.arena_team_sizes().to_vec();
        let groups = driver.pool.num_groups();
        let mut group = 0;
        driver
            .step(|actions| {
                for arena in (group..sizes.len()).step_by(groups) {
                    let (_, rows) = arena_span(&sizes, groups, arena);
                    for (car, row) in rows.enumerate() {
                        let mut rng =
                            fastrand::Rng::with_seed(mix_seed(step, (arena * 8 + car) as u64));
                        actions[row] = rng.i32(0..NUM_ACTIONS as i32);
                    }
                }
                group += 1;
            })
            .unwrap();
    }

    #[test]
    fn mixed_game_sizes_give_each_arena_its_own_cars_and_rows() {
        if !meshes_loaded() {
            return;
        }
        let sizes = [1, 2, 3];
        let mut mixed = PoolDriver::new(&sized_config(12, &sizes, false)).unwrap();
        let arena_sizes = mixed.pool.arena_team_sizes().to_vec();
        assert_eq!(arena_sizes, [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]);
        // Both groups hold one arena of each size twice: 2 x (2 + 4 + 6) rows each, the same shape.
        assert_eq!(mixed.pool.shape(0), mixed.pool.shape(1));
        assert_eq!(mixed.pool.shape(0).agents, 24);
        // Each arena plays exactly the game the same arena plays in a pool of its size alone.
        let mut alone: Vec<PoolDriver> = sizes
            .iter()
            .map(|&n| {
                let mut config = sized_config(12, &[], false);
                config.team_size = n;
                PoolDriver::new(&config).unwrap()
            })
            .collect();
        let obs = mixed.pool.shape(0).obs_size;
        for step in 0..120 {
            step_by_arena(&mut mixed, step);
            for driver in &mut alone {
                step_by_arena(driver, step);
            }
            for (arena, &n) in arena_sizes.iter().enumerate() {
                let (group, rows) = arena_span(&arena_sizes, 2, arena);
                let own = &alone[n - 1];
                let (_, own_rows) = arena_span(own.pool.arena_team_sizes(), 2, arena);
                let (a, b) = (&mixed.buffers[group], &own.buffers[group]);
                let at = format!("arena {arena} ({n}v{n}) at step {step}");
                assert_eq!(a.reward[rows.clone()], b.reward[own_rows.clone()], "{at}");
                assert_eq!(a.episode[rows.clone()], b.episode[own_rows.clone()], "{at}");
                assert_eq!(
                    a.terminal[rows.clone()],
                    b.terminal[own_rows.clone()],
                    "{at}"
                );
                let span = |r: &Range<usize>| r.start * obs..r.end * obs;
                assert_eq!(a.obs[span(&rows)], b.obs[span(&own_rows)], "{at}");
                assert_eq!(
                    a.final_obs[span(&rows)],
                    b.final_obs[span(&own_rows)],
                    "{at}"
                );
            }
        }
        let stats = mixed.pool.take_stats();
        assert_eq!(
            stats.play_by_size.len(),
            4,
            "one entry per size up to the largest"
        );
        let by_size: Vec<u64> = stats.play_by_size.iter().map(|p| p.car_steps).collect();
        // Four arenas of each size, every car counted, 120 steps.
        assert_eq!(by_size, [0, 4 * 2 * 120, 4 * 4 * 120, 4 * 6 * 120]);
        assert_eq!(stats.play.car_steps, by_size.iter().sum::<u64>());
        assert!(
            alone[0].pool.take_stats().play_by_size.is_empty(),
            "one size: no per-size statistics"
        );
    }

    #[test]
    fn shuffled_slots_move_cars_between_places_without_changing_the_games() {
        if !meshes_loaded() {
            return;
        }
        let mut plain = PoolDriver::new(&sized_config(4, &[1, 2], false)).unwrap();
        let mut shuffled = PoolDriver::new(&sized_config(4, &[1, 2], true)).unwrap();
        let layout = plain.pool.obs_layout();
        let (own, block) = (layout.self_part, layout.other_car);
        let mut moved = 0;
        for step in 0..300 {
            step_by_arena(&mut plain, step);
            step_by_arena(&mut shuffled, step);
            for (a, b) in plain.buffers.iter().zip(&shuffled.buffers) {
                assert_eq!(a.reward, b.reward, "the same games");
                assert_eq!(a.episode, b.episode);
                for (x, y) in a.obs.chunks(layout.size).zip(b.obs.chunks(layout.size)) {
                    assert_eq!(x[..own], y[..own], "a car's own part never moves");
                    // Teammates' places, then opponents': the same blocks, in some order.
                    for (start, places) in [
                        (own, layout.teammates),
                        (own + block * layout.teammates, layout.opponents),
                    ] {
                        let mut xs: Vec<&[f32]> =
                            x[start..start + places * block].chunks(block).collect();
                        let mut ys: Vec<&[f32]> =
                            y[start..start + places * block].chunks(block).collect();
                        moved += usize::from(xs != ys);
                        xs.sort_by(|p, q| p.partial_cmp(q).unwrap());
                        ys.sort_by(|p, q| p.partial_cmp(q).unwrap());
                        assert_eq!(xs, ys);
                    }
                }
            }
        }
        assert!(moved > 100, "slots were shuffled only {moved} times");
    }

    #[test]
    fn misuse_is_refused() {
        if !meshes_loaded() {
            return;
        }
        let mut game = play(2, 1, 1, 1);
        let shape = game.driver.pool.shape(0);
        let view = game.driver.buffers[0].view(shape);
        // The test holds an arena of the group, so its step cannot finish before the second submit.
        let held = game.driver.pool.shared.groups[0][0].lock().unwrap();
        game.driver.pool.submit(0, Work::Step, view).unwrap();
        assert!(
            game.driver.pool.submit(0, Work::Step, view).is_err(),
            "double submit must be refused"
        );
        drop(held);
        game.driver.pool.wait(0).unwrap();
        let wrong_shape = BufferShape {
            agents: shape.agents + 2,
            ..shape
        };
        game.driver.buffers[0].actions[1] = NUM_ACTIONS as i32;
        let view = game.driver.buffers[0].view(shape);
        let err = game.driver.pool.submit(0, Work::Step, view).unwrap_err();
        assert!(err.contains("action 90"), "{err}");
        assert!(
            !game.driver.pool.is_busy(0),
            "a refused step must not start"
        );
        game.driver.buffers[0].actions[1] = 0;
        game.driver.pool.submit(0, Work::Step, view).unwrap();
        game.driver.pool.wait(0).unwrap();
        let mut wrong = OwnedBuffers::new(wrong_shape);
        assert!(
            game.driver
                .pool
                .submit(0, Work::Step, wrong.view(wrong_shape))
                .is_err()
        );
    }

    /// A panic in an environment's step breaks the pool: the step says which environment failed, the group
    /// is free again and no further work is accepted.
    #[test]
    fn a_failed_step_stops_the_pool() {
        if !meshes_loaded() {
            return;
        }
        let mut game = RandomPlay {
            driver: PoolDriver::new(&pool_config(3, 1, 2, 3)).unwrap(),
            rng: fastrand::Rng::with_seed(1),
        };
        *game.driver.pool.shared.fail_at.lock().unwrap() = Some(1);
        let rng = &mut game.rng;
        let err = game
            .driver
            .step(|actions| randomise(rng, actions))
            .unwrap_err();
        assert_eq!(err, "environment 1 of group 0 failed: failing on purpose");
        assert!(!game.driver.pool.is_busy(0));
        let shape = game.driver.pool.shape(0);
        let view = game.driver.buffers[0].view(shape);
        let refused = game.driver.pool.submit(0, Work::Step, view).unwrap_err();
        assert!(
            refused.contains("stopped after an internal error"),
            "{refused}"
        );
    }

    /// The pool's statistics, each environment's taken in place, are those of the old `take_stats`
    /// (each environment's swapped for new ones), take after take: the same records in the same order.
    #[test]
    fn statistics_taken_in_place_are_those_taken_the_old_way() {
        if !meshes_loaded() {
            return;
        }
        fn the_old_way(pool: &EnvPool) -> EnvStats {
            let mut total = EnvStats::new(pool.reward_labels.len());
            for env in pool.shared.groups.iter().flatten() {
                total.merge(&env.lock().unwrap().take_stats());
            }
            total
        }
        // Their step times differ; as text, so NaNs (never touched) compare too.
        let untimed = |mut stats: EnvStats| {
            (stats.step_time, stats.physics_time) = (Duration::ZERO, Duration::ZERO);
            format!("{stats:?}")
        };
        let (mut in_place, mut replaced) = (play(8, 2, 3, 11), play(8, 2, 2, 11));
        let mut episodes = 0;
        for take in 0..6 {
            for _ in 0..120 {
                in_place.step();
                replaced.step();
            }
            let stats = in_place.driver.pool.take_stats();
            episodes += stats.episodes.len();
            assert_eq!(
                untimed(stats),
                untimed(the_old_way(&replaced.driver.pool)),
                "take {take}"
            );
        }
        assert!(episodes > 10, "only {episodes} episodes");
    }

    fn waking(threads: usize) -> RandomPlay {
        let config = pool_config(6, 3, threads, 8);
        RandomPlay {
            driver: PoolDriver::new(&config).unwrap(),
            rng: fastrand::Rng::with_seed(7),
        }
    }

    /// 12,000 submits and waits over three groups with random gaps of 0 to 200 µs (the workers asleep
    /// after some, spinning through others): no hang (a watchdog fails the test after 5 minutes) and the
    /// same buffers at one thread count as at another, after every wait.
    #[test]
    fn many_submits_and_waits_with_random_gaps_never_hang() {
        if !meshes_loaded() {
            return;
        }
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let gap = |rng: &mut fastrand::Rng| {
                let until = Instant::now() + Duration::from_micros(rng.u64(0..=200));
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            };
            let mut pools = [waking(1), waking(3), waking(4)];
            let (mut actions, mut gaps) =
                (fastrand::Rng::with_seed(3), fastrand::Rng::with_seed(4));
            for cycle in 0..12_000 {
                let group = cycle % 3;
                for play in &pools {
                    play.driver.pool.wait(group).unwrap();
                }
                for other in &pools[1..] {
                    let (x, y) = (
                        &pools[0].driver.buffers[group],
                        &other.driver.buffers[group],
                    );
                    assert_eq!(x.obs, y.obs, "cycle {cycle}");
                    assert_eq!(x.reward, y.reward, "cycle {cycle}");
                    assert_eq!(x.episode, y.episode, "cycle {cycle}");
                }
                let picked: Vec<i32> = (0..pools[0].driver.buffers[group].actions.len())
                    .map(|_| actions.i32(0..NUM_ACTIONS as i32))
                    .collect();
                for play in &mut pools {
                    let owned = &mut play.driver.buffers[group];
                    owned.actions.copy_from_slice(&picked);
                    let view = owned.view(play.driver.pool.shape(group));
                    play.driver.pool.submit(group, Work::Step, view).unwrap();
                    gap(&mut gaps);
                }
            }
            for play in &pools {
                play.driver.finish().unwrap();
            }
            done.send(()).unwrap();
        });
        match finished.recv_timeout(Duration::from_secs(300)) {
            Ok(()) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("the pools hung"),
            Err(err) => panic!("the stress test failed ({err}); see its panic above"),
        }
    }

    /// A `Shared` with no games, for watching how its workers sleep and wake.
    fn bare_shared(workers: usize) -> Shared {
        Shared {
            groups: Vec::new(),
            rows: Vec::new(),
            jobs: Vec::new(),
            shares: workers,
            spin: Duration::ZERO,
            epoch: Padded(AtomicU64::new(0)),
            sleepers: Padded(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
            sleep: Locked::new(()),
            schedule: Locked::new(Schedule {
                busy: Vec::new(),
                failures: Vec::new(),
                broken: None,
            }),
            fail_at: Mutex::new(None),
        }
    }

    /// Every sleeping worker wakes for every epoch: 16 and 40 workers that sleep at once, woken 3,000
    /// times, sometimes all asleep, sometimes while they fall asleep, with up to four wakes in a row (a few
    /// microseconds apart); a worker that missed a wake would still hold an old epoch at the deadline.
    #[test]
    fn no_sleeping_worker_misses_a_wake() {
        for workers in [16, 40] {
            let shared = Arc::new(bare_shared(workers));
            let seen: Arc<Vec<AtomicU64>> =
                Arc::new((0..workers).map(|_| AtomicU64::new(0)).collect());
            let threads: Vec<_> = (0..workers)
                .map(|me| {
                    let (shared, seen) = (Arc::clone(&shared), Arc::clone(&seen));
                    std::thread::spawn(move || {
                        loop {
                            let epoch = shared.epoch.load(Ordering::SeqCst);
                            seen[me].store(epoch, Ordering::SeqCst);
                            if shared.shutdown.load(Ordering::SeqCst) {
                                return;
                            }
                            shared.wait_for_work(epoch);
                        }
                    })
                })
                .collect();
            let asleep = || shared.sleepers.load(Ordering::SeqCst);
            let mut rng = fastrand::Rng::with_seed(9);
            let mut epoch = 0;
            for round in 1..=3_000u64 {
                let deadline = Instant::now() + Duration::from_secs(20);
                if rng.bool() {
                    while asleep() < workers {
                        assert!(Instant::now() < deadline, "round {round}: never all asleep");
                        std::thread::yield_now();
                    }
                }
                for wake in 0..rng.u32(1..=4) {
                    if wake > 0 {
                        let gap = Instant::now() + Duration::from_micros(rng.u64(0..20));
                        while Instant::now() < gap {
                            std::hint::spin_loop();
                        }
                    }
                    shared.wake();
                    epoch += 1;
                }
                while let Some(me) =
                    (0..workers).find(|&me| seen[me].load(Ordering::SeqCst) < epoch)
                {
                    assert!(
                        Instant::now() < deadline,
                        "{workers} workers, round {round}: worker {me} missed the wake"
                    );
                    std::thread::yield_now();
                }
            }
            shared.stop();
            for thread in threads {
                thread.join().unwrap();
            }
        }
    }

    /// Each word that many threads write (the epoch, the sleepers' count, the sleeping workers' lock
    /// and condvar, the schedule's, each group's count of environments stepped) sits alone on its 128-byte
    /// lines.
    #[test]
    fn every_word_threads_write_sits_alone_on_its_cache_lines() {
        use std::mem::{align_of, offset_of, size_of, size_of_val};
        const LINE: usize = 128;
        fn alone(fields: &[(&str, usize, usize, bool)]) {
            for &(name, offset, size, _) in fields.iter().filter(|field| field.3) {
                let lines = offset / LINE * LINE..(offset + size).div_ceil(LINE) * LINE;
                assert_eq!(offset % LINE, 0, "{name} starts a line");
                for &(other, at, length, _) in fields {
                    assert!(
                        other == name || at + length <= lines.start || at >= lines.end,
                        "{other} shares a line with {name}"
                    );
                }
            }
        }
        let shared = bare_shared(1);
        alone(&[
            (
                "groups",
                offset_of!(Shared, groups),
                size_of_val(&shared.groups),
                false,
            ),
            (
                "rows",
                offset_of!(Shared, rows),
                size_of_val(&shared.rows),
                false,
            ),
            (
                "jobs",
                offset_of!(Shared, jobs),
                size_of_val(&shared.jobs),
                false,
            ),
            (
                "shares",
                offset_of!(Shared, shares),
                size_of::<usize>(),
                false,
            ),
            (
                "spin",
                offset_of!(Shared, spin),
                size_of::<Duration>(),
                false,
            ),
            (
                "epoch",
                offset_of!(Shared, epoch),
                size_of_val(&shared.epoch),
                true,
            ),
            (
                "sleepers",
                offset_of!(Shared, sleepers),
                size_of_val(&shared.sleepers),
                true,
            ),
            (
                "shutdown",
                offset_of!(Shared, shutdown),
                size_of::<AtomicBool>(),
                false,
            ),
            (
                "sleep",
                offset_of!(Shared, sleep),
                size_of_val(&shared.sleep),
                true,
            ),
            (
                "schedule",
                offset_of!(Shared, schedule),
                size_of_val(&shared.schedule),
                true,
            ),
            (
                "fail_at",
                offset_of!(Shared, fail_at),
                size_of_val(&shared.fail_at),
                false,
            ),
        ]);
        alone(&[
            (
                "task",
                offset_of!(Job, task),
                size_of::<UnsafeCell<Option<Task>>>(),
                false,
            ),
            (
                "next",
                offset_of!(Job, next),
                size_of::<Vec<Counter>>(),
                false,
            ),
            (
                "done",
                offset_of!(Job, done),
                size_of::<Padded<AtomicUsize>>(),
                true,
            ),
        ]);
        // Their lines start where the memory holding them does (an `Arc`'s, a vector's), which their
        // alignment puts on a line too; one share's counter fills exactly a line.
        assert!(align_of::<Shared>() >= LINE && align_of::<Job>() >= LINE);
        assert_eq!((size_of::<Counter>(), align_of::<Counter>()), (LINE, LINE));
    }
}
