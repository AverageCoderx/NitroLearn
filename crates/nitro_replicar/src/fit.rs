//! The timing fit, a pass after replicar's own conversion: replicar knows what each car
//! pressed and the server's updates of it, but not always the exact tick. For each car, between its server
//! updates, the car is simulated alone in RocketSim and three kinds of timing are moved to bring it closest to
//! its next update: jump presses and releases (up to `jump_range` ticks), the ticks the updates land on
//! (`update_range`; past a jump or flip only when the car's jump flags there are the update's), and the tick
//! each throttle, steer, handbrake and boost change takes effect (`control_range`, at most `moves_per_gap`
//! changes moved between two updates). Coordinate descent, `rounds` rounds, as moving one changes what is best
//! for the others. No value is changed, only when it applies: the replay's controls and updates stay as they
//! are. Deterministic. Stretches where the replay has the car within reach of the ball or another car, or
//! without an update for `MAX_GAP` ticks, stay as replicar made them.

use replicar::rocketsim::{CarControls, CarState};
use replicar_format::header::PlayerInfo;
use replicar_format::record::Frame;

use crate::car::{self, Alone};
use crate::track::{self, Run};

/// The fit's limits; `DEFAULT` is what conversions use.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub jump_range: usize,
    pub update_range: usize,
    pub control_range: usize,
    pub moves_per_gap: usize,
    pub rounds: usize,
}

impl Settings {
    pub const DEFAULT: Self = Self {
        jump_range: 8,
        update_range: 4,
        control_range: 4,
        moves_per_gap: 3,
        rounds: 5,
    };
}

/// Updates further apart than this (ticks) are a gap the fit leaves alone.
pub const MAX_GAP: usize = 24;
/// Ticks a moved jump press or release keeps from its neighbours, or as replicar had them if closer.
const JUMP_APART: usize = 4;
/// The smallest gain (uu, summed over a window's updates) that moves anything: float noise moves nothing.
const MIN_GAIN: f64 = 1e-3;

/// What the fit did, summed over the cars.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub runs: usize,
    /// Updates whose error counts (the gap before them free of contacts), and their summed error (uu) before
    /// and after.
    pub updates_counted: usize,
    pub error_before: f64,
    pub error_after: f64,
    /// Per kind (updates, jump presses and releases, control changes): how many could move, how many moved.
    pub updates: [usize; 2],
    pub jump_edges: [usize; 2],
    pub control_changes: [usize; 2],
    pub rounds: usize,
}

impl Report {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "version": VERSION,
            "runs": self.runs,
            "updates_counted": self.updates_counted,
            "mean_update_error_before": self.error_before / self.updates_counted.max(1) as f64,
            "mean_update_error_after": self.error_after / self.updates_counted.max(1) as f64,
            "updates": {"free": self.updates[0], "moved": self.updates[1]},
            "jump_edges": {"free": self.jump_edges[0], "moved": self.jump_edges[1]},
            "control_changes": {"free": self.control_changes[0], "moved": self.control_changes[1]},
            "rounds": self.rounds,
        })
    }
}

/// The fit's version, recorded in each fitted file: a change to the method gives a new number.
pub const VERSION: u32 = 1;

/// A change of one control: rows `start..start + len` (a ramp of an analog stick is one change) moved by
/// `shift`, their values as they were.
#[derive(Debug, Clone)]
struct Change {
    start: usize,
    len: usize,
    shift: isize,
    values: Vec<f32>,
    /// The update gap it began in (moves per gap).
    gap: usize,
    /// A jump press's effect in replicar's rows (its own simulation, with every car and the ball): the car's
    /// jump flags on its row and two rows on. A press that changes none stays where replicar put it: the lone
    /// car cannot place it, and moving it on the lone car's differences made presses that do nothing land on
    /// the ground after a jump, labelled on the decision after a jump's.
    effect: Option<Effect>,
}

type Flags = (bool, bool, bool);
type Effect = (Flags, Flags);

/// What a candidate's summed error must stay under.
#[derive(Debug, Clone, Copy)]
struct Limit {
    bound: f64,
}

impl Change {
    fn at(&self) -> usize {
        self.start.wrapping_add_signed(self.shift)
    }
    fn end(&self) -> usize {
        self.at() + self.len - 1
    }
}

/// One control's changes. A control's value on a row: the change covering it, else the last value of the
/// change before it, else `initial`.
#[derive(Debug, Clone)]
struct Control {
    initial: f32,
    changes: Vec<Change>,
    range: usize,
    /// Rows of the held value kept between two changes (1 for analog controls: two ramps never merge).
    separation: usize,
    /// A move never brings a change closer to its neighbour than this or than replicar had them, whichever
    /// is less (jump: one replay frame, 4 ticks). Without it the fit turned a release and the next press
    /// replicar had frames apart into a one-tick release after a long hold, which a bot holding each action
    /// for a decision cannot do (the decision before a press labelled jump far more often).
    keep_apart: usize,
    /// Counted against `moves_per_gap`.
    counted: bool,
}

impl Control {
    fn value(&self, k: usize) -> f32 {
        let j = self.changes.partition_point(|c| c.at() <= k);
        if j == 0 {
            return self.initial;
        }
        let c = &self.changes[j - 1];
        c.values[(k - c.at()).min(c.len - 1)]
    }

    /// Whether change `j` may start at `at` in a run of `n` rows: inside the run after its first row, in order.
    fn allows(&self, j: usize, at: usize, n: usize) -> bool {
        let c = &self.changes[j];
        let apart = |before: &Change, after: &Change, new_gap: usize| {
            new_gap >= self.keep_apart.min(after.start - before.start)
        };
        at >= 1
            && at + c.len <= n
            && (j == 0 || {
                let before = &self.changes[j - 1];
                at > before.end() + self.separation && apart(before, c, at - before.at())
            })
            && self.changes.get(j + 1).is_none_or(|next| {
                next.at() > at + c.len - 1 + self.separation && apart(c, next, next.at() - at)
            })
    }
}

const THROTTLE: usize = 0;
const STEER: usize = 1;
const HANDBRAKE: usize = 2;
const BOOST: usize = 3;
const JUMP: usize = 4;

fn flag(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

fn channel(c: &CarControls, which: usize) -> f32 {
    match which {
        THROTTLE => c.throttle,
        STEER => c.steer,
        HANDBRAKE => flag(c.handbrake),
        BOOST => flag(c.boost),
        _ => flag(c.jump),
    }
}

#[derive(Debug, Clone, Copy)]
enum Var {
    Update(usize),
    Change(usize, usize),
}

/// One run of one car being fitted (positions are rows of the run).
struct Fit {
    settings: Settings,
    base: Vec<CarState>,
    original: Vec<CarControls>,
    updates: Vec<usize>,
    shift: Vec<isize>,
    /// Per gap between updates `g` and `g + 1`: left to replicar.
    rigid: Vec<bool>,
    /// Rigid rows counted up to each row (a range holds one when the counts differ).
    rigid_rows: Vec<u32>,
    controls: [Control; 5],
    /// The car's state just after each update (snapped to it) in the current solution.
    chain: Vec<CarState>,
    moved_in_gap: Vec<usize>,
    /// Updates whose window saw a move in the last round and in this one: a round after the first visits only
    /// the variables whose window holds one (the others would choose as before).
    touched: Vec<bool>,
    touched_now: Vec<bool>,
}

impl Fit {
    fn new(frames: &[Frame], run: &Run, settings: Settings) -> Result<Self, String> {
        let p = run.player;
        let car = |k: usize| {
            frames[run.rows[k]].state.cars[p]
                .as_ref()
                .expect("an active car")
        };
        let n = run.rows.len();
        let base: Vec<CarState> = (0..n)
            .map(|k| car::state_of(car(k)))
            .collect::<Result<_, _>>()?;
        let original: Vec<CarControls> = (0..n).map(|k| car::controls(&car(k).controls)).collect();
        let u = &run.updates;
        let rigid: Vec<bool> = u
            .windows(2)
            .map(|w| {
                w[1] - w[0] > MAX_GAP
                    || (w[0]..=w[1]).any(|k| car::near_contact(&frames[run.rows[k]], p))
            })
            .collect();
        let mut in_rigid = vec![false; n];
        for (g, w) in u.windows(2).enumerate() {
            if rigid[g] {
                in_rigid[w[0]..=w[1]].fill(true);
            }
        }
        // Rows before the first update and after the last are never moved into.
        in_rigid[..u[0]].fill(true);
        in_rigid[u[u.len() - 1]..].fill(true);
        let mut rigid_rows = Vec::with_capacity(n + 1);
        rigid_rows.push(0);
        for &r in &in_rigid {
            rigid_rows.push(rigid_rows.last().copied().unwrap_or(0) + u32::from(r));
        }
        let gap_of = |k: usize| u.partition_point(|&x| x <= k).saturating_sub(1);
        let make = |which: usize| {
            let jump = which == JUMP;
            let mut changes: Vec<Change> = Vec::new();
            for k in 1..n {
                let v = channel(&original[k], which);
                if v.to_bits() == channel(&original[k - 1], which).to_bits() {
                    continue;
                }
                match changes.last_mut() {
                    Some(c) if !jump && c.start + c.len == k => {
                        c.len += 1;
                        c.values.push(v);
                    }
                    _ => changes.push(Change {
                        start: k,
                        len: 1,
                        shift: 0,
                        values: vec![v],
                        gap: gap_of(k),
                        effect: (jump && v > 0.5).then(|| {
                            let after = &base[(k + 2).min(n - 1)];
                            (car::jump_flags(&base[k]), car::jump_flags(after))
                        }),
                    }),
                }
            }
            Control {
                initial: channel(&original[0], which),
                changes,
                range: if jump {
                    settings.jump_range
                } else {
                    settings.control_range
                },
                separation: usize::from(!jump),
                keep_apart: if jump { JUMP_APART } else { 0 },
                counted: !jump,
            }
        };
        let controls = [
            make(THROTTLE),
            make(STEER),
            make(HANDBRAKE),
            make(BOOST),
            make(JUMP),
        ];
        let m = u.len();
        let mut fit = Self {
            settings,
            chain: vec![base[u[0]]; m],
            base,
            original,
            updates: u.clone(),
            shift: vec![0; m],
            rigid,
            rigid_rows,
            controls,
            moved_in_gap: vec![0; m],
            touched: vec![true; m],
            touched_now: vec![false; m],
        };
        fit.chain[0] = fit.reset(0);
        Ok(fit)
    }

    fn pos(&self, i: usize) -> usize {
        self.updates[i].wrapping_add_signed(self.shift[i])
    }

    fn any_rigid(&self, lo: usize, hi: usize) -> bool {
        self.rigid_rows[hi] != self.rigid_rows[lo]
    }

    /// The effect replicar's rows show for the jump press on row `k` in the current solution, if one is there.
    fn press_at(&self, k: usize) -> Option<Effect> {
        let jumps = &self.controls[JUMP].changes;
        let j = jumps.partition_point(|e| e.at() <= k).checked_sub(1)?;
        let e = &jumps[j];
        (e.at() == k && e.values[0] > 0.5)
            .then_some(e.effect)
            .flatten()
    }

    /// The controls applied from row `k` in the current solution.
    fn controls_at(&self, k: usize) -> CarControls {
        let mut c = self.original[k];
        let v = |which: usize| self.controls[which].value(k);
        c.throttle = v(THROTTLE);
        c.steer = v(STEER);
        c.handbrake = v(HANDBRAKE) > 0.5;
        c.boost = v(BOOST) > 0.5;
        c.jump = v(JUMP) > 0.5;
        // A moved press takes its own pitch, yaw and roll along: RocketSim reads a dodge's way on its press tick.
        let jumps = &self.controls[JUMP].changes;
        let j = jumps.partition_point(|e| e.at() <= k);
        if let Some(e) = j.checked_sub(1).map(|j| &jumps[j])
            && e.at() == k
            && e.shift != 0
            && e.values[0] > 0.5
        {
            let from = &self.original[e.start];
            (c.pitch, c.yaw, c.roll) = (from.pitch, from.yaw, from.roll);
        }
        c
    }

    fn previous_controls(&self, k: usize) -> CarControls {
        if k == 0 {
            *self.base[0].prev_controls()
        } else {
            self.controls_at(k - 1)
        }
    }

    /// Replicar's state at update `i`, as a fresh start there.
    fn reset(&self, i: usize) -> CarState {
        let k = self.pos(i);
        let mut s = self.base[self.updates[i]];
        s.controls = self.controls_at(k);
        s.set_prev_controls(self.previous_controls(k));
        s
    }

    /// The car at update `i` from its simulated state there: the update's body and boost, the simulation's
    /// jump state; replicar's whole state where a stretch left to replicar begins or ends.
    fn snap(&self, i: usize, simulated: CarState) -> CarState {
        if self.rigid[i - 1] || self.rigid.get(i).copied().unwrap_or(false) {
            return self.reset(i);
        }
        let target = &self.base[self.updates[i]];
        let mut s = simulated;
        s.phys = target.phys;
        s.boost = target.boost;
        s
    }

    /// Simulates from `state` on row `lo` to update `to`, calling `visit` with each row's state (an update's
    /// after it is applied, with the update and its error) until it returns false. Returns the summed error at
    /// the updates after `lo` (uu). With a `limit` (a candidate): `None` when a moved update lands where the
    /// car's jump flags differ from the update's, a press does not do what replicar's rows show it doing (a
    /// move near a press changes it too: an earlier jump moved later left the car on the ground for the
    /// press after it), or the sum passes the limit's bound.
    fn simulate(
        &self,
        alone: &mut Alone,
        lo: usize,
        state: CarState,
        to: usize,
        limit: Option<Limit>,
        mut visit: impl FnMut(usize, &CarState, Option<(usize, f64)>) -> bool,
    ) -> Option<f64> {
        alone.start(state);
        let m = self.updates.len();
        let mut next = self.first_after(lo);
        let mut error = 0.0;
        let mut k = lo;
        // Presses on their way to showing their effect: (the row two on, the flags before, the effect).
        let mut pending: Vec<(usize, Flags, Effect)> = Vec::new();
        while next <= to && next < m {
            if limit.is_some()
                && let Some(effect) = self.press_at(k)
            {
                pending.push((k + 2, car::jump_flags(&alone.state()), effect));
            }
            alone.step(self.controls_at(k));
            k += 1;
            let mut s = alone.state();
            if let Some(&(_, before, effect)) = pending.first()
                && pending[0].0 == k
            {
                if (before, car::jump_flags(&s)) != effect {
                    return None;
                }
                pending.remove(0);
            }
            let mut update = None;
            if k == self.pos(next) {
                let i = next;
                let mut own = 0.0;
                if !self.rigid[i - 1] {
                    let target = &self.base[self.updates[i]];
                    if limit.is_some()
                        && self.shift[i] != 0
                        && car::jump_flags(&s) != car::jump_flags(target)
                    {
                        return None;
                    }
                    own = f64::from(s.phys.pos.distance(target.phys.pos));
                    error += own;
                    if limit.is_some_and(|l| error > l.bound) {
                        return None;
                    }
                }
                s = self.snap(i, s);
                alone.set(s);
                next += 1;
                update = Some((i, own));
            }
            if !visit(k, &s, update) {
                break;
            }
        }
        Some(error)
    }

    /// The car's state on row `k` (at or after update `from`'s row) in the current solution.
    fn state_at(&self, alone: &mut Alone, from: usize, k: usize) -> CarState {
        let start = self.pos(from);
        let mut out = self.chain[from];
        if k > start {
            self.simulate(
                alone,
                start,
                out,
                self.updates.len() - 1,
                None,
                |row, s, _| {
                    if row == k {
                        out = *s;
                    }
                    row < k
                },
            );
        }
        out
    }

    /// The chain again from update `from`, until it is as before past update `to`; returns the last update
    /// simulated.
    fn rechain(&mut self, alone: &mut Alone, from: usize, to: usize) -> usize {
        let m = self.updates.len();
        let mut fresh: Vec<(usize, CarState)> = Vec::new();
        let chain = &self.chain;
        self.simulate(
            alone,
            self.pos(from),
            chain[from],
            m - 1,
            None,
            |_, s, update| {
                let Some((i, _)) = update else { return true };
                fresh.push((i, *s));
                !(i >= to && car::same_internals(s, &chain[i]))
            },
        );
        let last = fresh.last().map_or(from, |(i, _)| *i);
        for (i, s) in fresh {
            self.chain[i] = s;
        }
        last
    }

    /// The window of a variable whose candidates touch rows `lo..hi`: the update to start from and the first at
    /// or after `hi`.
    fn window(&self, lo: usize, hi: usize) -> (usize, usize) {
        let m = self.updates.len();
        let from = self.first_after(lo).saturating_sub(1);
        let to = self
            .first_after(hi.saturating_sub(1))
            .max(from + 1)
            .min(m - 1);
        (from, to)
    }

    /// The first update on a row after `k` (`m` when none): update rows only grow along a run.
    fn first_after(&self, k: usize) -> usize {
        let (mut lo, mut hi) = (0, self.updates.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.pos(mid) <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn set(&mut self, var: Var, shift: isize) {
        match var {
            Var::Update(i) => self.shift[i] = shift,
            Var::Change(w, j) => self.controls[w].changes[j].shift = shift,
        }
    }

    fn shift_of(&self, var: Var) -> isize {
        match var {
            Var::Update(i) => self.shift[i],
            Var::Change(w, j) => self.controls[w].changes[j].shift,
        }
    }

    /// The rows a variable's candidates touch and its candidate shifts, or `None` when it stays where it is.
    fn candidates(&self, var: Var) -> Option<(usize, usize, Vec<isize>)> {
        let n = self.base.len();
        let order = |r: usize| {
            let r = r as isize;
            std::iter::once(0).chain((1..=r).flat_map(|d| [-d, d]))
        };
        match var {
            Var::Update(i) => {
                let r = self.settings.update_range;
                let u = self.updates[i];
                if i + 1 > self.rigid.len() || self.rigid[i - 1] || self.rigid[i] {
                    return None;
                }
                // From the row before the earliest candidate, whose step reaches it.
                let lo = u.checked_sub(r + 1)?;
                let hi = (u + r + 1).min(n);
                let (before, after) = (self.pos(i - 1), self.pos(i + 1));
                let shifts = order(r)
                    .filter(|&d| {
                        let at = u.wrapping_add_signed(d);
                        at > before && at < after
                    })
                    .collect();
                Some((lo, hi, shifts))
            }
            Var::Change(w, j) => {
                let control = &self.controls[w];
                let c = &control.changes[j];
                if c.effect.is_some_and(|(before, after)| before == after) {
                    return None;
                }
                let r = control.range;
                let lo = c.start.checked_sub(r)?;
                // A press's effect shows two rows on.
                let hi = c.start + c.len + r + if c.effect.is_some() { 2 } else { 0 };
                if hi > n || self.any_rigid(lo, hi) {
                    return None;
                }
                let limited = control.counted
                    && c.shift == 0
                    && self.moved_in_gap[c.gap] >= self.settings.moves_per_gap;
                let shifts = order(r)
                    .filter(|&d| d == 0 || !limited)
                    .filter(|&d| control.allows(j, c.start.wrapping_add_signed(d), n))
                    .collect();
                Some((lo, hi, shifts))
            }
        }
    }

    /// One coordinate step: the variable's best shift. Returns whether it moved.
    fn improve(&mut self, alone: &mut Alone, var: Var) -> bool {
        let Some((lo, hi, shifts)) = self.candidates(var) else {
            return false;
        };
        if shifts.len() < 2 {
            return false;
        }
        let (from, to) = self.window(lo, hi);
        if !self.touched[from..=to].iter().any(|t| *t) {
            return false;
        }
        let start_row = lo.max(self.pos(from));
        let start = self.state_at(alone, from, start_row);
        let current = self.shift_of(var);
        let cost = |fit: &Self, alone: &mut Alone, limit: Option<Limit>| {
            fit.simulate(alone, start_row, start, to, limit, |_, _, _| true)
        };
        let now = cost(self, alone, None).unwrap_or(f64::INFINITY);
        let mut best = (now, current);
        for &d in &shifts {
            if d == current {
                continue;
            }
            self.set(var, d);
            let limit = Limit {
                bound: best.0 - MIN_GAIN,
            };
            if let Some(e) = cost(self, alone, Some(limit))
                && e < best.0 - MIN_GAIN
            {
                best = (e, d);
            }
        }
        self.set(var, best.1);
        if best.1 == current {
            return false;
        }
        if let Var::Change(w, j) = var
            && self.controls[w].counted
        {
            let gap = self.controls[w].changes[j].gap;
            match (current == 0, best.1 == 0) {
                (true, false) => self.moved_in_gap[gap] += 1,
                (false, true) => self.moved_in_gap[gap] -= 1,
                _ => {}
            }
        }
        let last = self.rechain(alone, from, to);
        self.touched_now[from..=last].fill(true);
        true
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars: Vec<(usize, Var)> = (1..self.updates.len())
            .map(|i| (self.updates[i], Var::Update(i)))
            .collect();
        for (w, control) in self.controls.iter().enumerate() {
            vars.extend(
                control
                    .changes
                    .iter()
                    .enumerate()
                    .map(|(j, c)| (c.start, Var::Change(w, j))),
            );
        }
        vars.sort_by_key(|&(row, var)| (row, matches!(var, Var::Change(..))));
        vars.into_iter().map(|(_, v)| v).collect()
    }

    /// Each update's error in the current solution (0 for the first and after a stretch left to replicar).
    fn errors(&self, alone: &mut Alone) -> Vec<f64> {
        let m = self.updates.len();
        let mut out = vec![0.0; m];
        self.simulate(
            alone,
            self.pos(0),
            self.chain[0],
            m - 1,
            None,
            |_, _, update| {
                if let Some((i, e)) = update {
                    out[i] = e;
                }
                true
            },
        );
        out
    }

    /// The order of a round: the variables next to the largest errors first, so a badly placed update is moved
    /// before its neighbours make up for it; else in row order.
    fn round_order(&self, alone: &mut Alone, vars: &[Var]) -> Vec<Var> {
        let errors = self.errors(alone);
        let m = self.updates.len();
        let weight = |var: &Var| match *var {
            Var::Update(i) => errors[i] + errors.get(i + 1).copied().unwrap_or(0.0),
            Var::Change(w, j) => errors[(self.controls[w].changes[j].gap + 1).min(m - 1)],
        };
        let mut order: Vec<(f64, usize)> = vars
            .iter()
            .enumerate()
            .map(|(k, v)| (weight(v), k))
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        order.into_iter().map(|(_, k)| vars[k]).collect()
    }

    /// Writes the fitted rows: every gap whose controls or updates moved, or whose start the moves changed, is
    /// simulated again; the others keep replicar's rows.
    fn write(&self, alone: &mut Alone, run: &Run, frames: &mut [Frame]) {
        let p = run.player;
        let m = self.updates.len();
        let mut carried = false;
        for i in 0..m - 1 {
            let (a, b) = (self.pos(i), self.pos(i + 1));
            let moved = self.shift[i] != 0
                || self.shift[i + 1] != 0
                || (a..b).any(|k| !car::same_controls(&self.controls_at(k), &self.original[k]));
            let diverged =
                carried && !car::same_internals(&self.chain[i], &self.base[self.updates[i]]);
            carried = moved || diverged;
            if !carried {
                continue;
            }
            let mut rows: Vec<(usize, CarState)> = vec![(a, self.chain[i])];
            self.simulate(alone, a, self.chain[i], i + 1, None, |k, s, _| {
                if k < b {
                    rows.push((k, *s));
                }
                k < b
            });
            for (k, s) in rows {
                let frame = &mut frames[run.rows[k]];
                if let Some(Some(c)) = frame.state.cars.get_mut(p) {
                    car::write(&s, &self.controls_at(k), &self.previous_controls(k), c);
                }
            }
        }
        // All moved updates off their rows first: one moved onto the row another left must keep its flag.
        let moved: Vec<usize> = (0..m).filter(|&i| self.shift[i] != 0).collect();
        for &i in &moved {
            frames[run.rows[self.updates[i]]].updates.car_updated[p] = Some(false);
        }
        for &i in &moved {
            frames[run.rows[self.pos(i)]].updates.car_updated[p] = Some(true);
        }
    }
}

/// Fits every car's timing in `frames` (a conversion's frames, replicar's own) in place.
pub fn fit(
    players: &[PlayerInfo],
    frames: &mut [Frame],
    settings: Settings,
) -> Result<Report, String> {
    let mut report = Report::default();
    let runs = track::runs(frames, players.len());
    let mut arenas: Vec<Option<Alone>> = (0..players.len()).map(|_| None).collect();
    for run in &runs {
        let alone = match &mut arenas[run.player] {
            Some(a) => a,
            slot => slot.insert(Alone::new(&players[run.player])?),
        };
        let mut f = Fit::new(frames, run, settings)?;
        f.rechain(alone, 0, f.updates.len() - 1);
        let before: f64 = f.errors(alone).iter().sum();
        let counted = f.rigid.iter().filter(|r| !**r).count();
        let vars = f.vars();
        for &var in &vars {
            let free = f.candidates(var).is_some_and(|(_, _, s)| s.len() > 1);
            let slot = match var {
                Var::Update(_) => &mut report.updates,
                Var::Change(JUMP, _) => &mut report.jump_edges,
                Var::Change(..) => &mut report.control_changes,
            };
            slot[0] += usize::from(free);
        }
        let mut rounds = 0;
        for _ in 0..settings.rounds {
            rounds += 1;
            let mut changed = false;
            for var in f.round_order(alone, &vars) {
                changed |= f.improve(alone, var);
            }
            f.touched = std::mem::replace(&mut f.touched_now, vec![false; f.updates.len()]);
            if !changed {
                break;
            }
        }
        report.rounds = report.rounds.max(rounds);
        let after: f64 = f.errors(alone).iter().sum();
        for &var in &vars {
            if f.shift_of(var) != 0 {
                match var {
                    Var::Update(_) => report.updates[1] += 1,
                    Var::Change(JUMP, _) => report.jump_edges[1] += 1,
                    Var::Change(..) => report.control_changes[1] += 1,
                }
            }
        }
        report.runs += 1;
        report.updates_counted += counted;
        report.error_before += before;
        report.error_after += after;
        f.write(alone, run, frames);
    }
    Ok(report)
}

#[cfg(test)]
mod unit {
    use super::*;

    fn control(initial: f32, changes: &[(usize, &[f32])], separation: usize) -> Control {
        Control {
            initial,
            changes: changes
                .iter()
                .map(|&(start, values)| Change {
                    start,
                    len: values.len(),
                    shift: 0,
                    values: values.to_vec(),
                    gap: 0,
                    effect: None,
                })
                .collect(),
            range: 4,
            separation,
            keep_apart: if separation == 0 { JUMP_APART } else { 0 },
            counted: separation == 1,
        }
    }

    #[test]
    fn a_moved_change_keeps_its_values_and_the_held_ones() {
        let mut c = control(0.0, &[(5, &[0.2, 0.4]), (10, &[1.0])], 1);
        let row = |c: &Control| (0..14).map(|k| c.value(k)).collect::<Vec<_>>();
        assert_eq!(
            row(&c),
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0
            ]
        );
        c.changes[0].shift = -2;
        c.changes[1].shift = 2;
        assert_eq!(
            row(&c),
            [
                0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0
            ]
        );
    }

    #[test]
    fn changes_never_cross_or_merge() {
        let c = control(0.0, &[(5, &[0.2, 0.4]), (9, &[1.0])], 1);
        assert!(c.allows(0, 6, 20)); // ends on 7, one held row before 9
        assert!(!c.allows(0, 7, 20)); // would end on 8, next to 9: one ramp
        assert!(!c.allows(1, 7, 20));
        assert!(!c.allows(0, 0, 20)); // the run's first row holds the value before
        assert!(!c.allows(1, 20, 20));
        let jumps = control(0.0, &[(5, &[1.0]), (7, &[0.0])], 0);
        assert!(!jumps.allows(0, 6, 20)); // a 2-tick hold never shorter than replicar's
        assert!(jumps.allows(0, 3, 20));
        let jumps = control(0.0, &[(5, &[1.0]), (12, &[0.0]), (20, &[1.0])], 0);
        assert!(jumps.allows(1, 16, 30)); // 4 ticks from the next press
        assert!(!jumps.allows(1, 17, 30));
        assert!(!jumps.allows(0, 9, 30)); // a hold of 7 kept to 4 or more
        assert!(jumps.allows(0, 8, 30));
        let tap = control(0.0, &[(5, &[1.0]), (6, &[0.0])], 0);
        assert!(tap.allows(0, 3, 20) && !tap.allows(0, 6, 20)); // a one-tick tap stays one
    }
}
