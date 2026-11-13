//! Demonstrations for behaviour cloning: from a recorded game (RocketSim's RLPR format,
//! which human replays reach through a converter), for every car at every decision step, what the bot
//! would have seen and which of its actions the player's controls amount to.
//!
//! A decision at tick `t` sees the state at `t`; the action it chooses runs from tick `t + action_delay`
//! for `tick_skip` ticks, as in training. A record holds the controls that produced it, so that window's
//! controls are those of the records `t + action_delay + 1 ..= t + action_delay + tick_skip`. A recording
//! made from replicar (`replicar_rows`) holds in each row the controls applied next, so its window is read
//! one record earlier. Windows crossing a gap in the recording are left out.
//!
//! The controls are read as RocketSim plays them. Each control turns by the axis RocketSim uses in the
//! state it acts on, steer on the ground and yaw in the air (replicar writes yaw 0 on the ground and fits
//! it in the air; every table action has steer equal to yaw), counted once. The window's controls are
//! summarised (the axes averaged; jump on when pressed anew in the window or held for at least half of it;
//! boost and handbrake on when held for at least half) and every allowed action gets a share
//! `exp(-d / softness)` of the label, `d` being its weighted squared distance to the summary: an even mix
//! of two actions is split between them. A jump press or its absence outweighs every axis, and in the air
//! throttle and handbrake count for nothing: the air actions tie throttle to boost and carry handbrake only
//! on directional jumps (for wavedashes), so with every control counting the same, holding throttle in the
//! air without boost would be half a boost and air-rolling with the powerslide button half a dodge.
//!
//! A dodge goes the way RocketSim takes from the tick jump is pressed in the air, rounded to the table's
//! eight; the table's jump actions are measured against that, without throttle or handbrake (the
//! directional jumps' handbrake would tie a straight dodge with the plain jump). The previous action in
//! the observation is the label's first choice (a tie to the lower action). A recording does not hold the
//! boost pads: they are rebuilt from the cars' boost (a rise near a free pad took it; a kickoff frees them
//! all), and a car stays supersonic as RocketSim keeps it.
//!
//! Kickoffs only (`kickoff_only`): only the decisions in a kickoff window are kept, from the first tick of
//! each kickoff (`replay::is_kickoff`) to a set time after the ball's first touch after it (the ball
//! leaving its rest at the centre, or a car's touch flag), or to 10 s after the kickoff when nothing
//! touches it by then. Every decision is still read, so a kept one is the same sample bit for bit as
//! without the window (its previous action and the rebuilt pads included).

use std::f32::consts::FRAC_PI_4;

use rocketsim::consts::{TICK_TIME, boost_pads, car::boost::SPAWN_AMOUNT, car::supersonic};
use rocketsim::{CarBodyConfig, GameMode, Vec3A};

use crate::actions::{ActionTable, MaskRule, NUM_ACTIONS};
use crate::obs::ObsKind;
use crate::replay::{ball_at_kickoff, is_kickoff, teams_of};
use crate::rlpr::{CarRecord, Recording, Tick};
use crate::snapshot::{CarSnapshot, Controls, GameSnapshot, NUM_BOOST_PADS, PadSnapshot, Team};

#[derive(Clone, Copy, Debug)]
pub struct DemoRules {
    pub obs: ObsKind,
    /// Players a side the observation has places for (0 = the recording's own).
    pub obs_team_size: usize,
    pub tick_skip: u32,
    pub action_delay: u32,
    pub masks: MaskRule,
    /// How soft the labels are: 0.1 gives an even mix of two actions half to each and a whole unit of
    /// difference on one axis about 1/22,000 of the share.
    pub softness: f32,
    /// The recording's rows hold the controls applied next (made from replicar), not the controls that
    /// produced them.
    pub replicar_rows: bool,
    /// Only the decisions in kickoff windows, each to this many ticks after the ball's first touch;
    /// 0 = every decision.
    pub kickoff_only: u32,
}

/// The longest kickoff window after the first touch (`kickoff_only_seconds`), in seconds.
pub const KICKOFF_ONLY_MAX_SECONDS: f64 = 30.0;
/// A kickoff that nothing touches ends its window this many ticks (10 s) after its last kickoff tick.
const KICKOFF_UNTOUCHED_TICKS: u32 = 1200;

/// How much a jump press (or its absence) counts against one unit of an axis in a label.
const JUMP_WEIGHT: f32 = 10.0;

/// How much each control (in `Controls::to_floats` order) counts in a label's distance: a jump press
/// above all; the turn once, in yaw (`one_turn` has put the axis RocketSim uses in both); in the air no
/// throttle and no handbrake. Pitch and roll still count on the ground: the jump actions carry them into
/// the air.
pub fn label_weights(on_ground: bool) -> [f32; 8] {
    let ground = if on_ground { 1.0 } else { 0.0 };
    [ground, 0.0, 1.0, 1.0, 1.0, JUMP_WEIGHT, 1.0, ground]
}

/// `controls` with steer and yaw both the axis RocketSim turns by in that state: steer on the
/// ground, yaw in the air.
fn one_turn(mut controls: Controls, on_ground: bool) -> Controls {
    if on_ground {
        controls.yaw = controls.steer;
    } else {
        controls.steer = controls.yaw;
    }
    controls
}

/// The copy's own settings, which the pool's settings do not take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyKeys {
    /// `replicar_rows`: true or false (absent: false).
    pub replicar_rows: bool,
    /// `kickoff_only_seconds`: 0 to 30 (absent: 0, every decision), in ticks.
    pub kickoff_only: u32,
}

/// The core settings without the copy's own keys, and those keys.
pub fn split_copy_keys(settings_json: &str) -> Result<(String, CopyKeys), String> {
    let mut value: serde_json::Value = serde_json::from_str(settings_json)
        .map_err(|err| format!("invalid pool settings: {err}"))?;
    let mut take = |key: &str| value.as_object_mut().and_then(|map| map.remove(key));
    let replicar_rows = match take("replicar_rows") {
        None => false,
        Some(serde_json::Value::Bool(rows)) => rows,
        Some(other) => {
            return Err(format!("replicar_rows must be true or false, not {other}"));
        }
    };
    let kickoff_only = match take("kickoff_only_seconds") {
        None => 0,
        Some(serde_json::Value::Number(number))
            if number
                .as_f64()
                .is_some_and(|s| (0.0..=KICKOFF_ONLY_MAX_SECONDS).contains(&s)) =>
        {
            let seconds = number.as_f64().unwrap_or(0.0);
            // At least a tick when on at all.
            ((seconds / f64::from(TICK_TIME)).round() as u32).max(u32::from(seconds > 0.0))
        }
        Some(other) => {
            return Err(format!(
                "kickoff_only_seconds must be a number from 0 to {KICKOFF_ONLY_MAX_SECONDS}, not {other}"
            ));
        }
    };
    let keys = CopyKeys {
        replicar_rows,
        kickoff_only,
    };
    Ok((value.to_string(), keys))
}

/// Every sample of a recording, flat: `obs[i * obs_size..]`, `mask[i * NUM_ACTIONS..]`,
/// `label[i * NUM_ACTIONS..]` (shares summing to 1), and per sample the car and its team.
#[derive(Debug, Default)]
pub struct Demos {
    pub obs_size: usize,
    pub obs: Vec<f32>,
    pub mask: Vec<u8>,
    pub label: Vec<f32>,
    pub car: Vec<u8>,
    pub team: Vec<u8>,
    /// Decision steps left out because their window crossed a gap or a change in the cars.
    pub skipped: usize,
    /// Samples (car decisions) left out because they lie outside every kickoff window.
    pub outside_window: usize,
}

impl Demos {
    pub fn len(&self) -> usize {
        self.car.len()
    }

    pub fn is_empty(&self) -> bool {
        self.car.is_empty()
    }

    /// The samples as the large replay mode holds them, what `replays.compact` made with NumPy:
    /// observations in half precision (IEEE bits), the mask's bits packed (lowest action first), and each
    /// label's `LABEL_ENTRIES` largest shares (largest first, a tie to the lower action) summing to 1, in
    /// half precision.
    pub fn compact(&self) -> Compact {
        let count = self.len();
        let mut out = Compact {
            obs: self.obs.iter().map(|&v| half_bits(f64::from(v))).collect(),
            mask: vec![0; count * MASK_BYTES],
            label_index: Vec::with_capacity(count * LABEL_ENTRIES),
            label_share: Vec::with_capacity(count * LABEL_ENTRIES),
        };
        for i in 0..count {
            for (a, &allowed) in self.mask[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]
                .iter()
                .enumerate()
            {
                if allowed != 0 {
                    out.mask[i * MASK_BYTES + a / 8] |= 1 << (a % 8);
                }
            }
            let label = &self.label[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
            // Kept sorted, largest first; a later action enters only above an equal share.
            let mut top: Vec<usize> = Vec::with_capacity(LABEL_ENTRIES + 1);
            for (a, &share) in label.iter().enumerate() {
                if top.len() == LABEL_ENTRIES && share <= label[top[LABEL_ENTRIES - 1]] {
                    continue;
                }
                let at = top
                    .iter()
                    .position(|&b| label[b] < share)
                    .unwrap_or(top.len());
                top.insert(at, a);
                top.truncate(LABEL_ENTRIES);
            }
            let s: Vec<f64> = top.iter().map(|&a| f64::from(label[a])).collect();
            // In NumPy's order (its pairwise sum): eight running sums, then a tree of them.
            let mut r = [0.0f64; 8];
            for (k, &v) in s.iter().enumerate() {
                if k < 8 { r[k] = v } else { r[k % 8] += v }
            }
            let total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            let total = total.max(1e-12);
            out.label_index.extend(top.iter().map(|&a| a as u8));
            out.label_share
                .extend(s.iter().map(|&v| half_bits(v / total)));
        }
        out
    }
}

/// A label's largest shares kept in the compact form: on replicar's real 3v3 replays they hold 0.999999 of
/// a label on average and at least 0.92 (8 held at worst 0.61). A multiple of 8 (the sum's order).
pub const LABEL_ENTRIES: usize = 16;
const _: () = assert!(LABEL_ENTRIES.is_multiple_of(8));
/// Bytes of one packed mask.
pub const MASK_BYTES: usize = NUM_ACTIONS.div_ceil(8);

/// Samples in the compact form, flat per sample like `Demos`.
#[derive(Debug, Default)]
pub struct Compact {
    pub obs: Vec<u16>,
    pub mask: Vec<u8>,
    pub label_index: Vec<u8>,
    pub label_share: Vec<u16>,
}

/// `value` as IEEE half-precision bits, rounded to the nearest (a tie to the even one), as NumPy's
/// `astype(float16)` rounds both f32 and f64; beyond the range is infinity, a NaN stays a NaN.
pub fn half_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    if exponent == 0x7ff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exponent - 1023;
    if e >= 16 {
        return sign | 0x7c00;
    }
    if e < -25 {
        return sign;
    }
    // In units of the half's last place: 2^(e - 10) when normal, 2^-24 below 2^-14.
    let significand = fraction | (1 << 52);
    let shift = if e >= -14 { 42 } else { (28 - e) as u32 };
    let mut units = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if rest > halfway || (rest == halfway && units & 1 == 1) {
        units += 1;
    }
    // A carry out of the fraction lands in the exponent (up to infinity), as it should.
    let magnitude = if e >= -14 {
        (((e + 14) as u64) << 10) + units
    } else {
        units
    };
    sign | magnitude as u16
}

/// The frames each kickoff window spans: from a kickoff's first tick to `after` ticks past the ball's
/// first touch after it, or to `KICKOFF_UNTOUCHED_TICKS` past its last kickoff tick when nothing touches it
/// by then.
fn kickoff_windows(ticks: &[&Tick], after: u32) -> Vec<(u32, u32)> {
    let touched = |tick: &&&Tick| {
        !ball_at_kickoff(&tick.ball) || tick.cars.iter().any(|car| car.touching_ball)
    };
    let mut windows = Vec::new();
    let mut i = 0;
    while i < ticks.len() {
        if !is_kickoff(ticks[i]) {
            i += 1;
            continue;
        }
        let start = ticks[i].frame;
        while i + 1 < ticks.len() && is_kickoff(ticks[i + 1]) {
            i += 1;
        }
        let limit = ticks[i].frame.saturating_add(KICKOFF_UNTOUCHED_TICKS);
        let touch = ticks[i + 1..]
            .iter()
            .take_while(|tick| tick.frame <= limit)
            .find(touched);
        windows.push((
            start,
            touch.map_or(limit, |tick| tick.frame.saturating_add(after)),
        ));
        i += 1;
    }
    windows
}

/// The demonstrations in `recording` under `rules`.
pub fn demonstrations(recording: &Recording, rules: &DemoRules) -> Result<Demos, String> {
    let teams = teams_of(recording)?;
    let table = ActionTable::new();
    let per_side = |team: Team| teams.iter().filter(|&&t| t == team).count();
    let (blue, orange) = (per_side(Team::Blue), per_side(Team::Orange));
    let team_size = blue.max(orange);
    if rules.obs_team_size > 0 && team_size > rules.obs_team_size {
        return Err(format!(
            "the recording has {team_size} players a side; the observation has room for {}",
            rules.obs_team_size
        ));
    }
    if rules.obs_team_size == 0 && blue != orange && orange != 0 {
        return Err("uneven teams need a padded observation (env.obs_team_size)".to_owned());
    }
    let obs_size = rules
        .obs
        .layout(team_size, orange > 0, rules.obs_team_size)
        .size;
    let mut demos = Demos {
        obs_size,
        ..Demos::default()
    };
    let (skip, delay) = (rules.tick_skip as usize, rules.action_delay as usize);
    // Some recordings hold a frame twice in a row (RocketSim's `nexto_1v1`: almost every frame); the later
    // copy stands for the frame.
    let mut ticks: Vec<&Tick> = Vec::with_capacity(recording.ticks.len());
    for tick in &recording.ticks {
        match ticks.last_mut() {
            Some(last) if last.frame == tick.frame => *last = tick,
            _ => ticks.push(tick),
        }
    }
    // The record whose controls run first in a decision's window, from the decision's: the one after the
    // delay holds the controls that produced it, the one before (replicar's) those applied next.
    let first = delay + usize::from(!rules.replicar_rows);
    let mut rebuilt = Rebuilt::new();
    let windows = if rules.kickoff_only > 0 {
        kickoff_windows(&ticks, rules.kickoff_only)
    } else {
        Vec::new()
    };
    let mut previous: Vec<Controls> = vec![Controls::default(); teams.len()];
    let mut snap = GameSnapshot::default();
    let mut t = 0;
    while t + first + skip <= ticks.len() {
        // Every record read, from the decision's (or the one before, for jump held before) to the window's
        // last, consecutive and of every car; none before the first.
        let whole = (t + first).checked_sub(1).is_some_and(|held| {
            let window = &ticks[held.min(t)..t + first + skip];
            window
                .windows(2)
                .all(|pair| pair[1].frame == pair[0].frame + 1)
                && window.iter().all(|tick| tick.cars.len() == teams.len())
        });
        if !whole {
            demos.skipped += 1;
            previous.fill(Controls::default());
            t += skip;
            continue;
        }
        rebuilt.advance(&ticks, t);
        fill_snapshot(&mut snap, ticks[t], &teams, &rebuilt);
        // Outside every kickoff window the labels are still made, for the next decision's previous action.
        let frame = ticks[t].frame;
        let kept = rules.kickoff_only == 0
            || windows
                .iter()
                .any(|&(start, end)| start <= frame && frame <= end);
        let mut obs = vec![0.0; obs_size];
        let mut mask = [0u8; NUM_ACTIONS];
        for (car, &team) in teams.iter().enumerate() {
            if kept {
                rules
                    .obs
                    .build(&snap, car, previous[car], rules.obs_team_size, &mut obs);
            }
            table.write_mask(rules.masks, &snap.cars[car], &mut mask);
            // In either convention `applied[i]` acts on the state of record `t + delay + i`.
            let acts_on = |i: usize| ticks[t + delay + i].cars[car].on_ground;
            let applied: Vec<Controls> = ticks[t + first..t + first + skip]
                .iter()
                .enumerate()
                .map(|(i, tick)| one_turn(tick.cars[car].controls, acts_on(i)))
                .collect();
            let held_before = ticks[t + first - 1].cars[car].controls.jump;
            // A jump pressed anew on a tick the car is in the air is a dodge (or a double jump).
            let dodge = first_press(&applied, held_before)
                .filter(|&i| !acts_on(i))
                .map(|i| dodge_direction(&applied[i]));
            let label = soft_label(
                &table,
                &mask,
                &applied,
                held_before,
                rules.softness,
                &label_weights(snap.cars[car].on_ground),
                dodge,
            );
            previous[car] = table.controls(first_choice(&label));
            if !kept {
                demos.outside_window += 1;
                continue;
            }
            demos.obs.extend_from_slice(&obs);
            demos.mask.extend_from_slice(&mask);
            demos.label.extend_from_slice(&label);
            demos.car.push(car as u8);
            demos.team.push(team.index() as u8);
        }
        t += skip;
    }
    Ok(demos)
}

/// The label's first choice: its largest share, a tie to the lower action (as `Demos::compact` and
/// `argmax` take it).
pub fn first_choice(label: &[f32]) -> usize {
    let mut best = 0;
    for (action, &share) in label.iter().enumerate() {
        if share > label[best] {
            best = action;
        }
    }
    best
}

/// The first of `applied` that presses jump anew (not held on from the one before).
fn first_press(applied: &[Controls], held_before: bool) -> Option<usize> {
    let mut held = held_before;
    applied.iter().position(|c| {
        let edge = c.jump && !held;
        held = c.jump;
        edge
    })
}

/// What RocketSim makes of a jump pressed in the air with `controls` (`update_double_jump_or_flip`), as the
/// table's jump actions have it: (pitch, roll) of the nearest of their eight dodges, or (0, 0) for a double
/// jump (the sticks under the dodge deadzone) or a stall (no direction left).
pub fn dodge_direction(controls: &Controls) -> (f32, f32) {
    let [pitch, yaw, roll] =
        [controls.pitch, controls.yaw, controls.roll].map(|v| v.clamp(-1.0, 1.0));
    if pitch.abs() + yaw.abs() + roll.abs() < CarBodyConfig::OCTANE.dodge_deadzone {
        return (0.0, 0.0);
    }
    // Forward and sideways, as RocketSim's dodge direction.
    let (forward, side) = (-pitch, yaw + roll);
    if forward.abs() < 0.1 && side.abs() < 0.1 {
        return (0.0, 0.0);
    }
    // From a back flip (-180°) round to a back flip (180°) in eighths of a turn.
    const EIGHTHS: [(f32, f32); 9] = [
        (1.0, 0.0),
        (1.0, -1.0),
        (0.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
    ];
    let eighth = (side.atan2(forward) / FRAC_PI_4).round() as i32;
    EIGHTHS[(eighth.clamp(-4, 4) + 4) as usize]
}

/// The boost pads in the simulator's order (`Sim::read`): RocketSim's soccar pads, small then big, sorted
/// by y then x as its arena sorts them, each with whether it is big.
pub fn pad_layout() -> Vec<(Vec3A, bool)> {
    let mut pads: Vec<(Vec3A, bool)> = [false, true]
        .into_iter()
        .flat_map(|big| {
            boost_pads::get_locations(GameMode::Soccar, big)
                .iter()
                .map(move |&pos| (pos, big))
        })
        .collect();
    pads.sort_by(|a, b| a.0.y.total_cmp(&b.0.y).then(a.0.x.total_cmp(&b.0.x)));
    assert_eq!(pads.len(), NUM_BOOST_PADS);
    pads
}

/// How far (uu) beyond its trigger radius a car may be from a pad when its boost rises and still have taken
/// it (a replay's positions are the game's, not RocketSim's).
const PAD_REACH: f32 = 150.0;
/// A rise in boost (of 100) above which a big pad may have given it: a small one gives 12. A big pad may
/// also give less, when it fills the tank.
const BIG_PAD_RISE: f32 = 13.0;
/// A tank at least this full (of 100) is full.
const FULL_TANK: f32 = 99.5;

/// What a recording lacks, rebuilt from its ticks: the boost pads, and whether each car is supersonic.
struct Rebuilt {
    pads: Vec<(Vec3A, bool)>,
    /// Per pad, RocketSim's tick count when it gave boost (the frame before the rise shows); none while
    /// the pad is active.
    taken: [Option<i64>; NUM_BOOST_PADS],
    supersonic: Vec<bool>,
    /// Per car, RocketSim's time below the supersonic speed while still supersonic.
    grace: Vec<f32>,
    /// The next tick to read.
    next: usize,
}

impl Rebuilt {
    fn new() -> Self {
        Self {
            pads: pad_layout(),
            taken: [None; NUM_BOOST_PADS],
            supersonic: Vec::new(),
            grace: Vec::new(),
            next: 0,
        }
    }

    /// Reads the ticks up to `ticks[last]`.
    fn advance(&mut self, ticks: &[&Tick], last: usize) {
        while self.next <= last {
            let i = self.next;
            self.read(i.checked_sub(1).map(|before| ticks[before]), ticks[i]);
            self.next += 1;
        }
    }

    /// Pad `pad`'s cooldown at `frame`, as RocketSim reckons it.
    fn cooldown(&self, pad: usize, frame: i64) -> f32 {
        self.taken[pad].map_or(0.0, |gave| {
            let most = if self.pads[pad].1 {
                boost_pads::COOLDOWN_BIG
            } else {
                boost_pads::COOLDOWN_SMALL
            };
            (most - ((frame - gave) as f32) * TICK_TIME).max(0.0)
        })
    }

    fn read(&mut self, before: Option<&Tick>, tick: &Tick) {
        let follows =
            before.filter(|b| tick.frame == b.frame + 1 && b.cars.len() == tick.cars.len());
        if before.is_none() || is_kickoff(tick) {
            self.taken = [None; NUM_BOOST_PADS];
        } else if let Some(before) = follows {
            for (was, car) in before.cars.iter().zip(&tick.cars) {
                self.take_pad(was, car, i64::from(tick.frame));
            }
        }
        // RocketSim's rule (`post_tick_update`), on the squared speeds; anew after a gap.
        const START: f32 = supersonic::START_SPEED * supersonic::START_SPEED;
        const MAINTAIN: f32 = supersonic::MAINTAIN_MIN_SPEED * supersonic::MAINTAIN_MIN_SPEED;
        if follows.is_none() {
            self.supersonic.clear();
            self.supersonic.extend(
                tick.cars
                    .iter()
                    .map(|c| c.body.vel.length_squared() >= START),
            );
            self.grace.clear();
            self.grace.resize(tick.cars.len(), 0.0);
            return;
        }
        for ((on, grace), car) in self
            .supersonic
            .iter_mut()
            .zip(&mut self.grace)
            .zip(&tick.cars)
        {
            let speed = car.body.vel.length_squared();
            if !*on {
                *on = speed >= START;
                *grace = 0.0;
            } else if speed >= START {
                *grace = 0.0;
            } else if speed < MAINTAIN {
                (*on, *grace) = (false, 0.0);
            } else {
                *grace += TICK_TIME;
                if *grace >= supersonic::MAINTAIN_MAX_TIME {
                    (*on, *grace) = (false, 0.0);
                }
            }
        }
    }

    /// A rise in a car's boost (more than 1 of 100) took the nearest pad within reach that was free, a big
    /// one only for more than a small one gives or a full tank; a respawn's third of a tank took none.
    fn take_pad(&mut self, was: &CarRecord, car: &CarRecord, frame: i64) {
        let (before, now) = (was.boost * 100.0, car.boost * 100.0);
        let rise = now - before;
        if rise <= 1.0 || (before < 1.0 && (now - SPAWN_AMOUNT).abs() < 0.5) {
            return;
        }
        let big_too = rise > BIG_PAD_RISE || now >= FULL_TANK;
        let gave = frame - 1;
        let at = car.body.pos.truncate();
        let mut nearest: Option<(usize, f32)> = None;
        for (pad, &(pos, big)) in self.pads.iter().enumerate() {
            if (big && !big_too) || self.cooldown(pad, gave) > 0.0 {
                continue;
            }
            let reach = PAD_REACH
                + if big {
                    boost_pads::CYL_RAD_BIG
                } else {
                    boost_pads::CYL_RAD_SMALL
                };
            let distance = pos.truncate().distance_squared(at);
            if distance < reach * reach && nearest.is_none_or(|(_, d)| distance < d) {
                nearest = Some((pad, distance));
            }
        }
        if let Some((pad, _)) = nearest {
            self.taken[pad] = Some(gave);
        }
    }
}

/// The state of every car and the ball at `tick`, in the recording's car order, with `rebuilt`'s pads and
/// supersonic flags.
fn fill_snapshot(snap: &mut GameSnapshot, tick: &Tick, teams: &[Team], rebuilt: &Rebuilt) {
    snap.ball = tick.ball;
    snap.cars.clear();
    for (i, (record, &team)) in tick.cars.iter().zip(teams).enumerate() {
        snap.cars
            .push(car_snapshot(record, team, rebuilt.supersonic[i]));
    }
    for (pad, out) in snap.pads.iter_mut().enumerate() {
        let cooldown = rebuilt.cooldown(pad, i64::from(tick.frame));
        *out = PadSnapshot {
            active: cooldown <= 0.0,
            cooldown,
        };
    }
    snap.goal_scored = None;
}

fn car_snapshot(record: &CarRecord, team: Team, is_supersonic: bool) -> CarSnapshot {
    let flipped = record.double_jumped_or_flipped && (record.is_flipping || record.flip_time > 0.0);
    CarSnapshot {
        team,
        body: record.body,
        boost: record.boost * 100.0,
        on_ground: record.on_ground,
        has_jumped: record.has_jumped,
        has_double_jumped: record.double_jumped_or_flipped && !flipped,
        has_flipped: flipped,
        can_flip: record.has_flip,
        is_demoed: false,
        is_supersonic,
        demo_timer: 0.0,
        // RocketSim's flip window starts when the jump's force stops: its clock stands at 0 while the jump
        // is held, and from then on equals the jump count, which restarts at 1 on that tick.
        flip_time_left: CarSnapshot::flip_time_left_from(
            record.on_ground,
            record.has_jumped,
            record.has_flip,
            if record.is_jumping {
                0.0
            } else {
                record.jump_time
            },
        ),
    }
}

/// The label of one decision: shares of the allowed actions by their weighted distance to what was
/// applied. With `dodge`, the table's jump actions are measured against its (pitch, roll) and yaw 0 instead
/// of the window's means, and without throttle or handbrake.
pub fn soft_label(
    table: &ActionTable,
    mask: &[u8],
    applied: &[Controls],
    jump_held_before: bool,
    softness: f32,
    weights: &[f32; 8],
    dodge: Option<(f32, f32)>,
) -> [f32; NUM_ACTIONS] {
    let n = applied.len().max(1) as f32;
    let mean = |pick: fn(&Controls) -> f32| applied.iter().map(pick).sum::<f32>() / n;
    let share = |pick: fn(&Controls) -> bool| applied.iter().filter(|c| pick(c)).count() as f32 / n;
    let pressed_anew = first_press(applied, jump_held_before).is_some();
    let target = [
        mean(|c| c.throttle),
        mean(|c| c.steer),
        mean(|c| c.pitch),
        mean(|c| c.yaw),
        mean(|c| c.roll),
        f32::from(u8::from(pressed_anew || share(|c| c.jump) >= 0.5)),
        f32::from(u8::from(share(|c| c.boost) >= 0.5)),
        f32::from(u8::from(share(|c| c.handbrake) >= 0.5)),
    ];
    let jump_rule = dodge.map(|(pitch, roll)| {
        let (mut jump, mut jump_weights) = (target, *weights);
        (jump[2], jump[3], jump[4]) = (pitch, 0.0, roll);
        (jump_weights[0], jump_weights[7]) = (0.0, 0.0);
        (jump, jump_weights)
    });
    let mut label = [0.0f32; NUM_ACTIONS];
    let mut distances = [f32::INFINITY; NUM_ACTIONS];
    for (action, distance) in distances.iter_mut().enumerate() {
        if mask[action] != 0 {
            let c = table.controls(action);
            let (target, weights) = match &jump_rule {
                Some((jump, jump_weights)) if c.jump => (jump, jump_weights),
                _ => (&target, weights),
            };
            *distance = c
                .to_floats()
                .iter()
                .zip(target)
                .zip(weights)
                .map(|((a, b), w)| w * (a - b) * (a - b))
                .sum();
        }
    }
    let nearest = distances.iter().copied().fold(f32::INFINITY, f32::min);
    if !nearest.is_finite() {
        return label;
    }
    let mut total = 0.0;
    for (slot, &distance) in label.iter_mut().zip(&distances) {
        if distance.is_finite() {
            *slot = (-(distance - nearest) / softness.max(1e-6)).exp();
            total += *slot;
        }
    }
    for slot in &mut label {
        *slot /= total;
    }
    label
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{Body, FLIP_WINDOW_SECONDS};
    use rocketsim::{Mat3A, Vec3A};

    fn table_index(table: &ActionTable, want: Controls) -> usize {
        (0..NUM_ACTIONS)
            .find(|&a| table.controls(a) == want)
            .unwrap()
    }

    fn half_value(bits: u16) -> f64 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let (e, m) = (i32::from((bits >> 10) & 0x1f), f64::from(bits & 0x3ff));
        sign * if e == 0 {
            m * 2f64.powi(-24)
        } else {
            (1.0 + m / 1024.0) * 2f64.powi(e - 15)
        }
    }

    #[test]
    fn half_precision_is_every_half_exactly_and_rounds_halfway_to_the_even_one() {
        for sign in [0u16, 0x8000] {
            for h in 0..0x7c00u16 {
                let (value, bits) = (half_value(h | sign), h | sign);
                assert_eq!(half_bits(value), bits, "{value}");
                let next = half_value((h + 1) | sign); // 0x7c00 is infinity: the midpoint is 65520
                let next = if h == 0x7bff {
                    65536.0 * value.signum()
                } else {
                    next
                };
                let middle = (value + next) / 2.0;
                let step = (next - value) * 1e-3;
                let even = if h % 2 == 0 { h } else { h + 1 };
                assert_eq!(half_bits(middle), even | sign, "halfway above {value}");
                assert_eq!(half_bits(middle + step), (h + 1) | sign);
                assert_eq!(half_bits(middle - step), bits);
            }
        }
        assert_eq!(half_bits(f64::INFINITY), 0x7c00);
        assert_eq!(half_bits(-1e10), 0xfc00);
        assert_eq!(half_bits(f64::NAN) & 0x7e00, 0x7e00);
        assert_eq!(half_bits(1e-300), 0);
        assert_eq!(half_bits(-0.0), 0x8000);
        assert_eq!(half_bits(f64::from(0.1f32)), 0x2e66);
    }

    #[test]
    fn the_compact_form_keeps_the_mask_bits_and_the_largest_shares_a_tie_to_the_lower_action() {
        let mut demos = Demos {
            obs_size: 2,
            obs: vec![1.0, -2.5],
            mask: vec![0; NUM_ACTIONS],
            label: vec![0.0; NUM_ACTIONS],
            car: vec![0],
            team: vec![0],
            skipped: 0,
            outside_window: 0,
        };
        for a in [0, 9, 89] {
            demos.mask[a] = 1;
        }
        for (a, share) in [
            (80, 0.4),
            (5, 0.2),
            (3, 0.2),
            (40, 0.1),
            (7, 0.05),
            (2, 0.05),
        ] {
            demos.label[a] = share;
        }
        let compact = demos.compact();
        assert_eq!(compact.obs, [0x3c00, 0xc100]);
        let mut mask = vec![0u8; MASK_BYTES];
        (mask[0], mask[1], mask[11]) = (1, 2, 2);
        assert_eq!(compact.mask, mask);
        assert_eq!(
            compact.label_index,
            [80, 3, 5, 40, 2, 7, 0, 1, 4, 6, 8, 9, 10, 11, 12, 13]
        );
        let shares: Vec<f64> = compact.label_share.iter().map(|&b| half_value(b)).collect();
        assert!((shares[0] - 0.4).abs() < 1e-3 && (shares[1] - 0.2).abs() < 1e-3);
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 2e-3 && shares[6] == 0.0);
    }

    /// The label's share on actions with `pick`.
    fn share_of(
        table: &ActionTable,
        label: &[f32; NUM_ACTIONS],
        pick: impl Fn(Controls) -> bool,
    ) -> f32 {
        (0..NUM_ACTIONS)
            .filter(|&a| pick(table.controls(a)))
            .map(|a| label[a])
            .sum()
    }

    #[test]
    fn labels_read_throttle_and_air_roll_in_the_air_as_neither_boost_nor_dodge() {
        let table = ActionTable::new();
        let mut air = [0u8; NUM_ACTIONS];
        let mut car = CarSnapshot {
            on_ground: false,
            can_flip: true,
            has_jumped: true,
            boost: 50.0,
            ..CarSnapshot::default()
        };
        table.write_mask(MaskRule::Standard, &car, &mut air);
        let label =
            |mask: &[u8; NUM_ACTIONS], applied: &[Controls], held: bool, on_ground: bool| {
                soft_label(
                    &table,
                    mask,
                    applied,
                    held,
                    0.1,
                    &label_weights(on_ground),
                    None,
                )
            };
        // Throttle held in the air, no boost: no boost.
        let throttle = Controls {
            throttle: 1.0,
            pitch: 1.0,
            ..Controls::default()
        };
        let shares = label(&air, &[throttle; 8], true, false);
        assert!(share_of(&table, &shares, |c| c.boost) < 1e-3);
        // Air roll with the powerslide button (handbrake) and a flip available: no dodge.
        let roll = Controls {
            roll: -1.0,
            handbrake: true,
            ..Controls::default()
        };
        let shares = label(&air, &[roll; 8], true, false);
        assert!(share_of(&table, &shares, |c| c.jump) < 1e-3);
        // A dodge pressed in the air is still a dodge, the same way.
        let dodge = Controls {
            pitch: -1.0,
            jump: true,
            ..Controls::default()
        };
        let shares = label(&air, &[dodge; 8], false, false);
        assert!(share_of(&table, &shares, |c| c.jump && c.pitch < 0.0) > 0.99);
        // On the ground a jump pressed while driving is a jump.
        car.on_ground = true;
        let mut ground = [0u8; NUM_ACTIONS];
        table.write_mask(MaskRule::Standard, &car, &mut ground);
        let hop: Vec<Controls> = (0..8)
            .map(|i| Controls {
                throttle: 1.0,
                jump: i < 3,
                ..Controls::default()
            })
            .collect();
        let shares = label(&ground, &hop, false, true);
        assert!(share_of(&table, &shares, |c| c.jump) > 0.99);
        // Driving on the ground keeps its throttle and handbrake.
        let slide = Controls {
            throttle: 1.0,
            steer: 1.0,
            yaw: 1.0,
            handbrake: true,
            ..Controls::default()
        };
        let shares = label(&ground, &[slide; 8], false, true);
        assert!(shares[table_index(&table, slide)] > 0.99);
    }

    #[test]
    fn the_copys_own_keys_are_taken_out_of_the_pool_settings() {
        let (rest, keys) = split_copy_keys(r#"{"seed": 3, "replicar_rows": true}"#).unwrap();
        assert_eq!(
            keys,
            CopyKeys {
                replicar_rows: true,
                kickoff_only: 0,
            }
        );
        assert!(!rest.contains("replicar_rows"), "{rest}");
        assert!(rest.contains("seed"));
        let absent = CopyKeys {
            replicar_rows: false,
            kickoff_only: 0,
        };
        assert_eq!(split_copy_keys(r#"{"seed": 3}"#).unwrap().1, absent);
        let (_, keys) = split_copy_keys(r#"{"replicar_rows": false}"#).unwrap();
        assert_eq!(keys, absent);
        for (bad, says) in [
            (
                r#"{"replicar_rows": 1}"#,
                "replicar_rows must be true or false, not 1",
            ),
            (r#"{"kickoff_only_seconds": -1}"#, "from 0 to 30, not -1"),
            (r#"{"kickoff_only_seconds": 30.5}"#, "not 30.5"),
            (r#"{"kickoff_only_seconds": "3"}"#, r#"not "3""#),
        ] {
            let err = split_copy_keys(bad).unwrap_err();
            assert!(err.contains(says), "{bad}: {err}");
        }
        // Kickoffs only: seconds to ticks, at least one when on; 0 is off.
        for (seconds, ticks) in [
            ("3", 360),
            ("3.0", 360),
            ("0", 0),
            ("0.001", 1),
            ("30", 3600),
        ] {
            let json = format!(r#"{{"seed": 3, "kickoff_only_seconds": {seconds}}}"#);
            let (rest, keys) = split_copy_keys(&json).unwrap();
            assert_eq!(keys.kickoff_only, ticks, "{seconds}");
            assert!(!rest.contains("kickoff"), "{rest}");
        }
    }

    #[test]
    fn controls_that_are_one_action_label_that_action() {
        let table = ActionTable::new();
        let mask = [1u8; NUM_ACTIONS];
        let drive = Controls {
            throttle: 1.0,
            steer: 1.0,
            yaw: 1.0,
            ..Controls::default()
        };
        let label = soft_label(
            &table,
            &mask,
            &[drive; 8],
            false,
            0.1,
            &label_weights(true),
            None,
        );
        let best = table_index(&table, drive);
        assert!(label[best] > 0.99, "{}", label[best]);
        assert!((label.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn an_even_mix_is_split_and_a_short_jump_press_counts() {
        let table = ActionTable::new();
        let mask = [1u8; NUM_ACTIONS];
        let straight = Controls {
            throttle: 1.0,
            ..Controls::default()
        };
        let left = Controls {
            steer: -1.0,
            yaw: -1.0,
            ..straight
        };
        let half: Vec<Controls> = (0..8)
            .map(|i| if i < 4 { straight } else { left })
            .collect();
        let label = soft_label(&table, &mask, &half, false, 0.1, &label_weights(true), None);
        let (a, b) = (table_index(&table, straight), table_index(&table, left));
        assert!(
            (label[a] - label[b]).abs() < 1e-4 && label[a] > 0.3,
            "{} {}",
            label[a],
            label[b]
        );

        // Two ticks of jump, newly pressed: a jump action.
        let tap: Vec<Controls> = (0..8)
            .map(|i| Controls {
                jump: i < 2,
                ..straight
            })
            .collect();
        let label = soft_label(&table, &mask, &tap, false, 0.1, &label_weights(true), None);
        let best = (0..NUM_ACTIONS)
            .max_by(|&x, &y| label[x].total_cmp(&label[y]))
            .unwrap();
        assert!(table.controls(best).jump);
        // The tail of a jump held from before, released early: no new jump.
        let label = soft_label(&table, &mask, &tap, true, 0.1, &label_weights(true), None);
        let best = (0..NUM_ACTIONS)
            .max_by(|&x, &y| label[x].total_cmp(&label[y]))
            .unwrap();
        assert!(!table.controls(best).jump);
    }

    #[test]
    fn only_allowed_actions_get_a_share() {
        let table = ActionTable::new();
        let mut mask = [1u8; NUM_ACTIONS];
        let boost = Controls {
            throttle: 1.0,
            boost: true,
            ..Controls::default()
        };
        let best = table_index(&table, boost);
        mask[best] = 0;
        let label = soft_label(
            &table,
            &mask,
            &[boost; 8],
            false,
            0.1,
            &label_weights(true),
            None,
        );
        assert_eq!(label[best], 0.0);
        assert!((label.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    fn body(x: f32, y: f32) -> Body {
        Body {
            pos: Vec3A::new(x, y, 17.0),
            rot: Mat3A::IDENTITY,
            vel: Vec3A::ZERO,
            ang_vel: Vec3A::ZERO,
        }
    }

    fn record(x: f32, y: f32, controls: Controls) -> CarRecord {
        CarRecord {
            body: body(x, y),
            on_ground: true,
            is_jumping: false,
            is_flipping: false,
            jump_time: 0.0,
            flip_time: 0.0,
            has_jumped: false,
            double_jumped_or_flipped: false,
            has_flip: true,
            flip_rel_torque: Vec3A::ZERO,
            boost: 0.33,
            touching_ball: false,
            controls,
            touching_car: false,
        }
    }

    /// A 1v1 that starts at a kickoff (so the teams can be told apart): Blue drives straight on,
    /// Orange turns left; a gap in the frames after tick 40.
    fn recording(ticks: usize) -> Recording {
        let straight = Controls {
            throttle: 1.0,
            ..Controls::default()
        };
        let left = Controls {
            steer: -1.0,
            yaw: -1.0,
            ..straight
        };
        Recording {
            version: 6,
            num_cars: 2,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks: (0..ticks)
                .map(|t| Tick {
                    frame: if t > 40 { t as u32 + 100 } else { t as u32 },
                    cars: vec![
                        record(-2048.0, -2560.0, straight),
                        record(2048.0, 2560.0, left),
                    ],
                    ball: Body {
                        pos: Vec3A::new(0.0, 0.0, 93.15),
                        ..body(0.0, 0.0)
                    },
                    ball_world_contact: false,
                })
                .collect(),
        }
    }

    #[test]
    fn the_flip_window_runs_from_the_end_of_the_jump_as_in_rocketsim() {
        let mut car = record(0.0, 0.0, Controls::default());
        (car.on_ground, car.has_jumped, car.is_jumping, car.jump_time) = (false, true, true, 0.1);
        let held = car_snapshot(&car, Team::Blue, false).flip_time_left;
        assert_eq!(
            held, FLIP_WINDOW_SECONDS,
            "the clock stands while the jump is held"
        );
        (car.is_jumping, car.jump_time) = (false, 0.5);
        let after = car_snapshot(&car, Team::Blue, false).flip_time_left;
        assert!((after - (FLIP_WINDOW_SECONDS - 0.5)).abs() < 1e-6);
        car.has_flip = false;
        assert_eq!(car_snapshot(&car, Team::Blue, false).flip_time_left, 0.0);
    }

    #[test]
    fn a_recording_gives_one_sample_per_car_and_decision_with_its_label() {
        let rules = DemoRules {
            obs: ObsKind::Native,
            obs_team_size: 0,
            tick_skip: 8,
            action_delay: 7,
            masks: MaskRule::Standard,
            softness: 0.1,
            replicar_rows: false,
            kickoff_only: 0,
        };
        let demos = demonstrations(&recording(81), &rules).unwrap();
        let table = ActionTable::new();
        // Decisions at 0, 8, ..., 64 need ticks up to t + 15; those whose window holds the gap after
        // tick 40 (t = 32, 40) are left out.
        assert_eq!(demos.skipped, 2);
        assert_eq!(demos.len(), 2 * 7, "nine decisions, two left out, two cars");
        assert_eq!(demos.obs_size, ObsKind::Native.layout(1, true, 0).size);
        assert_eq!(demos.obs.len(), demos.len() * demos.obs_size);
        assert_eq!(demos.team[..2], [0, 1]);
        for i in 0..demos.len() {
            let label = &demos.label[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
            let best = (0..NUM_ACTIONS)
                .max_by(|&a, &b| label[a].total_cmp(&label[b]))
                .unwrap();
            let steer = table.controls(best).steer;
            assert_eq!(steer, if demos.car[i] == 0 { 0.0 } else { -1.0 });
        }
        // The same snapshot gives the same observation the environments would build.
        let mut snap = GameSnapshot::default();
        let rec = recording(81);
        let mut rebuilt = Rebuilt::new();
        rebuilt.advance(&rec.ticks.iter().collect::<Vec<_>>(), 0);
        fill_snapshot(&mut snap, &rec.ticks[0], &teams_of(&rec).unwrap(), &rebuilt);
        let mut expected = vec![0.0; demos.obs_size];
        ObsKind::Native.build(&snap, 1, Controls::default(), 0, &mut expected);
        assert_eq!(demos.obs[demos.obs_size..2 * demos.obs_size], expected[..]);
        // A recording holding every frame twice gives the same samples.
        let mut doubled = rec.clone();
        doubled.ticks = rec
            .ticks
            .iter()
            .flat_map(|t| [t.clone(), t.clone()])
            .collect();
        let again = demonstrations(&doubled, &rules).unwrap();
        assert_eq!((again.len(), again.skipped), (demos.len(), demos.skipped));
        // A padded observation fits too, and a game bigger than its room is refused.
        let padded = demonstrations(
            &rec,
            &DemoRules {
                obs_team_size: 2,
                ..rules
            },
        )
        .unwrap();
        assert_eq!(padded.obs_size, ObsKind::Native.layout(1, true, 2).size);
        // A 2v2 recording is refused by a room of 1 a side.
        let mut two = rec.clone();
        for tick in &mut two.ticks {
            let mut cars = tick.cars.clone();
            cars[0].body.pos.y = -3840.0;
            cars[1].body.pos.y = 3840.0;
            tick.cars.extend(cars);
        }
        two.num_cars = 4;
        let refused = demonstrations(
            &two,
            &DemoRules {
                obs_team_size: 1,
                ..rules
            },
        );
        assert!(refused.unwrap_err().contains("2 players a side"));
    }

    /// A 1v1 of 2,400 ticks with two kickoffs: one at frame 0 whose ball is first touched at frame
    /// 240, one at frame 900 that nothing touches.
    fn two_kickoffs() -> Recording {
        let drive = Controls {
            throttle: 1.0,
            boost: true,
            ..Controls::default()
        };
        let ticks = (0..2400u32)
            .map(|frame| {
                let at_kickoff = frame <= 5 || (900..=905).contains(&frame);
                let mut cars = vec![
                    record(-2048.0, -2560.0, drive),
                    record(2048.0, 2560.0, Controls::default()),
                ];
                if !at_kickoff {
                    cars[0].body.vel = Vec3A::new(0.0, 1000.0, 0.0);
                    cars[1].body.vel = Vec3A::new(0.0, -1000.0, 0.0);
                }
                let mut ball = Body {
                    pos: Vec3A::new(0.0, 0.0, 93.15),
                    ..body(0.0, 0.0)
                };
                if (240..900).contains(&frame) {
                    ball.pos.x = 100.0;
                    ball.vel = Vec3A::new(500.0, 0.0, 0.0);
                }
                Tick {
                    frame,
                    cars,
                    ball,
                    ball_world_contact: false,
                }
            })
            .collect();
        Recording {
            version: 6,
            num_cars: 2,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks,
        }
    }

    #[test]
    fn kickoffs_only_keep_the_kickoffs_to_the_set_time_after_the_first_touch() {
        let rec = two_kickoffs();
        let rules = copy_rules(true);
        let all = demonstrations(&rec, &rules).unwrap();
        assert_eq!(
            (all.len(), all.skipped, all.outside_window),
            (2 * 599, 0, 0)
        );
        let only = demonstrations(
            &rec,
            &DemoRules {
                kickoff_only: 120,
                ..rules
            },
        )
        .unwrap();
        // From frame 0 to 1 s after the touch at 240, and from 900 to 10 s after the last kickoff tick.
        let kept: Vec<usize> = (0..599)
            .filter(|&k| {
                let frame = 4 * k;
                frame <= 240 + 120 || (900..=905 + 1200).contains(&frame)
            })
            .collect();
        assert_eq!(kept.len(), 91 + 302);
        assert_eq!(only.len(), 2 * kept.len());
        assert_eq!(only.outside_window + only.len(), all.len());
        assert_eq!(only.skipped, all.skipped);
        // Each kept sample is the one made without the window, bit for bit.
        let (o, b) = (all.obs_size, NUM_ACTIONS);
        for (i, &k) in kept.iter().enumerate() {
            for car in 0..2 {
                let (from, to) = (2 * k + car, 2 * i + car);
                assert_eq!(
                    only.obs[to * o..(to + 1) * o],
                    all.obs[from * o..(from + 1) * o]
                );
                assert_eq!(
                    only.mask[to * b..(to + 1) * b],
                    all.mask[from * b..(from + 1) * b]
                );
                let (got, want) = (
                    &only.label[to * b..(to + 1) * b],
                    &all.label[from * b..(from + 1) * b],
                );
                assert!(
                    got.iter()
                        .zip(want)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                assert_eq!(
                    (only.car[to], only.team[to]),
                    (all.car[from], all.team[from])
                );
            }
        }
        // A car's touch flag counts as the touch too, even with the ball still at rest.
        let mut flagged = two_kickoffs();
        for tick in &mut flagged.ticks {
            if (240..900).contains(&tick.frame) {
                tick.ball = Body {
                    pos: Vec3A::new(0.0, 0.0, 93.15),
                    ..body(0.0, 0.0)
                };
            }
            tick.cars[0].touching_ball = (100..110).contains(&tick.frame);
        }
        let windows = kickoff_windows(&flagged.ticks.iter().collect::<Vec<_>>(), 120);
        assert_eq!(windows, [(0, 220), (900, 2105)]);
    }

    /// FNV-1a over `bytes`, from `hash`.
    fn fnv(hash: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(hash, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

    /// One of three exact turns about up (no trigonometry, so the same bits on every machine).
    fn turned(turn: u32) -> Mat3A {
        match turn % 3 {
            0 => Mat3A::IDENTITY,
            1 => Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z),
            _ => Mat3A::from_cols(
                Vec3A::new(0.6, 0.8, 0.0),
                Vec3A::new(-0.8, 0.6, 0.0),
                Vec3A::Z,
            ),
        }
    }

    /// Car `c` of the pin's recording at tick `t` (kickoff spot `spot`).
    fn pin_car(t: u32, c: u32, spot: (f32, f32)) -> CarRecord {
        // A value in [-1, 1] in steps of 0.001, from integers only.
        let r = |salt: u32| {
            let h = (t * 4 + c + salt * 977).wrapping_mul(2_654_435_761);
            ((h >> 8) % 2001) as f32 / 1000.0 - 1.0
        };
        let mut car = record(spot.0, spot.1, Controls::default());
        if t < 3 {
            return car;
        }
        let tf = t as f32;
        let steers = [-1.0, -0.6, 0.0, 0.8, 1.0];
        car.body.rot = turned(t + c);
        car.controls.throttle = 1.0;
        match c {
            // Blue: ground turns both ways, yaw 0 as replicar writes them.
            0 => {
                car.body.pos = Vec3A::new(spot.0 + tf * 10.0, spot.1 + tf * 15.0, 17.0);
                car.body.vel = Vec3A::new(600.0, 900.0, 0.0);
                car.controls.steer = steers[(t / 4 % 5) as usize];
                car.controls.boost = (t / 8).is_multiple_of(2);
                car.controls.handbrake = t.is_multiple_of(13);
                car.boost = 0.5;
            }
            // Orange: in the air for a while, a raw steer apart from the yaw, a dodge pressed at 40.
            1 => {
                car.body.pos = Vec3A::new(spot.0 - tf * 4.0, spot.1 - tf * 9.0, 17.0);
                car.body.vel = Vec3A::new(-300.0, -700.0, 0.0);
                car.controls.steer = r(5);
                if (30..60).contains(&t) {
                    car.on_ground = false;
                    car.has_jumped = true;
                    car.body.pos.z = 300.0;
                    car.body.vel.z = 250.0 - tf;
                    car.is_jumping = t < 33;
                    car.jump_time = (t - 29) as f32 / 120.0;
                    car.has_flip = t < 41;
                    car.double_jumped_or_flipped = t >= 41;
                    car.is_flipping = (41..50).contains(&t);
                    car.flip_time = if t >= 41 {
                        (t - 40) as f32 / 120.0
                    } else {
                        0.0
                    };
                    car.controls.pitch = r(1);
                    car.controls.yaw = r(2);
                    car.controls.roll = r(3);
                    car.controls.steer = r(4);
                    car.controls.jump = (30..32).contains(&t) || (40..42).contains(&t);
                }
                car.boost = 0.25 + r(7) * 0.2;
            }
            // Blue: fast (2150 to 2330 uu/s), partial steer with yaw 0, a ground jump at 70.
            2 => {
                car.body.pos = Vec3A::new(spot.0, spot.1 + tf * 18.0, 17.0);
                car.body.vel = Vec3A::new(0.0, 2150.0 + (t % 10) as f32 * 20.0, 0.0);
                car.controls.steer = r(6) * 0.7;
                car.controls.boost = true;
                car.controls.jump = (70..72).contains(&t);
                car.is_jumping = (71..74).contains(&t);
                car.boost = 0.9 - tf * 0.005;
            }
            // Orange: steer and yaw together (the bot's way), an empty tank for a while, air at 70.
            _ => {
                car.body.pos = Vec3A::new(spot.0 + tf, spot.1 - tf * 12.0, 17.0);
                car.body.vel = Vec3A::new(100.0, -1400.0, 0.0);
                car.controls.steer = steers[(t / 3 % 5) as usize];
                car.controls.yaw = car.controls.steer;
                car.controls.throttle = r(8);
                car.boost = if (10..30).contains(&t) { 0.0 } else { 0.4 };
                if (70..80).contains(&t) {
                    car.on_ground = false;
                    car.has_jumped = true;
                    car.body.pos.z = 120.0;
                    car.controls.pitch = r(9);
                    car.controls.roll = r(10);
                }
            }
        }
        car
    }

    /// A 2v2 from a kickoff with ground and air play, turns both ways, jump presses, a frame held twice
    /// (tick 20) and a gap after tick 55: the recording the v1 pin reads.
    fn pin_recording() -> Recording {
        let spots = [
            (-2048.0, -2560.0),
            (2048.0, 2560.0),
            (0.0, -4608.0),
            (0.0, 4608.0),
        ];
        let mut ticks = Vec::new();
        for t in 0..100u32 {
            let tf = t as f32;
            let ball = if t < 3 {
                Body {
                    pos: Vec3A::new(0.0, 0.0, 93.15),
                    ..body(0.0, 0.0)
                }
            } else {
                Body {
                    pos: Vec3A::new(tf * 5.0, -tf * 3.0, 93.15 + tf),
                    vel: Vec3A::new(100.0, -50.0, 20.0),
                    ang_vel: Vec3A::new(0.5, -1.0, 2.0),
                    ..body(0.0, 0.0)
                }
            };
            let tick = Tick {
                frame: if t > 55 { t + 20 } else { t },
                cars: (0..4).map(|c| pin_car(t, c, spots[c as usize])).collect(),
                ball,
                ball_world_contact: false,
            };
            if t == 20 {
                let mut earlier = tick.clone();
                earlier.cars[0].controls.steer = 0.3;
                ticks.push(earlier);
            }
            ticks.push(tick);
        }
        Recording {
            version: 6,
            num_cars: 4,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks,
        }
    }

    /// The samples' hashes: the observations (previous actions inside), masks, cars, teams and counts
    /// exactly; the labels exactly, and rounded to 1/1024. A label goes through `exp`, which comes from the
    /// platform's maths library, so its last bits may differ on another system.
    fn pin_hashes(demos: &Demos) -> (u64, u64, u64) {
        let mut inputs = fnv(FNV_START, &(demos.len() as u64).to_le_bytes());
        inputs = fnv(inputs, &(demos.skipped as u64).to_le_bytes());
        for v in &demos.obs {
            inputs = fnv(inputs, &v.to_bits().to_le_bytes());
        }
        inputs = fnv(inputs, &demos.mask);
        inputs = fnv(inputs, &demos.car);
        inputs = fnv(inputs, &demos.team);
        let (mut exact, mut rounded) = (FNV_START, FNV_START);
        for v in &demos.label {
            exact = fnv(exact, &v.to_bits().to_le_bytes());
            rounded = fnv(rounded, &((v * 1024.0).round() as i32).to_le_bytes());
        }
        (inputs, exact, rounded)
    }

    /// The samples stay the same bit for bit: these hashes pin them.
    #[test]
    fn the_samples_are_pinned() {
        let recording = pin_recording();
        let cases = [
            (
                DemoRules {
                    obs: ObsKind::Native,
                    obs_team_size: 0,
                    tick_skip: 4,
                    action_delay: 1,
                    masks: MaskRule::Standard,
                    softness: 0.1,
                    replicar_rows: false,
                    kickoff_only: 0,
                },
                (
                    0xd14b_0d58_6182_cab2,
                    0x35ab_cfe3_1072_6219,
                    0x8bb0_370b_603b_a4e5,
                ),
            ),
            (
                DemoRules {
                    obs: ObsKind::NitroV2,
                    obs_team_size: 3,
                    tick_skip: 8,
                    action_delay: 3,
                    masks: MaskRule::Standard,
                    softness: 0.1,
                    replicar_rows: true,
                    kickoff_only: 0,
                },
                (
                    0x517e_6c67_5f75_5e31,
                    0x8eb8_807c_bdad_43bc,
                    0x0e33_5ebe_5e07_9944,
                ),
            ),
        ];
        for (rules, (inputs, exact, rounded)) in cases {
            let demos = demonstrations(&recording, &rules).unwrap();
            let got = pin_hashes(&demos);
            assert_eq!((got.0, got.2), (inputs, rounded), "{rules:?}");
            if cfg!(target_os = "linux") {
                assert_eq!(got.1, exact, "{rules:?}");
            }
        }
    }

    /// A copy's rules at delay 1 and skip 4.
    fn copy_rules(replicar_rows: bool) -> DemoRules {
        DemoRules {
            obs: ObsKind::Native,
            obs_team_size: 0,
            tick_skip: 4,
            action_delay: 1,
            masks: MaskRule::Standard,
            softness: 0.1,
            replicar_rows,
            kickoff_only: 0,
        }
    }

    /// A tick at `frame` with the ball away from the centre (no kickoff).
    fn tick_of(frame: u32, cars: Vec<CarRecord>) -> Tick {
        Tick {
            frame,
            cars,
            ball: Body {
                pos: Vec3A::new(500.0, 500.0, 93.15),
                ..body(0.0, 0.0)
            },
            ball_world_contact: false,
        }
    }

    /// One car's rows at frames 0, 1, ... (a one-car recording without a kickoff is Blue's).
    fn one_car(rows: Vec<CarRecord>) -> Recording {
        Recording {
            version: 6,
            num_cars: 1,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks: rows
                .into_iter()
                .enumerate()
                .map(|(i, car)| tick_of(i as u32, vec![car]))
                .collect(),
        }
    }

    fn label_of(demos: &Demos, sample: usize) -> &[f32] {
        &demos.label[sample * NUM_ACTIONS..(sample + 1) * NUM_ACTIONS]
    }

    /// Where a native observation holds the previous action: after the ball's 9 numbers and the pads.
    const PREVIOUS: usize = 9 + NUM_BOOST_PADS;

    #[test]
    fn the_first_choice_is_the_compact_forms_first_entry_a_tie_to_the_lower_action() {
        let ties: [(&[(usize, f32)], usize); 4] = [
            (&[(7, 0.5), (3, 0.5)], 3),
            (&[(10, 0.25), (40, 0.25), (2, 0.25), (88, 0.25)], 2),
            (&[(0, 0.2), (89, 0.4), (1, 0.4)], 1),
            (&[(5, 1.0)], 5),
        ];
        for (shares, lowest) in ties {
            let mut demos = Demos {
                obs_size: 1,
                obs: vec![0.0],
                mask: vec![1; NUM_ACTIONS],
                label: vec![0.0; NUM_ACTIONS],
                car: vec![0],
                team: vec![0],
                skipped: 0,
                outside_window: 0,
            };
            for &(a, share) in shares {
                demos.label[a] = share;
            }
            assert_eq!(first_choice(&demos.label), lowest);
            assert_eq!(usize::from(demos.compact().label_index[0]), lowest);
        }
    }

    /// replicar's ground steering has yaw 0: the label turns as the player did, either way alike, and so
    /// does the previous action.
    #[test]
    fn ground_steering_without_yaw_turns_as_the_player_did() {
        let table = ActionTable::new();
        let mirror = table.mirror_permutation();
        let drive = |steer: f32| {
            let controls = Controls {
                throttle: 1.0,
                steer,
                ..Controls::default()
            };
            one_car((0..10).map(|_| record(0.0, 0.0, controls)).collect())
        };
        let turn = |steer: f32| {
            table_index(
                &table,
                Controls {
                    throttle: 1.0,
                    steer,
                    yaw: steer,
                    ..Controls::default()
                },
            )
        };
        for s in [1.0f32, 0.8, 0.6] {
            let made =
                [s, -s].map(|steer| demonstrations(&drive(steer), &copy_rules(true)).unwrap());
            let least = if s >= 0.8 { 0.98 } else { 0.85 };
            for (demos, side) in made.iter().zip([1.0, -1.0]) {
                assert_eq!(demos.len(), 2);
                let label = label_of(demos, 0);
                assert_eq!(first_choice(label), turn(side), "steer {s}, side {side}");
                assert!(label[turn(side)] > least, "{s}: {}", label[turn(side)]);
                assert_eq!(demos.obs[demos.obs_size + PREVIOUS + 1], side);
            }
            for (a, &m) in mirror.iter().enumerate() {
                assert_eq!(label_of(&made[0], 0)[a], label_of(&made[1], 0)[m]);
            }
        }
    }

    /// The bot's own recordings (and RocketSim's) steer with yaw alongside on the ground: each ground action
    /// is read back as itself.
    #[test]
    fn the_bots_own_ground_actions_are_read_back_as_themselves() {
        let table = ActionTable::new();
        let car = CarSnapshot {
            on_ground: true,
            can_flip: true,
            boost: 50.0,
            body: Body {
                pos: Vec3A::new(0.0, 0.0, 17.0),
                ..Body::RESTING
            },
            ..CarSnapshot::default()
        };
        let mut mask = [0u8; NUM_ACTIONS];
        table.write_mask(MaskRule::Standard, &car, &mut mask);
        for action in (0..NUM_ACTIONS).filter(|&a| !table.controls(a).jump && a < 24) {
            let window = [table.controls(action); 4];
            let label = soft_label(
                &table,
                &mask,
                &window,
                false,
                0.1,
                &label_weights(true),
                None,
            );
            assert_eq!(first_choice(&label), action);
        }
    }

    /// RocketSim ignores steer in the air, and the labels with it.
    #[test]
    fn labels_ignore_steer_in_the_air() {
        let table = ActionTable::new();
        let car = CarSnapshot {
            on_ground: false,
            can_flip: true,
            has_jumped: true,
            boost: 50.0,
            body: Body {
                pos: Vec3A::new(0.0, 0.0, 300.0),
                ..Body::RESTING
            },
            ..CarSnapshot::default()
        };
        let mut mask = [0u8; NUM_ACTIONS];
        table.write_mask(MaskRule::Standard, &car, &mut mask);
        let fitted: Vec<Controls> = (0..4)
            .map(|i| Controls {
                throttle: 1.0,
                steer: 0.7,
                pitch: 0.3 * i as f32 - 0.5,
                yaw: 0.7,
                roll: -0.2,
                ..Controls::default()
            })
            .collect();
        let raw: Vec<Controls> = fitted
            .iter()
            .zip([-1.0, 0.4, 0.0, 1.0])
            .map(|(c, steer)| Controls { steer, ..*c })
            .collect();
        let label = |window: &[Controls]| {
            soft_label(
                &table,
                &mask,
                window,
                false,
                0.1,
                &label_weights(false),
                None,
            )
        };
        assert_eq!(label(&fitted), label(&raw));
    }

    /// replicar's row k holds the controls applied from k on, a recording's record k those that produced
    /// it: the same jump tap sits one row earlier in replicar's and gives the same labels.
    #[test]
    fn replicars_rows_are_read_one_record_earlier() {
        let table = ActionTable::new();
        let tapped = |row: usize| {
            let rows = (0..14).map(|i| {
                let controls = Controls {
                    throttle: 1.0,
                    jump: i == row,
                    ..Controls::default()
                };
                record(0.0, 0.0, controls)
            });
            one_car(rows.collect())
        };
        let jumps = |demos: &Demos| -> Vec<bool> {
            (0..demos.len())
                .map(|i| table.controls(first_choice(label_of(demos, i))).jump)
                .collect()
        };
        let (replicar, rlpr) = (copy_rules(true), copy_rules(false));
        // Decisions at 0, 4 and 8: decision 0 reads rows 1 to 4, its jump held before from row 0.
        let early = demonstrations(&tapped(1), &replicar).unwrap();
        assert_eq!(jumps(&early), [true, false, false]);
        let late = demonstrations(&tapped(5), &replicar).unwrap();
        assert_eq!(jumps(&late), [false, true, false]);
        for (row, made) in [(1, &early), (5, &late)] {
            let same = demonstrations(&tapped(row + 1), &rlpr).unwrap();
            assert_eq!(
                (&same.label, &same.obs),
                (&made.label, &made.obs),
                "row {row}"
            );
        }
        // Without a delay replicar's first decision would need the row before the first: left out.
        let undelayed = DemoRules {
            action_delay: 0,
            ..replicar
        };
        let demos = demonstrations(&tapped(1), &undelayed).unwrap();
        assert_eq!((demos.skipped, demos.len()), (1, 2));
    }

    /// A jump pressed in the air goes the way RocketSim takes from that tick, whatever the sticks do after
    /// (replicar carries a side dodge in yaw and cancels a flip at once).
    #[test]
    fn a_dodge_goes_the_way_of_its_press() {
        let table = ActionTable::new();
        let mirror = table.mirror_permutation();
        let airborne = |controls: Controls| {
            let mut car = record(0.0, 0.0, controls);
            (car.on_ground, car.has_jumped, car.body.pos.z) = (false, true, 300.0);
            car
        };
        // replicar's rows: decision 0 reads rows 1 to 4, the press on row 1.
        let rows = |press: Controls, after: Controls| -> Vec<CarRecord> {
            (0..10)
                .map(|i| {
                    airborne(match i {
                        1 => Controls {
                            jump: true,
                            ..press
                        },
                        2..=4 => after,
                        _ => Controls::default(),
                    })
                })
                .collect()
        };
        let label = |rows: Vec<CarRecord>| {
            let demos = demonstrations(&one_car(rows), &copy_rules(true)).unwrap();
            label_of(&demos, 0).to_vec()
        };
        let jump = |pitch: f32, roll: f32| {
            let controls = Controls {
                pitch,
                roll,
                jump: true,
                handbrake: pitch != 0.0 || roll != 0.0,
                ..Controls::default()
            };
            table_index(&table, controls)
        };
        let sticks = |pitch: f32, yaw: f32, roll: f32| Controls {
            pitch,
            yaw,
            roll,
            ..Controls::default()
        };
        let side = label(rows(
            Controls {
                steer: -0.4,
                ..sticks(0.0, 1.0, 0.0)
            },
            sticks(0.0, 0.1, 0.5),
        ));
        assert!(side[jump(0.0, 1.0)] > 0.99, "{}", side[jump(0.0, 1.0)]);
        let mirrored = label(rows(
            Controls {
                steer: 0.4,
                ..sticks(0.0, -1.0, 0.0)
            },
            sticks(0.0, -0.1, -0.5),
        ));
        assert!(mirrored[jump(0.0, -1.0)] > 0.99);
        for (a, &m) in mirror.iter().enumerate() {
            assert_eq!(side[a], mirrored[m]);
        }
        let cases = [
            // A front flip cancelled at once.
            (sticks(-1.0, 0.0, 0.0), sticks(1.0, 0.0, 0.0), (-1.0, 0.0)),
            (sticks(-0.3, 0.3, 0.0), Controls::default(), (-1.0, 1.0)),
            // Under the deadzone: a double jump.
            (sticks(0.0, 0.45, 0.0), Controls::default(), (0.0, 0.0)),
            (sticks(0.0, 0.55, 0.0), Controls::default(), (0.0, 1.0)),
            (sticks(0.0, 0.0, 0.0), sticks(1.0, 0.0, 0.0), (0.0, 0.0)),
        ];
        for (press, after, (pitch, roll)) in cases {
            let got = label(rows(press, after));
            assert_eq!(first_choice(&got), jump(pitch, roll), "{press:?}");
            assert!(got[jump(pitch, roll)] > 0.99, "{press:?}");
        }
        // A first jump from the ground, and a press on a tick the car has landed, keep the window's means.
        let means = |rows: &[CarRecord]| {
            let snapshot = car_snapshot(&rows[0], Team::Blue, false);
            let mut mask = [0u8; NUM_ACTIONS];
            table.write_mask(MaskRule::Standard, &snapshot, &mut mask);
            let applied: Vec<Controls> = rows[1..5].iter().map(|r| r.controls).collect();
            let weights = label_weights(snapshot.on_ground);
            soft_label(&table, &mask, &applied, false, 0.1, &weights, None).to_vec()
        };
        let mut landed = rows(sticks(-1.0, 0.0, 0.0), sticks(1.0, 0.0, 0.0));
        landed[1].on_ground = true;
        let mut from_ground = landed.clone();
        from_ground[0].on_ground = true;
        for rows in [landed, from_ground] {
            assert_eq!(label(rows.clone()), means(&rows));
        }
    }

    /// `rows` (replicar's: each with the controls applied next) as a recording holds them (each record with
    /// the controls that produced it): the same states, each control one record later.
    fn recorded_from(rows: &[CarRecord]) -> Vec<CarRecord> {
        let mut recorded = rows.to_vec();
        recorded.push(rows[rows.len() - 1]);
        for (k, car) in recorded.iter_mut().enumerate() {
            car.controls = if k == 0 {
                Controls::default()
            } else {
                rows[k - 1].controls
            };
        }
        recorded
    }

    /// The samples of replicar's `rows` under `rules`, the same from the rows written the recording's way:
    /// in both a control acts on the state of the same record.
    fn both_ways(rows: &[CarRecord], rules: DemoRules) -> Demos {
        let replicar = DemoRules {
            replicar_rows: true,
            ..rules
        };
        let made = demonstrations(&one_car(rows.to_vec()), &replicar).unwrap();
        let recorded = DemoRules {
            replicar_rows: false,
            ..rules
        };
        let same = demonstrations(&one_car(recorded_from(rows)), &recorded).unwrap();
        assert_eq!((&same.label, &same.obs), (&made.label, &made.obs));
        made
    }

    fn airborne(mut car: CarRecord) -> CarRecord {
        (car.on_ground, car.has_jumped, car.body.pos.z) = (false, true, 300.0);
        car
    }

    /// Each control turns by the axis RocketSim uses on the tick it acts on, whatever the decision saw:
    /// landing at once after a decision in the air, the steering on the ground counts (replicar's yaw is
    /// 0 there); taking off after one on the ground, the fitted yaw counts and not the raw steer.
    #[test]
    fn each_control_turns_by_the_axis_rocketsim_uses_on_its_tick() {
        let table = ActionTable::new();
        let mirror = table.mirror_permutation();
        // Decision 0 reads rows 1 to 4 (replicar's rows, delay 1, skip 4).
        let landing = |side: f32| -> Vec<CarRecord> {
            (0..10)
                .map(|i| {
                    let controls = Controls {
                        throttle: 1.0,
                        steer: side,
                        ..Controls::default()
                    };
                    let car = record(0.0, 0.0, controls);
                    if i == 0 { airborne(car) } else { car }
                })
                .collect()
        };
        let taking_off = |side: f32| -> Vec<CarRecord> {
            let raw = [-1.0, -0.4, 0.0, 0.2];
            (0..10)
                .map(|i| {
                    let controls = Controls {
                        throttle: 1.0,
                        steer: raw[i % 4] * side,
                        yaw: side,
                        ..Controls::default()
                    };
                    let car = record(0.0, 0.0, controls);
                    if i == 0 { car } else { airborne(car) }
                })
                .collect()
        };
        let turning = |c: Controls, side: f32| c.steer == side && c.yaw == side && !c.jump;
        let cases: [fn(f32) -> Vec<CarRecord>; 2] = [landing, taking_off];
        for rows in cases {
            let made = [1.0, -1.0].map(|side| both_ways(&rows(side), copy_rules(true)));
            for (demos, side) in made.iter().zip([1.0, -1.0]) {
                let label: [f32; NUM_ACTIONS] = label_of(demos, 0).try_into().unwrap();
                let first = table.controls(first_choice(&label));
                assert!(turning(first, side), "{side}: {first:?}");
                let share = share_of(&table, &label, |c| turning(c, side));
                assert!(share > 0.99, "{side}: {share}");
            }
            // Either way alike (to the last bits of the shares' sum).
            for (a, &m) in mirror.iter().enumerate() {
                let (one, other) = (label_of(&made[0], 0)[a], label_of(&made[1], 0)[m]);
                assert!((one - other).abs() <= 1e-6 * one.max(other), "{a}");
            }
        }
    }

    /// A jump pressed anew in the air after a decision that saw the car on the ground (it has taken off
    /// since) is a dodge too, at any delay; and a straight dodge is its own row, not the plain jump, though
    /// the decision's weights count handbrake (only the directional jumps carry it).
    #[test]
    fn a_press_in_the_air_after_a_decision_on_the_ground_is_a_dodge() {
        let table = ActionTable::new();
        let jump = |pitch: f32, roll: f32| {
            let controls = Controls {
                pitch,
                roll,
                jump: true,
                handbrake: pitch != 0.0 || roll != 0.0,
                ..Controls::default()
            };
            table_index(&table, controls)
        };
        let held = Controls {
            throttle: 1.0,
            ..Controls::default()
        };
        let presses = [
            // A side dodge as replicar writes it: in yaw, roll 0, the raw steer apart.
            (
                Controls {
                    steer: -0.4,
                    yaw: 1.0,
                    ..held
                },
                (0.0, 1.0),
            ),
            (
                Controls {
                    steer: 0.4,
                    yaw: -1.0,
                    ..held
                },
                (0.0, -1.0),
            ),
            // Straight front and back flips.
            (
                Controls {
                    pitch: -1.0,
                    ..held
                },
                (-1.0, 0.0),
            ),
            (Controls { pitch: 1.0, ..held }, (1.0, 0.0)),
        ];
        for (skip, delay) in [(4u32, 3u32), (8, 7), (4, 1)] {
            // replicar's rows: on the ground at decision 0, in the air from row 1, the press on the window's
            // second row.
            let rows = |press: Controls| -> Vec<CarRecord> {
                (0..2 * (skip + delay) as usize)
                    .map(|i| {
                        let controls = if i == delay as usize + 1 {
                            Controls {
                                jump: true,
                                ..press
                            }
                        } else {
                            held
                        };
                        let car = record(0.0, 0.0, controls);
                        if i == 0 { car } else { airborne(car) }
                    })
                    .collect()
            };
            let rules = DemoRules {
                tick_skip: skip,
                action_delay: delay,
                ..copy_rules(true)
            };
            for (press, (pitch, roll)) in presses {
                let demos = both_ways(&rows(press), rules);
                let label = label_of(&demos, 0);
                let want = jump(pitch, roll);
                let case = format!("skip {skip}, delay {delay}, {press:?}");
                assert_eq!(first_choice(label), want, "{case}");
                assert!(label[want] > 0.99, "{case}: {}", label[want]);
            }
        }
    }

    #[test]
    fn a_dodges_way_mirrors_with_the_sticks_and_is_one_of_the_tables() {
        let table = ActionTable::new();
        let values = [-1.5f32, -1.0, -0.55, -0.3, -0.05, 0.0, 0.05, 0.3, 0.55, 1.0];
        for pitch in values {
            for yaw in values {
                for roll in values {
                    let sticks = Controls {
                        pitch,
                        yaw,
                        roll,
                        ..Controls::default()
                    };
                    let way = dodge_direction(&sticks);
                    let mirrored = dodge_direction(&Controls {
                        yaw: -yaw,
                        roll: -roll,
                        ..sticks
                    });
                    assert_eq!(mirrored, (way.0, -way.1), "{sticks:?}");
                    assert!((0..NUM_ACTIONS).any(|a| {
                        let c = table.controls(a);
                        c.jump && (c.pitch, c.roll) == way
                    }));
                }
            }
        }
    }

    fn pad_at(x: f32, y: f32) -> usize {
        pad_layout()
            .iter()
            .position(|(pos, _)| pos.x == x && pos.y == y)
            .unwrap()
    }

    fn rebuilt_through(ticks: &[Tick]) -> Rebuilt {
        let refs: Vec<&Tick> = ticks.iter().collect();
        let mut rebuilt = Rebuilt::new();
        rebuilt.advance(&refs, refs.len() - 1);
        rebuilt
    }

    fn holding(x: f32, y: f32, boost: f32) -> CarRecord {
        CarRecord {
            boost,
            ..record(x, y, Controls::default())
        }
    }

    /// The pads are rebuilt from the cars' boost as RocketSim keeps them. RocketSim gives the boost
    /// during the step into the tick that shows it, so a pad's cooldown counts from the tick before.
    #[test]
    fn the_pads_are_rebuilt_from_the_cars_boost() {
        let (small, big) = (pad_at(0.0, -4240.0), pad_at(3072.0, -4096.0));
        let rise = |x: f32, y: f32, from: f32, to: f32, frames: u32| -> Vec<Tick> {
            (0..frames)
                .map(|f| tick_of(f, vec![holding(x, y, if f == 0 { from } else { to })]))
                .collect()
        };
        // A small pad's 12: 4 s, active again 480 ticks after it gave the boost.
        let rebuilt = rebuilt_through(&rise(0.0, -4240.0, 0.2, 0.3216, 3));
        assert_eq!(rebuilt.cooldown(small, 1), 4.0 - TICK_TIME);
        assert_eq!(rebuilt.cooldown(small, 6), 4.0 - 6.0 * TICK_TIME);
        assert!(rebuilt.cooldown(small, 479) > 0.0 && rebuilt.cooldown(small, 480) == 0.0);
        assert_eq!(rebuilt.taken.iter().filter(|t| t.is_some()).count(), 1);
        // A gap carries the cooldowns on by its frames: 30 frames, a quarter of a second.
        assert!((rebuilt.cooldown(small, 1) - rebuilt.cooldown(small, 31) - 0.25).abs() < 1e-5);
        // A big pad's fill: 10 s. Only a rise above a small pad's takes a big one.
        let rebuilt = rebuilt_through(&rise(3072.0, -4096.0, 0.5, 1.0, 2));
        assert_eq!(rebuilt.cooldown(big, 1), 10.0 - TICK_TIME);
        let rebuilt = rebuilt_through(&rise(3072.0, -4096.0, 0.5, 0.62, 2));
        assert!(rebuilt.taken.iter().all(Option::is_none));
        // ... or one that fills the tank (the round trip below meets one: 87.6 to 100).
        let rebuilt = rebuilt_through(&rise(3072.0, -4096.0, 0.876, 1.0, 2));
        assert_eq!(rebuilt.cooldown(big, 1), 10.0 - TICK_TIME);
        // Within reach (the radius and 150 uu) the nearest free pad; beyond it, none.
        let rebuilt = rebuilt_through(&rise(0.0, -4240.0 + 144.0 + 140.0, 0.2, 0.32, 2));
        assert!(rebuilt.taken[small].is_some());
        let rebuilt = rebuilt_through(&rise(0.0, -4240.0 + 144.0 + 160.0, 0.2, 0.32, 2));
        assert!(rebuilt.taken.iter().all(Option::is_none));
        // A full tank crossing a pad takes nothing, and a respawn's third of a tank neither.
        for (from, to) in [(1.0, 1.0), (0.0, 0.3333)] {
            let rebuilt = rebuilt_through(&rise(0.0, -4240.0, from, to, 2));
            assert!(rebuilt.taken.iter().all(Option::is_none), "{from} -> {to}");
        }
        // Two cars rising on one pad in the same tick: the first takes it, the second finds it taken.
        let both = |f: u32, boost: f32| {
            tick_of(
                f,
                vec![holding(0.0, -4240.0, boost), holding(10.0, -4240.0, boost)],
            )
        };
        let rebuilt = rebuilt_through(&[both(0, 0.2), both(1, 0.32)]);
        assert_eq!(rebuilt.taken.iter().filter(|t| t.is_some()).count(), 1);
        // No rise across a gap counts, and a kickoff frees every pad.
        let mut ticks = rise(0.0, -4240.0, 0.2, 0.32, 2);
        ticks.push(tick_of(10, vec![holding(0.0, -4240.0, 0.5)]));
        let rebuilt = rebuilt_through(&ticks);
        assert_eq!(rebuilt.taken[small], Some(0));
        assert_eq!(rebuilt.taken.iter().filter(|t| t.is_some()).count(), 1);
        let mut kickoff = tick_of(11, vec![holding(0.0, -4608.0, 0.3333)]);
        kickoff.ball.pos = Vec3A::new(0.0, 0.0, 93.15);
        ticks.push(kickoff);
        assert!(rebuilt_through(&ticks).taken.iter().all(Option::is_none));
    }

    /// The observation shows the rebuilt pads, each team from its own end (Orange's turned around).
    #[test]
    fn the_observations_show_the_rebuilt_pads() {
        let mut rec = recording(9);
        for tick in &mut rec.ticks[3..] {
            // Blue on the small pad at its back, Orange on the big pad at its back right.
            (tick.cars[0].body.pos, tick.cars[0].boost) = (Vec3A::new(0.0, -4240.0, 17.0), 0.45);
            (tick.cars[1].body.pos, tick.cars[1].boost) = (Vec3A::new(-3072.0, 4096.0, 17.0), 1.0);
        }
        let (small, big) = (pad_at(0.0, -4240.0), pad_at(-3072.0, 4096.0));
        let rules = DemoRules {
            obs: ObsKind::Native,
            obs_team_size: 0,
            tick_skip: 4,
            action_delay: 0,
            masks: MaskRule::Standard,
            softness: 0.1,
            replicar_rows: false,
            kickoff_only: 0,
        };
        let demos = demonstrations(&rec, &rules).unwrap();
        // Decision 4, Blue then Orange; the pads were taken during the step into tick 3.
        let (blue, orange) = (2 * demos.obs_size, 3 * demos.obs_size);
        let shown = |cooldown: f32| 1.0 / (1.0 + cooldown);
        let (small_left, big_left) = (4.0 - 2.0 * TICK_TIME, 10.0 - 2.0 * TICK_TIME);
        let flipped = |pad: usize| NUM_BOOST_PADS - 1 - pad;
        assert_eq!(demos.obs[blue + 9 + small], shown(small_left));
        assert_eq!(demos.obs[blue + 9 + big], shown(big_left));
        assert_eq!(demos.obs[orange + 9 + flipped(small)], shown(small_left));
        assert_eq!(demos.obs[orange + 9 + flipped(big)], shown(big_left));
        let others = (0..NUM_BOOST_PADS).filter(|&p| p != small && p != big);
        assert!(others.clone().all(|p| demos.obs[blue + 9 + p] == 1.0));
    }

    fn speeds(speeds: &[(u32, f32)]) -> Vec<bool> {
        let ticks: Vec<Tick> = speeds
            .iter()
            .map(|&(frame, speed)| {
                let mut car = record(0.0, 0.0, Controls::default());
                car.body.vel = Vec3A::new(speed, 0.0, 0.0);
                tick_of(frame, vec![car])
            })
            .collect();
        let refs: Vec<&Tick> = ticks.iter().collect();
        let mut rebuilt = Rebuilt::new();
        (0..refs.len())
            .map(|i| {
                rebuilt.advance(&refs, i);
                rebuilt.supersonic[0]
            })
            .collect()
    }

    /// A car stays supersonic as RocketSim keeps it: on at 2200 uu/s, kept down to 2100 for up to a
    /// second. RocketSim adds up that second 1/120 at a time in f32, which after 120 ticks falls just
    /// short of 1, so the flag goes on the 121st tick below 2200.
    #[test]
    fn a_car_stays_supersonic_as_rocketsim_keeps_it() {
        let band: Vec<(u32, f32)> = (0..130)
            .map(|f| (f, if f < 5 { 2250.0 } else { 2150.0 }))
            .collect();
        let flags = speeds(&band);
        assert!(flags[..5 + 120].iter().all(|&on| on));
        assert!(!flags[5 + 120]);
        // Below 2100 the flag goes at once; below 2200 it never comes.
        assert_eq!(speeds(&[(0, 2250.0), (1, 2050.0)]), [true, false]);
        assert!(
            speeds(&[(0, 2150.0), (1, 2190.0), (2, 2199.0)])
                .iter()
                .all(|&on| !on)
        );
        // A gap starts the rule again.
        assert_eq!(
            speeds(&[(0, 2250.0), (1, 2150.0), (5, 2150.0)]),
            [true, true, false]
        );
    }

    /// The pads come in the simulator's order.
    #[test]
    fn the_rebuilt_pads_are_in_the_simulators_order() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let sim = crate::sim::Sim::new(1, false, 0);
        let ours: Vec<[f32; 4]> = pad_layout()
            .iter()
            .map(|&(pos, big)| [pos.x, pos.y, pos.z, f32::from(u8::from(big))])
            .collect();
        assert_eq!(ours, sim.pad_layout());
    }

    /// RocketSim's own count of the supersonic second: a car held at 2150 uu/s after 2250 keeps the flag
    /// exactly as long as the rebuild does.
    #[test]
    fn rocketsim_keeps_a_car_supersonic_as_long_as_the_rebuild() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let mut sim = crate::sim::Sim::new(1, false, 0);
        let arena = sim.arena_mut();
        let (mut band, mut flags) = (Vec::new(), Vec::new());
        for frame in 0..135u32 {
            arena.edit_car_state_full(0, |state| {
                state.phys.pos = Vec3A::new(0.0, 0.0, 1000.0);
                state.phys.vel = Vec3A::new(if frame < 5 { 2250.0 } else { 2150.0 }, 0.0, 0.0);
                state.phys.ang_vel = Vec3A::ZERO;
                state.set_contact_flags(false, [false; 4]);
            });
            arena.step_tick();
            let after = arena.get_car_state(0);
            flags.push(after.is_supersonic);
            band.push((frame, after.phys.vel));
        }
        let ticks: Vec<Tick> = band
            .iter()
            .map(|&(frame, vel)| {
                let mut car = record(0.0, 0.0, Controls::default());
                car.body.vel = vel;
                tick_of(frame, vec![car])
            })
            .collect();
        let refs: Vec<&Tick> = ticks.iter().collect();
        let mut rebuilt = Rebuilt::new();
        let ours: Vec<bool> = (0..refs.len())
            .map(|i| {
                rebuilt.advance(&refs, i);
                rebuilt.supersonic[0]
            })
            .collect();
        assert_eq!(ours, flags);
        assert_eq!(flags.iter().filter(|&&on| on).count(), 5 + 120);
    }

    /// The round trip: the simulator plays a 1v1 with scripted table actions under delay 1 and skip 4
    /// (boosting over pads, coasting at supersonic speed, turns both ways, a jump and a front flip), recorded
    /// tick by tick as a recording holds it (each record with the controls that produced it). The samples
    /// give back the scripted actions as first choices and the simulator's own observations, pads and
    /// supersonic included. Written as replicar writes it (each row with the controls applied next) and read with
    /// `replicar_rows`, the same samples.
    #[test]
    fn the_samples_give_back_what_the_simulator_played_and_saw() {
        use crate::sim::Sim;
        use crate::snapshot::StepEvents;
        if !crate::testing::meshes_loaded() {
            return;
        }
        let (skip, delay, ticks) = (4usize, 1usize, 840usize);
        let table = ActionTable::new();
        let mut sim = Sim::new(1, true, 0);
        sim.reset_to_kickoff(0);
        let spot = |x: f32, y: f32, yaw: f32| Body {
            pos: Vec3A::new(x, y, 17.0),
            rot: Mat3A::from_rotation_z(yaw),
            vel: Vec3A::ZERO,
            ang_vel: Vec3A::ZERO,
        };
        let half_pi = std::f32::consts::FRAC_PI_2;
        let ball = Body {
            pos: Vec3A::new(0.0, 0.0, 93.15),
            ..Body::RESTING
        };
        let spawn = SPAWN_AMOUNT;
        sim.place(
            ball,
            &[
                (spot(-2048.0, -2560.0, half_pi), spawn, true),
                (spot(2048.0, 2560.0, -half_pi), spawn, true),
            ],
        );
        // Each car's way: Blue up its left lane over pads, Orange down its own; each turns both ways.
        let ways = [
            [
                (-2048.0, -1036.0),
                (-3584.0, 0.0),
                (-1788.0, 2300.0),
                (-3200.0, 4000.0),
            ],
            [
                (2048.0, 1036.0),
                (3584.0, 0.0),
                (1788.0, -2300.0),
                (3200.0, -4000.0),
            ],
        ];
        let mut next_way = [0usize; 2];
        // Blue's speed: boost to 2280, coast to 2170, hold there with throttle for 1.3 s, then go on.
        let (mut coasting, mut holding) = (false, None::<usize>);
        let mut decide = |car: usize, k: usize, me: &CarSnapshot| -> Controls {
            let ahead = me.body.forward();
            let way = &mut next_way[car];
            let mut to = Vec3A::ZERO;
            while *way < 4 {
                let (x, y) = ways[car][*way];
                to = Vec3A::new(x, y, 0.0) - me.body.pos.with_z(0.0);
                if to.length() > 350.0 && to.dot(ahead) > 0.0 {
                    break;
                }
                *way += 1;
            }
            let cross = ahead.x * to.y - ahead.y * to.x;
            let angle = cross.atan2(ahead.dot(to));
            let steer = if *way >= 4 || angle.abs() < 0.15 {
                0.0
            } else {
                angle.signum()
            };
            let speed = me.body.vel.length();
            let (throttle, boost) = if car == 0 {
                if holding.is_some_and(|from| k > from + 40) {
                    (1.0, k % 6 < 3)
                } else if holding.is_some() {
                    (1.0, false)
                } else if coasting {
                    if speed < 2170.0 {
                        holding = Some(k);
                    }
                    (if speed < 2170.0 { 1.0 } else { 0.0 }, false)
                } else {
                    coasting = speed >= 2280.0;
                    (1.0, !coasting)
                }
            } else {
                (1.0, k > 20 && k % 4 < 2)
            };
            // Orange jumps at decision 10 and flips forwards at 12.
            if car == 1 && (k == 10 || k == 12) {
                let pitch = if k == 12 { -1.0 } else { 0.0 };
                return Controls {
                    pitch,
                    jump: true,
                    handbrake: pitch != 0.0,
                    ..Controls::default()
                };
            }
            Controls {
                throttle,
                steer,
                yaw: steer,
                boost: boost && me.boost > 0.0,
                ..Controls::default()
            }
        };
        let record_of = |snap: &GameSnapshot, events: &StepEvents, car: usize, controls| {
            let (c, d) = (&snap.cars[car], &events.details[car]);
            CarRecord {
                body: c.body,
                on_ground: c.on_ground,
                is_jumping: d.is_jumping,
                is_flipping: d.is_flipping,
                jump_time: d.jump_time,
                flip_time: d.flip_time,
                has_jumped: c.has_jumped,
                double_jumped_or_flipped: c.has_double_jumped || c.has_flipped,
                has_flip: c.can_flip,
                flip_rel_torque: d.flip_rel_torque,
                boost: c.boost / 100.0,
                touching_ball: false,
                controls,
                touching_car: false,
            }
        };
        let (mut snap, mut events) = (GameSnapshot::default(), StepEvents::default());
        sim.read(&mut snap);
        events.clear(2);
        sim.details(&mut events);
        let mut records = vec![Tick {
            frame: 0,
            cars: (0..2)
                .map(|car| record_of(&snap, &events, car, Controls::default()))
                .collect(),
            ball: snap.ball,
            ball_world_contact: false,
        }];
        // Per decision: each car's action and the simulator's observation then.
        let mut decisions: Vec<([usize; 2], [Vec<f32>; 2])> = Vec::new();
        let mut applied = [Controls::default(); 2];
        let mut mask = [0u8; NUM_ACTIONS];
        let mut seen = (0, 0, 0, 0);
        for tick in 0..ticks {
            if tick % skip == 0 {
                let k = tick / skip;
                let mut actions = [0; 2];
                let obs = [0, 1].map(|car| {
                    let mut wanted = decide(car, k, &snap.cars[car]);
                    table.write_mask(MaskRule::Standard, &snap.cars[car], &mut mask);
                    if !(0..NUM_ACTIONS).any(|a| mask[a] == 1 && table.controls(a) == wanted) {
                        // Not allowed here (in the air, say): turn the same way without throttle.
                        wanted = Controls {
                            steer: wanted.steer,
                            yaw: wanted.steer,
                            ..Controls::default()
                        };
                    }
                    actions[car] = table_index(&table, wanted);
                    assert_eq!(mask[actions[car]], 1, "car {car} decision {k}");
                    let mut out = vec![0.0; ObsKind::Native.layout(1, true, 0).size];
                    ObsKind::Native.build(&snap, car, Controls::default(), 0, &mut out);
                    out
                });
                let cars = &snap.cars;
                seen.0 += usize::from(snap.pads.iter().any(|p| !p.active));
                seen.1 += usize::from(
                    cars.iter()
                        .any(|c| c.is_supersonic && c.body.vel.length() < 2200.0),
                );
                seen.2 += usize::from(cars[1].has_flipped);
                let speed = cars[0].body.vel.length();
                seen.3 += usize::from(!cars[0].is_supersonic && (2100.0..2200.0).contains(&speed));
                decisions.push((actions, obs));
            }
            if tick >= delay && (tick - delay) % skip == 0 {
                let (actions, _) = decisions[(tick - delay) / skip];
                applied = actions.map(|a| table.controls(a));
            }
            for (car, &controls) in applied.iter().enumerate() {
                sim.set_controls(car, controls);
            }
            events.clear(2);
            events.step_seconds = 1.0 / 120.0;
            sim.step(1, &mut events);
            sim.read(&mut snap);
            sim.details(&mut events);
            records.push(Tick {
                frame: tick as u32 + 1,
                cars: (0..2)
                    .map(|car| record_of(&snap, &events, car, applied[car]))
                    .collect(),
                ball: snap.ball,
                ball_world_contact: false,
            });
        }
        assert!(
            seen.0 > 0 && seen.1 > 0 && seen.2 > 0 && seen.3 > 0,
            "pads taken, supersonic below 2200, a flip, the second below 2200 run out: {seen:?}"
        );
        let turns: Vec<f32> = decisions
            .iter()
            .flat_map(|(actions, _)| actions.map(|a| table.controls(a).steer))
            .collect();
        assert!(turns.contains(&1.0) && turns.contains(&-1.0));
        let recording = Recording {
            version: 6,
            num_cars: 2,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks: records,
        };
        let demos = demonstrations(&recording, &copy_rules(false)).unwrap();
        assert_eq!(demos.skipped, 0);
        assert_eq!(demos.len(), 2 * ((ticks - delay - skip) / skip + 1));
        let mut wrong = Vec::new();
        for (i, (actions, obs)) in decisions.iter().take(demos.len() / 2).enumerate() {
            for car in 0..2 {
                let sample = 2 * i + car;
                let choice = first_choice(label_of(&demos, sample));
                if choice != actions[car] {
                    wrong.push((i, car, actions[car], choice));
                }
                // The pads exactly; the rest but the previous action to a millionth (a record holds the
                // boost over 100, which does not always come back to the same last bit).
                let ours = &demos.obs[sample * demos.obs_size..(sample + 1) * demos.obs_size];
                for (slot, (a, b)) in ours.iter().zip(&obs[car]).enumerate() {
                    if (9..PREVIOUS).contains(&slot) {
                        assert_eq!(a, b, "decision {i}, car {car}, pad {}", slot - 9);
                    } else if !(PREVIOUS..PREVIOUS + 8).contains(&slot) {
                        assert!(
                            (a - b).abs() <= 1e-6,
                            "decision {i}, car {car}, slot {slot}: {a} {b}"
                        );
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
        // replicar's way: row k holds the controls applied from it, the last state left without them.
        let mut rows = recording.clone();
        for k in 0..rows.ticks.len() - 1 {
            for car in 0..2 {
                rows.ticks[k].cars[car].controls = recording.ticks[k + 1].cars[car].controls;
            }
        }
        rows.ticks.pop();
        let again = demonstrations(&rows, &copy_rules(true)).unwrap();
        assert_eq!((again.len(), again.skipped), (demos.len(), 0));
        assert_eq!(
            (&again.obs, &again.label, &again.mask),
            (&demos.obs, &demos.label, &demos.mask)
        );
    }
}
