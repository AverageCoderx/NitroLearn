//! Ball prediction: where the ball will be, from an arena of its own that holds only the ball.
//!
//! A `BallPath` keeps the ball's predicted path, one body every `stride` ticks. While the real ball is where
//! the path said it would be, the path carries on: its first bodies are dropped and the arena, which sits at
//! the path's far end, is stepped on to add as many again. A touch or a reset predicts the whole path afresh.
//! Checking the real ball at every step keeps the far end close too: a ball within the tolerance lands within
//! a few uu of where a fresh prediction would put it, so no refresh on a timer is needed. The idea is
//! RocketSim's `BallPredTracker` (MIT, ZealanL) and buta-ppo's observation (MIT, ButaMasta), which carries
//! on one decision at a time with tolerances; see CREDITS.md.
//!
//! Prediction only follows the ball: cars are not in its arena, so a path never foresees a touch.

use std::collections::VecDeque;

use crate::sim::Sim;
use crate::snapshot::{Body, StepEvents};

/// Physics ticks per second.
pub const TICK_RATE: f32 = 120.0;
/// The furthest a prediction time may reach: RLBot's own prediction covers 6 s, which the bot reads.
pub const MAX_SECONDS: f32 = 6.0;
/// RLBot's prediction is 720 slices a tick apart, the first one tick after the packet it is made with, so
/// it reaches 720 ticks ahead of that packet. The furthest decision is kept a tick short of that, so a
/// prediction a tick older than its packet still reaches it.
pub const MAX_TICKS: u32 = 719;
/// The most ticks RLBot's first slice may lie after the packet: 1 when the prediction is made from the
/// packet's own state; more means the two do not belong together.
pub const MAX_LEAD_TICKS: f32 = 2.0;
/// How far a slice that lies ahead of the packet may be from the packet's ball moved on at its own velocity,
/// per tick ahead (uu): well above the few uu a bounce in that tick bends the path (2.7 measured,
/// `a_bounce_after_the_packet_keeps_a_fresh_prediction`). A stale prediction (from before a touch or a
/// reset) is caught by its time instead: it is not ahead of its packet.
pub const LEAD_POS_PER_TICK: f32 = 30.0;
/// At most this many prediction times.
pub const MAX_TIMES: usize = 8;
/// Observation numbers per prediction time: position, velocity, position and velocity relative to the car.
pub const VALUES_PER_TIME: usize = 12;
/// The default times when ball prediction is on (seconds).
pub const DEFAULT_TIMES: [f32; 4] = [0.5, 1.0, 2.0, 3.0];

/// How close the real ball must be to the path's next body for the path to carry on (distances between
/// the two states, each part on its own).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// uu
    pub pos: f32,
    /// uu/s
    pub vel: f32,
    /// rad/s
    pub ang_vel: f32,
}

impl Tolerance {
    /// Nothing but rounding: a ball followed tick by tick.
    pub const EXACT: Self = Self {
        pos: 0.05,
        vel: 0.05,
        ang_vel: 0.001,
    };
    /// RLBot's prediction against the packet it is read for: its slice for now must be the packet's
    /// ball, give or take a tick of the game's own physics and the packet's rounding; a touch or a reset since
    /// it was made moves the ball far more.
    pub const RLBOT: Self = Self {
        pos: 2.0,
        vel: 10.0,
        // A ball rolling with a slip changes its spin by about 1.2 rad/s in one tick (measured), so a
        // prediction a tick old disagrees on spin by that much without being stale.
        ang_vel: 3.0,
    };
    /// For the observation: buta-ppo's position and velocity margins (1 uu, 3.5 uu/s); its spin
    /// margin (1.5 rad/s) is tightened to 0.1 rad/s, as spin changes every bounce. A ball off by a full
    /// margin moved the far end of a 3 s path by 9 uu at most, a tenth of the ball's radius
    /// (`a_ball_within_tolerance_lands_where_the_path_says`).
    pub const OBSERVATION: Self = Self {
        pos: 1.0,
        vel: 3.5,
        ang_vel: 0.1,
    };

    pub fn holds(&self, a: &Body, b: &Body) -> bool {
        a.pos.distance_squared(b.pos) <= self.pos * self.pos
            && a.vel.distance_squared(b.vel) <= self.vel * self.vel
            && a.ang_vel.distance_squared(b.ang_vel) <= self.ang_vel * self.ang_vel
    }
}

/// The decisions ahead each prediction time stands for: `times` (seconds) rounded to whole decisions of
/// `tick_skip` ticks, at least one. An error, saying why, for times that are not increasing, not above 0,
/// beyond `MAX_SECONDS`, too many, or two that round to the same decision.
pub fn decisions(times: &[f32], tick_skip: u32) -> Result<Vec<u32>, String> {
    if times.len() > MAX_TIMES {
        return Err(format!(
            "ball_prediction takes at most {MAX_TIMES} times, not {}",
            times.len()
        ));
    }
    let mut steps: Vec<u32> = Vec::with_capacity(times.len());
    for &time in times {
        if !(time > 0.0 && time <= MAX_SECONDS) {
            return Err(format!(
                "ball_prediction times must be above 0 and at most {MAX_SECONDS} seconds, not {time}"
            ));
        }
        let step = ((time * TICK_RATE / tick_skip as f32).round() as u32).max(1);
        if steps.last().is_some_and(|&last| step <= last) {
            return Err(format!(
                "ball_prediction times must increase and land on different decisions ({tick_skip} ticks \
                 each); {time} s does not"
            ));
        }
        steps.push(step);
    }
    // The last decision must lie within RLBot's prediction, which the bot picks it from, even when that
    // prediction is a tick older than its packet (`MAX_TICKS`).
    if let Some(&last) = steps.last()
        && last * tick_skip > MAX_TICKS
    {
        return Err(format!(
            "ball_prediction's last time rounds to {} ticks ahead ({:.3} s), beyond the {MAX_TICKS} ticks \
             (5.99 s) the bot can read from RLBot's prediction",
            last * tick_skip,
            (last * tick_skip) as f32 / TICK_RATE
        ));
    }
    Ok(steps)
}

/// The ball's predicted path: `path[k]` is the ball `(k + 1) * stride` ticks after the state it was last
/// given.
pub struct BallPath {
    sim: Sim,
    events: StepEvents,
    path: VecDeque<Body>,
    stride: u32,
    len: usize,
    tolerance: Tolerance,
    /// Paths predicted afresh, and calls to `update`.
    pub fresh: u64,
    pub updates: u64,
}

impl BallPath {
    /// A path of `len` bodies `stride` ticks apart, in a ball-only soccar arena. Needs the collision meshes
    /// loaded.
    pub fn new(stride: u32, len: usize, tolerance: Tolerance) -> Self {
        assert!(stride > 0 && len > 0, "a path needs a stride and a length");
        Self {
            sim: Sim::new(0, false, 0),
            events: StepEvents::default(),
            path: VecDeque::with_capacity(len),
            stride,
            len,
            tolerance,
            fresh: 0,
            updates: 0,
        }
    }

    /// The observation's path for `tick_skip` and the furthest decision ahead it reads (`last`).
    pub fn for_observation(tick_skip: u32, last: u32) -> Self {
        Self::new(tick_skip, last as usize, Tolerance::OBSERVATION)
    }

    /// The path for a ball now at `ball`, `strides` strides after the previous call: carried on when the
    /// ball is within the tolerance of where the path put it, else (and after any `forget`) predicted
    /// afresh.
    pub fn update(&mut self, ball: Body, strides: usize) -> &[Body] {
        self.updates += 1;
        let carries_on = self.path.len() == self.len
            && (1..=self.len).contains(&strides)
            && self.tolerance.holds(&self.path[strides - 1], &ball);
        if carries_on {
            self.path.drain(..strides);
        } else {
            self.sim.place(ball, &[]);
            self.path.clear();
            self.fresh += 1;
        }
        // The arena sits at the path's last body, so stepping on extends the path.
        while self.path.len() < self.len {
            self.events.clear(0);
            self.sim.step(self.stride, &mut self.events);
            self.path.push_back(self.sim.ball());
        }
        self.path.make_contiguous()
    }

    /// The next `update` predicts afresh (the game was reset or set to a new state).
    pub fn forget(&mut self) {
        self.path.clear();
    }

    /// The current path (empty before the first `update`).
    pub fn path(&mut self) -> &[Body] {
        self.path.make_contiguous()
    }
}

/// Decisions ahead (as `decisions` gives them) in a value that copies: for settings that are copied about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Steps {
    at: [u32; MAX_TIMES],
    len: u8,
}

impl Steps {
    /// At most `MAX_TIMES` of them (more is a programming error: `decisions` refuses them).
    pub fn new(steps: &[u32]) -> Self {
        assert!(
            steps.len() <= MAX_TIMES,
            "at most {MAX_TIMES} prediction times"
        );
        let mut at = [0; MAX_TIMES];
        at[..steps.len()].copy_from_slice(steps);
        Self {
            at,
            len: steps.len() as u8,
        }
    }
}

impl std::ops::Deref for Steps {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        &self.at[..usize::from(self.len)]
    }
}

/// Ball prediction for observations built from states kept elsewhere (recordings, converted replays): one
/// path per stream of consecutive decisions. A stream's next ball that matches its path
/// carries the path on, any other (another game, a cut, a reset) is predicted afresh, so a stream that is not
/// consecutive costs speed, never correctness.
pub struct Recorded {
    tick_skip: u32,
    steps: Vec<u32>,
    paths: Vec<Option<BallPath>>,
}

impl Recorded {
    /// For `steps` (decisions of `tick_skip` ticks ahead; empty = no prediction) over `streams` streams.
    pub fn new(tick_skip: u32, steps: &[u32], streams: usize) -> Self {
        Self {
            tick_skip,
            steps: steps.to_vec(),
            paths: (0..streams.max(1)).map(|_| None).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The predicted bodies for `ball`, the next decision of stream `stream`, into `out` (left empty without
    /// prediction).
    pub fn predict(&mut self, stream: usize, ball: Body, out: &mut Vec<Body>) {
        out.clear();
        let Some(&last) = self.steps.last() else {
            return;
        };
        let slot = stream % self.paths.len();
        let path =
            self.paths[slot].get_or_insert_with(|| BallPath::for_observation(self.tick_skip, last));
        pick(path.update(ball, 1), &self.steps, out);
    }

    /// Stream `stream`'s next ball is predicted afresh.
    pub fn forget(&mut self, stream: usize) {
        let slot = stream % self.paths.len();
        if let Some(path) = &mut self.paths[slot] {
            path.forget();
        }
    }

    /// Paths predicted afresh and updates, over every stream.
    pub fn counts(&self) -> (u64, u64) {
        self.paths
            .iter()
            .flatten()
            .fold((0, 0), |(f, u), p| (f + p.fresh, u + p.updates))
    }
}

/// The predicted bodies the observation reads: the path's body at each of `steps` (decisions ahead, as
/// `decisions` gives them), into `out`.
pub fn pick(path: &[Body], steps: &[u32], out: &mut Vec<Body>) {
    out.clear();
    out.extend(steps.iter().map(|&step| path[step as usize - 1]));
}

/// The packet a prediction is read for: its game time and its ball.
#[derive(Clone, Copy, Debug)]
pub struct Now {
    pub seconds: f32,
    pub ball: Body,
}

/// The observation's predicted bodies from a prediction given as timed slices, as RLBot sends one (one
/// slice a tick, the first a tick after the packet it was made with): for each of `steps`, the slice whose
/// game time (`seconds(slice)`) is within half a tick of `now + step * tick_skip` ticks. Nothing is assumed
/// about where the first slice lies; it is checked against the packet instead:
/// - a prediction older than the packet has a slice for now, which must be the packet's ball within
///   `Tolerance::RLBOT` (else it predates a touch or a reset: stale);
/// - a prediction whose first slice lies up to `MAX_LEAD_TICKS` after the packet was made from the packet's
///   own state: that slice must be the packet's ball moved on at its own velocity, within
///   `LEAD_POS_PER_TICK` a tick;
/// - one starting further ahead does not belong to this packet.
///
/// None when it does not fit, or when a slice needed is missing.
pub fn pick_timed<T>(
    slices: &[T],
    seconds: impl Fn(&T) -> f32,
    body: impl Fn(&T) -> Body,
    now: Now,
    steps: &[u32],
    tick_skip: u32,
    out: &mut Vec<Body>,
) -> Option<()> {
    out.clear();
    let first = seconds(slices.first()?);
    // Slices are one tick apart: the nearest is found from the first's time, then checked.
    let nearest = |target: f32| -> Option<&T> {
        let guess = ((target - first) * TICK_RATE).round();
        let slice = slices.get(usize::try_from(guess as i64).ok()?)?;
        (((seconds(slice) - target) * TICK_RATE).abs() <= 0.5).then_some(slice)
    };
    let lead = (first - now.seconds) * TICK_RATE;
    if lead <= 0.5 {
        if !Tolerance::RLBOT.holds(&body(nearest(now.seconds)?), &now.ball) {
            return None;
        }
    } else if lead <= MAX_LEAD_TICKS + 0.5 {
        let moved_on = now.ball.pos + now.ball.vel * (lead / TICK_RATE);
        if body(&slices[0]).pos.distance(moved_on) > LEAD_POS_PER_TICK * lead {
            return None;
        }
    } else {
        return None;
    }
    for &step in steps {
        out.push(body(nearest(
            now.seconds + (step * tick_skip) as f32 / TICK_RATE,
        )?));
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocketsim::Vec3A;

    fn ball(pos: [f32; 3], vel: [f32; 3], ang_vel: [f32; 3]) -> Body {
        Body {
            pos: Vec3A::from_array(pos),
            vel: Vec3A::from_array(vel),
            ang_vel: Vec3A::from_array(ang_vel),
            ..Body::RESTING
        }
    }

    /// Scripted balls: bouncing on the floor, off a side wall, off the ceiling, into a goal, rolling.
    pub(crate) fn scripted() -> Vec<(&'static str, Body)> {
        vec![
            (
                "bounce",
                ball(
                    [300.0, -800.0, 1200.0],
                    [400.0, 900.0, -300.0],
                    [1.0, -2.0, 0.5],
                ),
            ),
            (
                "wall",
                ball(
                    [3500.0, 0.0, 400.0],
                    [1800.0, 600.0, 200.0],
                    [0.0, 3.0, -1.0],
                ),
            ),
            (
                "ceiling",
                ball(
                    [-500.0, 1500.0, 1700.0],
                    [-300.0, 400.0, 1500.0],
                    [2.0, 0.0, 0.0],
                ),
            ),
            (
                "goal",
                ball(
                    [200.0, 3800.0, 300.0],
                    [-100.0, 2600.0, 250.0],
                    [0.0, 0.0, 1.0],
                ),
            ),
            (
                "roll",
                ball(
                    [-2000.0, -3000.0, 93.15],
                    [700.0, -200.0, 0.0],
                    [2.0, 7.0, 0.0],
                ),
            ),
        ]
    }

    #[test]
    fn times_round_to_whole_decisions_and_bad_ones_are_refused() {
        assert_eq!(decisions(&DEFAULT_TIMES, 8), Ok(vec![8, 15, 30, 45]));
        assert_eq!(
            decisions(&[0.01, 0.5], 8),
            Ok(vec![1, 8]),
            "never below one decision"
        );
        assert_eq!(decisions(&[], 8), Ok(vec![]));
        // At most 719 ticks ahead: 5.9 s (89 decisions, 712 ticks) is the furthest at tick skip 8.
        assert_eq!(decisions(&[5.9], 8), Ok(vec![89]));
        assert!(
            decisions(&[6.0], 8)
                .unwrap_err()
                .contains("beyond the 719 ticks")
        );
        assert!(decisions(&[1.0, 0.5], 8).unwrap_err().contains("increase"));
        assert!(
            decisions(&[0.5, 0.52], 8)
                .unwrap_err()
                .contains("different decisions")
        );
        assert!(decisions(&[0.0], 8).unwrap_err().contains("above 0"));
        assert!(decisions(&[f32::NAN], 8).unwrap_err().contains("above 0"));
        assert!(decisions(&[6.5], 8).unwrap_err().contains("at most 6"));
        assert!(decisions(&[0.1; 9], 8).unwrap_err().contains("at most 8"));
        // 5.99 s at 7 ticks rounds to 103 decisions, 721 ticks: past RLBot's last slice.
        assert!(decisions(&[5.99], 7).unwrap_err().contains("beyond"));
    }

    #[test]
    fn a_tolerance_holds_within_its_margins_only() {
        let a = ball([0.0, 0.0, 100.0], [10.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let t = Tolerance::OBSERVATION;
        let mut b = a;
        b.pos.x += 0.9;
        b.vel.y += 3.4;
        b.ang_vel.z += 0.09;
        assert!(t.holds(&a, &b));
        for change in [
            |b: &mut Body| b.pos.z += 1.1,
            |b: &mut Body| b.vel.x -= 3.6,
            |b: &mut Body| b.ang_vel.y += 0.11,
        ] {
            let mut c = a;
            change(&mut c);
            assert!(!t.holds(&a, &c));
        }
    }

    #[test]
    fn timed_slices_are_picked_at_whole_decisions_ahead() {
        // As RLBot sends them: 720 slices a tick apart, the first a tick after the packet the prediction
        // was made with (seen in a real match: packet 74.5917 s, first slice 74.6000 s). A ball moving
        // 1 uu a tick along x.
        let body = |x: f32| ball([x, 0.0, 100.0], [120.0, 0.0, 0.0], [0.0; 3]);
        let made_at = |tick: f32| -> Vec<(f32, Body)> {
            (1..=720)
                .map(|k| ((tick + k as f32) / TICK_RATE, body(tick + k as f32)))
                .collect()
        };
        let pick = |slices: &[(f32, Body)], tick: f32, steps: &[u32], out: &mut Vec<Body>| {
            let now = Now {
                seconds: tick / TICK_RATE,
                ball: body(tick),
            };
            pick_timed(slices, |s| s.0, |s| s.1, now, steps, 8, out)
        };
        let mut out = Vec::new();
        let fresh = made_at(1200.0);
        assert!(pick(&fresh, 1200.0, &[1, 8, 89, 90], &mut out).is_some());
        // k decisions ahead is the ball 8k ticks on, as training's path[k - 1]; the last slice is 720 ticks
        // on.
        assert_eq!(
            out.iter().map(|b| b.pos.x - 1200.0).collect::<Vec<_>>(),
            [8.0, 64.0, 712.0, 720.0]
        );
        assert!(
            pick(&fresh, 1200.0, &[91], &mut out).is_none(),
            "past the last slice"
        );
        // A prediction a tick older than the packet (made with the packet before): its first slice is now,
        // the packet's ball while nothing touched it. Its slices end a tick sooner.
        let late = made_at(1199.0);
        assert!(pick(&late, 1200.0, &[1, 89], &mut out).is_some());
        assert_eq!((out[0].pos.x, out[1].pos.x), (1208.0, 1912.0));
        assert!(pick(&late, 1200.0, &[90], &mut out).is_none());
        // RLBot's documented order (the first slice now) reads the same.
        let now_first: Vec<(f32, Body)> = (0..720)
            .map(|k| ((1200.0 + k as f32) / TICK_RATE, body(1200.0 + k as f32)))
            .collect();
        assert!(pick(&now_first, 1200.0, &[1, 89], &mut out).is_some());
        assert_eq!(out[0].pos.x, 1208.0);
        // Stale: a late prediction from before a touch (its ball for now elsewhere), or one from a second
        // ago that misses the ball now by 120 uu. A fresh one whose first slice is far from the packet's
        // ball, or that starts more than two ticks ahead, does not belong to it. None at all: nothing.
        let mut touched = made_at(1199.0);
        touched[0].1.vel.x += 300.0;
        assert!(pick(&touched, 1200.0, &[1], &mut out).is_none());
        let old: Vec<(f32, Body)> = (1..=720)
            .map(|k| {
                (
                    (1080.0 + k as f32) / TICK_RATE,
                    body(1080.0 + k as f32 - 120.0),
                )
            })
            .collect();
        assert!(pick(&old, 1200.0, &[1], &mut out).is_none());
        let mut elsewhere = made_at(1200.0);
        elsewhere[0].1.pos.y += 100.0;
        assert!(pick(&elsewhere, 1200.0, &[1], &mut out).is_none());
        assert!(pick(&made_at(1203.0), 1200.0, &[1], &mut out).is_none());
        assert!(pick(&[], 1200.0, &[1], &mut out).is_none());
        // Slices at another rate than one a tick are refused, not misread.
        let sparse: Vec<(f32, Body)> = (0..720)
            .map(|k| {
                (
                    (1201.0 + 2.0 * k as f32) / TICK_RATE,
                    body(1201.0 + 2.0 * k as f32),
                )
            })
            .collect();
        assert!(pick(&sparse, 1200.0, &[8], &mut out).is_none());
    }

    /// A bounce in the tick after the packet bends the path by far less than the lead's margin, so a fresh
    /// prediction of a bouncing ball is not refused.
    #[test]
    fn a_bounce_after_the_packet_keeps_a_fresh_prediction() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let mut worst = 0.0f32;
        for (vz, z) in [(-3000.0, 110.0), (-1500.0, 100.0), (-6000.0, 140.0)] {
            let start = ball([0.0, 0.0, z], [2000.0, 0.0, vz], [0.0; 3]);
            let next = BallPath::new(1, 1, Tolerance::EXACT).update(start, 1)[0];
            let moved_on = start.pos + start.vel / TICK_RATE;
            worst = worst.max(next.pos.distance(moved_on));
        }
        eprintln!("a bounce bent the next tick's position by {worst:.1} uu");
        assert!(worst < LEAD_POS_PER_TICK);
    }

    #[test]
    fn a_carried_on_path_matches_a_fresh_prediction() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let steps = decisions(&DEFAULT_TIMES, 8).unwrap();
        for (name, start) in scripted() {
            // The real ball: a ball-only arena too (no car touches it here), stepped a decision at a time.
            let mut real = BallPath::new(8, 1, Tolerance::EXACT);
            let mut carried = BallPath::for_observation(8, 45);
            let mut ball = start;
            for decision in 0..120 {
                let path = carried.update(ball, 1).to_vec();
                let fresh = BallPath::for_observation(8, 45).update(ball, 1).to_vec();
                for &step in &steps {
                    let (a, b) = (path[step as usize - 1], fresh[step as usize - 1]);
                    // Measured: 0 in flight, up to 0.01 uu for a rolling ball, whose contact with the floor
                    // a fresh arena has not cached yet.
                    assert!(
                        a.pos.distance(b.pos) < 0.1 && a.vel.distance(b.vel) < 0.1,
                        "{name}, decision {decision}, {step} ahead: {a:?} vs {b:?}"
                    );
                }
                ball = real.update(ball, 1)[0];
                real.forget();
            }
            // 120 decisions are 8 s, predicted afresh only at the start.
            assert_eq!(carried.fresh, 1, "{name}");
        }
    }

    #[test]
    fn a_touched_or_reset_ball_is_predicted_afresh() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let (_, start) = scripted()[0];
        let mut path = BallPath::for_observation(8, 45);
        let next = path.update(start, 1)[0];
        assert_eq!(path.fresh, 1);
        path.update(next, 1);
        assert_eq!(path.fresh, 1, "carried on");
        // A touch: the ball is 50 uu/s faster than predicted.
        let mut touched = path.path()[0];
        touched.vel.x += 50.0;
        let after = path.update(touched, 1).to_vec();
        assert_eq!(path.fresh, 2, "a touch predicts afresh");
        let fresh = BallPath::for_observation(8, 45).update(touched, 1).to_vec();
        assert_eq!(after, fresh);
        // A reset to a state the path happens to hold still predicts afresh.
        let same = path.path()[0];
        path.forget();
        path.update(same, 1);
        assert_eq!(path.fresh, 3);
        // Two decisions at once (a skipped call) carry on two strides.
        let two = path.path()[1];
        path.update(two, 2);
        assert_eq!(path.fresh, 3);
    }

    /// How far the far end moves when a ball is accepted within `Tolerance::OBSERVATION` of the path (the
    /// worst of each margin used fully), over the scripted balls.
    #[test]
    fn a_ball_within_tolerance_lands_where_the_path_says() {
        if !crate::testing::meshes_loaded() {
            return;
        }
        let t = Tolerance::OBSERVATION;
        let mut worst = 0.0f32;
        for (name, start) in scripted() {
            let exact = BallPath::for_observation(8, 45).update(start, 1).to_vec();
            for shift in [
                Vec3A::new(t.pos, 0.0, 0.0),
                Vec3A::new(0.0, 0.0, -t.pos),
                Vec3A::ZERO,
            ] {
                let mut nudged = start;
                nudged.pos += shift;
                nudged.vel += Vec3A::new(0.0, t.vel, 0.0) * f32::from(shift == Vec3A::ZERO);
                nudged.ang_vel.z += t.ang_vel;
                let path = BallPath::for_observation(8, 45).update(nudged, 1).to_vec();
                let far = exact[44].pos.distance(path[44].pos);
                eprintln!("{name}: far end moved {far:.2} uu");
                worst = worst.max(far);
            }
        }
        // Measured 8.8 uu at most (the bouncing ball).
        assert!(worst < 20.0, "the 3 s end moved {worst} uu");
    }
}
