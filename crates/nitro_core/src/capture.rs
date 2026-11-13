//! Frames of one arena every few physics ticks, for the match viewer. Off unless a caller starts it; the
//! training path pays one branch per simulated chunk.
//!
//! A frame is `HEAD + cars * CAR` numbers (layout `FORMAT`, the one the viewer reads).
//! Head: a marker; the ball's position, velocity, forward and up directions and spin (rad/s); the score
//! (Blue, Orange) since the capture began; then each boost pad's wait in seconds (0 while it is available).
//! Per car: its team, position, velocity, forward and up directions, spin, boost, flags (`FLAG_*`), the eight
//! controls it is using (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), state bits (`STATE_*`),
//! the smoothed handbrake (0 to 1), its body type (`BODY_OCTANE`: the only one simulated) and the seconds
//! until a demolished car is back.

use crate::sim::Sim;
use crate::snapshot::{Controls, GameSnapshot, NUM_BOOST_PADS, StepEvents, Team};

/// The frame layout's version, written into every recording.
pub const FORMAT: u32 = 2;
pub const HEAD: usize = 18 + NUM_BOOST_PADS;
pub const CAR: usize = 30;
/// Where the score and the first pad's wait sit in the head.
pub const SCORE_AT: usize = 16;
pub const PADS_AT: usize = 18;
/// Markers: an ordinary frame, the first frame of an episode (the viewer does not blend across it), and
/// the first frame after a goal by Blue or by Orange.
pub const PLAYING: f32 = 0.0;
pub const RESTART: f32 = 1.0;
pub const BLUE_GOAL: f32 = 2.0;
pub const ORANGE_GOAL: f32 = 3.0;
pub const FLAG_ON_GROUND: u32 = 1;
pub const FLAG_DEMOED: u32 = 2;
pub const FLAG_SUPERSONIC: u32 = 4;
pub const FLAG_JUMPED: u32 = 8;
pub const FLAG_DOUBLE_JUMPED: u32 = 16;
pub const FLAG_FLIPPED: u32 = 32;
/// The car touched the ball since the previous frame.
pub const FLAG_TOUCHED: u32 = 64;
/// The mechanics the car did in the step that ended at this frame (`mechanics::MARK_*`), from this bit
/// up (eight bits: the flags still fit the viewer's 16-bit column).
pub const FLAG_MARKS_SHIFT: u32 = 7;
/// Where a car's flags sit in its block.
const FLAGS_AT: usize = 17;
/// State bits: the wheel with this index touches something (front left, front right, back left, back
/// right: bits 0 to 3), the boost is firing, a jump is under way, a flip is under way.
pub const STATE_WHEEL: u32 = 1;
pub const STATE_BOOSTING: u32 = 16;
pub const STATE_JUMPING: u32 = 32;
pub const STATE_FLIPPING: u32 = 64;
pub const BODY_OCTANE: f32 = 0.0;

pub struct Capture {
    every: u32,
    ticks: u32,
    scratch: GameSnapshot,
    details: StepEvents,
    /// Goals since the capture began: Blue, Orange.
    score: [f32; 2],
    /// Per car: a touch already seen in this step's (latched) events, and a touch not yet written.
    seen: Vec<bool>,
    unwritten: Vec<bool>,
    goal_written: bool,
    pub frames: Vec<f32>,
}

impl Capture {
    /// Captures every `every` ticks (at least 1), starting with the arena's current state.
    pub fn new(every: u32, sim: &mut Sim, controls: &[Controls]) -> Self {
        let cars = sim.num_cars();
        let mut capture = Self {
            every: every.max(1),
            ticks: 0,
            scratch: GameSnapshot::default(),
            details: StepEvents::default(),
            score: [0.0; 2],
            seen: vec![false; cars],
            unwritten: vec![false; cars],
            goal_written: false,
            frames: Vec::new(),
        };
        capture.restart(sim, controls);
        capture
    }

    /// An episode starts: its first state is written at once, marked so the viewer jumps to it.
    pub fn restart(&mut self, sim: &mut Sim, controls: &[Controls]) {
        self.ticks = 0;
        self.goal_written = false;
        self.seen.fill(false);
        self.unwritten.fill(false);
        self.write(sim, controls, RESTART);
    }

    /// A step starts: the step's events were just cleared, so no touch has been seen in it yet.
    pub fn step_started(&mut self) {
        self.seen.fill(false);
    }

    /// After every simulated tick; writes a frame every `every` ticks.
    pub fn after_tick(&mut self, sim: &mut Sim, events: &StepEvents, controls: &[Controls]) {
        for (car, event) in events.cars.iter().enumerate() {
            // Events are latched for the whole step and cleared at its start.
            if event.touched_ball && !self.seen[car] {
                self.unwritten[car] = true;
            }
            self.seen[car] = event.touched_ball;
        }
        self.ticks += 1;
        if self.ticks.is_multiple_of(self.every) {
            self.write(sim, controls, PLAYING);
        }
    }

    fn write(&mut self, sim: &mut Sim, controls: &[Controls], marker: f32) {
        let mut snap = std::mem::take(&mut self.scratch);
        sim.read(&mut snap);
        let marker = match snap.goal_scored {
            Some(team) if marker == PLAYING && !self.goal_written => {
                self.goal_written = true;
                if team == Team::Blue {
                    self.score[0] += 1.0;
                    BLUE_GOAL
                } else {
                    self.score[1] += 1.0;
                    ORANGE_GOAL
                }
            }
            _ => marker,
        };
        sim.details(&mut self.details);
        self.frames.push(marker);
        self.frames.extend(snap.ball.pos.to_array());
        self.frames.extend(snap.ball.vel.to_array());
        self.frames.extend(snap.ball.forward().to_array());
        self.frames.extend(snap.ball.up().to_array());
        self.frames.extend(snap.ball.ang_vel.to_array());
        self.frames.extend(self.score);
        self.frames.extend(snap.pads.iter().map(|pad| {
            if pad.active {
                0.0
            } else {
                // A pad taken this very tick may still show no wait: it reads as taken either way.
                pad.cooldown.max(0.01)
            }
        }));
        for (car, state) in snap.cars.iter().enumerate() {
            let flags = [
                (state.on_ground, FLAG_ON_GROUND),
                (state.is_demoed, FLAG_DEMOED),
                (state.is_supersonic, FLAG_SUPERSONIC),
                (state.has_jumped, FLAG_JUMPED),
                (state.has_double_jumped, FLAG_DOUBLE_JUMPED),
                (state.has_flipped, FLAG_FLIPPED),
                (std::mem::take(&mut self.unwritten[car]), FLAG_TOUCHED),
            ]
            .iter()
            .filter(|(on, _)| *on)
            .fold(0, |all, (_, flag)| all | flag);
            let c = controls[car];
            let detail = &self.details.details[car];
            let bits = detail
                .wheel_contacts
                .iter()
                .enumerate()
                .filter(|(_, touching)| **touching)
                .fold(0, |all, (wheel, _)| all | STATE_WHEEL << wheel)
                | if detail.is_boosting {
                    STATE_BOOSTING
                } else {
                    0
                }
                | if detail.is_jumping { STATE_JUMPING } else { 0 }
                | if detail.is_flipping {
                    STATE_FLIPPING
                } else {
                    0
                };
            self.frames.push(state.team.index() as f32);
            self.frames.extend(state.body.pos.to_array());
            self.frames.extend(state.body.vel.to_array());
            self.frames.extend(state.body.forward().to_array());
            self.frames.extend(state.body.up().to_array());
            self.frames.extend(state.body.ang_vel.to_array());
            self.frames.extend([
                state.boost,
                flags as f32,
                c.throttle,
                c.steer,
                c.pitch,
                c.yaw,
                c.roll,
                f32::from(u8::from(c.jump)),
                f32::from(u8::from(c.boost)),
                f32::from(u8::from(c.handbrake)),
                bits as f32,
                detail.handbrake,
                BODY_OCTANE,
                state.demo_timer,
            ]);
        }
        self.scratch = snap;
    }

    /// The step that just ended: its last frame gets each car's mechanics (`mechanics::MARK_*`).
    pub fn mark(&mut self, marks: &[u32]) {
        mark_last_frame(&mut self.frames, marks);
    }
}

fn mark_last_frame(frames: &mut [f32], marks: &[u32]) {
    let width = HEAD + marks.len() * CAR;
    if frames.len() < width || marks.iter().all(|&m| m == 0) {
        return;
    }
    let start = frames.len() - width;
    for (car, &bits) in marks.iter().enumerate() {
        let flags = &mut frames[start + HEAD + car * CAR + FLAGS_AT];
        *flags = (*flags as u32 | bits << FLAG_MARKS_SHIFT) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mechanics::{MARK_DEMO, MARK_FLICK};

    #[test]
    fn marks_go_into_the_last_frames_flags_and_leave_the_rest() {
        let cars = 2;
        let width = HEAD + cars * CAR;
        let mut frames = vec![0.0f32; 2 * width];
        frames[width + HEAD + FLAGS_AT] = (FLAG_ON_GROUND | FLAG_TOUCHED) as f32;
        mark_last_frame(&mut frames, &[MARK_FLICK, MARK_DEMO]);
        let flags =
            |frame: usize, car: usize| frames[frame * width + HEAD + car * CAR + FLAGS_AT] as u32;
        assert_eq!(
            flags(1, 0),
            FLAG_ON_GROUND | FLAG_TOUCHED | MARK_FLICK << FLAG_MARKS_SHIFT
        );
        assert_eq!(flags(1, 1), MARK_DEMO << FLAG_MARKS_SHIFT);
        assert_eq!(
            (flags(0, 0), flags(0, 1)),
            (0, 0),
            "earlier frames are left alone"
        );
        let all = 0b111_1111 | 0xff << FLAG_MARKS_SHIFT;
        assert!(
            all <= i16::MAX as u32,
            "every flag fits the viewer's 16-bit column"
        );
        let before = frames.clone();
        mark_last_frame(&mut frames[..width - 1], &[MARK_FLICK, 0]);
        assert_eq!(frames, before, "no whole frame, nothing marked");
    }
}
