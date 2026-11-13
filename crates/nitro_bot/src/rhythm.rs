//! The decision rhythm: decide every `tick_skip` physics ticks, apply each decision `action_delay` ticks
//! later, hold the controls in between, and answer every packet.
//!
//! The game uses a sent control in its next physics step, with no extra latency (measured on three
//! matches), so `latency` is 0 by default; a game with latency gets each
//! decision sent that many ticks early, and the car acts exactly `action_delay` ticks after the
//! observation, as in training.
//!
//! Ticks are counted with the packet's `frame_num`, not by counting packets, so dropped or duplicated
//! packets never shift the rhythm: a decision falls due at its tick and is taken at the first packet at or
//! after it. If packets skip a whole decision, that decision is lost (counted in `stats.missed`), as
//! nothing can be decided for a state the bot never saw.

use nitro_core::snapshot::Controls;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RhythmStats {
    /// Decisions taken.
    pub decisions: u64,
    /// Decisions taken later than their due tick (a dropped packet).
    pub late: u64,
    /// Decisions that could not be taken at all (packets skipped past them).
    pub missed: u64,
}

#[derive(Clone, Debug)]
pub struct Rhythm {
    tick_skip: u32,
    /// Ticks after a decision at which its controls are sent: `action_delay` minus the game's latency.
    apply_after: u32,
    /// Tick of the next decision; None until the first movable packet after a reset.
    next_due: Option<u32>,
    /// Controls decided but not yet in effect (they take effect at this tick).
    pending: Option<(u32, Controls)>,
    /// A decision begun (`begin`) whose controls are not yet supplied; they take effect at this tick.
    awaited: Option<u32>,
    /// What the car is being told to do right now.
    current: Controls,
    /// The last decision, which the next observation reports as the previous action.
    last_decided: Controls,
    pub stats: RhythmStats,
}

impl Rhythm {
    /// A rhythm for a game that applies controls the moment they are sent (latency 0).
    pub fn new(tick_skip: u32, action_delay: u32) -> Self {
        Self::with_latency(tick_skip, action_delay, 0)
    }

    /// `latency`: ticks the game takes to apply a sent control; the decision is sent that many ticks
    /// early (never before the decision itself).
    pub fn with_latency(tick_skip: u32, action_delay: u32, latency: u32) -> Self {
        assert!(tick_skip > 0 && action_delay < tick_skip);
        Self {
            tick_skip,
            apply_after: action_delay.saturating_sub(latency),
            next_due: None,
            pending: None,
            awaited: None,
            current: Controls::default(),
            last_decided: Controls::default(),
            stats: RhythmStats::default(),
        }
    }

    /// Forgets everything but the statistics: the next movable packet starts a fresh rhythm with neutral
    /// controls, exactly like a training episode's first step (kickoff, after a goal, after a replay).
    pub fn reset(&mut self) {
        self.next_due = None;
        self.pending = None;
        self.awaited = None;
        self.current = Controls::default();
        self.last_decided = Controls::default();
    }

    /// Handles the packet of physics tick `frame`: takes a decision if one is due (`decide` gets the
    /// previous decision, as the observation's "previous action") and returns the controls to send.
    pub fn step(&mut self, frame: u32, decide: impl FnOnce(Controls) -> Controls) -> Controls {
        if let Some(previous) = self.begin(frame) {
            self.supply(decide(previous));
        }
        self.current(frame)
    }

    /// `step` in parts, for a decision taken off the packet's thread: `begin` says whether a
    /// decision is due at `frame` (with the previous decision); its controls go to `supply` once known,
    /// which must happen before any packet that `needs` them (`frame + action_delay` on) and before the
    /// next `begin`. `current` gives the controls to send.
    pub fn begin(&mut self, frame: u32) -> Option<Controls> {
        self.apply(frame);
        let due = *self.next_due.get_or_insert(frame);
        if frame < due {
            return None;
        }
        debug_assert!(
            self.awaited.is_none(),
            "the last decision was never supplied"
        );
        self.stats.decisions += 1;
        // A decision not yet in effect (late packets) is replaced by this one, as `step` always did.
        self.pending = None;
        self.awaited = Some(frame + self.apply_after);
        if frame > due {
            self.stats.late += 1;
        }
        let mut next = due + self.tick_skip;
        while next <= frame {
            next += self.tick_skip;
            self.stats.missed += 1;
        }
        self.next_due = Some(next);
        Some(self.last_decided)
    }

    /// Whether the packet of `frame` must wait for the decision begun last: its controls are in effect by
    /// then, or the next decision is due (which reads it as the previous action).
    pub fn needs(&self, frame: u32) -> bool {
        self.awaited
            .is_some_and(|at| frame >= at || self.next_due.is_some_and(|due| frame >= due))
    }

    /// Whether a decision is begun and not yet supplied.
    pub fn awaiting(&self) -> bool {
        self.awaited.is_some()
    }

    /// The controls of the decision begun last.
    pub fn supply(&mut self, chosen: Controls) {
        let at = self.awaited.take().expect("a decision was begun");
        self.last_decided = chosen;
        self.pending = Some((at, chosen));
    }

    /// The controls to send for the packet of `frame` (after any decision it `needs` is supplied).
    pub fn current(&mut self, frame: u32) -> Controls {
        debug_assert!(
            !self.needs(frame),
            "a decision in effect now was never supplied"
        );
        self.apply(frame);
        self.current
    }

    fn apply(&mut self, frame: u32) {
        if let Some((at, controls)) = self.pending
            && frame >= at
        {
            self.current = controls;
            self.pending = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controls(throttle: f32) -> Controls {
        Controls {
            throttle,
            ..Controls::default()
        }
    }

    /// Plays `frames` through the rhythm; every decision returns the frame it was taken at as throttle.
    fn play(rhythm: &mut Rhythm, frames: &[u32]) -> Vec<(u32, f32, f32)> {
        frames
            .iter()
            .map(|&frame| {
                let mut prev = f32::NAN;
                let sent = rhythm.step(frame, |previous| {
                    prev = previous.throttle;
                    controls(frame as f32)
                });
                (frame, sent.throttle, prev)
            })
            .collect()
    }

    #[test]
    fn decides_every_tick_skip_and_applies_after_the_delay() {
        let mut rhythm = Rhythm::new(8, 7);
        let frames: Vec<u32> = (100..125).collect();
        let played = play(&mut rhythm, &frames);
        // Decisions at 100, 108, 116, 124; each in effect from 7 ticks later.
        let sent: Vec<f32> = played.iter().map(|p| p.1).collect();
        assert_eq!(&sent[..7], &[0.0; 7]); // frames 100..106: nothing in effect yet
        assert!(sent[7..15].iter().all(|&t| t == 100.0)); // 107..114: decision of frame 100
        assert!(sent[15..23].iter().all(|&t| t == 108.0));
        assert!(sent[23..].iter().all(|&t| t == 116.0));
        // The observation's previous action is the previous decision.
        let prevs: Vec<f32> = played
            .iter()
            .filter(|p| !p.2.is_nan())
            .map(|p| p.2)
            .collect();
        assert_eq!(prevs, vec![0.0, 100.0, 108.0, 116.0]);
        assert_eq!(
            rhythm.stats,
            RhythmStats {
                decisions: 4,
                late: 0,
                missed: 0
            }
        );
    }

    #[test]
    fn the_games_latency_is_taken_off_the_delay() {
        // Training applies the action 7 ticks after the observation; the game adds 1 tick, so send at 6.
        let mut rhythm = Rhythm::with_latency(8, 7, 1);
        let sent: Vec<f32> = play(&mut rhythm, &(100..116).collect::<Vec<_>>())
            .iter()
            .map(|p| p.1)
            .collect();
        assert_eq!(&sent[..6], &[0.0; 6]);
        assert!(sent[6..14].iter().all(|&t| t == 100.0), "{sent:?}");
        assert!(sent[14..].iter().all(|&t| t == 108.0));
        // A latency longer than the delay cannot be taken off: the controls go out with the decision.
        let mut rhythm = Rhythm::with_latency(8, 1, 3);
        let sent: Vec<f32> = play(&mut rhythm, &[0, 1, 2]).iter().map(|p| p.1).collect();
        assert_eq!(sent, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn no_delay_applies_at_once() {
        let mut rhythm = Rhythm::new(4, 0);
        let played = play(&mut rhythm, &[0, 1, 2, 3, 4, 5]);
        let sent: Vec<f32> = played.iter().map(|p| p.1).collect();
        assert_eq!(sent, vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0]);
    }

    #[test]
    fn dropped_and_duplicated_packets_keep_the_rhythm() {
        let mut rhythm = Rhythm::new(8, 1);
        // Frames 8 and 9 never arrive: the decision due at 8 is taken at 10 (late); 10 arrives twice.
        let played = play(&mut rhythm, &[0, 1, 2, 10, 10, 11, 16, 17]);
        let decided: Vec<u32> = played
            .iter()
            .filter(|p| !p.2.is_nan())
            .map(|p| p.0)
            .collect();
        assert_eq!(decided, vec![0, 10, 16]);
        assert_eq!(
            rhythm.stats,
            RhythmStats {
                decisions: 3,
                late: 1,
                missed: 0
            }
        );
        // Packets jump over a whole decision: 40 is due, next packet is 60 -> decisions at 40 and 56 lost.
        let mut rhythm = Rhythm::new(8, 1);
        play(&mut rhythm, &[32, 60]);
        assert_eq!(
            rhythm.stats,
            RhythmStats {
                decisions: 2,
                late: 1,
                missed: 2
            }
        );
    }

    #[test]
    fn a_decision_supplied_later_sends_what_step_sends() {
        // Packets with drops, duplicates and a skipped decision; each decision is supplied at the last
        // moment (the first packet that needs it), as the bot does when deciding off the packet's thread.
        let frames = [
            0, 1, 2, 3, 4, 5, 7, 8, 8, 9, 10, 15, 16, 17, 30, 31, 32, 33, 34,
        ];
        for (tick_skip, delay) in [(4, 1), (4, 0), (4, 3), (8, 7)] {
            let mut sync = Rhythm::new(tick_skip, delay);
            let expected = play(&mut sync, &frames);
            let mut split = Rhythm::new(tick_skip, delay);
            let mut waiting: Option<(u32, f32)> = None;
            for (&frame, &(_, sent, prev)) in frames.iter().zip(&expected) {
                if split.needs(frame) {
                    split.supply(controls(waiting.take().unwrap().0 as f32));
                }
                let mut seen = f32::NAN;
                if let Some(previous) = split.begin(frame) {
                    seen = previous.throttle;
                    waiting = Some((frame, previous.throttle));
                    if split.needs(frame) {
                        split.supply(controls(waiting.take().unwrap().0 as f32));
                    }
                }
                assert_eq!(
                    split.current(frame).throttle,
                    sent,
                    "{tick_skip}/{delay} frame {frame}"
                );
                assert_eq!(
                    seen.to_bits(),
                    prev.to_bits(),
                    "{tick_skip}/{delay} frame {frame}"
                );
                assert_eq!(split.awaiting(), waiting.is_some());
            }
            assert_eq!(split.stats, sync.stats);
        }
    }

    #[test]
    fn reset_starts_afresh_with_neutral_controls() {
        let mut rhythm = Rhythm::new(8, 7);
        play(&mut rhythm, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(rhythm.step(9, |_| controls(9.0)).throttle, 0.0);
        rhythm.reset();
        // A new kickoff at frame 500: the first packet decides, nothing old is in effect or remembered.
        let played = play(&mut rhythm, &[500, 501]);
        assert_eq!(played[0].1, 0.0);
        assert_eq!(played[0].2, 0.0, "previous action after a reset is neutral");
        assert_eq!(rhythm.stats.decisions, 3);
    }
}
