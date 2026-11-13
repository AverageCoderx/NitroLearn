//! The network pass off the packets' threads. A car's thread hands its observation to the process's
//! one decider and answers the packet at once; the decision is needed only `action_delay` ticks later
//! (`Rhythm::needs`), so the network has that long, and every packet gets its controls on time (RLBot's
//! monitor counts a packet without an input as missed). The decider runs the observations that arrive
//! together, the cars of a hivemind deciding at the same tick, in one pass (`Policy::logits_batch`): the
//! weights are read once for all of them, and each car gets the bits it gets alone.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nitro_core::policy::Policy;
use nitro_core::team::Team;

/// How long the decider waits for the other cars' observations after the first one arrives. The cars of
/// a hivemind get each packet within microseconds of each other, and a car that is not deciding (one
/// seat only, or it coasts) costs at most this.
const GATHER: Duration = Duration::from_micros(500);

struct Job {
    obs: Vec<f32>,
    /// The car blocks (teammates, opponents) of an observation that is not the trained size.
    cars: Option<(usize, usize)>,
    answer: Sender<Answer>,
}

/// A decision's logits, and when the pass that made them ended.
pub struct Answer {
    pub logits: Vec<f32>,
    pub finished: Instant,
    /// Observations in that pass.
    pub batch: usize,
}

/// The process's decider: one thread running the policy for every car seated at it.
pub struct Decider {
    jobs: Mutex<Sender<Job>>,
    seated: Arc<AtomicUsize>,
}

impl Decider {
    /// Starts the decider's thread; each pass is split across `threads` threads (`Team`).
    pub fn start(policy: Arc<Policy>, threads: usize) -> Arc<Self> {
        let (jobs, inbox) = mpsc::channel();
        let seated = Arc::new(AtomicUsize::new(0));
        let counted = seated.clone();
        std::thread::Builder::new()
            .name("nitro_bot decider".into())
            .spawn(move || serve(&policy, threads.max(1), &inbox, &counted))
            .expect("the decider's thread starts");
        Arc::new(Self {
            jobs: Mutex::new(jobs),
            seated,
        })
    }

    /// A place for one car.
    pub fn seat(self: &Arc<Self>) -> Seat {
        self.seated.fetch_add(1, Ordering::AcqRel);
        Seat {
            decider: self.clone(),
            waiting: None,
        }
    }
}

/// The decider's loop: the first job, then any other seat's that arrive within `GATHER`, in one pass.
fn serve(policy: &Policy, threads: usize, inbox: &Receiver<Job>, seated: &AtomicUsize) {
    let actions = policy.num_actions();
    while let Ok(first) = inbox.recv() {
        let mut batch = vec![first];
        let until = Instant::now() + GATHER;
        while batch.len() < seated.load(Ordering::Acquire) {
            match inbox.recv_timeout(until.saturating_duration_since(Instant::now())) {
                Ok(job) => batch.push(job),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }
        let trained: Vec<&Job> = batch.iter().filter(|job| job.cars.is_none()).collect();
        let obs: Vec<f32> = trained
            .iter()
            .flat_map(|job| job.obs.iter().copied())
            .collect();
        let mut logits = vec![0.0f32; trained.len() * actions];
        let mut other = Vec::new();
        Team::run(threads, |team| {
            if !trained.is_empty() {
                policy.logits_batch(&obs, trained.len(), &mut logits, team);
            }
            for job in batch.iter().filter(|job| job.cars.is_some()) {
                let (teammates, opponents) = job.cars.unwrap_or_default();
                let mut out = vec![0.0f32; actions];
                policy.logits_cars_team(&job.obs, teammates, opponents, &mut out, team);
                other.push(out);
            }
        });
        let finished = Instant::now();
        let size = batch.len();
        let mut trained_logits = logits.chunks_exact(actions);
        let mut other = other.into_iter();
        for job in &batch {
            let logits = if job.cars.is_none() {
                trained_logits.next().map(<[f32]>::to_vec)
            } else {
                other.next()
            };
            let answer = Answer {
                logits: logits.expect("one answer per job"),
                finished,
                batch: size,
            };
            // A car that left (its match ended) or forgot the decision no longer listens.
            let _ = job.answer.send(answer);
        }
    }
}

/// One car's place at the decider: at most one decision in flight.
pub struct Seat {
    decider: Arc<Decider>,
    /// Where the decision in flight will be answered, and when it was sent.
    waiting: Option<(Receiver<Answer>, Instant)>,
}

impl Seat {
    /// Hands an observation to the decider (`cars`: its car blocks when not the trained size).
    pub fn send(&mut self, obs: Vec<f32>, cars: Option<(usize, usize)>) {
        // A channel for this decision alone: a forgotten one's answer goes nowhere, and if the decider's
        // thread dies, the job's end of it goes too and `wait` returns.
        let (answer, answers) = mpsc::channel();
        self.waiting = Some((answers, Instant::now()));
        let job = Job { obs, cars, answer };
        let jobs = self.decider.jobs.lock().unwrap_or_else(|p| p.into_inner());
        // The decider's thread only ends with a panic; `wait` then reports no answer.
        let _ = jobs.send(job);
    }

    /// Whether a decision is in flight.
    pub fn busy(&self) -> bool {
        self.waiting.is_some()
    }

    /// Drops the decision in flight (the rhythm restarted); its answer goes nowhere.
    pub fn forget(&mut self) {
        self.waiting = None;
    }

    /// Waits for the decision in flight: its answer and how long it took from `send`, and whether it had
    /// to wait (the answer was not there yet). None if no decision is in flight or the decider stopped.
    pub fn wait(&mut self) -> Option<(Answer, Duration, bool)> {
        let (answers, sent) = self.waiting.take()?;
        let (answer, waited) = match answers.try_recv() {
            Ok(answer) => (answer, false),
            Err(_) => (answers.recv().ok()?, true),
        };
        let took = answer.finished.saturating_duration_since(sent);
        Some((answer, took, waited))
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        self.decider.seated.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A policy of two linear layers (`obs_size` -> 7 -> 3 actions, ReLU between), random weights.
    fn policy(obs_size: usize) -> Policy {
        use nitro_core::policy::{MAGIC, PLAIN_FORMAT};
        let header = format!(
            r#"{{"format": 1, "game": {{"observation": "native", "action_masks": "standard", "team_size": 1,
            "opponents": true, "tick_skip": 8, "action_delay": 7}}, "obs_size": {obs_size},
            "num_actions": 3, "layers": [{{"kind": "linear", "inputs": {obs_size}, "outputs": 7}},
            {{"kind": "relu"}}, {{"kind": "linear", "inputs": 7, "outputs": 3}}]}}"#
        );
        let mut bytes = MAGIC.to_vec();
        bytes.extend(PLAIN_FORMAT.to_le_bytes());
        bytes.extend((header.len() as u32).to_le_bytes());
        bytes.extend(header.as_bytes());
        let mut rng = fastrand::Rng::with_seed(5);
        for _ in 0..obs_size * 7 + 7 + 7 * 3 + 3 {
            bytes.extend((rng.f32() - 0.5).to_le_bytes());
        }
        Policy::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn every_seat_gets_its_own_bits_and_a_forgotten_answer_is_dropped() {
        let size = 12;
        let policy = Arc::new(policy(size));
        let decider = Decider::start(policy.clone(), 2);
        let mut seats: Vec<Seat> = (0..3).map(|_| decider.seat()).collect();
        let mut rng = fastrand::Rng::with_seed(3);
        let mut obs = || -> Vec<f32> { (0..size).map(|_| rng.f32() * 2.0 - 1.0).collect() };
        let alone = |o: &[f32]| {
            let mut out = vec![0.0f32; 3];
            policy.logits(o, &mut out);
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        };
        let mut shared = 0;
        for _ in 0..5 {
            let all: Vec<Vec<f32>> = seats.iter().map(|_| obs()).collect();
            for (seat, o) in seats.iter_mut().zip(&all) {
                seat.send(o.clone(), None);
                assert!(seat.busy());
            }
            for (seat, o) in seats.iter_mut().zip(&all) {
                let (answer, _, _) = seat.wait().unwrap();
                let bits: Vec<u32> = answer.logits.iter().map(|x| x.to_bits()).collect();
                assert_eq!(bits, alone(o));
                assert!((1..=3).contains(&answer.batch));
                shared += usize::from(answer.batch > 1);
                assert!(!seat.busy());
            }
        }
        // Sent within microseconds of each other, the seats' decisions share passes.
        assert!(shared > 0);
        // A decision forgotten (the rhythm restarted): the next answer is the new decision's.
        let (old, new) = (obs(), obs());
        seats[0].send(old, None);
        seats[0].forget();
        assert!(seats[0].wait().is_none());
        seats[0].send(new.clone(), None);
        let (answer, _, _) = seats[0].wait().unwrap();
        let bits: Vec<u32> = answer.logits.iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits, alone(&new));
        drop(seats);
        assert_eq!(decider.seated.load(Ordering::Acquire), 0);
    }
}
