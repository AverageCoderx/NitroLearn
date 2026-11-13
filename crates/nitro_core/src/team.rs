//! A few threads sharing one job at a time, for the bot's network passes (the cars of a match deciding in
//! one pass, split across threads). `Team::run` starts the members for the length of a closure; inside it,
//! `split` gives every member its share of a job and returns once all are done. Members spin between jobs,
//! so a job starts within a microsecond.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

type Job = &'static (dyn Fn(usize) + Sync);

pub struct Team<'a> {
    members: usize,
    shared: Option<&'a Shared>,
}

struct Shared {
    /// Bumped once per job; a member starts the job when it changes.
    generation: AtomicUsize,
    /// Members other than the caller still on the current job.
    busy: AtomicUsize,
    job: Mutex<Option<Job>>,
    panicked: AtomicBool,
    stop: AtomicBool,
}

impl Team<'_> {
    /// One member: `split` runs the job on the calling thread.
    pub const ALONE: Team<'static> = Team {
        members: 1,
        shared: None,
    };

    /// Runs `body` with a team of `members` threads (the calling one included).
    pub fn run<R>(members: usize, body: impl FnOnce(&Team<'_>) -> R) -> R {
        if members <= 1 {
            return body(&Team::ALONE);
        }
        let shared = Shared {
            generation: AtomicUsize::new(0),
            busy: AtomicUsize::new(0),
            job: Mutex::new(None),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        };
        std::thread::scope(|scope| {
            for member in 1..members {
                let shared = &shared;
                scope.spawn(move || shared.serve(member));
            }
            // Members stop however `body` ends, so the scope can join them.
            struct Stop<'s>(&'s Shared);
            impl Drop for Stop<'_> {
                fn drop(&mut self) {
                    self.0.stop.store(true, Ordering::Release);
                }
            }
            let _stop = Stop(&shared);
            body(&Team {
                members,
                shared: Some(&shared),
            })
        })
    }

    pub fn members(&self) -> usize {
        self.members
    }

    /// Runs `work(index, item)` once for every item, the items shared out among the members.
    pub fn each_mut<T: Send>(&self, items: &mut [T], work: impl Fn(usize, &mut T) + Sync) {
        let (base, count) = (Items(items.as_mut_ptr()), items.len());
        self.split(&|member| {
            // The whole wrapper, not its pointer field alone, so the closure is Sync.
            #[allow(clippy::redundant_locals)]
            let base = base;
            for index in share(count, 1, member, self.members) {
                // SAFETY: every index lies in one member's share alone, inside `items`.
                work(index, unsafe { &mut *base.0.add(index) });
            }
        });
    }

    /// Runs `job(member)` once for every member, `0..members`, at the same time; returns when all are done.
    pub fn split(&self, job: &(dyn Fn(usize) + Sync)) {
        let Some(shared) = self.shared else {
            job(0);
            return;
        };
        // SAFETY: the job is cleared below before this call returns, and no member reads it after
        // `busy` reaches zero, so it never outlives this borrow.
        let job_static: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        *shared.job.lock().unwrap_or_else(|p| p.into_inner()) = Some(job_static);
        shared.busy.store(self.members - 1, Ordering::Release);
        shared.generation.fetch_add(1, Ordering::AcqRel);
        let own = catch_unwind(AssertUnwindSafe(|| job(0)));
        while shared.busy.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        *shared.job.lock().unwrap_or_else(|p| p.into_inner()) = None;
        if let Err(panic) = own {
            std::panic::resume_unwind(panic);
        }
        assert!(
            !shared.panicked.load(Ordering::Acquire),
            "a team member's job panicked"
        );
    }
}

struct Items<T>(*mut T);
impl<T> Clone for Items<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Items<T> {}
// SAFETY: each member reaches only its own items (`each_mut`), which may move between threads.
unsafe impl<T: Send> Send for Items<T> {}
unsafe impl<T: Send> Sync for Items<T> {}

impl Shared {
    fn serve(&self, member: usize) {
        let mut seen = 0;
        loop {
            let generation = loop {
                let now = self.generation.load(Ordering::Acquire);
                if now != seen {
                    break now;
                }
                if self.stop.load(Ordering::Acquire) {
                    return;
                }
                std::hint::spin_loop();
            };
            seen = generation;
            let job = *self.job.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(job) = job
                && catch_unwind(AssertUnwindSafe(|| job(member))).is_err()
            {
                self.panicked.store(true, Ordering::Release);
            }
            self.busy.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// The share `member` of `members` takes of `0..count`, in contiguous ranges of whole `grain`s.
pub fn share(count: usize, grain: usize, member: usize, members: usize) -> std::ops::Range<usize> {
    let grains = count.div_ceil(grain);
    let (each, extra) = (grains / members, grains % members);
    let start = member * each + member.min(extra);
    let end = start + each + usize::from(member < extra);
    (start * grain).min(count)..(end * grain).min(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_member_takes_its_share_once_per_job() {
        for members in 1..=5 {
            Team::run(members, |team| {
                assert_eq!(team.members(), members);
                for round in 0..200 {
                    let hits: Vec<AtomicUsize> =
                        (0..members).map(|_| AtomicUsize::new(0)).collect();
                    team.split(&|member| {
                        hits[member].fetch_add(round + 1, Ordering::Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == round + 1));
                }
            });
        }
    }

    #[test]
    fn each_item_is_worked_once() {
        Team::run(3, |team| {
            let mut items: Vec<usize> = vec![0; 10];
            team.each_mut(&mut items, |index, item| *item += index + 1);
            assert_eq!(items, (1..=10).collect::<Vec<_>>());
        });
    }

    #[test]
    fn shares_cover_the_range_in_whole_grains() {
        for count in 0..40 {
            for grain in 1..4 {
                for members in 1..6 {
                    let ranges: Vec<_> = (0..members)
                        .map(|m| share(count, grain, m, members))
                        .collect();
                    let mut next = 0;
                    for range in &ranges {
                        assert_eq!(range.start, next);
                        assert!(range.start == count || range.start % grain == 0);
                        next = range.end;
                    }
                    assert_eq!(next, count);
                }
            }
        }
    }

    #[test]
    fn a_member_panic_reaches_the_caller() {
        let caught = catch_unwind(|| {
            Team::run(3, |team| {
                team.split(&|member| assert_ne!(member, 2, "boom"));
            });
        });
        assert!(caught.is_err());
    }
}
