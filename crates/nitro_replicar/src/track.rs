//! A car's runs: rows in play, one tick apart, with the car active, and the rows the server updated it on.

use replicar_format::CarStatus;
use replicar_format::record::Frame;

/// One unbroken stretch of a car's rows: indices into the conversion's frames.
pub struct Run {
    pub player: usize,
    pub rows: Vec<usize>,
    /// Positions in `rows` of the rows that carry a server update of the car.
    pub updates: Vec<usize>,
}

fn active(frame: &Frame, p: usize) -> bool {
    frame.state.car_status.get(p) == Some(&CarStatus::Active)
        && frame
            .state
            .cars
            .get(p)
            .copied()
            .flatten()
            .is_some_and(|c| !c.internals.is_demoed)
}

/// Every car's runs with at least two updates, in player then row order.
pub fn runs(frames: &[Frame], players: usize) -> Vec<Run> {
    let mut out = Vec::new();
    for p in 0..players {
        let mut current: Option<Run> = None;
        let mut last: Option<(u32, u64)> = None;
        for (i, frame) in frames.iter().enumerate() {
            let usable = frame.segment.is_some() && active(frame, p);
            let follows = usable
                && last.is_some_and(|(s, t)| Some(s) == frame.segment && frame.sim_tick == t + 1);
            if !follows {
                if let Some(run) = current.take()
                    && run.updates.len() >= 2
                {
                    out.push(run);
                }
                if usable {
                    current = Some(Run {
                        player: p,
                        rows: Vec::new(),
                        updates: Vec::new(),
                    });
                }
            }
            if let Some(run) = current.as_mut() {
                if frame.updates.car_updated.get(p) == Some(&Some(true)) {
                    run.updates.push(run.rows.len());
                }
                run.rows.push(i);
            }
            last = usable.then(|| (frame.segment.unwrap_or(0), frame.sim_tick));
        }
        if let Some(run) = current
            && run.updates.len() >= 2
        {
            out.push(run);
        }
    }
    out
}
