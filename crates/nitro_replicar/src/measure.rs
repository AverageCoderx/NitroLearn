//! The car error: from a car's row at a server update, its labelled controls replayed for about
//! `horizon` ticks with the car alone in RocketSim, and the distance to where the replay's update at the end puts
//! it. The windows are chosen once, on the conversion before the fit (free of contacts from end to end), and
//! measured on both conversions by the updates' order in each run, so before and after count the same windows.

use replicar_format::header::PlayerInfo;
use replicar_format::record::Frame;

use crate::car::{self, Alone};
use crate::track;

/// The longest a window may run past `horizon` (ticks) to end on an update.
const SLACK: usize = 8;

/// A window: player, the run's first row, and the first and last update by their order in the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    player: usize,
    first_row: usize,
    from: usize,
    to: usize,
}

pub fn windows(frames: &[Frame], players: usize, horizon: usize) -> Vec<Window> {
    let mut out = Vec::new();
    for run in track::runs(frames, players) {
        let u = &run.updates;
        let clear: Vec<bool> = run
            .rows
            .iter()
            .map(|&r| !car::near_contact(&frames[r], run.player))
            .collect();
        for a in 0..u.len() {
            let Some(b) = (a + 1..u.len()).find(|&b| u[b] - u[a] >= horizon) else {
                break;
            };
            if u[b] - u[a] <= horizon + SLACK && clear[u[a]..=u[b]].iter().all(|c| *c) {
                out.push(Window {
                    player: run.player,
                    first_row: run.rows[0],
                    from: a,
                    to: b,
                });
            }
        }
    }
    out
}

/// Each window's error (uu), `None` where the conversion's runs do not hold it.
pub fn errors(
    players: &[PlayerInfo],
    frames: &[Frame],
    windows: &[Window],
) -> Result<Vec<Option<f32>>, String> {
    let runs = track::runs(frames, players.len());
    let mut arenas: Vec<Option<Alone>> = (0..players.len()).map(|_| None).collect();
    let mut out = Vec::with_capacity(windows.len());
    for w in windows {
        let run = runs
            .iter()
            .find(|r| r.player == w.player && r.rows[0] == w.first_row);
        let Some(run) = run.filter(|r| w.to < r.updates.len()) else {
            out.push(None);
            continue;
        };
        let alone = match &mut arenas[w.player] {
            Some(a) => a,
            slot => slot.insert(Alone::new(&players[w.player])?),
        };
        let car = |k: usize| {
            frames[run.rows[k]].state.cars[w.player]
                .as_ref()
                .expect("an active car")
        };
        let (a, b) = (run.updates[w.from], run.updates[w.to]);
        alone.start(car::state_of(car(a))?);
        for k in a..b {
            alone.step(car::controls(&car(k).controls));
        }
        let end = glam::Vec3A::from(car(b).body.position);
        out.push(Some(alone.state().phys.pos.distance(end)));
    }
    Ok(out)
}
