//! A recorded game scored with a bot's rewards: what each reward would have paid each player, step by step,
//! and the game as viewer frames (`capture`), for the reward chart over a real match ("what would my
//! rewards pay a pro?").
//!
//! The recording gives the states; the events the rewards read (touches, bumps, demolitions, goals, wheel
//! and flip details) come from RocketSim: at every decision the arena is put in the recorded state and plays
//! the step with the players' recorded controls. The rewards are then paid on the recorded states at the
//! step's two ends, so a reward that reads positions sees the real game. Rewards that remember start again
//! at every kickoff and after every goal or longer gap. The pads are taken as all available (a recording does not
//! hold them).

use crate::capture::{self, BLUE_GOAL, ORANGE_GOAL, PLAYING, RESTART};
use crate::demos::car_snapshot;
use crate::replay::{is_kickoff, teams_of};
use crate::rewards::{RewardSet, RewardStats, RewardTerm};
use crate::rlpr::{CarRecord, Recording, Tick};
use crate::sim::Sim;
use crate::snapshot::{
    CarDetail, CarMoves, Controls, GameSnapshot, MAX_CARS, PadSnapshot, StepEvents, Team,
};

/// Past this |y| the ball's centre is in a goal (the goal line plus the ball's radius).
const SCORED_Y: f32 = 5124.25 + 92.75;
/// A car counts as supersonic from this speed (uu/s).
const SUPERSONIC_SPEED: f32 = 2200.0;
/// Ticks a step may be missing from a recording and still be played (more: the game starts again there).
const MISSED_TICKS: usize = 5;
/// A recording leaves out the moments after a goal: a gap that ends with the ball at the centre spot, the
/// ball last seen within this of a goal line (uu, |y|), is taken as a goal in that net.
const NEAR_GOAL_Y: f32 = 4400.0;
/// A flip's window after the jump (s): a car that can no longer flip is at least this far past its jump.
const FLIP_WINDOW: f32 = 1.25;

pub struct Scored {
    /// The cars in the frames and the pay, as the viewer orders them (Blue first).
    pub teams: Vec<Team>,
    /// Viewer frames (`capture` layout), flat.
    pub frames: Vec<f32>,
    /// Per step: the viewer frame it ends on, and each term's pay to each car, `[term * cars + car]`.
    pub step_frames: Vec<u32>,
    pub pay: Vec<f32>,
    pub pads: Vec<[f32; 4]>,
}

pub struct ScoreRules<'a> {
    pub rewards: &'a [RewardTerm],
    pub tick_skip: u32,
    /// Ticks per viewer frame: a divisor of `tick_skip`.
    pub every: u32,
    /// A recording made from replays holds in each row the controls applied next (`demos`).
    pub replicar_rows: bool,
    /// At most this many ticks of the recording are read.
    pub max_ticks: usize,
    /// The recorded cars' teams when they are known (a converted replay: Blue first, half each); else they
    /// are told from the kickoffs (`replay::teams_of`).
    pub teams: Option<Vec<Team>>,
}

/// Scores `recording` (the meshes must be loaded).
pub fn score(recording: &Recording, rules: &ScoreRules) -> Result<Scored, String> {
    let k = rules.tick_skip as usize;
    if rules.every == 0 || !rules.tick_skip.is_multiple_of(rules.every) {
        return Err("the frame rate must divide the tick skip".to_owned());
    }
    let teams = match &rules.teams {
        Some(teams) if teams.len() == recording.num_cars as usize => teams.clone(),
        Some(_) => return Err("the teams given do not fit the recording".to_owned()),
        None => teams_of(recording)?,
    };
    let blue = teams.iter().filter(|&&t| t == Team::Blue).count();
    let orange = teams.len() - blue;
    if blue == 0 || (orange != 0 && orange != blue) || teams.len() > MAX_CARS {
        return Err(format!("a {blue} against {orange} game cannot be scored"));
    }
    if let Some(tick) = recording
        .ticks
        .iter()
        .find(|tick| tick.cars.len() != teams.len())
    {
        return Err(format!(
            "frame {} of the recording holds {} cars, not {}",
            tick.frame,
            tick.cars.len(),
            teams.len()
        ));
    }
    let mut sim = Sim::new(blue, orange > 0, 0);
    let cars = sim.num_cars();
    // Simulated car `slot` plays the recorded car `order[slot]`, of the same team.
    let mut pools: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (c, team) in teams.iter().enumerate().rev() {
        pools[team.index()].push(c);
    }
    let order: Vec<usize> = (0..cars)
        .map(|slot| {
            pools[sim.team(slot).index()]
                .pop()
                .expect("teams of equal size")
        })
        .collect();
    // The viewer shows the recorded cars Blue first: viewer car `v` is simulated slot `shown[v]`.
    let mut shown: Vec<usize> = (0..cars).collect();
    shown.sort_by_key(|&slot| (sim.team(slot).index(), order[slot]));
    // A frame held twice in a row (some recordings hold almost every frame twice): the later copy stands.
    let mut ticks: Vec<&Tick> = Vec::with_capacity(recording.ticks.len());
    for tick in &recording.ticks {
        let full = ticks.len() >= rules.max_ticks;
        match ticks.last_mut() {
            Some(last) if last.frame == tick.frame => *last = tick,
            _ if full => break,
            _ => ticks.push(tick),
        }
    }
    let mut set = RewardSet::new(rules.rewards.to_vec(), rules.tick_skip);
    set.keep_parts(true);
    let terms = rules.rewards.len();
    let mut stats = RewardStats::new(terms);
    let mut out = Scored {
        teams: shown.iter().map(|&slot| sim.team(slot)).collect(),
        frames: Vec::new(),
        step_frames: Vec::new(),
        pay: Vec::new(),
        pads: sim.pad_layout(),
    };
    let (mut prev, mut cur, mut played) = (
        GameSnapshot::default(),
        GameSnapshot::default(),
        GameSnapshot::default(),
    );
    let mut events = StepEvents::default();
    let mut paid = vec![0.0f32; cars];
    let counted = vec![true; cars];
    let mut touched = vec![false; cars];
    let mut score = [0.0f32; 2];
    let mut frame_count = 0u32;
    // Recordings drop the odd tick: a step may miss a few and still be played as one.
    let consecutive = |a: usize, b: usize| {
        ticks[b].frame as usize <= ticks[a].frame as usize + (b - a) + MISSED_TICKS
    };
    let mut t = 0usize;
    let mut fresh = true;
    while t + k < ticks.len() {
        if !consecutive(t, t + k) {
            // A gap: the next game part starts at the first tick after it.
            t = (t + 1..t + k + 1)
                .find(|&j| !consecutive(t, j))
                .unwrap_or(t + k);
            fresh = true;
            continue;
        }
        if fresh {
            set.reset();
            events.last_touch = None;
            snapshot(&mut prev, ticks[t], &teams, &order);
            write(
                &mut out.frames,
                &prev,
                ticks[t],
                &order,
                &shown,
                RESTART,
                score,
                &mut touched,
            );
            frame_count += 1;
            fresh = false;
        } else {
            snapshot(&mut prev, ticks[t], &teams, &order);
        }
        place(&mut sim, &prev, ticks[t], &order);
        events.clear(cars);
        events.step_seconds = rules.tick_skip as f32 / 120.0;
        sim.details(&mut events);
        std::mem::swap(&mut events.details, &mut events.prev_details);
        for m in 1..=k {
            let row = if rules.replicar_rows {
                t + m - 1
            } else {
                t + m
            };
            for (slot, &rec) in order.iter().enumerate() {
                sim.set_controls(slot, ticks[row].cars[rec].controls);
            }
            sim.step(1, &mut events);
        }
        sim.read(&mut played);
        sim.details(&mut events);
        events.tick = sim.tick();
        snapshot(&mut cur, ticks[t + k], &teams, &order);
        // The last tick before the next gap, when the next step cannot be played whole: a goal the
        // recording cut out ends the game part there.
        let end = t + k;
        let last = (end..ticks.len().min(end + k))
            .find(|&e| e + 1 >= ticks.len() || !consecutive(e, e + 1))
            .filter(|&e| e + 1 < ticks.len() && at_centre(&ticks[e + 1].ball));
        let net = |y: f32| if y > 0.0 { Team::Blue } else { Team::Orange };
        let scored = played
            .goal_scored
            .or_else(|| (cur.ball.pos.y.abs() > SCORED_Y).then(|| net(cur.ball.pos.y)))
            .or_else(|| {
                let y = ticks[last?].ball.pos.y;
                (y.abs() > NEAR_GOAL_Y).then(|| net(y))
            });
        cur.goal_scored = scored;
        set.compute(&prev, &cur, &events, &counted, &mut paid, &mut stats);
        let parts = set.parts();
        for term in 0..terms {
            out.pay
                .extend(shown.iter().map(|&slot| parts[term * cars + slot]));
        }
        for (slot, car) in events.cars.iter().enumerate() {
            touched[slot] |= car.touched_ball;
        }
        for m in (rules.every as usize..=k).step_by(rules.every as usize) {
            let marker = match scored {
                Some(team) if m == k => {
                    score[team.index()] += 1.0;
                    if team == Team::Blue {
                        BLUE_GOAL
                    } else {
                        ORANGE_GOAL
                    }
                }
                _ => PLAYING,
            };
            let mut state = GameSnapshot::default();
            snapshot(&mut state, ticks[t + m], &teams, &order);
            let touches = if m == k {
                &mut touched[..]
            } else {
                &mut [false; 8][..cars]
            };
            write(
                &mut out.frames,
                &state,
                ticks[t + m],
                &order,
                &shown,
                marker,
                score,
                touches,
            );
            frame_count += 1;
        }
        out.step_frames.push(frame_count - 1);
        t += k;
        if scored.is_some() {
            // The ball is in the net until the next kickoff: the game goes on from there.
            t = (t..ticks.len())
                .find(|&j| is_kickoff(ticks[j]) || at_centre(&ticks[j].ball))
                .unwrap_or(ticks.len());
            fresh = true;
        }
    }
    Ok(out)
}

/// The ball at the centre spot and not moving across the field: a kickoff is about to start (it may still
/// be dropping onto the spot).
fn at_centre(ball: &crate::snapshot::Body) -> bool {
    ball.pos.x.abs() < 5.0 && ball.pos.y.abs() < 5.0 && ball.vel.x.hypot(ball.vel.y) < 10.0
}

/// The recorded tick as the simulated arena orders its cars.
fn snapshot(snap: &mut GameSnapshot, tick: &Tick, teams: &[Team], order: &[usize]) {
    snap.ball = tick.ball;
    snap.cars.clear();
    snap.cars.extend(order.iter().map(|&rec| {
        let record = &tick.cars[rec];
        car_snapshot(
            record,
            teams[rec],
            record.body.vel.length() >= SUPERSONIC_SPEED,
        )
    }));
    snap.pads.fill(PadSnapshot::default());
    snap.goal_scored = None;
}

/// Puts the arena in the recorded state of `tick` (`snap`, already in the arena's order).
fn place(sim: &mut Sim, snap: &GameSnapshot, tick: &Tick, order: &[usize]) {
    let cars: Vec<_> = snap
        .cars
        .iter()
        .map(|car| (car.body, car.boost, car.on_ground))
        .collect();
    sim.place(snap.ball, &cars);
    for (slot, &rec) in order.iter().enumerate() {
        sim.set_moves(slot, &moves(&tick.cars[rec], &snap.cars[slot]));
    }
    sim.forget();
}

fn moves(record: &CarRecord, car: &crate::snapshot::CarSnapshot) -> CarMoves {
    CarMoves {
        has_jumped: record.has_jumped,
        has_double_jumped: car.has_double_jumped,
        has_flipped: car.has_flipped,
        is_jumping: record.is_jumping,
        is_flipping: record.is_flipping,
        jump_time: record.jump_time,
        flip_time: record.flip_time,
        flip_rel_torque: record.flip_rel_torque,
        air_time: 0.0,
        air_time_since_jump: if record.on_ground || record.has_flip {
            0.0
        } else {
            FLIP_WINDOW
        },
    }
}

/// One viewer frame of the recorded `state` (in the arena's order), its cars shown Blue first.
#[allow(clippy::too_many_arguments)]
fn write(
    frames: &mut Vec<f32>,
    state: &GameSnapshot,
    tick: &Tick,
    order: &[usize],
    shown: &[usize],
    marker: f32,
    score: [f32; 2],
    touched: &mut [bool],
) {
    let mut view = state.clone();
    view.cars = shown.iter().map(|&slot| state.cars[slot]).collect();
    let mut details = Vec::with_capacity(shown.len());
    let mut controls = Vec::with_capacity(shown.len());
    let mut touches = Vec::with_capacity(shown.len());
    for &slot in shown {
        let record = &tick.cars[order[slot]];
        let c: Controls = record.controls;
        controls.push(c);
        touches.push(std::mem::take(&mut touched[slot]));
        details.push(CarDetail {
            controls: c,
            wheel_contacts: [record.on_ground; 4],
            is_jumping: record.is_jumping,
            is_flipping: record.is_flipping,
            is_boosting: c.boost && record.boost > 0.0,
            handbrake: f32::from(u8::from(c.handbrake)),
            ..CarDetail::default()
        });
    }
    capture::write_frame(
        frames,
        &view,
        &details,
        &controls,
        marker,
        score,
        &mut touches,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::meshes_loaded;

    /// One of the recordings RocketSim ships, in cargo's checkout of it (None when it is not there).
    fn bundled(name: &str) -> Option<std::path::PathBuf> {
        let home = std::env::var_os("CARGO_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cargo"))
            })?;
        std::fs::read_dir(home.join("git").join("checkouts"))
            .ok()?
            .flatten()
            .filter(|d| d.file_name().to_string_lossy().starts_with("rocketsim-"))
            .flat_map(|d| std::fs::read_dir(d.path()).into_iter().flatten().flatten())
            .map(|rev| {
                rev.path()
                    .join("rocketsim_test")
                    .join("recordings")
                    .join(name)
            })
            .find(|file| file.is_file())
    }

    /// A still car on a kickoff spot of `team`'s half.
    fn parked(team: Team, x: f32) -> CarRecord {
        let y = if team == Team::Blue { -4608.0 } else { 4608.0 };
        CarRecord {
            body: crate::snapshot::Body {
                pos: rocketsim::Vec3A::new(x, y, 17.0),
                rot: rocketsim::Mat3A::IDENTITY,
                vel: rocketsim::Vec3A::ZERO,
                ang_vel: rocketsim::Vec3A::ZERO,
            },
            on_ground: true,
            is_jumping: false,
            is_flipping: false,
            jump_time: 0.0,
            flip_time: 0.0,
            has_jumped: false,
            double_jumped_or_flipped: false,
            has_flip: true,
            flip_rel_torque: rocketsim::Vec3A::ZERO,
            boost: 0.33,
            touching_ball: false,
            controls: Controls::default(),
            touching_car: false,
        }
    }

    /// `ticks` kickoff ticks of `blue` against `orange` cars.
    fn kickoffs(blue: usize, orange: usize, ticks: u32) -> Recording {
        let cars: Vec<CarRecord> = (0..blue)
            .map(|i| parked(Team::Blue, i as f32 * 300.0))
            .chain((0..orange).map(|i| parked(Team::Orange, i as f32 * 300.0)))
            .collect();
        let ball = crate::snapshot::Body {
            pos: rocketsim::Vec3A::new(0.0, 0.0, 93.15),
            rot: rocketsim::Mat3A::IDENTITY,
            vel: rocketsim::Vec3A::ZERO,
            ang_vel: rocketsim::Vec3A::ZERO,
        };
        Recording {
            version: 6,
            num_cars: (blue + orange) as u32,
            hitbox_min_bt: rocketsim::Vec3A::ZERO,
            hitbox_max_bt: rocketsim::Vec3A::ZERO,
            ticks: (0..ticks)
                .map(|frame| Tick {
                    frame,
                    cars: cars.clone(),
                    ball,
                    ball_world_contact: false,
                })
                .collect(),
        }
    }

    fn rules(rewards: &[RewardTerm]) -> ScoreRules<'_> {
        ScoreRules {
            rewards,
            tick_skip: 8,
            every: 4,
            replicar_rows: false,
            max_ticks: 120 * 300,
            teams: None,
        }
    }

    #[test]
    fn a_recording_with_a_tick_short_of_cars_is_refused_in_words() {
        if !meshes_loaded() {
            return;
        }
        let mut recording = kickoffs(1, 1, 64);
        recording.ticks[40].cars.pop();
        let rewards: Vec<RewardTerm> =
            serde_json::from_str(r#"[{"kind": {"name": "touch_ball"}, "weight": 1.0}]"#).unwrap();
        let err = score(&recording, &rules(&rewards)).err().expect("refused");
        assert!(err.contains("holds 1 cars, not 2"), "{err}");
    }

    #[test]
    fn a_game_of_more_cars_than_an_arena_holds_is_refused_in_words() {
        if !meshes_loaded() {
            return;
        }
        let recording = kickoffs(5, 5, 64);
        let rewards: Vec<RewardTerm> =
            serde_json::from_str(r#"[{"kind": {"name": "touch_ball"}, "weight": 1.0}]"#).unwrap();
        let err = score(&recording, &rules(&rewards)).err().expect("refused");
        assert!(err.contains("5 against 5"), "{err}");
    }

    #[test]
    fn a_recorded_game_is_scored_step_by_step_and_shown_frame_by_frame() {
        if !meshes_loaded() {
            return;
        }
        let Some(file) = bundled("nexto_1v1_300s.rlpr.zst") else {
            return;
        };
        let recording = crate::rlpr::read_file(&file).unwrap();
        let rewards: Vec<RewardTerm> = serde_json::from_str(
            r#"[{"kind": {"name": "touch_ball"}, "weight": 1.0},
                {"kind": {"name": "goal", "concede_scale": -1.0}, "weight": 10.0}]"#,
        )
        .unwrap();
        let rules = ScoreRules {
            rewards: &rewards,
            tick_skip: 8,
            every: 4,
            replicar_rows: false,
            max_ticks: 120 * 300,
            teams: None,
        };
        let scored = score(&recording, &rules).unwrap();
        let cars = scored.teams.len();
        let width = capture::HEAD + cars * capture::CAR;
        assert_eq!(scored.frames.len() % width, 0);
        let frames = scored.frames.len() / width;
        let steps = scored.step_frames.len();
        assert!(
            steps > 3000,
            "most of five minutes is played: {steps} steps"
        );
        assert_eq!(scored.pay.len(), steps * rewards.len() * cars);
        assert!(
            scored
                .teams
                .windows(2)
                .all(|w| w[0].index() <= w[1].index()),
            "Blue first"
        );
        assert!(scored.step_frames.windows(2).all(|w| w[0] < w[1]));
        assert!(frames as u32 > *scored.step_frames.last().unwrap());
        let step = |s: usize| &scored.pay[s * rewards.len() * cars..(s + 1) * rewards.len() * cars];
        let touches = (0..steps)
            .filter(|&s| step(s)[..cars].iter().any(|&v| v > 0.0))
            .count();
        assert!(touches > 50, "players touch the ball often: {touches}");
        // A goal pays the scorer 10 and charges the other side 10, on the step whose frame says so.
        let goals: Vec<usize> = (0..steps)
            .filter(|&s| step(s)[cars..].iter().any(|&v| v != 0.0))
            .collect();
        assert!(!goals.is_empty(), "five minutes of a real 1v1 hold goals");
        for &s in &goals {
            let pay = &step(s)[cars..];
            assert_eq!(pay.iter().sum::<f32>(), 0.0);
            let marker = scored.frames[scored.step_frames[s] as usize * width];
            let scorer = if pay[0] > 0.0 { BLUE_GOAL } else { ORANGE_GOAL };
            assert_eq!(marker, scorer);
        }
    }
}
