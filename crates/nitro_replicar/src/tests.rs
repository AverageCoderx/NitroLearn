//! The timing fit and the car error on a car whose true controls are known: a car simulated alone with scripted
//! presses becomes a conversion's rows (an update every 4 ticks), the labelled timing is put a few ticks off, and
//! the fit must find the true ticks. Needs the collision meshes (`NITRO_TEST_MESHES` or the repository's
//! collision_meshes); skips without them, as the core's tests do.

use std::path::PathBuf;

use replicar::rocketsim::{CarControls, CarState};
use replicar_format::header::PlayerInfo;
use replicar_format::record::{Frame, Game, State, Updates};
use replicar_format::{CarStatus, ClockPhase, FrameIndex, Period};

use crate::car::{self, Alone};
use crate::{fit, measure};

fn meshes() -> Option<replicar::Meshes> {
    let dir = std::env::var_os("NITRO_TEST_MESHES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../collision_meshes"),
        PathBuf::from,
    );
    if !dir.join("soccar").is_dir() {
        println!("SKIPPED: no collision meshes");
        return None;
    }
    Some(replicar::Meshes::load(&dir).expect("meshes"))
}

fn player() -> PlayerInfo {
    PlayerInfo {
        index: 0,
        key: "test".to_owned(),
        name: None,
        team: 0,
        body_product_id: None,
        hitbox: "octane".to_owned(),
        final_stats: std::collections::BTreeMap::default(),
    }
}

/// The scripted controls applied from tick `t`: throttle, a steer ramp at 40-43, boost 60-99, a jump held
/// 120-129, then a forward dodge pressed at 150.
fn script(t: usize) -> CarControls {
    CarControls {
        throttle: 1.0,
        steer: match t {
            0..40 => 0.0,
            40..44 => 0.15 * (t - 39) as f32,
            _ => 0.6,
        },
        pitch: if t >= 140 { -1.0 } else { 0.0 },
        jump: (120..130).contains(&t) || (150..156).contains(&t),
        boost: (60..100).contains(&t),
        ..CarControls::default()
    }
}

const TICKS: usize = 240;

/// The rows of the scripted run, true to the simulation: the state before each tick and the controls applied
/// from it.
fn truth() -> (Vec<Frame>, Vec<CarState>) {
    let mut alone = Alone::new(&player()).expect("arena");
    let mut start = CarState::default();
    start.phys.pos = replicar::rocketsim::Vec3A::new(0.0, -2000.0, 17.0);
    start.set_contact_flags(true, [false; 4]);
    start.boost = 100.0;
    alone.start(start);
    for _ in 0..10 {
        alone.step(CarControls::default()); // settled on its wheels
    }
    let mut states = Vec::new();
    let mut frames = Vec::new();
    for t in 0..TICKS {
        let mut s = alone.state();
        s.controls = script(t);
        states.push(s);
        let mut record = replicar_format::record::Car::default();
        let previous = if t == 0 {
            CarControls::default()
        } else {
            script(t - 1)
        };
        car::write(&s, &script(t), &previous, &mut record);
        frames.push(Frame {
            frame: FrameIndex(t as u32),
            frame_row: true,
            segment: Some(0),
            replay_time: t as f32 / 120.0,
            replay_tick: t as u32,
            sim_tick: t as u64,
            state: State {
                car_status: vec![CarStatus::Active],
                car_status_inferred: vec![false],
                cars: vec![Some(record)],
                ..State::default()
            },
            game: Game {
                period: Period::Regulation,
                clock_phase: ClockPhase::Running,
                seconds_remaining: None,
                overtime_seconds: None,
                scores: [None; 2],
                events: Vec::new(),
                stat_events: Vec::new(),
                ball_contacts: Vec::new(),
                boost_pickups: Vec::new(),
            },
            updates: Updates {
                car_updated: vec![Some(t % 4 == 0)],
                ..Updates::default()
            },
            future: None,
        });
        alone.step(script(t));
    }
    // The ball far from the car on every row: no contacts.
    for f in &mut frames {
        f.state.ball.body.position = [3000.0, 4000.0, 1800.0];
        f.state.ball.body.rotation = [0.0, 0.0, 0.0, 1.0];
    }
    (frames, states)
}

/// The rows replicar would make from `controls`: the car simulated with them, its body and boost put back to
/// the truth at each update (every 4 ticks), as replicar applies the server's updates.
fn as_replicar<F: Fn(usize) -> CarControls>(
    true_frames: &[Frame],
    states: &[CarState],
    controls: F,
) -> Vec<Frame> {
    let mut frames = true_frames.to_vec();
    let mut alone = Alone::new(&player()).expect("arena");
    alone.start(states[0]);
    for t in 0..TICKS {
        if t > 0 && t % 4 == 0 {
            let mut s = alone.state();
            s.phys = states[t].phys;
            s.boost = states[t].boost;
            alone.set(s);
        }
        let previous = if t == 0 {
            CarControls::default()
        } else {
            controls(t - 1)
        };
        let car = frames[t].state.cars[0].as_mut().expect("car");
        car::write(&alone.state(), &controls(t), &previous, car);
        alone.step(controls(t));
    }
    frames
}

/// The true rows with other controls written on them (the states unchanged).
fn labelled<F: Fn(usize) -> CarControls>(frames: &mut [Frame], controls: F) {
    for (t, f) in frames.iter_mut().enumerate() {
        let c = f.state.cars[0].as_mut().expect("car");
        let given = controls(t);
        c.controls.throttle = given.throttle;
        c.controls.steer = given.steer;
        c.controls.pitch = given.pitch;
        c.controls.jump = given.jump;
        c.controls.boost = given.boost;
    }
}

fn jump_presses(frames: &[Frame]) -> Vec<usize> {
    let jump = |t: usize| {
        frames[t].state.cars[0]
            .as_ref()
            .is_some_and(|c| c.controls.jump)
    };
    (1..frames.len())
        .filter(|&t| jump(t) && !jump(t - 1))
        .collect()
}

#[test]
fn the_fit_finds_the_true_ticks_of_presses_put_off() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, states) = truth();
    // The converter's guess: the first jump 3 ticks late, the dodge 2 early, the boost 2 late, the steer ramp
    // 2 late.
    let mut frames = as_replicar(&true_frames, &states, |t| CarControls {
        jump: (123..133).contains(&t) || (148..154).contains(&t),
        boost: (62..102).contains(&t),
        steer: script(t.saturating_sub(2)).steer,
        ..script(t)
    });
    let players = [player()];
    let report = fit::fit(&players, &mut frames, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(jump_presses(&frames), vec![120, 150], "{report:?}");
    let get = |frames: &[Frame], t: usize| frames[t].state.cars[0].as_ref().expect("car").controls;
    for t in 0..TICKS {
        let (got, want) = (get(&frames, t), get(&true_frames, t));
        // A release after a dodge changes nothing RocketSim simulates: its tick cannot be found.
        if t < 150 {
            assert_eq!(got.jump, want.jump, "tick {t}");
        }
        assert_eq!(got.boost, want.boost, "tick {t}");
        assert_eq!(got.steer.to_bits(), want.steer.to_bits(), "tick {t}");
    }
    assert!(
        report.error_after < 0.05 * report.error_before,
        "{report:?}"
    );
    // The values were never changed: the dodge keeps its own pitch.
    assert_eq!(get(&frames, 150).pitch, -1.0);
}

#[test]
fn true_timing_stays_where_it_is() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, _) = truth();
    let mut frames = true_frames.clone();
    let report = fit::fit(&[player()], &mut frames, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(
        report.jump_edges[1] + report.control_changes[1] + report.updates[1],
        0,
        "{report:?}"
    );
    assert_eq!(frames, true_frames);
}

#[test]
fn an_update_off_by_ticks_comes_back() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, _) = truth();
    let mut frames = true_frames.clone();
    // The update of tick 80 put on tick 82 (its body, the replay's, as the converter applied it there).
    let at80 = frames[80].state.cars[0];
    frames[82].state.cars[0] = at80;
    frames[80].updates.car_updated[0] = Some(false);
    frames[82].updates.car_updated[0] = Some(true);
    let report = fit::fit(&[player()], &mut frames, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(frames[80].updates.car_updated[0], Some(true), "{report:?}");
    assert_eq!(frames[82].updates.car_updated[0], Some(false));
    let moved: Vec<usize> = (0..TICKS)
        .filter(|&t| frames[t].updates.car_updated[0] != true_frames[t].updates.car_updated[0])
        .collect();
    assert_eq!(report.updates[1], 1, "{moved:?}");
}

#[test]
fn the_fit_is_deterministic() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, states) = truth();
    let mut a = as_replicar(&true_frames, &states, |t| CarControls {
        jump: (122..132).contains(&t) || (150..156).contains(&t),
        ..script(t)
    });
    let mut b = a.clone();
    fit::fit(&[player()], &mut a, fit::Settings::DEFAULT).expect("fit");
    fit::fit(&[player()], &mut b, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(a, b);
}

#[test]
fn the_car_error_is_small_with_the_true_controls_and_counts_the_same_windows() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, _) = truth();
    let mut off = true_frames.clone();
    labelled(&mut off, |t| CarControls {
        boost: (66..106).contains(&t),
        ..script(t)
    });
    let windows = measure::windows(&true_frames, 1, 20);
    assert!(windows.len() >= 40, "{}", windows.len());
    let truth = measure::errors(&[player()], &true_frames, &windows).expect("errors");
    let wrong = measure::errors(&[player()], &off, &windows).expect("errors");
    let truth: Vec<f32> = truth.into_iter().map(|e| e.expect("held")).collect();
    let wrong: Vec<f32> = wrong.into_iter().map(|e| e.expect("held")).collect();
    // A state restored from a row lacks RocketSim's wheel contacts, which costs a few uu over 20 ticks while
    // the car speeds up on the ground; the true controls are still far closer.
    let sorted = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v
    };
    let (truth, wrong) = (sorted(truth), sorted(wrong));
    assert!(truth[truth.len() / 2] < 0.1, "{truth:?}");
    assert!(
        wrong[wrong.len() - 1] > 2.0 * truth[truth.len() - 1],
        "{truth:?} {wrong:?}"
    );
}

#[test]
fn a_car_near_the_ball_is_left_to_replicar() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, _) = truth();
    let mut frames = true_frames.clone();
    labelled(&mut frames, |t| CarControls {
        jump: (123..133).contains(&t) || (150..156).contains(&t),
        ..script(t)
    });
    for f in &mut frames {
        f.state.ball.body.position = f.state.cars[0].expect("car").body.position;
    }
    let before = frames.clone();
    let report = fit::fit(&[player()], &mut frames, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(report.updates_counted, 0);
    assert_eq!(frames, before);
}

#[test]
fn a_press_is_judged_by_what_it_did_in_replicars_rows() {
    if meshes().is_none() {
        return;
    }
    let (true_frames, states) = truth();
    // A press at 138 that replicar's rows show doing nothing (its simulation, with the ball and the other cars,
    // had no dodge there); the lone car would dodge with it, so the fit could only move it on the lone car's
    // differences.
    let controls = |t: usize| CarControls {
        jump: (120..130).contains(&t) || (138..140).contains(&t) || (150..156).contains(&t),
        ..script(t)
    };
    let mut frames = as_replicar(&true_frames, &states, script);
    labelled(&mut frames, controls);
    let report = fit::fit(&[player()], &mut frames, fit::Settings::DEFAULT).expect("fit");
    assert_eq!(jump_presses(&frames), vec![120, 138, 150], "{report:?}");
}

#[test]
fn an_update_moved_onto_the_row_another_left_keeps_its_flag() {
    if meshes().is_none() {
        return;
    }
    // Updates at 84 and 86, each put 2 ticks early by the converter: both move 2 later, the first onto the row
    // the second left.
    let (true_frames, _) = truth();
    let mut truth_frames = true_frames.clone();
    truth_frames[86].updates.car_updated[0] = Some(true);
    let mut frames = truth_frames.clone();
    let (at84, at86) = (frames[84].state.cars[0], frames[86].state.cars[0]);
    frames[82].state.cars[0] = at84;
    frames[84].state.cars[0] = at86;
    frames[82].updates.car_updated[0] = Some(true);
    frames[86].updates.car_updated[0] = Some(false);
    let report = fit::fit(&[player()], &mut frames, fit::Settings::DEFAULT).expect("fit");
    let flags: Vec<usize> = (76..95)
        .filter(|&t| frames[t].updates.car_updated[0] == Some(true))
        .collect();
    assert_eq!(flags, vec![76, 80, 84, 86, 88, 92], "{report:?}");
}
