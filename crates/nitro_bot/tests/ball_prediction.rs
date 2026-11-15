//! Ball prediction in the bot: RLBot's prediction (6 s, a slice per tick), read at whole decisions
//! ahead, gives the observation training builds from its own prediction, for balls that bounce, hit a wall
//! or the ceiling, fly into a goal or roll. Without a prediction the bot coasts; with the meshes it
//! predicts the ball itself.

mod common;

use common::{field, packet, physics, random_snapshot};
use nitro_bot::packet::PadMap;
use nitro_bot::{Choice, NitroBot};
use nitro_core::actions::NUM_ACTIONS as BASE_ACTIONS;
use nitro_core::ball_prediction::{self, BallPath, Tolerance};
use nitro_core::custom::prelude::{Mat3A, Vec3A};
use nitro_core::obs::ObsKind;
use nitro_core::policy::{MAGIC, PLAIN_FORMAT, Policy};
use nitro_core::snapshot::{Body, Controls, GameSnapshot};
use rlbot::flat::{BallPrediction, MatchPhase, PredictionSlice};

const TIMES: [f32; 4] = [0.5, 1.0, 2.0, 3.0];

/// The prediction's numbers in a native observation: right after the ball's 9 (the rest of the observation
/// from a packet is checked in parity.rs).
fn prediction_part(obs: &[f32]) -> &[f32] {
    &obs[9..9 + 12 * TIMES.len()]
}

fn meshes_loaded() -> bool {
    let folder = std::env::var_os("NITRO_TEST_MESHES").map_or_else(
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../collision_meshes"),
        std::path::PathBuf::from,
    );
    match nitro_core::meshes::load(&folder) {
        Ok(report) if report.is_complete() => true,
        _ => {
            eprintln!("SKIPPED: no collision meshes at {}", folder.display());
            false
        }
    }
}

/// A one-layer native 1v1 policy trained with ball prediction at `TIMES`, favouring action 0.
fn predicting_policy() -> Policy {
    let obs_size = ObsKind::Native
        .layout(1, true, 0)
        .with_ball_prediction(TIMES.len())
        .size;
    let header = format!(
        r#"{{"format": 1, "run": "test", "agent_steps": 7, "game": {{"observation": "native",
        "action_masks": "standard", "team_size": 1, "opponents": true, "ball_prediction": {TIMES:?},
        "tick_skip": 8, "action_delay": 7}}, "obs_size": {obs_size}, "num_actions": {BASE_ACTIONS},
        "layers": [{{"kind": "linear", "inputs": {obs_size}, "outputs": {BASE_ACTIONS}}}]}}"#
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(PLAIN_FORMAT.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    for _ in 0..obs_size * BASE_ACTIONS {
        bytes.extend(0.0f32.to_le_bytes());
    }
    for action in 0..BASE_ACTIONS {
        bytes.extend(if action == 0 { 5.0f32 } else { 0.0 }.to_le_bytes());
    }
    Policy::from_bytes(&bytes).unwrap()
}

fn ball(pos: [f32; 3], vel: [f32; 3], ang_vel: [f32; 3]) -> Body {
    Body {
        pos: Vec3A::from_array(pos),
        vel: Vec3A::from_array(vel),
        ang_vel: Vec3A::from_array(ang_vel),
        ..Body::RESTING
    }
}

fn scripted() -> Vec<(&'static str, Body)> {
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

/// A 1v1 game with the ball at `ball`, the bot's car on the ground.
fn game(ball: Body, seed: u64) -> GameSnapshot {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut snap = random_snapshot(&mut rng, 2);
    snap.ball = ball;
    snap.cars[0].on_ground = true;
    snap.cars[0].body.rot = Mat3A::IDENTITY;
    snap
}

/// What training would observe: its own prediction, a decision at a time.
fn training_observation(snap: &GameSnapshot, me: usize) -> Vec<f32> {
    let steps = ball_prediction::decisions(&TIMES, 8).unwrap();
    let mut path = BallPath::for_observation(8, *steps.last().unwrap());
    let mut predicted = Vec::new();
    ball_prediction::pick(path.update(snap.ball, 1), &steps, &mut predicted);
    let size = ObsKind::Native
        .layout(1, true, 0)
        .with_ball_prediction(steps.len())
        .size;
    let mut out = vec![0.0; size];
    ObsKind::Native.build_predicted(snap, me, Controls::default(), 0, &predicted, &mut out);
    out
}

/// RLBot's ball prediction made with the packet of `frame` (its ball `ball`), encoded as RLBot v5 sends it
/// in a real match: 720 slices a tick apart, the first a tick after the packet (packet 74.5917 s, first
/// slice 74.6000 s), RocketSim's path a tick at a time.
fn rlbot_prediction(ball: Body, frame: u32) -> BallPrediction {
    let mut path = BallPath::new(1, 720, Tolerance::EXACT);
    let slices = path
        .update(ball, 1)
        .iter()
        .enumerate()
        .map(|(k, body)| PredictionSlice {
            game_seconds: (frame as usize + 1 + k) as f32 / 120.0,
            physics: physics(body),
        })
        .collect();
    BallPrediction { slices }
}

#[test]
fn rlbots_prediction_gives_the_observation_training_builds() {
    if !meshes_loaded() {
        return;
    }
    let pads = PadMap::new(&field()).unwrap();
    // Early in a match, at the real match's 74.59 s, and 10 minutes in (game times as 32-bit floats).
    for frame in [20, 8951, 72_000] {
        for (seed, (name, start)) in scripted().into_iter().enumerate() {
            let snap = game(start, seed as u64);
            let expected = training_observation(&snap, 0);
            let mut bot =
                NitroBot::new(predicting_policy(), &field(), 0, Choice::MostLikely, 1, 0).unwrap();
            assert!(bot.wants_ball_prediction());
            bot.set_ball_prediction(&rlbot_prediction(start, frame));
            bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff));
            assert!(bot.decided(), "{name} at frame {frame}");
            let seen = bot.observation();
            assert_eq!(seen.len(), expected.len());
            let worst = prediction_part(seen)
                .iter()
                .zip(prediction_part(&expected))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Both predictions are RocketSim's from the same state: measured 0.
            assert!(worst < 1e-4, "{name} at frame {frame}: off by {worst}");
        }
    }
}

/// A game at frames 20.. whose ball flies as RocketSim's: the frames' snapshots, from frame 20.
fn flight() -> Vec<GameSnapshot> {
    let (_, start) = scripted()[1];
    let mut path = BallPath::new(1, 40, Tolerance::EXACT);
    let mut balls = vec![start];
    balls.extend_from_slice(path.update(start, 1));
    balls.into_iter().map(|ball| game(ball, 7)).collect()
}

/// The bot's controls for frames 20..44 of `flight`, given before each packet the prediction `feed` makes
/// for its frame (if any); and how many decisions it took.
fn play(
    bot: &mut NitroBot,
    flight: &[GameSnapshot],
    feed: impl Fn(u32) -> Option<BallPrediction>,
) -> (Vec<Controls>, u64) {
    let pads = PadMap::new(&field()).unwrap();
    let sent = (20..44)
        .map(|frame| {
            if let Some(prediction) = feed(frame) {
                bot.set_ball_prediction(&prediction);
            }
            bot.controls(&packet(
                &flight[frame as usize - 20],
                &pads,
                frame,
                MatchPhase::Kickoff,
            ))
        })
        .collect();
    (sent, bot.rhythm_stats().decisions)
}

fn bot() -> NitroBot {
    NitroBot::new(predicting_policy(), &field(), 0, Choice::MostLikely, 1, 0).unwrap()
}

#[test]
fn without_a_fitting_prediction_the_bot_coasts_and_with_meshes_it_predicts_itself() {
    if !meshes_loaded() {
        return;
    }
    let flight = flight();
    let favourite = nitro_core::actions::ActionTable::new().controls(0);
    let ball_at = |frame: u32| flight[frame as usize - 20].ball;
    // No prediction: it decides at 20, 28 and 36, each time coasting.
    let (sent, decisions) = play(&mut bot(), &flight, |_| None);
    assert_eq!(decisions, 3);
    assert!(!sent.contains(&favourite), "no prediction: coasting");
    // RLBot's prediction made with each packet (its first slice a tick after it): the action from frame 27
    // on (the action delay is 7), and training's numbers.
    let mut fresh = bot();
    let (sent, _) = play(&mut fresh, &flight, |frame| {
        Some(rlbot_prediction(ball_at(frame), frame))
    });
    assert!(sent[..7].iter().all(|c| *c != favourite) && sent[7..].iter().all(|c| *c == favourite));
    let worst = prediction_part(fresh.observation())
        .iter()
        .zip(prediction_part(&training_observation(&flight[16], 0)))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "off by {worst}");
    // A tick late (made with the packet before): its first slice is now, this packet's ball, so it is used
    // and gives training's numbers.
    let mut late = bot();
    let (sent, _) = play(&mut late, &flight, |frame| {
        (frame > 20).then(|| rlbot_prediction(ball_at(frame - 1), frame - 1))
    });
    assert!(
        sent[15..].iter().all(|c| *c == favourite),
        "decided at 28 and 36: {sent:?}"
    );
    let expected = training_observation(&flight[16], 0); // the decision at frame 36
    let worst = prediction_part(late.observation())
        .iter()
        .zip(prediction_part(&expected))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "off by {worst}");
    // A stale prediction: made 20 ticks before, from a ball that is elsewhere now.
    let (sent, decisions) = play(&mut bot(), &flight, |frame| {
        Some(rlbot_prediction(ball_at(20), frame - 20))
    });
    assert_eq!(decisions, 3);
    assert!(!sent.contains(&favourite), "a stale prediction is not used");
    // With the meshes, the bot predicts the ball itself when RLBot's prediction is missing or stale.
    let mut own = bot();
    own.predict_ball_itself();
    let (sent, _) = play(&mut own, &flight, |_| None);
    assert!(sent[7..].iter().all(|c| *c == favourite), "{sent:?}");
    assert_eq!(
        prediction_part(own.observation()),
        prediction_part(&training_observation(&flight[16], 0))
    );
}

#[test]
fn a_plain_policy_wants_no_prediction() {
    let obs_size = ObsKind::Native.layout(1, true, 0).size;
    let header = format!(
        r#"{{"format": 1, "game": {{"observation": "native", "action_masks": "standard", "team_size": 1,
        "opponents": true, "tick_skip": 8, "action_delay": 7}}, "obs_size": {obs_size},
        "num_actions": {BASE_ACTIONS}, "layers": [{{"kind": "linear", "inputs": {obs_size},
        "outputs": {BASE_ACTIONS}}}]}}"#
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(PLAIN_FORMAT.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(vec![0u8; 4 * (obs_size + 1) * BASE_ACTIONS]);
    let bot = NitroBot::new(
        Policy::from_bytes(&bytes).unwrap(),
        &field(),
        0,
        Choice::MostLikely,
        1,
        0,
    )
    .unwrap();
    assert!(!bot.wants_ball_prediction());
    assert!(!nitro_bot::wants_ball_prediction(
        &Policy::from_bytes(&bytes).unwrap()
    ));
}
