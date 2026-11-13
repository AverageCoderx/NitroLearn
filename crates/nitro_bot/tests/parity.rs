//! Parity between a simulated state and the same state seen through an RLBot v5 packet:
//! the snapshot the bot builds from a synthesised packet matches the simulator's, so the
//! observation and the action mask are the same; and the bot decides on the training rhythm.

mod common;

use common::{field, packet, random_snapshot};
use nitro_bot::packet::{GROUND_JUMP_TICKS, JumpStarts, PadMap, snapshot};
use nitro_bot::rhythm::RhythmStats;
use nitro_bot::{Choice, NitroBot};
use nitro_core::actions::{ActionTable, MaskRule, NUM_ACTIONS};
use nitro_core::custom::prelude::{Mat3A, Vec3A};
use nitro_core::obs::ObsKind;
use nitro_core::policy::{MAGIC, PLAIN_FORMAT, Policy};
use nitro_core::snapshot::{Body, Controls, FLIP_WINDOW_SECONDS, GameSnapshot, PadSnapshot};
use rlbot::flat::{AirState, BoostPadState, MatchPhase};

fn close(a: Vec3A, b: Vec3A, tolerance: f32) -> bool {
    (a - b).length() <= tolerance
}

fn assert_bodies_match(a: &Body, b: &Body) {
    assert_eq!(a.pos, b.pos);
    assert_eq!(a.vel, b.vel);
    assert_eq!(a.ang_vel, b.ang_vel);
    for col in 0..3 {
        assert!(
            close(a.rot.col(col), b.rot.col(col), 2e-6),
            "{:?} vs {:?}",
            a.rot,
            b.rot
        );
    }
}

#[test]
fn a_packet_rebuilds_the_simulated_snapshot_and_its_observation() {
    let mut rng = fastrand::Rng::with_seed(11);
    let pads = PadMap::new(&field()).unwrap();
    let table = ActionTable::new();
    for team_size in [1usize, 2, 3] {
        for _ in 0..200 {
            let snap = random_snapshot(&mut rng, team_size * 2);
            let mut rebuilt = GameSnapshot::default();
            snapshot(
                &packet(&snap, &pads, 100, MatchPhase::Active),
                &pads,
                &mut JumpStarts::default(),
                &mut rebuilt,
            )
            .unwrap();
            assert_bodies_match(&snap.ball, &rebuilt.ball);
            assert_eq!(snap.cars.len(), rebuilt.cars.len());
            for (a, b) in snap.cars.iter().zip(&rebuilt.cars) {
                assert_bodies_match(&a.body, &b.body);
                assert_eq!(
                    (
                        a.team,
                        a.boost,
                        a.on_ground,
                        a.has_jumped,
                        a.has_double_jumped,
                        a.has_flipped
                    ),
                    (
                        b.team,
                        b.boost,
                        b.on_ground,
                        b.has_jumped,
                        b.has_double_jumped,
                        b.has_flipped
                    )
                );
                assert_eq!(
                    (
                        a.can_flip,
                        a.is_demoed,
                        a.is_supersonic,
                        a.demo_timer,
                        a.flip_time_left
                    ),
                    (
                        b.can_flip,
                        b.is_demoed,
                        b.is_supersonic,
                        b.demo_timer,
                        b.flip_time_left
                    )
                );
            }
            for (slot, (a, b)) in snap.pads.iter().zip(&rebuilt.pads).enumerate() {
                assert_eq!(a.active, b.active, "pad {slot}");
                assert!(
                    (a.cooldown - b.cooldown).abs() < 1e-5,
                    "pad {slot}: {a:?} vs {b:?}"
                );
            }
            // The observations and masks built from both agree (rotation goes through Euler angles).
            let layout = ObsKind::Native.layout(team_size, true, 0);
            let prev = Controls {
                throttle: 1.0,
                boost: true,
                ..Controls::default()
            };
            for me in 0..snap.cars.len() {
                let (mut from_sim, mut from_packet) =
                    (vec![0.0; layout.size], vec![0.0; layout.size]);
                ObsKind::Native.build(&snap, me, prev, 0, &mut from_sim);
                ObsKind::Native.build(&rebuilt, me, prev, 0, &mut from_packet);
                let worst = from_sim
                    .iter()
                    .zip(&from_packet)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(worst <= 1e-5, "observation differs by {worst}");
                let (mut mask_sim, mut mask_packet) = ([0u8; NUM_ACTIONS], [0u8; NUM_ACTIONS]);
                table.write_mask(MaskRule::Standard, &snap.cars[me], &mut mask_sim);
                table.write_mask(MaskRule::Standard, &rebuilt.cars[me], &mut mask_packet);
                assert_eq!(mask_sim, mask_packet);
            }
        }
    }
}

#[test]
fn a_jump_from_the_wheels_reads_as_on_the_ground_for_its_first_ticks_as_in_training() {
    // RocketSim keeps the car on the ground for the jump's first 6 ticks; RLBot says "jumping" from the
    // first, and "in the air" once a short press ends.
    let mut rng = fastrand::Rng::with_seed(5);
    let pads = PadMap::new(&field()).unwrap();
    let table = ActionTable::new();
    let mut snap = random_snapshot(&mut rng, 2);
    let car = &mut snap.cars[0];
    (
        car.on_ground,
        car.has_jumped,
        car.has_double_jumped,
        car.has_flipped,
    ) = (true, false, false, false);
    (car.can_flip, car.flip_time_left, car.is_demoed) = (true, FLIP_WINDOW_SECONDS, false);
    let mut jumps = JumpStarts::default();
    let mut rebuilt = GameSnapshot::default();
    let mut read = |frame: u32, state: AirState, jumped: bool, jumps: &mut JumpStarts| {
        let mut p = packet(&snap, &pads, frame, MatchPhase::Active);
        let player = &mut p.players[0];
        (player.air_state, player.has_jumped) = (state, jumped);
        player.dodge_timeout = if state == AirState::OnGround {
            -1.0
        } else {
            1.3
        };
        snapshot(&p, &pads, jumps, &mut rebuilt).unwrap();
        rebuilt.cars[0]
    };
    assert!(read(100, AirState::OnGround, false, &mut jumps).on_ground);
    // A 4-tick press (tick skip 4): jumping for 4 packets, then in the air.
    for frame in 101..=112 {
        let state = if frame <= 104 {
            AirState::Jumping
        } else {
            AirState::InAir
        };
        let seen = read(frame, state, true, &mut jumps);
        let ground = frame - 101 < GROUND_JUMP_TICKS;
        assert_eq!(seen.on_ground, ground, "frame {frame}");
        assert!(seen.can_flip && seen.has_jumped);
        let window = if ground { FLIP_WINDOW_SECONDS } else { 1.3 };
        assert_eq!(seen.flip_time_left, window, "frame {frame}");
        if ground {
            // The mask is the one training built for the car still on its wheels.
            let mut sim = snap.cars[0];
            sim.has_jumped = true;
            let (mut want, mut got) = ([0u8; NUM_ACTIONS], [0u8; NUM_ACTIONS]);
            table.write_mask(MaskRule::Standard, &sim, &mut want);
            table.write_mask(MaskRule::Standard, &seen, &mut got);
            assert_eq!(want, got, "frame {frame}");
        }
    }
    // A car first seen mid-jump, or one that dodged, is in the air; a new jump starts a new window.
    let mut fresh = JumpStarts::default();
    assert!(!read(200, AirState::Jumping, true, &mut fresh).on_ground);
    assert!(read(300, AirState::OnGround, false, &mut jumps).on_ground);
    assert!(read(301, AirState::Jumping, true, &mut jumps).on_ground);
    assert!(!read(302, AirState::Dodging, true, &mut jumps).on_ground);
    assert!(!read(303, AirState::InAir, true, &mut jumps).on_ground);
}

#[test]
fn pads_are_mapped_by_position_and_timers_become_cooldowns() {
    let pads = PadMap::new(&field()).unwrap();
    let mut snap = GameSnapshot::default();
    // Snapshot slot 13 is (-2048, -1036) and slot 14 is (0, -1024): RLBot lists them the other way round.
    snap.pads[14] = PadSnapshot {
        active: false,
        cooldown: 3.0,
    };
    let mut rebuilt = GameSnapshot::default();
    let mut p = packet(&snap, &pads, 1, MatchPhase::Active);
    assert_eq!(
        p.boost_pads[13].timer, 1.0,
        "RLBot slot 13 is the small pad at (0, -1024)"
    );
    snapshot(&p, &pads, &mut JumpStarts::default(), &mut rebuilt).unwrap();
    assert_eq!(rebuilt.pads[14].cooldown, 3.0);
    assert!(rebuilt.pads[13].active);
    // A big pad taken 4 s ago comes back in 6 s.
    p.boost_pads[3] = BoostPadState {
        is_active: false,
        timer: 4.0,
    };
    snapshot(&p, &pads, &mut JumpStarts::default(), &mut rebuilt).unwrap();
    assert_eq!(rebuilt.pads[3].cooldown, 6.0);

    let mut short = field();
    short.boost_pads.pop();
    assert!(PadMap::new(&short).is_err());
}

/// A policy file whose only layer maps the observation to logits favouring action `favourite`.
fn policy_bytes(favourite: usize, obs_size: usize) -> Vec<u8> {
    padded_policy_bytes(favourite, obs_size, 0)
}

/// The same, trained with an observation that has places for `room` players a side.
fn padded_policy_bytes(favourite: usize, obs_size: usize, room: usize) -> Vec<u8> {
    policy_file(favourite, obs_size, room, "", NUM_ACTIONS)
}

/// A one-layer policy file over `num_actions` actions favouring `favourite`; `extra_game` adds settings
/// to its header's game section (each ending in a comma).
fn policy_file(
    favourite: usize,
    obs_size: usize,
    room: usize,
    extra_game: &str,
    num_actions: usize,
) -> Vec<u8> {
    let header = format!(
        r#"{{"format": 1, "run": "test", "agent_steps": 7, "game": {{"observation": "native",
        "action_masks": "standard", "team_size": 1, "opponents": true, "obs_team_size": {room},
        {extra_game} "tick_skip": 8, "action_delay": 7}},
        "obs_size": {obs_size}, "num_actions": {num_actions},
        "layers": [{{"kind": "linear", "inputs": {obs_size}, "outputs": {num_actions}}}]}}"#
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(PLAIN_FORMAT.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    for _ in 0..obs_size * num_actions {
        bytes.extend(0.0f32.to_le_bytes());
    }
    for action in 0..num_actions {
        bytes.extend(if action == favourite { 5.0f32 } else { 0.0 }.to_le_bytes());
    }
    bytes
}

#[test]
fn a_policy_with_another_number_of_actions_is_refused() {
    let obs_size = ObsKind::Native.layout(1, true, 0).size;
    let bot = |bytes: &[u8]| {
        NitroBot::new(
            Policy::from_bytes(bytes).unwrap(),
            &field(),
            0,
            Choice::MostLikely,
            1,
            0,
        )
    };
    let wrong = policy_file(0, obs_size, 0, "", NUM_ACTIONS + 16);
    let err = bot(&wrong).err().unwrap();
    assert!(
        err.contains("chooses among 106 actions") && err.contains("has 90"),
        "{err}"
    );
    // Game keys the bot does not read change nothing.
    let extra = r#""an_old_key": {"a": [], "b": ["x"]}, "another_old_key": 0,"#;
    assert!(bot(&policy_file(0, obs_size, 0, extra, NUM_ACTIONS)).is_ok());
}

#[test]
fn the_bot_decides_on_the_training_rhythm_and_coasts_outside_play() {
    let obs_size = ObsKind::Native.layout(1, true, 0).size;
    // Action 0 is a ground action (throttle -1); the car starts on the ground, so it is allowed.
    let policy = Policy::from_bytes(&policy_bytes(0, obs_size)).unwrap();
    let pads = PadMap::new(&field()).unwrap();
    let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 0).unwrap();
    let mut rng = fastrand::Rng::with_seed(2);
    let mut snap = random_snapshot(&mut rng, 2);
    snap.cars[0].on_ground = true;
    snap.cars[0].body.rot = Mat3A::IDENTITY;
    let expected = ActionTable::new().controls(0);

    // Countdown: neutral controls, no decisions.
    assert_eq!(
        bot.controls(&packet(&snap, &pads, 10, MatchPhase::Countdown)),
        Controls::default()
    );
    // Kickoff at tick 20: decision at 20, in effect from tick 27; next at 28.
    let mut sent = vec![];
    for frame in 20..40 {
        sent.push(bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff)));
    }
    assert!(sent[..7].iter().all(|c| *c == Controls::default()));
    assert!(sent[7..].iter().all(|c| *c == expected), "{sent:?}");
    assert_eq!(
        bot.rhythm_stats(),
        RhythmStats {
            decisions: 3,
            late: 0,
            missed: 0
        }
    );
    // A goal replay resets everything; the next live packet decides afresh.
    assert_eq!(
        bot.controls(&packet(&snap, &pads, 41, MatchPhase::Replay)),
        Controls::default()
    );
    assert_eq!(
        bot.controls(&packet(&snap, &pads, 500, MatchPhase::Active)),
        Controls::default()
    );
    assert_eq!(bot.rhythm_stats().decisions, 4);
    // A match with the wrong number of players, or the right number on the wrong teams: the bot coasts
    // instead of guessing.
    let three = random_snapshot(&mut rng, 3);
    assert_eq!(
        bot.controls(&packet(&three, &pads, 600, MatchPhase::Active)),
        Controls::default()
    );
    let mut same_team = random_snapshot(&mut rng, 2);
    same_team.cars[1].team = same_team.cars[0].team;
    assert_eq!(
        bot.controls(&packet(&same_team, &pads, 700, MatchPhase::Active)),
        Controls::default()
    );
    assert_eq!(
        bot.rhythm_stats().decisions,
        4,
        "no decision on a malformed match"
    );
}

#[test]
fn a_latency_makes_the_bot_send_that_many_ticks_early() {
    let obs_size = ObsKind::Native.layout(1, true, 0).size;
    let policy = Policy::from_bytes(&policy_bytes(0, obs_size)).unwrap();
    let pads = PadMap::new(&field()).unwrap();
    let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 1).unwrap();
    let mut rng = fastrand::Rng::with_seed(3);
    let mut snap = random_snapshot(&mut rng, 2);
    snap.cars[0].on_ground = true;
    snap.cars[0].body.rot = Mat3A::IDENTITY;
    let expected = ActionTable::new().controls(0);
    // Decision at tick 20; with a 1-tick latency (`--latency 1`; the game's is 0) the controls go out
    // at 26 and act at 27, as in training (action delay 7).
    let sent: Vec<Controls> = (20..30)
        .map(|frame| bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff)))
        .collect();
    assert!(
        sent[..6].iter().all(|c| *c == Controls::default()),
        "{sent:?}"
    );
    assert!(sent[6..].iter().all(|c| *c == expected), "{sent:?}");
    // The decision is timed against what the delay leaves it: 6 ticks after its packet plus the tick it
    // acts on, here.
    let times = bot.decision_times();
    assert_eq!(times.decisions, bot.rhythm_stats().decisions);
    assert!((times.budget_ms - 7.0 * 1000.0 / 120.0).abs() < 1e-9);
    assert!(times.slowest_ms >= 0.0 && times.total_ms >= times.slowest_ms);
}

#[test]
fn a_padded_policy_plays_every_game_up_to_its_room_and_coasts_in_bigger_ones() {
    let room = 2;
    let obs_size = ObsKind::Native.layout(1, true, room).size;
    let pads = PadMap::new(&field()).unwrap();
    let expected = ActionTable::new().controls(0);
    for (cars, plays) in [(2, true), (4, true), (6, false)] {
        let policy = Policy::from_bytes(&padded_policy_bytes(0, obs_size, room)).unwrap();
        let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 0).unwrap();
        let mut rng = fastrand::Rng::with_seed(4);
        let mut snap = random_snapshot(&mut rng, cars);
        snap.cars[0].on_ground = true;
        snap.cars[0].body.rot = Mat3A::IDENTITY;
        let sent: Vec<Controls> = (20..30)
            .map(|frame| bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff)))
            .collect();
        let acted = sent.contains(&expected);
        assert_eq!(acted, plays, "{cars} cars: {sent:?}");
    }
}

/// An attention policy (format 2) trained with room for `room` players a side: an encoder 4 wide with
/// zero weights, then a linear layer whose bias favours action 0.
fn attention_policy_bytes(room: usize) -> Vec<u8> {
    attention_policy_with(room, None)
}

/// `attention_policy_bytes`, with `seed` every weight random in [-1, 1) instead.
fn attention_policy_with(room: usize, seed: Option<u64>) -> Vec<u8> {
    let layout = ObsKind::Native.layout(1, true, room);
    let (own, block) = (layout.self_part, layout.other_car);
    let linear = |i: usize, o: usize| i * o + o;
    let encoder = linear(own, 4)
        + linear(4, 4)
        + linear(block - 1, 4)
        + linear(4, 4)
        + 2 * 4
        + 4
        + (6 * 4 + 4 * linear(4, 4) + linear(4, 16) + linear(16, 4));
    let header = format!(
        r#"{{"format": 2, "game": {{"observation": "native", "action_masks": "standard", "team_size": 1,
        "opponents": true, "obs_team_size": {room}, "tick_skip": 8, "action_delay": 7}},
        "obs_size": {}, "num_actions": {NUM_ACTIONS}, "layers": [{{"kind": "entity_encoder", "own": {own},
        "block": {block}, "teammates": {}, "opponents": {room}, "width": 4, "steps": 1, "heads": 2}},
        {{"kind": "linear", "inputs": 4, "outputs": {NUM_ACTIONS}}}]}}"#,
        layout.size,
        room - 1
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(2u32.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    if let Some(seed) = seed {
        let mut rng = fastrand::Rng::with_seed(seed);
        for _ in 0..encoder + 5 * NUM_ACTIONS {
            bytes.extend((rng.f32() * 2.0 - 1.0).to_le_bytes());
        }
        return bytes;
    }
    for _ in 0..encoder + 4 * NUM_ACTIONS {
        bytes.extend(0.0f32.to_le_bytes());
    }
    for action in 0..NUM_ACTIONS {
        bytes.extend(if action == 0 { 5.0f32 } else { 0.0 }.to_le_bytes());
    }
    bytes
}

#[test]
fn a_packet_of_any_game_size_gives_the_padded_observation_training_and_nitro_eval_read() {
    // Within its room (2 here) the bot builds the padded observation; beyond it one place per car, which is
    // the padded observation with room for the game's own size, as nitro eval and watch play it.
    let pads = PadMap::new(&field()).unwrap();
    let mut rng = fastrand::Rng::with_seed(12);
    let prev = Controls {
        steer: -0.5,
        jump: true,
        ..Controls::default()
    };
    for kind in [ObsKind::Native, ObsKind::NitroV2] {
        for team_size in 1..=4 {
            let snap = random_snapshot(&mut rng, team_size * 2);
            let mut rebuilt = GameSnapshot::default();
            let p = packet(&snap, &pads, 100, MatchPhase::Active);
            snapshot(&p, &pads, &mut JumpStarts::default(), &mut rebuilt).unwrap();
            let room = team_size.max(2);
            for me in 0..snap.cars.len() {
                let mut sim = vec![0.0; kind.layout(team_size, true, room).size];
                kind.build(&snap, me, prev, room, &mut sim);
                let mut bot = vec![0.0; sim.len()];
                if team_size <= 2 {
                    kind.build(&rebuilt, me, prev, 2, &mut bot);
                } else {
                    let places = kind.build_cars(&rebuilt, me, prev, &mut bot);
                    assert_eq!(places, (team_size - 1, team_size));
                }
                assert_eq!(bot.len(), sim.len(), "{kind:?} {team_size}v{team_size}");
                let worst = sim
                    .iter()
                    .zip(&bot)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= 1e-5,
                    "{kind:?} {team_size}v{team_size}: differs by {worst}"
                );
            }
        }
    }
}

#[test]
fn an_attention_bot_decides_in_every_game_size_as_its_network_does_on_the_padded_observation() {
    let pads = PadMap::new(&field()).unwrap();
    let table = ActionTable::new();
    let mut rng = fastrand::Rng::with_seed(13);
    let mut checked = 0;
    for cars in [2usize, 4, 6, 8] {
        for _ in 0..10 {
            let policy = Policy::from_bytes(&attention_policy_with(2, Some(3))).unwrap();
            let snap = random_snapshot(&mut rng, cars);
            let room = (cars / 2).max(2);
            let mut obs = vec![0.0; ObsKind::Native.layout(1, true, room).size];
            ObsKind::Native.build(&snap, 0, Controls::default(), room, &mut obs);
            let mut logits = vec![0.0; NUM_ACTIONS];
            policy.logits_cars(&obs, room - 1, room, &mut logits);
            let mut mask = [0u8; NUM_ACTIONS];
            table.write_mask(MaskRule::Standard, &snap.cars[0], &mut mask);
            let mut allowed: Vec<f32> = (0..NUM_ACTIONS)
                .filter(|&a| mask[a] == 1)
                .map(|a| logits[a])
                .collect();
            allowed.sort_by(|a, b| b.total_cmp(a));
            if allowed.len() > 1 && allowed[0] - allowed[1] < 1e-3 {
                continue; // a near tie, which the packet's rounding may tip
            }
            let expected = table.controls(Policy::argmax(&logits, &mask));
            let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 0).unwrap();
            let sent: Vec<Controls> = (20..28)
                .map(|frame| bot.controls(&packet(&snap, &pads, frame, MatchPhase::Active)))
                .collect();
            assert_eq!(bot.rhythm_stats().decisions, 1, "{cars} cars");
            assert_eq!(sent[7], expected, "{cars} cars: {sent:?}");
            checked += 1;
        }
    }
    assert!(checked >= 30, "only {checked} decisions checked");
}

#[test]
fn an_attention_policy_plays_every_game_size_even_beyond_its_room() {
    let pads = PadMap::new(&field()).unwrap();
    let expected = ActionTable::new().controls(0);
    // Room for 2 a side: 1v1 and 2v2 as padded, 3v3, 4v4 and RLBot's 12v12 with one place per car.
    for cars in [2, 4, 6, 8, 24] {
        let policy = Policy::from_bytes(&attention_policy_bytes(2)).unwrap();
        assert!(policy.reads_any_cars());
        let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 0).unwrap();
        let mut rng = fastrand::Rng::with_seed(4);
        let mut snap = random_snapshot(&mut rng, cars);
        snap.cars[0].on_ground = true;
        snap.cars[0].body.rot = Mat3A::IDENTITY;
        let sent: Vec<Controls> = (20..30)
            .map(|frame| bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff)))
            .collect();
        assert!(sent.contains(&expected), "{cars} cars: {sent:?}");
        assert_eq!(bot.rhythm_stats().decisions, 2, "{cars} cars");
    }
}

#[test]
fn trained_size_only_coasts_beyond_the_room_and_plays_within_it() {
    let pads = PadMap::new(&field()).unwrap();
    for (cars, plays) in [(4, true), (6, false)] {
        let policy = Policy::from_bytes(&attention_policy_bytes(2)).unwrap();
        let mut bot = NitroBot::new(policy, &field(), 0, Choice::MostLikely, 1, 0).unwrap();
        bot.keep_to_trained_size();
        let mut rng = fastrand::Rng::with_seed(4);
        let mut snap = random_snapshot(&mut rng, cars);
        snap.cars[0].on_ground = true;
        snap.cars[0].body.rot = Mat3A::IDENTITY;
        let sent: Vec<Controls> = (20..30)
            .map(|frame| bot.controls(&packet(&snap, &pads, frame, MatchPhase::Kickoff)))
            .collect();
        let decisions = bot.rhythm_stats().decisions;
        assert_eq!(decisions, if plays { 2 } else { 0 }, "{cars} cars");
        if !plays {
            assert!(
                sent.iter().all(|c| *c == Controls::default()),
                "{cars} cars coast"
            );
        }
    }
}

#[test]
fn a_policy_that_does_not_fit_the_game_is_refused() {
    let wrong_size = Policy::from_bytes(&policy_bytes(0, 50)).unwrap();
    let Err(err) = NitroBot::new(wrong_size, &field(), 0, Choice::MostLikely, 0, 1) else {
        panic!("a policy with the wrong observation size was accepted");
    };
    assert!(err.contains("observation"), "{err}");
}

/// A 2v2 policy of one linear layer with random weights over `num_actions` base actions, on the rhythm
/// `tick_skip` / `action_delay`.
fn random_2v2_policy(tick_skip: u32, action_delay: u32, seed: u64) -> Policy {
    let obs_size = ObsKind::Native.layout(2, true, 0).size;
    let header = format!(
        r#"{{"format": 1, "run": "test", "agent_steps": 7, "game": {{"observation": "native",
        "action_masks": "standard", "team_size": 2, "opponents": true, "tick_skip": {tick_skip},
        "action_delay": {action_delay}}}, "obs_size": {obs_size}, "num_actions": {NUM_ACTIONS},
        "layers": [{{"kind": "linear", "inputs": {obs_size}, "outputs": {NUM_ACTIONS}}}]}}"#
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(PLAIN_FORMAT.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    let mut rng = fastrand::Rng::with_seed(seed);
    for _ in 0..(obs_size + 1) * NUM_ACTIONS {
        bytes.extend((rng.f32() * 2.0 - 1.0).to_le_bytes());
    }
    Policy::from_bytes(&bytes).unwrap()
}

#[test]
fn the_decider_sends_every_packet_what_deciding_on_the_packet_thread_sends() {
    use nitro_bot::decider::Decider;
    use std::sync::Arc;
    let pads = PadMap::new(&field()).unwrap();
    // Kickoff with dropped and repeated packets, a goal replay, live play again, a malformed match, play.
    let mut script: Vec<(u32, MatchPhase, usize)> = Vec::new();
    for frame in (20..60).filter(|f| f % 7 != 3) {
        script.push((frame, MatchPhase::Kickoff, 4));
    }
    script.push((59, MatchPhase::Kickoff, 4));
    script.push((61, MatchPhase::Replay, 4));
    script.extend((100..140).map(|frame| (frame, MatchPhase::Active, 4)));
    script.push((141, MatchPhase::Active, 6));
    script.extend((142..170).map(|frame| (frame, MatchPhase::Active, 4)));
    for (tick_skip, delay) in [(4, 1), (4, 0), (8, 7), (4, 3)] {
        for choice in [Choice::MostLikely, Choice::Sample] {
            let policy = Arc::new(random_2v2_policy(tick_skip, delay, 9));
            let decider = Decider::start(policy.clone(), 2);
            // The two Blue cars of a hivemind (indices 0 and 2), each deciding both ways.
            let make = |index: usize| {
                NitroBot::new(policy.clone(), &field(), index, choice, 5 + index as u64, 0).unwrap()
            };
            let mut here = [make(0), make(2)];
            let mut there = [make(0), make(2)];
            for bot in &mut there {
                bot.use_decider(decider.seat());
            }
            let mut rng = fastrand::Rng::with_seed(8);
            for &(frame, phase, cars) in &script {
                let snap = random_snapshot(&mut rng, cars);
                let p = packet(&snap, &pads, frame, phase);
                for (a, b) in here.iter_mut().zip(&mut there) {
                    assert_eq!(
                        a.controls(&p),
                        b.controls(&p),
                        "{tick_skip}/{delay} {choice:?} frame {frame}"
                    );
                    assert_eq!(a.decided(), b.decided());
                }
            }
            for (a, b) in here.iter().zip(&there) {
                assert_eq!(a.rhythm_stats(), b.rhythm_stats());
                assert!(a.rhythm_stats().decisions > 10);
                let times = b.decision_times();
                // Every decision was timed but those dropped in flight at the replay and the malformed
                // packet, and the one in flight at the end.
                assert!(times.decisions + 3 >= a.decision_times().decisions);
                assert!(times.decisions <= a.decision_times().decisions);
            }
        }
    }
}
