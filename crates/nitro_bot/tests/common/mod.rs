//! Synthesised RLBot v5 packets for the bot's tests: a standard field, and a packet showing any simulated
//! `GameSnapshot` at a given tick (the inverse of `nitro_bot::packet::snapshot`).
#![allow(dead_code)]

use nitro_bot::angles::{rotation_matrix, rotator};
use nitro_bot::packet::PadMap;
use nitro_core::custom::prelude::Vec3A;
use nitro_core::snapshot::{Body, CarSnapshot, GameSnapshot, NUM_BOOST_PADS, PadSnapshot, Team};
use rlbot::flat::{
    AirState, BallInfo, BoostPad, BoostPadState, BoxShape, CollisionShape, ControllerState,
    FieldInfo, GamePacket, MatchInfo, MatchPhase, Physics, PlayerInfo, Rotator, ScoreInfo,
    SphereShape, Vector2, Vector3,
};
use std::f32::consts::PI;

/// Rocket League's boost pads in RLBot's order (two pairs differ from the by-position order).
pub const RLBOT_PADS: [(f32, f32); NUM_BOOST_PADS] = [
    (0.0, -4240.0),
    (-1792.0, -4184.0),
    (1792.0, -4184.0),
    (-3072.0, -4096.0),
    (3072.0, -4096.0),
    (-940.0, -3308.0),
    (940.0, -3308.0),
    (0.0, -2816.0),
    (-3584.0, -2484.0),
    (3584.0, -2484.0),
    (-1788.0, -2300.0),
    (1788.0, -2300.0),
    (-2048.0, -1036.0),
    (0.0, -1024.0),
    (2048.0, -1036.0),
    (-3584.0, 0.0),
    (-1024.0, 0.0),
    (1024.0, 0.0),
    (3584.0, 0.0),
    (-2048.0, 1036.0),
    (0.0, 1024.0),
    (2048.0, 1036.0),
    (-1788.0, 2300.0),
    (1788.0, 2300.0),
    (-3584.0, 2484.0),
    (3584.0, 2484.0),
    (0.0, 2816.0),
    (-940.0, 3308.0),
    (940.0, 3308.0),
    (-3072.0, 4096.0),
    (3072.0, 4096.0),
    (-1792.0, 4184.0),
    (1792.0, 4184.0),
    (0.0, 4240.0),
]; // fmt: skip

pub fn is_big(x: f32, y: f32) -> bool {
    (x.abs() == 3072.0 && y.abs() == 4096.0) || (x.abs() == 3584.0 && y == 0.0)
}

pub fn field() -> FieldInfo {
    FieldInfo {
        boost_pads: RLBOT_PADS
            .iter()
            .map(|&(x, y)| BoostPad {
                location: Vector3 { x, y, z: 0.0 },
                is_full_boost: is_big(x, y),
            })
            .collect(),
        goals: vec![],
        tiles: vec![],
    }
}

pub fn vec3(v: Vec3A) -> Vector3 {
    Vector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

pub fn physics(body: &Body) -> Physics {
    let (pitch, yaw, roll) = rotator(body.rot);
    Physics {
        location: vec3(body.pos),
        rotation: Rotator { pitch, yaw, roll },
        velocity: vec3(body.vel),
        angular_velocity: vec3(body.ang_vel),
    }
}

pub fn player(car: &CarSnapshot) -> PlayerInfo {
    PlayerInfo {
        physics: physics(&car.body),
        score_info: ScoreInfo::default(),
        // Octane, as RLBot reports it (RocketSim adds its 2.5 uu collision margin).
        hitbox: Box::new(BoxShape {
            length: 118.0074,
            width: 84.1994,
            height: 36.1591,
        }),
        hitbox_offset: Vector3 {
            x: 13.8757,
            y: 0.0,
            z: 20.755,
        },
        latest_touch: None,
        air_state: if car.on_ground {
            AirState::OnGround
        } else {
            AirState::InAir
        },
        // The window's remaining time while a jump was used and nothing else; -1 otherwise.
        dodge_timeout: if !car.on_ground && car.has_jumped && car.can_flip {
            car.flip_time_left
        } else {
            -1.0
        },
        demolished_timeout: if car.is_demoed { car.demo_timer } else { -1.0 },
        is_supersonic: car.is_supersonic,
        is_bot: true,
        name: "test".into(),
        team: car.team.index() as u32,
        boost: car.boost,
        player_id: 0,
        accolades: vec![],
        last_input: ControllerState::default(),
        has_jumped: car.has_jumped,
        has_double_jumped: car.has_double_jumped,
        has_dodged: car.has_flipped,
        dodge_elapsed: 0.0,
        dodge_dir: Vector2 { x: 0.0, y: 0.0 },
        rumble_item: None,
        time_until_next_item: 0.0,
        max_time_until_next_item: 0.0,
    }
}

/// A packet showing `snap` at tick `frame`; the snapshot's pads (by position) go to RLBot's slots.
pub fn packet(snap: &GameSnapshot, pads: &PadMap, frame: u32, phase: MatchPhase) -> GamePacket {
    let field = field();
    let mut boost_pads = vec![BoostPadState::default(); NUM_BOOST_PADS];
    for (index, pad) in field.boost_pads.iter().enumerate() {
        // Which snapshot slot this RLBot pad is: rank by (y, x), as PadMap does.
        let slot = field
            .boost_pads
            .iter()
            .filter(|other| {
                other.location.y < pad.location.y
                    || (other.location.y == pad.location.y && other.location.x < pad.location.x)
            })
            .count();
        let state = snap.pads[slot];
        let respawn = if pad.is_full_boost { 10.0 } else { 4.0 };
        boost_pads[index] = BoostPadState {
            is_active: state.active,
            timer: if state.active {
                0.0
            } else {
                respawn - state.cooldown
            },
        };
    }
    let _ = pads;
    GamePacket {
        players: snap.cars.iter().map(player).collect(),
        boost_pads,
        balls: vec![BallInfo {
            physics: physics(&snap.ball),
            shape: CollisionShape::SphereShape(Box::new(SphereShape { diameter: 182.5 })),
            charge_level: 0,
            target_speed: 0.0,
        }],
        match_info: Box::new(MatchInfo {
            seconds_elapsed: frame as f32 / 120.0,
            game_time_remaining: 300.0,
            is_overtime: false,
            is_unlimited_time: false,
            match_phase: phase,
            world_gravity_z: -650.0,
            game_speed: 1.0,
            last_spectated: 0,
            frame_num: frame,
        }),
        teams: vec![],
        tiles: vec![],
    }
}

pub fn random_body(rng: &mut fastrand::Rng, scale: f32) -> Body {
    let r = |rng: &mut fastrand::Rng| rng.f32() * 2.0 - 1.0;
    Body {
        pos: Vec3A::new(r(rng) * 4000.0, r(rng) * 5000.0, 17.0 + rng.f32() * 1500.0),
        rot: rotation_matrix(r(rng) * 1.5, r(rng) * PI, r(rng) * PI),
        vel: Vec3A::new(r(rng), r(rng), r(rng)) * scale,
        ang_vel: Vec3A::new(r(rng), r(rng), r(rng)) * 5.0,
    }
}

pub fn random_snapshot(rng: &mut fastrand::Rng, cars: usize) -> GameSnapshot {
    let mut snap = GameSnapshot {
        ball: random_body(rng, 3000.0),
        ..GameSnapshot::default()
    };
    for i in 0..cars {
        let on_ground = rng.bool();
        let has_jumped = !on_ground && rng.bool();
        let has_double_jumped = has_jumped && rng.bool();
        let has_flipped = has_jumped && !has_double_jumped && rng.bool();
        // RocketSim's rule, with the jump window open or closed at random.
        let can_flip =
            on_ground || (!has_flipped && !has_double_jumped && (!has_jumped || rng.bool()));
        let is_demoed = rng.u32(..10) == 0;
        snap.cars.push(CarSnapshot {
            team: if i % 2 == 0 { Team::Blue } else { Team::Orange },
            body: random_body(rng, 2300.0),
            boost: rng.f32() * 100.0,
            on_ground,
            has_jumped,
            has_double_jumped,
            has_flipped,
            can_flip,
            is_demoed,
            is_supersonic: rng.bool(),
            demo_timer: if is_demoed {
                0.1 + rng.f32() * 2.9
            } else {
                0.0
            },
            flip_time_left: CarSnapshot::flip_time_left_from(
                on_ground,
                has_jumped,
                can_flip,
                rng.f32() * 1.2,
            ),
        });
    }
    for pad in &mut snap.pads {
        if rng.u32(..4) == 0 {
            *pad = PadSnapshot {
                active: false,
                cooldown: rng.f32() * 4.0,
            };
        }
    }
    snap
}
