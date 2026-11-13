//! RLBot v5 `GamePacket` → `GameSnapshot`, the state training builds its observations from.
//! Every field has a documented packet source, so the observation and the action mask
//! the bot computes are the ones the policy saw in training.

use nitro_core::custom::prelude::Vec3A;
use nitro_core::snapshot::{
    Body, CarSnapshot, FLIP_WINDOW_SECONDS, GameSnapshot, NUM_BOOST_PADS, PadSnapshot, Team,
    window_open,
};
use rlbot::flat::{AirState, FieldInfo, GamePacket, Physics, PlayerInfo};

use crate::angles::{rotation_matrix, vec3};

/// A big pad gives a full tank and comes back after 10 s, a small one after 4 s (RLBot schema).
const BIG_PAD_RESPAWN: f32 = 10.0;
const SMALL_PAD_RESPAWN: f32 = 4.0;

/// Where each of RLBot's boost pads sits in the snapshot's order (by y, then x), and how long it takes
/// to come back: RLBot's `timer` counts up since the pickup, the snapshot's `cooldown` counts down.
#[derive(Clone, Debug)]
pub struct PadMap {
    slot: Vec<usize>,
    respawn: Vec<f32>,
}

impl PadMap {
    pub fn new(field: &FieldInfo) -> Result<Self, String> {
        let pads = &field.boost_pads;
        if pads.len() != NUM_BOOST_PADS {
            return Err(format!(
                "the field has {} boost pads; the policy was trained on a standard soccar field with \
                 {NUM_BOOST_PADS}",
                pads.len()
            ));
        }
        let mut order: Vec<usize> = (0..pads.len()).collect();
        order.sort_by(|&a, &b| {
            let (pa, pb) = (pads[a].location, pads[b].location);
            pa.y.total_cmp(&pb.y).then(pa.x.total_cmp(&pb.x))
        });
        let mut slot = vec![0; pads.len()];
        for (rank, &index) in order.iter().enumerate() {
            slot[index] = rank;
        }
        let respawn = pads
            .iter()
            .map(|pad| {
                if pad.is_full_boost {
                    BIG_PAD_RESPAWN
                } else {
                    SMALL_PAD_RESPAWN
                }
            })
            .collect();
        Ok(Self { slot, respawn })
    }
}

fn body(physics: &Physics) -> Body {
    let rotation = &physics.rotation;
    Body {
        pos: vec3(physics.location),
        rot: rotation_matrix(rotation.pitch, rotation.yaw, rotation.roll),
        vel: vec3(physics.velocity),
        ang_vel: vec3(physics.angular_velocity),
    }
}

/// Ticks RocketSim still counts a car on the ground after its jump starts from its wheels, which keep
/// touching while the jump lifts it: 6 on 90% of 2,485 random jumps, 5–7 on 95%, however long the jump
/// is held. RLBot reports "jumping" from the first tick.
pub const GROUND_JUMP_TICKS: u32 = 6;

/// When each car's last jump left the ground, so the bot reads that jump's first `GROUND_JUMP_TICKS`
/// as on the ground, as training did.
#[derive(Clone, Debug, Default)]
pub struct JumpStarts {
    /// Per player: the air state in the last packet, and the frame its jump from the ground started.
    cars: Vec<(AirState, Option<u32>)>,
}

impl JumpStarts {
    /// Whether `player` (the packet's player `index`) is on the ground at `frame` as RocketSim counts it.
    fn on_ground(&mut self, index: usize, player: &PlayerInfo, frame: u32) -> bool {
        if self.cars.len() <= index {
            self.cars.resize(index + 1, (AirState::InAir, None));
        }
        let (last, start) = &mut self.cars[index];
        let state = player.air_state;
        *start = match state {
            AirState::Jumping if *last == AirState::OnGround => Some(frame),
            // The jump's force can end before the wheels leave (a short press), which RLBot shows as in air.
            AirState::Jumping | AirState::InAir => {
                start.filter(|&s| frame >= s && frame - s < GROUND_JUMP_TICKS)
            }
            _ => None,
        };
        *last = state;
        state == AirState::OnGround || start.is_some()
    }
}

/// A car as training sees it; the derivations are the ones documented on `CarSnapshot`.
pub fn car(player: &PlayerInfo, on_ground: bool) -> CarSnapshot {
    let (has_jumped, has_double_jumped, has_flipped) = (
        player.has_jumped,
        player.has_double_jumped,
        player.has_dodged,
    );
    // RocketSim's has_flip_or_jump: on the ground, or airborne with neither used and the jump window
    // still open; a car that fell off a surface never jumped, so its window never closes.
    let can_flip = on_ground
        || (!has_flipped
            && !has_double_jumped
            && (!has_jumped || window_open(player.dodge_timeout)));
    CarSnapshot {
        team: if player.team == 0 {
            Team::Blue
        } else {
            Team::Orange
        },
        body: body(&player.physics),
        boost: player.boost,
        on_ground,
        has_jumped,
        has_double_jumped,
        has_flipped,
        can_flip,
        // -1 while the car is alive.
        is_demoed: player.demolished_timeout >= 0.0,
        is_supersonic: player.is_supersonic,
        demo_timer: player.demolished_timeout.max(0.0),
        // RLBot's dodge_timeout is the window's remaining time while a first jump has it running.
        flip_time_left: if can_flip && !on_ground && has_jumped {
            player.dodge_timeout.max(0.0)
        } else if can_flip {
            FLIP_WINDOW_SECONDS
        } else {
            0.0
        },
    }
}

/// Fills `out` from `packet`; `Err` when the packet cannot stand for a game the policy knows. `jumps`
/// carries the cars' jump starts from packet to packet.
pub fn snapshot(
    packet: &GamePacket,
    pads: &PadMap,
    jumps: &mut JumpStarts,
    out: &mut GameSnapshot,
) -> Result<(), String> {
    let ball = packet
        .balls
        .first()
        .ok_or_else(|| "the packet has no ball".to_owned())?;
    out.ball = body(&ball.physics);
    out.cars.clear();
    let frame = packet.match_info.frame_num;
    jumps.cars.truncate(packet.players.len());
    out.cars.extend(
        packet
            .players
            .iter()
            .enumerate()
            .map(|(index, player)| car(player, jumps.on_ground(index, player, frame))),
    );
    if packet.boost_pads.len() != pads.slot.len() {
        return Err(format!(
            "the packet lists {} boost pads, the field {}",
            packet.boost_pads.len(),
            pads.slot.len()
        ));
    }
    for (index, state) in packet.boost_pads.iter().enumerate() {
        out.pads[pads.slot[index]] = PadSnapshot {
            active: state.is_active,
            cooldown: if state.is_active {
                0.0
            } else {
                (pads.respawn[index] - state.timer).max(0.0)
            },
        };
    }
    // Training-only (rewards); the bot never reads it.
    out.goal_scored = None;
    Ok(())
}

/// Sanity check for debugging: the snapshot's ball and cars are finite.
pub fn is_finite(snap: &GameSnapshot) -> bool {
    let finite = |v: Vec3A| v.is_finite();
    finite(snap.ball.pos)
        && finite(snap.ball.vel)
        && snap.cars.iter().all(|c| {
            finite(c.body.pos)
                && finite(c.body.vel)
                && finite(c.body.ang_vel)
                && c.boost.is_finite()
        })
}
