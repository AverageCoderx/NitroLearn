//! Observation builders. Both read only `CarSnapshot`/ball/pad fields, so the RLBot bot builds the
//! exact same vectors from a `GamePacket`.

use rocketsim::{Mat3A, Vec3A};
use serde::Deserialize;

use crate::snapshot::{Body, CarSnapshot, Controls, GameSnapshot, MAX_CARS, NUM_BOOST_PADS, Team};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsKind {
    /// NitroLearn's own observation.
    #[default]
    Native,
    /// The native observation plus the relative, goal-aware and scalar features strong bots use:
    /// 148 values in 1v1. A setting, unproven until compared.
    NitroV2,
}

/// Where each part of a native observation sits, so the network can treat other cars as a set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObsLayout {
    pub size: usize,
    /// Ball, pads, own previous action, own car, own egocentric extras.
    pub self_part: usize,
    /// Size of each other-car block; teammates come first, then opponents.
    pub other_car: usize,
    /// Places for teammates and for opponents (with `room`, filled up with empty places).
    pub teammates: usize,
    pub opponents: usize,
    /// Players per team the observation has places for; 0 = exactly the game's (no padding).
    pub room: usize,
}

const NATIVE_CAR: usize = 23;
const NATIVE_SELF_EXTRA: usize = 9;
const NATIVE_OTHER: usize = NATIVE_CAR + 6;
const NATIVE_GLOBAL: usize = 9 + NUM_BOOST_PADS + 8;
// v2: ball 9 + speed 1 + on-floor flag 1 + to each goal (vector 3 + distance 1) × 2 + speed toward goal 1.
const V2_BALL: usize = 20;
// v2 car: pos, forward, up, vel, ang vel (15) + speed 1 + boost 1 + 7 flags + demo timer 1 + flip window 1.
const V2_CAR: usize = 26;
// v2 own extras: local ang vel 3, local vel 3, ball rel pos 3, ball direction 3, ball rel vel 3, ball distance 1,
// both goals in the car's frame 3 + 3.
const V2_SELF_EXTRA: usize = 22;
// v2 other-car extras: rel pos 3, rel vel 3, distance 1 (car frame); to ball 3, speed toward ball 1, ball distance 1.
const V2_OTHER_EXTRA: usize = 12;
const V2_GLOBAL: usize = V2_BALL + NUM_BOOST_PADS + 8;

impl ObsKind {
    /// Layout for `team_size` cars per team, with or without an opposing team. With `room` (players per
    /// team, at least `team_size`; 0 = none), there are places for `room - 1` teammates and `room`
    /// opponents whatever the game, each other-car block ends in a "car present" flag, and empty places
    /// are zeros: one network fits every game up to `room` a side, the one-car practice tasks included
    ///.
    pub fn layout(self, team_size: usize, has_opponents: bool, room: usize) -> ObsLayout {
        let (self_part, base_other) = self.parts();
        let (teammates, opponents, other_car) = if room == 0 {
            let opponents = if has_opponents { team_size } else { 0 };
            (team_size - 1, opponents, base_other)
        } else {
            (room - 1, room, base_other + 1)
        };
        ObsLayout {
            size: self_part + other_car * (teammates + opponents),
            self_part,
            other_car,
            teammates,
            opponents,
            room,
        }
    }

    /// The own part's size and an unflagged other-car block's.
    fn parts(self) -> (usize, usize) {
        match self {
            Self::Native => (NATIVE_GLOBAL + NATIVE_CAR + NATIVE_SELF_EXTRA, NATIVE_OTHER),
            Self::NitroV2 => (V2_GLOBAL + V2_CAR + V2_SELF_EXTRA, V2_CAR + V2_OTHER_EXTRA),
        }
    }

    /// The layout with exactly `teammates` and `opponents` flagged car places and none empty (`build_cars`):
    /// what an attention policy reads in a game bigger than its room.
    pub fn cars_layout(self, teammates: usize, opponents: usize) -> ObsLayout {
        let (self_part, base_other) = self.parts();
        ObsLayout {
            size: self_part + (base_other + 1) * (teammates + opponents),
            self_part,
            other_car: base_other + 1,
            teammates,
            opponents,
            room: (teammates + 1).max(opponents),
        }
    }

    /// `layout` for settings that may be wrong: the same rules as the core's settings check, and only
    /// these, so a saved network's layout never depends on the rest of a settings file.
    pub fn checked_layout(
        self,
        team_size: usize,
        has_opponents: bool,
        room: usize,
    ) -> Result<ObsLayout, String> {
        if team_size == 0 || team_size * 2 > crate::snapshot::MAX_CARS {
            return Err("team_size must be 1 to 4".to_owned());
        }
        if room != 0 && (room < team_size || room * 2 > crate::snapshot::MAX_CARS) {
            return Err("obs_team_size must be 0 or from team_size to 4".to_owned());
        }
        Ok(self.layout(team_size, has_opponents, room))
    }

    /// The kind a settings file names (`env.observation`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "native" => Some(Self::Native),
            "nitro_v2" => Some(Self::NitroV2),
            _ => None,
        }
    }

    /// Writes the observation of car `me` with places for `room` players a side (0 = the game's own);
    /// `out.len()` must equal `layout(.., room).size`.
    pub fn build(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        room: usize,
        out: &mut [f32],
    ) {
        self.build_slotted(snap, me, prev_action, room, &[], out);
    }

    /// `build` with each car at place `slots[car]` among its team's `room` places (`env.shuffle_slots`);
    /// a car's teammates keep their places' order with its own left out. Empty `slots` = the cars in
    /// order, as `build`.
    pub fn build_slotted(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        room: usize,
        slots: &[u8],
        out: &mut [f32],
    ) {
        assert!(
            slots.is_empty() || room > 0,
            "slots need a padded observation"
        );
        self.write(
            snap,
            me,
            prev_action,
            (room > 0).then(|| (room - 1, room)),
            slots,
            out,
        );
    }

    /// Writes the observation of car `me` with one flagged block per other car in the game and no empty
    /// places (`cars_layout`), into `out` resized to fit; returns the (teammates, opponents) written. For a
    /// policy whose attention encoder reads any number of cars, in a game bigger than its room (any size,
    /// e.g. RLBot's 12v12).
    pub fn build_cars(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        out: &mut Vec<f32>,
    ) -> (usize, usize) {
        let team = snap.cars[me].team;
        let teammates = snap.cars.iter().filter(|car| car.team == team).count() - 1;
        let opponents = snap.cars.len() - 1 - teammates;
        out.clear();
        out.resize(self.cars_layout(teammates, opponents).size, 0.0);
        self.write(
            snap,
            me,
            prev_action,
            Some((teammates, opponents)),
            &[],
            out,
        );
        (teammates, opponents)
    }

    fn write(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        places: Places,
        slots: &[u8],
        out: &mut [f32],
    ) {
        let mut writer = Writer { out, len: 0 };
        let places = (places, slots);
        match self {
            Self::Native => native(snap, me, prev_action, places, &mut writer),
            Self::NitroV2 => nitro_v2(snap, me, prev_action, places, &mut writer),
        }
        assert_eq!(writer.len, writer.out.len(), "observation size mismatch");
    }
}

/// The other cars' places: `None` = exactly the game's, unflagged (unpadded); `Some((teammates, opponents))`
/// = that many places a side, each block ending in a 1/0 "car present" flag, the empty ones zeros.
type Places = Option<(usize, usize)>;

struct Writer<'a> {
    out: &'a mut [f32],
    len: usize,
}

impl Writer<'_> {
    fn scalar(&mut self, value: f32) {
        self.out[self.len] = value;
        self.len += 1;
    }

    fn vec(&mut self, value: Vec3A) {
        self.out[self.len..self.len + 3].copy_from_slice(&value.to_array());
        self.len += 3;
    }

    fn flag(&mut self, value: bool) {
        self.scalar(f32::from(u8::from(value)));
    }

    fn zeros(&mut self, count: usize) {
        self.out[self.len..self.len + count].fill(0.0);
        self.len += count;
    }
}

// Orange sees the field rotated 180° about the vertical axis, so both teams attack +Y.
const TEAM_FLIP: Vec3A = Vec3A::new(-1.0, -1.0, 1.0);

fn team_view(body: Body, team: Team) -> Body {
    if team == Team::Blue {
        return body;
    }
    Body {
        pos: body.pos * TEAM_FLIP,
        rot: Mat3A::from_cols(
            body.rot.x_axis * TEAM_FLIP,
            body.rot.y_axis * TEAM_FLIP,
            body.rot.z_axis * TEAM_FLIP,
        ),
        vel: body.vel * TEAM_FLIP,
        ang_vel: body.ang_vel * TEAM_FLIP,
    }
}

/// Components of `v` along the car's forward, right and up axes.
fn local(rot: Mat3A, v: Vec3A) -> Vec3A {
    rot.transpose() * v
}

/// The pad that sits where pad `i` would be after the 180° team flip (pads are ordered by y then x).
const fn flipped_pad(i: usize) -> usize {
    NUM_BOOST_PADS - 1 - i
}

/// Pads sorted by y then x come in rows of equal y; mirroring the field left-right (x → −x) reverses each
/// row. Row sizes of the 34 soccar pads, in order of y (checked against the simulator in sim.rs's tests).
const PAD_ROWS: [usize; 19] = [1, 2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 2, 2, 1, 2, 2, 2, 1];

/// The pad at (−x, y) for the pad at (x, y).
pub const fn mirrored_pad(i: usize) -> usize {
    let mut start = 0;
    let mut row = 0;
    while row < PAD_ROWS.len() {
        let size = PAD_ROWS[row];
        if i < start + size {
            return start + (start + size - 1 - i);
        }
        start += size;
        row += 1;
    }
    i
}

fn pad_value(active: bool, cooldown: f32) -> f32 {
    if active { 1.0 } else { 1.0 / (1.0 + cooldown) }
}

fn other_cars(snap: &GameSnapshot, me: usize) -> impl Iterator<Item = &CarSnapshot> {
    let team = snap.cars[me].team;
    let others = snap
        .cars
        .iter()
        .enumerate()
        .filter(move |&(i, _)| i != me)
        .map(|(_, car)| car);
    others
        .clone()
        .filter(move |car| car.team == team)
        .chain(others.filter(move |car| car.team != team))
}

/// Writes every other car's `block`-long entry with `write`, teammates first. With `places`, each entry
/// ends in a "car present" flag and each team's places are filled up with zeros to that many; with slots
/// too, each car sits at its slot's place (`ObsKind::build_slotted`).
fn write_others(
    snap: &GameSnapshot,
    me: usize,
    (places, slots): (Places, &[u8]),
    block: usize,
    w: &mut Writer<'_>,
    mut write: impl FnMut(&CarSnapshot, &mut Writer<'_>),
) {
    let Some((teammate_places, opponent_places)) = places else {
        for car in other_cars(snap, me) {
            write(car, w);
        }
        return;
    };
    let team = snap.cars[me].team;
    for (teammates, places) in [(true, teammate_places), (false, opponent_places)] {
        let cars = snap.cars.iter().enumerate();
        let cars = cars.filter(|&(i, car)| i != me && (car.team == team) == teammates);
        if slots.is_empty() {
            let mut used = 0;
            for (_, car) in cars {
                write(car, w);
                w.flag(true);
                used += 1;
            }
            assert!(
                used <= places,
                "the game has more cars than the observation has room for"
            );
            w.zeros((places - used) * (block + 1));
            continue;
        }
        let mut at: [Option<&CarSnapshot>; MAX_CARS] = [None; MAX_CARS];
        for (i, car) in cars {
            let slot = usize::from(slots[i]);
            // A teammate's place among the room - 1 left once the car's own slot is taken out.
            let place = if teammates && slot > usize::from(slots[me]) {
                slot - 1
            } else {
                slot
            };
            assert!(
                place < places && at[place].is_none(),
                "car slots must be distinct places within the observation's room"
            );
            at[place] = Some(car);
        }
        for car in &at[..places] {
            match car {
                Some(car) => {
                    write(car, w);
                    w.flag(true);
                }
                None => w.zeros(block + 1),
            }
        }
    }
}

mod native_scale {
    use rocketsim::Vec3A;
    pub const POS: Vec3A = Vec3A::new(1.0 / 4096.0, 1.0 / 5120.0, 1.0 / 2044.0);
    pub const REL_POS: f32 = 1.0 / 2300.0;
    pub const VEL: f32 = 1.0 / 2300.0;
    pub const CAR_ANG_VEL: f32 = 1.0 / 5.5;
    pub const BALL_ANG_VEL: f32 = 1.0 / 6.0;
    pub const BOOST: f32 = 1.0 / 100.0;
}

fn native(
    snap: &GameSnapshot,
    me: usize,
    prev_action: Controls,
    places: (Places, &[u8]),
    w: &mut Writer<'_>,
) {
    use native_scale as s;
    let my = &snap.cars[me];
    let team = my.team;
    let ball = team_view(snap.ball, team);
    w.vec(ball.pos * s::POS);
    w.vec(ball.vel * s::VEL);
    w.vec(ball.ang_vel * s::BALL_ANG_VEL);
    for i in 0..NUM_BOOST_PADS {
        let pad = snap.pads[if team == Team::Blue {
            i
        } else {
            flipped_pad(i)
        }];
        w.scalar(pad_value(pad.active, pad.cooldown));
    }
    for value in prev_action.to_floats() {
        w.scalar(value);
    }

    let me_body = team_view(my.body, team);
    native_car(my, me_body, w);
    w.vec(local(me_body.rot, me_body.ang_vel) * s::CAR_ANG_VEL);
    w.vec(local(me_body.rot, ball.pos - me_body.pos) * s::REL_POS);
    w.vec(local(me_body.rot, ball.vel - me_body.vel) * s::VEL);

    write_others(snap, me, places, NATIVE_OTHER, w, |car, w| {
        let body = team_view(car.body, team);
        native_car(car, body, w);
        w.vec(local(me_body.rot, body.pos - me_body.pos) * s::REL_POS);
        w.vec(local(me_body.rot, body.vel - me_body.vel) * s::VEL);
    });
}

fn native_car(car: &CarSnapshot, body: Body, w: &mut Writer<'_>) {
    use native_scale as s;
    w.vec(body.pos * s::POS);
    w.vec(body.forward());
    w.vec(body.up());
    w.vec(body.vel * s::VEL);
    w.vec(body.ang_vel * s::CAR_ANG_VEL);
    w.scalar(car.boost * s::BOOST);
    w.flag(car.on_ground);
    w.flag(car.has_jumped);
    w.flag(car.has_double_jumped);
    w.flag(car.has_flipped);
    w.flag(car.can_flip);
    w.flag(car.is_demoed);
    w.flag(car.is_supersonic);
}

mod v2_scale {
    pub use super::native_scale::{BALL_ANG_VEL, BOOST, CAR_ANG_VEL, POS, REL_POS, VEL};
    use rocketsim::Vec3A;
    /// Distances and goal vectors: the goal line is 5120 away from the centre.
    pub const DIST: f32 = 1.0 / 5120.0;
    /// A demolished car is back after 3 s; the flip window lasts 1.25 s.
    pub const DEMO_TIMER: f32 = 1.0 / 3.0;
    pub const FLIP_WINDOW: f32 = 1.0 / crate::snapshot::FLIP_WINDOW_SECONDS;
    /// The ball rests at z ≈ 93; below this it is "on the floor" (Opti's idea).
    pub const BALL_ON_FLOOR_Z: f32 = 100.0;
    /// Goal centres in the team view (Blue attacks +Y), at half the goal height.
    pub const OPPONENT_GOAL: Vec3A = Vec3A::new(0.0, 5120.0, 642.775 / 2.0);
    pub const OWN_GOAL: Vec3A = Vec3A::new(0.0, -5120.0, 642.775 / 2.0);
}

/// Speed along the direction to `target` (uu/s), 0 when already there.
fn speed_toward(vel: Vec3A, from: Vec3A, target: Vec3A) -> f32 {
    vel.dot((target - from).normalize_or_zero())
}

/// The second observation. Everything in the team view, then the
/// agent's own frame for what concerns aiming; every feature comes from `CarSnapshot`, ball and pad
/// fields, so the RLBot bot builds the same vector.
fn nitro_v2(
    snap: &GameSnapshot,
    me: usize,
    prev_action: Controls,
    places: (Places, &[u8]),
    w: &mut Writer<'_>,
) {
    use v2_scale as s;
    let my = &snap.cars[me];
    let team = my.team;
    let ball = team_view(snap.ball, team);
    // Ball: where it is and moves, how fast, whether it is on the floor, and how it stands to both goals.
    w.vec(ball.pos * s::POS);
    w.vec(ball.vel * s::VEL);
    w.vec(ball.ang_vel * s::BALL_ANG_VEL);
    w.scalar(ball.vel.length() * s::VEL);
    w.flag(ball.pos.z < s::BALL_ON_FLOOR_Z);
    let to_opponent_goal = s::OPPONENT_GOAL - ball.pos;
    w.vec(to_opponent_goal * s::DIST);
    w.scalar(to_opponent_goal.length() * s::DIST);
    let to_own_goal = s::OWN_GOAL - ball.pos;
    w.vec(to_own_goal * s::DIST);
    w.scalar(to_own_goal.length() * s::DIST);
    w.scalar(speed_toward(ball.vel, ball.pos, s::OPPONENT_GOAL) * s::VEL);
    for i in 0..NUM_BOOST_PADS {
        let pad = snap.pads[if team == Team::Blue {
            i
        } else {
            flipped_pad(i)
        }];
        w.scalar(pad_value(pad.active, pad.cooldown));
    }
    for value in prev_action.to_floats() {
        w.scalar(value);
    }

    let me_body = team_view(my.body, team);
    v2_car(my, me_body, w);
    // The agent's own frame: how it spins and slides, where the ball is and heads, where the goals are.
    w.vec(local(me_body.rot, me_body.ang_vel) * s::CAR_ANG_VEL);
    w.vec(local(me_body.rot, me_body.vel) * s::VEL);
    let to_ball = ball.pos - me_body.pos;
    w.vec(local(me_body.rot, to_ball) * s::REL_POS);
    w.vec(local(me_body.rot, to_ball.normalize_or_zero()));
    w.vec(local(me_body.rot, ball.vel - me_body.vel) * s::VEL);
    w.scalar(to_ball.length() * s::DIST);
    w.vec(local(me_body.rot, s::OPPONENT_GOAL - me_body.pos) * s::DIST);
    w.vec(local(me_body.rot, s::OWN_GOAL - me_body.pos) * s::DIST);

    write_others(snap, me, places, V2_CAR + V2_OTHER_EXTRA, w, |car, w| {
        let body = team_view(car.body, team);
        v2_car(car, body, w);
        let gap = body.pos - me_body.pos;
        w.vec(local(me_body.rot, gap) * s::REL_POS);
        w.vec(local(me_body.rot, body.vel - me_body.vel) * s::VEL);
        w.scalar(gap.length() * s::DIST);
        let other_to_ball = ball.pos - body.pos;
        w.vec(other_to_ball * s::REL_POS);
        w.scalar(speed_toward(body.vel, body.pos, ball.pos) * s::VEL);
        w.scalar(other_to_ball.length() * s::DIST);
    });
}

fn v2_car(car: &CarSnapshot, body: Body, w: &mut Writer<'_>) {
    use v2_scale as s;
    w.vec(body.pos * s::POS);
    w.vec(body.forward());
    w.vec(body.up());
    w.vec(body.vel * s::VEL);
    w.vec(body.ang_vel * s::CAR_ANG_VEL);
    w.scalar(body.vel.length() * s::VEL);
    w.scalar(car.boost * s::BOOST);
    w.flag(car.on_ground);
    w.flag(car.has_jumped);
    w.flag(car.has_double_jumped);
    w.flag(car.has_flipped);
    w.flag(car.can_flip);
    w.flag(car.is_demoed);
    w.flag(car.is_supersonic);
    w.scalar(car.demo_timer * s::DEMO_TIMER);
    w.scalar(car.flip_time_left * s::FLIP_WINDOW);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_is_named_as_in_the_settings_and_its_layout_checked_alone() {
        for name in ["native", "nitro_v2"] {
            let parsed: ObsKind = serde_json::from_str(&format!("\"{name}\"")).unwrap();
            assert_eq!(ObsKind::from_name(name), Some(parsed));
        }
        assert_eq!(ObsKind::from_name("nexto"), None);
        let padded = ObsKind::Native.checked_layout(2, true, 3).unwrap();
        assert_eq!(padded, ObsKind::Native.layout(2, true, 3));
        assert!(ObsKind::Native.checked_layout(3, true, 2).is_err());
        assert!(ObsKind::Native.checked_layout(0, true, 0).is_err());
    }
    use crate::snapshot::PadSnapshot;
    use std::f32::consts::PI;

    fn body(seed: f32) -> Body {
        let rot = Mat3A::from_rotation_z(0.3 * seed)
            * Mat3A::from_rotation_y(-0.2 * seed)
            * Mat3A::from_rotation_x(0.1 * seed);
        Body {
            pos: Vec3A::new(100.0 * seed, -250.0 * seed, 17.0 + seed),
            rot,
            vel: Vec3A::new(300.0, -40.0 * seed, 5.0),
            ang_vel: Vec3A::new(0.1, 0.2 * seed, -0.3),
        }
    }

    fn snapshot(team_size: usize) -> GameSnapshot {
        let mut snap = GameSnapshot {
            ball: body(7.0),
            ..GameSnapshot::default()
        };
        for i in 0..team_size * 2 {
            snap.cars.push(CarSnapshot {
                team: if i % 2 == 0 { Team::Blue } else { Team::Orange },
                body: body(i as f32 + 1.0),
                boost: 10.0 * i as f32,
                on_ground: i % 3 == 0,
                can_flip: true,
                ..CarSnapshot::default()
            });
        }
        snap.pads[3] = PadSnapshot {
            active: false,
            cooldown: 4.0,
        };
        snap.pads[30] = PadSnapshot {
            active: false,
            cooldown: 1.5,
        };
        snap
    }

    /// The same game seen from the other side: everything rotated 180°, teams swapped.
    fn swapped_sides(snap: &GameSnapshot) -> GameSnapshot {
        let mut out = snap.clone();
        out.ball = team_view(snap.ball, Team::Orange);
        for car in &mut out.cars {
            car.body = team_view(car.body, Team::Orange);
            car.team = car.team.opposite();
        }
        for i in 0..NUM_BOOST_PADS {
            out.pads[i] = snap.pads[flipped_pad(i)];
        }
        out
    }

    fn obs(kind: ObsKind, snap: &GameSnapshot, me: usize) -> Vec<f32> {
        let mut out = vec![0.0; kind.layout(snap.cars.len() / 2, true, 0).size];
        kind.build(
            snap,
            me,
            Controls {
                throttle: 1.0,
                jump: true,
                ..Controls::default()
            },
            0,
            &mut out,
        );
        out
    }

    #[test]
    fn sizes() {
        assert_eq!(ObsKind::Native.layout(1, true, 0).size, 112);
        // Self part: ball 9 + pads 34 + previous action 8 + own car 23 + egocentric extras 9 = 83.
        assert_eq!(ObsKind::Native.layout(3, true, 0).size, 83 + 29 * 5);
        // A lone car sees only its own part; two Blue cars without opponents see one teammate.
        assert_eq!(ObsKind::Native.layout(1, false, 0).size, 83);
        // v2: ball 20 + pads 34 + previous action 8 + own car 26 + own extras 22 = 110; 38 per other car.
        assert_eq!(ObsKind::NitroV2.layout(1, true, 0).size, 148);
        assert_eq!(ObsKind::NitroV2.layout(1, false, 0).size, 110);
        assert_eq!(ObsKind::NitroV2.layout(2, true, 0).size, 110 + 38 * 3);
        assert_eq!(ObsKind::Native.layout(2, false, 0).size, 83 + 29);
    }

    #[test]
    fn native_is_identical_for_both_teams_in_mirrored_situations() {
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            for team_size in 1..=3 {
                let snap = snapshot(team_size);
                let swapped = swapped_sides(&snap);
                for me in 0..snap.cars.len() {
                    assert_eq!(
                        obs(kind, &snap, me),
                        obs(kind, &swapped, me),
                        "{kind:?} car {me}"
                    );
                }
            }
        }
    }

    #[test]
    fn v2_features_are_checked_by_hand() {
        // A Blue car at the centre facing +Y (toward the opponent's goal) with the ball 1000 ahead, rolling
        // toward that goal at 500 uu/s; an Orange car behind the ball.
        let mut snap = snapshot(1);
        snap.cars[0].body = Body {
            pos: Vec3A::new(0.0, 0.0, 17.0),
            rot: Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2),
            vel: Vec3A::new(0.0, 230.0, 0.0),
            ..Body::RESTING
        };
        snap.cars[0].on_ground = true;
        snap.cars[0].flip_time_left = 1.25;
        snap.ball = Body {
            pos: Vec3A::new(0.0, 1000.0, 93.0),
            vel: Vec3A::new(0.0, 500.0, 0.0),
            ..Body::RESTING
        };
        snap.cars[1].body = Body {
            pos: Vec3A::new(0.0, 2000.0, 17.0),
            rot: Mat3A::from_rotation_z(-std::f32::consts::FRAC_PI_2),
            vel: Vec3A::new(0.0, -1150.0, 0.0),
            ..Body::RESTING
        };
        let o = obs(ObsKind::NitroV2, &snap, 0);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-4;
        // Ball block.
        assert!(close(o[9], 500.0 / 2300.0), "ball speed");
        assert_eq!(o[10], 1.0, "ball on the floor");
        assert!(
            close(o[11], 0.0) && close(o[12], 4120.0 / 5120.0),
            "ball to opponent goal"
        );
        assert!(close(
            o[14],
            ((4120.0f32).powi(2) + (321.3875f32 - 93.0).powi(2)).sqrt() / 5120.0
        ));
        assert!(close(o[16], -6120.0 / 5120.0), "ball to own goal");
        // Toward the goal centre, which is 228.4 above the ball, so a little less than the full 500.
        let toward = 500.0 * 4120.0 / (4120.0f32.powi(2) + 228.3875f32.powi(2)).sqrt();
        assert!(
            close(o[19], toward / 2300.0),
            "ball speed toward the opponent's goal"
        );
        // Own car block starts after ball 20 + pads 34 + previous action 8 = 62: speed at index 62 + 15.
        assert!(close(o[62 + 15], 0.1), "own speed 230 / 2300");
        // The timers: alive (0) and, on the ground with a flip available, the full window (1.25 / 1.25).
        assert!(close(o[62 + 24], 0.0) && close(o[62 + 25], 1.0), "timers");
        let extras = 62 + 26;
        assert!(
            o[extras..extras + 3].iter().all(|&v| close(v, 0.0)),
            "no spin"
        );
        // Local velocity: forward is +Y for this car, so the car moves purely forward.
        let local_vel = &o[extras + 3..extras + 6];
        assert!(close(local_vel[0], 0.1) && close(local_vel[1], 0.0) && close(local_vel[2], 0.0));
        // Ball straight ahead: direction (1, 0, small up), distance 1000 / 5120.
        let dir = &o[extras + 9..extras + 12];
        assert!(dir[0] > 0.99 && close(dir[1], 0.0), "{dir:?}");
        assert!(close(
            o[extras + 15],
            (1000.0f32.powi(2) + 76.0f32.powi(2)).sqrt() / 5120.0
        ));
        // The opponent's goal is straight ahead in the car's frame, the own goal straight behind.
        assert!(close(o[extras + 16], 5120.0 / 5120.0) && close(o[extras + 19], -1.0));
        // Other car: 38 entries from 110; its speed toward the ball is 1150 (it drives at the ball).
        let other = 110;
        assert!(close(o[other + 15], 0.5), "opponent speed");
        assert!(
            close(o[other + 26 + 6], 2000.0 / 5120.0),
            "gap to the opponent"
        );
        // It drives at the ball, which sits 76 above its own height, so nearly its full 1150 counts.
        let at_ball = 1150.0 * 1000.0 / (1000.0f32.powi(2) + 76.0f32.powi(2)).sqrt();
        assert!(
            close(o[other + 26 + 10], at_ball / 2300.0),
            "opponent speed toward the ball"
        );
        assert!(close(
            o[other + 26 + 11],
            (1000.0f32.powi(2) + 76.0f32.powi(2)).sqrt() / 5120.0
        ));
    }

    #[test]
    fn local_frame_puts_a_ball_ahead_on_the_forward_axis() {
        let mut snap = snapshot(1);
        snap.cars[0].body = Body {
            pos: Vec3A::new(0.0, 0.0, 17.0),
            ..Body::RESTING
        };
        snap.ball = Body {
            pos: Vec3A::new(2300.0, 0.0, 17.0),
            ..Body::RESTING
        };
        let o = obs(ObsKind::Native, &snap, 0);
        let ball_rel = NATIVE_GLOBAL + NATIVE_CAR + 3;
        assert_eq!(&o[ball_rel..ball_rel + 3], &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_padded_observation_keeps_one_size_and_marks_its_empty_places() {
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            let room = 3;
            let size = kind.layout(1, true, room).size;
            for (team_size, has_opponents) in
                [(1, true), (1, false), (2, true), (3, true), (2, false)]
            {
                let layout = kind.layout(team_size, has_opponents, room);
                assert_eq!(layout.size, size, "{kind:?} {team_size} {has_opponents}");
                assert_eq!((layout.teammates, layout.opponents), (2, 3));
            }
            // A 1v1 game: own part as unpadded; teammates' places empty; the opponent's block as unpadded
            // plus its "present" flag; the other two opponent places empty.
            let snap = snapshot(1);
            let exact = kind.layout(1, true, 0);
            let padded = kind.layout(1, true, room);
            let (mut plain, mut full) = (vec![0.0; exact.size], vec![9.0; padded.size]);
            kind.build(&snap, 0, Controls::default(), 0, &mut plain);
            kind.build(&snap, 0, Controls::default(), room, &mut full);
            let own = exact.self_part;
            assert_eq!(full[..own], plain[..own]);
            let block = padded.other_car;
            assert_eq!(block, exact.other_car + 1);
            let teammates = &full[own..own + 2 * block];
            assert!(
                teammates.iter().all(|&v| v == 0.0),
                "no teammates: empty places"
            );
            let opponent = &full[own + 2 * block..own + 3 * block];
            assert_eq!(opponent[..block - 1], plain[own..own + exact.other_car]);
            assert_eq!(
                opponent[block - 1],
                1.0,
                "the opponent's place says a car is there"
            );
            assert!(full[own + 3 * block..].iter().all(|&v| v == 0.0));
            // A practice game without opponents: every opponent place empty, same size.
            let mut alone = snapshot(1);
            alone.cars.truncate(1);
            let mut lone = vec![9.0; padded.size];
            kind.build(&alone, 0, Controls::default(), room, &mut lone);
            assert_eq!(lone[..own], full[..own]);
            assert!(lone[own..].iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn slots_put_each_car_at_its_place_and_the_empty_places_between() {
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            // 2v2 in a room of 3, cars Blue, Orange, Blue, Orange; their places among their team's.
            let snap = snapshot(2);
            let layout = kind.layout(2, true, 3);
            let (own, block) = (layout.self_part, layout.other_car);
            let build = |me: usize, slots: &[u8]| {
                let mut out = vec![9.0; layout.size];
                kind.build_slotted(&snap, me, Controls::default(), 3, slots, &mut out);
                out
            };
            let slots = [2, 0, 0, 2];
            let (plain, slotted) = (build(2, &[]), build(2, &slots));
            let place =
                |obs: &[f32], k: usize| obs[own + k * block..own + (k + 1) * block].to_vec();
            assert_eq!(slotted[..own], plain[..own]);
            // Car 2 (slot 0) sees its teammate (slot 2) at the second of its two places, the first empty.
            assert!(place(&slotted, 0).iter().all(|&v| v == 0.0));
            assert_eq!(place(&slotted, 1), place(&plain, 0));
            // Opponents: car 1 at place 0, place 1 empty, car 3 at place 2.
            assert_eq!(place(&slotted, 2), place(&plain, 2));
            assert!(place(&slotted, 3).iter().all(|&v| v == 0.0));
            assert_eq!(place(&slotted, 4), place(&plain, 3));
            // Car 0 (slot 2) has its teammate (slot 0) at place 0: as unslotted.
            assert_eq!(
                build(0, &slots)[..own + 2 * block],
                build(0, &[])[..own + 2 * block]
            );
            assert_eq!(
                build(1, &[]),
                build(1, &[0, 0, 1, 1]),
                "places in the cars' order"
            );
        }
    }

    #[test]
    fn one_place_per_car_present_matches_the_padded_observation_in_any_game_size() {
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            // Within a room of 3 the places for the cars present hold the padded observation's numbers.
            for team_size in 1..=3 {
                let snap = snapshot(team_size);
                let padded = kind.layout(team_size, true, 3);
                let mut full = vec![9.0; padded.size];
                kind.build(&snap, 1, Controls::default(), 3, &mut full);
                let mut cars = Vec::new();
                let (teammates, opponents) =
                    kind.build_cars(&snap, 1, Controls::default(), &mut cars);
                assert_eq!((teammates, opponents), (team_size - 1, team_size));
                let layout = kind.cars_layout(teammates, opponents);
                assert_eq!(cars.len(), layout.size);
                let (own, block) = (layout.self_part, layout.other_car);
                assert_eq!((own, block), (padded.self_part, padded.other_car));
                assert_eq!(cars[..own], full[..own]);
                let teammate_part = (team_size - 1) * block;
                assert_eq!(
                    cars[own..own + teammate_part],
                    full[own..own + teammate_part]
                );
                let opponents_at = own + 2 * block;
                assert_eq!(
                    cars[own + teammate_part..],
                    full[opponents_at..opponents_at + team_size * block]
                );
            }
            // Bigger and uneven games than any room: 12v12 and 5v2, every block flagged.
            let mut uneven = snapshot(5);
            let mut orange = 0;
            uneven.cars.retain(|car| {
                orange += usize::from(car.team == Team::Orange);
                car.team == Team::Blue || orange <= 2
            });
            for (snap, expected) in [(snapshot(12), (11, 12)), (uneven, (4, 2))] {
                let mut cars = Vec::new();
                assert_eq!(
                    kind.build_cars(&snap, 0, Controls::default(), &mut cars),
                    expected
                );
                let layout = kind.cars_layout(expected.0, expected.1);
                assert_eq!(cars.len(), layout.size);
                for place in 0..expected.0 + expected.1 {
                    let flag = layout.self_part + (place + 1) * layout.other_car - 1;
                    assert_eq!(cars[flag], 1.0, "{kind:?} place {place}");
                }
                assert!(cars.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn pad_mirroring_undoes_itself_and_fixes_the_centre_pads() {
        for i in 0..NUM_BOOST_PADS {
            assert_eq!(mirrored_pad(mirrored_pad(i)), i);
        }
        // Pads on the centre line (x = 0) are their own image: slots 0, 7, 14, 19, 26, 33 by position.
        for i in [0, 7, 14, 19, 26, 33] {
            assert_eq!(mirrored_pad(i), i, "pad {i}");
        }
        assert_eq!(mirrored_pad(1), 2);
        assert_eq!(mirrored_pad(15), 18);
        assert_eq!(mirrored_pad(16), 17);
    }

    /// Deterministic random games against a committed reference file, so any change to the v2 observation's
    /// values shows. Set NITRO_WRITE_REFERENCE=1 to rewrite the file after a deliberate change to the
    /// observation.
    #[test]
    fn v2_reference_cases_match_the_committed_file() {
        use std::fmt::Write as _;
        let mut rng = fastrand::Rng::with_seed(20261012);
        // Version 2: each car's flags are drawn the way a RocketSim car can have them, and its air time since
        // the jump follows the flip timer, so an adapter that derives the flags from a real car state can be
        // checked on every value.
        let mut text = String::from("nitro_v2_reference 2\ncases 4\n");
        let mut air_times = Vec::new();
        let mut next_state = 0usize;
        for (team_size, opponents) in [(1usize, true), (1, false), (2, true), (3, true)] {
            let cars = team_size * if opponents { 2 } else { 1 };
            let mut snap = GameSnapshot::default();
            let random_body = |rng: &mut fastrand::Rng, z_min: f32| Body {
                pos: Vec3A::new(
                    (rng.f32() * 2.0 - 1.0) * 4000.0,
                    (rng.f32() * 2.0 - 1.0) * 5000.0,
                    z_min + rng.f32() * 1500.0,
                ),
                rot: Mat3A::from_rotation_z((rng.f32() * 2.0 - 1.0) * PI)
                    * Mat3A::from_rotation_y((rng.f32() * 2.0 - 1.0) * 1.5)
                    * Mat3A::from_rotation_x((rng.f32() * 2.0 - 1.0) * PI),
                vel: Vec3A::new(
                    (rng.f32() * 2.0 - 1.0) * 2300.0,
                    (rng.f32() * 2.0 - 1.0) * 2300.0,
                    (rng.f32() * 2.0 - 1.0) * 1000.0,
                ),
                ang_vel: Vec3A::new(
                    (rng.f32() * 2.0 - 1.0) * 5.5,
                    (rng.f32() * 2.0 - 1.0) * 5.5,
                    (rng.f32() * 2.0 - 1.0) * 5.5,
                ),
            };
            snap.ball = random_body(&mut rng, 93.0);
            air_times.clear();
            for i in 0..cars {
                let body = random_body(&mut rng, 17.0);
                // As in RocketSim: on the ground nothing is used up; in the air a car may have jumped, then
                // flipped or double-jumped (not both), and keeps its flip only within the window. The cars
                // take the six states in turn, so every one appears among the reference's 13 cars.
                let state = next_state % 6;
                next_state += 1;
                let on_ground = state == 0;
                let has_jumped = state >= 2;
                let has_flipped = state == 4;
                let has_double_jumped = state == 5;
                let air_time = match state {
                    2 => rng.f32() * 1.2,       // jumped, flip still there
                    3 => 1.3 + rng.f32() * 0.7, // jumped, flip window over
                    4 | 5 => rng.f32() * 2.0,   // flipped or double-jumped
                    _ => 0.0,                   // on the ground, or drove off a wall
                };
                let can_flip = on_ground
                    || (!has_flipped
                        && !has_double_jumped
                        && (!has_jumped || air_time < crate::snapshot::FLIP_WINDOW_SECONDS));
                let is_demoed = rng.usize(0..4) == 0;
                air_times.push(air_time);
                snap.cars.push(CarSnapshot {
                    team: if i % 2 == 0 || !opponents {
                        Team::Blue
                    } else {
                        Team::Orange
                    },
                    boost: rng.f32() * 100.0,
                    on_ground,
                    has_jumped,
                    has_double_jumped,
                    has_flipped,
                    can_flip,
                    is_demoed,
                    is_supersonic: body.vel.length() >= 2200.0,
                    demo_timer: if is_demoed { rng.f32() * 3.0 } else { 0.0 },
                    flip_time_left: CarSnapshot::flip_time_left_from(
                        on_ground, has_jumped, can_flip, air_time,
                    ),
                    body,
                });
            }
            for pad in &mut snap.pads {
                *pad = PadSnapshot {
                    active: rng.bool(),
                    cooldown: rng.f32() * 10.0,
                };
                if pad.active {
                    pad.cooldown = 0.0;
                }
            }
            let axis = |rng: &mut fastrand::Rng| [-1.0f32, 0.0, 1.0][rng.usize(0..3)];
            let prev = Controls {
                throttle: axis(&mut rng),
                steer: axis(&mut rng),
                pitch: axis(&mut rng),
                yaw: axis(&mut rng),
                roll: axis(&mut rng),
                jump: rng.bool(),
                boost: rng.bool(),
                handbrake: rng.bool(),
            };
            let _ = writeln!(
                text,
                "case team_size {team_size} opponents {} cars {cars}",
                u8::from(opponents)
            );
            let floats = |values: &[f32]| {
                values
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let body_floats = |b: &Body| {
                let mut v = b.pos.to_array().to_vec();
                v.extend(b.rot.x_axis.to_array());
                v.extend(b.rot.y_axis.to_array());
                v.extend(b.rot.z_axis.to_array());
                v.extend(b.vel.to_array());
                v.extend(b.ang_vel.to_array());
                v
            };
            let ball = [
                snap.ball.pos.to_array(),
                snap.ball.vel.to_array(),
                snap.ball.ang_vel.to_array(),
            ]
            .concat();
            let _ = writeln!(text, "ball {}", floats(&ball));
            for (car, air_time) in snap.cars.iter().zip(&air_times) {
                let flags = [
                    car.on_ground,
                    car.has_jumped,
                    car.has_double_jumped,
                    car.has_flipped,
                    car.can_flip,
                    car.is_demoed,
                    car.is_supersonic,
                ];
                let _ = writeln!(
                    text,
                    "car {} {} {:?} {} {:?} {:?} {:?}",
                    car.team.index(),
                    floats(&body_floats(&car.body)),
                    car.boost,
                    flags
                        .iter()
                        .map(|&f| u8::from(f).to_string())
                        .collect::<Vec<_>>()
                        .join(" "),
                    car.demo_timer,
                    car.flip_time_left,
                    air_time
                );
            }
            let pads: Vec<String> = snap
                .pads
                .iter()
                .map(|p| format!("{} {:?}", u8::from(p.active), p.cooldown))
                .collect();
            let _ = writeln!(text, "pads {}", pads.join(" "));
            let _ = writeln!(text, "prev {}", floats(&prev.to_floats()));
            let layout = ObsKind::NitroV2.layout(team_size, opponents, 0);
            for me in 0..cars {
                let mut out = vec![0.0; layout.size];
                ObsKind::NitroV2.build(&snap, me, prev, 0, &mut out);
                let _ = writeln!(text, "obs car {me} size {} {}", layout.size, floats(&out));
            }
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/nitro_v2_reference.txt"
        );
        if std::env::var_os("NITRO_WRITE_REFERENCE").is_some() {
            std::fs::write(path, &text).unwrap();
            return;
        }
        let committed = std::fs::read_to_string(path).expect("testdata/nitro_v2_reference.txt");
        // Token by token with a tolerance: Windows' sin/cos round the generated games' rotation matrices
        // differently in the last digit, and git may check the file out with CRLF line ends.
        let (mine, theirs): (Vec<&str>, Vec<&str>) = (
            text.split_whitespace().collect(),
            committed.split_whitespace().collect(),
        );
        assert_eq!(
            mine.len(),
            theirs.len(),
            "the reference file has a different shape"
        );
        for (index, (a, b)) in mine.iter().zip(&theirs).enumerate() {
            let same = match (a.parse::<f32>(), b.parse::<f32>()) {
                (Ok(x), Ok(y)) => (x - y).abs() <= 1e-5 * x.abs().max(y.abs()).max(1.0),
                _ => a == b,
            };
            assert!(
                same,
                "the v2 observation changed (token {index}: {a} vs {b}): rerun with NITRO_WRITE_REFERENCE=1 \
                 if that was meant"
            );
        }
    }

    #[test]
    fn observations_are_finite() {
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            let snap = snapshot(2);
            for me in 0..4 {
                assert!(obs(kind, &snap, me).iter().all(|v| v.is_finite()));
            }
        }
    }
}
