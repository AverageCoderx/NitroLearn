//! Observation builders. Both read only `CarSnapshot`/ball/pad fields, so the RLBot bot builds the
//! exact same vectors from a `GamePacket`.

use rocketsim::{Mat3A, Vec3A};
use serde::Deserialize;

use crate::ball_prediction::VALUES_PER_TIME;
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
    /// GigaLearn's `AdvancedObs`, reproduced exactly (including its mirrored boost-pad timers).
    GigalearnAdvanced,
    /// GigaLearn's `DefaultObs`, reproduced exactly.
    GigalearnDefault,
    /// AdvancedObsPadderGGL, a community GigaLearn observation, reproduced exactly: places for
    /// `obs_team_size` players a side (its `teamSize`; 0 = the game's), the empty ones all zeros.
    GigalearnPadder,
    /// rlgym-tools' `AdvancedObsPadder`: `gigalearn_padder` with each car's `has_jump` (not jumped yet) where
    /// that has its 0 (bots brought over from RLGym-PPO).
    RlgymPadder,
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

impl ObsLayout {
    /// This layout with the ball's predicted path at `times` prediction times: `VALUES_PER_TIME` numbers
    /// each, in the own part right after the ball's.
    pub fn with_ball_prediction(self, times: usize) -> Self {
        let extra = times * VALUES_PER_TIME;
        Self {
            size: self.size + extra,
            self_part: self.self_part + extra,
            ..self
        }
    }
}

const NATIVE_CAR: usize = 23;
const NATIVE_SELF_EXTRA: usize = 9;
const NATIVE_OTHER: usize = NATIVE_CAR + 6;
const NATIVE_GLOBAL: usize = 9 + NUM_BOOST_PADS + 8;
const GIGALEARN_PLAYER: usize = 29;
const GIGALEARN_GLOBAL: usize = 9 + 8 + NUM_BOOST_PADS;
const GIGALEARN_DEFAULT_PLAYER: usize = 19;
// AdvancedObsPadderGGL: 26 numbers a car, and 6 more (relative to the own car) for every other car.
const PADDER_PLAYER: usize = 26;
const PADDER_OTHER: usize = PADDER_PLAYER + 6;
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
    /// are zeros: one network fits every game up to `room` a side, the one-car practice tasks included.
    /// GigaLearn's padder has `room` places its own way (unflagged); its other copies take no room.
    pub fn layout(self, team_size: usize, has_opponents: bool, room: usize) -> ObsLayout {
        let (self_part, base_other) = self.parts();
        if matches!(self, Self::GigalearnPadder | Self::RlgymPadder) {
            // Its places are always `teamSize - 1` teammates and `teamSize` opponents, unflagged.
            let places = if room == 0 { team_size } else { room };
            return ObsLayout {
                size: self_part + base_other * (2 * places - 1),
                self_part,
                other_car: base_other,
                teammates: places - 1,
                opponents: places,
                room,
            };
        }
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
            Self::GigalearnAdvanced => (GIGALEARN_GLOBAL + GIGALEARN_PLAYER, GIGALEARN_PLAYER),
            Self::GigalearnDefault => (
                GIGALEARN_GLOBAL + GIGALEARN_DEFAULT_PLAYER,
                GIGALEARN_DEFAULT_PLAYER,
            ),
            Self::GigalearnPadder | Self::RlgymPadder => {
                (GIGALEARN_GLOBAL + PADDER_PLAYER, PADDER_OTHER)
            }
        }
    }

    /// The layout with exactly `teammates` and `opponents` flagged car places and none empty (`build_cars`):
    /// what an attention policy reads in a game bigger than its room. Native and nitro_v2 only.
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

    /// Whether this observation has NitroLearn's flagged car places, so it can be padded, shuffled, read by
    /// an attention encoder or given the ball prediction (GigaLearn's exact copies cannot).
    pub fn pads(self) -> bool {
        matches!(self, Self::Native | Self::NitroV2)
    }

    /// Whether this observation takes `obs_team_size` (room for more players than the game has): NitroLearn's
    /// padded ones and GigaLearn's padder (which pads its own way, without flags).
    pub fn takes_room(self) -> bool {
        self.pads() || matches!(self, Self::GigalearnPadder | Self::RlgymPadder)
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
        if room != 0 && !self.takes_room() {
            return Err(
                "obs_team_size needs the native, nitro_v2 or gigalearn_padder observation"
                    .to_owned(),
            );
        }
        Ok(self.layout(team_size, has_opponents, room))
    }

    /// The kind a settings file names (`env.observation`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "native" => Some(Self::Native),
            "nitro_v2" => Some(Self::NitroV2),
            "gigalearn_advanced" => Some(Self::GigalearnAdvanced),
            "gigalearn_default" => Some(Self::GigalearnDefault),
            "gigalearn_padder" => Some(Self::GigalearnPadder),
            "rlgym_padder" => Some(Self::RlgymPadder),
            _ => None,
        }
    }

    /// The name a settings file gives this kind (`from_name`'s).
    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::NitroV2 => "nitro_v2",
            Self::GigalearnAdvanced => "gigalearn_advanced",
            Self::GigalearnDefault => "gigalearn_default",
            Self::GigalearnPadder => "gigalearn_padder",
            Self::RlgymPadder => "rlgym_padder",
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
        self.build_predicted(snap, me, prev_action, room, &[], out);
    }

    /// `build` with the ball's predicted bodies at each prediction time (world frame, as
    /// `ball_prediction::pick` gives them); `out.len()` must equal the layout's `with_ball_prediction` size.
    pub fn build_predicted(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        room: usize,
        prediction: &[Body],
        out: &mut [f32],
    ) {
        self.build_slotted(snap, me, prev_action, room, &[], prediction, out);
    }

    /// `build` with each car at place `slots[car]` among its team's `room` places (`env.shuffle_slots`);
    /// a car's teammates keep their places' order with its own left out. Empty `slots` = the cars in
    /// order, as `build`. `prediction` as in `build_predicted`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_slotted(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        room: usize,
        slots: &[u8],
        prediction: &[Body],
        out: &mut [f32],
    ) {
        assert!(
            slots.is_empty() || (room > 0 && self.pads()),
            "slots need a padded observation"
        );
        assert!(room == 0 || self.takes_room(), "{self:?} cannot be padded");
        self.write(
            snap,
            me,
            prev_action,
            (room > 0).then(|| (room - 1, room)),
            slots,
            prediction,
            out,
        );
    }

    /// Writes the observation of car `me` with one flagged block per other car in the game and no empty
    /// places (`cars_layout`), into `out` resized to fit; returns the (teammates, opponents) written. For a
    /// policy whose attention encoder reads any number of cars, in a game bigger than its room (any size,
    /// e.g. RLBot's 12v12). `prediction` as in `build_predicted`.
    pub fn build_cars(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        prediction: &[Body],
        out: &mut Vec<f32>,
    ) -> (usize, usize) {
        assert!(self.pads(), "{self:?} has no car places");
        let team = snap.cars[me].team;
        let teammates = snap.cars.iter().filter(|car| car.team == team).count() - 1;
        let opponents = snap.cars.len() - 1 - teammates;
        out.clear();
        let layout = self
            .cars_layout(teammates, opponents)
            .with_ball_prediction(prediction.len());
        out.resize(layout.size, 0.0);
        self.write(
            snap,
            me,
            prev_action,
            Some((teammates, opponents)),
            &[],
            prediction,
            out,
        );
        (teammates, opponents)
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        self,
        snap: &GameSnapshot,
        me: usize,
        prev_action: Controls,
        places: Places,
        slots: &[u8],
        prediction: &[Body],
        out: &mut [f32],
    ) {
        assert!(
            prediction.is_empty() || self.pads(),
            "{self:?} has no ball prediction"
        );
        let mut writer = Writer { out, len: 0 };
        let slotted = (places, slots);
        match self {
            Self::Native => native(snap, me, prev_action, slotted, prediction, &mut writer),
            Self::NitroV2 => nitro_v2(snap, me, prev_action, slotted, prediction, &mut writer),
            Self::GigalearnAdvanced => gigalearn_advanced(snap, me, prev_action, &mut writer),
            Self::GigalearnDefault => gigalearn_default(snap, me, prev_action, &mut writer),
            Self::GigalearnPadder => {
                gigalearn_padder(snap, me, prev_action, places, false, &mut writer);
            }
            Self::RlgymPadder => gigalearn_padder(snap, me, prev_action, places, true, &mut writer),
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

/// Snapshot index of the pad in slot `slot` of GigaLearn's (RLBot's) pad list. That list is ordered by
/// y then x except at y = ∓1024, where the centre pad sits between the two pads at y = ∓1036.
pub(crate) const fn gigalearn_pad(slot: usize) -> usize {
    match slot {
        13 => 14,
        14 => 13,
        19 => 20,
        20 => 19,
        other => other,
    }
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
    prediction: &[Body],
    w: &mut Writer<'_>,
) {
    use native_scale as s;
    let my = &snap.cars[me];
    let team = my.team;
    let ball = team_view(snap.ball, team);
    w.vec(ball.pos * s::POS);
    w.vec(ball.vel * s::VEL);
    w.vec(ball.ang_vel * s::BALL_ANG_VEL);
    write_prediction(prediction, team, team_view(my.body, team), w);
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

/// The ball's predicted path, per prediction time: its position and velocity, then both relative to the
/// observing car (`me`, in the team view), all in the team view and the ball features' scales. Relative
/// vectors stay in world axes: the own-car part already gives the ball in the car's frame.
fn write_prediction(prediction: &[Body], team: Team, me: Body, w: &mut Writer<'_>) {
    use native_scale as s;
    for &body in prediction {
        let ball = team_view(body, team);
        w.vec(ball.pos * s::POS);
        w.vec(ball.vel * s::VEL);
        w.vec((ball.pos - me.pos) * s::REL_POS);
        w.vec((ball.vel - me.vel) * s::VEL);
    }
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
    prediction: &[Body],
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
    write_prediction(prediction, team, team_view(my.body, team), w);
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

mod gigalearn_scale {
    pub const POS: f32 = 1.0 / 5000.0;
    pub const VEL: f32 = 1.0 / 2300.0;
    pub const ANG_VEL: f32 = 1.0 / 3.0;
}

fn gigalearn_advanced(snap: &GameSnapshot, me: usize, prev_action: Controls, w: &mut Writer<'_>) {
    use gigalearn_scale as s;
    let team = snap.cars[me].team;
    let ball = team_view(snap.ball, team);
    w.vec(ball.pos * s::POS);
    w.vec(ball.vel * s::VEL);
    w.vec(ball.ang_vel * s::ANG_VEL);
    for value in prev_action.to_floats() {
        w.scalar(value);
    }
    for slot in 0..NUM_BOOST_PADS {
        // GigaLearn's GetBoostPadTimers returns the opposite team's list, so the timer comes from the
        // mirrored pad. Reproduced on purpose.
        let pad = gigalearn_pad(slot);
        let (active_pad, timer_pad) = if team == Team::Blue {
            (pad, flipped_pad(pad))
        } else {
            (flipped_pad(pad), pad)
        };
        w.scalar(pad_value(
            snap.pads[active_pad].active,
            snap.pads[timer_pad].cooldown,
        ));
    }
    gigalearn_player(&snap.cars[me], team, ball, w);
    for car in other_cars(snap, me) {
        gigalearn_player(car, team, ball, w);
    }
}

mod gigalearn_default_scale {
    use rocketsim::Vec3A;
    // Side wall, back wall and ceiling.
    pub const POS: Vec3A = Vec3A::new(1.0 / 4096.0, 1.0 / 5120.0, 1.0 / 2044.0);
    pub const VEL: f32 = 1.0 / 2300.0;
    // The car's top angular speed.
    pub const ANG_VEL: f32 = 1.0 / 5.5;
}

fn gigalearn_default(snap: &GameSnapshot, me: usize, prev_action: Controls, w: &mut Writer<'_>) {
    use gigalearn_default_scale as s;
    let team = snap.cars[me].team;
    let ball = team_view(snap.ball, team);
    w.vec(ball.pos * s::POS);
    w.vec(ball.vel * s::VEL);
    w.vec(ball.ang_vel * s::ANG_VEL);
    for value in prev_action.to_floats() {
        w.scalar(value);
    }
    for slot in 0..NUM_BOOST_PADS {
        let pad = gigalearn_pad(slot);
        let pad = if team == Team::Blue {
            pad
        } else {
            flipped_pad(pad)
        };
        w.flag(snap.pads[pad].active);
    }
    gigalearn_default_player(&snap.cars[me], team, w);
    for car in other_cars(snap, me) {
        gigalearn_default_player(car, team, w);
    }
}

fn gigalearn_default_player(car: &CarSnapshot, viewer: Team, w: &mut Writer<'_>) {
    use gigalearn_default_scale as s;
    let body = team_view(car.body, viewer);
    w.vec(body.pos * s::POS);
    w.vec(body.forward());
    w.vec(body.up());
    w.vec(body.vel * s::VEL);
    w.vec(body.ang_vel * s::ANG_VEL);
    w.scalar(car.boost / 100.0);
    w.flag(car.on_ground);
    w.flag(car.can_flip);
    w.flag(car.is_demoed);
}

/// AdvancedObsPadderGGL, a community GigaLearn observation: positions and speeds over 2300, spins over pi;
/// pads as gigalearn_default; the own car, then up to `teammates` teammates and `opponents` opponents in
/// car order (more are left out, as there), each also relative to the own car, the missing places all
/// zeros. `places` None: every other car (the game's size).
/// `has_jump`: rlgym-tools' version, each car's has_jump (not jumped yet) where the source has its 0
/// placeholder.
fn gigalearn_padder(
    snap: &GameSnapshot,
    me: usize,
    prev_action: Controls,
    places: Places,
    has_jump: bool,
    w: &mut Writer<'_>,
) {
    const SCALE: f32 = 1.0 / 2300.0;
    let team = snap.cars[me].team;
    let ball = team_view(snap.ball, team);
    w.vec(ball.pos * SCALE);
    w.vec(ball.vel * SCALE);
    w.vec(ball.ang_vel / std::f32::consts::PI);
    for value in prev_action.to_floats() {
        w.scalar(value);
    }
    for slot in 0..NUM_BOOST_PADS {
        let pad = gigalearn_pad(slot);
        let pad = if team == Team::Blue {
            pad
        } else {
            flipped_pad(pad)
        };
        w.flag(snap.pads[pad].active);
    }
    let player = |car: &CarSnapshot, w: &mut Writer<'_>| {
        let body = team_view(car.body, team);
        w.vec((ball.pos - body.pos) * SCALE);
        w.vec((ball.vel - body.vel) * SCALE);
        w.vec(body.pos * SCALE);
        w.vec(body.forward());
        w.vec(body.up());
        w.vec(body.vel * SCALE);
        w.vec(body.ang_vel / std::f32::consts::PI);
        w.scalar(car.boost / 100.0);
        w.flag(car.on_ground);
        w.flag(car.can_flip);
        w.flag(car.is_demoed);
        w.flag(has_jump && !car.has_jumped); // the GigaLearn source's 0 placeholder
        body
    };
    let mine = player(&snap.cars[me], w);
    let others: Vec<&CarSnapshot> = other_cars(snap, me).collect();
    let teammates = others.iter().filter(|car| car.team == team).count();
    let (room_mates, room_opponents) = places.unwrap_or((teammates, others.len() - teammates));
    for (own_side, room) in [(true, room_mates), (false, room_opponents)] {
        let side = others.iter().filter(|car| (car.team == team) == own_side);
        let mut written = 0;
        for car in side.take(room) {
            let body = player(car, w);
            w.vec((body.pos - mine.pos) * SCALE);
            w.vec((body.vel - mine.vel) * SCALE);
            written += 1;
        }
        w.zeros((room - written) * PADDER_OTHER);
    }
}

fn gigalearn_player(car: &CarSnapshot, viewer: Team, ball: Body, w: &mut Writer<'_>) {
    use gigalearn_scale as s;
    let body = team_view(car.body, viewer);
    w.vec(body.pos * s::POS);
    w.vec(body.forward());
    w.vec(body.up());
    w.vec(body.vel * s::VEL);
    w.vec(body.ang_vel * s::ANG_VEL);
    w.vec(local(body.rot, body.ang_vel) * s::ANG_VEL);
    w.vec(local(body.rot, ball.pos - body.pos) * s::POS);
    w.vec(local(body.rot, ball.vel - body.vel) * s::VEL);
    w.scalar(car.boost / 100.0);
    w.flag(car.on_ground);
    w.flag(car.can_flip);
    w.flag(car.is_demoed);
    w.flag(car.has_jumped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_is_named_as_in_the_settings_and_its_layout_checked_alone() {
        for name in [
            "native",
            "nitro_v2",
            "gigalearn_advanced",
            "gigalearn_default",
            "gigalearn_padder",
            "rlgym_padder",
        ] {
            let parsed: ObsKind = serde_json::from_str(&format!("\"{name}\"")).unwrap();
            assert_eq!(ObsKind::from_name(name), Some(parsed));
            assert_eq!(parsed.name(), name);
        }
        assert_eq!(ObsKind::from_name("nexto"), None);
        let padded = ObsKind::Native.checked_layout(2, true, 3).unwrap();
        assert_eq!(padded, ObsKind::Native.layout(2, true, 3));
        assert!(ObsKind::Native.checked_layout(3, true, 2).is_err());
        assert!(
            ObsKind::GigalearnAdvanced
                .checked_layout(1, true, 1)
                .is_err()
        );
        assert!(
            ObsKind::GigalearnDefault
                .checked_layout(1, true, 2)
                .is_err()
        );
        assert!(ObsKind::GigalearnPadder.checked_layout(1, true, 3).is_ok());
        assert!(ObsKind::Native.checked_layout(0, true, 0).is_err());
    }

    /// The ball prediction sits right after the ball, has its size, reads the same for both teams in
    /// mirrored games, and is checked by hand.
    #[test]
    fn the_ball_prediction_follows_the_ball() {
        let predicted = [body(11.0), body(13.0), body(17.0)];
        let prev = Controls {
            steer: 1.0,
            ..Controls::default()
        };
        for kind in [ObsKind::Native, ObsKind::NitroV2] {
            for (team_size, room) in [(1, 0), (2, 0), (2, 3)] {
                let snap = snapshot(team_size);
                let layout = kind.layout(team_size, true, room);
                let with = layout.with_ball_prediction(predicted.len());
                assert_eq!(with.size, layout.size + 36);
                assert_eq!(with.self_part, layout.self_part + 36);
                assert_eq!((with.other_car, with.room), (layout.other_car, layout.room));
                let ball = if kind == ObsKind::Native { 9 } else { 20 };
                for me in 0..snap.cars.len() {
                    let (mut plain, mut out) = (vec![0.0; layout.size], vec![0.0; with.size]);
                    kind.build(&snap, me, prev, room, &mut plain);
                    kind.build_predicted(&snap, me, prev, room, &predicted, &mut out);
                    // Everything else is where it was, shifted past the prediction.
                    assert_eq!(out[..ball], plain[..ball]);
                    assert_eq!(out[ball + 36..], plain[ball..]);
                    // Seen from the other side: the same numbers.
                    let swapped = swapped_sides(&snap);
                    let turned = predicted.map(|b| team_view(b, Team::Orange));
                    let mut other = vec![0.0; with.size];
                    kind.build_predicted(&swapped, me, prev, room, &turned, &mut other);
                    assert_eq!(out, other, "{kind:?} car {me}");
                }
            }
        }
        // With attention's one place per car, the same numbers after the ball.
        let snap = snapshot(2);
        let mut cars = Vec::new();
        ObsKind::Native.build_cars(&snap, 0, prev, &predicted, &mut cars);
        let layout = ObsKind::Native.cars_layout(1, 2).with_ball_prediction(3);
        assert_eq!(cars.len(), layout.size);
        let mut padded = vec![
            0.0;
            ObsKind::Native
                .layout(2, true, 0)
                .with_ball_prediction(3)
                .size
        ];
        ObsKind::Native.build_predicted(&snap, 0, prev, 0, &predicted, &mut padded);
        assert_eq!(cars[9..45], padded[9..45]);
        // By hand: an Orange car at (100, 200, 17) moving at (0, 300, 0); the ball predicted at (500, -1000,
        // 300) moving at (0, -1200, 0). Orange sees it rotated 180°.
        let mut snap = snapshot(1);
        snap.cars[1].body = Body {
            pos: Vec3A::new(100.0, 200.0, 17.0),
            vel: Vec3A::new(0.0, 300.0, 0.0),
            ..Body::RESTING
        };
        let ahead = Body {
            pos: Vec3A::new(500.0, -1000.0, 300.0),
            vel: Vec3A::new(0.0, -1200.0, 0.0),
            ..Body::RESTING
        };
        let layout = ObsKind::Native.layout(1, true, 0).with_ball_prediction(1);
        let mut out = vec![0.0; layout.size];
        ObsKind::Native.build_predicted(&snap, 1, prev, 0, &[ahead], &mut out);
        let expected = [
            -500.0 / 4096.0,
            1000.0 / 5120.0,
            300.0 / 2044.0,
            0.0,
            1200.0 / 2300.0,
            0.0,
            -400.0 / 2300.0,
            1200.0 / 2300.0,
            283.0 / 2300.0,
            0.0,
            1500.0 / 2300.0,
            0.0,
        ];
        for (i, (&got, want)) in out[9..21].iter().zip(expected).enumerate() {
            assert!((got - want).abs() < 1e-6, "value {i}: {got} vs {want}");
        }
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
        assert_eq!(ObsKind::GigalearnAdvanced.layout(1, true, 0).size, 109);
        assert_eq!(ObsKind::GigalearnAdvanced.layout(2, true, 0).size, 167);
        assert_eq!(ObsKind::GigalearnAdvanced.layout(1, false, 0).size, 80);
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
    fn gigalearn_reads_pad_timers_from_the_mirrored_pad() {
        let snap = snapshot(1);
        let o = obs(ObsKind::GigalearnAdvanced, &snap, 0);
        let pads = 9 + 8;
        // Blue: pad 3 is taken and its timer is read from pad 30 (1.5 s); pad 30 is taken and read from pad 3.
        assert_eq!(o[pads + 3], 1.0 / 2.5);
        assert_eq!(o[pads + 30], 1.0 / 5.0);
        let native = obs(ObsKind::Native, &snap, 0);
        assert_eq!(native[9 + 3], 1.0 / 5.0);
    }

    #[test]
    fn gigalearn_lists_the_centre_pads_in_rlbot_order() {
        let mut snap = snapshot(1);
        // Snapshot pad 14 is the one at (0, -1024); GigaLearn calls it slot 13. Its mirror is pad 19.
        snap.pads[14] = PadSnapshot {
            active: false,
            cooldown: 4.0,
        };
        snap.pads[19] = PadSnapshot {
            active: false,
            cooldown: 1.5,
        };
        let o = obs(ObsKind::GigalearnAdvanced, &snap, 0);
        let pads = 9 + 8;
        assert_eq!(o[pads + 13], 1.0 / 2.5);
        assert_eq!(o[pads + 14], 1.0);
        assert_eq!(o[pads + 20], 1.0 / 5.0);
    }

    #[test]
    fn gigalearn_default_is_defaultobs() {
        // Ball 9 + previous action 8 + pads 34 + own car 19, then 19 per other car.
        assert_eq!(ObsKind::GigalearnDefault.layout(1, true, 0).size, 89);
        assert_eq!(
            ObsKind::GigalearnDefault.layout(3, true, 0).size,
            51 + 19 * 6
        );
        let mut snap = snapshot(1);
        snap.cars[1].body.pos = Vec3A::new(4096.0, -5120.0, 1022.0);
        snap.cars[1].body.vel = Vec3A::new(230.0, 0.0, 0.0);
        snap.cars[1].body.ang_vel = Vec3A::new(0.0, 0.0, 5.5);
        snap.cars[1].is_demoed = true;
        let o = obs(ObsKind::GigalearnDefault, &snap, 1);
        // Orange sees the field turned half a turn; positions over the walls and ceiling, speeds over the tops.
        assert_eq!(&o[51..54], &[-1.0, 1.0, 0.5]);
        assert_eq!(&o[60..63], &[-0.1, 0.0, 0.0]);
        assert_eq!(o[65], 1.0);
        assert_eq!(
            &o[66..70],
            &[0.1, 0.0, 1.0, 1.0],
            "boost, on ground, can flip, demoed"
        );
        // Pads: active flags only, no timers; Orange's list is the turned one (pad 3 is its slot 30).
        assert_eq!(o[17 + 30], 0.0);
        assert_eq!(o[17 + 3], 0.0);
        assert_eq!(o[17], 1.0);
        assert_eq!(
            &o[9..17],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "previous action"
        );
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
                kind.build_slotted(&snap, me, Controls::default(), 3, slots, &[], &mut out);
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
                    kind.build_cars(&snap, 1, Controls::default(), &[], &mut cars);
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
                    kind.build_cars(&snap, 0, Controls::default(), &[], &mut cars),
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

    /// One random game of the reference files, drawn from `rng` in a fixed order; `air_times` gets each car's
    /// air time since its jump.
    fn reference_game(
        rng: &mut fastrand::Rng,
        team_size: usize,
        opponents: bool,
        next_state: &mut usize,
        air_times: &mut Vec<f32>,
    ) -> (GameSnapshot, Controls) {
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
        snap.ball = random_body(&mut *rng, 93.0);
        air_times.clear();
        for i in 0..cars {
            let body = random_body(&mut *rng, 17.0);
            // As in RocketSim: on the ground nothing is used up; in the air a car may have jumped, then
            // flipped or double-jumped (not both), and keeps its flip only within the window. The cars
            // take the six states in turn, so every one appears among the reference's 13 cars.
            let state = *next_state % 6;
            *next_state += 1;
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
            throttle: axis(&mut *rng),
            steer: axis(&mut *rng),
            pitch: axis(&mut *rng),
            yaw: axis(&mut *rng),
            roll: axis(&mut *rng),
            jump: rng.bool(),
            boost: rng.bool(),
            handbrake: rng.bool(),
        };
        (snap, prev)
    }

    fn reference_floats(values: &[f32]) -> String {
        values
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The game's lines of a reference file: the case, ball, cars, pads and previous action.
    fn write_reference_game(
        text: &mut String,
        team_size: usize,
        opponents: bool,
        snap: &GameSnapshot,
        prev: Controls,
        air_times: &[f32],
    ) {
        use std::fmt::Write as _;
        let cars = snap.cars.len();
        let _ = writeln!(
            text,
            "case team_size {team_size} opponents {} cars {cars}",
            u8::from(opponents)
        );
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
        let _ = writeln!(text, "ball {}", reference_floats(&ball));
        for (car, air_time) in snap.cars.iter().zip(air_times) {
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
                reference_floats(&body_floats(&car.body)),
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
        let _ = writeln!(text, "prev {}", reference_floats(&prev.to_floats()));
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
            let (snap, prev) = reference_game(
                &mut rng,
                team_size,
                opponents,
                &mut next_state,
                &mut air_times,
            );
            let cars = snap.cars.len();
            write_reference_game(&mut text, team_size, opponents, &snap, prev, &air_times);
            let layout = ObsKind::NitroV2.layout(team_size, opponents, 0);
            for me in 0..cars {
                let mut out = vec![0.0; layout.size];
                ObsKind::NitroV2.build(&snap, me, prev, 0, &mut out);
                let _ = writeln!(
                    text,
                    "obs car {me} size {} {}",
                    layout.size,
                    reference_floats(&out)
                );
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
        check_reference(&text, path, "v2");
    }

    /// `text` against the committed reference file at `path`, token by token with a tolerance: Windows'
    /// sin/cos round the generated games' rotation matrices differently in the last digit, and git may check
    /// the file out with CRLF line ends.
    fn check_reference(text: &str, path: &str, what: &str) {
        let committed = std::fs::read_to_string(path).expect(path);
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
                "the {what} observation changed (token {index}: {a} vs {b}): rerun with \
                 NITRO_WRITE_REFERENCE=1 if that was meant"
            );
        }
    }

    #[test]
    fn gigalearn_padder_layout_pads_with_zeros_and_leaves_extra_cars_out() {
        // Own part 51 + 26, then 32 per place: teamSize - 1 teammates and teamSize opponents.
        assert_eq!(ObsKind::GigalearnPadder.layout(1, true, 0).size, 109);
        assert_eq!(ObsKind::GigalearnPadder.layout(1, true, 3).size, 237);
        assert_eq!(ObsKind::GigalearnPadder.layout(2, true, 0).size, 173);
        let snap = snapshot(1);
        let room3 = {
            let mut out = vec![0.0; 237];
            let prev = Controls {
                throttle: 1.0,
                jump: true,
                ..Controls::default()
            };
            ObsKind::GigalearnPadder.build(&snap, 0, prev, 3, &mut out);
            out
        };
        let exact = obs(ObsKind::GigalearnPadder, &snap, 0);
        // Two empty teammate places, then the one opponent, then two empty opponent places.
        assert_eq!(&room3[..77], &exact[..77]);
        assert!(room3[77..141].iter().all(|&v| v == 0.0));
        assert_eq!(&room3[141..173], &exact[77..109]);
        assert!(room3[173..].iter().all(|&v| v == 0.0));
        assert_eq!(exact[76], 0.0, "the source's placeholder after demoed");
        // rlgym-tools' version has the car's has_jump there, and the same everywhere else.
        let mut jumped = snap.clone();
        jumped.cars[1].has_jumped = true;
        let rlgym = obs(ObsKind::RlgymPadder, &jumped, 0);
        assert_eq!(rlgym[76], 1.0, "the own car has not jumped");
        assert_eq!(rlgym[77 + 25], 0.0, "the opponent has");
        let plain = obs(ObsKind::GigalearnPadder, &jumped, 0);
        let differ: Vec<usize> = (0..109).filter(|&i| rlgym[i] != plain[i]).collect();
        assert_eq!(differ, [76]);
    }

    /// AdvancedObsPadderGGL's reference: its own C++ source, run on these games, wrote the same numbers.
    /// Needs the meshes for the pads' positions (written in the file for the C++ side's pad order).
    #[test]
    fn padder_reference_cases_match_the_committed_file() {
        use std::fmt::Write as _;
        if !crate::testing::meshes_loaded() {
            return;
        }
        let sim = crate::sim::Sim::new(1, true, 0);
        let mut text = String::from("padder_reference 1\n");
        let positions: Vec<f32> = sim
            .pad_layout()
            .iter()
            .flat_map(|p| p[..3].to_vec())
            .collect();
        let _ = writeln!(text, "padpos {}", reference_floats(&positions));
        let mut rng = fastrand::Rng::with_seed(20261116);
        let (mut next_state, mut air_times) = (0usize, Vec::new());
        for (team_size, room) in [(1usize, 0usize), (1, 3), (2, 0), (2, 3), (3, 3)] {
            let (snap, prev) =
                reference_game(&mut rng, team_size, true, &mut next_state, &mut air_times);
            write_reference_game(&mut text, team_size, true, &snap, prev, &air_times);
            let layout = ObsKind::GigalearnPadder.layout(team_size, true, room);
            let places = if room == 0 { team_size } else { room };
            for me in 0..snap.cars.len() {
                let mut out = vec![0.0; layout.size];
                ObsKind::GigalearnPadder.build(&snap, me, prev, room, &mut out);
                let _ = writeln!(
                    text,
                    "obs car {me} teamsize {places} size {} {}",
                    layout.size,
                    reference_floats(&out)
                );
            }
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/gigalearn_padder_reference.txt"
        );
        if std::env::var_os("NITRO_WRITE_REFERENCE").is_some() {
            std::fs::write(path, &text).unwrap();
            return;
        }
        check_reference(&text, path, "gigalearn_padder");
    }

    /// GigaLearn's own observations and action masks on these games: integrations/gigalearn/ggl_check (main
    /// line) runs GigaLearn's AdvancedObs.cpp, DefaultObs.cpp and DefaultAction.cpp, unchanged, on this file
    /// and finds every number and mask the same. Games of 1v1 to 3v3 with every pad's state drawn, the six car
    /// states, demolitions, empty tanks and turtled cars. Needs the meshes for the pads' positions (GigaLearn
    /// orders pads by position, CommonValues::BOOST_LOCATIONS).
    #[test]
    fn gigalearn_reference_cases_match_the_committed_file() {
        use crate::actions::{ActionTable, MaskRule};
        use std::fmt::Write as _;
        if !crate::testing::meshes_loaded() {
            return;
        }
        let sim = crate::sim::Sim::new(1, true, 0);
        let mut text = String::from("gigalearn_reference 1\n");
        let positions: Vec<f32> = sim
            .pad_layout()
            .iter()
            .flat_map(|p| p[..3].to_vec())
            .collect();
        let _ = writeln!(text, "padpos {}", reference_floats(&positions));
        let table = ActionTable::new();
        for action in 0..90 {
            let _ = writeln!(
                text,
                "action {action} {}",
                reference_floats(&table.controls(action).to_floats())
            );
        }
        let mut rng = fastrand::Rng::with_seed(20261117);
        let (mut next_state, mut air_times) = (0usize, Vec::new());
        let mut drawn = 0usize;
        for team_size in [1usize, 1, 2, 2, 3, 3] {
            let (mut snap, prev) =
                reference_game(&mut rng, team_size, true, &mut next_state, &mut air_times);
            for car in &mut snap.cars {
                drawn += 1;
                if drawn % 5 == 1 {
                    car.boost = 0.0;
                }
                if drawn % 7 == 3 {
                    // Turtled: on its roof near the floor, not on its wheels.
                    car.on_ground = false;
                    car.body.pos.z = 40.0;
                    car.body.rot =
                        Mat3A::from_rotation_x(PI) * Mat3A::from_rotation_z(rng.f32() * 6.0);
                }
            }
            write_reference_game(&mut text, team_size, true, &snap, prev, &air_times);
            for me in 0..snap.cars.len() {
                for kind in [ObsKind::GigalearnAdvanced, ObsKind::GigalearnDefault] {
                    let mut out = vec![0.0; kind.layout(team_size, true, 0).size];
                    kind.build(&snap, me, prev, 0, &mut out);
                    let _ = writeln!(
                        text,
                        "obs {} car {me} {}",
                        kind.name(),
                        reference_floats(&out)
                    );
                }
                let mut mask = vec![0u8; 90];
                table.write_mask(MaskRule::GigaLearn, &snap.cars[me], &mut mask);
                let _ = writeln!(
                    text,
                    "mask car {me} turtled {} {}",
                    u8::from(snap.cars[me].is_turtled()),
                    mask.iter().map(u8::to_string).collect::<Vec<_>>().join(" ")
                );
            }
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/gigalearn_reference.txt"
        );
        if std::env::var_os("NITRO_WRITE_REFERENCE").is_some() {
            std::fs::write(path, &text).unwrap();
            return;
        }
        check_reference(&text, path, "gigalearn");
    }

    #[test]
    fn observations_are_finite() {
        for kind in [
            ObsKind::Native,
            ObsKind::NitroV2,
            ObsKind::GigalearnAdvanced,
            ObsKind::GigalearnDefault,
            ObsKind::GigalearnPadder,
        ] {
            let snap = snapshot(2);
            for me in 0..4 {
                assert!(obs(kind, &snap, me).iter().all(|v| v.is_finite()));
            }
        }
    }
}
