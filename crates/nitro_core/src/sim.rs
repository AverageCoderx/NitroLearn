//! Thin adapter over RocketSim v3. Nothing outside this module uses RocketSim types.

use rocketsim::{
    Arena, ArenaConfig, ArenaEvent, ArenaMemWeightMode, BallState, BoostPadConfig, BoostPadState,
    CarBodyConfig, CarControls, CarCore, CarState, GameMode, UserInfoType, Vec3A,
};

use crate::snapshot::{
    BALL_FLOOR_Z, Body, CarDetail, CarSnapshot, Controls, GameSnapshot, MAX_CARS, NUM_BOOST_PADS,
    PadSnapshot, StepEvents, Team,
};

/// Floor contact: an arena surface facing up (normal z above this) and low down. The curve from the
/// floor up to a wall reaches 45° about 77 uu up, so the heights keep only its lowest part.
pub(crate) const FLOOR_NORMAL_Z: f32 = 0.7;
/// A wheel's hit point or the ball's contact point below this height (uu) is on the floor.
pub(crate) const FLOOR_CONTACT_Z: f32 = 60.0;
/// The car body's contact point below this height (uu) is on the floor.
pub(crate) const BODY_FLOOR_Z: f32 = 40.0;
/// Beyond this |y| (uu, the goal lines) the ball is inside a goal.
const GOAL_LINE_Y: f32 = 5120.0;
/// A car's `flip_time` at an episode start (s). RocketSim 0.3 counts it from the car's last flip change, also
/// on the ground, and holds the car's air pitch for 0.3 s after one. An episode's car has not flipped
/// for a while (the game's kickoff countdown is 3 s), so its pitch is free from the first tick, as in 0.2.9.
pub(crate) const SETTLED_FLIP_TIME: f32 = 3.0;
/// `TickFacts::last_hitter` when no car touched the ball.
const NO_CAR: u8 = 0xFF;
/// A tick's first car-car events, in event order, set the cars' latest bumps (`details`).
const CONTACT_BUMPS: usize = 8;

fn on_floor(point: Vec3A, normal: Vec3A, max_z: f32) -> bool {
    normal.z > FLOOR_NORMAL_Z && point.z < max_z
}

/// Most car-car events one tick can hold; more (every car against most others at once) keeps the first.
pub const MAX_BUMPS: usize = 32;

/// What one physics tick did, as `Sim::absorb` needs it, in RocketSim's event order. Bits are per car index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickFacts {
    /// The arena's tick count after the tick.
    pub tick: u64,
    /// The ball's position after the tick (uu).
    pub ball_pos: [f32; 3],
    /// `is_ball_scored()` after the tick.
    pub ball_scored: u8,
    /// Some ball-world contact this tick was on the floor (`on_floor`, `FLOOR_CONTACT_Z`).
    pub ball_floor: u8,
    /// Cars with a car-ball contact this tick.
    pub hit_mask: u8,
    /// The car of the tick's last car-ball event, 0xFF for none.
    pub last_hitter: u8,
    /// Cars whose body touched the arena this tick.
    pub body_arena: u8,
    /// Cars whose body touched the arena on the floor (`BODY_FLOOR_Z`).
    pub body_floor: u8,
    /// Cars (not demolished) with a wheel on the arena after the tick.
    pub wheel_arena: u8,
    /// Cars (not demolished) with a wheel on the arena's floor after the tick.
    pub wheel_floor: u8,
    /// Car-car events this tick (at most `MAX_BUMPS`).
    pub n_bumps: u8,
    /// (bumper, victim, is_demo) of each car-car event, in event order.
    pub bumps: [[u8; 3]; MAX_BUMPS],
}

impl Default for TickFacts {
    fn default() -> Self {
        Self {
            tick: 0,
            ball_pos: [0.0; 3],
            ball_scored: 0,
            ball_floor: 0,
            hit_mask: 0,
            last_hitter: NO_CAR,
            body_arena: 0,
            body_floor: 0,
            wheel_arena: 0,
            wheel_floor: 0,
            n_bumps: 0,
            bumps: [[0; 3]; MAX_BUMPS],
        }
    }
}

pub struct Sim {
    arena: Arena,
    teams: Vec<Team>,
    /// Per car, this episode: the tick of its latest ball touch and the ball's position then; the car it
    /// last bumped or was bumped by, and the tick (for `details`).
    hits: Vec<Option<(u64, Vec3A)>>,
    contacts: Vec<Option<(usize, u64)>>,
}

impl Sim {
    /// A soccar arena with `team_size` cars per team, ordered Blue, Orange, Blue, Orange, ...; without
    /// `opponents`, only the Blue cars.
    pub fn new(team_size: usize, opponents: bool, seed: u64) -> Self {
        let (arena, teams) = build_arena(team_size, opponents, seed);
        let cars = teams.len();
        Self {
            arena,
            teams,
            hits: vec![None; cars],
            contacts: vec![None; cars],
        }
    }

    pub fn num_cars(&self) -> usize {
        self.teams.len()
    }

    /// Car `car`'s team.
    pub fn team(&self, car: usize) -> Team {
        self.teams[car]
    }

    /// The arena's RocketSim (tests).
    #[cfg(test)]
    pub(crate) fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }

    /// Each boost pad's position (x, y, z) and 1 for a big pad, 0 for a small one, in the snapshot's order.
    pub fn pad_layout(&self) -> Vec<[f32; 4]> {
        (0..NUM_BOOST_PADS)
            .map(|i| {
                let pad = self.pad_config(i);
                [
                    pad.pos.x,
                    pad.pos.y,
                    pad.pos.z,
                    f32::from(u8::from(pad.is_big)),
                ]
            })
            .collect()
    }

    pub fn set_controls(&mut self, car: usize, controls: Controls) {
        self.arena.set_car_controls(car, car_controls(controls));
    }

    /// Advances `ticks` physics ticks, adding what happened to `events`.
    pub fn step(&mut self, ticks: u32, events: &mut StepEvents) {
        for _ in 0..ticks {
            let facts = cpu_tick(&mut self.arena, self.teams.len());
            self.absorb(&facts, events);
        }
    }

    /// Adds what one physics tick did to `events` and to the latest touches and bumps.
    pub fn absorb(&mut self, facts: &TickFacts, events: &mut StepEvents) {
        if facts.last_hitter != NO_CAR {
            events.last_touch = Some(usize::from(facts.last_hitter));
        }
        let bumps = &facts.bumps[..usize::from(facts.n_bumps)];
        for &[bumper, victim, is_demo] in bumps {
            let (bumper, victim) = (usize::from(bumper), usize::from(victim));
            if self.teams[bumper] != self.teams[victim] {
                events.cars[bumper].bumped_opponent = true;
                events.cars[bumper].demoed_opponent |= is_demo != 0;
            } else {
                events.cars[bumper].bumped_teammate = true;
                events.cars[victim].bumped_teammate = true;
            }
        }
        let (tick, ball) = (facts.tick, Vec3A::from_array(facts.ball_pos));
        let hit = u32::from(facts.hit_mask);
        for (car, slot) in self.hits.iter_mut().enumerate() {
            if hit & (1 << car) != 0 {
                *slot = Some((tick, ball));
            }
        }
        for &[bumper, victim, _] in &bumps[..bumps.len().min(CONTACT_BUMPS)] {
            let (bumper, victim) = (usize::from(bumper), usize::from(victim));
            self.contacts[bumper] = Some((victim, tick));
            self.contacts[victim] = Some((bumper, tick));
        }
        if ball.z < BALL_FLOOR_Z {
            events.ball_on_floor = true;
        }
        // After the goal's tick a touch still counts, but not in the tick-ordered contacts.
        let after_goal = events.goal_tick.is_some_and(|goal| tick > goal);
        if events.goal_tick.is_none() && facts.ball_scored != 0 {
            events.goal_tick = Some(tick);
        }
        if after_goal {
            for (car, car_events) in events.cars.iter_mut().enumerate() {
                car_events.add_tick_after_goal(hit & (1 << car) != 0);
            }
            return;
        }
        let ball_down =
            (facts.ball_floor != 0 || ball.z < BALL_FLOOR_Z) && ball.y.abs() <= GOAL_LINE_Y;
        if ball_down {
            events.ball_on_field_floor = true;
            events.ball_field_floor_last_tick = Some(tick);
        }
        let on_floor = u32::from(facts.wheel_floor | facts.body_floor);
        let on_arena = u32::from(facts.wheel_arena | facts.body_arena);
        for (car, car_events) in events.cars.iter_mut().enumerate() {
            let bit = 1u32 << car;
            car_events.add_tick(
                tick,
                hit & bit != 0,
                on_floor & bit != 0,
                on_arena & bit != 0,
                ball_down,
            );
        }
    }

    /// A new episode: the latest touches and bumps are forgotten.
    pub fn forget(&mut self) {
        self.hits.fill(None);
        self.contacts.fill(None);
    }

    /// The arena's tick count.
    pub fn tick(&self) -> u64 {
        self.arena.tick_count()
    }

    /// Each car's state beyond the snapshot, into `events.details` (for rewards of one's own).
    pub fn details(&self, events: &mut StepEvents) {
        events.details.clear();
        events.details.extend((0..self.teams.len()).map(|index| {
            let car = self.car_state(index);
            let c = car.controls;
            let (hit, contact) = (self.hits[index], self.contacts[index]);
            CarDetail {
                controls: Controls {
                    throttle: c.throttle,
                    steer: c.steer,
                    pitch: c.pitch,
                    yaw: c.yaw,
                    roll: c.roll,
                    jump: c.jump,
                    boost: c.boost,
                    handbrake: c.handbrake,
                },
                wheel_contacts: car.wheels_with_contact().map(|w| w.is_some()),
                world_contact_normal: car.world_contact_normal(),
                is_jumping: car.is_jumping,
                is_flipping: car.is_flipping,
                is_auto_flipping: car.is_auto_flipping,
                is_boosting: car.is_boosting,
                jump_time: car.jump_time(),
                flip_time: car.flip_time,
                air_time: car.air_time,
                air_time_since_jump: car.air_time_since_jump,
                boosting_time: car.boosting_time,
                flip_rel_torque: car.flip_rel_torque,
                handbrake: car.handbrake_val,
                last_hit_tick: hit.map(|(tick, _)| tick),
                last_hit_ball_pos: hit.map_or(Vec3A::ZERO, |(_, pos)| pos),
                last_contact_car: contact.map(|(other, _)| other),
                last_contact_tick: contact.map(|(_, tick)| tick),
                bump_cooldown: car.bump_cooldown_timer,
            }
        }));
    }

    pub fn read(&self, snapshot: &mut GameSnapshot) {
        let ball = self.ball_state();
        snapshot.ball = body_of(&ball.phys);
        snapshot.cars.clear();
        for (index, &team) in self.teams.iter().enumerate() {
            let car = self.car_state(index);
            snapshot.cars.push(CarSnapshot {
                team,
                body: body_of(&car.phys),
                boost: car.boost,
                on_ground: car.is_on_ground(),
                has_jumped: car.has_jumped,
                has_double_jumped: car.has_double_jumped,
                has_flipped: car.has_flipped,
                can_flip: car.has_flip_or_jump(),
                is_demoed: car.is_demoed,
                is_supersonic: car.is_supersonic,
                demo_timer: if car.is_demoed {
                    car.demo_respawn_timer
                } else {
                    0.0
                },
                flip_time_left: CarSnapshot::flip_time_left_from(
                    car.is_on_ground(),
                    car.has_jumped,
                    car.has_flip_or_jump(),
                    car.air_time_since_jump,
                ),
            });
        }
        for (index, pad) in snapshot.pads.iter_mut().enumerate() {
            let state = self.pad_state(index);
            *pad = PadSnapshot {
                active: state.is_active(),
                cooldown: state.cooldown,
            };
        }
        // The ball sits in the goal of the team that conceded.
        snapshot.goal_scored = self
            .is_ball_scored()
            .then(|| Team::from_field_y(snapshot.ball.pos.y).opposite());
    }

    /// A kickoff with the spots shuffled by `seed`, so a start depends only on the caller's random stream
    /// (the arena's own generator also serves demolition respawns, which differ between policies).
    pub fn reset_to_kickoff(&mut self, seed: u64) {
        self.arena.reset_to_random_kickoff(Some(seed));
        for car in 0..self.teams.len() {
            self.arena
                .edit_car_state_full(car, |s| s.flip_time = SETTLED_FLIP_TIME);
        }
    }

    /// Teleports the ball and every car (body, boost, ground contact), dropping cached contacts so the
    /// next tick behaves like a freshly built arena.
    pub fn place(&mut self, ball: Body, cars: &[(Body, f32, bool)]) {
        self.set_ball_state(BallState {
            phys: phys_of(ball),
            ..BallState::DEFAULT
        });
        for (index, &(body, boost, on_ground)) in cars.iter().enumerate() {
            self.put_car(index, body, boost, on_ground);
        }
        self.clear_persistent_manifolds();
    }

    /// A car at `body` with `boost`, fresh jump and flip state, its contacts dropped. RocketSim 0.3's setter
    /// leaves a placed car off the ground until its next tick; NitroLearn keeps the caller's `on_ground` (as
    /// 0.2.9 took it from the state), so a car placed on its wheels can jump on its first tick.
    fn put_car(&mut self, car: usize, body: Body, boost: f32, on_ground: bool) {
        let mut core = CarCore::grounded(phys_of(body), boost);
        core.flip_time = SETTLED_FLIP_TIME;
        self.arena.set_car_state(car, core);
        self.arena
            .edit_car_state_full(car, |s| s.set_contact_flags(on_ground, [false; 4]));
    }

    // RocketSim's calls.

    fn car_state(&self, car: usize) -> &CarState {
        self.arena.get_car_state(car)
    }

    fn ball_state(&self) -> &BallState {
        self.arena.get_ball_state()
    }

    fn pad_config(&self, pad: usize) -> &BoostPadConfig {
        self.arena.get_boost_pad_config(pad)
    }

    fn pad_state(&self, pad: usize) -> BoostPadState {
        self.arena.get_boost_pad_state(pad)
    }

    fn is_ball_scored(&self) -> bool {
        self.arena.is_ball_scored()
    }

    fn set_ball_state(&mut self, state: BallState) {
        self.arena.set_ball_state(state);
    }

    fn clear_persistent_manifolds(&mut self) {
        self.arena.clear_persistent_manifolds();
    }
}

/// One physics tick of `arena` (with `cars` cars), and what it did. Bits per car; the events in RocketSim's
/// order.
pub(crate) fn cpu_tick(arena: &mut Arena, cars: usize) -> TickFacts {
    let mut facts = TickFacts::default();
    let mut ball_floor = false;
    for event in arena.step_tick() {
        match *event {
            ArenaEvent::CarHitBall(hit_ball) => {
                facts.last_hitter = hit_ball.car_idx as u8;
                facts.hit_mask |= 1 << hit_ball.car_idx;
            }
            ArenaEvent::CarHitCar(hit_car) => {
                // RocketSim 0.3 keeps a bump's cooldown for that victim only, so a car can bump
                // several cars in one tick: at most cars × (cars − 1) events, 30 at 3v3. A 4v4 tick with
                // more than MAX_BUMPS (every car against most others at once) keeps the first ones.
                let count = usize::from(facts.n_bumps);
                if count == MAX_BUMPS {
                    continue;
                }
                facts.bumps[count] = [
                    hit_car.bumper_car_idx as u8,
                    hit_car.victim_car_idx as u8,
                    u8::from(hit_car.is_demo),
                ];
                facts.n_bumps += 1;
            }
            ArenaEvent::CarHitWorld(hit_world) => {
                facts.body_arena |= 1 << hit_world.car_idx;
                if on_floor(
                    hit_world.contact_point,
                    hit_world.contact_normal,
                    BODY_FLOOR_Z,
                ) {
                    facts.body_floor |= 1 << hit_world.car_idx;
                }
            }
            ArenaEvent::BallHitWorld(hit_world) => {
                ball_floor |= on_floor(
                    hit_world.contact_point,
                    hit_world.contact_normal,
                    FLOOR_CONTACT_Z,
                );
            }
            _ => {}
        }
    }
    facts.ball_floor = u8::from(ball_floor);
    facts.tick = arena.tick_count();
    facts.ball_pos = arena.get_ball_state().phys.pos.to_array();
    facts.ball_scored = u8::from(arena.is_ball_scored());
    for car in 0..cars {
        let state = arena.get_car_state(car);
        // A demolished car keeps its last wheel contacts, but it touches nothing.
        if state.is_demoed {
            continue;
        }
        // Wheels on the ball or a car hit a body with its own `user_info`, not the arena's.
        for wheel in state.wheels_with_contact().iter().flatten() {
            if wheel.user_info == UserInfoType::None {
                facts.wheel_arena |= 1 << car;
                if on_floor(wheel.hit_point, wheel.hit_normal, FLOOR_CONTACT_Z) {
                    facts.wheel_floor |= 1 << car;
                }
            }
        }
    }
    facts
}

/// A soccar arena's config in the memory mode NitroLearn measured with: RocketSim 0.3.1 made Balanced the
/// default, Heavy stays explicit until a setting chooses.
pub(crate) fn soccar_config() -> ArenaConfig {
    let mut config = ArenaConfig::new(GameMode::Soccar);
    config.mem_weight_mode = ArenaMemWeightMode::Heavy;
    config
}

/// The arena `Sim::new` describes and its cars' teams.
fn build_arena(team_size: usize, opponents: bool, seed: u64) -> (Arena, Vec<Team>) {
    let config = soccar_config().with_rng_seed(seed);
    let mut arena = Arena::new_with_config(config);
    let sides: &[Team] = if opponents {
        &[Team::Blue, Team::Orange]
    } else {
        &[Team::Blue]
    };
    let mut teams = Vec::with_capacity(team_size * sides.len());
    for _ in 0..team_size {
        for &team in sides {
            arena.add_car(to_rs_team(team), CarBodyConfig::OCTANE);
            teams.push(team);
        }
    }
    assert_eq!(arena.num_boost_pads(), NUM_BOOST_PADS);
    // A tick's facts hold one bit per car.
    assert!(teams.len() <= MAX_CARS, "at most {MAX_CARS} cars");
    (arena, teams)
}

/// NitroLearn's controls as RocketSim's.
pub(crate) fn car_controls(controls: Controls) -> CarControls {
    CarControls {
        throttle: controls.throttle,
        steer: controls.steer,
        pitch: controls.pitch,
        yaw: controls.yaw,
        roll: controls.roll,
        jump: controls.jump,
        boost: controls.boost,
        handbrake: controls.handbrake,
    }
}

fn to_rs_team(team: Team) -> rocketsim::Team {
    match team {
        Team::Blue => rocketsim::Team::Blue,
        Team::Orange => rocketsim::Team::Orange,
    }
}

fn body_of(phys: &rocketsim::PhysState) -> Body {
    Body {
        pos: phys.pos,
        rot: phys.rot_mat,
        vel: phys.vel,
        ang_vel: phys.ang_vel,
    }
}

fn phys_of(body: Body) -> rocketsim::PhysState {
    rocketsim::PhysState {
        pos: body.pos,
        rot_mat: body.rot,
        vel: body.vel,
        ang_vel: body.ang_vel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::meshes_loaded;
    use rocketsim::{Mat3A, Vec3A};

    /// The arena of a `Sim`.
    fn cpu(sim: &Sim) -> &Arena {
        &sim.arena
    }

    /// `Sim::step` as it was before it was split into `cpu_tick` and `absorb`, kept word for word (with
    /// `sim` for `self`) to show the split changed nothing.
    fn step_legacy(sim: &mut Sim, ticks: u32, events: &mut StepEvents) {
        for _ in 0..ticks {
            let arena = &mut sim.arena;
            // The touches, bumps and arena contacts of the tick, kept once it is done (the events borrow
            // the arena). One bit per car.
            let (mut hit, mut body_arena, mut body_floor) = (0u32, 0u32, 0u32);
            let mut ball_floor = false;
            let mut bumps = [(0usize, 0usize); 8];
            let mut count = 0;
            for event in arena.step_tick() {
                match *event {
                    ArenaEvent::CarHitBall(hit_ball) => {
                        events.last_touch = Some(hit_ball.car_idx);
                        hit |= 1 << hit_ball.car_idx;
                    }
                    ArenaEvent::CarHitCar(hit_car) => {
                        let (bumper, victim) = (hit_car.bumper_car_idx, hit_car.victim_car_idx);
                        if count < bumps.len() {
                            bumps[count] = (bumper, victim);
                            count += 1;
                        }
                        if sim.teams[bumper] != sim.teams[victim] {
                            events.cars[bumper].bumped_opponent = true;
                            events.cars[bumper].demoed_opponent |= hit_car.is_demo;
                        } else {
                            events.cars[bumper].bumped_teammate = true;
                            events.cars[victim].bumped_teammate = true;
                        }
                    }
                    ArenaEvent::CarHitWorld(hit_world) => {
                        body_arena |= 1 << hit_world.car_idx;
                        if on_floor(
                            hit_world.contact_point,
                            hit_world.contact_normal,
                            BODY_FLOOR_Z,
                        ) {
                            body_floor |= 1 << hit_world.car_idx;
                        }
                    }
                    ArenaEvent::BallHitWorld(hit_world) => {
                        ball_floor |= on_floor(
                            hit_world.contact_point,
                            hit_world.contact_normal,
                            FLOOR_CONTACT_Z,
                        );
                    }
                    _ => {}
                }
            }
            let (tick, ball) = (arena.tick_count(), arena.get_ball_state().phys.pos);
            for (car, slot) in sim.hits.iter_mut().enumerate() {
                if hit & (1 << car) != 0 {
                    *slot = Some((tick, ball));
                }
            }
            for &(bumper, victim) in &bumps[..count] {
                sim.contacts[bumper] = Some((victim, tick));
                sim.contacts[victim] = Some((bumper, tick));
            }
            if ball.z < BALL_FLOOR_Z {
                events.ball_on_floor = true;
            }
            // After the goal's tick a touch still counts, but not in the tick-ordered contacts.
            let after_goal = events.goal_tick.is_some_and(|goal| tick > goal);
            if events.goal_tick.is_none() && arena.is_ball_scored() {
                events.goal_tick = Some(tick);
            }
            if after_goal {
                for (car, car_events) in events.cars.iter_mut().enumerate() {
                    car_events.add_tick_after_goal(hit & (1 << car) != 0);
                }
                continue;
            }
            let ball_down = (ball_floor || ball.z < BALL_FLOOR_Z) && ball.y.abs() <= GOAL_LINE_Y;
            if ball_down {
                events.ball_on_field_floor = true;
                events.ball_field_floor_last_tick = Some(tick);
            }
            for (car, car_events) in events.cars.iter_mut().enumerate() {
                let state = arena.get_car_state(car);
                let (mut wheel_arena, mut wheel_floor) = (false, false);
                // A demolished car keeps its last wheel contacts, but it touches nothing.
                if !state.is_demoed {
                    // Wheels on the ball or a car hit a body with its own `user_info`, not the arena's.
                    for wheel in state.wheels_with_contact().iter().flatten() {
                        if wheel.user_info == UserInfoType::None {
                            wheel_arena = true;
                            wheel_floor |=
                                on_floor(wheel.hit_point, wheel.hit_normal, FLOOR_CONTACT_Z);
                        }
                    }
                }
                let bit = 1u32 << car;
                car_events.add_tick(
                    tick,
                    hit & bit != 0,
                    wheel_floor || body_floor & bit != 0,
                    wheel_arena || body_arena & bit != 0,
                    ball_down,
                );
            }
        }
    }

    #[test]
    fn pads_are_rocketsims_soccar_pads_sorted_and_mirror_symmetric() {
        if !meshes_loaded() {
            return;
        }
        let sim = Sim::new(1, true, 0);
        let pos = |i: usize| cpu(&sim).get_boost_pad_config(i).pos;
        let pads = rocketsim::consts::boost_pads::get_locations;
        let mut rocketsims: Vec<(f32, f32, bool)> = [true, false]
            .into_iter()
            .flat_map(|big| {
                pads(GameMode::Soccar, big)
                    .iter()
                    .map(move |p| (p.x, p.y, big))
            })
            .collect();
        rocketsims.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));
        assert_eq!(rocketsims.len(), NUM_BOOST_PADS);
        for (i, &(x, y, big)) in rocketsims.iter().enumerate() {
            if i > 0 {
                let (a, b) = (pos(i - 1), pos(i));
                assert!(
                    a.y < b.y || (a.y == b.y && a.x < b.x),
                    "pads {} and {i} out of order",
                    i - 1
                );
            }
            let mirrored = pos(NUM_BOOST_PADS - 1 - i);
            assert_eq!((mirrored.x, mirrored.y), (-pos(i).x, -pos(i).y), "pad {i}");
            let config = cpu(&sim).get_boost_pad_config(i);
            assert_eq!(
                (config.pos.x, config.pos.y, config.is_big),
                (x, y, big),
                "pad {i}"
            );
        }
    }

    #[test]
    fn details_follow_the_cars_state_and_the_last_touch_is_kept() {
        if !meshes_loaded() {
            return;
        }
        let mut sim = Sim::new(1, true, 0);
        let at = |x, y, z| Body {
            pos: Vec3A::new(x, y, z),
            ..Body::RESTING
        };
        // Blue drives into a ball just in front of it (an unturned car faces +x); Orange jumps.
        sim.place(
            at(250.0, -500.0, 93.15),
            &[
                (at(0.0, -500.0, 17.0), 100.0, true),
                (at(0.0, 2000.0, 17.0), 100.0, true),
            ],
        );
        let drive = Controls {
            throttle: 1.0,
            boost: true,
            ..Controls::default()
        };
        sim.set_controls(0, drive);
        sim.set_controls(
            1,
            Controls {
                jump: true,
                ..Controls::default()
            },
        );
        let mut events = StepEvents::default();
        events.clear(2);
        sim.step(60, &mut events);
        sim.details(&mut events);
        let (blue, orange) = (events.details[0], events.details[1]);
        assert_eq!(blue.controls, drive);
        assert!(blue.is_boosting && blue.boosting_time > 0.0 && blue.wheel_contacts == [true; 4]);
        assert!(
            orange.air_time > 0.0 && orange.wheel_contacts == [false; 4] && orange.jump_time > 0.0
        );
        assert_eq!(events.last_touch, Some(0), "Blue hit the ball");
        let at_hit = blue.last_hit_tick.expect("the touch is kept");
        assert!(at_hit <= sim.tick() && blue.last_hit_ball_pos.x > 200.0);
        assert!(orange.last_hit_tick.is_none() && orange.last_contact_car.is_none());
        sim.forget();
        sim.details(&mut events);
        assert!(
            events.detail(0).last_hit_tick.is_none(),
            "a new episode forgets it"
        );
        events.clear(2);
        assert_eq!(events.last_touch, Some(0), "kept from step to step");
    }

    /// A lone car: plays `steps` steps of 4 ticks from `ball` and `car` (on the ground or not), driven by
    /// `controls`; what happened in each step and the state it ended in.
    fn play(
        ball: Body,
        car: (Body, bool),
        controls: Controls,
        steps: usize,
    ) -> Vec<(StepEvents, GameSnapshot)> {
        let mut sim = Sim::new(1, false, 0);
        sim.place(ball, &[(car.0, 50.0, car.1)]);
        sim.set_controls(0, controls);
        (0..steps)
            .map(|_| {
                let (mut events, mut snap) = (StepEvents::default(), GameSnapshot::default());
                events.clear(1);
                sim.step(4, &mut events);
                events.tick = sim.tick();
                sim.read(&mut snap);
                (events, snap)
            })
            .collect()
    }

    fn at(x: f32, y: f32, z: f32) -> Body {
        Body {
            pos: Vec3A::new(x, y, z),
            ..Body::RESTING
        }
    }

    #[test]
    fn a_resting_car_is_on_the_floor_and_a_car_on_a_side_wall_is_not() {
        if !meshes_loaded() {
            return;
        }
        let ball_far = at(0.0, 3000.0, 93.15);
        for (events, _) in play(
            ball_far,
            (at(0.0, -2000.0, 17.0), true),
            Controls::default(),
            5,
        ) {
            let car = events.cars[0];
            assert!(car.on_floor && car.on_arena && !car.touched_ball);
            assert_eq!(
                car.first_arena_tick,
                Some(events.tick - 3),
                "from the step's first tick"
            );
        }
        // Nose up the right side wall, wheels on it (the car's up points into the field).
        let wall = Body {
            rot: Mat3A::from_rotation_y(-std::f32::consts::FRAC_PI_2),
            ..at(4096.0 - 17.0, 0.0, 800.0)
        };
        for (events, snap) in play(ball_far, (wall, true), Controls::default(), 5) {
            let car = events.cars[0];
            assert!(car.on_arena && !car.on_floor, "{:?}", snap.cars[0].body.pos);
        }
    }

    /// A jump from the wheels keeps the car on the ground for its first 6 ticks however long it is held:
    /// the bot reads a jump's start so (`nitro_bot::packet::GROUND_JUMP_TICKS`), measured again on
    /// RocketSim 0.3. A car placed on its wheels can jump on its first tick (`Sim::place`).
    #[test]
    fn a_jump_keeps_the_wheels_down_for_six_ticks() {
        if !meshes_loaded() {
            return;
        }
        for hold in [1, 3, 12, 24] {
            let mut sim = Sim::new(1, false, 0);
            sim.place(
                at(0.0, 3000.0, 93.15),
                &[(at(0.0, -2000.0, 17.0), 50.0, true)],
            );
            let mut snap = GameSnapshot::default();
            sim.read(&mut snap);
            assert!(
                snap.cars[0].on_ground && snap.cars[0].can_flip,
                "placed on its wheels"
            );
            let mut grounded = 0;
            for tick in 0..40 {
                sim.set_controls(
                    0,
                    Controls {
                        jump: tick < hold,
                        ..Controls::default()
                    },
                );
                sim.step(1, &mut StepEvents::default());
                sim.read(&mut snap);
                assert!(
                    snap.cars[0].has_jumped,
                    "hold {hold}: the jump starts on the first tick"
                );
                if snap.cars[0].on_ground {
                    grounded += 1;
                } else {
                    break;
                }
            }
            assert_eq!(grounded, 6, "hold {hold}");
        }
    }

    #[test]
    fn wheels_on_the_ball_are_neither_the_floor_nor_the_arena() {
        if !meshes_loaded() {
            return;
        }
        // Upside down under a ball, rising into it: a flip reset contact (wheels on the ball).
        let mut sim = Sim::new(1, false, 0);
        let car = Body {
            rot: Mat3A::from_rotation_x(std::f32::consts::PI),
            vel: Vec3A::new(0.0, 0.0, 150.0),
            ..at(0.0, 0.0, 880.0)
        };
        sim.place(at(8.75, 0.0, 1000.0), &[(car, 50.0, false)]);
        let mut events = StepEvents::default();
        events.clear(1);
        let mut on_ball = 0;
        for _ in 0..40 {
            sim.step(1, &mut events);
            let state = cpu(&sim).get_car_state(0);
            let mut wheels = state.wheels_with_contact().iter().flatten();
            if state.is_on_ground() && wheels.all(|w| w.user_info == UserInfoType::Ball) {
                on_ball += 1;
            }
        }
        assert!(on_ball > 20, "{on_ball} ticks with the wheels on the ball");
        let car = events.cars[0];
        assert!(!car.on_floor && !car.on_arena && car.first_arena_tick.is_none());
        assert!(!car.touched_ball_on_floor);
        assert!(!events.ball_on_floor && !events.ball_on_field_floor);
    }

    #[test]
    fn a_body_on_the_floor_is_on_the_floor_and_against_the_ceiling_only_on_the_arena() {
        if !meshes_loaded() {
            return;
        }
        let ball_far = at(0.0, 3000.0, 93.15);
        // Dropped on its roof: no wheel touches anything, so the body's contacts tell.
        let roof = Body {
            rot: Mat3A::from_rotation_x(std::f32::consts::PI),
            ..at(0.0, 0.0, 60.0)
        };
        let mut sim = Sim::new(1, false, 0);
        sim.place(ball_far, &[(roof, 50.0, false)]);
        sim.step(60, &mut StepEvents::default());
        for _ in 0..5 {
            let mut events = StepEvents::default();
            events.clear(1);
            sim.step(4, &mut events);
            let state = cpu(&sim).get_car_state(0);
            assert!(state.wheels_with_contact().iter().all(Option::is_none));
            assert!(events.cars[0].on_floor && events.cars[0].on_arena);
        }
        // Rising roof first into the ceiling.
        let rising = Body {
            vel: Vec3A::new(0.0, 0.0, 600.0),
            ..at(0.0, 0.0, 1960.0)
        };
        let steps = play(ball_far, (rising, false), Controls::default(), 6);
        assert!(steps.iter().any(|(e, _)| e.cars[0].on_arena));
        assert!(steps.iter().all(|(e, _)| !e.cars[0].on_floor));
    }

    #[test]
    fn touches_from_the_floor_from_a_wall_and_in_the_air_are_told_apart() {
        if !meshes_loaded() {
            return;
        }
        let throttle = Controls {
            throttle: 1.0,
            ..Controls::default()
        };
        // Driving into a ball on the floor (which sits on the field's floor until then).
        let steps = play(
            at(250.0, -500.0, 93.15),
            (at(0.0, -500.0, 17.0), true),
            throttle,
            15,
        );
        let touch = steps
            .iter()
            .position(|(e, _)| e.cars[0].touched_ball)
            .expect("a touch");
        let car = steps[touch].0.cars[0];
        assert!(car.touched_ball_on_floor && !car.touched_ball_in_air && car.on_floor);
        assert!(
            steps[..=touch]
                .iter()
                .all(|(e, _)| e.ball_on_field_floor && e.ball_on_floor)
        );

        // Moving into a ball in the air.
        let flying = Body {
            vel: Vec3A::new(1000.0, 0.0, 0.0),
            ..at(0.0, 0.0, 500.0)
        };
        let steps = play(
            at(200.0, 0.0, 520.0),
            (flying, false),
            Controls::default(),
            10,
        );
        let touch = steps
            .iter()
            .find(|(e, _)| e.cars[0].touched_ball)
            .expect("a touch");
        let car = touch.0.cars[0];
        assert!(car.touched_ball_in_air && !car.touched_ball_on_floor && !car.on_arena);
        assert!(car.first_touch_tick.is_some() && car.first_touch_tick <= car.last_touch_tick);
        assert_eq!(
            car.air_start_tick, car.first_air_touch_tick,
            "a starting touch"
        );
        assert!(steps.iter().all(|(e, _)| !e.ball_on_field_floor));

        // Driving up the right side wall from the floor into a ball falling down the wall: a touch from
        // the wall, neither from the floor nor in the air.
        let toward_wall = Body {
            vel: Vec3A::new(1400.0, 0.0, 0.0),
            ..at(3300.0, 0.0, 17.0)
        };
        let steps = play(at(4000.0, 0.0, 840.0), (toward_wall, true), throttle, 30);
        assert!(steps[0].0.cars[0].on_floor, "it starts on the floor");
        let touch = steps
            .iter()
            .position(|(e, _)| e.cars[0].touched_ball)
            .expect("a touch");
        let (events, snap) = &steps[touch];
        let car = events.cars[0];
        assert!(
            snap.cars[0].body.pos.z > 300.0,
            "{:?}",
            snap.cars[0].body.pos
        );
        assert!(car.on_arena && !car.on_floor);
        assert!(!car.touched_ball_on_floor && !car.touched_ball_in_air);
        assert_eq!(car.air_start_tick, None);
    }

    /// A car in the air hitting a ball that rests on the floor (a jump pop) touches it in the air, but that
    /// touch is no starting touch (`air_start_tick`).
    #[test]
    fn a_touch_in_the_air_on_a_ball_on_the_floor_is_no_starting_touch() {
        if !meshes_loaded() {
            return;
        }
        let flying = Body {
            vel: Vec3A::new(1000.0, 0.0, 0.0),
            ..at(0.0, 0.0, 120.0)
        };
        let steps = play(
            at(200.0, 0.0, 93.15),
            (flying, false),
            Controls::default(),
            10,
        );
        let touch = steps
            .iter()
            .position(|(e, _)| e.cars[0].touched_ball)
            .expect("a touch");
        let (events, snap) = &steps[touch];
        let car = events.cars[0];
        assert!(
            car.touched_ball_in_air && !car.on_arena,
            "{car:?} {:?}",
            snap.cars[0].body.pos
        );
        assert!(car.first_air_touch_tick.is_some() && car.air_start_tick.is_none());
        assert!(events.ball_on_field_floor);
        assert!(
            steps
                .iter()
                .all(|(e, _)| e.cars[0].air_start_tick.is_none())
        );
    }

    /// The tick the ball is first scored is kept, and after it a touch counts as a touch but not in the
    /// car's tick-ordered contacts, so play in the air is judged up to the goal.
    #[test]
    fn touches_after_the_goal_tick_count_but_not_in_the_ordered_contacts() {
        if !meshes_loaded() {
            return;
        }
        // The ball flying into Orange's goal; a car flying faster behind it, `gap` uu short of touching it.
        let chase = |gap: f32| {
            let mut sim = Sim::new(1, false, 0);
            let ball = Body {
                vel: Vec3A::new(0.0, 1500.0, 0.0),
                ..at(0.0, 5000.0, 300.0)
            };
            let car = Body {
                vel: Vec3A::new(0.0, 2300.0, 0.0),
                rot: Mat3A::from_rotation_z(std::f32::consts::FRAC_PI_2),
                ..at(0.0, 5000.0 - 165.0 - gap, 300.0)
            };
            sim.place(ball, &[(car, 50.0, false)]);
            let mut events = StepEvents::default();
            events.clear(1);
            sim.step(40, &mut events);
            sim.details(&mut events);
            assert!(cpu(&sim).is_ball_scored());
            (events.goal_tick.expect("a goal"), events)
        };
        let (goal, late) = chase(165.0);
        let hit = late.detail(0).last_hit_tick.expect("a touch");
        assert!(
            hit > goal,
            "touched on tick {hit}, after the goal on {goal}"
        );
        let car = late.cars[0];
        assert!(car.touched_ball && car.first_touch_tick.is_none() && !car.touched_ball_in_air);
        let (goal, early) = chase(40.0);
        let car = early.cars[0];
        assert!(car.touched_ball_in_air && car.first_touch_tick.is_some_and(|t| t <= goal));
        assert!(car.last_touch_tick.is_some_and(|t| t <= goal));
    }

    #[test]
    fn the_ball_on_the_floor_of_a_goal_is_not_on_the_field_floor() {
        if !meshes_loaded() {
            return;
        }
        let parked = (at(-3000.0, 0.0, 17.0), true);
        let field = play(at(1000.0, 1000.0, 300.0), parked, Controls::default(), 30);
        let landed = field
            .iter()
            .position(|(e, _)| e.ball_on_floor)
            .expect("it lands");
        assert!(field[landed].0.ball_on_field_floor);
        assert!(field[..landed].iter().all(|(e, _)| !e.ball_on_field_floor));

        // Landing on the bottom of the curve up to a side wall: on the floor, though the ball's centre stays
        // higher than on the flat floor.
        let curve = play(
            at(4096.0 - 180.0, 0.0, 400.0),
            parked,
            Controls::default(),
            40,
        );
        let landed = curve
            .iter()
            .position(|(e, _)| e.ball_on_field_floor)
            .expect("it lands");
        assert!(!curve[landed].0.ball_on_floor);
        assert!(curve[landed].1.ball.pos.x > 3800.0);

        let goal = play(at(0.0, 5500.0, 300.0), parked, Controls::default(), 30);
        assert!(
            goal.iter().any(|(e, _)| e.ball_on_floor),
            "it lands in the goal"
        );
        assert!(goal.iter().all(|(e, _)| !e.ball_on_field_floor));
        assert!(goal.iter().all(|(_, s)| s.ball.pos.y > 5120.0));
    }

    /// Random controls: mostly driving, sometimes boosting, jumping or drifting.
    fn random_controls(rng: &mut fastrand::Rng) -> Controls {
        let axis = |rng: &mut fastrand::Rng| [-1.0, -0.5, 0.0, 0.5, 1.0][rng.usize(..5)];
        Controls {
            throttle: if rng.u8(..4) == 0 { axis(rng) } else { 1.0 },
            steer: axis(rng),
            pitch: axis(rng),
            yaw: axis(rng),
            roll: axis(rng),
            jump: rng.u8(..6) == 0,
            boost: rng.bool(),
            handbrake: rng.u8(..10) == 0,
        }
    }

    /// A random start the same on both arenas: a kickoff, cars scattered round the ball, cars racing at each
    /// other (bumps and demolitions) or the ball flying into a goal; or a forgotten episode.
    fn random_reset(rng: &mut fastrand::Rng, sims: [&mut Sim; 2]) {
        let cars = sims[0].num_cars();
        let kind = rng.u8(..9);
        let kickoff_seed = rng.u64(..);
        let mut ball = at(
            rng.f32() * 6000.0 - 3000.0,
            rng.f32() * 8000.0 - 4000.0,
            93.15 + rng.f32() * 1000.0,
        );
        ball.vel = Vec3A::new(rng.f32() - 0.5, rng.f32() - 0.5, rng.f32() - 0.5) * 2000.0;
        let mut placed: Vec<(Body, f32, bool)> = (0..cars)
            .map(|_| {
                let on_ground = rng.bool();
                let mut car = at(
                    ball.pos.x + rng.f32() * 2000.0 - 1000.0,
                    ball.pos.y + rng.f32() * 2000.0 - 1000.0,
                    if on_ground {
                        17.0
                    } else {
                        100.0 + rng.f32() * 800.0
                    },
                );
                // Half of them head for the ball.
                let to_ball = ball.pos - car.pos;
                let yaw = if rng.bool() {
                    to_ball.y.atan2(to_ball.x)
                } else {
                    rng.f32() * std::f32::consts::TAU
                };
                car.rot = Mat3A::from_rotation_z(yaw);
                car.vel = car.rot.x_axis * (rng.f32() * 2300.0);
                (car, rng.f32() * 100.0, on_ground)
            })
            .collect();
        if kind == 2 {
            // Pairs racing head on at supersonic speed, any two cars (teammates too).
            let mut order: Vec<usize> = (0..cars).collect();
            rng.shuffle(&mut order);
            for (lane, pair) in order.chunks(2).enumerate() {
                let y = lane as f32 * 600.0 - 1500.0;
                let speed = 2000.0 + rng.f32() * 300.0;
                for (side, &car) in pair.iter().enumerate() {
                    let sign = if side == 0 { 1.0 } else { -1.0 };
                    placed[car].0 = Body {
                        rot: Mat3A::from_rotation_z(if side == 0 {
                            0.0
                        } else {
                            std::f32::consts::PI
                        }),
                        vel: Vec3A::new(sign * speed, 0.0, 0.0),
                        ..at(-sign * (150.0 + rng.f32() * 300.0), y, 17.0)
                    };
                    placed[car].2 = true;
                }
            }
            ball = at(0.0, 3000.0, 93.15);
        } else if kind == 7 {
            // Every car driving at a resting ball from a few hundred uu away.
            ball.pos.z = 93.15;
            ball.vel = Vec3A::ZERO;
            for (car, _, on_ground) in &mut placed {
                let angle = rng.f32() * std::f32::consts::TAU;
                let away = Vec3A::new(angle.cos(), angle.sin(), 0.0);
                *car = Body {
                    rot: Mat3A::from_rotation_z(angle + std::f32::consts::PI),
                    vel: away * -(800.0 + rng.f32() * 1200.0),
                    ..at(
                        ball.pos.x + away.x * (250.0 + rng.f32() * 300.0),
                        ball.pos.y + away.y * (250.0 + rng.f32() * 300.0),
                        17.0,
                    )
                };
                *on_ground = true;
            }
        } else if kind == 3 {
            let side = if rng.bool() { 1.0 } else { -1.0 };
            ball = Body {
                vel: Vec3A::new(rng.f32() * 400.0 - 200.0, side * 3000.0, 0.0),
                ..at(rng.f32() * 1000.0 - 500.0, side * 4700.0, 300.0)
            };
        }
        for sim in sims {
            match kind {
                0 => sim.reset_to_kickoff(kickoff_seed),
                1..=3 | 7 => sim.place(ball, &placed),
                6 => sim.forget(),
                _ => {}
            }
        }
    }

    /// The hits with the ball's position as bits, so a NaN compares too.
    fn hit_bits(sim: &Sim) -> Vec<Option<(u64, [u32; 3])>> {
        sim.hits
            .iter()
            .map(|h| h.map(|(tick, pos)| (tick, pos.to_array().map(f32::to_bits))))
            .collect()
    }

    /// `step` split into one `TickFacts` per tick and `absorb` gives exactly what the old loop gave,
    /// over random 1v1 to 4v4 play with kickoffs, teleports, bumps, demolitions and goals.
    #[test]
    fn the_split_step_matches_the_old_loop_in_random_play() {
        if !meshes_loaded() {
            return;
        }
        let (mut touches, mut bumps, mut teammates, mut demos, mut goals, mut past_goal) =
            (0, 0, 0, 0, 0, 0);
        let mut ticks_run = 0;
        for team_size in 1..=4 {
            for game in 0..16u64 {
                let seed = 1000 * team_size as u64 + game;
                let mut rng = fastrand::Rng::with_seed(seed);
                let mut split = Sim::new(team_size, true, seed);
                let mut old = Sim::new(team_size, true, seed);
                let cars = split.num_cars();
                let (mut split_events, mut old_events) =
                    (StepEvents::default(), StepEvents::default());
                let mut scored = false;
                for step in 0..70 {
                    if step == 0 || scored || rng.u8(..12) == 0 {
                        random_reset(&mut rng, [&mut split, &mut old]);
                    }
                    for car in 0..cars {
                        let controls = random_controls(&mut rng);
                        split.set_controls(car, controls);
                        old.set_controls(car, controls);
                    }
                    // Mostly a fresh step; sometimes the events run on over several (as in a test).
                    if step == 0 || rng.u8(..4) != 0 {
                        split_events.clear(cars);
                        old_events.clear(cars);
                    }
                    let ticks = rng.u32(1..=8);
                    split.step(ticks, &mut split_events);
                    step_legacy(&mut old, ticks, &mut old_events);
                    ticks_run += ticks;

                    let (a, b) = (&split_events, &old_events);
                    let when = format!("{team_size}v{team_size}, game {game}, step {step}");
                    assert_eq!(a.cars, b.cars, "{when}");
                    assert_eq!(a.ball_on_floor, b.ball_on_floor, "{when}");
                    assert_eq!(a.ball_on_field_floor, b.ball_on_field_floor, "{when}");
                    assert_eq!(
                        a.ball_field_floor_last_tick, b.ball_field_floor_last_tick,
                        "{when}"
                    );
                    assert_eq!(a.last_touch, b.last_touch, "{when}");
                    assert_eq!(a.goal_tick, b.goal_tick, "{when}");
                    assert_eq!(hit_bits(&split), hit_bits(&old), "{when}");
                    assert_eq!(split.contacts, old.contacts, "{when}");
                    let (mut snap_a, mut snap_b) =
                        (GameSnapshot::default(), GameSnapshot::default());
                    split.read(&mut snap_a);
                    old.read(&mut snap_b);
                    assert_eq!(format!("{snap_a:?}"), format!("{snap_b:?}"), "{when}");
                    // A new start after a goal most of the time, as an environment makes.
                    scored = snap_a.goal_scored.is_some() && rng.u8(..4) != 0;
                    let (mut details_a, mut details_b) =
                        (StepEvents::default(), StepEvents::default());
                    split.details(&mut details_a);
                    old.details(&mut details_b);
                    assert_eq!(
                        format!("{:?}", details_a.details),
                        format!("{:?}", details_b.details),
                        "{when}"
                    );

                    touches += a.cars.iter().filter(|c| c.touched_ball).count();
                    bumps += a.cars.iter().filter(|c| c.bumped_opponent).count();
                    teammates += a.cars.iter().filter(|c| c.bumped_teammate).count();
                    demos += a.cars.iter().filter(|c| c.demoed_opponent).count();
                    goals += usize::from(a.goal_tick.is_some());
                    past_goal += a.cars.iter().filter(|c| c.past_goal).count();
                }
            }
        }
        println!(
            "{ticks_run} ticks: {touches} touches, {bumps} bumps, {teammates} teammate bumps, \
             {demos} demolitions, {goals} goals, {past_goal} cars past a goal"
        );
        assert!(
            touches > 50 && bumps > 20 && teammates > 5 && demos > 5,
            "too few events"
        );
        assert!(goals > 5 && past_goal > 5, "too few goals");
    }

    /// More car-car events than `details` keeps (RocketSim cannot make them in one tick, the record can):
    /// every one sets the team flags, the first 8 the latest bumps, as the old loop did.
    #[test]
    fn every_bump_sets_the_team_flags_and_the_first_eight_the_latest_bumps() {
        if !meshes_loaded() {
            return;
        }
        let mut sim = Sim::new(4, true, 0);
        let mut facts = TickFacts {
            tick: 7,
            ..TickFacts::default()
        };
        // Eight bumps among cars 0-5 (Blue 0, 2, 4; Orange 1, 3, 5), then car 6 (Blue) bumps 7 (Orange) and
        // car 7 demolishes 0.
        let pairs = [
            (0, 2),
            (2, 4),
            (4, 0),
            (1, 3),
            (3, 5),
            (5, 1),
            (0, 2),
            (2, 4),
            (6, 7),
            (7, 0),
        ];
        for (i, &(bumper, victim)) in pairs.iter().enumerate() {
            facts.bumps[i] = [bumper, victim, u8::from(i == 9)];
        }
        facts.n_bumps = pairs.len() as u8;
        let mut events = StepEvents::default();
        events.clear(8);
        sim.absorb(&facts, &mut events);
        assert!(events.cars[6].bumped_opponent && !events.cars[6].demoed_opponent);
        assert!(events.cars[7].bumped_opponent && events.cars[7].demoed_opponent);
        assert!((0..6).all(|car| events.cars[car].bumped_teammate));
        assert_eq!(sim.contacts[6], None, "the ninth bump is not kept");
        assert_eq!(sim.contacts[7], None);
        assert_eq!(
            sim.contacts[0],
            Some((2, 7)),
            "the seventh bump, not the tenth"
        );
        assert_eq!(sim.contacts[4], Some((2, 7)), "the eighth bump");
    }
}
