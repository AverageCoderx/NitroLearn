//! A car alone in RocketSim, started from a converted row: the simulator the timing fit (`fit`) and the car
//! error (`measure`) share. The ball is parked high in the corner farthest from the car and there are no boost
//! pad in reach, so only the arena's walls touch the car; stretches where the replay has the car near the ball or
//! another car are left to replicar (`near_contact`).

use glam::{Mat3A, Quat, Vec3A};
use replicar::hitbox::Hitbox;
use replicar::rocketsim::{
    Arena, ArenaConfig, BallState, BoostPadConfig, CarControls, CarState, GameMode, PhysState, Team,
};
use replicar_format::header::PlayerInfo;
use replicar_format::record::{Body, Car, Controls, Frame};

/// Centre distance (uu) at which a car may touch the ball or another car within a tick: an Octane's half
/// diagonal (about 85) plus the ball's radius (92) or another car's half diagonal, plus a margin.
pub const CONTACT_DISTANCE: f32 = 250.0;

fn phys(body: &Body) -> PhysState {
    PhysState {
        pos: Vec3A::from(body.position),
        rot_mat: Mat3A::from_quat(Quat::from_array(body.rotation).normalize()),
        vel: Vec3A::from(body.velocity),
        ang_vel: Vec3A::from(body.angular_velocity),
    }
}

fn body(phys: &PhysState) -> Body {
    let q = Quat::from_mat3a(&phys.rot_mat).normalize();
    let q = if q.w < 0.0 { -q } else { q }; // replicar's sign
    Body {
        position: phys.pos.to_array(),
        velocity: phys.vel.to_array(),
        angular_velocity: phys.ang_vel.to_array(),
        rotation: q.to_array(),
    }
}

pub fn controls(c: &Controls) -> CarControls {
    CarControls {
        throttle: c.throttle,
        steer: c.steer,
        pitch: c.pitch,
        yaw: c.yaw,
        roll: c.roll,
        jump: c.jump,
        boost: c.boost,
        handbrake: c.handbrake,
    }
}

fn record_controls(c: &CarControls) -> Controls {
    Controls {
        throttle: c.throttle,
        steer: c.steer,
        pitch: c.pitch,
        yaw: c.yaw,
        roll: c.roll,
        jump: c.jump,
        boost: c.boost,
        handbrake: c.handbrake,
    }
}

pub fn same_controls(a: &CarControls, b: &CarControls) -> bool {
    a.throttle.to_bits() == b.throttle.to_bits()
        && a.steer.to_bits() == b.steer.to_bits()
        && a.pitch.to_bits() == b.pitch.to_bits()
        && a.yaw.to_bits() == b.yaw.to_bits()
        && a.roll.to_bits() == b.roll.to_bits()
        && (a.jump, a.boost, a.handbrake) == (b.jump, b.boost, b.handbrake)
}

/// A row's car as RocketSim's state (the mapping replicar's `restore` uses: wheels only as touching or not).
/// RocketSim keeps the contact cache private: the ground flag and touching wheels go in through
/// `set_contact_flags`, the previous controls through `set_prev_controls`, and the world-contact normal, which
/// has no setter, through the state's serde form.
pub fn state_of(c: &Car) -> Result<CarState, String> {
    let i = &c.internals;
    // Fields the file does not store keep their defaults.
    let mut s = CarState::DEFAULT;
    s.phys = phys(&c.body);
    s.controls = controls(&c.controls);
    s.has_jumped = i.has_jumped;
    s.has_double_jumped = i.has_double_jumped;
    s.has_flipped = i.has_flipped;
    s.flip_rel_torque = Vec3A::from(i.flip_relative_torque);
    s.jump_ticks = i.jump_ticks;
    s.flip_time = i.flip_time;
    s.is_flipping = i.is_flipping;
    s.is_jumping = i.is_jumping;
    s.air_time = i.air_time;
    s.air_time_since_jump = i.air_time_since_jump;
    s.boost = c.boost;
    s.time_since_boosted = i.time_since_boosted;
    s.is_boosting = i.is_boosting;
    s.boosting_time = i.boosting_time;
    s.is_supersonic = i.is_supersonic;
    s.supersonic_grace_timer = i.supersonic_grace_timer;
    s.handbrake_val = i.handbrake_value;
    s.is_auto_flipping = i.is_auto_flipping;
    s.auto_flip_timer = i.auto_flip_timer;
    s.auto_flip_torque_scale = i.auto_flip_torque_scale;
    s.bump_cooldown_timer = i.bump_cooldown_timer;
    s.last_extra_hit_tick = i.last_extra_hit_tick;
    s.is_demoed = i.is_demoed;
    s.demo_respawn_timer = i.demo_respawn_timer;
    s.set_contact_flags(i.is_on_ground, i.wheels_with_contact);
    s.set_prev_controls(controls(&c.previous_controls));
    match i.world_contact_normal {
        None => Ok(s),
        Some(normal) => with_world_contact_normal(s, Vec3A::from(normal)),
    }
}

fn with_world_contact_normal(state: CarState, normal: Vec3A) -> Result<CarState, String> {
    let fail = |e: serde_json::Error| format!("world contact normal: {e}");
    let mut value = serde_json::to_value(state).map_err(fail)?;
    let cache = value
        .get_mut("cache")
        .and_then(|c| c.get_mut("world_contact_normal"))
        .ok_or("RocketSim's car state has no cache.world_contact_normal")?;
    *cache = serde_json::to_value(Some(normal)).map_err(fail)?;
    serde_json::from_value(value).map_err(fail)
}

/// Writes a simulated state into a row's car: its body, boost and internals, and the controls applied from it.
pub fn write(s: &CarState, applied: &CarControls, previous: &CarControls, car: &mut Car) {
    car.body = body(&s.phys);
    car.boost = s.boost;
    car.controls = record_controls(applied);
    car.previous_controls = record_controls(previous);
    let i = &mut car.internals;
    i.is_on_ground = s.is_on_ground();
    i.wheels_with_contact = s.wheels_with_contact().map(|w| w.is_some());
    i.has_jumped = s.has_jumped;
    i.has_double_jumped = s.has_double_jumped;
    i.has_flipped = s.has_flipped;
    i.flip_relative_torque = s.flip_rel_torque.to_array();
    i.jump_ticks = s.jump_ticks;
    i.flip_time = s.flip_time;
    i.is_flipping = s.is_flipping;
    i.is_jumping = s.is_jumping;
    i.air_time = s.air_time;
    i.air_time_since_jump = s.air_time_since_jump;
    i.time_since_boosted = s.time_since_boosted;
    i.is_boosting = s.is_boosting;
    i.boosting_time = s.boosting_time;
    i.is_supersonic = s.is_supersonic;
    i.supersonic_grace_timer = s.supersonic_grace_timer;
    i.handbrake_value = s.handbrake_val;
    i.is_auto_flipping = s.is_auto_flipping;
    i.auto_flip_timer = s.auto_flip_timer;
    i.auto_flip_torque_scale = s.auto_flip_torque_scale;
    i.bump_cooldown_timer = s.bump_cooldown_timer;
    i.last_extra_hit_tick = s.last_extra_hit_tick;
    i.world_contact_normal = s.world_contact_normal().map(|n| n.to_array());
    i.is_demoed = s.is_demoed;
    i.demo_respawn_timer = s.demo_respawn_timer;
}

/// The jump state the fit's rule compares when an update moves (has jumped, double jumped, flipped).
pub fn jump_flags(s: &CarState) -> (bool, bool, bool) {
    (s.has_jumped, s.has_double_jumped, s.has_flipped)
}

/// What carries a car from one update to the next besides its body (equal: the simulation goes on the same).
pub fn same_internals(a: &CarState, b: &CarState) -> bool {
    let f = |x: f32, y: f32| x.to_bits() == y.to_bits();
    (
        a.is_on_ground(),
        a.has_jumped,
        a.has_double_jumped,
        a.has_flipped,
    ) == (
        b.is_on_ground(),
        b.has_jumped,
        b.has_double_jumped,
        b.has_flipped,
    ) && (a.is_flipping, a.is_jumping, a.jump_ticks, a.is_boosting)
        == (b.is_flipping, b.is_jumping, b.jump_ticks, b.is_boosting)
        && (a.is_supersonic, a.is_auto_flipping) == (b.is_supersonic, b.is_auto_flipping)
        && f(a.flip_time, b.flip_time)
        && f(a.air_time, b.air_time)
        && f(a.air_time_since_jump, b.air_time_since_jump)
        && f(a.boost, b.boost)
        && f(a.boosting_time, b.boosting_time)
        && f(a.handbrake_val, b.handbrake_val)
        && f(a.supersonic_grace_timer, b.supersonic_grace_timer)
        && a.flip_rel_torque.to_array().map(f32::to_bits)
            == b.flip_rel_torque.to_array().map(f32::to_bits)
        && same_controls(a.prev_controls(), b.prev_controls())
}

/// One car alone in an arena without pads.
pub struct Alone {
    arena: Arena,
}

impl Alone {
    pub fn new(player: &PlayerInfo) -> Result<Self, String> {
        let hitbox = Hitbox::from_name(&player.hitbox)
            .ok_or_else(|| format!("unknown hitbox {}", player.hitbox))?;
        let mut config = ArenaConfig::new(GameMode::Soccar);
        config.rng_seed = Some(0);
        // RocketSim needs a pad: one under the floor, out of reach.
        config.custom_boost_pads = Some(vec![BoostPadConfig {
            pos: Vec3A::new(0.0, 0.0, -2000.0),
            is_big: false,
        }]);
        let mut arena = Arena::new_with_config(config);
        let team = if player.team == 0 {
            Team::Blue
        } else {
            Team::Orange
        };
        arena.add_car(team, hitbox.config());
        Ok(Self { arena })
    }

    /// Puts the car in `state` as a fresh start (no contacts carried from what was simulated before).
    pub fn start(&mut self, state: CarState) {
        let p = state.phys.pos;
        let mut ball = BallState::default();
        // The corner farthest from the car, high: it falls about 20 uu in a quarter second.
        ball.phys.pos = Vec3A::new(
            if p.x > 0.0 { -3000.0 } else { 3000.0 },
            if p.y > 0.0 { -4000.0 } else { 4000.0 },
            1800.0,
        );
        self.arena.set_ball_state(ball);
        self.arena.reset_car_transient_contacts(0);
        self.arena.set_car_state_full(0, state);
        self.arena.clear_persistent_manifolds();
    }

    /// Replaces the car's state mid-run (an update), keeping its contacts.
    pub fn set(&mut self, state: CarState) {
        self.arena.set_car_state_full(0, state);
        // RocketSim asks for this after moving a car mid-drive.
        self.arena.refresh_car_sticky_gate(0);
    }

    pub fn step(&mut self, controls: CarControls) {
        self.arena.set_car_controls(0, controls);
        self.arena.step_tick();
    }

    pub fn state(&self) -> CarState {
        *self.arena.get_car_state(0)
    }
}

/// Whether the replay has car `p` within reach of the ball or another car on row `frame`.
pub fn near_contact(frame: &Frame, p: usize) -> bool {
    let Some(Some(car)) = frame.state.cars.get(p) else {
        return true;
    };
    let at = Vec3A::from(car.body.position);
    let reach = |other: [f32; 3]| at.distance(Vec3A::from(other)) < CONTACT_DISTANCE;
    reach(frame.state.ball.body.position)
        || frame
            .state
            .cars
            .iter()
            .enumerate()
            .any(|(q, c)| q != p && c.as_ref().is_some_and(|c| reach(c.body.position)))
}
