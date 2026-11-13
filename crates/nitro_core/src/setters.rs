//! How each episode starts, picked at random from a weighted mix in the settings file.

use std::f32::consts::{FRAC_PI_2, PI};

use rocketsim::{Mat3A, Vec3A};
use serde::Deserialize;

use crate::replay::ReplayLibrary;
use crate::sim::Sim;
use crate::snapshot::{Body, GameSnapshot};

const BALL_RADIUS: f32 = 92.75;
const CAR_REST_Z: f32 = 17.0;
const MAX_CAR_SPEED: f32 = 2300.0;
const MAX_CAR_ANG_SPEED: f32 = 5.5;
pub(crate) const PLACEMENT_ATTEMPTS: usize = 100;
/// Closest two car centres (or a car and the ball) may start; a car is about 120 uu long.
pub(crate) const MIN_CAR_GAP: f32 = 250.0;
/// From any ball position on the field there is room for a car this far away (the field is 7000 x 8400
/// inside the margins), so the ball-touch setter always finds a spot.
const MAX_TOUCH_DISTANCE: f32 = 4000.0;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum SetterKind {
    /// A normal kickoff, optionally with each car moved up to `fuzz_uu` in a random direction on the floor.
    Kickoff { fuzz_uu: f32 },
    /// Ball and cars anywhere on the field with random speeds and orientations.
    Random {
        max_ball_speed: f32,
        /// Chance that each car starts on its wheels on the floor instead of in the air.
        ground_chance: f32,
    },
    /// Practice: the ball at rest on the floor; each car at rest on its wheels, `min_distance_uu` to
    /// `max_distance_uu` from the ball, facing a random direction.
    BallTouch {
        min_distance_uu: f32,
        max_distance_uu: f32,
    },
    /// Practice: each car `min_height_uu` to `max_height_uu` above the floor, turned and spinning at
    /// random; the ball at rest on the floor. With `max_ball_distance_uu` above 0, each car starts
    /// `min_ball_distance_uu` to `max_ball_distance_uu` from the ball along the floor; otherwise anywhere.
    Airborne {
        min_height_uu: f32,
        max_height_uu: f32,
        #[serde(default)]
        min_ball_distance_uu: f32,
        #[serde(default)]
        max_ball_distance_uu: f32,
    },
    /// Practice: each car at rest on its wheels with no boost, every boost pad available; the ball at
    /// rest on the floor.
    BoostPickup {},
    /// Practice: an empty-net shot. The ball at rest `min_distance_uu` to `max_distance_uu` in front of
    /// the goal the first car attacks; that car `car_distance_uu` behind the ball, facing it. Any other
    /// car starts at rest somewhere else on the field.
    Shot {
        min_distance_uu: f32,
        max_distance_uu: f32,
        car_distance_uu: f32,
    },
    /// Practice: a save. The ball rolls from the first car's half toward its goal mouth at `min_speed` to
    /// `max_speed` uu/s; that car waits in its goal facing the field. Any other car starts at rest
    /// somewhere else on the field.
    Save { min_speed: f32, max_speed: f32 },
    /// Practice: an aerial. The ball `min_height_uu` to `max_height_uu` above the floor, thrown up to
    /// `max_ball_speed` uu/s (mostly upward, so it hangs); the first car at rest on its wheels 800–2500 uu
    /// away with a full tank, facing the ball. Any other car starts at rest somewhere else on the field.
    AerialBall {
        min_height_uu: f32,
        max_height_uu: f32,
        max_ball_speed: f32,
    },
    /// A situation from a recording of a real game (`crate::replay`): `files` are RLPR recordings (plain or
    /// zstd-compressed, as the settings loader resolved them); `air_bias` favours frames with something high
    /// up (a frame's weight is 1 + air_bias × height / ceiling height).
    Replay { files: Vec<String>, air_bias: f32 },
}

/// The goal line (uu); Blue defends -Y and attacks +Y.
const GOAL_Y: f32 = 5120.0;
/// Inside the goal mouth with room to spare (the posts are at x = ±893, the bar at z = 643).
const GOAL_MOUTH_X: f32 = 700.0;
const GOAL_MOUTH_Z: f32 = 500.0;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedSetter {
    pub kind: SetterKind,
    pub weight: f32,
}

impl WeightedSetter {
    /// What is wrong with these values, if anything (the Python loader checks the same, with line numbers).
    pub fn problem(&self) -> Option<&'static str> {
        let within = |value: f32, low: f32, high: f32| (low..=high).contains(&value);
        if !within(self.weight, 0.0, f32::MAX) {
            return Some("state setter weights must be 0 or more");
        }
        match self.kind {
            SetterKind::Kickoff { fuzz_uu } if !within(fuzz_uu, 0.0, 300.0) => {
                Some("kickoff fuzz_uu must be 0 to 300")
            }
            SetterKind::Random { max_ball_speed, .. } if !within(max_ball_speed, 0.0, 6000.0) => {
                Some("random max_ball_speed must be 0 to 6000")
            }
            SetterKind::Random { ground_chance, .. } if !within(ground_chance, 0.0, 1.0) => {
                Some("random ground_chance must be 0 to 1")
            }
            SetterKind::BallTouch {
                min_distance_uu,
                max_distance_uu,
            } if !(within(min_distance_uu, MIN_CAR_GAP, MAX_TOUCH_DISTANCE)
                && within(max_distance_uu, min_distance_uu, MAX_TOUCH_DISTANCE)) =>
            {
                Some("ball_touch needs 250 <= min_distance_uu <= max_distance_uu <= 4000")
            }
            SetterKind::Airborne {
                min_height_uu,
                max_height_uu,
                ..
            } if !(within(min_height_uu, 100.0, 1800.0)
                && within(max_height_uu, min_height_uu, 1800.0)) =>
            {
                Some("airborne needs 100 <= min_height_uu <= max_height_uu <= 1800")
            }
            SetterKind::Airborne {
                min_ball_distance_uu,
                max_ball_distance_uu,
                ..
            } if (max_ball_distance_uu == 0.0 && min_ball_distance_uu != 0.0)
                || (max_ball_distance_uu != 0.0
                    && !(within(min_ball_distance_uu, MIN_CAR_GAP, MAX_TOUCH_DISTANCE)
                        && within(
                            max_ball_distance_uu,
                            min_ball_distance_uu,
                            MAX_TOUCH_DISTANCE,
                        ))) =>
            {
                Some(
                    "airborne needs 250 <= min_ball_distance_uu <= max_ball_distance_uu <= 4000 (or both 0)",
                )
            }
            SetterKind::Shot {
                min_distance_uu,
                max_distance_uu,
                car_distance_uu,
            } if !(within(min_distance_uu, 500.0, 4500.0)
                && within(max_distance_uu, min_distance_uu, 4500.0)
                && within(car_distance_uu, MIN_CAR_GAP, 2500.0)) =>
            {
                Some(
                    "shot needs 500 <= min_distance_uu <= max_distance_uu <= 4500 and 250 <= car_distance_uu <= 2500",
                )
            }
            SetterKind::Save {
                min_speed,
                max_speed,
            } if !(within(min_speed, 200.0, 4000.0) && within(max_speed, min_speed, 4000.0)) => {
                Some("save needs 200 <= min_speed <= max_speed <= 4000")
            }
            SetterKind::AerialBall {
                min_height_uu,
                max_height_uu,
                max_ball_speed,
            } if !(within(min_height_uu, 300.0, 1800.0)
                && within(max_height_uu, min_height_uu, 1800.0)
                && within(max_ball_speed, 0.0, 2000.0)) =>
            {
                Some(
                    "aerial_ball needs 300 <= min_height_uu <= max_height_uu <= 1800 and 0 <= max_ball_speed <= 2000",
                )
            }
            SetterKind::Replay { ref files, .. } if files.is_empty() => {
                Some("replay needs at least one recording file")
            }
            SetterKind::Replay { air_bias, .. } if !within(air_bias, 0.0, 10.0) => {
                Some("replay air_bias must be 0 to 10")
            }
            _ => None,
        }
    }
}

pub struct SetterMix {
    setters: Vec<SetterKind>,
    weights: Vec<f32>,
    /// The loaded recordings of each replay setter (None for the other kinds).
    replays: Vec<Option<ReplayLibrary>>,
}

impl SetterMix {
    /// Loads any replay setter's recordings for a `team_size`-a-side game. Panics on an empty list or a
    /// non-positive total weight (validated in the settings loader); a recording that cannot be read or
    /// does not fit the game is an error.
    pub fn new(
        weighted: Vec<WeightedSetter>,
        team_size: usize,
        opponents: bool,
    ) -> Result<Self, String> {
        let total: f32 = weighted.iter().map(|s| s.weight).sum();
        assert!(
            total > 0.0,
            "state setter weights must sum to more than zero"
        );
        let weights: Vec<f32> = weighted.iter().map(|s| s.weight).collect();
        let replays = weighted
            .iter()
            .map(|setter| match &setter.kind {
                SetterKind::Replay { files, air_bias } => {
                    ReplayLibrary::load(files, team_size, opponents, *air_bias)
                        .map(Some)
                        .map_err(|err| format!("replay state setter: {err}"))
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            setters: weighted.into_iter().map(|setter| setter.kind).collect(),
            weights,
            replays,
        })
    }

    /// Setter `index`'s weight.
    pub fn weight(&self, index: usize) -> f32 {
        self.weights[index]
    }

    /// Number of setters in the mix.
    pub fn len(&self) -> usize {
        self.setters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.setters.is_empty()
    }

    /// The kind of setter `index`.
    pub fn kind(&self, index: usize) -> &SetterKind {
        &self.setters[index]
    }

    /// Puts `sim` into a fresh starting state and returns which setter made it (its index in the list);
    /// `scratch` is reused to avoid allocating.
    pub fn apply(
        &self,
        sim: &mut Sim,
        rng: &mut fastrand::Rng,
        scratch: &mut GameSnapshot,
    ) -> usize {
        let index = {
            let count = self.setters.len();
            let weight = |i: usize| self.weight(i);
            let total: f32 = (0..count).map(weight).sum();
            // A running sum of the weights.
            let pick = rng.f32() * total;
            let mut sum = 0.0;
            (0..count)
                .position(|i| {
                    sum += weight(i);
                    pick < sum
                })
                .or_else(|| (0..count).rev().find(|&i| weight(i) > 0.0))
                .unwrap_or(count - 1)
        };
        sim.reset_to_kickoff(rng.u64(..));
        match self.setters[index] {
            SetterKind::Replay { .. } => {
                let library = self.replays[index]
                    .as_ref()
                    .expect("loaded in SetterMix::new");
                let (ball, cars) = library.sample(rng);
                sim.place(ball, &cars);
            }
            SetterKind::Kickoff { fuzz_uu } => {
                if fuzz_uu > 0.0 {
                    sim.read(scratch);
                    let mut cars: Vec<(Body, f32, bool)> = Vec::with_capacity(scratch.cars.len());
                    for car in &scratch.cars {
                        let mut body = car.body;
                        // A random point within `fuzz_uu`, kept clear of the cars already moved; the
                        // exact kickoff spot is the fallback.
                        let spot = (0..PLACEMENT_ATTEMPTS)
                            .map(|_| body.pos + random_in_disc(rng) * fuzz_uu)
                            .find(|p| {
                                cars.iter()
                                    .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
                            });
                        body.pos = spot.unwrap_or(body.pos);
                        cars.push((body, car.boost, true));
                    }
                    sim.place(scratch.ball, &cars);
                }
            }
            SetterKind::Random {
                max_ball_speed,
                ground_chance,
            } => random_state(sim, rng, max_ball_speed, ground_chance),
            SetterKind::BallTouch {
                min_distance_uu,
                max_distance_uu,
            } => ball_touch(sim, rng, min_distance_uu, max_distance_uu),
            SetterKind::Airborne {
                min_height_uu,
                max_height_uu,
                max_ball_distance_uu: 0.0,
                ..
            } => airborne(sim, rng, min_height_uu, max_height_uu),
            SetterKind::Airborne {
                min_height_uu,
                max_height_uu,
                min_ball_distance_uu,
                max_ball_distance_uu,
            } => airborne_near_ball(
                sim,
                rng,
                (min_height_uu, max_height_uu),
                (min_ball_distance_uu, max_ball_distance_uu),
            ),
            SetterKind::BoostPickup {} => boost_pickup(sim, rng),
            SetterKind::Shot {
                min_distance_uu,
                max_distance_uu,
                car_distance_uu,
            } => shot(sim, rng, min_distance_uu, max_distance_uu, car_distance_uu),
            SetterKind::Save {
                min_speed,
                max_speed,
            } => save(sim, rng, min_speed, max_speed),
            SetterKind::AerialBall {
                min_height_uu,
                max_height_uu,
                max_ball_speed,
            } => aerial_ball(sim, rng, min_height_uu, max_height_uu, max_ball_speed),
        }
        index
    }
}

/// The ball in the air, hanging or rising; the first car on the floor some way off, facing it, full tank.
fn aerial_ball(
    sim: &mut Sim,
    rng: &mut fastrand::Rng,
    min_height: f32,
    max_height: f32,
    max_speed: f32,
) {
    // Away from the walls, so the car has room to approach from any side.
    let ball_pos = Vec3A::new(
        sym(rng) * 2500.0,
        sym(rng) * 3000.0,
        min_height + rng.f32() * (max_height - min_height),
    );
    // Mostly upward, with a sideways drift of up to half the speed; never faster than `max_speed` in all.
    let sideways = random_in_disc(rng) * max_speed * 0.5;
    let thrown = Vec3A::new(sideways.x, sideways.y, rng.f32() * max_speed);
    let speed = thrown.length();
    let vel = if speed > max_speed {
        thrown * (max_speed / speed)
    } else {
        thrown
    };
    let ball = Body {
        pos: ball_pos,
        vel,
        ..Body::RESTING
    };
    let under = Vec3A::new(ball_pos.x, ball_pos.y, CAR_REST_Z);
    let pos = (0..PLACEMENT_ATTEMPTS)
        .map(|_| {
            let angle = sym(rng) * PI;
            let distance = 800.0 + rng.f32() * 1700.0;
            under + Vec3A::new(angle.cos(), angle.sin(), 0.0) * distance
        })
        .find(|&p| in_field(p))
        .unwrap_or(Vec3A::new(0.0, 0.0, CAR_REST_Z));
    let mut cars = vec![(facing(pos, under), 100.0, true)];
    park_the_rest(sim, rng, ball.pos, &mut cars);
    sim.place(ball, &cars);
}

/// Uniform in [-1, 1).
fn sym(rng: &mut fastrand::Rng) -> f32 {
    rng.f32() * 2.0 - 1.0
}

/// Uniform in the unit disc on the floor plane.
fn random_in_disc(rng: &mut fastrand::Rng) -> Vec3A {
    loop {
        let v = Vec3A::new(sym(rng), sym(rng), 0.0);
        if v.length_squared() <= 1.0 {
            return v;
        }
    }
}

fn random_direction(rng: &mut fastrand::Rng) -> Vec3A {
    loop {
        let v = Vec3A::new(sym(rng), sym(rng), sym(rng));
        let len = v.length();
        if len > 1e-3 && len <= 1.0 {
            return v / len;
        }
    }
}

/// Inside the field with a margin from the walls and the rounded corners.
pub(crate) fn random_field_point(rng: &mut fastrand::Rng, z_min: f32, z_max: f32) -> Vec3A {
    loop {
        let p = Vec3A::new(
            sym(rng) * 3500.0,
            sym(rng) * 4200.0,
            z_min + rng.f32() * (z_max - z_min),
        );
        if p.x.abs() + p.y.abs() < 6500.0 {
            return p;
        }
    }
}

fn random_state(sim: &mut Sim, rng: &mut fastrand::Rng, max_ball_speed: f32, ground_chance: f32) {
    let ball = Body {
        pos: random_field_point(rng, BALL_RADIUS, 1800.0),
        vel: random_direction(rng) * rng.f32() * max_ball_speed,
        ..Body::RESTING
    };
    let mut cars: Vec<(Body, f32, bool)> = Vec::with_capacity(sim.num_cars());
    for _ in 0..sim.num_cars() {
        let on_ground = rng.f32() < ground_chance;
        let yaw = sym(rng) * PI;
        // Rejection sampling keeps cars apart from each other and from the ball.
        let pos = (0..PLACEMENT_ATTEMPTS)
            .map(|_| {
                if on_ground {
                    random_field_point(rng, CAR_REST_Z, CAR_REST_Z)
                } else {
                    random_field_point(rng, 150.0, 1800.0)
                }
            })
            .find(|p| {
                p.distance(ball.pos) > MIN_CAR_GAP
                    && cars
                        .iter()
                        .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
            })
            .expect("no free spot for a car (field too crowded)");
        let body = if on_ground {
            Body {
                pos,
                rot: Mat3A::from_rotation_z(yaw),
                vel: Mat3A::from_rotation_z(yaw).x_axis * rng.f32() * MAX_CAR_SPEED,
                ang_vel: Vec3A::ZERO,
            }
        } else {
            Body {
                pos,
                rot: Mat3A::from_rotation_z(yaw)
                    * Mat3A::from_rotation_y(sym(rng) * FRAC_PI_2)
                    * Mat3A::from_rotation_x(sym(rng) * PI),
                vel: random_direction(rng) * rng.f32() * MAX_CAR_SPEED,
                ang_vel: random_direction(rng) * rng.f32() * MAX_CAR_ANG_SPEED,
            }
        };
        cars.push((body, rng.f32() * 100.0, on_ground));
    }
    sim.place(ball, &cars);
}

fn in_field(p: Vec3A) -> bool {
    p.x.abs() <= 3500.0 && p.y.abs() <= 4200.0 && p.x.abs() + p.y.abs() < 6500.0
}

fn resting_ball(rng: &mut fastrand::Rng) -> Body {
    Body {
        pos: random_field_point(rng, BALL_RADIUS, BALL_RADIUS),
        ..Body::RESTING
    }
}

/// A car at rest on its wheels facing a random direction.
pub(crate) fn parked(pos: Vec3A, rng: &mut fastrand::Rng) -> Body {
    Body {
        pos,
        rot: Mat3A::from_rotation_z(sym(rng) * PI),
        ..Body::RESTING
    }
}

fn ball_touch(sim: &mut Sim, rng: &mut fastrand::Rng, min_distance: f32, max_distance: f32) {
    // A ball near a wall may leave no room at the drawn distances: then draw the ball again.
    for _ in 0..PLACEMENT_ATTEMPTS {
        let ball = resting_ball(rng);
        let mut cars: Vec<(Body, f32, bool)> = Vec::with_capacity(sim.num_cars());
        for _ in 0..sim.num_cars() {
            let spot = (0..PLACEMENT_ATTEMPTS)
                .map(|_| {
                    let angle = sym(rng) * PI;
                    let distance = min_distance + rng.f32() * (max_distance - min_distance);
                    let offset = Vec3A::new(angle.cos(), angle.sin(), 0.0) * distance;
                    Vec3A::new(ball.pos.x + offset.x, ball.pos.y + offset.y, CAR_REST_Z)
                })
                .find(|&p| {
                    in_field(p)
                        && cars
                            .iter()
                            .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
                });
            let Some(pos) = spot else { break };
            cars.push((parked(pos, rng), rng.f32() * 100.0, true));
        }
        if cars.len() == sim.num_cars() {
            sim.place(ball, &cars);
            return;
        }
    }
    panic!("no room for the cars at the ball_touch distances");
}

fn airborne(sim: &mut Sim, rng: &mut fastrand::Rng, min_height: f32, max_height: f32) {
    let ball = resting_ball(rng);
    let mut cars: Vec<(Body, f32, bool)> = Vec::with_capacity(sim.num_cars());
    for _ in 0..sim.num_cars() {
        let pos = (0..PLACEMENT_ATTEMPTS)
            .map(|_| random_field_point(rng, min_height, max_height))
            .find(|p| {
                p.distance(ball.pos) > MIN_CAR_GAP
                    && cars
                        .iter()
                        .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
            })
            .expect("no free spot for a car (field too crowded)");
        cars.push((tumbling(pos, rng), rng.f32() * 100.0, false));
    }
    sim.place(ball, &cars);
}

/// Each car turned and spinning at random in the air, a set distance from the ball along the floor.
fn airborne_near_ball(
    sim: &mut Sim,
    rng: &mut fastrand::Rng,
    heights: (f32, f32),
    distances: (f32, f32),
) {
    // A ball near a wall may leave no room at the drawn distances: then draw the ball again.
    for _ in 0..PLACEMENT_ATTEMPTS {
        let ball = resting_ball(rng);
        if let Some(spots) = spots_near(rng, ball.pos, sim.num_cars(), heights, distances) {
            let cars: Vec<(Body, f32, bool)> = spots
                .into_iter()
                .map(|pos| (tumbling(pos, rng), rng.f32() * 100.0, false))
                .collect();
            sim.place(ball, &cars);
            return;
        }
    }
    panic!("no room for the cars at the airborne ball distances");
}

/// `count` points `heights` above the floor and `distances` from `ball` along it, inside the field and
/// apart from each other; `None` when there is no room.
fn spots_near(
    rng: &mut fastrand::Rng,
    ball: Vec3A,
    count: usize,
    (low, high): (f32, f32),
    (near, far): (f32, f32),
) -> Option<Vec<Vec3A>> {
    let mut spots: Vec<Vec3A> = Vec::with_capacity(count);
    for _ in 0..count {
        let spot = (0..PLACEMENT_ATTEMPTS)
            .map(|_| {
                let angle = sym(rng) * PI;
                let distance = near + rng.f32() * (far - near);
                let height = low + rng.f32() * (high - low);
                Vec3A::new(
                    ball.x + angle.cos() * distance,
                    ball.y + angle.sin() * distance,
                    height,
                )
            })
            .find(|&p| in_field(p) && spots.iter().all(|other| p.distance(*other) > MIN_CAR_GAP))?;
        spots.push(spot);
    }
    Some(spots)
}

/// A car at `pos` turned and spinning at random, drifting up to 500 uu/s (the airborne start's car).
fn tumbling(pos: Vec3A, rng: &mut fastrand::Rng) -> Body {
    Body {
        pos,
        rot: Mat3A::from_rotation_z(sym(rng) * PI)
            * Mat3A::from_rotation_y(sym(rng) * FRAC_PI_2)
            * Mat3A::from_rotation_x(sym(rng) * PI),
        vel: random_direction(rng) * rng.f32() * 500.0,
        ang_vel: random_direction(rng) * rng.f32() * MAX_CAR_ANG_SPEED,
    }
}

/// A car at rest on its wheels at `pos`, facing `target`.
fn facing(pos: Vec3A, target: Vec3A) -> Body {
    Body {
        pos,
        rot: Mat3A::from_rotation_z((target.y - pos.y).atan2(target.x - pos.x)),
        ..Body::RESTING
    }
}

/// The other cars of a practice start: at rest on their wheels, away from the ball and from each other.
fn park_the_rest(
    sim: &Sim,
    rng: &mut fastrand::Rng,
    ball: Vec3A,
    cars: &mut Vec<(Body, f32, bool)>,
) {
    while cars.len() < sim.num_cars() {
        let pos = (0..PLACEMENT_ATTEMPTS)
            .map(|_| random_field_point(rng, CAR_REST_Z, CAR_REST_Z))
            .find(|p| {
                p.distance(ball) > MIN_CAR_GAP
                    && cars
                        .iter()
                        .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
            })
            .expect("no free spot for a car (field too crowded)");
        cars.push((parked(pos, rng), rng.f32() * 100.0, true));
    }
}

/// The first car attacks +Y (Blue) and defends -Y; see `Sim::new` for the car order.
fn shot(
    sim: &mut Sim,
    rng: &mut fastrand::Rng,
    min_distance: f32,
    max_distance: f32,
    car_distance: f32,
) {
    let distance = min_distance + rng.f32() * (max_distance - min_distance);
    let ball = Body {
        pos: Vec3A::new(sym(rng) * 1500.0, GOAL_Y - distance, BALL_RADIUS),
        ..Body::RESTING
    };
    // Behind the ball (toward the car's own goal), up to 30° off the straight line, facing the ball.
    let angle = -FRAC_PI_2 + sym(rng) * (PI / 6.0);
    let pos = ball.pos + Vec3A::new(angle.cos(), angle.sin(), 0.0) * car_distance;
    let pos = Vec3A::new(
        pos.x.clamp(-3400.0, 3400.0),
        pos.y.clamp(-4200.0, 4200.0),
        CAR_REST_Z,
    );
    let mut cars = vec![(facing(pos, ball.pos), rng.f32() * 100.0, true)];
    park_the_rest(sim, rng, ball.pos, &mut cars);
    sim.place(ball, &cars);
}

fn save(sim: &mut Sim, rng: &mut fastrand::Rng, min_speed: f32, max_speed: f32) {
    let start = Vec3A::new(sym(rng) * 2000.0, -800.0 - rng.f32() * 1800.0, BALL_RADIUS);
    // Aimed inside the goal mouth, just past the line; gravity drops it a little on the way.
    let target = Vec3A::new(
        sym(rng) * GOAL_MOUTH_X,
        -GOAL_Y - 50.0,
        BALL_RADIUS + rng.f32() * (GOAL_MOUTH_Z - BALL_RADIUS),
    );
    let speed = min_speed + rng.f32() * (max_speed - min_speed);
    let ball = Body {
        pos: start,
        vel: (target - start).normalize() * speed,
        ..Body::RESTING
    };
    // In the goal mouth, facing the field give or take 17°.
    let body = Body {
        pos: Vec3A::new(sym(rng) * 500.0, -4900.0 + rng.f32() * 400.0, CAR_REST_Z),
        rot: Mat3A::from_rotation_z(FRAC_PI_2 + sym(rng) * 0.3),
        ..Body::RESTING
    };
    let mut cars = vec![(body, rng.f32() * 100.0, true)];
    park_the_rest(sim, rng, ball.pos, &mut cars);
    sim.place(ball, &cars);
}

fn boost_pickup(sim: &mut Sim, rng: &mut fastrand::Rng) {
    let ball = resting_ball(rng);
    let mut cars: Vec<(Body, f32, bool)> = Vec::with_capacity(sim.num_cars());
    for _ in 0..sim.num_cars() {
        let pos = (0..PLACEMENT_ATTEMPTS)
            .map(|_| random_field_point(rng, CAR_REST_Z, CAR_REST_Z))
            .find(|p| {
                p.distance(ball.pos) > MIN_CAR_GAP
                    && cars
                        .iter()
                        .all(|(other, _, _)| p.distance(other.pos) > MIN_CAR_GAP)
            })
            .expect("no free spot for a car (field too crowded)");
        cars.push((parked(pos, rng), 0.0, true));
    }
    sim.place(ball, &cars);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::meshes_loaded;

    /// The meshes must be loaded first (`meshes_loaded`).
    fn start(kind: SetterKind, cars: usize, seed: u64) -> GameSnapshot {
        let mut sim = Sim::new(cars, false, seed);
        let mut rng = fastrand::Rng::with_seed(seed);
        let mix = SetterMix::new(vec![WeightedSetter { kind, weight: 1.0 }], cars, false).unwrap();
        let mut snap = GameSnapshot::default();
        mix.apply(&mut sim, &mut rng, &mut snap);
        sim.read(&mut snap);
        snap
    }

    fn assert_ball_at_rest_on_the_floor(snap: &GameSnapshot) {
        assert!((snap.ball.pos.z - BALL_RADIUS).abs() < 1.0 && in_field(snap.ball.pos));
        assert_eq!(snap.ball.vel, Vec3A::ZERO);
    }

    #[test]
    fn ball_touch_parks_each_car_at_the_requested_distance() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::BallTouch {
                    min_distance_uu: 1000.0,
                    max_distance_uu: 3000.0,
                },
                2,
                seed,
            );
            assert_ball_at_rest_on_the_floor(&snap);
            for car in &snap.cars {
                let distance = (car.body.pos - snap.ball.pos).truncate().length();
                assert!(
                    (999.0..=3001.0).contains(&distance),
                    "seed {seed}: {distance}"
                );
                assert!(in_field(car.body.pos) && car.on_ground && car.body.vel == Vec3A::ZERO);
                assert!(car.body.up().z > 0.99, "upright");
            }
            assert!(snap.cars[0].body.pos.distance(snap.cars[1].body.pos) > MIN_CAR_GAP);
        }
    }

    #[test]
    fn airborne_cars_start_in_the_air_within_the_height_range() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::Airborne {
                    min_height_uu: 300.0,
                    max_height_uu: 1200.0,
                    min_ball_distance_uu: 0.0,
                    max_ball_distance_uu: 0.0,
                },
                1,
                seed,
            );
            assert_ball_at_rest_on_the_floor(&snap);
            let car = &snap.cars[0];
            assert!(!car.on_ground && (300.0..=1200.0).contains(&car.body.pos.z));
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::Airborne {
                    min_height_uu: 300.0,
                    max_height_uu: 1200.0,
                    min_ball_distance_uu: 1000.0,
                    max_ball_distance_uu: 3000.0,
                },
                2,
                seed,
            );
            assert_ball_at_rest_on_the_floor(&snap);
            for car in &snap.cars {
                let along = (car.body.pos - snap.ball.pos) * Vec3A::new(1.0, 1.0, 0.0);
                assert!(!car.on_ground && (300.0..=1200.0).contains(&car.body.pos.z));
                assert!(
                    (999.0..=3001.0).contains(&along.length()),
                    "{}",
                    along.length()
                );
            }
        }
    }

    #[test]
    fn spots_near_the_ball_keep_their_distance_height_and_room() {
        let mut rng = fastrand::Rng::with_seed(4);
        let ball = Vec3A::new(0.0, 0.0, BALL_RADIUS);
        let spots = spots_near(&mut rng, ball, 3, (300.0, 1200.0), (1000.0, 3000.0)).unwrap();
        for (i, p) in spots.iter().enumerate() {
            let along = Vec3A::new(p.x - ball.x, p.y - ball.y, 0.0).length();
            assert!((1000.0..=3000.0).contains(&along) && (300.0..=1200.0).contains(&p.z));
            assert!(in_field(*p));
            assert!(spots[..i].iter().all(|q| q.distance(*p) > MIN_CAR_GAP));
        }
        let corner = Vec3A::new(3400.0, 3000.0, BALL_RADIUS);
        let far = spots_near(&mut rng, corner, 1, (300.0, 400.0), (3990.0, 4000.0));
        assert!(
            far.is_none_or(|s| in_field(s[0])),
            "outside the field is never a spot"
        );
    }

    #[test]
    fn aerial_ball_hangs_in_the_air_with_the_car_below_facing_it() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::AerialBall {
                    min_height_uu: 400.0,
                    max_height_uu: 1000.0,
                    max_ball_speed: 500.0,
                },
                2,
                seed,
            );
            assert!(
                (400.0..=1000.0).contains(&snap.ball.pos.z),
                "{:?}",
                snap.ball.pos
            );
            assert!(snap.ball.vel.z >= 0.0 && snap.ball.vel.length() <= 500.0 * 1.0001);
            let car = &snap.cars[0];
            assert!(car.on_ground && car.boost == 100.0);
            let flat = Vec3A::new(
                snap.ball.pos.x - car.body.pos.x,
                snap.ball.pos.y - car.body.pos.y,
                0.0,
            );
            assert!(
                (800.0 - 1.0..=2500.0 + 1.0).contains(&flat.length()),
                "{}",
                flat.length()
            );
            assert!(
                car.body.forward().dot(flat.normalize()) > 0.99,
                "facing the ball"
            );
            assert!(snap.cars[1].on_ground, "the other car is parked");
        }
    }

    #[test]
    fn boost_pickup_starts_empty_with_every_pad_available() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..20 {
            let snap = start(SetterKind::BoostPickup {}, 1, seed);
            assert_ball_at_rest_on_the_floor(&snap);
            assert_eq!(snap.cars[0].boost, 0.0);
            assert!(snap.cars[0].on_ground);
            assert!(snap.pads.iter().all(|pad| pad.active));
        }
    }

    #[test]
    fn shot_puts_the_ball_in_front_of_the_goal_with_the_car_behind_it_facing_it() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::Shot {
                    min_distance_uu: 1500.0,
                    max_distance_uu: 3500.0,
                    car_distance_uu: 1000.0,
                },
                2,
                seed,
            );
            assert_ball_at_rest_on_the_floor(&snap);
            let to_goal = GOAL_Y - snap.ball.pos.y;
            assert!(
                (1499.0..=3501.0).contains(&to_goal),
                "seed {seed}: {to_goal}"
            );
            let car = &snap.cars[0];
            let gap = (car.body.pos - snap.ball.pos).truncate();
            assert!((gap.length() - 1000.0).abs() < 1.0 || car.body.pos.x.abs() >= 3399.0);
            assert!(car.body.pos.y < snap.ball.pos.y, "behind the ball");
            let facing = car.body.forward().truncate().normalize();
            assert!(
                facing.dot(-gap.normalize()) > 0.999,
                "seed {seed}: facing the ball"
            );
            assert!(car.on_ground && car.body.vel == Vec3A::ZERO && car.body.up().z > 0.99);
            let other = &snap.cars[1];
            assert!(other.on_ground && other.body.pos.distance(snap.ball.pos) > MIN_CAR_GAP);
            assert!(other.body.pos.distance(car.body.pos) > MIN_CAR_GAP);
        }
    }

    #[test]
    fn save_rolls_the_ball_at_the_goal_mouth_with_the_car_waiting_in_goal() {
        if !meshes_loaded() {
            return;
        }
        for seed in 0..50 {
            let snap = start(
                SetterKind::Save {
                    min_speed: 1200.0,
                    max_speed: 2200.0,
                },
                1,
                seed,
            );
            let ball = snap.ball;
            assert!(
                ball.pos.y < -799.0 && in_field(ball.pos),
                "seed {seed}: {}",
                ball.pos
            );
            let speed = ball.vel.length();
            assert!((1199.0..=2201.0).contains(&speed), "seed {seed}: {speed}");
            assert!(ball.vel.y < 0.0, "toward the car's own goal");
            // Where the straight line crosses the goal line: inside the posts.
            let t = (-GOAL_Y - ball.pos.y) / ball.vel.y;
            let x = ball.pos.x + ball.vel.x * t;
            assert!(
                x.abs() <= GOAL_MOUTH_X + 1.0,
                "seed {seed}: aimed at x = {x}"
            );
            let car = &snap.cars[0];
            assert!(car.body.pos.y <= -4499.0 && car.body.pos.x.abs() <= 501.0);
            assert!(car.body.forward().y > 0.95, "facing the field");
            assert!(car.on_ground && car.body.vel == Vec3A::ZERO);
        }
    }

    #[test]
    fn practice_setter_values_are_checked() {
        let setter = |kind| WeightedSetter { kind, weight: 1.0 };
        assert!(
            setter(SetterKind::BallTouch {
                min_distance_uu: 3000.0,
                max_distance_uu: 1000.0
            })
            .problem()
            .is_some()
        );
        assert!(
            setter(SetterKind::Airborne {
                min_height_uu: 50.0,
                max_height_uu: 500.0,
                min_ball_distance_uu: 0.0,
                max_ball_distance_uu: 0.0,
            })
            .problem()
            .is_some()
        );
        let near_ball = |min_ball_distance_uu, max_ball_distance_uu| {
            setter(SetterKind::Airborne {
                min_height_uu: 300.0,
                max_height_uu: 1200.0,
                min_ball_distance_uu,
                max_ball_distance_uu,
            })
            .problem()
        };
        assert!(near_ball(0.0, 0.0).is_none() && near_ball(1000.0, 3000.0).is_none());
        assert!(near_ball(3000.0, 1000.0).is_some() && near_ball(100.0, 3000.0).is_some());
        assert!(near_ball(1000.0, 5000.0).is_some() && near_ball(1000.0, 0.0).is_some());
        assert!(setter(SetterKind::BoostPickup {}).problem().is_none());
        assert!(
            setter(SetterKind::Shot {
                min_distance_uu: 1500.0,
                max_distance_uu: 3500.0,
                car_distance_uu: 100.0
            })
            .problem()
            .is_some()
        );
        assert!(
            setter(SetterKind::Shot {
                min_distance_uu: 1500.0,
                max_distance_uu: 3500.0,
                car_distance_uu: 1000.0
            })
            .problem()
            .is_none()
        );
        assert!(
            setter(SetterKind::Save {
                min_speed: 2200.0,
                max_speed: 1200.0
            })
            .problem()
            .is_some()
        );
        assert!(
            setter(SetterKind::Save {
                min_speed: 1200.0,
                max_speed: 2200.0
            })
            .problem()
            .is_none()
        );
    }
}
