//! Field numbers and small helpers for rewards of one's own, in the prelude. Distances in uu, speeds in
//! uu/s, Blue defends −y and attacks +y.

use rocketsim::Vec3A;

use crate::snapshot::{CarSnapshot, GameSnapshot, Team};

pub const BALL_RADIUS: f32 = 91.25;
/// The ball's height resting on the floor.
pub const BALL_REST_Z: f32 = 93.15;
/// The speed above which a car is supersonic.
pub const SUPERSONIC_SPEED: f32 = 2200.0;
pub const CAR_MAX_ANG_VEL: f32 = 5.5;
/// A car's height resting on its wheels.
pub const CAR_REST_Z: f32 = 17.0;
/// The side walls' distance from the centre.
pub const SIDE_WALL_X: f32 = 4096.0;
/// The back walls' distance from the centre (the goal lines).
pub const BACK_WALL_Y: f32 = 5120.0;
/// The back of each goal's net.
pub const BACK_NET_Y: f32 = 6000.0;
pub const GOAL_HEIGHT: f32 = 642.775;
pub const GOAL_HALF_WIDTH: f32 = 892.755;
/// One physics tick, in seconds.
pub const TICK_SECONDS: f32 = 1.0 / 120.0;

/// +1 for Blue, which attacks +y; −1 for Orange.
pub fn attack_sign(team: Team) -> f32 {
    if team == Team::Blue { 1.0 } else { -1.0 }
}

/// The centre of the goal mouth `team` scores in: on its goal line, half the goal's height up.
pub fn attack_goal(team: Team) -> Vec3A {
    Vec3A::new(0.0, attack_sign(team) * BACK_WALL_Y, GOAL_HEIGHT / 2.0)
}

/// The centre of the goal mouth `team` defends.
pub fn own_goal(team: Team) -> Vec3A {
    attack_goal(team.opposite())
}

/// The back of the net `team` scores in, half the goal's height up.
pub fn attack_net_back(team: Team) -> Vec3A {
    Vec3A::new(0.0, attack_sign(team) * BACK_NET_Y, GOAL_HEIGHT / 2.0)
}

/// The back of the net `team` defends.
pub fn own_net_back(team: Team) -> Vec3A {
    attack_net_back(team.opposite())
}

/// `v` as `team` sees it when everything is turned so that it attacks +y (Orange's view turned half a
/// turn; Blue's unchanged): for maths written for Blue only.
pub fn as_blue(team: Team, v: Vec3A) -> Vec3A {
    if team == Team::Blue {
        v
    } else {
        Vec3A::new(-v.x, -v.y, v.z)
    }
}

/// The other cars of `car`'s team.
pub fn teammates(game: &GameSnapshot, car: usize) -> impl Iterator<Item = usize> + '_ {
    let team = game.cars[car].team;
    (0..game.cars.len()).filter(move |&other| other != car && game.cars[other].team == team)
}

/// The cars of the other team.
pub fn opponents(game: &GameSnapshot, car: usize) -> impl Iterator<Item = usize> + '_ {
    let team = game.cars[car].team;
    (0..game.cars.len()).filter(move |&other| game.cars[other].team != team)
}

/// A kickoff: the ball still, on the centre spot.
pub fn is_kickoff(game: &GameSnapshot) -> bool {
    game.ball.pos.x == 0.0 && game.ball.pos.y == 0.0 && game.ball.vel == Vec3A::ZERO
}

/// The car of `team` nearest the ball (the first such car on a tie).
pub fn closest_to_ball(game: &GameSnapshot, team: Team) -> Option<usize> {
    let distance = |car: &CarSnapshot| car.body.pos.distance(game.ball.pos);
    (0..game.cars.len())
        .filter(|&car| game.cars[car].team == team)
        .min_by(|&a, &b| distance(&game.cars[a]).total_cmp(&distance(&game.cars[b])))
}

/// `kph` km/h in uu/s.
pub fn kph(kph: f32) -> f32 {
    kph / 0.036
}

/// The cosine of the angle between `a` and `b` (0 when either is zero).
pub fn cosine(a: Vec3A, b: Vec3A) -> f32 {
    a.normalize_or_zero().dot(b.normalize_or_zero())
}

#[cfg(test)]
pub mod testing {
    //! Games for the rewards' own tests.

    use super::*;
    use crate::snapshot::{Body, CarEvents, CarSnapshot, StepEvents};

    /// A game with the ball at `ball` and a car of each `(team, position)`, all at rest on their wheels.
    pub fn game(ball: Vec3A, cars: &[(Team, Vec3A)]) -> GameSnapshot {
        GameSnapshot {
            ball: Body {
                pos: ball,
                ..Body::RESTING
            },
            cars: cars
                .iter()
                .map(|&(team, pos)| CarSnapshot {
                    team,
                    body: Body {
                        pos,
                        ..Body::RESTING
                    },
                    boost: 33.0,
                    on_ground: true,
                    can_flip: true,
                    ..CarSnapshot::default()
                })
                .collect(),
            ..GameSnapshot::default()
        }
    }

    /// Nothing happened to any of `cars` cars.
    pub fn quiet(cars: usize) -> StepEvents {
        let mut events = StepEvents {
            step_seconds: 8.0 / 120.0,
            ..StepEvents::default()
        };
        events.clear(cars);
        events.details = vec![Default::default(); cars];
        events.prev_details = events.details.clone();
        events
    }

    /// Car `car` touched the ball (and so touched it last).
    pub fn touched(mut events: StepEvents, car: usize) -> StepEvents {
        events.cars[car] = CarEvents {
            touched_ball: true,
            ..events.cars[car]
        };
        events.last_touch = Some(car);
        events
    }
}
