//! Scripted benchmark opponents: simple players written here, each deciding once a step
//! from the game state alone, with no randomness, so the same start gives the same game. They give a bot a
//! fixed yardstick ("beats the chaser 97% of points") that does not move as training goes on.
//!
//! - `idle`: does nothing (the bottom of the scale).
//! - `chaser`: drives at the ball, lined up to hit it toward the other goal, boosts, dodges into it.
//! - `goalie`: shadows between the ball and its own goal, facing the ball; challenges when the ball comes
//!   near its goal.
//! - `kickoff`: rushes every kickoff with a dodge into the ball, then plays as the chaser.
//! - `aerial`: the chaser, plus a jump-and-boost at a ball in the air in front of it.

use rocketsim::Vec3A;

use crate::snapshot::{CarSnapshot, Controls, GameSnapshot, Team};

const GOAL_Y: f32 = 5120.0;
const FIELD_X: f32 = 3900.0;
const FIELD_Y: f32 = 4900.0;
const BALL_RADIUS: f32 = 93.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    Idle,
    Chaser,
    Goalie,
    Kickoff,
    Aerial,
}

pub const NAMES: [&str; 5] = ["idle", "chaser", "goalie", "kickoff", "aerial"];

impl Script {
    pub fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "idle" => Self::Idle,
            "chaser" => Self::Chaser,
            "goalie" => Self::Goalie,
            "kickoff" => Self::Kickoff,
            "aerial" => Self::Aerial,
            other => {
                return Err(format!(
                    "no scripted opponent called '{other}' (choose from {})",
                    NAMES.join(", ")
                ));
            }
        })
    }
}

/// What a scripted car remembers between steps: where it is in a dodge or a jump.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Memory {
    /// 0: nothing; 1: first jump held; 2: released; 3: dodging (until it lands). 10, 11: an aerial's jump.
    phase: u8,
    /// Pitch and yaw of the dodge being made.
    dodge: (f32, f32),
}

fn flat(v: Vec3A) -> Vec3A {
    Vec3A::new(v.x, v.y, 0.0)
}

fn own_goal(team: Team) -> Vec3A {
    Vec3A::new(0.0, if team == Team::Blue { -GOAL_Y } else { GOAL_Y }, 0.0)
}

fn their_goal(team: Team) -> Vec3A {
    -own_goal(team)
}

/// `v` in the car's frame: x forward, y to its steering side, z up (as `crate::extras` reads it).
fn local(car: &CarSnapshot, v: Vec3A) -> Vec3A {
    car.body.rot.transpose() * v
}

/// Level out in the air: pitch and roll toward the floor.
fn recover(car: &CarSnapshot) -> Controls {
    let up = local(car, Vec3A::Z);
    Controls {
        pitch: (-up.x * 2.0).clamp(-1.0, 1.0),
        roll: (up.y * 2.0).clamp(-1.0, 1.0),
        throttle: 1.0,
        ..Controls::default()
    }
}

/// Full speed toward `target` on the ground: steer at it, slide round sharp turns, boost on straight runs.
fn drive_to(car: &CarSnapshot, target: Vec3A, boost: bool) -> Controls {
    let to = local(car, target - car.body.pos);
    let angle = to.y.atan2(to.x);
    let distance = flat(to).length();
    let speed = car.body.vel.dot(car.body.forward());
    Controls {
        throttle: 1.0,
        steer: (angle * 3.0).clamp(-1.0, 1.0),
        boost: boost && angle.abs() < 0.35 && car.boost > 0.0 && speed < 2250.0 && distance > 600.0,
        handbrake: angle.abs() > 1.9 && speed > 500.0,
        ..Controls::default()
    }
}

/// Where a chaser drives: a little behind the ball on its line to the other goal, so it hits the ball that
/// way; straight at it when close.
fn chase_target(snap: &GameSnapshot, car: &CarSnapshot) -> Vec3A {
    let ball = flat(snap.ball.pos);
    let aim = (flat(their_goal(car.team)) - ball).normalize_or(Vec3A::Y);
    let distance = flat(car.body.pos).distance(ball);
    let back = (distance * 0.35).min(500.0);
    let target = ball - aim * back;
    Vec3A::new(
        target.x.clamp(-FIELD_X, FIELD_X),
        target.y.clamp(-FIELD_Y, FIELD_Y),
        0.0,
    )
}

fn is_kickoff(snap: &GameSnapshot) -> bool {
    flat(snap.ball.pos).length() < 1.0 && snap.ball.vel.length() < 1.0
}

/// The next step of a dodge already started, or None when no dodge is under way.
fn dodging(car: &CarSnapshot, memory: &mut Memory) -> Option<Controls> {
    match memory.phase {
        1 => {
            memory.phase = 2;
            Some(Controls {
                throttle: 1.0,
                ..Controls::default()
            })
        }
        2 => {
            memory.phase = 3;
            Some(Controls {
                throttle: 1.0,
                jump: true,
                pitch: memory.dodge.0,
                yaw: memory.dodge.1,
                ..Controls::default()
            })
        }
        3 if !car.on_ground => Some(Controls {
            throttle: 1.0,
            boost: car.boost > 0.0 && memory.dodge.0 < -0.5,
            ..recover(car)
        }),
        3 => {
            memory.phase = 0;
            None
        }
        _ => None,
    }
}

/// Starts a dodge toward `target` (forward and to its side).
fn start_dodge(car: &CarSnapshot, target: Vec3A, memory: &mut Memory) -> Controls {
    let to = local(car, target - car.body.pos);
    let angle = to.y.atan2(to.x);
    memory.phase = 1;
    memory.dodge = (-angle.cos().clamp(-1.0, 1.0), angle.sin().clamp(-1.0, 1.0));
    Controls {
        throttle: 1.0,
        jump: true,
        ..Controls::default()
    }
}

/// The chaser: line up behind the ball, drive at it, dodge into it when close and low.
fn chaser(snap: &GameSnapshot, car: &CarSnapshot, memory: &mut Memory) -> Controls {
    if let Some(controls) = dodging(car, memory) {
        return controls;
    }
    if !car.on_ground {
        return recover(car);
    }
    let ball = snap.ball.pos;
    let to = local(car, ball - car.body.pos);
    let distance = flat(to).length();
    let speed = car.body.vel.dot(car.body.forward());
    if distance < 380.0 + speed * 0.1 && ball.z < 2.5 * BALL_RADIUS && to.x > 0.0 && speed > 600.0 {
        return start_dodge(car, ball, memory);
    }
    drive_to(car, chase_target(snap, car), true)
}

/// The goalie: between the ball and its goal, nearer the goal the further the ball is; challenges a ball
/// near its goal.
fn goalie(snap: &GameSnapshot, car: &CarSnapshot, memory: &mut Memory) -> Controls {
    if let Some(controls) = dodging(car, memory) {
        return controls;
    }
    if !car.on_ground {
        return recover(car);
    }
    let goal = own_goal(car.team);
    let ball = flat(snap.ball.pos);
    let ball_to_goal = ball.distance(goal);
    let coming = snap.ball.vel.dot((goal - ball).normalize_or(Vec3A::ZERO)) > 300.0;
    if ball_to_goal < 1800.0 || (coming && ball_to_goal < 3000.0) {
        return chaser(snap, car, memory);
    }
    let stand =
        goal + (ball - goal).normalize_or(Vec3A::Y) * (ball_to_goal * 0.25).clamp(500.0, 1800.0);
    let to = local(car, stand - car.body.pos);
    let distance = flat(to).length();
    let speed = car.body.vel.dot(car.body.forward());
    if distance < 250.0 {
        // In place: face the ball, coming to rest.
        let at_ball = local(car, snap.ball.pos - car.body.pos);
        let facing = at_ball.y.atan2(at_ball.x);
        return Controls {
            throttle: if facing.abs() > 0.5 {
                0.5
            } else {
                (-speed / 400.0).clamp(-1.0, 1.0)
            },
            steer: (facing * 2.0).clamp(-1.0, 1.0),
            ..Controls::default()
        };
    }
    let mut controls = drive_to(car, stand, distance > 2000.0);
    // Slower as it arrives.
    if speed > distance * 1.5 + 300.0 {
        controls.throttle = 0.0;
        controls.boost = false;
    }
    controls
}

/// The kickoff rusher: full boost at the ball from kickoff, a dodge into it near it; the chaser otherwise.
fn kickoff(snap: &GameSnapshot, car: &CarSnapshot, memory: &mut Memory) -> Controls {
    if let Some(controls) = dodging(car, memory) {
        return controls;
    }
    if !is_kickoff(snap) {
        return chaser(snap, car, memory);
    }
    let distance = flat(car.body.pos).distance(flat(snap.ball.pos));
    if car.on_ground && distance < 750.0 {
        return start_dodge(car, snap.ball.pos, memory);
    }
    Controls {
        boost: car.boost > 0.0,
        ..drive_to(car, snap.ball.pos, true)
    }
}

/// The chaser, which also jumps and boosts at a ball in the air ahead of it.
fn aerial(snap: &GameSnapshot, car: &CarSnapshot, memory: &mut Memory) -> Controls {
    let ball = snap.ball.pos;
    let to = local(car, ball - car.body.pos);
    match memory.phase {
        10 => {
            memory.phase = 11;
            return Controls {
                jump: true,
                pitch: 1.0,
                ..Controls::default()
            };
        }
        11 if !car.on_ground => {
            // Nose at the ball, boosting once roughly lined up.
            let dir = to.normalize_or(Vec3A::X);
            return Controls {
                pitch: (dir.z * 4.0).clamp(-1.0, 1.0),
                yaw: (dir.y * 4.0).clamp(-1.0, 1.0),
                roll: (-local(car, Vec3A::Z).y).clamp(-1.0, 1.0),
                boost: car.boost > 0.0 && dir.x > 0.7,
                throttle: 1.0,
                ..Controls::default()
            };
        }
        11 => memory.phase = 0,
        _ => {}
    }
    if memory.phase == 0 && car.on_ground && car.boost > 20.0 && ball.z > 300.0 {
        let distance = flat(to).length();
        let ahead = to.y.atan2(to.x).abs() < 0.3;
        if ahead && distance < ball.z * 2.0 + 400.0 && distance > ball.z * 0.4 {
            memory.phase = 10;
            return Controls {
                jump: true,
                ..Controls::default()
            };
        }
    }
    chaser(snap, car, memory)
}

/// Car `car`'s controls for the next step under `script`.
pub fn controls(script: Script, snap: &GameSnapshot, car: usize, memory: &mut Memory) -> Controls {
    let me = &snap.cars[car];
    if me.is_demoed {
        *memory = Memory::default();
        return Controls::default();
    }
    match script {
        Script::Idle => Controls::default(),
        Script::Chaser => chaser(snap, me, memory),
        Script::Goalie => goalie(snap, me, memory),
        Script::Kickoff => kickoff(snap, me, memory),
        Script::Aerial => aerial(snap, me, memory),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{Body, CarSnapshot};

    fn car(team: Team, x: f32, y: f32) -> CarSnapshot {
        CarSnapshot {
            team,
            body: Body {
                pos: Vec3A::new(x, y, 17.0),
                ..Body::RESTING
            },
            on_ground: true,
            boost: 33.0,
            ..CarSnapshot::default()
        }
    }

    fn game(cars: Vec<CarSnapshot>, ball: Vec3A) -> GameSnapshot {
        let mut snap = GameSnapshot {
            cars,
            ..GameSnapshot::default()
        };
        snap.ball.pos = ball;
        snap
    }

    #[test]
    fn names_parse_and_unknown_ones_are_refused() {
        for name in NAMES {
            assert!(Script::parse(name).is_ok());
        }
        assert!(Script::parse("pro").unwrap_err().contains("chaser"));
    }

    #[test]
    fn idle_does_nothing_and_the_chaser_steers_at_the_ball() {
        // Car at the origin facing +x; ball on its +y side (the side positive steering turns to).
        let snap = game(
            vec![car(Team::Blue, 0.0, 0.0)],
            Vec3A::new(0.0, 2000.0, 93.0),
        );
        let mut memory = Memory::default();
        assert_eq!(
            controls(Script::Idle, &snap, 0, &mut memory),
            Controls::default()
        );
        let c = controls(Script::Chaser, &snap, 0, &mut memory);
        assert_eq!(c.throttle, 1.0);
        assert!(c.steer > 0.5, "{c:?}");
    }

    #[test]
    fn the_goalie_stays_home_when_the_ball_is_far() {
        // Blue goalie at its goal, ball deep in Orange's half: it stays near its goal.
        let snap = game(
            vec![car(Team::Blue, 0.0, -4600.0)],
            Vec3A::new(0.0, 3000.0, 93.0),
        );
        let goal = own_goal(Team::Blue);
        let stand = goal + Vec3A::Y * (8120.0f32 * 0.25).clamp(500.0, 1800.0);
        assert!(stand.y < -3000.0);
        let mut memory = Memory::default();
        let c = controls(Script::Goalie, &snap, 0, &mut memory);
        assert!(!c.jump);
    }

    #[test]
    fn the_kickoff_bot_dodges_near_the_ball_and_the_dodge_runs_its_course() {
        let mut snap = game(
            vec![car(Team::Blue, 600.0, 0.0)],
            Vec3A::new(0.0, 0.0, 93.0),
        );
        // Face the ball (-x).
        snap.cars[0].body.rot = rocketsim::Mat3A::from_rotation_z(std::f32::consts::PI);
        let mut memory = Memory::default();
        let first = controls(Script::Kickoff, &snap, 0, &mut memory);
        assert!(first.jump);
        let second = controls(Script::Kickoff, &snap, 0, &mut memory);
        assert!(!second.jump);
        snap.cars[0].on_ground = false;
        let third = controls(Script::Kickoff, &snap, 0, &mut memory);
        assert!(third.jump && third.pitch < -0.9, "{third:?}");
        snap.cars[0].on_ground = true;
        controls(Script::Kickoff, &snap, 0, &mut memory);
        assert_eq!(
            memory.phase, 1,
            "a new dodge once landed, the ball still there"
        );
    }

    #[test]
    fn the_same_state_gives_the_same_controls() {
        let snap = game(
            vec![car(Team::Orange, 1200.0, 800.0)],
            Vec3A::new(-400.0, 300.0, 500.0),
        );
        for script in NAMES.map(|n| Script::parse(n).unwrap()) {
            let (mut a, mut b) = (Memory::default(), Memory::default());
            assert_eq!(
                controls(script, &snap, 0, &mut a),
                controls(script, &snap, 0, &mut b)
            );
        }
    }
}
