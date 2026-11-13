//! Start states drawn from recordings of real games: RocketSim's
//! bundled captures of pro and bot play and recordings made from replays, read through `rlpr`.
//!
//! A recording does not say which team a car is on, so the teams are read off its kickoffs: at a kickoff
//! every car sits still on its own half at a kickoff spot. A frame then serves a game by taking, from each
//! team, the cars nearest the ball (a 1v1 from a 3v3 recording keeps the two most involved cars). Half
//! the draws turn the field around (a half-turn about the centre swaps the halves and the teams), so both
//! ends are used alike and a Blue bot also meets Orange's situations. Frames with the ball in a goal and
//! frames identical to the one before (a pause, a countdown) are left out. The physics state placed is
//! the bodies, boost and ground contact; jump and flip state is not carried (an airborne car starts with
//! its jump unused), as every replay-state setter in the community has it.

use std::f32::consts::PI;
use std::path::Path;

use rocketsim::{Mat3A, Vec3A};

use crate::rlpr::{self, Recording, Tick};
use crate::snapshot::{Body, Team};

const GOAL_Y: f32 = 5120.0;
const CEILING_Z: f32 = 2044.0;
/// A car within this of a kickoff spot's |y| (and on the ground) is on its kickoff spot.
const KICKOFF_SPOT_TOLERANCE: f32 = 60.0;
/// A car on its spot below this speed (uu/s) has not left it yet: a converted replay's first kickoff tick
/// can already have the cars rolling; a few ticks of full boost reach about 200.
const KICKOFF_RELEASE_SPEED: f32 = 200.0;
const KICKOFF_SPOT_Y: [f32; 3] = [4608.0, 3840.0, 2560.0];
/// Below this speed (uu/s) a body counts as at rest.
const AT_REST: f32 = 5.0;

/// A car as `Sim::place` takes it: body, boost (0 to 100), on the ground.
pub type PlacedCar = (Body, f32, bool);

#[derive(Clone, Debug)]
struct Frame {
    ball: Body,
    cars: Vec<(PlacedCar, Team)>,
}

pub struct ReplayLibrary {
    frames: Vec<Frame>,
    cumulative: Vec<f32>,
    team_size: usize,
    opponents: bool,
}

impl std::fmt::Debug for ReplayLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReplayLibrary({} frames)", self.frames.len())
    }
}

impl ReplayLibrary {
    /// Loads every file (plain `.rlpr` or `.rlpr.zst`) for a game of `team_size` per team.
    pub fn load(
        files: &[String],
        team_size: usize,
        opponents: bool,
        air_bias: f32,
    ) -> Result<Self, String> {
        if files.is_empty() {
            return Err("the replay state setter needs at least one recording file".to_owned());
        }
        let recordings = files
            .iter()
            .map(|file| rlpr::read_file(Path::new(file)).map(|r| (file.as_str(), r)))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_recordings(&recordings, team_size, opponents, air_bias)
    }

    /// `recordings` are (name for messages, recording).
    pub fn from_recordings(
        recordings: &[(&str, Recording)],
        team_size: usize,
        opponents: bool,
        air_bias: f32,
    ) -> Result<Self, String> {
        let mut frames = Vec::new();
        let mut left_out = Vec::new();
        for (name, recording) in recordings {
            match frames_of(recording, team_size) {
                Ok(found) => frames.extend(found),
                // A recording of a smaller game (the bundled 1v1 in a 2v2 run) is left out, not fatal.
                Err(err) => left_out.push(format!("{name}: {err}")),
            }
        }
        if frames.is_empty() {
            return Err(format!(
                "no recording fits this game or holds a usable frame (ball in play, cars moving): {}",
                left_out.join("; ")
            ));
        }
        let mut total = 0.0;
        let cumulative = frames
            .iter()
            .map(|frame| {
                let height = frame
                    .cars
                    .iter()
                    .map(|((body, _, _), _)| body.pos.z)
                    .fold(frame.ball.pos.z, f32::max);
                total += 1.0 + air_bias * (height / CEILING_Z).clamp(0.0, 1.0);
                total
            })
            .collect();
        Ok(Self {
            frames,
            cumulative,
            team_size,
            opponents,
        })
    }

    /// Usable frames across every recording.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// A start state in the simulator's car order (Blue, Orange, Blue, Orange, ...; Blue only without
    /// opponents): the ball and, per team, the `team_size` cars nearest the ball; turned around half the
    /// time.
    pub fn sample(&self, rng: &mut fastrand::Rng) -> (Body, Vec<PlacedCar>) {
        let pick = rng.f32() * self.cumulative[self.cumulative.len() - 1];
        let index = self
            .cumulative
            .partition_point(|&c| c <= pick)
            .min(self.frames.len() - 1);
        let frame = &self.frames[index];
        let turned = rng.bool();
        let ball = if turned {
            turned_around(frame.ball)
        } else {
            frame.ball
        };
        let by_team = |team: Team| -> Vec<PlacedCar> {
            let mut cars: Vec<PlacedCar> = frame
                .cars
                .iter()
                // Turning the field around swaps the teams: Orange's cars become Blue's situation.
                .filter(|(_, recorded)| (*recorded == team) != turned)
                .map(|&((body, boost, on_ground), _)| {
                    (
                        if turned { turned_around(body) } else { body },
                        boost,
                        on_ground,
                    )
                })
                .collect();
            cars.sort_by(|a, b| {
                a.0.pos
                    .distance_squared(ball.pos)
                    .total_cmp(&b.0.pos.distance_squared(ball.pos))
            });
            cars.truncate(self.team_size);
            cars
        };
        let blue = by_team(Team::Blue);
        let orange = if self.opponents {
            by_team(Team::Orange)
        } else {
            Vec::new()
        };
        let mut cars = Vec::with_capacity(blue.len() + orange.len());
        for slot in 0..self.team_size {
            cars.push(blue[slot]);
            if self.opponents {
                cars.push(orange[slot]);
            }
        }
        (ball, cars)
    }
}

/// The same situation seen from the other end of the field: a half-turn about the centre line.
pub fn turned_around(body: Body) -> Body {
    let half_turn = Mat3A::from_rotation_z(PI);
    Body {
        pos: Vec3A::new(-body.pos.x, -body.pos.y, body.pos.z),
        rot: half_turn * body.rot,
        vel: Vec3A::new(-body.vel.x, -body.vel.y, body.vel.z),
        ang_vel: Vec3A::new(-body.ang_vel.x, -body.ang_vel.y, body.ang_vel.z),
    }
}

/// The ball at rest at the centre, where a kickoff puts it (until a car touches it).
pub(crate) fn ball_at_kickoff(ball: &Body) -> bool {
    ball.pos.x.abs() < 5.0 && ball.pos.y.abs() < 5.0 && ball.vel.length() < AT_REST
}

/// Every car at rest on its own kickoff spot, the ball at rest at the centre.
pub(crate) fn is_kickoff(tick: &Tick) -> bool {
    ball_at_kickoff(&tick.ball)
        && tick.cars.iter().all(|car| {
            car.on_ground
                && car.body.vel.length() < KICKOFF_RELEASE_SPEED
                && KICKOFF_SPOT_Y
                    .iter()
                    .any(|&spot| (car.body.pos.y.abs() - spot).abs() < KICKOFF_SPOT_TOLERANCE)
        })
}

/// Which team each car is on, read off the kickoffs: Blue starts on the -Y half. A one-car recording with no
/// kickoff (a clip of one player's move, written attacking +y) is Blue.
pub fn teams_of(recording: &Recording) -> Result<Vec<Team>, String> {
    let num_cars = recording.num_cars as usize;
    let mut blue_votes = vec![0usize; num_cars];
    let mut kickoffs = 0usize;
    for tick in recording
        .ticks
        .iter()
        .filter(|t| t.cars.len() == num_cars && is_kickoff(t))
    {
        kickoffs += 1;
        for (votes, car) in blue_votes.iter_mut().zip(&tick.cars) {
            *votes += usize::from(car.body.pos.y < 0.0);
        }
    }
    if kickoffs == 0 && num_cars == 1 {
        return Ok(vec![Team::Blue]);
    }
    if kickoffs == 0 {
        return Err(
            "no kickoff in the recording, so the cars' teams cannot be told apart (a recording must \
             include at least one kickoff with every car on its spot)"
                .to_owned(),
        );
    }
    Ok(blue_votes
        .iter()
        .map(|&votes| {
            if votes * 2 >= kickoffs {
                Team::Blue
            } else {
                Team::Orange
            }
        })
        .collect())
}

fn frames_of(recording: &Recording, team_size: usize) -> Result<Vec<Frame>, String> {
    let teams = teams_of(recording)?;
    let blue = teams.iter().filter(|&&t| t == Team::Blue).count();
    let orange = teams.len() - blue;
    if blue < team_size || orange < team_size {
        return Err(format!(
            "the recording has {blue} Blue and {orange} Orange cars; a {team_size}v{team_size} game needs \
             {team_size} of each"
        ));
    }
    let mut frames = Vec::with_capacity(recording.ticks.len());
    let mut previous: Option<&Tick> = None;
    for tick in &recording.ticks {
        let frozen = previous.is_some_and(|last| {
            last.ball.pos == tick.ball.pos
                && last.cars.len() == tick.cars.len()
                && last
                    .cars
                    .iter()
                    .zip(&tick.cars)
                    .all(|(a, b)| a.body.pos == b.body.pos)
        });
        previous = Some(tick);
        if frozen || tick.cars.len() != teams.len() || tick.ball.pos.y.abs() > GOAL_Y {
            continue;
        }
        frames.push(Frame {
            ball: tick.ball,
            cars: tick
                .cars
                .iter()
                .zip(&teams)
                .map(|(car, &team)| ((car.body, car.boost * 100.0, car.on_ground), team))
                .collect(),
        });
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlpr::{CarRecord, Tick};
    use crate::snapshot::Controls;

    fn car_at(pos: Vec3A, vel: Vec3A, on_ground: bool) -> CarRecord {
        CarRecord {
            body: Body {
                pos,
                rot: Mat3A::IDENTITY,
                vel,
                ang_vel: Vec3A::ZERO,
            },
            on_ground,
            is_jumping: false,
            is_flipping: false,
            jump_time: 0.0,
            flip_time: 0.0,
            has_jumped: false,
            double_jumped_or_flipped: false,
            has_flip: true,
            flip_rel_torque: Vec3A::ZERO,
            boost: 0.5,
            touching_ball: false,
            controls: Controls::default(),
            touching_car: false,
        }
    }

    fn ball_at(pos: Vec3A, vel: Vec3A) -> Body {
        Body {
            pos,
            rot: Mat3A::IDENTITY,
            vel,
            ang_vel: Vec3A::ZERO,
        }
    }

    /// A 2v2 recording: car order Orange, Blue, Blue, Orange (as the game lists them, not by team); a
    /// kickoff, then play frames, a frozen repeat, a frame with the ball in the goal.
    fn recording() -> Recording {
        let rest = Vec3A::ZERO;
        let kickoff = Tick {
            frame: 1,
            cars: vec![
                car_at(Vec3A::new(2048.0, 2560.0, 17.0), rest, true),
                car_at(Vec3A::new(-2048.0, -2560.0, 17.0), rest, true),
                car_at(Vec3A::new(0.0, -4608.0, 17.0), rest, true),
                car_at(Vec3A::new(256.0, 3840.0, 17.0), rest, true),
            ],
            ball: ball_at(Vec3A::new(0.0, 0.0, 93.15), rest),
            ball_world_contact: true,
        };
        let moving = Vec3A::new(500.0, 0.0, 0.0);
        let play = Tick {
            frame: 2,
            cars: vec![
                car_at(Vec3A::new(1000.0, 1000.0, 17.0), moving, true), // Orange, near the ball
                car_at(Vec3A::new(-300.0, 200.0, 17.0), moving, true),  // Blue, nearest
                car_at(Vec3A::new(-3000.0, -3000.0, 17.0), moving, true), // Blue, far
                car_at(Vec3A::new(3000.0, 4000.0, 600.0), moving, false), // Orange, far and airborne
            ],
            ball: ball_at(Vec3A::new(0.0, 500.0, 200.0), moving),
            ball_world_contact: false,
        };
        let frozen = Tick {
            frame: 3,
            ..play.clone()
        };
        let mut in_goal = play.clone();
        in_goal.frame = 4;
        in_goal.ball.pos = Vec3A::new(0.0, 5300.0, 200.0);
        Recording {
            version: 6,
            num_cars: 4,
            hitbox_min_bt: Vec3A::ZERO,
            hitbox_max_bt: Vec3A::ZERO,
            ticks: vec![kickoff, play, frozen, in_goal],
        }
    }

    #[test]
    fn teams_come_from_the_kickoff_and_unusable_frames_are_dropped() {
        let recording = recording();
        assert_eq!(
            teams_of(&recording).unwrap(),
            [Team::Orange, Team::Blue, Team::Blue, Team::Orange]
        );
        let library =
            ReplayLibrary::from_recordings(&[("test", recording.clone())], 2, true, 0.0).unwrap();
        assert_eq!(
            library.len(),
            2,
            "the kickoff and the play frame; the frozen and in-goal ones go"
        );
        let mut rolling = recording.clone();
        for car in &mut rolling.ticks[0].cars {
            car.body.vel = Vec3A::new(0.0, 120.0, 0.0);
        }
        assert_eq!(
            teams_of(&rolling).unwrap(),
            teams_of(&recording).unwrap(),
            "cars just released from their spots still tell the teams"
        );
        let mut no_kickoff = recording.clone();
        no_kickoff.ticks.remove(0);
        let err = ReplayLibrary::from_recordings(&[("x", no_kickoff)], 1, true, 0.0).unwrap_err();
        assert!(err.contains("no kickoff"), "{err}");
        let err =
            ReplayLibrary::from_recordings(&[("x", recording.clone())], 3, true, 0.0).unwrap_err();
        assert!(
            err.contains("no recording fits") && err.contains("2 Blue and 2 Orange"),
            "{err}"
        );
        // A recording of a smaller game beside one that fits is left out silently.
        let mut small = recording.clone();
        small.num_cars = 2;
        for tick in &mut small.ticks {
            tick.cars.truncate(2);
        }
        let mixed =
            ReplayLibrary::from_recordings(&[("small", small), ("ok", recording)], 2, true, 0.0)
                .unwrap();
        assert_eq!(mixed.len(), 2);
    }

    #[test]
    fn a_1v1_takes_the_cars_nearest_the_ball_from_each_team_and_turns_half_the_draws() {
        let library =
            ReplayLibrary::from_recordings(&[("test", recording())], 1, true, 0.0).unwrap();
        let mut rng = fastrand::Rng::with_seed(4);
        let (mut plain, mut turned, mut kickoffs) = (0, 0, 0);
        for _ in 0..200 {
            let (ball, cars) = library.sample(&mut rng);
            assert_eq!(cars.len(), 2);
            if ball.pos.y > 0.0 {
                // The play frame as recorded (ball at y = 500).
                plain += 1;
                assert_eq!(
                    cars[0].0.pos,
                    Vec3A::new(-300.0, 200.0, 17.0),
                    "Blue's nearest"
                );
                assert_eq!(
                    cars[1].0.pos,
                    Vec3A::new(1000.0, 1000.0, 17.0),
                    "Orange's nearest"
                );
                assert_eq!(cars[0].1, 50.0, "boost on the simulator's scale");
            } else if ball.pos.y < 0.0 {
                // Turned around: Orange's nearest car (1000, 1000) becomes Blue's at (-1000, -1000).
                turned += 1;
                assert_eq!(cars[0].0.pos, Vec3A::new(-1000.0, -1000.0, 17.0));
                assert_eq!(cars[1].0.pos, Vec3A::new(300.0, -200.0, 17.0));
                assert_eq!(cars[0].0.vel, Vec3A::new(-500.0, 0.0, 0.0));
            } else {
                // The kickoff, either way round: Blue's car is on the -Y half.
                kickoffs += 1;
                assert!(cars[0].0.pos.y < 0.0 && cars[1].0.pos.y > 0.0);
            }
        }
        assert!(
            plain > 30 && turned > 30 && kickoffs > 60,
            "{plain} plain, {turned} turned, {kickoffs} kickoffs"
        );
    }

    #[test]
    fn turning_around_is_a_half_turn_that_undoes_itself() {
        let body = Body {
            pos: Vec3A::new(100.0, -200.0, 300.0),
            rot: Mat3A::from_rotation_z(0.4) * Mat3A::from_rotation_y(0.2),
            vel: Vec3A::new(1.0, 2.0, 3.0),
            ang_vel: Vec3A::new(0.1, 0.2, 0.3),
        };
        let twice = turned_around(turned_around(body));
        assert!((twice.pos - body.pos).length() < 1e-4);
        assert!((twice.vel - body.vel).length() < 1e-5);
        assert!((twice.ang_vel - body.ang_vel).length() < 1e-5);
        for col in 0..3 {
            assert!((twice.rot.col(col) - body.rot.col(col)).length() < 1e-5);
        }
        // Forward keeps its height and flips in the plane.
        let once = turned_around(body);
        assert!((once.forward().z - body.forward().z).abs() < 1e-6);
        assert!((once.forward().x + body.forward().x).abs() < 1e-6);
    }

    #[test]
    fn air_bias_favours_high_frames() {
        let library =
            ReplayLibrary::from_recordings(&[("test", recording())], 2, true, 10.0).unwrap();
        let mut rng = fastrand::Rng::with_seed(5);
        let high = (0..4000)
            .filter(|_| {
                let (_, cars) = library.sample(&mut rng);
                cars.iter().any(|(body, _, _)| body.pos.z > 100.0)
            })
            .count();
        // Weights: kickoff 1 + 10 * 17/2044 = 1.08, play frame 1 + 10 * 600/2044 = 3.94 -> 78% of draws.
        assert!(
            (2900..3400).contains(&high),
            "{high} of 4000 draws were the airborne frame"
        );
    }
}
