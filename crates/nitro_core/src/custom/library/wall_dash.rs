//! Wall dashes: a short jump off a wall and a dodge that puts the wheels straight back on it, paid by the
//! speed the dash gained along the wall.
//! The rules (what counts as a make, the timing and flip angle bands, the speed window, the wall types,
//! chains and the spam guard) follow walldashcoach by samuelkassi (github.com/samuelkassi/walldashcoach),
//! Copyright (c) 2026 samuelkassi, MIT licence (full text in LICENSES/walldashcoach-MIT.txt).
//! Modified: written in Rust for whole steps of `tick_skip` ticks instead of single ticks; pays a reward
//! instead of showing a verdict.
//! group: moves

use crate::custom::prelude::*;

/// Wheels on the wall within this many ticks of the first touch back...
const MAKE_WINDOW_TICKS: f32 = 30.0;
/// ...this many of them...
const MAKE_MIN_WHEELS: usize = 3;
/// ...with the roof this close to the wall's normal (a cosine).
const UPRIGHT_DOT: f32 = 0.9;
/// The peak speed after a make is looked for this many ticks after the wheels are back.
const SPEED_WINDOW_TICKS: f32 = 18.0;
/// An attempt that has not come back after this many ticks is a miss.
const TIMEOUT_TICKS: f32 = 120.0;
/// This many jump presses in one attempt is mashing, not a dash.
const SPAM_PRESSES: f32 = 4.0;
/// Below this wheel height a touch back is the floor, not the wall (uu).
const FLOOR_HEIGHT: f32 = 60.0;
/// The roof's up.z on the floor (at least) and the ceiling (at most): no attempt from there.
const FLOOR_UP_Z: f32 = 0.98;
const CEILING_UP_Z: f32 = -0.7;
/// A wall whose normal has more |z| than this is a curve...
const CURVE_UP_Z: f32 = 0.1;
/// ...and one whose normal is less than this along x or y is a corner.
const CORNER_AXIS: f32 = 0.98;
/// The flip angle needs the car turned sideways on the wall: its right side's |z| at least this.
const ANGLE_MIN_RIGHT_Z: f32 = 0.2;
/// Flip angle bands (degrees toward the ceiling): green inside, yellow inside, red outside.
const FLIP_GREEN: (f32, f32) = (50.0, 70.0);
const FLIP_YELLOW: (f32, f32) = (45.0, 85.0);
/// A diagonal dodge of the action table is 45 degrees exactly, up to rounding.
const ANGLE_EPS: f32 = 0.01;
/// First jump held: green up to 6 ticks (50 ms), yellow up to 9 (75 ms).
const HOLD_BAND: (f32, f32) = (6.0, 9.0);
/// Press to press (green up to, yellow up to; ticks) by the speed along the wall at the first press: under
/// 1300, 1300 to 1500, 1500 to 1700, 1700 to 1900, 1900 and up (uu/s).
const PRESS_BAND_SPEEDS: [f32; 4] = [1300.0, 1500.0, 1700.0, 1900.0];
const PRESS_BANDS: [(f32, f32); 5] = [
    (12.0, 14.0),
    (13.0, 13.0),
    (13.0, 14.0),
    (14.0, 15.0),
    (15.0, 16.0),
];
/// A chain goes on when the next first jump comes this soon after the wheels were back...
const CHAIN_GAP_TICKS: f32 = 120.0;
/// ...without the wheels off every surface longer than this in between.
const CHAIN_MAX_AIR_TICKS: f32 = 6.0;
/// The flip angle when it cannot be measured.
const NO_ANGLE: f32 = 1000.0;

// Memory, one number each (all 0 when an episode starts).
/// 0 no attempt, 1 attempt under way, 2 a make waiting for its speed window.
const PHASE: usize = 0;
/// Ticks since the first press, at the step's end.
const TICKS: usize = 1;
/// The wall's normal at the first press (x, y, z).
const NORMAL: usize = 2;
/// Speed along the wall at the first press.
const SPEED_START: usize = 5;
/// Ticks the first jump was held, press to press (−1 until known), jump presses so far.
const HOLD: usize = 6;
const PRESS_TO_PRESS: usize = 7;
const PRESSES: usize = 8;
/// 1 once the second press gave a dodge.
const DODGED: usize = 9;
const FLIP_ANGLE: usize = 10;
/// 1 once the wheels were off the wall.
const LEFT_WALL: usize = 11;
/// Ticks at the first touch back and when the wheels were back (−1 until then).
const FIRST_TOUCH: usize = 12;
const REATTACH: usize = 13;
/// The highest speed along the wall after the wheels were back.
const PEAK: usize = 14;
/// 0 flat wall, 1 curve, 2 corner.
const SURFACE: usize = 15;
/// Chain: 1 while it can go on, ticks since the last make's wheels were back, ticks in the air since, and
/// how many attempts it has so far (the current one included).
const CHAIN_ALIVE: usize = 16;
const CHAIN_SINCE: usize = 17;
const CHAIN_AIR: usize = 18;
const CHAIN_LENGTH: usize = 19;
const MEMORY: usize = 20;

/// Wall dashes, by the rules of samuelkassi's Wall Dash Coach: a first jump off a wall (not the floor or the
/// ceiling), a second press that dodges, and 3+ wheels back on the wall within 30 ticks of the first touch,
/// roof within about 26 degrees of the wall's normal. A make pays the speed gained along the wall (the
/// highest in the 18 ticks after the wheels are back, minus the speed at the first press) over `full_gain`,
/// at most 1, times the timing and flip angle grades (green 1, yellow `yellow_factor`, red `red_factor`
/// each), times 1 + `chain_bonus` when it continues a chain. Nothing for a make with 4+ jump presses or
/// gaining no more than `min_speed_gain`; −`miss_cost` for a dodge off the wall that is not a make.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WallDash {
    /// Speed gained along the wall for the full pay, before the grades (uu/s).
    #[serde(default = "five_hundred")]
    pub full_gain: f32,
    /// A make must gain more than this along the wall to pay (uu/s).
    #[serde(default)]
    pub min_speed_gain: f32,
    /// What a yellow grade (press to press, hold or flip angle) multiplies the pay by.
    #[serde(default = "half")]
    pub yellow_factor: f32,
    /// What a red grade multiplies the pay by.
    #[serde(default = "quarter")]
    pub red_factor: f32,
    /// Extra pay, as a share of its own, for a make that continues a chain: its first jump within 120 ticks
    /// of the last make's wheels coming back. 0 = off.
    #[serde(default)]
    pub chain_bonus: f32,
    /// Also pays dashes on the curves between the walls and the floor or ceiling.
    #[serde(default)]
    pub curves: bool,
    /// Also pays dashes in the corners between two walls.
    #[serde(default)]
    pub corners: bool,
    /// Charged (as a negative pay) for a dodge off an allowed wall that is not a make. 0 = off.
    #[serde(default)]
    pub miss_cost: f32,
}

fn five_hundred() -> f32 {
    500.0
}

fn half() -> f32 {
    0.5
}

fn quarter() -> f32 {
    0.25
}

/// Green, yellow or red.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grade {
    Green,
    Yellow,
    Red,
}

/// `ticks` against the band (green up to, yellow up to). The step's whole ticks are all a bot can press on,
/// so each edge is rounded up to the next whole step of `step_ticks`.
fn timing_grade(ticks: f32, (green, yellow): (f32, f32), step_ticks: f32) -> Grade {
    let on_steps = |edge: f32| (edge / step_ticks).ceil() * step_ticks;
    if ticks <= on_steps(green) {
        Grade::Green
    } else if ticks <= on_steps(yellow) {
        Grade::Yellow
    } else {
        Grade::Red
    }
}

/// The press to press band for this speed along the wall at the first press.
fn press_band(speed: f32) -> (f32, f32) {
    let index = PRESS_BAND_SPEEDS
        .iter()
        .position(|&limit| speed < limit)
        .unwrap_or(PRESS_BANDS.len() - 1);
    PRESS_BANDS[index]
}

/// The flip angle's grade; a dodge aimed at the floor or backwards is red.
fn angle_grade(degrees: f32) -> Grade {
    let inside =
        |(low, high): (f32, f32)| degrees >= low - ANGLE_EPS && degrees <= high + ANGLE_EPS;
    if inside(FLIP_GREEN) {
        Grade::Green
    } else if inside(FLIP_YELLOW) {
        Grade::Yellow
    } else {
        Grade::Red
    }
}

/// How far the dodge points toward the ceiling, in degrees: 0 straight ahead, 90 straight up the wall,
/// negative toward the floor. None when the car is not turned sideways enough on the wall to tell.
fn flip_angle(torque: Vec3A, right: Vec3A) -> Option<f32> {
    if right.z.abs() < ANGLE_MIN_RIGHT_Z {
        return None;
    }
    // RocketSim keeps the dodge's direction as torque (−sideways, forward) in the car's frame.
    let (forward, sideways) = (torque.y, -torque.x);
    Some((sideways * right.z.signum()).atan2(forward).to_degrees())
}

/// The wall type of a normal: 0 flat wall, 1 curve, 2 corner.
fn surface(normal: Vec3A) -> f32 {
    if normal.z.abs() > CURVE_UP_Z {
        1.0
    } else if normal.x.abs() >= CORNER_AXIS || normal.y.abs() >= CORNER_AXIS {
        0.0
    } else {
        2.0
    }
}

/// `velocity` along the plane with this normal.
fn speed_along(velocity: Vec3A, normal: Vec3A) -> f32 {
    (velocity - normal * velocity.dot(normal)).length()
}

impl WallDash {
    fn grade_factor(&self, grade: Grade) -> f32 {
        match grade {
            Grade::Green => 1.0,
            Grade::Yellow => self.yellow_factor,
            Grade::Red => self.red_factor,
        }
    }

    fn surface_allowed(&self, kind: f32) -> bool {
        kind == 0.0 || (kind == 1.0 && self.curves) || (kind == 2.0 && self.corners)
    }

    /// Ends the attempt in `m` and returns its pay.
    fn finish(&self, m: &mut [f32], step_ticks: f32) -> f32 {
        let make = m[PHASE] == 2.0;
        m[PHASE] = 0.0;
        let allowed = self.surface_allowed(m[SURFACE]);
        if !make {
            m[CHAIN_ALIVE] = 0.0;
            m[CHAIN_LENGTH] = 0.0;
            return if allowed && m[DODGED] == 1.0 {
                -self.miss_cost
            } else {
                0.0
            };
        }
        m[CHAIN_ALIVE] = 1.0;
        m[CHAIN_SINCE] = m[TICKS] - m[REATTACH];
        m[CHAIN_AIR] = 0.0;
        let gain = m[PEAK] - m[SPEED_START];
        if !allowed || m[PRESSES] >= SPAM_PRESSES || gain <= self.min_speed_gain.max(0.0) {
            return 0.0;
        }
        let press = timing_grade(m[PRESS_TO_PRESS], press_band(m[SPEED_START]), step_ticks);
        let hold = timing_grade(m[HOLD], HOLD_BAND, step_ticks);
        let mut quality = self.grade_factor(press) * self.grade_factor(hold);
        if m[FLIP_ANGLE] != NO_ANGLE {
            quality *= self.grade_factor(angle_grade(m[FLIP_ANGLE]));
        }
        let chain = if m[CHAIN_LENGTH] >= 2.0 {
            1.0 + self.chain_bonus
        } else {
            1.0
        };
        (gain / self.full_gain.max(1.0)).min(1.0) * quality * chain
    }

    /// A first jump off a wall at the step's start: a new attempt.
    fn start(m: &mut [f32], was: &CarSnapshot, step_ticks: f32) {
        let normal = was.body.up();
        let continuing = m[CHAIN_ALIVE] == 1.0;
        let length = if continuing {
            m[CHAIN_LENGTH] + 1.0
        } else {
            1.0
        };
        m[..CHAIN_ALIVE].fill(0.0);
        m[PHASE] = 1.0;
        m[TICKS] = step_ticks;
        m[NORMAL..NORMAL + 3].copy_from_slice(&normal.to_array());
        m[SPEED_START] = speed_along(was.body.vel, normal);
        m[HOLD] = -1.0;
        m[PRESS_TO_PRESS] = -1.0;
        m[PRESSES] = 1.0;
        m[FLIP_ANGLE] = NO_ANGLE;
        m[FIRST_TOUCH] = -1.0;
        m[REATTACH] = -1.0;
        m[SURFACE] = surface(normal);
        m[CHAIN_ALIVE] = 0.0;
        m[CHAIN_LENGTH] = length;
    }
}

impl Reward for WallDash {
    fn memory(&self) -> usize {
        MEMORY
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        let (was, now) = (&prev.cars[car], &cur.cars[car]);
        let (detail, before) = (events.detail(car), events.prev_detail(car));
        let step_ticks = (events.step_seconds / TICK_SECONDS).round().max(1.0);
        if now.is_demoed {
            m.fill(0.0);
            return 0.0;
        }
        let pressed = detail.controls.jump && !before.controls.jump;
        let wheels = detail.wheel_contacts.iter().filter(|&&w| w).count();
        let mut pay = 0.0;

        let first_jump = pressed && was.on_ground && !was.has_jumped && now.has_jumped;
        if first_jump {
            // The press that starts the next jump belongs to the next attempt.
            if m[PHASE] != 0.0 {
                pay += self.finish(m, step_ticks);
            }
            let up_z = was.body.up().z;
            if up_z >= FLOOR_UP_Z || up_z <= CEILING_UP_Z {
                m[CHAIN_ALIVE] = 0.0;
                m[CHAIN_LENGTH] = 0.0;
            } else {
                Self::start(m, was, step_ticks);
            }
            return pay;
        }

        if m[PHASE] == 0.0 {
            if m[CHAIN_ALIVE] == 1.0 {
                m[CHAIN_SINCE] += step_ticks;
                if wheels > 0 {
                    m[CHAIN_AIR] = 0.0;
                } else if !events.cars[car].on_arena {
                    m[CHAIN_AIR] += step_ticks;
                }
                let on_floor =
                    wheels > 0 && (now.body.pos.z < FLOOR_HEIGHT || now.body.up().z >= FLOOR_UP_Z);
                if m[CHAIN_SINCE] > CHAIN_GAP_TICKS
                    || m[CHAIN_AIR] > CHAIN_MAX_AIR_TICKS
                    || on_floor
                {
                    m[CHAIN_ALIVE] = 0.0;
                    m[CHAIN_LENGTH] = 0.0;
                }
            }
            return pay;
        }

        let normal = Vec3A::from_slice(&m[NORMAL..NORMAL + 3]);
        let step_start = m[TICKS];
        m[TICKS] += step_ticks;
        let ticks = m[TICKS];
        if pressed {
            m[PRESSES] += 1.0;
            let dodged = (now.has_flipped && !was.has_flipped)
                || (detail.is_flipping && detail.flip_time <= events.step_seconds + 1e-4);
            if m[DODGED] == 0.0 && dodged {
                m[DODGED] = 1.0;
                m[PRESS_TO_PRESS] = step_start;
                m[FLIP_ANGLE] =
                    flip_angle(detail.flip_rel_torque, was.body.rot.y_axis).unwrap_or(NO_ANGLE);
            } else if m[PRESS_TO_PRESS] < 0.0 {
                m[PRESS_TO_PRESS] = step_start;
            }
        }
        if !detail.controls.jump && m[HOLD] < 0.0 {
            m[HOLD] = step_start;
        }

        if m[PHASE] == 2.0 {
            m[PEAK] = m[PEAK].max(speed_along(now.body.vel, normal));
            if ticks - m[REATTACH] >= SPEED_WINDOW_TICKS {
                pay += self.finish(m, step_ticks);
            }
            return pay;
        }

        if m[LEFT_WALL] == 0.0 {
            if wheels == 0 {
                m[LEFT_WALL] = 1.0;
            }
        } else if wheels > 0 {
            if now.body.pos.z < FLOOR_HEIGHT {
                return pay + self.finish(m, step_ticks);
            }
            if m[FIRST_TOUCH] < 0.0 {
                m[FIRST_TOUCH] = ticks;
            }
            if wheels >= MAKE_MIN_WHEELS && ticks - m[FIRST_TOUCH] <= MAKE_WINDOW_TICKS {
                let upright = now.body.up().dot(normal) > UPRIGHT_DOT;
                if m[DODGED] == 1.0 && upright {
                    m[PHASE] = 2.0;
                    m[REATTACH] = ticks;
                    m[PEAK] = speed_along(now.body.vel, normal);
                    return pay;
                }
                return pay + self.finish(m, step_ticks);
            }
        }
        let late_touch = m[FIRST_TOUCH] >= 0.0 && ticks - m[FIRST_TOUCH] > MAKE_WINDOW_TICKS;
        if late_touch || ticks >= TIMEOUT_TICKS {
            pay += self.finish(m, step_ticks);
        }
        pay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// The car's turn on the +x side wall heading +y: forward +y, right side down, roof toward the field.
    const ON_RIGHT_WALL: Mat3A = Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_Z, Vec3A::NEG_X);

    fn reward() -> WallDash {
        serde_json::from_str("{}").unwrap()
    }

    /// The dodge's torque for a flip `degrees` toward the ceiling, for a car turned as `ON_RIGHT_WALL`.
    fn toward_ceiling(degrees: f32) -> Vec3A {
        let (sin, cos) = degrees.to_radians().sin_cos();
        Vec3A::new(sin, cos, 0.0)
    }

    /// Made-up steps of one car, `skip` ticks each, starting on a wall at 1500 uu/s along it.
    struct Run {
        reward: WallDash,
        game: GameSnapshot,
        events: StepEvents,
        memory: [f32; MEMORY],
    }

    impl Run {
        fn new(reward: WallDash, skip: f32, roof: Mat3A) -> Self {
            let mut game = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::ZERO)]);
            let car = &mut game.cars[0];
            car.body.rot = roof;
            car.body.pos = Vec3A::new(SIDE_WALL_X - CAR_REST_Z, 0.0, 800.0);
            car.body.vel = roof.x_axis * 1500.0;
            let mut events = quiet(1);
            events.step_seconds = skip / 120.0;
            events.details[0].wheel_contacts = [true; 4];
            events.prev_details = events.details.clone();
            Self {
                reward,
                game,
                events,
                memory: [0.0; MEMORY],
            }
        }

        /// One step with jump held or not; `edit` sets the car's state at the step's end.
        fn step(&mut self, jump: bool, edit: impl FnOnce(&mut CarSnapshot, &mut CarDetail)) -> f32 {
            let prev = self.game.clone();
            self.events.prev_details = self.events.details.clone();
            let mut detail = CarDetail {
                wheel_contacts: self.events.details[0].wheel_contacts,
                ..CarDetail::default()
            };
            detail.controls.jump = jump;
            edit(&mut self.game.cars[0], &mut detail);
            self.events.cars[0].on_arena = detail.wheel_contacts.contains(&true);
            self.events.details = vec![detail];
            let (game, events) = (&self.game, &self.events);
            self.reward
                .value_remembering(&prev, game, events, 0, &mut self.memory)
        }

        fn jump_off(&mut self) -> f32 {
            self.step(true, |car, _| car.has_jumped = true)
        }

        fn fly(&mut self, jump: bool) -> f32 {
            self.step(jump, |car, d| {
                car.on_ground = false;
                d.wheel_contacts = [false; 4];
            })
        }

        fn dodge(&mut self, torque: Vec3A) -> f32 {
            let flip_time = self.events.step_seconds;
            self.step(true, |car, d| {
                car.on_ground = false;
                car.has_flipped = true;
                d.wheel_contacts = [false; 4];
                d.is_flipping = true;
                d.flip_time = flip_time;
                d.flip_rel_torque = torque;
            })
        }

        /// Wheels on something at the step's end, the roof as `roof`, going `speed` along +y.
        fn wheels(&mut self, count: usize, roof: Mat3A, speed: f32) -> f32 {
            self.step(false, |car, d| {
                car.on_ground = count >= 3;
                car.has_flipped = false;
                car.has_jumped = false;
                car.body.rot = roof;
                car.body.vel = Vec3A::new(0.0, speed, 0.0);
                d.wheel_contacts = [0, 1, 2, 3].map(|w| w < count);
                d.is_flipping = true;
            })
        }

        /// Riding the wall at `speed` for `steps` steps; what they paid.
        fn ride(&mut self, steps: usize, speed: f32) -> Vec<f32> {
            (0..steps)
                .map(|_| self.wheels(4, ON_RIGHT_WALL, speed))
                .collect()
        }

        /// Jump, release, dodge at `degrees` toward the ceiling: what it paid (nothing yet on a make).
        fn dash(&mut self, degrees: f32) -> f32 {
            self.jump_off() + self.fly(false) + self.dodge(toward_ceiling(degrees))
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn a_clean_make_on_a_flat_wall_pays_its_speed_gain_after_the_speed_window() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        assert_eq!(run.dash(60.0), 0.0);
        assert_eq!(
            run.wheels(4, ON_RIGHT_WALL, 1700.0),
            0.0,
            "back: the speed window opens"
        );
        let paid = run.ride(5, 1750.0);
        // Back at 16 ticks; the window ends at the first step 18+ ticks later (36).
        assert_eq!(paid[..4], [0.0; 4]);
        assert!(approx(paid[4], 250.0 / 500.0), "{paid:?}");
        assert_eq!(run.ride(3, 1750.0), [0.0; 3], "paid once");
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 2400.0);
        assert!(approx(run.ride(5, 2400.0)[4], 1.0), "at most 1");
    }

    #[test]
    fn the_flip_angle_grades_a_make_and_a_flip_away_from_the_wall_is_a_miss() {
        for (degrees, factor) in [
            (55.0, 1.0),
            (45.0, 0.5),
            (80.0, 0.5),
            (30.0, 0.25),
            (88.0, 0.25),
        ] {
            let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
            run.dash(degrees);
            run.wheels(4, ON_RIGHT_WALL, 1700.0);
            let paid = run.ride(5, 1700.0)[4];
            assert!(approx(paid, 0.4 * factor), "{degrees}: {paid}");
        }
        let miss_cost = WallDash {
            miss_cost: 0.3,
            ..reward()
        };
        let mut run = Run::new(miss_cost, 4.0, ON_RIGHT_WALL);
        assert_eq!(run.dash(-40.0), 0.0, "aimed at the floor");
        let paid: Vec<f32> = (0..30).map(|_| run.fly(false)).collect();
        // The attempt times out 120 ticks after the first press: 30 steps of 4 ticks, 3 of them above.
        assert_eq!(paid.iter().filter(|&&p| p != 0.0).count(), 1);
        assert!(approx(paid[26], -0.3), "the miss costs miss_cost once");
    }

    #[test]
    fn a_late_second_press_is_graded_red_and_pays_nothing_when_it_misses() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.jump_off();
        (0..4).for_each(|_| assert_eq!(run.fly(false), 0.0));
        run.dodge(toward_ceiling(60.0));
        // Press to press 20 ticks at 1500 uu/s: yellow ends at 14 ticks, 16 on whole steps of 4.
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert!(approx(run.ride(5, 1700.0)[4], 0.4 * 0.25));
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.jump_off();
        (0..4).for_each(|_| assert_eq!(run.fly(false), 0.0));
        run.dodge(toward_ceiling(60.0));
        run.wheels(2, ON_RIGHT_WALL, 1700.0);
        let paid: Vec<f32> = (0..10)
            .map(|_| run.wheels(2, ON_RIGHT_WALL, 1700.0))
            .collect();
        assert_eq!(paid, [0.0; 10], "only two wheels back: a miss");
        assert_eq!(
            run.memory[PHASE], 0.0,
            "over 30 ticks after the first touch"
        );
    }

    #[test]
    fn not_landing_back_upright_on_the_wall_is_a_miss() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        let twisted = Mat3A::from_rotation_y(0.6) * ON_RIGHT_WALL;
        assert!(twisted.z_axis.dot(Vec3A::NEG_X) < UPRIGHT_DOT);
        run.wheels(4, twisted, 1700.0);
        assert_eq!(run.ride(8, 1700.0), [0.0; 8], "came back twisted");
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.step(false, |car, d| {
            car.body.pos.z = 40.0;
            d.wheel_contacts = [true; 4];
        });
        assert_eq!(run.memory[PHASE], 0.0, "landed on the floor");
        assert_eq!(run.ride(8, 1700.0), [0.0; 8]);
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.jump_off();
        run.fly(false);
        run.step(true, |car, d| {
            car.has_double_jumped = true;
            d.wheel_contacts = [false; 4];
        });
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert_eq!(run.ride(8, 1700.0), [0.0; 8], "a double jump, not a dodge");
    }

    #[test]
    fn curves_and_corners_pay_only_when_allowed_and_no_attempt_starts_on_the_floor_or_ceiling() {
        let curve = Mat3A::from_rotation_y(-0.6) * ON_RIGHT_WALL;
        assert!(curve.z_axis.z.abs() > CURVE_UP_Z);
        let corner = Mat3A::from_rotation_z(0.4) * ON_RIGHT_WALL;
        assert!(corner.z_axis.x.abs() < CORNER_AXIS && corner.z_axis.z.abs() < CURVE_UP_Z);
        for (roof, allowed) in [
            (
                curve,
                WallDash {
                    curves: true,
                    ..reward()
                },
            ),
            (
                corner,
                WallDash {
                    corners: true,
                    ..reward()
                },
            ),
        ] {
            for (reward, pays) in [(reward(), false), (allowed, true)] {
                let mut run = Run::new(reward, 4.0, roof);
                run.dash(60.0);
                run.wheels(4, roof, 2000.0);
                let paid = (0..5).map(|_| run.wheels(4, roof, 2000.0)).sum::<f32>();
                assert_eq!(paid > 0.0, pays, "{roof:?}");
            }
        }
        for roof in [
            Mat3A::IDENTITY,
            Mat3A::from_rotation_x(std::f32::consts::PI),
        ] {
            let mut run = Run::new(reward(), 4.0, roof);
            run.jump_off();
            assert_eq!(run.memory[PHASE], 0.0, "no attempt from {roof:?}");
        }
    }

    #[test]
    fn mashing_jump_pays_nothing() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.fly(false);
        run.fly(true);
        run.fly(false);
        run.fly(true);
        assert_eq!(run.memory[PRESSES], 4.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert_eq!(run.ride(6, 1700.0), [0.0; 6]);
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.fly(false);
        run.fly(true);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert!(
            run.ride(6, 1700.0).iter().sum::<f32>() > 0.0,
            "three presses still pay"
        );
    }

    #[test]
    fn a_make_must_gain_speed_along_the_wall() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1400.0);
        assert_eq!(
            run.ride(6, 1450.0),
            [0.0; 6],
            "slower than at the first press"
        );
        let mut run = Run::new(
            WallDash {
                min_speed_gain: 300.0,
                ..reward()
            },
            4.0,
            ON_RIGHT_WALL,
        );
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert_eq!(run.ride(6, 1750.0), [0.0; 6], "250 gained, 300 needed");
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.game.cars[0].body.vel = Vec3A::new(-900.0, 1500.0, 0.0);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert!(
            approx(run.ride(5, 1700.0)[4], 0.4),
            "speed into the wall does not count"
        );
    }

    #[test]
    fn a_chain_pays_its_bonus_when_the_next_dash_comes_soon_enough() {
        let chained = || WallDash {
            chain_bonus: 0.5,
            ..reward()
        };
        let mut run = Run::new(chained(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert!(approx(run.ride(5, 1700.0)[4], 0.4), "the first of a chain");
        run.ride(2, 1700.0);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1900.0);
        assert!(approx(run.ride(5, 1900.0)[4], 0.4 * 1.5), "the second");
        assert_eq!(run.memory[CHAIN_LENGTH], 2.0);
        // Jumping again inside the speed window ends the make there and chains on.
        run.ride(2, 1900.0);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 2100.0);
        let early = run.ride(1, 2100.0)[0] + run.jump_off();
        assert!(approx(early, 0.4 * 1.5), "{early}");
        assert_eq!(run.memory[CHAIN_LENGTH], 4.0);

        let mut run = Run::new(chained(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        run.ride(5, 1700.0);
        run.ride(30, 1500.0);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        assert!(
            approx(run.ride(5, 1700.0)[4], 0.4),
            "too long between: a new chain"
        );
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1700.0);
        run.ride(5, 1700.0);
        run.dash(60.0);
        run.wheels(4, ON_RIGHT_WALL, 1900.0);
        assert!(
            approx(run.ride(5, 1900.0)[4], 0.4),
            "chain_bonus is off by default"
        );
    }

    #[test]
    fn a_demolished_car_drops_its_attempt() {
        let mut run = Run::new(reward(), 4.0, ON_RIGHT_WALL);
        run.dash(60.0);
        run.step(false, |car, _| car.is_demoed = true);
        assert_eq!(run.memory, [0.0; MEMORY]);
    }

    #[test]
    fn timings_are_graded_on_whole_steps() {
        let band = press_band(1400.0);
        assert_eq!(band, (13.0, 13.0));
        assert_eq!(press_band(1500.0), (13.0, 14.0));
        assert_eq!(press_band(1299.0), (12.0, 14.0));
        assert_eq!(press_band(1900.0), (15.0, 16.0));
        let at = |ticks, skip| timing_grade(ticks, band, skip);
        assert_eq!([at(13.0, 1.0), at(14.0, 1.0)], [Grade::Green, Grade::Red]);
        assert_eq!([at(16.0, 4.0), at(20.0, 4.0)], [Grade::Green, Grade::Red]);
        assert_eq!([at(16.0, 8.0), at(24.0, 8.0)], [Grade::Green, Grade::Red]);
        let hold = |ticks, skip| timing_grade(ticks, HOLD_BAND, skip);
        assert_eq!(
            [hold(6.0, 1.0), hold(9.0, 1.0), hold(10.0, 1.0)],
            [Grade::Green, Grade::Yellow, Grade::Red]
        );
        assert_eq!(
            [hold(8.0, 4.0), hold(12.0, 4.0), hold(16.0, 4.0)],
            [Grade::Green, Grade::Yellow, Grade::Red]
        );
        // At tick skip 1 the press to press counts tick by tick.
        let mut run = Run::new(reward(), 1.0, ON_RIGHT_WALL);
        run.jump_off();
        (0..13).for_each(|_| {
            run.fly(false);
        });
        run.dodge(toward_ceiling(60.0));
        assert_eq!((run.memory[HOLD], run.memory[PRESS_TO_PRESS]), (1.0, 14.0));
    }

    #[test]
    fn the_flip_angle_counts_toward_the_ceiling_on_either_wall() {
        let right = ON_RIGHT_WALL.y_axis;
        let angle = |torque, right| flip_angle(torque, right).unwrap();
        assert!((angle(toward_ceiling(60.0), right) - 60.0).abs() < 1e-3);
        assert!((angle(toward_ceiling(-20.0), right) + 20.0).abs() < 1e-3);
        // The left wall: right side up, so a dodge to the right goes up.
        assert!((angle(Vec3A::new(-1.0, 1.0, 0.0), -right) - 45.0).abs() < 1e-3);
        assert_eq!(
            angle_grade(angle(Vec3A::new(1.0, 1.0, 0.0), right)),
            Grade::Yellow
        );
        assert_eq!(
            flip_angle(Vec3A::Y, Vec3A::new(0.0, 1.0, 0.1)),
            None,
            "car pointing up the wall"
        );
        assert_eq!(angle_grade(135.0), Grade::Red, "backwards");
    }

    /// Scripted in RocketSim at tick skip 4 (action delay 1), from 1500 uu/s along the +x side wall: a jump
    /// held one step, a dodge diagonally up the wall two steps after the first press makes the dash and is
    /// paid; the same dodge down the wall misses and is charged `miss_cost`; a dodge straight up the wall comes back with
    /// almost no speed gained and earns almost nothing.
    #[test]
    fn a_scripted_wall_dash_in_rocketsim_pays_and_a_bad_one_does_not() {
        use crate::sim::Sim;
        use crate::testing::meshes_loaded;

        if !meshes_loaded() {
            return;
        }
        let (skip, delay) = (4, 1);
        let drive = Controls {
            throttle: 1.0,
            ..Controls::default()
        };
        let dash = |pitch: f32, roll: f32| {
            let mut sim = Sim::new(1, false, 7);
            let car = Body {
                pos: Vec3A::new(SIDE_WALL_X - CAR_REST_Z, 0.0, 800.0),
                rot: ON_RIGHT_WALL,
                vel: Vec3A::new(0.0, 1500.0, 0.0),
                ang_vel: Vec3A::ZERO,
            };
            sim.place(Body::RESTING, &[(car, 100.0, true)]);
            let (mut game, mut events) = (GameSnapshot::default(), StepEvents::default());
            sim.read(&mut game);
            events.clear(1);
            sim.details(&mut events);
            let reward = WallDash {
                miss_cost: 1.0,
                ..reward()
            };
            let mut memory = [0.0; MEMORY];
            let stick = Controls {
                pitch,
                roll,
                ..drive
            };
            let mut paid = Vec::new();
            for step in 0..40 {
                let controls = match step {
                    0..4 => drive,
                    4 => Controls {
                        jump: true,
                        ..drive
                    },
                    5 => stick,
                    6 => Controls {
                        jump: true,
                        ..stick
                    },
                    _ => drive,
                };
                let prev = game.clone();
                events.clear(1);
                events.step_seconds = skip as f32 / 120.0;
                sim.step(delay, &mut events);
                sim.set_controls(0, controls);
                sim.step(skip - delay, &mut events);
                sim.read(&mut game);
                std::mem::swap(&mut events.details, &mut events.prev_details);
                sim.details(&mut events);
                paid.push(reward.value_remembering(&prev, &game, &events, 0, &mut memory));
            }
            paid
        };
        let up_the_wall = dash(-1.0, -1.0);
        let made: Vec<f32> = up_the_wall.iter().copied().filter(|&p| p != 0.0).collect();
        assert!(made.len() == 1 && made[0] > 0.2, "{up_the_wall:?}");
        let down_the_wall = dash(-1.0, 1.0);
        let missed: Vec<f32> = down_the_wall
            .iter()
            .copied()
            .filter(|&p| p != 0.0)
            .collect();
        assert_eq!(missed, [-1.0], "only miss_cost, once");
        let straight_up = dash(0.0, -1.0);
        assert!(
            straight_up.iter().all(|&p| (0.0..0.01).contains(&p)),
            "{straight_up:?}"
        );
    }
}
