//! When an episode ends, and whether that ending is real (a goal) or an artificial cut-off.

use serde::Deserialize;

use crate::snapshot::{BALL_FLOOR_Z, GameSnapshot, StepEvents};

fn zero() -> f32 {
    0.0
}

const TICK_RATE: f32 = 120.0;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalConfig {
    /// Cut the episode after this many seconds without any car touching the ball (0 = never).
    pub no_touch_seconds: f32,
    /// Cut the episode after this many seconds in total (0 = never).
    pub max_episode_seconds: f32,
    /// Practice tasks: the episode ends (task done) when any car touches the ball.
    pub end_on_touch: bool,
    /// Practice tasks: the episode ends (task done) once every car is on its wheels.
    pub end_on_landing: bool,
    /// Chance (0 to 1) that the episode is cut off (not done) when the ball comes down to the floor after
    /// being in the air. 1 in the aerial-touch task, so only an aerial touch can end it; a smaller
    /// chance ends match episodes at natural pauses now and then.
    #[serde(default = "zero")]
    pub truncate_on_ball_landing: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    Continues,
    /// The game itself ended the episode (a goal), or a practice task ended: the value of what follows is
    /// zero.
    Terminal,
    /// We cut the episode short; the game would have continued, so the learner bootstraps.
    Truncated,
}

#[derive(Clone)]
pub struct EpisodeClock {
    config: TerminalConfig,
    ticks: u32,
    ticks_since_touch: u32,
    /// The ball was off the floor at the last look, so reaching the floor now is a landing.
    ball_was_up: bool,
}

impl EpisodeClock {
    pub fn new(config: TerminalConfig) -> Self {
        Self {
            config,
            ticks: 0,
            ticks_since_touch: 0,
            ball_was_up: false,
        }
    }

    /// Starts timing a new episode from its first state.
    pub fn restart(&mut self, snap: &GameSnapshot) {
        self.ticks = 0;
        self.ticks_since_touch = 0;
        self.ball_was_up = snap.ball.pos.z >= BALL_FLOOR_Z;
    }

    /// Ticks since the episode started.
    pub fn ticks(&self) -> u32 {
        self.ticks
    }

    /// `rng` is drawn from only when a landing happens and the truncation chance is strictly between 0 and 1,
    /// so runs without the option keep their random stream.
    pub fn advance(
        &mut self,
        ticks: u32,
        snap: &GameSnapshot,
        events: &StepEvents,
        rng: &mut fastrand::Rng,
    ) -> Ending {
        self.ticks += ticks;
        let touched = events.cars.iter().any(|car| car.touched_ball);
        if touched {
            self.ticks_since_touch = 0;
        } else {
            self.ticks_since_touch += ticks;
        }
        // A landing: the ball was off the floor at the last look and touched it during this step (the
        // contact is latched tick by tick in `Sim::step`; a bounce may be over by the step's end).
        let landed = self.ball_was_up && events.ball_on_floor;
        self.ball_was_up = snap.ball.pos.z >= BALL_FLOOR_Z;
        let chance = self.config.truncate_on_ball_landing;
        let cut_on_landing = landed && chance > 0.0 && (chance >= 1.0 || rng.f32() < chance);
        let exceeded =
            |seconds: f32, ticks: u32| seconds > 0.0 && ticks as f32 >= seconds * TICK_RATE;
        let landed = || snap.cars.iter().all(|car| car.on_ground);
        if snap.goal_scored.is_some()
            || (self.config.end_on_touch && touched)
            || (self.config.end_on_landing && landed())
        {
            Ending::Terminal
        } else if exceeded(self.config.no_touch_seconds, self.ticks_since_touch)
            || exceeded(self.config.max_episode_seconds, self.ticks)
            || cut_on_landing
        {
            Ending::Truncated
        } else {
            Ending::Continues
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{CarEvents, CarSnapshot, Team};

    const NO_LIMITS: TerminalConfig = TerminalConfig {
        no_touch_seconds: 0.0,
        max_episode_seconds: 0.0,
        end_on_touch: false,
        end_on_landing: false,
        truncate_on_ball_landing: 0.0,
    };

    fn rng() -> fastrand::Rng {
        fastrand::Rng::with_seed(1)
    }

    fn events(touched: bool) -> StepEvents {
        StepEvents {
            cars: vec![CarEvents {
                touched_ball: touched,
                ..CarEvents::default()
            }],
            ..StepEvents::default()
        }
    }

    /// A step during which the ball was on the floor at some tick.
    fn floor(touched: bool) -> StepEvents {
        StepEvents {
            ball_on_floor: true,
            ..events(touched)
        }
    }

    #[test]
    fn goals_are_terminal_and_timeouts_are_truncations() {
        let config = TerminalConfig {
            no_touch_seconds: 1.0,
            max_episode_seconds: 3.0,
            ..NO_LIMITS
        };
        let mut clock = EpisodeClock::new(config);
        let mut snap = GameSnapshot::default();
        assert_eq!(
            clock.advance(64, &snap, &events(false), &mut rng()),
            Ending::Continues
        );
        assert_eq!(
            clock.advance(64, &snap, &events(false), &mut rng()),
            Ending::Truncated
        );

        clock.restart(&snap);
        for _ in 0..44 {
            assert_eq!(
                clock.advance(8, &snap, &events(true), &mut rng()),
                Ending::Continues
            );
        }
        assert_eq!(
            clock.advance(8, &snap, &events(true), &mut rng()),
            Ending::Truncated,
            "3 s total"
        );

        clock.restart(&snap);
        snap.goal_scored = Some(Team::Orange);
        assert_eq!(
            clock.advance(8, &snap, &events(false), &mut rng()),
            Ending::Terminal
        );
    }

    #[test]
    fn a_landing_cuts_the_episode_with_the_given_chance() {
        let mut snap = GameSnapshot::default();
        let up = |snap: &mut GameSnapshot, z: f32| snap.ball.pos.z = z;
        let mut clock = EpisodeClock::new(TerminalConfig {
            truncate_on_ball_landing: 1.0,
            ..NO_LIMITS
        });
        // The ball starts on the floor: staying there (floor contact every tick) is no landing.
        up(&mut snap, 93.15);
        clock.restart(&snap);
        assert_eq!(
            clock.advance(8, &snap, &floor(false), &mut rng()),
            Ending::Continues
        );
        // Up, then a floor contact during a step: a landing, even though the ball has bounced back up by
        // the step's end.
        up(&mut snap, 400.0);
        assert_eq!(
            clock.advance(8, &snap, &events(false), &mut rng()),
            Ending::Continues
        );
        up(&mut snap, 300.0);
        assert_eq!(
            clock.advance(8, &snap, &floor(false), &mut rng()),
            Ending::Truncated
        );
        // A touch on the landing step still ends the task (terminal beats truncation).
        up(&mut snap, 400.0);
        let mut touch_first = EpisodeClock::new(TerminalConfig {
            end_on_touch: true,
            truncate_on_ball_landing: 1.0,
            ..NO_LIMITS
        });
        touch_first.restart(&snap);
        assert_eq!(
            touch_first.advance(8, &snap, &floor(true), &mut rng()),
            Ending::Terminal
        );

        // Half the landings, over many episodes, with one draw per landing.
        let mut clock = EpisodeClock::new(TerminalConfig {
            truncate_on_ball_landing: 0.5,
            ..NO_LIMITS
        });
        let mut rng = rng();
        let mut cut = 0;
        for _ in 0..1000 {
            up(&mut snap, 400.0);
            clock.restart(&snap);
            up(&mut snap, 95.0);
            if clock.advance(8, &snap, &floor(false), &mut rng) == Ending::Truncated {
                cut += 1;
            }
        }
        assert!(
            (440..=560).contains(&cut),
            "{cut} of 1000 landings cut the episode"
        );
    }

    #[test]
    fn zero_disables_a_limit() {
        let mut clock = EpisodeClock::new(NO_LIMITS);
        let snap = GameSnapshot::default();
        assert_eq!(
            clock.advance(120 * 3600, &snap, &events(false), &mut rng()),
            Ending::Continues
        );
    }

    #[test]
    fn practice_tasks_end_on_a_touch_or_once_every_car_has_landed() {
        let snap = GameSnapshot::default();
        let mut clock = EpisodeClock::new(TerminalConfig {
            end_on_touch: true,
            ..NO_LIMITS
        });
        assert_eq!(
            clock.advance(8, &snap, &events(false), &mut rng()),
            Ending::Continues
        );
        assert_eq!(
            clock.advance(8, &snap, &events(true), &mut rng()),
            Ending::Terminal
        );
        assert_eq!(clock.ticks(), 16);

        let mut clock = EpisodeClock::new(TerminalConfig {
            end_on_landing: true,
            ..NO_LIMITS
        });
        let car = |on_ground| CarSnapshot {
            on_ground,
            ..CarSnapshot::default()
        };
        let mut snap = GameSnapshot {
            cars: vec![car(true), car(false)],
            ..GameSnapshot::default()
        };
        assert_eq!(
            clock.advance(8, &snap, &events(false), &mut rng()),
            Ending::Continues
        );
        snap.cars[1].on_ground = true;
        assert_eq!(
            clock.advance(8, &snap, &events(false), &mut rng()),
            Ending::Terminal
        );
    }

    /// One tick of a car: touched the ball, on the floor, on another arena surface (a wall).
    #[derive(Clone, Copy)]
    enum Tick {
        Air,
        AirTouch,
        Wall,
        WallTouch,
        Floor,
        FloorTouch,
    }

    /// The contact fields of a step made of `ticks`, the first at arena tick `first`.
    fn step_of(first: u64, ticks: &[Tick]) -> CarEvents {
        step_with_ball_down(first, ticks, None)
    }

    /// `step_of`, with the ball on the field's floor on arena tick `down` (if any).
    fn step_with_ball_down(first: u64, ticks: &[Tick], down: Option<u64>) -> CarEvents {
        let mut car = CarEvents::default();
        for (tick, &kind) in (first..).zip(ticks) {
            let (touched, floor, arena) = match kind {
                Tick::Air => (false, false, false),
                Tick::AirTouch => (true, false, false),
                Tick::Wall => (false, false, true),
                Tick::WallTouch => (true, false, true),
                Tick::Floor => (false, true, false),
                Tick::FloorTouch => (true, true, false),
            };
            car.add_tick(tick, touched, floor, arena, down == Some(tick));
        }
        car
    }

    #[test]
    fn the_contact_fields_keep_the_order_of_a_step() {
        use Tick::*;
        let car = step_of(100, &[Wall, AirTouch, Air, Floor, FloorTouch]);
        assert!(
            car.on_floor && car.on_arena && car.touched_ball_on_floor && car.touched_ball_in_air
        );
        assert_eq!(
            (car.first_touch_tick, car.last_touch_tick),
            (Some(101), Some(104))
        );
        assert_eq!(car.first_arena_tick, Some(100));
        assert_eq!(car.first_air_touch_tick, Some(101));
        assert_eq!(car.air_start_tick, Some(101));
        assert_eq!(car.arena_after_air_start_tick, Some(103));
        let wall = step_of(0, &[Air, WallTouch]);
        assert!(
            wall.on_arena && !wall.on_floor,
            "a wall is arena, not floor"
        );
        assert!(!wall.touched_ball_on_floor && !wall.touched_ball_in_air);
        assert_eq!(wall.arena_after_air_start_tick, None);
        assert_eq!(wall.first_air_touch_tick, None);
        // A touch in the air on a ball on the floor (a jump pop) is no starting touch; a landing after it
        // is no arena contact after a start
        let pop = step_with_ball_down(0, &[AirTouch, Air, Floor, Air, AirTouch], Some(0));
        assert!(pop.touched_ball_in_air);
        assert_eq!(pop.first_air_touch_tick, Some(0));
        assert_eq!(pop.air_start_tick, Some(4));
        assert_eq!(pop.arena_after_air_start_tick, None);
        let floor = step_of(0, &[Floor]);
        assert!(floor.on_arena, "the floor is arena too");
        assert_eq!(CarEvents::default(), step_of(0, &[Air, Air]));
    }
}
