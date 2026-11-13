//! Wavedash: a dodge whose wheels land on the floor right after it starts, paid by the speed it gained.
//! group: moves

use crate::custom::prelude::*;

/// The dodge must have started at most this long before the wheels touch down (s).
const MAX_FLIP_AGE: f32 = 0.2;
/// Speed gained for the full reward (uu/s): a dodge's own push is 500.
const FULL_GAIN: f32 = 500.0;

/// The car came down onto its wheels this step from a dodge that had just started: in the air on the step
/// before and dodging for at most `MAX_FLIP_AGE`, or still able to dodge and pressing jump with the stick
/// tilted this step (dodge and landing inside one step).
fn dash_landing(prev: &GameSnapshot, cur: &GameSnapshot, events: &StepEvents, car: usize) -> bool {
    let (was, now) = (&prev.cars[car], &cur.cars[car]);
    if was.on_ground || !now.on_ground || now.is_demoed {
        return false;
    }
    let before = events.prev_detail(car);
    let fresh_dodge = was.has_flipped && before.flip_time <= MAX_FLIP_AGE;
    let c = events.detail(car).controls;
    let dodge_this_step = was.can_flip
        && c.jump
        && !before.controls.jump
        && c.pitch.abs() + c.yaw.abs() + c.roll.abs() > 0.1;
    fresh_dodge || dodge_this_step
}

/// On a dash landing onto the floor: the speed gained since before the dodge over `FULL_GAIN`, 0 to 1.
/// Memory: [speed before the dodge, 1 once that is known].
fn dash_value(
    prev: &GameSnapshot,
    cur: &GameSnapshot,
    events: &StepEvents,
    car: usize,
    memory: &mut [f32],
) -> f32 {
    let me = &cur.cars[car];
    let speed = me.body.vel.length();
    let floor = me.body.up().z > 0.7;
    let paid = if floor && memory[1] > 0.0 && dash_landing(prev, cur, events, car) {
        ((speed - memory[0]) / FULL_GAIN).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if !me.has_flipped {
        memory[0] = speed;
        memory[1] = 1.0;
    }
    paid
}

/// When a dodge lands the car's wheels on the floor right after it starts (a wavedash): the speed it
/// gained since before the dodge over 500 uu/s, 0 to 1. 0 otherwise.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wavedash {}

impl Reward for Wavedash {
    fn memory(&self) -> usize {
        2
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        dash_value(prev, cur, events, car, memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// The reward on a landing whose speed before the dodge was 1000 uu/s and is `after` now.
    fn pay<R: Reward>(
        r: &R,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        e: &StepEvents,
        after: f32,
    ) -> f32 {
        let mut cur = cur.clone();
        cur.cars[0].body.vel = Vec3A::new(0.0, after, 0.0);
        r.value_remembering(prev, &cur, e, 0, &mut [1000.0, 1.0])
    }

    /// A car in the air on the step before, dodging for `flip_age` s, and on its wheels now; `on_wall`
    /// turns it so its roof faces sideways.
    fn landing(flip_age: f32, on_wall: bool) -> (GameSnapshot, GameSnapshot, StepEvents) {
        let mut prev = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[(Team::Blue, Vec3A::new(0.0, 0.0, 30.0))],
        );
        prev.cars[0].on_ground = false;
        prev.cars[0].has_flipped = true;
        let mut cur = prev.clone();
        cur.cars[0].on_ground = true;
        cur.cars[0].has_flipped = false;
        if on_wall {
            for g in [&mut prev, &mut cur] {
                g.cars[0].body.rot = Mat3A::from_rotation_y(std::f32::consts::FRAC_PI_2);
            }
        }
        let mut events = quiet(1);
        events.prev_details[0].is_flipping = true;
        events.prev_details[0].flip_time = flip_age;
        (prev, cur, events)
    }

    #[test]
    fn wavedash_pays_the_speed_gained_on_the_floor() {
        let (prev, cur, e) = landing(0.05, false);
        assert_eq!(pay(&Wavedash {}, &prev, &cur, &e, 1500.0), 1.0);
        assert!((pay(&Wavedash {}, &prev, &cur, &e, 1250.0) - 0.5).abs() < 1e-5);
        assert_eq!(
            pay(&Wavedash {}, &prev, &cur, &e, 900.0),
            0.0,
            "slower: nothing"
        );
        assert_eq!(pay(&Wavedash {}, &prev, &cur, &e, 2500.0), 1.0, "at most 1");
        let (prev, cur, e) = landing(0.05, true);
        assert_eq!(
            pay(&Wavedash {}, &prev, &cur, &e, 1500.0),
            0.0,
            "a wall is not the floor"
        );
    }

    #[test]
    fn nothing_for_a_whole_flip_or_a_plain_landing() {
        let (prev, cur, e) = landing(0.9, false);
        assert_eq!(
            pay(&Wavedash {}, &prev, &cur, &e, 1500.0),
            0.0,
            "a full flip that lands later"
        );
        let (mut prev, cur, mut e) = landing(0.05, false);
        prev.cars[0].has_flipped = false;
        e.prev_details[0].is_flipping = false;
        assert_eq!(pay(&Wavedash {}, &prev, &cur, &e, 1500.0), 0.0, "no dodge");
        let (prev, _, e) = landing(0.05, false);
        assert_eq!(
            pay(&Wavedash {}, &prev, &prev, &e, 1500.0),
            0.0,
            "still in the air"
        );
    }

    #[test]
    fn a_dodge_and_landing_inside_one_step_counts() {
        let (mut prev, cur, mut e) = landing(0.0, false);
        prev.cars[0].has_flipped = false;
        prev.cars[0].can_flip = true;
        e.prev_details[0].is_flipping = false;
        e.details[0].controls = Controls {
            jump: true,
            pitch: -1.0,
            ..Controls::default()
        };
        assert_eq!(pay(&Wavedash {}, &prev, &cur, &e, 1500.0), 1.0);
        e.details[0].controls.pitch = 0.0;
        assert_eq!(
            pay(&Wavedash {}, &prev, &cur, &e, 1500.0),
            0.0,
            "a plain jump is not a dodge"
        );
    }

    #[test]
    fn remembers_the_speed_before_the_dodge() {
        let (prev, cur, e) = landing(0.05, false);
        let mut mem = [0.0, 0.0];
        let mut driving = prev.clone();
        driving.cars[0].has_flipped = false;
        driving.cars[0].body.vel = Vec3A::new(0.0, 1000.0, 0.0);
        Wavedash {}.value_remembering(&driving, &driving, &quiet(1), 0, &mut mem);
        assert_eq!(mem, [1000.0, 1.0]);
        let mut dodging = prev.clone();
        dodging.cars[0].body.vel = Vec3A::new(0.0, 1400.0, 0.0);
        Wavedash {}.value_remembering(&driving, &dodging, &quiet(1), 0, &mut mem);
        assert_eq!(mem[0], 1000.0, "kept while dodging");
        let mut landed = cur.clone();
        landed.cars[0].body.vel = Vec3A::new(0.0, 1250.0, 0.0);
        let r = Wavedash {}.value_remembering(&dodging, &landed, &e, 0, &mut mem);
        assert!((r - 0.5).abs() < 1e-5);
        let fresh = Wavedash {}.value_remembering(&dodging, &landed, &e, 0, &mut [0.0, 0.0]);
        assert_eq!(fresh, 0.0, "no speed from before the dodge known yet");
    }
}
