//! Dribbling with the ball held just above the car (dribble_cradle, dribble_upfield) and the flick from
//! it (dribble_flick).
//! Based on rewards in rishieissocool's reward collection.

use crate::custom::library::aerial_kit::RLGYM_BALL_RADIUS;
use crate::custom::prelude::*;

/// The cradle test (dribble_cradle without its opponent check): the car's centre below the
/// ball's, the ball 112.75 to 292.75 uu up and within 170 uu of the car (centre to centre), away from the walls.
fn cradled(cur: &GameSnapshot, car: usize) -> bool {
    let (ball, pos) = (cur.ball.pos, cur.cars[car].body.pos);
    pos.z < ball.z
        && RLGYM_BALL_RADIUS + 20.0 < ball.z
        && ball.z < RLGYM_BALL_RADIUS + 200.0
        && pos.distance(ball) <= 170.0
        && ball.x.abs() < 3946.0
        && ball.y.abs() < 4970.0
}

/// A car of the other team nearer the ball than `reach` (uu, centre to centre).
fn challenged(cur: &GameSnapshot, car: usize, reach: f32) -> bool {
    opponents(cur, car).any(|other| cur.cars[other].body.pos.distance(cur.ball.pos) < reach)
}

/// Every step the ball sits just above the car (the cradle test: ball 112.75 to 292.75 uu up, within 170 uu,
/// away from the walls) with no opponent within `minimum_barrier` of the ball: 1. Else 0.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleCradle {
    /// An opponent this near the ball (uu) stops the pay; 0 turns the check off.
    #[serde(default = "two_hundred")]
    pub minimum_barrier: f32,
}

fn two_hundred() -> f32 {
    200.0
}

impl Reward for DribbleCradle {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        let blocked = self.minimum_barrier > 0.0 && challenged(cur, car, self.minimum_barrier);
        f32::from(u8::from(cradled(cur, car) && !blocked))
    }
}

/// DribbleCradle paid by how far up the field the car is: √((y + 5000) / 100) / 10 with y towards the opponents'
/// net, 0 behind y = −5000, 0.71 at the centre line, 1 at y = 5000.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleUpfield {
    /// An opponent this near the ball (uu) stops the pay; 0 turns the check off.
    #[serde(default = "two_hundred_broken")]
    pub minimum_barrier: f32,
}

fn two_hundred_broken() -> f32 {
    200.0
}

impl Reward for DribbleUpfield {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        let blocked = self.minimum_barrier > 0.0 && challenged(cur, car, self.minimum_barrier);
        if !cradled(cur, car) || blocked {
            return 0.0;
        }
        let me = &cur.cars[car];
        let y = as_blue(me.team, me.body.pos).y;
        ((y + 5000.0) / 100.0).max(0.0).sqrt() / 10.0
    }
}

/// Every step the ball sits on the car (the cradle test): 0.5, and with the ball 112.75 to 172.75 uu up
/// moving within `max_vel_diff` of the car: +1 unchallenged; challenged (an opponent within `minimum_barrier`
/// of the ball) −0.5 on the ground, +2 in the air with a flip left, +5 once it is used (the flick). −0.5 to 5.5.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleFlick {
    /// An opponent this near the ball (uu) makes the dribble challenged.
    #[serde(default = "four_hundred")]
    pub minimum_barrier: f32,
    /// The ball's speed may differ from the car's by this much for a steady carry (uu/s).
    #[serde(default = "four_hundred_speed")]
    pub max_vel_diff: f32,
    /// Pay the 0.5 for holding the ball; false keeps only the carry and flick amounts.
    #[serde(default = "yes")]
    pub training: bool,
}

fn four_hundred() -> f32 {
    400.0
}

fn four_hundred_speed() -> f32 {
    400.0
}

fn yes() -> bool {
    true
}

impl Reward for DribbleFlick {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !cradled(cur, car) {
            return 0.0;
        }
        let me = &cur.cars[car];
        let reward = if self.training { 0.5 } else { 0.0 };
        let ball = &cur.ball;
        let stable = RLGYM_BALL_RADIUS + 20.0 < ball.pos.z
            && ball.pos.z < RLGYM_BALL_RADIUS + 80.0
            && (me.body.vel - ball.vel).length() <= self.max_vel_diff;
        if !stable {
            reward
        } else if !challenged(cur, car, self.minimum_barrier) {
            reward + 1.0
        } else if me.on_ground {
            reward - 0.5
        } else if me.can_flip {
            reward + 2.0
        } else {
            reward + 5.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// A car of `team` on the floor at (300, y) in its own view, carrying the ball 140 uu above it (and a
    /// little to the side), and an opponent on the floor `opponent_gap` uu up the field from the ball.
    fn carrying(team: Team, y: f32, opponent_gap: f32) -> GameSnapshot {
        let s = attack_sign(team);
        let car = Vec3A::new(300.0 * s, y * s, 17.0);
        let ball = car + Vec3A::new(10.0 * s, 20.0 * s, 140.0);
        let other = Vec3A::new(ball.x, ball.y + opponent_gap * s, 17.0);
        game(ball, &[(team, car), (team.opposite(), other)])
    }

    #[test]
    fn dribble_tg_pays_the_cradle_unless_an_opponent_is_close() {
        let reward: DribbleCradle = serde_json::from_value(serde_json::json!({})).unwrap();
        for team in [Team::Blue, Team::Orange] {
            let g = carrying(team, 1000.0, 3000.0);
            assert_eq!(reward.value(&g, &g, &quiet(2), 0), 1.0, "{team:?}");
            assert_eq!(reward.value(&g, &g, &quiet(2), 1), 0.0, "the far car");
            let close = carrying(team, 1000.0, 100.0);
            assert_eq!(reward.value(&close, &close, &quiet(2), 0), 0.0, "{team:?}");
            let off = DribbleCradle {
                minimum_barrier: 0.0,
            };
            assert_eq!(off.value(&close, &close, &quiet(2), 0), 1.0, "check off");
        }
        let mut high = carrying(Team::Blue, 1000.0, 3000.0);
        high.ball.pos.z = 300.0;
        assert_eq!(reward.value(&high, &high, &quiet(2), 0), 0.0, "too high");
        let mut beside = carrying(Team::Blue, 1000.0, 3000.0);
        beside.ball.pos.x += 200.0;
        assert_eq!(reward.value(&beside, &beside, &quiet(2), 0), 0.0, "too far");
        let mut by_wall = carrying(Team::Blue, 1000.0, 3000.0);
        by_wall.ball.pos.x = 3950.0;
        by_wall.cars[0].body.pos.x = 3940.0;
        assert_eq!(
            reward.value(&by_wall, &by_wall, &quiet(2), 0),
            0.0,
            "by a wall"
        );
        let mut above = carrying(Team::Blue, 1000.0, 3000.0);
        above.cars[0].body.pos.z = 200.0;
        assert_eq!(
            reward.value(&above, &above, &quiet(2), 0),
            0.0,
            "car above the ball"
        );
    }

    #[test]
    fn dribble_broken_pays_more_up_the_field_for_either_team() {
        let reward: DribbleUpfield = serde_json::from_value(serde_json::json!({})).unwrap();
        for team in [Team::Blue, Team::Orange] {
            let middle = carrying(team, 0.0, 3000.0);
            let at_middle = reward.value(&middle, &middle, &quiet(2), 0);
            assert!((at_middle - 50f32.sqrt() / 10.0).abs() < 1e-5, "{team:?}");
            let deep = carrying(team, 4500.0, -3000.0);
            let at_deep = reward.value(&deep, &deep, &quiet(2), 0);
            assert!((at_deep - 95f32.sqrt() / 10.0).abs() < 1e-5, "{team:?}");
            let own_end = carrying(team, -4900.0, 3000.0);
            let at_own_end = reward.value(&own_end, &own_end, &quiet(2), 0);
            assert!((at_own_end - 0.1).abs() < 1e-5, "{team:?}");
            let close = carrying(team, 0.0, 100.0);
            assert_eq!(
                reward.value(&close, &close, &quiet(2), 0),
                0.0,
                "an opponent at the ball"
            );
        }
        let mut behind = carrying(Team::Blue, -4900.0, 3000.0);
        behind.cars[0].body.pos.y = -5010.0;
        behind.ball.pos.y = -4960.0;
        let v = reward.value(&behind, &behind, &quiet(2), 0);
        assert!(v == 0.0, "behind y = −5000: 0, not NaN");
    }

    #[test]
    fn dribble_flick_pays_the_carry_and_most_for_the_flick() {
        let reward: DribbleFlick = serde_json::from_value(serde_json::json!({})).unwrap();
        for team in [Team::Blue, Team::Orange] {
            let free = carrying(team, 1000.0, 3000.0);
            assert_eq!(reward.value(&free, &free, &quiet(2), 0), 1.5, "{team:?}");
            let mut challenged = carrying(team, 1000.0, 300.0);
            assert_eq!(
                reward.value(&challenged, &challenged, &quiet(2), 0),
                0.0,
                "challenged on the ground"
            );
            challenged.cars[0].on_ground = false;
            assert_eq!(reward.value(&challenged, &challenged, &quiet(2), 0), 2.5);
            challenged.cars[0].can_flip = false;
            assert_eq!(reward.value(&challenged, &challenged, &quiet(2), 0), 5.5);
            let mut unsteady = challenged.clone();
            unsteady.ball.vel = Vec3A::new(0.0, 0.0, 500.0);
            assert_eq!(reward.value(&unsteady, &unsteady, &quiet(2), 0), 0.5);
        }
        let mut high = carrying(Team::Blue, 1000.0, 3000.0);
        high.ball.pos.z = 200.0;
        high.cars[0].body.pos.z = 60.0;
        assert_eq!(
            reward.value(&high, &high, &quiet(2), 0),
            0.5,
            "held, not a steady carry"
        );
        high.ball.pos.z = 300.0;
        assert_eq!(reward.value(&high, &high, &quiet(2), 0), 0.0, "not held");
        let no_base = DribbleFlick {
            minimum_barrier: 400.0,
            max_vel_diff: 400.0,
            training: false,
        };
        let free = carrying(Team::Blue, 1000.0, 3000.0);
        assert_eq!(no_base.value(&free, &free, &quiet(2), 0), 1.0);
    }
}
