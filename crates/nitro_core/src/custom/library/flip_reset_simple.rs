//! Flip resets: getting one (flip_reset_simple), and the approach to one (flip_reset_approach): upside
//! down under a high ball, nose to the opponents' net.
//! Based on Opti's flip reset rewards.

use crate::custom::library::aerial_kit::*;
use crate::custom::prelude::*;

/// The source's fixed "time until the kickoff is finished" (steps) and its quick-reset norm (steps): the
/// first reset of an episode pays `quick_flip_reset_w` × 100 / 1000.
const KICKOFF_TIMER: f32 = 1000.0;
const QUICK_FLIP_RESET_NORM_STEPS: f32 = 100.0;

kept! {
    /// What FlipResetSimple keeps for each car.
    struct ResetMemory {
        /// The car has had a reset this episode.
        got_reset: bool,
        /// Counted resets since the car was last on the ground.
        in_a_row: i32,
        /// A reset since the car was last on the ground (the source's timer, which never advanced, set).
        since_ground: bool,
    }
}

/// On a flip reset (the car's jump comes back in the air with the ball above 200 uu, within 110 uu and under
/// its wheels): `flip_reset_w` (10) for the first of each flight, every one with `prevent_chain_reset` off,
/// + 0.5 for the car's first of the episode, + up to `inc_flip_reset_w` (2) from the second counted in a row.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlipResetSimple {
    /// What a counted reset pays.
    #[serde(default = "ten")]
    pub flip_reset_w: f32,
    /// The car's first reset of the episode adds a tenth of this.
    #[serde(default = "five")]
    pub quick_flip_reset_w: f32,
    /// The n-th counted reset in a row (n ≥ 2) adds this × min(1.4ⁿ, 6) / 6.
    #[serde(default = "two")]
    pub inc_flip_reset_w: f32,
    /// Count only the first reset until the car is on the ground again; false counts every reset and pays
    /// the bonus for resets in a row.
    #[serde(default = "yes")]
    pub prevent_chain_reset: bool,
}

fn ten() -> f32 {
    10.0
}

fn five() -> f32 {
    5.0
}

fn two() -> f32 {
    2.0
}

fn yes() -> bool {
    true
}

impl Reward for FlipResetSimple {
    fn memory(&self) -> usize {
        ResetMemory::SIZE
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let mut kept = ResetMemory::load(memory);
        let me = &cur.cars[car];
        let (ball, pos) = (cur.ball.pos, me.body.pos);
        // rlgym-sim's "not has_jump before, has_jump now": the first jump came back during the step.
        let jump_back = prev.cars[car].has_jumped && !me.has_jumped;
        let reset = jump_back
            && ball.z > 200.0
            && ball.distance(pos) < 110.0
            && cosine(ball - pos, -me.body.up()) > 0.9;
        let mut reward = 0.0;
        if reset {
            if !kept.got_reset {
                reward += self.quick_flip_reset_w * QUICK_FLIP_RESET_NORM_STEPS / KICKOFF_TIMER;
            }
            kept.got_reset = true;
            if !(self.prevent_chain_reset && kept.since_ground) {
                reward += self.flip_reset_w;
                kept.in_a_row += 1;
                if kept.in_a_row > 1 {
                    reward += self.inc_flip_reset_w * 1.4f32.powi(kept.in_a_row).min(6.0) / 6.0;
                }
            }
            kept.since_ground = true;
        } else if me.on_ground {
            kept.in_a_row = 0;
            kept.since_ground = false;
        }
        kept.store(memory);
        reward
    }
}

/// Every step: `flip_reset_help_w` × 2·cos(car→ball, car's underside) × min(1, |ball x| / 1300) ×
/// min(1, ball height / 1700) × cos(car→opponents' net, nose) × min(1, 40·(how far the wheels face the
/// ceiling) / (distance to the ball, height doubled, + 1)). Up to about 2; below 0 when one factor is.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlipResetApproach {
    /// The weight of the whole product (0 turns it off).
    #[serde(default = "one")]
    pub flip_reset_help_w: f32,
}

fn one() -> f32 {
    1.0
}

impl Reward for FlipResetApproach {
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
        if self.flip_reset_help_w == 0.0 {
            return 0.0;
        }
        let me = &cur.cars[car];
        let (ball, pos, down) = (cur.ball.pos, me.body.pos, -me.body.up());
        let upness = cosine(Vec3A::new(0.0, 0.0, CEILING_Z - pos.z), down);
        let from_wall = (ball.x.abs() / 1300.0).min(1.0);
        let height = (ball.z / 1700.0).min(1.0);
        let bottom_ball = 2.0 * cosine(ball - pos, down);
        let align = cosine(attack_net_back(me.team) - pos, me.body.forward());
        let mut gap = ball - pos;
        gap.z *= 2.0;
        // The source's np.clip(-1, 1, x) (its arguments out of order) is min(1, x): no lower bound.
        let near = (40.0 * upness / (gap.length() + 1.0)).min(1.0);
        self.flip_reset_help_w * bottom_ball * from_wall * height * align * near
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};
    use std::f32::consts::{FRAC_PI_2, PI};

    /// A car of `team` upside down 1000 uu up, nose towards the opponents' net, the ball just above its
    /// wheels (centre 100 uu away), and an opponent far off; the step before, its jump was used.
    fn reset_step(team: Team) -> (GameSnapshot, GameSnapshot) {
        let s = attack_sign(team);
        let pos = Vec3A::new(300.0 * s, 1000.0 * s, 1000.0);
        let mut cur = game(
            pos + Vec3A::new(0.0, 0.0, 100.0),
            &[(team, pos), (team.opposite(), Vec3A::new(0.0, 0.0, 17.0))],
        );
        cur.cars[0].body.rot = Mat3A::from_rotation_y(PI) * Mat3A::from_rotation_z(FRAC_PI_2 * s);
        cur.cars[0].on_ground = false;
        let mut prev = cur.clone();
        prev.cars[0].has_jumped = true;
        (prev, cur)
    }

    #[test]
    fn pays_a_reset_once_a_flight_and_more_for_the_first() {
        let reward: FlipResetSimple = serde_json::from_value(serde_json::json!({})).unwrap();
        for team in [Team::Blue, Team::Orange] {
            let (prev, cur) = reset_step(team);
            let mut memory = [0.0f32; 3];
            let mut step = |prev: &GameSnapshot, cur: &GameSnapshot| {
                reward.value_remembering(prev, cur, &quiet(2), 0, &mut memory)
            };
            assert!(
                (step(&prev, &cur) - 10.5).abs() < 1e-5,
                "{team:?}: 10 + 0.5"
            );
            assert_eq!(step(&cur, &cur), 0.0, "no jump came back");
            assert_eq!(step(&prev, &cur), 0.0, "a second reset in the same flight");
            let mut landed = cur.clone();
            landed.cars[0].on_ground = true;
            assert_eq!(step(&landed, &landed), 0.0);
            assert_eq!(
                step(&prev, &cur),
                10.0,
                "next flight; not the episode's first"
            );
            let mut fresh = [0.0f32; 3];
            let again = reward.value_remembering(&prev, &cur, &quiet(2), 0, &mut fresh);
            assert!((again - 10.5).abs() < 1e-5, "fresh memory starts again");
        }
    }

    #[test]
    fn chains_resets_when_allowed_and_needs_the_ball_under_the_wheels() {
        let chained = FlipResetSimple {
            flip_reset_w: 10.0,
            quick_flip_reset_w: 5.0,
            inc_flip_reset_w: 2.0,
            prevent_chain_reset: false,
        };
        let (prev, cur) = reset_step(Team::Blue);
        let mut memory = [0.0f32; 3];
        let first = chained.value_remembering(&prev, &cur, &quiet(2), 0, &mut memory);
        assert!((first - 10.5).abs() < 1e-5);
        let second = chained.value_remembering(&prev, &cur, &quiet(2), 0, &mut memory);
        assert!(
            (second - (10.0 + 2.0 * 1.96 / 6.0)).abs() < 1e-4,
            "{second}"
        );
        let reward: FlipResetSimple = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut upright = cur.clone();
        upright.cars[0].body.rot = Mat3A::IDENTITY;
        let mut fresh = [0.0f32; 3];
        assert_eq!(
            reward.value_remembering(&prev, &upright, &quiet(2), 0, &mut fresh),
            0.0,
            "the ball on the roof"
        );
        let mut far = cur.clone();
        far.ball.pos.z += 50.0;
        assert_eq!(
            reward.value_remembering(&prev, &far, &quiet(2), 0, &mut fresh),
            0.0
        );
        let mut low = cur.clone();
        low.ball.pos.z = 190.0;
        low.cars[0].body.pos.z = 90.0;
        assert_eq!(
            reward.value_remembering(&prev, &low, &quiet(2), 0, &mut fresh),
            0.0,
            "the ball too low"
        );
    }

    #[test]
    fn helper_pays_upside_down_under_a_high_ball_off_the_centre_line() {
        let reward: FlipResetApproach = serde_json::from_value(serde_json::json!({})).unwrap();
        for team in [Team::Blue, Team::Orange] {
            let s = attack_sign(team);
            let pos = Vec3A::new(1300.0 * s, 0.0, 1600.0);
            let mut g = game(pos + Vec3A::new(0.0, 0.0, 100.0), &[(team, pos)]);
            g.cars[0].body.rot = Mat3A::from_rotation_y(PI) * Mat3A::from_rotation_z(FRAC_PI_2 * s);
            g.cars[0].on_ground = false;
            let to_net = attack_net_back(team) - pos;
            let align = to_net.y.abs() / to_net.length();
            let expected = 2.0 * align * (40.0 / 201.0);
            let got = reward.value(&g, &g, &quiet(1), 0);
            assert!(
                (got - expected).abs() < 1e-5,
                "{team:?}: {got} vs {expected}"
            );
            let mut centre = g.clone();
            centre.ball.pos.x = 0.0;
            assert_eq!(reward.value(&centre, &centre, &quiet(1), 0), 0.0);
            let off = FlipResetApproach {
                flip_reset_help_w: 0.0,
            };
            assert_eq!(off.value(&g, &g, &quiet(1), 0), 0.0);
        }
    }
}
