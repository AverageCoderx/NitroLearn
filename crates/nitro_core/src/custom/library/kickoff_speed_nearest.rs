//! Kickoffs: speed at the ball once the boost is used up (kickoff_speed_no_boost), and being the team
//! nearest the still ball (kickoff_nearest_team).
//! Based on rewards from apollo-bot by redd-rl; kickoff_nearest_team also builds on one by aether6837.

use crate::custom::prelude::*;

/// The source's kickoff test: the ball standing still (anywhere).
fn ball_still(game: &GameSnapshot) -> bool {
    game.ball.vel == Vec3A::ZERO
}

/// Every step the ball stands still (a kickoff) while the car's boost is below 2 (of 100): its speed
/// towards the ball over 2300 uu/s, 0 to 1, and 0 when moving away. 0 otherwise. (`kickoff_speed_no_boost_v2` pays
/// differently.)
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffSpeedNoBoost {}

impl Reward for KickoffSpeedNoBoost {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        let me = &cur.cars[car];
        // The source's `boostFraction < 0.02`, on the 0-100 scale.
        if !ball_still(cur) || me.boost >= 2.0 {
            return 0.0;
        }
        let towards = (cur.ball.pos - me.body.pos).normalize_or_zero();
        (me.body.vel.dot(towards) / CAR_MAX_SPEED).max(0.0)
    }
}

/// Every step the ball stands still (a kickoff): 1 when a car of this car's team (itself included) is nearer
/// the ball than every opponent, else −1 (also with no opponent). 0 once the ball moves.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffNearestTeam {}

impl Reward for KickoffNearestTeam {
    fn every_step(&self) -> bool {
        true
    }

    fn value(&self, _prev: &GameSnapshot, cur: &GameSnapshot, _: &StepEvents, car: usize) -> f32 {
        if !ball_still(cur) {
            return 0.0;
        }
        let nearest = |team: Team| {
            cur.cars
                .iter()
                .filter(|other| other.team == team)
                .map(|other| other.body.pos.distance(cur.ball.pos))
                .min_by(f32::total_cmp)
        };
        let team = cur.cars[car].team;
        match (nearest(team), nearest(team.opposite())) {
            (Some(ours), Some(theirs)) if ours < theirs => 1.0,
            _ => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    /// A 2v2 kickoff: Blue at y −2000 and −4000, Orange at 3000 and 4000, all driving at the ball.
    fn kickoff() -> GameSnapshot {
        let mut g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 93.15)),
                (Team::Blue, Vec3A::new(0.0, -4000.0, 93.15)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 93.15)),
                (Team::Orange, Vec3A::new(0.0, 4000.0, 93.15)),
            ],
        );
        for car in &mut g.cars {
            let towards = attack_sign(car.team);
            car.body.vel = Vec3A::new(0.0, 1150.0 * towards, 0.0);
            car.boost = 0.0;
        }
        g
    }

    #[test]
    fn speedflip_kickoff_pays_speed_at_the_ball_once_the_boost_is_gone() {
        let reward: KickoffSpeedNoBoost = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = kickoff();
        assert!((reward.value(&g, &g, &quiet(4), 0) - 0.5).abs() < 1e-6);
        assert!(
            (reward.value(&g, &g, &quiet(4), 2) - 0.5).abs() < 1e-6,
            "Orange alike"
        );
        let mut boosted = g.clone();
        boosted.cars[0].boost = 2.0;
        assert_eq!(
            reward.value(&boosted, &boosted, &quiet(4), 0),
            0.0,
            "boost left"
        );
        let mut away = g.clone();
        away.cars[0].body.vel.y = -1000.0;
        assert_eq!(reward.value(&away, &away, &quiet(4), 0), 0.0, "moving away");
        let mut moving = g.clone();
        moving.ball.vel.x = 1.0;
        assert_eq!(
            reward.value(&moving, &moving, &quiet(4), 0),
            0.0,
            "ball moving"
        );
        let mut at_ball = g.clone();
        at_ball.cars[0].body.pos = at_ball.ball.pos;
        assert_eq!(
            reward.value(&at_ball, &at_ball, &quiet(4), 0),
            0.0,
            "no NaN"
        );
    }

    #[test]
    fn kickoff_proximity_pays_the_whole_nearer_team() {
        let reward: KickoffNearestTeam = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = kickoff();
        let paid: Vec<f32> = (0..4)
            .map(|car| reward.value(&g, &g, &quiet(4), car))
            .collect();
        assert_eq!(paid, [1.0, 1.0, -1.0, -1.0], "Blue's front car is nearest");
        let mut mirrored = g.clone();
        for car in &mut mirrored.cars {
            car.team = car.team.opposite();
            car.body.pos = as_blue(Team::Orange, car.body.pos);
        }
        let paid: Vec<f32> = (0..4)
            .map(|car| reward.value(&mirrored, &mirrored, &quiet(4), car))
            .collect();
        assert_eq!(paid, [1.0, 1.0, -1.0, -1.0], "mirrored: Orange alike");
        let mut beaten = g.clone();
        beaten.cars[0].body.pos.y = -3500.0;
        let paid: Vec<f32> = (0..4)
            .map(|car| reward.value(&beaten, &beaten, &quiet(4), car))
            .collect();
        assert_eq!(paid, [-1.0, -1.0, 1.0, 1.0], "Orange nearest");
        let mut moving = g.clone();
        moving.ball.vel.y = 10.0;
        assert_eq!(
            reward.value(&moving, &moving, &quiet(4), 0),
            0.0,
            "ball moving"
        );
        let alone = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[(Team::Blue, Vec3A::new(0.0, -2000.0, 17.0))],
        );
        assert_eq!(
            reward.value(&alone, &alone, &quiet(1), 0),
            -1.0,
            "no opponent"
        );
    }
}
