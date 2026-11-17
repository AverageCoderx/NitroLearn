//! Kickoffs: being nearer the still ball than any opponent, from community GigaLearn bots' rewards
//! (KickoffProximityReward in their CommonRewards.h), whose author let it be used (CREDITS.md).
//! Changed: nothing.

use crate::custom::prelude::*;

/// +1 nearer the ball than every opponent, −1 otherwise (ties too), 0 with no opponent; only while the ball
/// moves slower than 1 uu/s. `skip_near` gives 0 within 200 uu.
fn one_v_one(cur: &GameSnapshot, car: usize, skip_near: bool) -> f32 {
    if cur.ball.vel.length() >= 1.0 {
        return 0.0;
    }
    let mine = cur.cars[car].body.pos.distance(cur.ball.pos);
    let Some(nearest) = opponents(cur, car)
        .map(|other| cur.cars[other].body.pos.distance(cur.ball.pos))
        .min_by(f32::total_cmp)
    else {
        return 0.0;
    };
    if skip_near && mine <= 200.0 {
        0.0
    } else if mine < nearest {
        1.0
    } else {
        -1.0
    }
}

/// Every step while the ball is still (slower than 1 uu/s): +1 when this car is nearer the ball than every
/// opponent, −1 when not (ties too). 0 with the ball moving or no opponent. Meant for 1v1 (there it is
/// zero-sum); in team modes teammates get the same.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KickoffProximity {
    /// true: 0 while this car is within 200 uu of the ball (a variant some bots use); false: the original.
    #[serde(default)]
    pub zero_within_200: bool,
}

impl Reward for KickoffProximity {
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
        one_v_one(cur, car, self.zero_within_200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn mirrored(g: &GameSnapshot) -> GameSnapshot {
        let mut m = g.clone();
        m.ball.pos = as_blue(Team::Orange, m.ball.pos);
        for car in &mut m.cars {
            car.team = car.team.opposite();
            car.body.pos = as_blue(Team::Orange, car.body.pos);
            car.body.vel = as_blue(Team::Orange, car.body.vel);
        }
        m
    }

    #[test]
    fn nearer_than_the_opponent_on_a_still_ball() {
        let r: KickoffProximity = serde_json::from_value(serde_json::json!({})).unwrap();
        let g = game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -150.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 3000.0, 17.0)),
            ],
        );
        let events = quiet(2);
        assert_eq!(r.value(&g, &g, &events, 0), 1.0);
        assert_eq!(r.value(&g, &g, &events, 1), -1.0);
        let m = mirrored(&g);
        assert_eq!(m.cars[0].team, Team::Orange);
        assert_eq!(r.value(&m, &m, &events, 0), 1.0, "Orange the same as Blue");
        let mut moving = g.clone();
        moving.ball.vel = Vec3A::new(0.0, 5.0, 0.0);
        assert_eq!(r.value(&moving, &moving, &events, 0), 0.0);
        let alone = game(Vec3A::ZERO, &[(Team::Blue, Vec3A::new(0.0, -150.0, 17.0))]);
        assert_eq!(r.value(&alone, &alone, &quiet(1), 0), 0.0);
        let guarded = KickoffProximity {
            zero_within_200: true,
        };
        assert_eq!(guarded.value(&g, &g, &events, 0), 0.0, "within 200 uu");
        assert_eq!(guarded.value(&g, &g, &events, 1), -1.0);
    }
}
