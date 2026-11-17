//! GigaLearn's built-in rewards that NitroLearn's own list lacks, with GigaLearn's exact formulas, so a bot
//! brought over from GigaLearn keeps its rewards. From GigaLearnCPP's `RLGymCPP/Rewards/CommonRewards.h`
//! (GigaLearnCPP-Leak, commit 86e11c3, cleared for use in NitroLearn; CREDITS.md): VelocityReward,
//! VelocityBallToGoalReward with ownGoal, StrongTouchReward, TouchAccelReward, WavedashReward, BumpedPenalty,
//! DemoedPenalty. The others (air, face ball, ball and player velocities, touch, boost, bump, demo, goal) are
//! NitroLearn's built-in rewards already.
//! Changed: nothing.

use crate::custom::prelude::*;

/// Every step: the car's speed over 2300 uu/s (any direction), or minus that with `negative`
/// (GigaLearn's VelocityReward; its SpeedReward is the same with `negative = false`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnVelocity {
    #[serde(default)]
    pub negative: bool,
}

impl Reward for GigalearnVelocity {
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
        let speed = cur.cars[car].body.vel.length() / CAR_MAX_SPEED;
        if self.negative { -speed } else { speed }
    }
}

/// Every step: the ball's speed toward the back of a net over 6000 uu/s; the net the car attacks, or with
/// `own_goal` the one it defends (GigaLearn's VelocityBallToGoalReward).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnVelocityBallToGoal {
    #[serde(default)]
    pub own_goal: bool,
}

impl Reward for GigalearnVelocityBallToGoal {
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
        let team = cur.cars[car].team;
        let target = if self.own_goal {
            own_net_back(team)
        } else {
            attack_net_back(team)
        };
        (target - cur.ball.pos)
            .normalize_or_zero()
            .dot(cur.ball.vel / BALL_MAX_SPEED)
    }
}

/// On a step the car touched the ball: the change in the ball's velocity over `max_speed_kph`, at most 1,
/// and 0 below `min_speed_kph` (GigaLearn's StrongTouchReward; the built-in strong_touch ramps from 0 at
/// `min_speed_kph` instead).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnStrongTouch {
    /// Below this change in the ball's speed (km/h) the touch pays 0 (GigaLearn: 20).
    #[serde(default = "twenty")]
    pub min_speed_kph: f32,
    /// The change (km/h) that pays 1 (GigaLearn: 130).
    #[serde(default = "one_thirty")]
    pub max_speed_kph: f32,
}

fn twenty() -> f32 {
    20.0
}

fn one_thirty() -> f32 {
    130.0
}

impl Reward for GigalearnStrongTouch {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        let hit = (cur.ball.vel - prev.ball.vel).length();
        if hit < kph(self.min_speed_kph) {
            0.0
        } else {
            (hit / kph(self.max_speed_kph)).min(1.0)
        }
    }
}

/// On a step the car touched the ball: how much the touch raised the ball's speed as a share of 110 km/h
/// (both speeds capped there), 0 when it did not speed it up (GigaLearn's TouchAccelReward).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnTouchAccel {}

impl Reward for GigalearnTouchAccel {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !events.cars[car].touched_ball {
            return 0.0;
        }
        let top = kph(110.0);
        let before = (prev.ball.vel.length() / top).min(1.0);
        let after = (cur.ball.vel.length() / top).min(1.0);
        (after - before).max(0.0)
    }
}

/// 1 on the step the car lands on its wheels from a flip it was doing in the air (GigaLearn's WavedashReward).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnWavedash {}

impl Reward for GigalearnWavedash {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let landed = cur.cars[car].on_ground && !prev.cars[car].on_ground;
        f32::from(u8::from(landed && events.prev_detail(car).is_flipping))
    }
}

/// -1 on a step an opponent bumped the car (GigaLearn's BumpedPenalty).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnBumpedPenalty {}

impl Reward for GigalearnBumpedPenalty {
    fn value(
        &self,
        _prev: &GameSnapshot,
        _cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        -f32::from(u8::from(events.cars[car].bumped_by_opponent))
    }
}

/// -1 on a step an opponent demolished the car (GigaLearn's DemoedPenalty).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GigalearnDemoedPenalty {}

impl Reward for GigalearnDemoedPenalty {
    fn value(
        &self,
        _prev: &GameSnapshot,
        _cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        -f32::from(u8::from(events.cars[car].demoed_by_opponent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn two_cars() -> GameSnapshot {
        game(
            Vec3A::new(0.0, 0.0, 93.15),
            &[
                (Team::Blue, Vec3A::new(0.0, -2000.0, 17.0)),
                (Team::Orange, Vec3A::new(0.0, 2000.0, 17.0)),
            ],
        )
    }

    #[test]
    fn velocity_pays_speed_in_any_direction_or_charges_it() {
        let mut g = two_cars();
        g.cars[0].body.vel = Vec3A::new(-1150.0, 0.0, 0.0);
        let plain = GigalearnVelocity { negative: false };
        let negative = GigalearnVelocity { negative: true };
        assert!(
            (plain.value(&g, &g, &quiet(2), 0) - 0.5).abs() < 1e-6,
            "reverse counts too"
        );
        assert!((negative.value(&g, &g, &quiet(2), 0) + 0.5).abs() < 1e-6);
    }

    #[test]
    fn ball_to_goal_aims_at_either_net_from_the_cars_team() {
        let mut g = two_cars();
        g.ball.vel = Vec3A::new(0.0, 3000.0, 0.0);
        let attack = GigalearnVelocityBallToGoal { own_goal: false };
        let own = GigalearnVelocityBallToGoal { own_goal: true };
        let blue = attack.value(&g, &g, &quiet(2), 0);
        assert!(
            blue > 0.49 && blue < 0.5,
            "toward Orange's net, slightly upward: {blue}"
        );
        assert!((own.value(&g, &g, &quiet(2), 0) + blue).abs() < 1e-6);
        assert!(
            (attack.value(&g, &g, &quiet(2), 1) + blue).abs() < 1e-6,
            "Orange attacks -y"
        );
    }

    #[test]
    fn strong_touch_pays_the_change_over_the_top_from_the_minimum_on() {
        let prev = two_cars();
        let mut cur = two_cars();
        let reward = GigalearnStrongTouch {
            min_speed_kph: 20.0,
            max_speed_kph: 130.0,
        };
        cur.ball.vel = Vec3A::new(0.0, kph(65.0), 0.0);
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0, "no touch");
        assert!((reward.value(&prev, &cur, &touched(quiet(2), 0), 0) - 0.5).abs() < 1e-6);
        cur.ball.vel = Vec3A::new(0.0, kph(19.0), 0.0);
        assert_eq!(
            reward.value(&prev, &cur, &touched(quiet(2), 0), 0),
            0.0,
            "below the minimum"
        );
        cur.ball.vel = Vec3A::new(0.0, kph(300.0), 0.0);
        assert_eq!(
            reward.value(&prev, &cur, &touched(quiet(2), 0), 0),
            1.0,
            "capped"
        );
        let defaults: GigalearnStrongTouch = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(
            (defaults.min_speed_kph, defaults.max_speed_kph),
            (20.0, 130.0)
        );
    }

    #[test]
    fn touch_accel_pays_the_capped_gain_on_a_touch_only() {
        let prev = two_cars();
        let mut cur = two_cars();
        cur.ball.vel = Vec3A::new(0.0, kph(55.0), 0.0);
        let reward = GigalearnTouchAccel {};
        assert_eq!(reward.value(&prev, &cur, &quiet(2), 0), 0.0, "no touch");
        let gained = reward.value(&prev, &cur, &touched(quiet(2), 0), 0);
        assert!((gained - 0.5).abs() < 1e-6);
        cur.ball.vel = Vec3A::new(0.0, 5000.0, 0.0);
        assert!(
            (reward.value(&prev, &cur, &touched(quiet(2), 0), 0) - 1.0).abs() < 1e-6,
            "capped"
        );
        assert_eq!(
            reward.value(&cur, &prev, &touched(quiet(2), 0), 0),
            0.0,
            "slowed down"
        );
    }

    #[test]
    fn wavedash_pays_landing_from_a_flip() {
        let mut prev = two_cars();
        prev.cars[0].on_ground = false;
        let cur = two_cars();
        let mut events = quiet(2);
        let reward = GigalearnWavedash {};
        assert_eq!(
            reward.value(&prev, &cur, &events, 0),
            0.0,
            "landing without a flip"
        );
        events.prev_details[0].is_flipping = true;
        assert_eq!(reward.value(&prev, &cur, &events, 0), 1.0);
        assert_eq!(
            reward.value(&cur, &cur, &events, 0),
            0.0,
            "already on the ground"
        );
    }

    #[test]
    fn penalties_charge_the_car_that_was_hit() {
        let g = two_cars();
        let mut events = quiet(2);
        events.cars[1].bumped_by_opponent = true;
        events.cars[1].demoed_by_opponent = true;
        assert_eq!(GigalearnBumpedPenalty {}.value(&g, &g, &events, 1), -1.0);
        assert_eq!(GigalearnDemoedPenalty {}.value(&g, &g, &events, 1), -1.0);
        assert_eq!(GigalearnBumpedPenalty {}.value(&g, &g, &events, 0), 0.0);
    }
}
