//! Ball speed towards the goal, paid only while the ball flies almost straight at it.

use crate::custom::prelude::*;

/// How far (as the distance between the two unit directions) the ball's flight may be off the line to the
/// net before nothing is paid: about 41 degrees.
const MAX_UNALIGNMENT: f32 = 0.7;

/// Every step: the ball's speed towards the back of the opponents' net (1 at 6000 uu/s), times how straight
/// at it the ball flies (all of it exactly at the net, none from about 41 degrees off). 0 to about 1; never
/// negative.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VelocityBallToGoalAligned {
    /// true = towards the car's own net instead.
    #[serde(default)]
    pub own_goal: bool,
}

impl Reward for VelocityBallToGoalAligned {
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
        let to_goal = (target - cur.ball.pos).normalize_or_zero();
        let heading = cur.ball.vel.normalize_or_zero();
        let unalignment = (to_goal - heading).length();
        let aligned = 1.0 - unalignment.min(MAX_UNALIGNMENT) / MAX_UNALIGNMENT;
        (aligned * to_goal.dot(cur.ball.vel / BALL_MAX_SPEED)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    fn paid(own_goal: bool, team: Team, ball_vel: Vec3A) -> f32 {
        let mut cur = game(
            Vec3A::new(0.0, 0.0, GOAL_HEIGHT / 2.0),
            &[(team, Vec3A::new(0.0, 0.0, 17.0))],
        );
        cur.ball.vel = ball_vel;
        VelocityBallToGoalAligned { own_goal }.value(&cur, &cur, &quiet(1), 0)
    }

    #[test]
    fn pays_the_speed_towards_the_net_when_straight_at_it_for_either_team() {
        let blue = paid(false, Team::Blue, Vec3A::new(0.0, 3000.0, 0.0));
        assert!((blue - 0.5).abs() < 1e-6);
        let orange = paid(false, Team::Orange, Vec3A::new(0.0, -3000.0, 0.0));
        assert!((orange - blue).abs() < 1e-6);
    }

    #[test]
    fn less_when_off_line_and_nothing_beyond_the_limit_away_or_at_rest() {
        let angle = 30f32.to_radians();
        let off = paid(
            false,
            Team::Blue,
            Vec3A::new(angle.sin(), angle.cos(), 0.0) * 3000.0,
        );
        let expected = (1.0 - 2.0 * 15f32.to_radians().sin() / 0.7) * angle.cos() * 0.5;
        assert!((off - expected).abs() < 1e-5, "{off} vs {expected}");
        let wide = 45f32.to_radians();
        let beyond = Vec3A::new(wide.sin(), wide.cos(), 0.0) * 3000.0;
        assert_eq!(paid(false, Team::Blue, beyond), 0.0);
        assert_eq!(paid(false, Team::Blue, Vec3A::new(0.0, -3000.0, 0.0)), 0.0);
        assert_eq!(paid(false, Team::Blue, Vec3A::ZERO), 0.0);
    }

    #[test]
    fn own_goal_turns_it_round_and_defaults_to_off() {
        let towards_own = paid(true, Team::Orange, Vec3A::new(0.0, 3000.0, 0.0));
        assert!((towards_own - 0.5).abs() < 1e-6);
        let default: VelocityBallToGoalAligned =
            serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!default.own_goal);
    }
}
