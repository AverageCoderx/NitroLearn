//! Dribble: the ball carried just above the car, more on a touch.

use crate::custom::prelude::*;

/// The ball radius this reward measures from (rlgym-sim's 92.75 uu, not the game's 91.25).
const ITS_BALL_RADIUS: f32 = 92.75;
/// How far the car may be from the ball across, and along, the field (uu).
const NEAR: f32 = 150.0;

/// Every step the ball is 112.75 to 292.75 uu up and within 150 uu of the car across and along the field:
/// 0.2, plus 0.8 when the car touches it then (0 to 1). "Within" compares distances from the centre
/// lines, so a car mirrored across a centre line from the ball also passes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DribbleCarry {
    // No settings.
}

impl Reward for DribbleCarry {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        let (me, ball) = (cur.cars[car].body.pos, cur.ball.pos);
        // The band's lower edge also keeps the ball above 100 uu.
        let in_band = ball.z > ITS_BALL_RADIUS + 20.0 && ball.z < ITS_BALL_RADIUS + 200.0;
        let near =
            (me.x.abs() - ball.x.abs()).abs() < NEAR && (me.y.abs() - ball.y.abs()).abs() < NEAR;
        if !(in_band && near) {
            return 0.0;
        }
        if events.cars[car].touched_ball {
            1.0
        } else {
            0.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet, touched};

    fn paid(ball: Vec3A, car: Vec3A, touch: bool) -> f32 {
        let cur = game(ball, &[(Team::Blue, car)]);
        let events = if touch {
            touched(quiet(1), 0)
        } else {
            quiet(1)
        };
        DribbleCarry {}.value(&cur, &cur, &events, 0)
    }

    const CAR: Vec3A = Vec3A::new(1000.0, -2000.0, 17.0);

    #[test]
    fn pays_a_fifth_with_the_ball_on_the_car_and_all_on_a_touch() {
        let ball = Vec3A::new(1050.0, -2050.0, 200.0);
        assert!((paid(ball, CAR, false) - 0.2).abs() < 1e-6);
        assert_eq!(paid(ball, CAR, true), 1.0);
    }

    #[test]
    fn nothing_with_the_ball_on_the_floor_too_high_or_away() {
        assert_eq!(paid(Vec3A::new(1000.0, -2000.0, 93.15), CAR, true), 0.0);
        assert_eq!(paid(Vec3A::new(1000.0, -2000.0, 300.0), CAR, true), 0.0);
        assert_eq!(paid(Vec3A::new(1200.0, -2000.0, 200.0), CAR, true), 0.0);
    }

    #[test]
    fn a_car_mirrored_across_the_centre_line_passes() {
        let mirrored = Vec3A::new(-1000.0, -2000.0, 17.0);
        assert!((paid(Vec3A::new(1000.0, -2000.0, 200.0), mirrored, false) - 0.2).abs() < 1e-6);
    }
}
