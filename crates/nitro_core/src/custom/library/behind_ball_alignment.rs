//! Lined up behind the ball: the direction to the ball against the direction to the opponents' net.
//! Adapted from Necto (github.com/Rolv-Arild/Necto), licensed CC BY-NC-SA 4.0
//! (https://creativecommons.org/licenses/by-nc-sa/4.0/), and changed from the original. This file is shared
//! under the same licence.

use crate::custom::prelude::*;

/// Every step: `align_w` × the cosine between car→ball and car→the back of the opponents' net, −0.75 to
/// 0.75 with the default (most with the ball straight between the car and that net).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehindBallAlignment {
    /// The reward with the car, the ball and the net in one line.
    #[serde(default = "three_quarters")]
    pub align_w: f32,
}

fn three_quarters() -> f32 {
    0.75
}

impl Reward for BehindBallAlignment {
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
        let me = &cur.cars[car];
        let pos = me.body.pos;
        self.align_w * cosine(cur.ball.pos - pos, attack_net_back(me.team) - pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn pays_the_ball_between_the_car_and_the_opponents_net() {
        let reward: BehindBallAlignment = serde_json::from_value(serde_json::json!({})).unwrap();
        // The source's net backs are 321.3875 uu up, as kit's.
        let up = Vec3A::new(0.0, 0.0, 321.3875);
        let lined_up = game(
            up,
            &[
                (Team::Blue, Vec3A::new(0.0, -1000.0, 321.3875)),
                (Team::Orange, Vec3A::new(0.0, 1000.0, 321.3875)),
            ],
        );
        assert!((reward.value(&lined_up, &lined_up, &quiet(2), 0) - 0.75).abs() < 1e-5);
        assert!(
            (reward.value(&lined_up, &lined_up, &quiet(2), 1) - 0.75).abs() < 1e-5,
            "Orange alike"
        );
        let wrong_side = game(
            up,
            &[
                (Team::Blue, Vec3A::new(0.0, 1000.0, 321.3875)),
                (Team::Orange, Vec3A::new(0.0, -1000.0, 321.3875)),
            ],
        );
        assert!((reward.value(&wrong_side, &wrong_side, &quiet(2), 0) + 0.75).abs() < 1e-5);
        assert!((reward.value(&wrong_side, &wrong_side, &quiet(2), 1) + 0.75).abs() < 1e-5);
        let beside = game(up, &[(Team::Blue, Vec3A::new(-1000.0, 0.0, 321.3875))]);
        let across = reward.value(&beside, &beside, &quiet(1), 0);
        assert!(across > 0.0 && across < 0.75, "partly lined up: {across}");
        let on_ball = game(up, &[(Team::Blue, up)]);
        assert_eq!(reward.value(&on_ball, &on_ball, &quiet(1), 0), 0.0);
    }
}
