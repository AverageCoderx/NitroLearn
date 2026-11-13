//! An example of a reward of one's own, always built in: copy this file, rename
//! the struct, change the maths, add the copy on the dashboard's Resources page and press Build.

use crate::custom::prelude::*;

/// Every step: how high the ball is, 0 on the ground to 1 at the ceiling; `power` bends the curve.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExampleBallHeight {
    /// 1 = in proportion to the height; 2 = pays much more near the ceiling.
    #[serde(default = "one")]
    pub power: f32,
}

fn one() -> f32 {
    1.0
}

impl Reward for ExampleBallHeight {
    fn every_step(&self) -> bool {
        true
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        _car: usize,
    ) -> f32 {
        (cur.ball.pos.z / CEILING_Z)
            .clamp(0.0, 1.0)
            .powf(self.power)
    }
}
