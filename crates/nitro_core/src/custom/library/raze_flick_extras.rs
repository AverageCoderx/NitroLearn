//! Two rewards of the flick sets: a goal scored inside the goal mouth (goal_in_mouth), and shaping that
//! gets the ball onto the roof (ball_to_roof_shaping).
//! group: moves

use crate::custom::prelude::*;

use crate::custom::library::flick_kit::*;

/// +1 on the step the ball gets into the goal mouth this car attacks, -1 into the one it defends.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalInMouth {}

/// The ball inside a goal mouth, past its line.
fn in_net(ball: &Body) -> bool {
    ball.pos.y.abs() > BACK_WALL_Y + CV_BALL_RADIUS * 0.5
        && ball.pos.x.abs() < GOAL_HALF_WIDTH
        && ball.pos.z < GOAL_HEIGHT
}

impl Reward for GoalInMouth {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
    ) -> f32 {
        if !in_net(&cur.ball) || in_net(&prev.ball) {
            return 0.0;
        }
        let scored = (cur.ball.pos.y > 0.0) == (cur.cars[car].team == Team::Blue);
        if scored { 1.0 } else { -1.0 }
    }
}

/// Each step until the ball is first under control on the roof this episode: how much closer it came to
/// the roof spot (-1 at d_max or further, 0 on it), so an episode pays at most 1; 0 when it starts there.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallToRoofShaping {
    /// Distances from the roof spot count up to this (uu). Default 1500.
    #[serde(default = "kd::d_max")]
    pub d_max: f32,
    /// Ball under control on the roof: its centre at least this far above the car along its roof direction (uu). Default 105.
    #[serde(default = "kd::dribble_min_up")]
    pub dribble_min_up: f32,
    /// ... and at most this far (uu). Default 185.
    #[serde(default = "kd::dribble_max_up")]
    pub dribble_max_up: f32,
    /// ... best this far forward of the car's centre (uu). Default 20.
    #[serde(default = "kd::dribble_target_forward")]
    pub dribble_target_forward: f32,
    /// ... no control this far or further from that spot, sideways (uu). Default 90.
    #[serde(default = "kd::dribble_max_planar_dist")]
    pub dribble_max_planar_dist: f32,
    /// ... nor at this speed relative to the car (uu/s). Default 600.
    #[serde(default = "kd::dribble_max_rel_speed")]
    pub dribble_max_rel_speed: f32,
    /// Control quality (0-1) that counts as the ball under control. Default 0.25.
    #[serde(default = "kd::control_threshold")]
    pub control_threshold: f32,
}

/// BallToRoofShaping's defaults.
mod kd {
    pub fn d_max() -> f32 {
        1500.0
    }
    pub fn dribble_min_up() -> f32 {
        105.0
    }
    pub fn dribble_max_up() -> f32 {
        185.0
    }
    pub fn dribble_target_forward() -> f32 {
        20.0
    }
    pub fn dribble_max_planar_dist() -> f32 {
        90.0
    }
    pub fn dribble_max_rel_speed() -> f32 {
        600.0
    }
    pub fn control_threshold() -> f32 {
        0.25
    }
}

// BallToRoofShaping's memory, per car.
const INIT: usize = 0;
const DONE: usize = 1;
const HAS: usize = 2;
const PHI: usize = 3;

impl BallToRoofShaping {
    fn dribble(&self) -> Dribble {
        Dribble {
            min_up: self.dribble_min_up,
            max_up: self.dribble_max_up,
            target_forward: self.dribble_target_forward,
            max_planar_dist: self.dribble_max_planar_dist,
            max_rel_speed: self.dribble_max_rel_speed,
        }
    }

    fn controlled(&self, car: &CarSnapshot, ball: &Body) -> bool {
        control_quality(car, ball, self.dribble()) >= self.control_threshold
    }

    /// The potential: minus the ball's distance to the roof spot (car-local 20 forward, 133 up), over d_max.
    fn phi(&self, car: &CarSnapshot, ball: &Body) -> f32 {
        let rot = car.body.rot;
        let spot = car.body.pos + rot.x_axis * 20.0 + rot.z_axis * 133.0;
        -div(ball.pos.distance(spot).min(self.d_max), self.d_max)
    }
}

impl Reward for BallToRoofShaping {
    fn memory(&self) -> usize {
        4
    }

    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        _events: &StepEvents,
        car: usize,
        m: &mut [f32],
    ) -> f32 {
        if m[INIT] == 0.0 {
            m[INIT] = 1.0;
            m[DONE] = flag(self.controlled(&prev.cars[car], &prev.ball));
        }
        if m[DONE] != 0.0 {
            return 0.0;
        }
        let me = &cur.cars[car];
        let phi = self.phi(me, &cur.ball);
        let paid = if m[HAS] != 0.0 { phi - m[PHI] } else { 0.0 };
        m[PHI] = phi;
        m[HAS] = 1.0;
        if self.controlled(me, &cur.ball) {
            m[DONE] = 1.0; // paid up to here, then off
        }
        paid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::kit::testing::{game, quiet};

    #[test]
    fn a_goal_in_the_mouth_pays_once_each_way() {
        let reward: GoalInMouth =
            serde_json::from_value(serde_json::json!({})).expect("no settings");
        let cars = [
            (Team::Blue, Vec3A::ZERO),
            (Team::Orange, Vec3A::new(0.0, 100.0, 17.0)),
        ];
        let before = game(Vec3A::new(0.0, 5100.0, 300.0), &cars);
        let inside = game(Vec3A::new(0.0, 5200.0, 300.0), &cars);
        let events = quiet(2);
        assert_eq!(reward.value(&before, &inside, &events, 0), 1.0);
        assert_eq!(reward.value(&before, &inside, &events, 1), -1.0);
        assert_eq!(reward.value(&inside, &inside, &events, 0), 0.0, "once");
        assert_eq!(reward.value(&before, &before, &events, 0), 0.0);
        let wide = game(Vec3A::new(1000.0, 5200.0, 300.0), &cars);
        assert_eq!(reward.value(&before, &wide, &events, 0), 0.0);
        let theirs = game(Vec3A::new(0.0, -5200.0, 300.0), &cars);
        assert_eq!(
            reward.value(&before, &theirs, &events, 1),
            1.0,
            "Orange scores at -y"
        );
    }

    #[test]
    fn pickup_shaping_pays_the_approach_until_control() {
        let reward: BallToRoofShaping =
            serde_json::from_value(serde_json::json!({})).expect("defaults");
        let level = Mat3A::from_cols(Vec3A::Y, Vec3A::NEG_X, Vec3A::Z);
        let at = |ball: Vec3A| {
            let mut g = game(ball, &[(Team::Blue, Vec3A::new(0.0, 0.0, 17.0))]);
            g.cars[0].body.rot = level;
            g
        };
        let games = [
            at(Vec3A::new(0.0, 800.0, 93.0)),
            at(Vec3A::new(0.0, 600.0, 93.0)),
            at(Vec3A::new(0.0, 300.0, 93.0)),
            at(Vec3A::new(0.0, 20.0, 150.0)),
            at(Vec3A::new(0.0, 600.0, 93.0)),
        ];
        let events = quiet(1);
        let mut memory = vec![0.0; reward.memory()];
        let paid: Vec<f32> = (1..games.len())
            .map(|i| reward.value_remembering(&games[i - 1], &games[i], &events, 0, &mut memory))
            .collect();
        assert_eq!(paid[0], 0.0, "the first step only measures");
        assert!(paid[1] > 0.1 && paid[2] > 0.1, "{paid:?}");
        assert_eq!(paid[3], 0.0, "off once the ball was under control");
        let mut fresh = vec![0.0; reward.memory()];
        let on_roof = at(Vec3A::new(0.0, 20.0, 150.0));
        assert_eq!(
            reward.value_remembering(&on_roof, &games[1], &events, 0, &mut fresh),
            0.0
        );
        assert_eq!(
            reward.value_remembering(&games[1], &games[2], &events, 0, &mut fresh),
            0.0
        );
    }

    /// The flick rewards (all of this file family's, as the reward sets use them) in a real 1v1 game
    /// of random play from kickoffs and random states, at tick skip 4: always finite.
    #[test]
    fn the_flick_rewards_stay_finite_in_the_simulator() {
        use std::sync::Arc;

        use crate::config::PoolConfig;
        use crate::env::{AgentRows, Env};
        use crate::pool::{BufferShape, OwnedBuffers};
        use crate::testing::{meshes_loaded, test_meshes};

        if !meshes_loaded() {
            return;
        }
        let custom = |kind: &str, settings: &str| {
            format!(
                r#"{{"kind": {{"name": "custom", "kind": "{kind}", "settings": {settings}}}, "weight": 1.0, "zero_sum": null}}"#
            )
        };
        let rewards = [
            custom(
                "musty_flick",
                r#"{"on_net_bonus": 0.4, "stretch_shot_weight": 0.25}"#,
            ),
            custom("breezi_flick", r#"{"musty_share": 1.0}"#),
            custom("musty_pose_shaping", "{}"),
            custom("tornado_shaping", "{}"),
            custom(
                "raze_flick",
                r#"{"arm_credit": 0.1, "rebound_credit": 0.2, "possession_at_reset": true}"#,
            ),
            custom("raze_flick_setup", r#"{"possession_at_reset": true}"#),
            custom("goal_in_mouth", "{}"),
            custom("ball_to_roof_shaping", "{}"),
        ]
        .join(", ");
        let json = format!(
            r#"{{"collision_meshes": {meshes:?}, "seed": 7, "team_size": 1, "opponents": true, "num_arenas": 1,
            "groups": 1, "threads": 1, "tick_skip": 4, "action_delay": 3, "observation": "native",
            "action_masks": "standard",
            "terminals": {{"no_touch_seconds": 10.0, "max_episode_seconds": 12.0, "end_on_touch": false,
                           "end_on_landing": false}},
            "rewards": [{rewards}],
            "state_setters": [
                {{"kind": {{"name": "kickoff", "fuzz_uu": 0.0}}, "weight": 1.0}},
                {{"kind": {{"name": "random", "max_ball_speed": 2000.0, "ground_chance": 0.5}}, "weight": 1.0}}
            ]}}"#,
            meshes = test_meshes().display().to_string(),
        );
        let rules = Arc::new(
            PoolConfig::from_json(&json)
                .expect("config")
                .rules()
                .expect("rules"),
        );
        let n = rules.actions.len();
        let mut owned = OwnedBuffers::new(BufferShape {
            agents: 2,
            obs_size: rules.obs_size(),
            num_actions: n,
        });
        let mut env = Env::new(Arc::clone(&rules), 0, 7);
        let mut rng = fastrand::Rng::with_seed(3);
        let mut paid = 0;
        for _ in 0..6000 {
            let actions: Vec<i32> = (0..2)
                .map(|agent| {
                    let allowed: Vec<usize> =
                        (0..n).filter(|&a| owned.mask[agent * n + a] != 0).collect();
                    let pick = if allowed.is_empty() {
                        rng.usize(..n)
                    } else {
                        allowed[rng.usize(..allowed.len())]
                    };
                    pick as i32
                })
                .collect();
            env.step(
                &actions,
                &mut AgentRows {
                    obs: &mut owned.obs,
                    mask: &mut owned.mask,
                    reward: &mut owned.reward,
                    terminal: &mut owned.terminal,
                    truncated: &mut owned.truncated,
                    episode: &mut owned.episode,
                    step: &mut owned.step,
                    final_obs: &mut owned.final_obs,
                },
            );
            assert!(
                owned.reward.iter().all(|r| r.is_finite()),
                "{:?}",
                owned.reward
            );
            paid += owned.reward.iter().filter(|&&r| r != 0.0).count();
        }
        assert!(paid > 100, "the shaping rewards pay now and then: {paid}");
    }
}
