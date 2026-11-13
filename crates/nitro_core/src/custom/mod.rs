//! Rewards of one's own, written in Rust and compiled into NitroLearn.
//!
//! A reward is a struct whose fields are its settings and which implements [`Reward`]; its name in a
//! settings file is the struct's name in snake case (`FlipReset` -> `flip_reset`). The files to compile are
//! listed in `my-settings/rust-files.txt` (the dashboard's Resources page writes it; one path a line), and
//! `nitro rebuild` builds NitroLearn with them. `example.rs` is one, always built in.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Deserializer};

use crate::snapshot::{GameSnapshot, StepEvents};

// `example.rs`, `library/` and the listed files come in through the generated code below, each as a module.
pub mod kit;
#[cfg(test)]
mod scan;

/// What a reward file starts with: `use crate::custom::prelude::*;`.
pub mod prelude {
    pub use super::Reward;
    pub use super::kit::*;
    pub use crate::snapshot::{
        Body, CarDetail, CarEvents, CarSnapshot, Controls, GameSnapshot, NUM_BOOST_PADS,
        PadSnapshot, StepEvents, Team,
    };
    pub use rocketsim::{Mat3A, Vec3A};
    pub use serde::Deserialize;

    /// The car's top speed (uu/s).
    pub const CAR_MAX_SPEED: f32 = 2300.0;
    /// The ball's top speed (uu/s).
    pub const BALL_MAX_SPEED: f32 = 6000.0;
    /// The ceiling's height (uu).
    pub const CEILING_Z: f32 = 2044.0;
    /// The goal lines' distance from the centre (uu); Blue defends −y.
    pub const GOAL_LINE_Y: f32 = 5120.0;
}

/// A reward of one's own. `value` is what car `car` earns for the step from `prev` to `cur`; the weight,
/// zero-sum sharing and the rest come from its `[[rewards]]` block, as for the built-in rewards.
/// A reward that has to remember something from step to step (a count of touches, a time in the air)
/// says how many numbers per car in `memory` and implements `value_remembering` instead.
pub trait Reward: Send + Sync + 'static {
    /// Paid on every step (a reward of the state, scaled to the tick skip like the built-in ones) rather
    /// than on events. Return a plain `true` (the dashboard reads it from the file).
    fn every_step(&self) -> bool {
        false
    }

    /// How many numbers this reward keeps per car between steps, all 0 when an episode starts. 0 (the
    /// default) for a reward of the step alone. A reward that remembers cannot be a part of an
    /// `all_of` group.
    fn memory(&self) -> usize {
        0
    }

    fn value(
        &self,
        _prev: &GameSnapshot,
        _cur: &GameSnapshot,
        _events: &StepEvents,
        _car: usize,
    ) -> f32 {
        0.0
    }

    /// `value` for a reward that remembers: `memory` holds this car's `memory()` numbers, as this
    /// method left them on the step before.
    fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        let _ = memory;
        self.value(prev, cur, events, car)
    }
}

/// A reward of one's own as a settings file names it.
pub struct Info {
    pub name: &'static str,
    pub doc: &'static str,
    pub every_step: bool,
    /// The file it was compiled from.
    pub file: &'static str,
    pub params: &'static [Param],
}

/// One of its settings (a field of its struct).
pub struct Param {
    pub name: &'static str,
    /// The Rust type, as written.
    pub kind: &'static str,
    /// It may be left out (marked `#[serde(default)]`).
    pub optional: bool,
    pub doc: &'static str,
    /// What it is when left out, as JSON, when its file says so plainly; empty otherwise.
    pub default: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/custom_rewards.rs"));

/// A reward of one's own inside the core's list of rewards: made once from its settings
/// (`{ "kind": name, "settings": {...} }`), then asked for values like a built-in one.
#[derive(Clone)]
pub struct CustomReward {
    name: &'static str,
    settings: serde_json::Value,
    reward: Arc<dyn Reward>,
}

impl CustomReward {
    pub fn new(name: &str, settings: serde_json::Value) -> Result<Self, String> {
        let info = REWARDS
            .iter()
            .find(|info| info.name == name)
            .ok_or_else(|| {
                format!("there is no reward called '{name}' in this build of NitroLearn")
            })?;
        let reward = make(name, settings.clone())?;
        Ok(Self {
            name: info.name,
            settings,
            reward,
        })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn every_step(&self) -> bool {
        self.reward.every_step()
    }

    pub fn memory(&self) -> usize {
        self.reward.memory()
    }

    pub fn value_remembering(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
        memory: &mut [f32],
    ) -> f32 {
        self.reward
            .value_remembering(prev, cur, events, car, memory)
    }

    pub fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 {
        self.reward.value(prev, cur, events, car)
    }
}

#[cfg(test)]
impl CustomReward {
    /// A reward made in a test, without a file.
    pub fn of(name: &'static str, reward: impl Reward) -> Self {
        Self {
            name,
            settings: serde_json::Value::Null,
            reward: Arc::new(reward),
        }
    }
}

impl PartialEq for CustomReward {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.settings == other.settings
    }
}

impl fmt::Debug for CustomReward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name, self.settings)
    }
}

impl<'de> Deserialize<'de> for CustomReward {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Written {
            kind: String,
            #[serde(default)]
            settings: serde_json::Map<String, serde_json::Value>,
        }
        let written = Written::deserialize(deserializer)?;
        Self::new(&written.kind, serde_json::Value::Object(written.settings))
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{CarSnapshot, GameSnapshot};

    #[test]
    fn the_example_is_built_in_listed_with_its_details_and_made_from_its_settings() {
        let info = REWARDS
            .iter()
            .find(|i| i.name == "example_ball_height")
            .expect("built in");
        assert!(info.every_step && info.doc.starts_with("Every step: how high the ball is"));
        assert_eq!(info.params.len(), 1);
        assert_eq!(
            (
                info.params[0].name,
                info.params[0].kind,
                info.params[0].optional
            ),
            ("power", "f32", true)
        );
        let reward =
            CustomReward::new("example_ball_height", serde_json::json!({ "power": 2.0 })).unwrap();
        let mut game = GameSnapshot::default();
        game.cars.push(CarSnapshot::default());
        game.ball.pos.z = 1022.0;
        let events = StepEvents::default();
        assert!((reward.value(&game, &game, &events, 0) - 0.25).abs() < 1e-6);
        assert!(reward.every_step() && reward.name() == "example_ball_height");
        let default = CustomReward::new("example_ball_height", serde_json::json!({})).unwrap();
        assert!(
            (default.value(&game, &game, &events, 0) - 0.5).abs() < 1e-6,
            "power defaults to 1"
        );
        assert!(
            CustomReward::new("example_ball_height", serde_json::json!({ "powr": 2 })).is_err()
        );
        assert!(
            CustomReward::new("nothing", serde_json::json!({}))
                .unwrap_err()
                .contains("no reward called")
        );
    }

    #[test]
    fn rewards_are_found_with_their_details_and_clashing_names_refused() {
        let source = r#"
use crate::custom::prelude::*;

/// Pays for flip resets.
/// Twice as much on the way up.
#[derive(Deserialize)]
pub struct FlipReset {
    /// How much one is worth.
    pub scale: f32,
    #[serde(default)]
    pub upward: bool,
    #[serde(default = "one_and_a_half")]
    pub boost: f32,
    #[serde(default = "side")]
    pub side: String,
    #[serde(default = "computed")]
    pub angle: f32,
}

fn one_and_a_half() -> f32 {
    1_000.5_f32
}
fn side() -> String { "left".into() }
fn computed() -> f32 { std::f32::consts::PI / 4.0 }

// impl Reward for NotThis {} is a comment
impl crate::custom::Reward for FlipReset {
    fn value(&self, _: &GameSnapshot, _: &GameSnapshot, _: &StepEvents, _: usize) -> f32 { self.scale }
}

pub struct Height;
impl Reward for Height {
    fn every_step(&self) -> bool { true }
    fn value(&self, _: &GameSnapshot, c: &GameSnapshot, _: &StepEvents, _: usize) -> f32 { c.ball.pos.z }
}

/// No settings, written as `{}` so that `{}` in a settings file reads.
pub struct Plain {}
impl Reward for Plain {
    fn value(
        &self,
        prev: &GameSnapshot,
        cur: &GameSnapshot,
        events: &StepEvents,
        car: usize,
    ) -> f32 { 1.0 }
}
"#;
        let found = scan::rewards(source);
        assert_eq!(found.len(), 3);
        assert!(found[2].params.is_empty(), "nothing read past `{{}}`");
        assert_eq!(found[0].name, "flip_reset");
        assert_eq!(
            found[0].doc,
            "Pays for flip resets. Twice as much on the way up."
        );
        assert!(!found[0].every_step);
        assert_eq!(
            found[0].params[..2],
            [
                scan::Param {
                    name: "scale".into(),
                    kind: "f32".into(),
                    optional: false,
                    doc: "How much one is worth.".into(),
                    default: String::new(),
                },
                scan::Param {
                    name: "upward".into(),
                    kind: "bool".into(),
                    optional: true,
                    doc: String::new(),
                    default: "false".into(),
                },
            ]
        );
        let defaults: Vec<&str> = found[0].params[2..]
            .iter()
            .map(|p| p.default.as_str())
            .collect();
        assert_eq!(
            defaults,
            ["1000.5", "\"left\"", ""],
            "a literal default is read, a computed one not"
        );
        assert!(found[1].every_step && found[1].params.is_empty() && found[1].name == "height");
        let files = vec![("a.rs".to_string(), source.to_string())];
        let code = scan::generate(&files).unwrap();
        assert!(code.contains("pub mod file_0;") && code.contains("\"flip_reset\" => Ok("));
        let twice = vec![files[0].clone(), ("b.rs".to_string(), source.to_string())];
        assert!(
            scan::generate(&twice)
                .unwrap_err()
                .contains("a second reward is called 'flip_reset'")
        );
        let built_in = vec![(
            "c.rs".to_string(),
            "pub struct Goal; impl Reward for Goal {}".to_string(),
        )];
        assert!(
            scan::generate(&built_in)
                .unwrap_err()
                .contains("built-in reward")
        );
        assert_eq!(scan::snake("ABall"), "a_ball");
    }
}
