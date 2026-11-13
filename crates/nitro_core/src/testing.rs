//! Shared helpers for the crate's tests.

use std::path::{Path, PathBuf};

use crate::config::PoolConfig;
use crate::meshes::{self, MeshError};

/// Meshes used by the tests: `$NITRO_TEST_MESHES` or the project's `collision_meshes` folder.
pub fn test_meshes() -> PathBuf {
    std::env::var_os("NITRO_TEST_MESHES").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../collision_meshes"),
        PathBuf::from,
    )
}

/// Loads the test meshes; `false`, with a notice on stderr, when they are not on this machine. They are
/// Rocket League assets that never travel with the repo, so a test that needs them returns early instead
/// of failing (`tools/check.sh` counts the notices so skipped tests are never mistaken for passed ones).
pub fn meshes_loaded() -> bool {
    let folder = test_meshes();
    match meshes::load(&folder) {
        Ok(_) => true,
        Err(MeshError::FolderMissing(_) | MeshError::NoSoccarMeshes(_)) => {
            eprintln!("SKIPPED: no collision meshes at {}", folder.display());
            false
        }
        Err(err) => panic!("loading {}: {err}", folder.display()),
    }
}

/// A small 1v1 pool: the default rewards, kickoffs plus random states, short no-touch timeout.
pub fn pool_config(num_arenas: usize, groups: usize, threads: usize, seed: u64) -> PoolConfig {
    let json = format!(
        r#"{{
        "collision_meshes": {meshes:?}, "seed": {seed}, "team_size": 1, "opponents": true, "num_arenas": {num_arenas},
        "groups": {groups}, "threads": {threads}, "tick_skip": 8, "action_delay": 1,
        "observation": "native", "action_masks": "standard",
        "terminals": {{"no_touch_seconds": 2.0, "max_episode_seconds": 30.0, "end_on_touch": false,
                       "end_on_landing": false}},
        "rewards": [
            {{"kind": {{"name": "air"}}, "weight": 0.25, "zero_sum": null}},
            {{"kind": {{"name": "face_ball"}}, "weight": 0.25, "zero_sum": null}},
            {{"kind": {{"name": "velocity_player_to_ball"}}, "weight": 4.0, "zero_sum": null}},
            {{"kind": {{"name": "strong_touch", "min_speed_kph": 20, "max_speed_kph": 100}}, "weight": 60.0, "zero_sum": null}},
            {{"kind": {{"name": "velocity_ball_to_goal"}}, "weight": 2.0, "zero_sum": {{"team_spirit": 1.0, "opponent_scale": 1.0}}}},
            {{"kind": {{"name": "pickup_boost"}}, "weight": 10.0, "zero_sum": null}},
            {{"kind": {{"name": "save_boost", "exponent": 0.5}}, "weight": 0.2, "zero_sum": null}},
            {{"kind": {{"name": "bump"}}, "weight": 20.0, "zero_sum": {{"team_spirit": 0.5, "opponent_scale": 1.0}}}},
            {{"kind": {{"name": "demo"}}, "weight": 80.0, "zero_sum": {{"team_spirit": 0.5, "opponent_scale": 1.0}}}},
            {{"kind": {{"name": "goal", "concede_scale": -1.0}}, "weight": 150.0, "zero_sum": null}}
        ],
        "state_setters": [
            {{"kind": {{"name": "kickoff", "fuzz_uu": 0.0}}, "weight": 1.0}},
            {{"kind": {{"name": "random", "max_ball_speed": 2000.0, "ground_chance": 0.5}}, "weight": 1.0}}
        ]
    }}"#,
        meshes = test_meshes().display().to_string(),
    );
    PoolConfig::from_json(&json).expect("test config")
}

/// A policy file of a plain network with `widths` hidden layers (layer norm, then `act`) and random
/// weights, for a 1v1 game with the native observation.
pub fn random_policy(
    obs_size: usize,
    num_actions: usize,
    widths: &[usize],
    act: &str,
    seed: u64,
) -> Vec<u8> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut weights: Vec<f32> = Vec::new();
    let mut net = |widths: &[usize], outputs: usize, weights: &mut Vec<f32>| {
        let mut specs = Vec::new();
        let mut inputs = obs_size;
        for &width in widths {
            specs.push(format!(
                r#"{{"kind": "linear", "inputs": {inputs}, "outputs": {width}}}"#
            ));
            specs.push(format!(r#"{{"kind": "layer_norm", "size": {width}}}"#));
            specs.push(format!(r#"{{"kind": "{act}"}}"#));
            let scale = 1.0 / (inputs as f32).sqrt();
            weights.extend((0..width * inputs).map(|_| (rng.f32() * 2.0 - 1.0) * scale));
            weights.extend((0..width).map(|_| rng.f32() * 0.2 - 0.1));
            weights.extend((0..width).map(|_| 0.5 + rng.f32()));
            weights.extend((0..width).map(|_| rng.f32() * 0.2 - 0.1));
            inputs = width;
        }
        specs.push(format!(
            r#"{{"kind": "linear", "inputs": {inputs}, "outputs": {outputs}}}"#
        ));
        weights.extend((0..outputs * inputs).map(|_| rng.f32() * 2.0 - 1.0));
        weights.extend((0..outputs).map(|_| rng.f32()));
        specs.join(", ")
    };
    let layers = net(widths, num_actions, &mut weights);
    let format = crate::policy::PLAIN_FORMAT;
    let header = format!(
        r#"{{"format": {format}, "game": {{"observation": "native", "action_masks": "standard", "team_size": 1,
        "opponents": true, "tick_skip": 8, "action_delay": 1}}, "obs_size": {obs_size},
        "num_actions": {num_actions}, "layers": [{layers}]}}"#
    );
    let mut bytes = crate::policy::MAGIC.to_vec();
    bytes.extend(format.to_le_bytes());
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(header.as_bytes());
    for w in &weights {
        bytes.extend(w.to_le_bytes());
    }
    bytes
}
