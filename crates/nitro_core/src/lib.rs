//! Rust core of NitroLearn: simulator access, environments and rollout collection.

pub mod actions;
pub mod capture;
pub mod config;
pub mod custom;
pub mod demos;
pub mod driver;
pub mod env;
pub mod mechanics;
pub mod meshes;
pub mod obs;
pub mod policy;
pub mod pool;
pub mod replay;
pub mod replicar;
pub mod rewards;
pub mod rlpr;
pub mod setters;
pub mod sim;
pub mod snapshot;
pub mod team;
pub mod terminals;
#[cfg(test)]
mod testing;

/// Which physics engine this build steps.
pub const SIMULATOR: &str = "RocketSim v3";

/// Version of the core, shared with the Python package through the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
