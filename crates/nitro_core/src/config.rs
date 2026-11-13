//! Environment settings as sent by the Python front end (already validated there, re-checked here).

use std::path::PathBuf;

use serde::Deserialize;

use crate::actions::{ActionTable, MaskRule};
use crate::env::EnvRules;
use crate::obs::ObsKind;
use crate::rewards::RewardTerm;
use crate::setters::{SetterMix, WeightedSetter};
use crate::snapshot::MAX_CARS;
use crate::terminals::TerminalConfig;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    pub collision_meshes: PathBuf,
    pub seed: u64,
    pub team_size: usize,
    /// Mixed game sizes: the sizes played side by side, each arena one of them for the whole run;
    /// empty = every arena `team_size` (which is then the largest of these).
    #[serde(default)]
    pub team_sizes: Vec<usize>,
    /// Each of `team_sizes`'s share of the arenas, in its order; empty = equal shares.
    #[serde(default)]
    pub team_size_shares: Vec<f32>,
    /// Each episode every car takes a random place among its team's places in the padded observation;
    /// off = the cars in order.
    #[serde(default)]
    pub shuffle_slots: bool,
    /// False: only the Blue team plays (practice tasks for a lone car).
    pub opponents: bool,
    pub num_arenas: usize,
    /// Arenas are split into this many groups that take turns on the CPU and the GPU.
    pub groups: usize,
    /// Worker threads (0 = all logical cores but two).
    pub threads: usize,
    pub tick_skip: u32,
    pub action_delay: u32,
    pub observation: ObsKind,
    /// Players per team the observation has places for; 0 = the game's own.
    #[serde(default)]
    pub obs_team_size: usize,
    pub action_masks: MaskRule,
    pub terminals: TerminalConfig,
    pub rewards: Vec<RewardTerm>,
    pub state_setters: Vec<WeightedSetter>,
}

impl PoolConfig {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let config: Self =
            serde_json::from_str(json).map_err(|err| format!("invalid pool settings: {err}"))?;
        config.check()?;
        Ok(config)
    }

    fn check(&self) -> Result<(), String> {
        let problems = [
            (
                self.team_size == 0 || self.team_size * 2 > MAX_CARS,
                "team_size must be 1 to 4",
            ),
            (
                self.groups == 0 || self.num_arenas < self.groups,
                "need at least one arena per group",
            ),
            (self.tick_skip == 0, "tick_skip must be at least 1"),
            (
                self.action_delay >= self.tick_skip,
                "action_delay must be smaller than tick_skip",
            ),
            (
                self.obs_team_size != 0
                    && (self.obs_team_size < self.team_size || self.obs_team_size * 2 > MAX_CARS),
                "obs_team_size must be 0 or from team_size to 4",
            ),
            (
                self.state_setters.is_empty(),
                "at least one state setter is needed",
            ),
            (
                self.state_setters.iter().map(|s| s.weight).sum::<f32>() <= 0.0,
                "state setter weights must sum to more than 0",
            ),
            (
                !(0.0..=1.0).contains(&self.terminals.truncate_on_ball_landing),
                "truncate_on_ball_landing must be between 0 and 1",
            ),
        ];
        let first = problems
            .iter()
            .find(|(bad, _)| *bad)
            .map(|&(_, message)| message)
            .or_else(|| self.rewards.iter().find_map(RewardTerm::problem))
            .or_else(|| self.state_setters.iter().find_map(WeightedSetter::problem))
            .or_else(|| {
                let limits = [
                    self.terminals.no_touch_seconds,
                    self.terminals.max_episode_seconds,
                ];
                (!limits.iter().all(|s| (0.0..=f32::MAX).contains(s)))
                    .then_some("episode time limits must be 0 or more")
            });
        if let Some(message) = first {
            return Err(message.to_owned());
        }
        if let Some(problem) = self.mixed_sizes_problem() {
            return Err(problem);
        }
        if self.shuffle_slots && self.obs_team_size == 0 {
            return Err("shuffle_slots needs a padded observation (obs_team_size)".to_owned());
        }
        Ok(())
    }

    fn mixed_sizes_problem(&self) -> Option<String> {
        let sizes = &self.team_sizes;
        if sizes.is_empty() {
            return (!self.team_size_shares.is_empty())
                .then(|| "team_size_shares needs team_sizes".to_owned());
        }
        let largest = sizes.iter().copied().max().unwrap_or(0);
        let distinct = sizes.iter().collect::<std::collections::HashSet<_>>().len() == sizes.len();
        let problems = [
            (sizes.len() < 2, "team_sizes needs two or more game sizes"),
            (
                sizes.iter().any(|&n| n == 0 || n * 2 > MAX_CARS),
                "team_sizes must each be 1 to 4",
            ),
            (!distinct, "team_sizes names a game size twice"),
            (
                self.team_size != largest,
                "team_size must be the largest of team_sizes",
            ),
            (
                !self.team_size_shares.is_empty() && self.team_size_shares.len() != sizes.len(),
                "team_size_shares needs one share per game size in team_sizes",
            ),
            (
                self.team_size_shares
                    .iter()
                    .any(|s| !(s.is_finite() && *s > 0.0)),
                "team_size_shares must each be above 0",
            ),
            (!self.opponents, "team_sizes needs opponents"),
            (
                self.obs_team_size < largest,
                "team_sizes needs an observation with room for the largest size (obs_team_size)",
            ),
            (
                self.state_setters
                    .iter()
                    .any(|s| matches!(s.kind, crate::setters::SetterKind::Replay { .. })),
                "team_sizes cannot be used with replay state setters yet",
            ),
        ];
        if let Some((_, message)) = problems.iter().find(|(bad, _)| *bad) {
            return Some((*message).to_owned());
        }
        let counts = self.size_counts();
        counts.iter().zip(sizes).find(|(c, _)| **c == 0).map(|(_, n)| {
            format!(
                "with {} arenas in {} groups the share of {n}v{n} gets no arena; raise num_arenas or its \
                 share in team_size_shares",
                self.num_arenas, self.groups
            )
        })
    }

    /// Places in a group (an arena's place is `arena / groups`, its group `arena % groups`).
    fn places(&self) -> usize {
        self.num_arenas.div_ceil(self.groups)
    }

    /// How many places of each group play each of `team_sizes` (largest remainder of the shares).
    fn size_counts(&self) -> Vec<usize> {
        let n = self.team_sizes.len();
        let shares: Vec<f64> = if self.team_size_shares.is_empty() {
            vec![1.0; n]
        } else {
            self.team_size_shares
                .iter()
                .map(|&s| f64::from(s))
                .collect()
        };
        let total: f64 = shares.iter().sum();
        let places = self.places();
        let exact: Vec<f64> = shares.iter().map(|s| s / total * places as f64).collect();
        let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let fraction = |i: usize| exact[i] - exact[i].floor();
            fraction(b).total_cmp(&fraction(a)).then(a.cmp(&b))
        });
        let left = places - counts.iter().sum::<usize>();
        for &i in order.iter().take(left) {
            counts[i] += 1;
        }
        counts
    }

    /// Every arena's players a side. With `team_sizes`, an arena's size follows its place in its
    /// group, so every group holds the same sizes in the same order (and the same rows), and the sizes are
    /// spread evenly over the places, so any run of neighbouring arenas (a self-play block) mixes them.
    pub fn arena_team_sizes(&self) -> Vec<usize> {
        if self.team_sizes.is_empty() {
            return vec![self.team_size; self.num_arenas];
        }
        let counts = self.size_counts();
        let places = self.places();
        let mut given = vec![0usize; counts.len()];
        let mut by_place = Vec::with_capacity(places);
        for place in 0..places {
            // The size furthest behind its even spread so far, in places x `places` (ties: the first listed).
            let due = |i: usize| (counts[i] * (place + 1)) as i64 - (given[i] * places) as i64;
            let next = (0..counts.len())
                .filter(|&i| given[i] < counts[i])
                .max_by(|&a, &b| due(a).cmp(&due(b)).then(b.cmp(&a)))
                .expect("the counts add up to the places");
            given[next] += 1;
            by_place.push(self.team_sizes[next]);
        }
        (0..self.num_arenas)
            .map(|arena| by_place[arena / self.groups])
            .collect()
    }

    pub fn worker_threads(&self) -> usize {
        if self.threads > 0 {
            return self.threads;
        }
        let logical = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        logical.saturating_sub(2).max(1)
    }

    /// The environment rules; an error when a state setter's recordings cannot be loaded.
    pub fn rules(&self) -> Result<EnvRules, String> {
        let actions = ActionTable::new();
        let setters = SetterMix::new(self.state_setters.clone(), self.team_size, self.opponents)?;
        Ok(EnvRules {
            team_size: self.team_size,
            opponents: self.opponents,
            tick_skip: self.tick_skip,
            action_delay: self.action_delay,
            obs: self.observation,
            obs_team_size: self.obs_team_size,
            masks: self.action_masks,
            terminals: self.terminals,
            rewards: self.rewards.clone(),
            setters,
            actions,
            shuffle_slots: self.shuffle_slots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PoolConfig;
    use crate::rewards::RewardKind;
    use crate::setters::SetterKind;
    use crate::testing::pool_config;

    #[test]
    fn values_the_loader_would_reject_are_rejected_here_too() {
        assert_eq!(pool_config(2, 1, 1, 0).check(), Ok(()));

        let mut config = pool_config(2, 1, 1, 0);
        config.state_setters[0].weight = -0.5;
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("weights must be 0 or more")
        );

        let mut config = pool_config(2, 1, 1, 0);
        config.state_setters[1].kind = SetterKind::Random {
            max_ball_speed: 2000.0,
            ground_chance: f32::NAN,
        };
        assert!(config.check().unwrap_err().contains("ground_chance"));

        let mut config = pool_config(2, 1, 1, 0);
        config.rewards[3].kind = RewardKind::StrongTouch {
            min_speed_kph: 50.0,
            max_speed_kph: 50.0,
        };
        assert!(config.check().unwrap_err().contains("strong_touch"));

        let mut config = pool_config(2, 1, 1, 0);
        config.rewards[0].weight = f32::INFINITY;
        assert!(config.check().unwrap_err().contains("finite"));

        let mut config = pool_config(2, 1, 1, 0);
        config.terminals.truncate_on_ball_landing = -0.1;
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("truncate_on_ball_landing")
        );

        let mut config = pool_config(2, 1, 1, 0);
        config.obs_team_size = 3;
        assert_eq!(config.check(), Ok(()));
        config.team_size = 4;
        assert!(config.check().unwrap_err().contains("obs_team_size"));
    }

    #[test]
    fn mixed_sizes_spread_evenly_over_every_group_and_bad_ones_are_refused() {
        let mut config = pool_config(12, 2, 1, 0);
        assert_eq!(
            config.arena_team_sizes(),
            vec![1; 12],
            "off: every arena team_size"
        );
        config.team_size = 3;
        config.obs_team_size = 3;
        config.team_sizes = vec![1, 2, 3];
        assert_eq!(config.check(), Ok(()));
        let sizes = config.arena_team_sizes();
        // Six places a group, two of each size, spread out; both groups alike (arena i: place i / 2).
        assert_eq!(sizes, vec![1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]);
        config.team_size_shares = vec![4.0, 1.0, 1.0];
        assert_eq!(
            config.arena_team_sizes(),
            vec![1, 1, 1, 1, 2, 2, 1, 1, 3, 3, 1, 1]
        );
        config.num_arenas = 11; // the last place holds one arena
        assert_eq!(config.arena_team_sizes().len(), 11);

        let refused = |change: &dyn Fn(&mut PoolConfig), says: &str| {
            let mut bad = pool_config(12, 2, 1, 0);
            (bad.team_size, bad.obs_team_size, bad.team_sizes) = (3, 3, vec![1, 2, 3]);
            change(&mut bad);
            let problem = bad.check().unwrap_err();
            assert!(problem.contains(says), "{problem:?} should say {says:?}");
        };
        refused(&|c| c.team_sizes = vec![3], "two or more");
        refused(&|c| c.team_sizes = vec![3, 3], "twice");
        refused(&|c| c.team_size = 2, "largest of team_sizes");
        refused(&|c| c.obs_team_size = 2, "obs_team_size must be 0 or from");
        refused(&|c| c.obs_team_size = 0, "room for the largest");
        refused(&|c| c.opponents = false, "needs opponents");
        refused(&|c| c.team_size_shares = vec![1.0], "one share per");
        refused(&|c| c.team_size_shares = vec![1.0, 0.0, 1.0], "above 0");
        refused(
            &|c| c.team_size_shares = vec![100.0, 1.0, 1.0],
            "gets no arena",
        );
        let mut config = pool_config(2, 1, 1, 0);
        config.team_size_shares = vec![1.0];
        assert!(config.check().unwrap_err().contains("needs team_sizes"));
        config.team_size_shares.clear();
        config.shuffle_slots = true;
        assert!(config.check().unwrap_err().contains("padded observation"));
        config.obs_team_size = 2;
        assert_eq!(config.check(), Ok(()));
    }
}
