# RLGym

NitroLearn carries pieces of RLGym, used under the Apache License 2.0, for bots brought over from RLGym-PPO
(`nitro import-rlgym`):

- rlgym-sim (github.com/AechPro/rocket-league-gym-sim) and rlgym-tools v1.8.3 (github.com/RLGym/rlgym-tools):
  their observation builders (`DefaultObs`, `AdvancedObs`, `AdvancedObsPadder`), game-state classes, common
  values and `LookupAction`, copied into `python/nitro/rlgym_reference/` with each project's licence
  (`LICENSE-rlgym-sim`, `LICENSE-rlgym-tools`) and a `NOTICE.md` of the small changes made. They are used only
  to check an imported bot: the import runs RLGym's own observation code on real game states and holds
  NitroLearn's observation to the same numbers.
- The formulas of `EventReward`'s parts and the Liu distance rewards (rlgym-sim), written again in Rust:
  `crates/nitro_core/src/custom/library/rlgym_common.rs`.
- The `rlgym_padder` observation (`crates/nitro_core/src/obs.rs`) is rlgym-tools' `AdvancedObsPadder`, built
  from NitroLearn's copy of AdvancedObsPadderGGL (see `GigaLearn-notice.md`) with each car's `has_jump`.

RLGym-PPO's and rlgym-learn's save formats and settings are read by `python/nitro/rlgym_import.py`; no code of
theirs is copied. The licence's full text: `Apache-2.0.txt`.
