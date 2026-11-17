# GigaLearn and AdvancedObsPadderGGL

## GigaLearn

GigaLearnCPP-Leak, by ZealanL (commit `86e11c3`), has no licence file. Its use in NitroLearn
was cleared: "gigalearn is open source and you can use it how you like". NitroLearn adapts these parts of it, written
again in Rust, for bots brought over from GigaLearn:

- Its `AdvancedObs` and `DefaultObs` observation builders: the `gigalearn_advanced` and
  `gigalearn_default` observations (`crates/nitro_core/src/obs.rs`), with `CommonValues::BOOST_LOCATIONS`'
  pad order (test data in `crates/nitro_core/src/sim.rs`).
- Its `DefaultAction` action parser: the `gigalearn` action masks
  (`crates/nitro_core/src/actions.rs`); its action list is NitroLearn's 90-action table, in the same order.
- Both observations, the action list and the masks are checked against numbers GigaLearn's own C++
  (`AdvancedObs.cpp`, `DefaultObs.cpp`, `DefaultAction.cpp`, unchanged) agreed with on the games in
  `crates/nitro_core/testdata/gigalearn_reference.txt` (pads in `BOOST_LOCATIONS`' order, by position, and
  GigaLearn's own pad-timer accessor, which gives each team the other side's timers).
- Its `CommonRewards.h` and `ZeroSumReward.cpp`: the formulas of the
  `gigalearn_*` rewards (`crates/nitro_core/src/custom/library/gigalearn_common.rs`) and of the test that
  checks imported rewards against them (`crates/nitro_core/src/gigalearn_rewards.rs`).
- Its save format (`Util/Models.cpp`, `PPO/PPOLearner.cpp`, `Learner.cpp`, `Util/WelfordStat.h`) and its
  settings' defaults (`LearnerConfig.h`, `PPOLearnerConfig.h`): `python/nitro/gigalearn_import.py`, which
  reads a GigaLearn bot's saves and settings.

## AdvancedObsPadderGGL

AdvancedObsPadderGGL, a community GigaLearn observation, and two rewards of the same community bots
(KickoffProximityReward, SpeedTowardsOpponentReward), whose author let them be used in NitroLearn. They are
written again in Rust: the `gigalearn_padder` observation (`crates/nitro_core/src/obs.rs`, checked against
numbers its own C++ wrote, `crates/nitro_core/testdata/gigalearn_padder_reference.txt`) and the
`kickoff_proximity` and `speed_towards_opponent` rewards (`crates/nitro_core/src/custom/library/`). The
two rewards' C++ is also copied, unchanged, into a test project for the import's tests
(`tests/fixtures/gigalearn/project/`).
