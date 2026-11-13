# How NitroLearn fits together

NitroLearn has three parts: a Rust core that runs the games, a Python package that learns, and a Rust bot
program that plays a trained policy in RLBot v5. The core's observation and action code is compiled once
and linked into both the trainer and the bot, so a bot sees the game exactly as it did in training.

```
settings.toml ─▶ nitro.config (checked) ─▶ TrainingRun ─┬─ inference (CUDA graphs on a graphics card)
                                                        ├─ PPO learner (PyTorch, FP32 or FP16)
                                                        ├─ self-play: past versions, ratings, matchmaking
                                                        └─ metrics, saves, the console report, W&B
                         ▲  shared buffers (observations, masks, rewards, ends, actions)
                         │  PyO3 module nitro._core
crates/nitro_core: EnvPool (worker threads, game groups) → RocketSim arenas → GameSnapshot →
                   observations, rewards, game ends, start states → written straight into the buffers
crates/nitro_bot:  RLBot v5 packet → GameSnapshot → the same observations → the policy on the CPU → controls
```

## The Rust core (`crates/nitro_core`)
- `sim.rs` wraps RocketSim v3 (pinned in `Cargo.toml`): arenas, state, controls, steps, touches and goals.
- `snapshot.rs` is the only view of a game that observations, rewards and game ends read; the bot builds the
  same snapshot from an RLBot packet.
- `obs.rs` (the `native` and `nitro_v2` observations, padded rooms for several game sizes), `actions.rs`
  (the 90-action table), `rewards.rs` (built-in rewards), `custom/` (the reward library and rewards of your
  own), `setters.rs` and `replay.rs` (how games start), `terminals.rs` (when a game ends).
- `pool.rs` and `env.rs` step many arenas on worker threads; every arena has its own seeded random stream,
  so the games do not depend on the thread count.
- `policy.rs` reads exported policy files (`.nwt`) and runs them on the CPU for the bot.

## The Python package (`python/nitro`)
- `config/`: the settings schema with every setting's help, loaded from TOML and checked before a run.
- `training.py` and `trainer.py`: a run in its folder (`runs/<name>`), its iterations, saves and resumes.
- `learner.py`, `ppo.py`, `model.py`: clipped PPO with PopArt, the networks (MLP, attention encoders,
  transformer, shared and residual trunks, extra value heads) and the optimizers (Adam, Muon).
- `selfplay.py`, `ratings.py`: playing frozen past versions with skill ratings.
- `pretrain.py`, `replays.py`, `replay_mix.py`: copying players from replays, before or during training.
- `export.py`: the RLBot bot folder; `eval.py`, `skills.py`, `tasks/`, `ab.py`: measuring bots and settings.
- `dashboard/`: the dashboard window (pywebview), its page in parts (`dashboard/web/`) and its texts
  (`dashboard/winapp.py`: one window at a time, its own taskbar identity and icon, the shortcut record);
  `cli.py`: the `nitro` commands; `update.py`: `nitro update` and the dashboard's Update NitroLearn (the
  newest GitHub release, by git or by its source archive, then the installer again).
- `viewer/`: the 3D match viewer (three.js, bundled) that `nitro watch` and the dashboard show games in,
  recorded or live; `watch.py` plays those games; `brand/`: the logo, the app icon (`tools/make_icon.py`)
  and the fonts both windows share.
- `multigpu.py`, `cards_run.py`: one bot trained on several graphics cards (Linux).

## The bot (`crates/nitro_bot`)
One small program per bot folder, with no Python and no graphics card: it connects to RLBot v5, builds each
car's observation from the packet, runs the policy and sends the controls on the training rhythm. A 2v2 or
3v3 bot drives every car of its team.

## Tests
Rust tests live in each crate (`cargo test --workspace`); Python tests in `tests/` (`pytest`). Tests that
simulate games need the collision meshes and skip themselves without them; tests that need a graphics card
skip themselves on the CPU. `tools/check.sh` (Linux) and `tools/check.ps1` (Windows) run every check.
