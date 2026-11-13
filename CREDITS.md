# Credits

NitroLearn is written from scratch. This file lists the code it depends on, the small pieces of data and
formulas taken from elsewhere, and the projects whose ideas shaped it. The rewards that come with
NitroLearn, with who made each one and under which licence, are listed in `docs/rewards.md`. The pieces
adapted from Apache-2.0 code keep their notices in their files; the licence's full text is in
`LICENSES/Apache-2.0.txt`.

The compiled parts carry the licences of the Rust crates built into them, in `THIRD_PARTY_NOTICES.md`: next to
the compiled core (`python/nitro`; the release's ready-made core brings it, and the installer writes it after a
build from the code, or run `python python/nitro/notices.py core --out python/nitro/THIRD_PARTY_NOTICES.md`),
next to the release's ready-made programs, and in every exported bot folder (`NOTICES.md`).

## Dependencies (used as libraries, not copied)
- **RocketSim v3** (crates.io `rocketsim` 0.3.3, published from `4572287`, branch `v3-rust`) — ZealanL,
  VirxEC and contributors, MIT. `github.com/ZealanL/RocketSim`. The simulator every game runs in. Much of its
  physics step is a port of **Bullet Physics** (zlib, Copyright (c) 2003-2006 Erwin Coumans). Its bundled
  real-game recordings are used for the `replay` start states and the tests.
- **RLBot v5 rust-interface** (`rlbot`, `rlbot_flat` 0.6.0) — RLBot contributors, MIT. The bot
  (`crates/nitro_bot`) uses it for the connection, the handshake and the packet types. The flatbuffers
  schema it builds on has no licence file, so it is used only through the MIT-licensed interface packages.
- **replicar** — Rolv Arild, MIT (Copyright (c) 2026 Rolv Arild). `github.com/Rolv-Arild/replicar`. Turns
  Rocket League replays into RocketSim states with the players' controls, for the replay copy
  (`crates/nitro_replicar` builds on its library). The release's ready-made programs include its own
  command-line tool (replicar-cli 2.2.1), built unchanged; the programs' `THIRD_PARTY_NOTICES.md` carries its
  licence and those of every crate compiled into each program.
- **uv** (Astral, MIT or Apache-2.0) — the installer puts it (a pinned version, the wheel from PyPI) in
  NitroLearn's folder to fetch Python and the packages; Python itself is Astral's **python-build-standalone**
  build (Python Software Foundation licence, with the licences of the libraries built in). The setup wizard
  carries uv.exe; the release carries copies of both, unchanged (`LICENSES/uv-MIT.txt`,
  `LICENSES/Python-PSF.txt`, `LICENSES/python-build-standalone.md`).
- **Inno Setup** (Jordan Russell and Martijn Laan; free for any use, `LICENSES/Inno-Setup.txt`) — makes the
  Windows setup wizard, `NitroLearn-Setup.exe`.
- **The release's own copies** — so an install depends on other servers as little as possible, each release
  also carries, unchanged and each under its own licence: the source code of every Rust crate the builds use
  (`nitrolearn-crates-<version>.tar.gz`, listed in its `VENDORED-CRATES.md`) and every Python package an
  install takes from PyPI except PyTorch (`nitrolearn-wheels-<version>-<platform>.zip`, listed in its
  `NOTICES.md`). `LICENSES/BUNDLED.md` lists every copy and what is not carried (PyTorch, the arena meshes,
  Qt, the NVIDIA driver, RLBot), and why.
- **RLBot v5** (RLBot contributors; launcher and core MIT, with parts RLBot keeps closed) — a separate
  program, never shipped or hosted with NitroLearn. On request (part 2 of the installer's list of parts, a
  tick box on NitroLearn-Setup's Components page, or `nitro install-rlbot`), NitroLearn downloads RLBot's own
  installer from RLBot's releases on GitHub and opens it; RLBot installs itself under its own licence terms,
  in its own folder.
- **PyTorch** (BSD-3-Clause), **NumPy** (BSD-3-Clause); optional: **Weights & Biases SDK** (`wandb`, MIT) for
  run tracking, **pyarrow** (Apache-2.0) for converted replays, **pywebview** (BSD-3-Clause) and **tomlkit**
  (MIT) for the dashboard.
- **three.js** 0.186.1 — the three.js authors, MIT. `github.com/mrdoob/three.js`. Draws the match viewer.
  The one library whose file comes inside NitroLearn: one script, unchanged, with its licence beside it
  (`python/nitro/viewer/vendor/`).
- **Fonts** of the dashboard and the match viewer: Archivo (Omnibus-Type) and JetBrains Mono (JetBrains),
  SIL Open Font License 1.1, bundled with their licence texts (`python/nitro/brand/fonts/LICENSES.txt`).

## Data, constants and formulas
- **90-action lookup table construction** (ground 24 + aerial 66 actions) — RLGym's LookupTableAction
  (rlgym-rocket-league 2.0.1, Copyright 2021 Lucas Emery, Apache-2.0), adapted in Rust for NitroLearn's
  action table (`crates/nitro_core/src/actions.rs`).
- **Rewards adapted from Apache-2.0 code** — seven built-in rewards follow RLGym's common rewards (RLGym 1.2.2,
  Copyright 2021 Lucas Emery; `crates/nitro_core/src/rewards.rs`), `align_ball_goal.rs` and
  `ball_y_coordinate.rs` are adapted from rlgym-sim, and `aerial_chain_distance` follows rlgym-tools'
  AerialDistanceReward (Rolv Arild Braaten). Each file says what was changed; `docs/rewards.md` lists every
  reward's origin.
- **Wall dash rules from samuelkassi's walldashcoach (MIT)** — github.com/samuelkassi/walldashcoach, Copyright
  (c) 2026 samuelkassi; the licence's text is in `LICENSES/walldashcoach-MIT.txt`. The `wall_dash` reward
  (`crates/nitro_core/src/custom/library/wall_dash.rs`) uses its rules: what counts as a made dash,
  the timing and flip angle bands, the speed window, the wall types, chains and the spam guard.
- **Muon** (`python/nitro/muon.py`) — Keller Jordan's optimizer, written from its public description; the
  Newton-Schulz coefficients (3.4445, −4.7750, 2.0315) are the published ones.
- **Game constants** shared with RLGym's common values: max car speed 2300, max ball speed 6000, back of the
  net y = 6000, goal height 642.775, km/h → uu/s factor 250/9.
- **CartPole dynamics** (`python/nitro/cartpole.py`) — written to match Gymnasium's CartPole-v1 (Farama
  Foundation, MIT; the cart-pole problem of Barto, Sutton & Anderson, 1983) so the learning proof can be
  checked against it.
- **Discount ↔ horizon formula** `H½ = (tick_skip/120)·ln 0.5/ln γ` — from Kuenec's (Kue's) guide to bot
  training.
- **Rocket League's rotator convention** (pitch, yaw, roll → forward/right/up matrix, `crates/nitro_bot/src/angles.rs`)
  — the standard trigonometric form every RLBot bot library uses for the game's angles (RLUtilities, MIT;
  RocketSim's C++ `Angle::ToRotMat`, MIT); written from the convention, checked by tests.
- **Boost-pad respawn times** (big 10 s, small 4 s) and the meaning of RLBot's pad `timer` — from the
  RLBot v5 flatbuffers schema documentation, through the MIT `rlbot_flat` crate.

## Ideas adopted (no code copied)
- **Rocket Learn** (Rolv Arild Braaten, Daniel Downs): the rated opponent pool, p(1−p) matchmaking, scoreboard
  context, stat trackers.
- **Kuenec (Kue)**: scaling the rewards paid every step by how often the bot decides (the tick skip), so a
  weight means the same at any decision rate, and warning when one reward makes up nearly all that is paid
  (a sign the bot is farming it).
- **GigaLearn** (ZealanL): action masking, entropy normalised by log(#actions), return-std reward scaling.
- **Pulsar2's design** (none of its code): the goal-and-concede-only ("GCO") setup's residual shared trunk
  and residual policy, the larger value side with several value heads trained with Muon, and the learning
  signals it describes (twin optimistic critics, the goal critic, learning from the samples with the
  largest advantages, a target for the policy's randomness); also, from Pulsar's authors, that training
  every game size from scratch at once can stall a run.
- **buta-ppo and Pulsar**: shuffling which place in the observation each car takes, for networks that play
  several game sizes (`env.shuffle_slots`).
- **EnvPool**: keeping each process's threads on its own share of the processor (its NUMA idea;
  `run.cards_pin_threads`).
- **RLMind** (corresation; read, not copied): counting what the bots do in play (flips, aerials, resets and
  similar gameplay statistics), whatever they are paid for.
- **Lucy-SKG** (Moschopoulos et al., arXiv 2305.15801; paper read, no code): the multiplicative "AND"
  combination of reward parts (`all_of`) and the car–ball–goal alignment shaping term (`alignment`), both
  written from the description.

## Start-state ideas (the `replay` state setter)
- Necto (Rolv Arild Braaten, `state.py`) and Seer (Table 3.6): about 70% of training episodes started from real-game
  states, the rest kickoffs and drills; the idea of a replay-state setter and of weighting toward height.
- RocketSim (ZealanL, VirxEC): the RLPR recording format and the bundled recordings the setter reads.
  The five recordings (`rocketsim_test/recordings` at commit `4572287`, added by VirxEC) are part of
  RocketSim's repository under its MIT licence (Copyright (c) 2022 ZealanL, Copyright (c) 2025 VirxEC); the
  repository names no other terms for them. They are not in NitroLearn's files: `nitro fetch-recordings`, or a
  run that needs them, downloads them, checked by hash, into `rocketsim_recordings/`, from NitroLearn's
  release (`rocketsim-recordings.zip`: the five files unchanged with RocketSim's licence,
  `LICENSES/RocketSim-MIT.txt`), else from that commit on GitHub.

## Viewer ideas (the match viewer, `python/nitro/viewer/`; ideas only, no code and no art)
- **RocketSimVis** — ZealanL, and **RLViser** — VirxEC: what a RocketSim match viewer shows (cars, ball,
  pads, trails, cameras). Ours is a page in a window instead of a program listening on a port.
- **RLMind** (corresation; read, not copied): the ball's height marker and ring, and the choice of cameras.
- The arena is drawn from RocketSim's collision meshes when the user has them; the car bodies, ball,
  stadium and effects are our own designs. The steering-angle curve and the body sizes are RocketSim's
  constants.

## Observation design ideas (the `nitro_v2` observation; read, not copied)
- **RLGym 2 `DefaultObs` and rlgym-tools `RelativeDefaultObs`** — RLGym contributors, Apache-2.0: the
  relative-to-the-agent encoding of the ball and other cars, pad timers as a block, the partially
  observable flag block. `github.com/RLGym/rlgym`, `github.com/RLGym/rlgym-tools`.
- **Necto / Nexto observation builders** — Rolv Arild Braaten and contributors (repository under
  CC BY-NC-SA 4.0): entity-relative positions and velocities, yaw-only and full car frames side by side,
  boost pads as entities with type and timer. `github.com/Rolv-Arild/Necto`.
- **Opti (`CoyoteObs`)** — Kaiyotech: the ball's scalar speed and on-the-floor flag, each other car's
  relation to the ball (vector, speed toward it, distance) and its distance to the agent.
  `github.com/Kaiyotech/Opti`.
- The goal-relative features (vector and distance from the ball to each goal, the goals in the car's
  frame, the ball's speed toward the opponent's goal) and the unit ball direction are ours.
- **Padded observations** (`env.obs_team_size`) — a common idea (RLMind's `DefaultObsPadded`, read after
  ours was built). The "a car is here" number at the end of every place is ours.
