# All commands

Everything the dashboard does can also be typed. On Windows, open PowerShell in NitroLearn's folder and
type the commands as shown; on Linux, use `./nitro.sh` instead of `.\nitro`.

| Command | What it does |
|---|---|
| `.\nitro doctor` | Checks this PC (the dashboard's **Check this PC**): Python, PyTorch, the graphics card and its driver, the core, the arena meshes, disk space and more, and says how to fix what is wrong. |
| `.\nitro install-rlbot` | Windows: downloads RLBot v5's official installer from RLBot's releases on GitHub and opens it (RLBot is a separate program; `--status` only says whether it is installed). |
| `.\nitro fetch-meshes` | Downloads the arena's collision meshes from RLGym's package on PyPI, checked by hash (the installer runs it). |
| `.\nitro fetch-recordings` | Downloads the five real-game recordings RocketSim ships (119 MB, from RocketSim's GitHub, checked by hash) for `bundled = true` start states and `pretrain --rocketsim`; they also download by themselves the first time they are needed. |
| `.\nitro check-settings my_bot.toml` | Checks a settings file and lists every mistake with its line number. |
| `.\nitro train my_bot.toml --name NAME` | Starts a training run in `runs\NAME\`. |
| `.\nitro resume NAME` | Continues a run from its last save. |
| `.\nitro pretrain SETTINGS RECORDINGS` | Teaches a bot to copy recorded games; train on from it with `--from` (`--rocketsim`: RocketSim's own recordings). |
| `.\nitro train SETTINGS --from RUN` | Starts a new run from another run's trained networks (transfer learning). |
| `.\nitro new SETTINGS --name NAME` | Makes a bot from a settings file without starting it (to pack it, say). |
| `.\nitro copy-save NAME` | Saves a bot's unfinished copy of replays as its save (the copy also does this itself). |
| `.\nitro reward-check NAME` | Plays the bot's newest save, learning nothing, and says what each reward would pay it. |
| `.\nitro get-replays --size 1v1 --count N` | Downloads N high-level replays (best first, only what you do not have; a stopped download carries on) into `replays\dataset\<size>` and converts them. `--list` shows what can be downloaded, `--no-convert` only downloads. Coming soon: until the replays are online it says they are not available yet. |
| `.\nitro convert-replays FOLDERS` | Converts replays ahead of a later copy of replays, learning nothing, with the timing fit (`--no-fit-timing` without it). Replays converted before are left as they are; `--reconvert` converts them again with the fit (hours for a big store). |
| `.\nitro wandb NAME` | Sends a run's charts to Weights & Biases (its copy of replays and its training). |
| `.\nitro dashboard` | The dashboard window: make, train, change, watch and export bots without typing. |
| `.\nitro update` | Updates NitroLearn to its newest release, keeping your bots, settings, replays and saves (`--check`: only says whether one is out). |
| `.\nitro rebuild` | Builds NitroLearn again with your own Rust rewards (`my-settings\rust-files.txt`, docs/your_own_rewards.md). |
| `.\nitro watch NAME` | A game played by the run's newest save, in the 3D match viewer (`--follow`: a new game after every save; `--live`: played and shown live in a window of its own until you close it; `--rewards`: a chart of what each reward paid; `--moments`: practise moments from the page). |
| `.\nitro score-replay NAME GAME` | Pays every player of a real game (a converted replay, or an `.rlpr` recording) with the bot's rewards and shows it in the 3D viewer with the reward chart: what would your rewards pay a pro? |
| `.\nitro practice NAME FILE` | Adds moments saved by the match viewer (or `--at SECONDS --car N` from the last recording) to the bot's training starts, with `--weight`. |
| `.\nitro eval NAME` | Plays the run's newest save against its oldest (or `--vs-at STEPS`, `--vs OTHER`) and reports the goal share with an interval; `--both-ways` plays each colour once from the same starts (`--log` keeps the result in evals.csv). |
| `.\nitro skills NAME` | Scores the run's newest save on the practice skills and drills against each one's bar; appends to `runs\NAME\skills.csv`; `--weak` also shows where it does worst ([Measuring your bot](measuring.md)). |
| `.\nitro gauntlet NAME` | Plays the run's newest save against four scripted players (chaser, goalie, kickoff, aerial) and prints a report card of its goal shares with ranges; kept in `runs\NAME\gauntlet.json`. |
| `.\nitro strength NAME` | Prints the run's strength ratings against its pinned anchor saves (`[strength] anchors`). |
| `.\nitro export NAME` | Writes an RLBot v5 bot folder (`bots\NAME\`) from the run's newest save. |
| `.\nitro pack NAME` / `.\nitro unpack FILE` | Puts a bot in one file to train it on another computer, and adds it there (or back home). |
| `.\nitro import-gigalearn SAVE --source PROJECT --into NAME` | Makes `NAME` from a GigaLearn bot, to keep training it here (see [Bringing a GigaLearn bot over](gigalearn.md)). |
| `.\nitro import-rlgym SAVE --source PROJECT --into NAME` | Makes `NAME` from an RLGym-PPO bot, to keep training it here (see [Bringing an RLGym-PPO bot over](rlgym.md)). |
| `.\nitro surgery NAME --widen 2` | Makes `NAME-x2`, the same bot with networks twice as wide, to resume with bigger networks. |
| `.\nitro ab A.toml B.toml --task TASK` (or `--speed`) | Compares two settings files. |

Add `--help` to any command for its options.
