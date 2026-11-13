# v1-attention

A Rocket League bot trained with NitroLearn (run `v1-attention`, 1,280 agent steps; 1v1,
tick skip 4, action delay 1 tick).

## Fair play
Play it offline only: RLBot matches, private matches against your own bots, and bot tournaments. Never use it
in online matches against real players: that is cheating, and it breaks Rocket League's terms of service.

## Playing it in RLBot v5
1. Install RLBot v5 (https://rlbot.org) and start it.
2. In RLBot, add this folder (or its parent) as a bot folder: the bot appears as **v1-attention**.
3. Put it in a match and start. The game must be running; RLBot starts the bot program itself.

## Files
| File | What it is |
|---|---|
| `bot.toml` | How RLBot starts the bot; rewritten by every export (`--bot-name` sets the name shown). |
| `policy.nwt` | The trained policy: the network's weights and the game settings it was trained with. |
| `loadout.toml` | The car's look (item ids). Cosmetic only. |
| `nitro_bot` | The bot program. It needs no Python and no graphics card. |
| `NOTICES.md` | The licence notices of the bot program and this folder: keep it with them. |

The bot program reads the game, builds the observation exactly as in training, runs the network and sends
the controls.

## Options
Add `--sample` to `run_command` in `bot.toml` to make the bot pick actions randomly in proportion to its
confidence, as it did during training. Without it, the bot always takes its most likely action.
If the game uses a sent control one tick late, add `--latency 1` so the bot sends each decision a tick
early and acts exactly as in training. By default the bot sends them on the training rhythm (`--latency 0`).
A bot with an attention network plays matches bigger than it trained for (one place per car, any size; how
well is not measured); add `--trained-size-only` to make it sit those out, as other bots do.
The bot answers every game packet at once and runs the network on its own thread until the decision is needed;
the cars of a 2v2 or 3v3 decide in one pass, split across 2 threads (`--threads 2`, the default; the same
decisions with any number). Use `--threads 1` on a machine with few free cores, and `--no-decider` to decide
on each car's own thread instead.

**The bot program is not here yet:** building it was skipped (`nitro export --no-build`). Run `nitro export`
again without `--no-build` to build `nitro_bot` and copy it here.

Any RLBot bot folder works from any location: copy this whole folder wherever you keep your bots.
