# More settings

The starters work as they are. This page is for later: settings worth knowing, the choices to make before a
long run, bigger networks, attention networks for 2v2 and 3v3, starting from a bot you already have, and
comparing two settings. Every setting also has its own explanation in the bot's **Settings** tab (**See
more**).

## Settings worth knowing
Each is a line or two in your settings file; all are off unless you write them.

**Switching a reward off:** `enabled = false` in a reward's block leaves it out of the run (not paid, not
measured) while keeping its settings for later; delete the line to switch it back on. `metrics.csv` shows
what each reward paid per step after its weight and zero-sum sharing (`paid_<reward>`) next to its raw value
(`raw_<reward>`); the dashboard's Rewards tab ranks them.

**A richer view for the bot:** `observation = "nitro_v2"` under `[env]` gives the bot everything the
standard view has plus the relative, goal-aware and speed numbers the strongest public bots use (148
numbers in 1v1 instead of 112). **The presets use it**: on the shot practice task
bots with it scored on 88% of shots against 53% with the standard view (every seed); on ball
touch and on saves it was no different, and no task measured worse. A bot made before keeps its
own view. Choose before starting a run: the view is fixed for a run.

**One bot for 1v1, 2v2 and 3v3:** `obs_team_size = 3` under `[env]` gives the bot's view a place for
every car of a 3v3 game, with empty places marked as empty in smaller games. The same
network then plays 1v1, 2v2 and 3v3, both in training and as an exported RLBot bot, and a run can start
in 1v1 and continue as 2v2 or 3v3 later: change `team_size` and resume. Any number from your
`team_size` up to 4 works; 0 (the default) keeps the view sized for one game size. It is off
in the presets because nothing has measured yet whether it learns as fast; it costs a few more numbers per
car.

**1v1, 2v2 and 3v3 in the same run (off by default, not proven yet):** list the sizes and every arena plays
one of them for the whole run, spread evenly, so each batch holds all three:
```toml
[env]
team_size = 3                       # the largest of team_sizes
team_sizes = [1, 2, 3]
team_size_shares = [0.5, 0.25, 0.25] # share of the arenas; left out = equal
obs_team_size = 3                   # room for the largest (or use an attention/transformer encoder)
shuffle_slots = true                # optional: cars take random places in the view each game
```
A 3v3 arena has six cars to a 1v1's two, so equal shares give 3v3 half of the steps; the log says each size's
share when the run starts. Steps per second still count every player's step, as always. metrics.csv gains
per-size columns (`1v1_episode_seconds`, `2v2_goals_per_minute`, `3v3_touches_per_minute`, ...). Self-play
works (past versions drive Orange in every size; one rating for all). Not yet with replay start states:
`nitro check-settings` says so. **Start
in 1v1** (with `obs_team_size = 3`) and add the bigger sizes on a resume once the bot touches the ball well:
Pulsar's authors found that training every size from scratch spread the first touches thin and the run
stalled. `shuffle_slots` helps a plain network learn that every place in its view can hold a car; attention
networks do not need it.

**Seeing where the ball will be:** see [Ball prediction](ball-prediction.md).


## Playing against past versions

With `enabled = true` under `[selfplay]` in the settings, one game in
five is played against a saved earlier version of the bot (a copy joins the pool every 20M steps; the
pool lives in `runs\my-first-bot\pool\`). The report then shows the bot's rating against its past
selves and how even the matches are. It is off in the presets: only full games could show whether it helps.
It is standard practice in self-play training.

## How strong is it getting?

A report card against scripted players (`nitro gauntlet`), a strength chart against a few pinned saves
(`[strength] anchors`), and the practice skills with the bot's weak spots, also on a schedule while it trains
(`[run] skills_every`): see [Measuring your bot](measuring.md).

## Before your first long run
These are fixed for the whole run (they shape the bot's networks), so choose them first. All are lines in
your copy of a preset.
- **Game sizes:** `obs_team_size = 3` under `[env]` if the bot may ever train at 2v2 or 3v3 (one bot for
  all three); leave it out for a bot that trains at one size. An attention network needs nothing
  here: left out, it gets room for the run's own team size, it plays any size anyway (see Attention networks), and its
  team size and room may change between stop and resume.
- **What the bot sees:** the presets' `observation = "nitro_v2"` under `[env]` (it scored on 88% of shots
  against 53%).
- **Network size and precision:** the presets' `[256, 256, 256]` trains at about 394,000 steps a second on
  the test machine. Bigger networks can learn more but train slower: `[1024, 1024, 1024, 1024]` runs at about
  140,000 in FP16 (`precision = "fp16"` under `[run]`; it learned as well as FP32 on every check) with
  `fused_norm = true` under `[model]` (LayerNorm and ReLU as one graphics-card routine, measured 11% faster
  at that size; recommended there). With the presets' networks `fused_norm` is no faster, so leave it off
  there.
- **Plain or attention network:** for 1v1, the plain network; attention is not for 1v1 (it learned
  slower there; the settings check says so). For 2v2 and 3v3, either the
  plain network (with `obs_team_size`) or attention (see Attention networks; the 2v2 and 3v3 presets hold its settings,
  commented out, at the recommended sizes: 2 blocks, 4 heads, width 128). Attention trains at about a
  fifth of the presets' speed. Its learning was compared only with no other car (slower) and in 1v1 (the
  same), never in 2v2 or 3v3, where it is meant to help; there the tests check that it works.
- **Playing past versions** (`[selfplay] enabled = true`) can be switched on or off at any time; it
  has not had a long-run check yet.

## Will an update break my bot?
No, from version 1.0 on. A run you start on a release keeps resuming, exporting and playing the same in
every later version: its settings keep their meaning, new features are off unless you turn them on, and
tests built from real saves of each release check this for every change. Choose these
before a long run, because they shape the bot's networks: how many players a side it may ever play
(`env.obs_team_size`), the observation (the presets use `nitro_v2`), the network sizes and the precision.

## Starting from a bot you already have

**Starting from a bot you already trained (transfer learning):** in the dashboard, **New bot from this one**
with its box ticked makes a bot that starts from this one's newest save, taking everything it learned (the
copy keeps its rewards). The new bot's Settings tab opens with **Transfer learning** at the top: a switch (on
or off), which bot and which save, and what to take (everything, or the policy only with a fresh value
network). Change any of it before the first start; if you change the rewards first, the dashboard switches it
to the policy only and says so. The transfer happens once, at the first start: from then on the bot trains its
own networks (its saves), so every later start carries on from those, and the section shows where it started,
read-only. From a terminal, `.\nitro train my-settings.toml --from my-first-bot` starts a
new run from that run's newest save (`--from-at 50000000` for an older one); or write it in the settings:
```toml
[start_from]
enabled = true                # false: start from scratch, keeping run for later (also over --from's)
run = "my-first-bot"
at = 0                        # which save, by its agent steps; 0 = its newest when this bot starts
parts = "policy"              # "all" (default): policy, value network and scaling, for the same rewards;
                              # "policy": a fresh value network, for new rewards; "matching": the policy
                              # whole and every part of the value network that fits, for the same rewards
                              # with another value network, e.g. a lighter critic
```
Both runs need the same network sizes and observation. To move a bot from 1v1 to 2v2 or 3v3, train it
with `obs_team_size = 3` from the start; then a 3v3 run can start from it.

## Attention networks (for 2v2 and 3v3)
For 2v2 and 3v3 only, never for 1v1. `model.encoder = "attention"` puts an attention encoder in front of
the policy and value layers: the
bot's own car looks at every other car through one shared set of weights, so what it learns about one
opponent carries to the others and the order of the cars does not matter. Expect a gain in 2v2 and 3v3
rather than in 1v1. It is new and not proven yet; `nitro export`
makes RLBot bots from it as from any run. Its size is set by `attention_width` (128),
`attention_blocks` (2) and `attention_heads` (4); with the presets' [256, 256, 256] layers that makes
about three times the parameters of the plain network. **It is slow to train:** about 81,000 steps a
second on the test machine, about a fifth of the presets' plain network's speed and under half of a plain
network of its size (197,000); the bot itself stays fast (0.14 ms per decision). **Not yet proven where it should help:** it has been
checked only where it has little to do: ball touch has no other car (it learned slower than a plain
network of its size) and kickoffs are 1v1 (no difference). It has not been compared in 2v2 or 3v3, where
the other cars are what it is for; there the tests only check that it works. Options, all off: `attention_skip = true` also gives the
policy and value layers the ball's, your own car's and the pads' numbers and the last action directly (its bots need this version's
`nitro_bot`), `[ppo] encoder_lr_scale` lets the encoder learn faster or slower than the layers after it, and
`attention_fast = true` does the same maths in fewer, bigger graphics-card steps (same answers and saves) and trained 10%
faster: recommended whenever you use attention.

**Attention with talking (`encoder = "attention_communication"`):** a separate network kind. First the other
cars, the ball and your own car "talk": each looks at all the others, so the network can work out how they relate (who
is nearer the ball, who covers whom); then your car looks at them as plain attention does. It is all one bot's thinking:
nothing is sent between bots, so teammates still coordinate only through what they see. Much slower to train: 31% of
plain attention's speed; the bot stays far inside RLBot's time per decision (0.48 ms, 6% of a tick).
Its bots need this version's `nitro_bot` (bot file format 4).
**Seeing what it pays attention to:** with either attention network, `nitro watch` draws yellow lines from the car you
follow to what its network looked at (thicker means more) and lists the shares on its card; with the talking network,
pale lines show which car (or the ball) each one listened to most. View > Attention lines hides them.

**Any number of cars, like Nexto.** An attention network uses one shared set of weights for every car, so
its weights do not depend on how many cars there are. Training still needs a fixed-size observation: the
simulator writes every game's numbers into one block of memory of a fixed length, the networks run on
whole batches of them at once, and the rollout memory of a run is made once for that length. So each car
gets a place of its own, marked present or empty, and the room (the number of places) is fixed while a run
trains. That is all padding was for, and an empty place changes nothing: the RLBot bot's answer for the
same game is the same to the bit with room for 2 or 3. An attention bot without `obs_team_size` gets room for its own team size
by itself, and at play time it reads however many cars there are: the RLBot bot gives the observation one
place per car in the match (RLBot's 12v12 included), and `nitro watch --team-size` and `nitro eval
--team-size` play it at 1v1 to 4v4 (the simulator's limit). At the sizes it trained at it answers exactly
as with a fixed room; beyond them it works, but how well it plays there is not measured. To train it at another size,
stop it, change `team_size` (and `obs_team_size`, if written) in its settings and resume, or start a new
bot from it with `--from`.

**The transformer (`encoder = "transformer"`, not proven yet):** a standard transformer. Your car, the ball, every
other car and, with `transformer_pads = true`, each of the 34 boost pads are tokens; in each block every
token looks at every other one, then thinks on what it saw, and the policy and value layers read your car's
token (or, with `transformer_readout = "summary"`, one extra learned token). Sizes: `transformer_layers`
(2), `transformer_width` (128), `transformer_heads` (4), `transformer_ff` (512); `shared`, `attention_skip`
and `[ppo] encoder_lr_scale` work with it as with attention. It uses PyTorch's fused attention on the graphics
card. Its speed and learning are not measured: it is checked for working only. Its bots need this version's
`nitro_bot` (bot file format 6); `nitro watch` lists what it looked at ("Boost pads" for all pads).

## Bigger networks without starting over
```powershell
.\nitro surgery my-first-bot --widen 2          # makes my-first-bot-x2; then: .\nitro resume my-first-bot-x2
```
Makes a new run from the newest save with every hidden layer twice as wide (`--widen 3` for three times),
built so that it plays like the old bot at first (the networks compute the same numbers to within the
computer's rounding, checked before anything is written) and then keeps learning with the bigger networks.
The old run stays as it is; the chosen save and the self-play pool are widened into the new one. A fresh
run of the same length usually ends up stronger than a widened one, so use this when the run so far is worth
more than a restart.

## Comparing two settings
`nitro ab` trains a fresh bot with each of two settings files on the same seeds and tells you in plain
English whether one is better, allowing for the luck between seeds:
```powershell
.\nitro ab a.toml b.toml --task ball_touch     # learning, on a short practice task (30 min to 2 h)
.\nitro ab a.toml b.toml --speed               # training speed (a few minutes; leave the PC alone)
```
Practice tasks: `ball_touch` (touch a ball at rest), `recovery` (land on the wheels), `boost_pickup`,
`kickoff`, `shot` (empty net), `save`, `aerial_touch` (a ball hanging in the air), `touch_mix` (half
ground touches, half aerial ones, for settings about mixing kinds of starts). On a task only the settings
for how the bot learns count ([model], [ppo], precision, the observation, and how the training
data is arranged); the task provides the rest. Add `--with-rewards`
and the two files' `[[rewards]]` replace the task's, so two reward designs can be compared on the same skill
(still judged by the task's measure, never by the reward). `--also-likeliest` also scores each trained bot
taking its likeliest move, as the RLBot bot plays (the sampled score still decides); `--likeliest-decides`
makes that score decide, written into the comparison's plan before the first run. Use one of them for
settings that change how random the bot is: the sampled score alone favours the less random one.
The answer is "B is better", "B is worse", "borderline" or "no measurable difference", with every seed's
result listed. A practice task shows whether a setting helps the bot learn *that* skill; your own full
training is the real test. If the comparison stops, run the same command again to carry on. Add
`--checkpoints 4` to also score each bot at a quarter, half and three quarters of its training and see which
one learns faster (the learning curve and its average, scored sampled as in training and shown for
information; the final score still decides). It goes with `--also-likeliest` and
`--likeliest-decides`, which score only the trained bot.
