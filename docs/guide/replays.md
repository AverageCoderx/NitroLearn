# Learning from replays

Your saved Rocket League replays show real players at work. NitroLearn can use them two ways: **before
training**, a new bot first learns to choose what the players chose (a copy of their play), and **while
training**, a share of every update keeps copying the players. Neither needs anything extra installed.

## Before training: copy the players first

**Learning from your own replays first:** add this to your settings file (each preset has it, commented
out) and start a new run as usual with `.\nitro train` (the dashboard: **+ New bot**, **Learn from my
replays first**, then **Copy replays** on the bot's page):
```toml
[pretrain]
enabled = true
limit = 30                  # the newest 30 replays of this run's team size
passes = 10
value_warmup_steps = 10_000_000   # first train only the new value network, holding the copied play
```
The run finds your saved replays in the game's own folder (`Documents\My Games\Rocket League\TAGame\
DemosEpic`, or `Demos` for Steam; `replays = "D:/some folder"` for another), converts the ones of its team
size with Rolv's replicar (ready-made with the install; each replay is converted once and kept, compressed,
in NitroLearn's `replays` folder), learns to choose as the players
did, prints how often its first choice matches the player's on replays it kept back, saves that copy as the
run's first save (0 agent steps) and stops. Nothing else to install or run.
Training has not started: watch the copy play, try rewards on it (next part), then start training with
`.\nitro resume <name>` (the dashboard: Start training). `--then-train` on `nitro train` or `nitro resume`
trains straight after the copy instead. It has been shown to learn the players' choices; whether it makes a
better bot than training from scratch is not proven yet. A copy stopped part way keeps what it has done and carries
on at the next start.

## Try rewards on the copy

**Trying rewards on the copied bot before training:** guessing rewards that suit a bot copied from
replays is hard, so try them first. `.\nitro reward-check <name>` plays the bot as last saved, learning
nothing (its policy and value networks stay as they are, as with a learning rate of 0 for both), for a
million agent steps, and prints each reward's share of all the reward paid, what it pays a step and a minute
of play, and its value before the weight; `--rewards some-set.toml` tries other rewards (a reward set or a
settings file), `--steps` plays longer. Nothing of the bot changes; the result is kept in its
`reward_checks` folder. In the dashboard: the Rewards tab, Try these rewards, which tries the rewards as
shown there, changes not yet applied included. Change the weights until no reward drowns out the others and
each one you care about pays something, then Apply and start training.

## No replays of your own? Download some

**Coming soon:** the replays are not online yet. Until they are, the command and the dashboard say
"Replay downloads aren't available yet"; everything below works once they are.

`.\nitro get-replays --size 1v1 --count 1000` (the dashboard:
Resources, Download replays) downloads 1v1 or 2v2 ranked games played at Grand Champion and above, the most
skilled games first, and only as many as you ask for (about 1 MB each to download, about 10 MB once converted;
it checks the free disk space first). A stopped download carries on where it stopped, replays you already have
are never downloaded again, and they are converted straight away like `convert-replays`. They are saved in
`replays\dataset\1v1` (or `2v2`): put that folder in `replays = "..."` (see "Before training" above). No ballchasing.com key needed.
The replays are a copy of Rolv Arild's High-Level Rocket League Replay Dataset (games from ballchasing.com),
re-hosted for download: the dataset is Rolv Arild's work, not NitroLearn's;
`--list` shows how many can be downloaded.

## Thousands of replays

**The large replay mode:** the way above holds every replay in memory at once, so
it tops out at a few hundred. For an archive, add `batch_replays` and point `replays` at the folder:
```toml
[pretrain]
enabled = true
replays = "my-replays/2v2"
limit = 10_000              # how many replays in all
batch_replays = 200         # converted and learned from this many at a time
passes = 2                  # passes over each round's replays
```
Replays are converted several at once (half your PC's threads; `workers` sets it) into a compact form
kept in memory only, about 2 MB per 5-minute 1v1 replay, 6 MB for 2v2 and 13 MB for 3v3; after replicar, the
Rust core does almost all the rest of each replay's work, so the threads run side by side. Each batch is
learned from together with the one before it (`keep_batches = 2`) and then dropped, so memory and disk stay small however
many replays you ask for, and no recordings are written. Progress is saved after every batch: if you stop
it or the PC turns off, `.\nitro resume <run>` carries on where it was. Each round prints how many replays are
done, how often the copy picks the player's choice on replays it keeps back, and about how long is left.

The copy goes as fast as your processor converts replays: the graphics card learns a batch in a fraction of
the time and then waits for the next. By default (`converter = "nitro"`) NitroLearn converts with its
own build of the same replicar (ready-made with the install), which gives exactly the same states and
controls but skips writing the columns the copy never reads: 14% more replays a minute.
`converter = "replicar"` uses replicar's own tool.

## More accurate conversion (the timing fit)

When NitroLearn converts a replay, it re-simulates the game and then works out on exactly which tick each
jump, each update from the game's server and each throttle, steering, powerslide and boost change took
effect, so the simulated cars stay close to where the replay shows them. It is on for bots made with 1.0.4 or
later (`fit_timing = true` under `[pretrain]`, the large replay mode with NitroLearn's converter) and for
`.\nitro convert-replays`; converting takes about 2.5 to 3 times as long. Replays converted with and without
it are kept in separate folders and never mixed, and bots made before 1.0.4 keep using the replays they had.
Nothing is converted again unless you ask: `.\nitro convert-replays --reconvert`, or the tick box on the
Converted replays card (Resources page). For a big collection that takes hours (roughly 12 hours for 40,000
replays on a fast PC). `--no-fit-timing` converts without it.

## Every game size at once

**One bot for every game size, copied from all of them:** with room for 3 a side
(`obs_team_size = 3`), the copy can take 1v1s, 2v2s and 3v3s together: list a folder of each and the sizes,
and the folders are taken in turn, so every batch mixes them:
```toml
[pretrain]
replays = ["my-replays/1v1", "my-replays/2v2", "my-replays/3v3"]
team_sizes = [1, 2, 3]
limit = 7_500               # in all: about 2,500 of each
```
A bot copied this way can use an attention network with a shared trunk (`shared` with
`encoder = "attention"`: one encoder and the shared layers serve both networks), the `nitro_v2`
observation, a decision every 4 ticks, and a save every 200 million steps (`checkpoint_steps`, the newest 5
kept); `.\nitro resume <run>` trains on from the copy. Training plays one game size at a time unless
`env.team_sizes` is set ([More settings](more-settings.md)); otherwise change `team_size` in the run's settings between stop and resume. With a graphics card and
`precision = "fp16"`, the copy learns in FP16 too.

## More replays later

**Learning from more replays, without training:** `continue_from = "<run>"` carries on another run's copy
with new replays, and `new_only = true` leaves out the replays a copy on this PC used before. Nothing in it
needs the internet.

## By hand, from a terminal

**Copying recorded games by hand:** `.\nitro pretrain my-settings.toml recordings\ --name cloned` teaches a
new bot to choose what the players in recorded games chose, then a normal run continues from
it: `.\nitro train my-settings.toml --from cloned` with `parts = "policy"`. Recordings are RocketSim's
format: replays converted ahead of time with `.\.venv\Scripts\python -I tools\replays.py` (into
`replays\1v1` and so on), or `--rocketsim` for the five real-game recordings RocketSim ships (about 25
minutes, 1v1 to 3v3, 119 MB downloaded the first time; use `obs_team_size = 3` to learn from all of them). Converted replays and RocketSim's
own recordings place the controls one row apart: inside `replays\` conversions are read as such by
themselves; for conversions kept anywhere else add `--rows replicar`. `nitro pretrain` says how many it read each way. You can
`nitro watch` and `nitro export` a cloned bot like any run.

## While training: replays in training

A bot that already plays well can keep learning from the players while it trains against itself. After
each update a few extra learning steps copy the players' choices from your converted replays; they change
how the bot chooses, never how it judges a moment.

1. Convert replays of the bot's game size first: the **Converted replays** card on the **Resources** page,
   or `.\nitro convert-replays` with your replay folders.
2. On the bot's page, open the **Replays in training** tab, switch it on, keep the suggested share (0.2) and
   learning rate (3e-5) to begin with, and press **Apply** (or **Start training**).
3. The same tab shows how close the bot is to the players (the copy loss, lower is closer) and how often its
   first choice is the player's.

Use it late in a bot's training, when it has stopped improving or plays oddly against people; it can also
be on from the start. Without converted replays of its size the bot says so and trains as usual. Each
update takes a little longer.
