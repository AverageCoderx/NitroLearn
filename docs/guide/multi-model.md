# Multi-model bots

A multi-model bot is several of your trained bots in one. At every moment a small network of its own, the
**selector**, picks which of them (its **models**) drives each car, so a bot that is good at kickoffs, one
that is good at aerials and a full 1v1 bot can each play the moments they are best at. The selector learns
when to hand the car over from the games it plays. It is off unless you make one, and how well a
multi-model bot plays is not measured: try it, and compare it with your best single bot (`nitro eval`).

## Making one

In the dashboard: **+ New bot**, then **A multi-model bot from my bots** (it appears once you have two
bots; it is also under a bot's **Tools**). On that page:
1. Tick two to eight of your bots. Each shows what it is (its network, game size and observation); anything
   that would stop it from driving is shown in red, in plain words. A bot needs at least one save.
2. Choose the bot whose game, rewards and starts it plays: use a full bot that plays against opponents.
3. Choose the selector's size (small is usually enough: it only picks a model) and how long a choice holds.
4. Optionally **Train the models too**, so they learn the moments they are handed. The selector learns
   alone first (the page suggests 20,000,000 steps, a guess) so the models never learn from its early,
   clumsy choices.
5. Give it a name and press **Make the bot**, then **Start** on its page.

The bots you pick are copied into the new bot's saves; nothing changes in them, and the new bot never
needs their folders again.

## While it trains

Its **Overview** shows each model's share of the driving in the newest iteration (with a line at the
smallest share you aimed for), how often a car is handed over each minute and, with goal guessers on,
how often their guesses were right. The console report has a **Models** line with the same numbers.

## In the settings file

```toml
[selector]
specialists = ["kickoff", "aerial", "main-1v1"]   # its models: bots in the runs folder
min_hold_seconds = 0.5      # a choice holds at least this long
min_share = [0, 10, 0]      # optional: aim for at least 10% of the driving for "aerial"
train_models = true         # optional: the models learn too, pulled back toward where they started
models_start_after = 20000000
```
Every setting is in the dashboard's Settings tab under **Multi-model bot**, each with a See more text.
The networks in `[model]` are the selector's own.

## What it cannot do yet

A multi-model bot cannot start from another bot, copy replays (first or while training), use exploiters, train
several game sizes at once or use several graphics cards; the settings check says so. `nitro eval` and `nitro
watch` play it at the game size it trained at; in RLBot it also plays a smaller match when its selector has
room for it and every model can see it, and otherwise sits still and says why. The strength chart, skill
checks and the gauntlet skip it. Every model runs at every decision, so it trains and plays about as slowly as
all its models together.

## Playing it in RLBot

`nitro export` makes one RLBot bot of the selector and all its models (the bot program reads it like any
other). Its README names its models. A choice starts afresh at every kickoff.
