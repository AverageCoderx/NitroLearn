# Rewards

A bot learns by being paid. Each **reward** pays it for something: moving towards the ball, touching it,
the ball moving towards the other net, a goal. Each has a **weight**: how much that thing is worth next to
the others. The bot slowly learns to do whatever pays the most, so rewards decide what kind of player it
becomes.

The starter bots come with a proven mix, so you do not have to touch rewards for a first bot.

## The Rewards tab

On a bot's page, the **Rewards** tab lists every reward the bot has. For each one you can:
- switch it on or off (an off reward stays in the settings for later, but is not paid or measured);
- change its weight and its own settings;
- make it **zero-sum** (what one team gains the other loses), with **team spirit** in 2v2 and 3v3 (how much a
  car shares its teammates' pay);
- read **See more** (what it pays, when to use it and what to watch out for) and the **Nerd explanation**
  (its block for the settings file);
- write a note.

**+ Add reward** picks another from the list: the built-in ones, trick moves and many more, each with a line
on what it pays. [The rewards that come with NitroLearn](../rewards.md) lists them all with where each came
from. **Load a reward set** adds several at once, with weights chosen for one move.

Changes wait until you press **Apply**. If the bot is training, Apply saves it, stops it and starts it again
with them; nothing it learned is lost.

## What each reward pays

Beside the list, **what each reward pays** ranks every reward by its share of all the reward paid over the
last minute of training (or 10 minutes, or the last hour). This is the most useful view of rewards: a weight
alone does not tell you how much a reward actually pays, because that depends on how often the bot earns it.

What to look for:
- **One reward makes up over 90%** of everything paid: the dashboard warns. The bot may be farming it (below).
- **A share suddenly doubles**: the dashboard warns too. The bot found a new way to earn it; check that it is
  what you wanted.
- **A reward you care about pays almost nothing**: raise its weight, or the bot will ignore it.

## Farming: paid without playing

A bot is "farming" a reward when it finds a way to be paid that is not good Rocket League: driving in
circles next to the ball, juggling it in a corner, never letting a game end. Signs:
- one reward's share of the pay climbs far above the others;
- the reward chart rises while the charts of what the bots actually do (touches, speed, goals) stay flat or
  fall;
- games get much longer while the goal's share of the pay falls. The console warns when this happens.

What to do: watch it play (**Watch it play**) to see what it is doing, then lower the weight of the reward it
farms, or add a reward that pays for what it should do instead. The [training coach](coach.md) points out
these signs too.

## Try rewards before training

**Try these rewards** on the Rewards tab plays the bot as last saved, learning nothing, for a short while,
and shows what each reward would pay it, changes not yet applied included. Nothing about the bot changes.
It is the quick way to set weights so that no reward drowns out the others, especially for a bot copied
from replays ([Learning from replays](replays.md)). From a terminal: `.\nitro reward-check <bot>`.

## What each reward paid in a game

`.\nitro watch my-first-bot --rewards --moments` adds a
chart under the game: what each of the bot's rewards paid the followed car in every step (pay above the line,
charges below), moving with the game. Click a spike to jump to the moment just before it; click a reward's
name to hide it; R hides the chart. `--moments` keeps every car's exact state too: pause where the bot should
have done better, follow its car, open Practice (top right) and press Practice this moment. The page saves a
small file in your Downloads folder and shows the command that adds it to the bot's training starts, e.g.
`.\nitro practice my-first-bot practice-20250101120000.csv --weight 0.2` (a bare file name is looked for in
Downloads too; or straight from the recording:
`.\nitro practice my-first-bot --at 12.5 --car 0`). "Place a start by hand" in the same panel lets you drag
the car (and its nose, to turn it) and the ball on the field from above and type heights and speeds. The
moments go into `practice_starts.csv` in the bot's folder, and its settings get one `snapshots` start with that
weight; they are used from the bot's next start (`.\nitro resume`).

## What your rewards would pay a pro

`.\nitro score-replay my-first-bot "D:\Replays\some game.replay"` (a
replay converted with `.\nitro convert-replays`, or an `.rlpr` recording) pays every player of that real game with
your bot's rewards, step by step, and opens it with the same reward chart, plus a table of what each reward paid
each player. In the dashboard: Watch it play, then Score a real game. In the dashboard's own viewer, Practice this
moment adds the moment to the bot straight away.

## Rewards of your own

Rewards of your own are written in Rust and built into NitroLearn from the **Resources** page's **Build**
button. [Your own rewards, in Rust](../your_own_rewards.md) shows how, starting from an example. Most people
never need this.

## In the settings file

Each reward is a `[[rewards]]` block with its `name`, a `weight` and its own settings. `enabled = false` in a
block leaves that reward out of the run while keeping its settings. `metrics.csv` shows what each reward paid
per step after its weight and zero-sum sharing (`paid_<reward>`) next to its raw value (`raw_<reward>`).
