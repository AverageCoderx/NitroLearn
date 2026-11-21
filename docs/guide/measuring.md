# Measuring your bot

Three tools tell you how good your bot is getting, apart from its reward. They are for your own reading of
a run: a bot that does better on them is not proof that one setting beats another.

## The report card: `nitro gauntlet`

```
.\nitro gauntlet my-first-bot
```

Your bot's newest save plays from kickoffs against four scripted players that come with NitroLearn and never
learn, each driving every car of the other team:

- **chaser**: drives at the ball, lined up to hit it toward your goal, boosts and dodges into it.
- **goalie**: stays between the ball and its own goal, facing the ball; clears a ball that comes near.
- **kickoff**: rushes every kickoff with a dodge into the ball, then plays as the chaser.
- **aerial**: the chaser, plus a jump and a boost at a ball in the air in front of it.

It prints a line for each, such as `beats chaser 97% (92-99) of 100 goals`: your bot's share of the goals,
and the range it is 95% sure to lie in. "Beats" means even the low end of the range is above half; "loses to"
means even the high end is below; "even with" means the range holds half. The card is kept in
`runs\my-first-bot\gauntlet.json` and the newest one shows on the bot's Overview.

Options: `--at STEPS` (another save), `--opponents chaser goalie` (only some; `idle` is a player that does
nothing), `--games 200` (points against each; more narrow the range), `--seed` (the same seed plays the same
games).

## The strength chart: `[strength]`

```toml
[strength]
anchors = 4
```

With `anchors` above 0, a few of your bot's saves become **anchors**: the first save, then one every
`anchor_every_steps` (a billion by default). Anchors are kept for good (never deleted with old saves) and your
bot never trains against them. After each save, the bot plays a short match from kickoffs against every
anchor (`points` points each, 100 by default) and gets a rating on a fixed scale: the first anchor is 0, and
400 more means about ten goals for every one against. An anchor keeps the rating it had when it was pinned,
so the scale never drifts.

The rating is charted as **Strength against the pinned anchors** in the Self-play folder of the Overview's
charts, and `.\nitro strength my-first-bot` prints every rating with its 95% range. Training waits while the
matches play; `rate_every = 2` rates every other save to spend less time on them. The bot needs opponents.

## Skills, weak spots and drills: `nitro skills`

```
.\nitro skills my-first-bot --weak
```

Scores the save on the practice skills (touching the ball, landing, boost, a shot, a save, an aerial touch)
and three drills: **kickoff_duel** (kickoffs against the scripted kickoff rusher), **fifty_fifty** (a ball
between your bot and the scripted chaser, both the same distance away) and **wall_read** (alone: a ball rolled
off a side wall). The two drills against a scripted player are left out for a bot that trains without
opponents (`env.opponents = false`). Each skill has a bar; the drills' bars are what a scripted player reaches in the
same situations. Results go to `runs\my-first-bot\skills.csv`.

`--weak` also prints where the bot does worst: each skill's situations grouped by how they started (where
the ball was, how high and fast, how fast the car was going, which way it faced, how far away), the groups
worst first. Groups of under 20 situations are marked as mostly luck. It is saved in `weakness.json`.

To score the skills automatically while training, set `skills_every = 5` under `[run]` (every fifth save;
`skills_games` sets the situations per skill, 128 by default). Training waits while they play, and the
console says how long that took.
