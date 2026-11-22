# Playing your bot in RLBot

RLBot is a free program that runs bots in real Rocket League matches on your own PC, offline. NitroLearn's
bots are RLBot v5 bots. The installer can install RLBot v5 for you (or get it from <https://rlbot.org>);
Rocket League itself (Epic Games or Steam) must be installed too.

## From the dashboard

1. On the bot's page, press **Make RLBot bot**, then **Make it**. When it says the bot is ready, press
   **Open folder**.
2. Press **Copy folder path** (or, in the folder window, click the address bar at the top and copy it with
   Ctrl+C). NitroLearn-Setup's folder is inside AppData, which Windows hides in its file pickers, so pasting
   the address is the easy way to reach it.
3. Open RLBot v5 (the **Open RLBot** button in NitroLearn's top band starts it), add a bot folder and paste
   the address into the folder picker's address bar.
4. Put your bot on a team and start a match. RLBot starts Rocket League for you.

Make the RLBot bot again after more training: the folder holds the save it was made from, and making it
again brings in the newest.

## From a terminal

```powershell
.\nitro export my-first-bot
```
This turns the run's newest save into a bot folder, `bots\my-first-bot\`, with everything RLBot v5 needs:
the trained policy as a small file, the bot program (ready-made with the install), RLBot's settings
file, a README and NOTICES.md (the licence notices that go with the bot: keep it in the folder when you
share the bot). In RLBot v5 (https://rlbot.org), add that folder as a bot folder and the bot appears
under the run's name (`--bot-name "Nitro One"` picks another). The bot needs no Python and no graphics
card; it builds the exact observation it saw in training and takes its most likely action
(`--sample` in the folder's `bot.toml` makes it pick randomly in proportion to its confidence, as during
training). Export again after more training to refresh the folder; an edited `loadout.toml` (the car's
look) is kept.

In the dashboard this is **Make RLBot bot** on the bot's page; **Open folder** then shows the folder. To
reach it from RLBot's folder picker, copy the address from the folder window's address bar and paste it
into the picker's (NitroLearn-Setup's folder is inside AppData, which Windows hides). Rocket League (Epic
Games or Steam) must be installed for RLBot to start a match.

## Is it better than before?

In the dashboard, the Tools tab's test of the newest save against an older one does this. From a terminal:
```powershell
.\nitro eval my-first-bot                      # newest save against the run's oldest save
.\nitro eval my-first-bot --vs-at 50000000      # against this run's save at 50M agent steps
.\nitro eval my-first-bot --vs other-run         # against another run's newest save
```
The two saves play each other in every arena at once under the run's own settings (512 points from kickoff
by default; `--games` changes it) and you get Blue's share of the goals with a 95% interval, in plain
words: "clearly better", "clearly worse" or "no clear difference with this many points". The interval is
there so that the luck of a few hundred points is never mistaken for progress. The same `--seed` plays the
same games. The game's own measure is the real test; this is the quick one for between sessions. A run whose games
have no time limit (`terminals.max_episode_seconds = 0`) needs one for the match: `--max-seconds 300`
(a point lasting longer ends with no goal; the run's settings are not changed).

Both bots pick their moves as in training, at random in proportion to their confidence. The exported RLBot
bot always takes its likeliest move instead, which scored about 10 points of goal share more on a long run's bot.
`--as-bot` plays both sides that way; `--blue-temperature T` and `--orange-temperature T` set one
side (0 = likeliest, 1 = as training, in between = less random). The report says how each side picked and
how random each bot's policy was as trained (0 = always one move, 1 = any allowed move alike), whichever way
it picked: a side taking its likeliest move plays no move at random, yet still shows its policy's number.

The start states can favour one colour (one bot's eval read 69% as Blue and 52% as Orange), so for a number
to keep, play both ways round: `--both-ways` plays `--games` points with the run's save as Blue, then as
many as Orange from the same starts, and gives its share of the goals over both; `--log` adds the result to
`runs/<name>/evals.csv`.
