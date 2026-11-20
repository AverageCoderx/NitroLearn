# Your first bot

## Quick start (Windows)
1. **Install:** download
   [NitroLearn-Setup.exe](https://github.com/AverageCoderx/NitroLearn/releases/latest/download/NitroLearn-Setup.exe)
   and double-click it. If Windows says "Windows protected your PC", click **More info**, then **Run anyway**.
   Accept the licence, keep the folder and the ticked parts as they are, and wait (about 20 minutes on a
   normal connection). **Finish** opens NitroLearn.
2. **Make a bot:** press **+ New bot**, choose the **1v1 starter**, give it a name, keep the suggested size,
   and make it.
3. **Train it:** on the bot's page, press **Start training**. Training goes on when you close the window;
   open NitroLearn again to see how it is doing, and press **Stop and save** to stop it.
4. **How long?** Roughly, with an NVIDIA graphics card: the first save comes a few minutes in, the bot starts
   chasing and touching the ball within the first hour, and it needs many hours (a day or more) of training
   before it is fun to play against. Without an NVIDIA card everything takes many times longer.
5. **Make its RLBot bot:** on the bot's page, press **Make RLBot bot**, then **Make it**. When it says the
   bot is ready, press **Open folder**.
6. **Copy the folder's address:** press **Copy folder path** (or, in the folder window, click the address bar
   at the top and copy it with Ctrl+C). NitroLearn's folder is inside AppData, which Windows hides in its
   file pickers, so pasting the address is the easy way to reach it.
7. **Play it:** open RLBot v5 (the **Open RLBot** button in NitroLearn's top band starts it), add a bot
   folder and paste the address into the folder picker's address bar, then put your bot on a team and start a
   match. Rocket League (Epic Games or Steam) must be installed; RLBot starts it for you.

The rest of this page explains each part in more detail. [Reading the dashboard](dashboard.md) explains
what the bot's page shows while it trains, and [Playing your bot in RLBot](rlbot.md) has more on step 5 to 7.

## Your first training run
```powershell
.\nitro train configs\presets\1v1_starter.toml --name my-first-bot
```
For team play use `configs\presets\2v2_starter.toml` or `3v3_starter.toml` instead (same learner and rewards;
their arena counts are guesses until measured). `configs\presets\gco_1v1_starter.toml` (the GCO starter) is
an untested design to try: a 1v1 bot that learns from goals alone (+1 for scoring, -1 for conceding), with the extra learning signals
that design needs; its comments explain each part. It is sized for a graphics card with 8 GB; the design's
full size, for cards with 12 GB or more, is written in it as a comment.
This starts a 1v1 bot from scratch with a preset that works as it is (a proven starting mix of rewards,
1024 games at once). Every optional feature (learning from your replays first, attention for 2v2 and 3v3,
starting from another bot, W&B charts)
is in the presets too, commented out with a line or two on what it does: remove the `#` in front of its
lines to turn it on. Everything about the run goes into `runs\my-first-bot\`:

| File | What it is |
|---|---|
| `report.html` | Charts of the run (speed, reward, how the bot's choices and predictions develop), plus any warnings. Open it in a browser; it updates every 2 minutes. |
| `metrics.csv` | Every number in the report, one row per learning update, for your own analysis. |
| `train.log` | A detailed log of every update, and every warning. |
| `settings.toml` | The run's own copy of its settings. |
| `checkpoints\` | Saves of the bot: the newest 5, plus one per billion steps. A save is made every 15 minutes. |

The console shows a progress line every 10 seconds. **Warnings** appear in plain English when something
looks off, saying what to change (for example "The bot has become almost certain of its choices ... raise
ppo.entropy a little"). Training stops by itself, with a clear message, only if the maths breaks (numbers
that are not numbers, or learning updates that keep failing).

**Stopping and continuing:** press Ctrl+C once: the run finishes its current update, saves and stops
(press it twice to quit at once without saving). Continue later, even after a reboot, with:
```powershell
.\nitro resume my-first-bot
```
A resumed run carries on exactly where its last save left off. **To change a setting** (say a reward
weight or a learning rate), stop the run, edit `runs\my-first-bot\settings.toml`, and resume. Settings
that the saved networks depend on (team size, network sizes, precision and a few others) can't change
mid-run; `resume` says which line and why.

## Double-click shortcuts (no typing)
The `Shortcuts` folder in the project has a file for each everyday job. Double-click one and answer its
questions (type a number, press Enter); it runs the right `nitro` command for you and the window stays open
until you press a key. To put one on your desktop: right-click it, Show more options, Send to, Desktop
(create shortcut).

| Double-click | What it does |
|---|---|
| `New settings file` | Copies a preset into `my-settings\` under a name you choose and opens it in Notepad; it is checked when you close Notepad. |
| `Edit a settings file` | Opens one of your `my-settings` files in Notepad, then checks it. |
| `Train a new bot` | Pick the settings and a name, and it starts; its progress page opens after a few minutes. Ctrl+C once stops and saves. |
| `Resume a run` | Pick a run (newest first) and it carries on from its last save. |
| `Show training progress` | Opens a run's progress page. |
| `Watch a bot play` | Replays a game of a run's newest bot in your browser. |
| `Export a bot for RLBot` | Makes `bots\<run>`, the RLBot v5 bot of a run's newest save, and opens the folder. |
| `Check this PC` | `nitro doctor`: is everything installed and found? |
| `Open the dashboard` | The dashboard window ([Reading the dashboard](dashboard.md)): everything in one place. |

Keep your own settings in `my-settings\` rather than editing the presets: updates replace the presets and
never touch `my-settings`. It stays on your PC only, so copy it somewhere if you want a backup. Everything
on this page can still be typed in PowerShell instead.

## Watching your bot
**Watching the bot:** `.\nitro watch my-first-bot` plays a 60-second game with the newest save (the bot
on both sides) and opens it in your browser in the 3D match viewer: the arena, the cars, the ball and the
boost pads, with the score and the clock on top. Cameras: Broadcast (from the side, like TV), Ball cam and
Car cam behind any car, Top (the whole field from above) and Free (drag to look around, wheel to zoom).
Each car has a card with its boost and speed. Click a card to follow that car and see what its bot is
thinking: how good it thinks the moment is for its team (its value estimate), how sure it was of its move,
how random its choices are, and the controls it is using. The timeline under the picture shows every car's
value estimate, with goals, touches and kick-offs marked; click or drag it to jump. Letters on a car's card
name the mechanics it spotted (aerial touches, dribbles, flicks, double taps, flip resets, shots, saves,
demolitions). The View button switches name tags, attention lines, hitboxes and a minimap, and sets the
image quality: Low, Medium, High or Ultra, or Auto, which picks what runs smoothly. Press ? on the page
for every key and mouse action. You can use it while the run is training, and `--follow` keeps it going:
every new save plays a new game, and the open page shows it from its next game on. `--at 50000000`
watches an older save (the step counts are the folder names in `runs\my-first-bot\checkpoints`),
`--seed 5` another game, `--seconds 120` a longer one and `--fps 120` every physics tick.

The page (`runs\my-first-bot\watch.html`) is a single file that works offline, with nothing to install and
nothing listening. On this PC it draws the arena from your collision meshes, which it reads from one file
beside the runs (`runs\.viewer\arena.js`). The meshes are not inside the page, so you can pass the page on:
without that file it draws a generated arena instead. The recording itself is saved next to the page as
`watch.nlrec`.

**Watching live:** `.\nitro watch my-first-bot --live` plays the newest save on and on in real time and shows
the game as it is played, in a window of its own: a LIVE badge, the newest moment followed, the last two
minutes on the timeline. Pause or click the timeline to look back, and Go live (or L) takes you back to now.
Add `--follow` and each new save takes over from its next game. Closing the window stops it (so does Ctrl+C
in the console). Nothing is served on any port: the game is played inside the command and handed straight
to its window. The window needs pywebview, which the normal install brings for the dashboard.

## What can my bot do?
```powershell
.\nitro skills my-first-bot                     # newest save; --at 50000000 for an older one
```
The save plays each practice skill's situations (touching a resting ball, landing on its wheels, collecting
boost, an empty-net shot, a save, an aerial touch) with every other car parked (the opponent and, in 2v2 or
3v3, its own teammates, so the save's first car does the skill alone), and each skill is scored by
its own measure against a bar: the bar is what a fresh bot trained on that one skill alone reaches, so
"above the bar" means the skill is there ("under the bar" for a time, where lower is better). The rows are
kept in `runs\my-first-bot\skills.csv`, one per skill and save, so you can see skills arrive over a run.
`--temperature 0` scores the bot taking its likeliest move, as the RLBot bot plays; those rows go to `skills_likeliest.csv`, so `skills.csv` stays one series.
