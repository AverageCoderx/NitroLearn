# Reading the dashboard

The dashboard is a window for doing everything without a terminal: the Start menu's **NitroLearn**, or
double-click `Shortcuts\Open the dashboard` (or type `.\nitro dashboard`). Every button that has a matching command shows it, so you can learn the commands as you go.

## The Garage

**The Garage page** shows the bot that is training on top, with its steps, speed and saves and the two buttons
you need most (**Stop and save**, **Watch it play**). Under it a short **Getting set up** checklist (this PC,
the arena files, your first bot, where RLBot is) that goes away once everything is done, and then every other
bot as a row with the one button that fits it (Start training, Carry on, See progress, Fix settings). Each bot
has a plate (two letters or digits, like 07 or AB) and two colours, picked from its name, so you know it at a glance; click its number plate on
its page to change them.

## Making a bot

**+ New bot** is four steps: the kind of bot (a 1v1, 2v2 or 3v3 starter, the
[GCO starter](first-bot.md#your-first-training-run) (an untested design that learns from goals alone), learning from your
replays first, or starting from one of your bots), its name with its race number and colours, its brain size
(four sizes with their parameter counts, or Custom for every detail), and a review that lists what is fixed
after its first save. The bot is made but not started, so you can set up its rewards first.

## Finding anything

**The search box** in the top band (Ctrl+K) finds bots, pages, the open bot's settings (by their plain name,
by their name in the file such as `ppo.policy_lr`, or by a word of their explanation) and actions.

## A bot's page

- **The header:** start or carry on training, stop it (it saves first), **Watch it play** and **Make RLBot
  bot**. Under it four steps (set up, training, watch it play, play it in RLBot) show how far the bot is, and
  a **Next** board says the most useful next thing to do.
- **Overview:** charts of its reward, speed, randomness, value prediction, touches and game length, each with a
  line saying which way is good; point at a chart to read a value, and switch between all of its training, the
  last hour and the last 10 minutes. [Reading the charts](#reading-the-charts) below says what each one tells you.
- **Coach:** a plan in four stages (basics, shooting, air play, duels and reads) read from its skill checks, with "ready for
  the next stage" when it gets there, and hints when its training looks flat, farmed, too certain or held
  back by its learning rate. Suggestions only; see [The training coach](coach.md).
- **Tools** (beside the charts and as a tab), each with one line saying what it does: test the newest save
  against an older one, a skill check (nine short drills), compare two settings, make a wider copy of the bot
  (**Change the network's size** makes a new bot and leaves this one alone), make its RLBot bot, check a
  settings file, make a new bot from this one, pack it for another computer, its progress report and its
  folder. Results show in the tool and its output goes to the Console tab. One heavy tool runs at a time, and
  never beside that bot's own training.
- **Watch it play** opens the 3D match viewer inside the dashboard window. **Watch live:** the newest save
  plays on and on, and while the bot trains each new save takes over (it runs on the processor, not the
  graphics card). **Record a game:** 30 seconds to 5 minutes, at any game size the bot can play, with the
  newest or an older save. The last five recordings are kept in the bot's `recordings` folder, and `watch.html`
  in the bot's folder is the newest one as a page you can pass on (there it draws a generated arena).
- **Rewards:** every reward with a search box: switch it on or off (an off reward stays in the settings for
  later), its weight and own settings, zero-sum with team spirit, a note, **See more** and a **Nerd
  explanation** (the block for the settings file), and **+ Add reward** to pick another from the list with
  what each pays for. Beside them, **what each reward pays**: each reward's share of all reward paid over the
  last minute (or 10 minutes, or hour) of training, ranked, with a warning when one reward makes up over 90%
  (the bot may be farming it) or its share at least doubles.
- **Settings:** in plain-named groups with **Basics** first (the few settings most people change); **Show
  advanced** reveals the rest, and a search box looks through all of them. Every setting has its explanation,
  **See more**, a **Nerd explanation** (its name in the file, type, default, limits, and the exact line for
  the settings file with the commands to check it and carry on), **Back to default** and a note you can write.
  Self-play has its own on/off card. A change waits (marked, with what it was) until you press **Apply**:
  change as many as you like, then one Apply writes them all. If the bot is training, Apply saves it, stops it
  and starts it again with them; nothing it learned is lost. Settings its trained networks depend on (sizes,
  team size, observation) are marked fixed: make a new bot from this one to change them.
- **Console:** what its training and tools print, live. **Notes:** anything to remember about it.

Training started from the dashboard keeps going when you close the window; open it again to see it (and bots
started from a terminal or a shortcut show up as training too). One dashboard is open at a time: starting it
again brings the open window to the front. Dark by default; a button switches to light.

## The top band

**Check this PC** (in the top band) runs the same checks as `nitro doctor` and says how to fix what is wrong.
**Open RLBot** starts RLBot, and the first time says what RLBot is for and asks where its program is. **App
settings** has the look, where RLBot is, and an About block with the version and the licences of what comes
with NitroLearn. **Help** holds this handbook, a **Copy a report for Discord** button, and where to ask for help, report a
bug or suggest an idea (the Vanta Discord server). Nothing in the dashboard listens on a network port.

## Resources

**Resources** (in the top band) is the library: **Add a file…** takes a settings file, a reward set (a file
with only `[[rewards]]` blocks) or a Rust file, recognises which it is (and says why when NitroLearn
cannot use one: C++ files and Python files are refused with the reason), and remembers where it
is. Settings files, with your files in `my-settings` and the presets, are starting points in the new-bot
form; reward sets load into a bot from its Rewards tab, which can also save its rewards as one (into
`my-settings\reward-sets`). Notes can go on every file and on the built-in rewards, observations and the
action table. **Rewards of your own** are written in Rust (`docs/your_own_rewards.md` shows how, starting from an
example) and compiled into NitroLearn by the Resources page's **Build** button (or `.\nitro rebuild`,
a few minutes); then they are in every bot's Add reward list. **84 more rewards come with it**, in every
bot's Add reward list, and 8 reward sets are in Load a reward set; `docs/rewards.md` lists them all with
where each came from. Your own observation builders and action parsers come later.

## Reading the charts

Each chart on the Overview tab has a line saying which way is good. What they tell you, in short:

| Chart | What it shows | What to look for |
|---|---|---|
| Reward | what the bots are paid per step | rising early on; flat later is normal in self-play (it plays a better opponent too) |
| Speed | agent steps a second | steady; a sudden drop means something else is using the PC or the card |
| Randomness | how random its choices are (1 any move, 0 always the same) | falling slowly; near 0 it has stopped exploring |
| Value prediction | how well it predicts how a moment will go (1 perfect, 0 no better than the average) | rising towards 1; below 0 for long is a warning |
| Touches | ball touches per car and minute | rising in the first hours |
| Game length | how long its games last | falling as it learns to score; rising a lot can mean farming |

The console warns in plain words when one of these goes wrong, and the [Coach](coach.md) tab reads them for
you.

## Following a run
The report (`report.html`) and `metrics.csv` show how learning goes; two more ways to look:

**What the bots do:** besides the rewards, the report charts what the bots actually do each iteration:
ball touches per minute, time in the air, speed, speed toward the ball, boost held, and time spent
supersonic or demolished. If the rewards climb while these look wrong, the bots have found
a way to be paid without playing.

**The console report:** with `console = "report"` under `[run]` (in the starter presets), the console
prints a full report every 10 seconds while training: the
speed and where each iteration's time goes, the reward and each reward's part, how the policy and value
learn, what the bots do, self-play, the bot the run started from, the next save and new
warnings. A copy of replays in the large mode prints one for every round (replays learned, losses,
agreement with the players, speed, time left). `console = "line"` keeps the one line of before.

**Charts on Weights & Biases (optional):** add this to your settings file to also send the run's
numbers to wandb.ai, where you can watch them from any device and put runs side by side. It works for every
kind of training: from scratch, resumed, started from another run, and every copy of replays (the large
mode round by round, against the replays learned in all, so a copy carried on continues the chart; a copy
learned all at once and `nitro pretrain` pass by pass). It also sends the time each part of an iteration
takes and how far each update moves the networks:
```toml
[wandb]
enabled = true
project = "my-bots"
```
Make a free account at wandb.ai and log in once with `.venv\Scripts\wandb login` (W&B keeps your key on
this PC; it never goes in the settings file). `nitro resume` continues the same W&B run. Training is the
same with or without it, and if W&B stops working mid-run the run keeps training and `metrics.csv` keeps
every number. Without internet (or before you log in) the run goes on with W&B offline, its files kept
in the run folder (logged in but offline, W&B first tries for about 45 seconds); `.\nitro wandb <run>`
uploads them later, and also sends whatever a run made before W&B was on (its copy of replays and its
training); use it while that run is not training. `mode = "offline"` always works that way. While a run
sends, W&B's helper program talks to it through a port on this PC that only this PC can reach.
