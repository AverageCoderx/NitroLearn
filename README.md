# NitroLearn

A fast Rocket League bot training framework for Windows (and Linux), built to train bots quickly and to run
them in RLBot v5. Training works, and a trained bot plays real matches in RLBot v5. More features are coming
in later versions.

Website and downloads: <https://nitrolearn.net>

## Fair play: NitroLearn is not a cheating tool

NitroLearn trains bots for **offline play only**: [RLBot](https://rlbot.org) matches, private matches against
your own bots, bot tournaments, and research into how bots learn.

- **Never use a NitroLearn bot, or anything made with NitroLearn, in online matches against real players.**
  That includes ranked, casual, tournaments and any other online playlist. It is cheating, it ruins the game
  for other people, and it breaks Rocket League's terms of service.
- NitroLearn has no feature for joining online games, hiding from anti-cheat, or controlling the game for you
  online, and none will be added. Requests for such features will be refused.
- Bots made with NitroLearn are meant to run through RLBot, which starts local matches on your own PC.

If you see someone using a bot online, report them in the game.

## Psyonix and Epic Games

NitroLearn is a fan project. It is not affiliated with, endorsed by or sponsored by Psyonix or Epic Games.
Rocket League is a trademark of Psyonix LLC.

If Psyonix or Epic Games has any concerns about NitroLearn, please join this Discord server,
<https://discord.gg/heFPmxNzz8>, and mention **@gamingshorts** in it (rather than sending a friend request).
We will work with you to fix any problem.

## License

NitroLearn is free to use, change and share for **non-commercial** purposes. Selling NitroLearn, or a changed
copy of it, is not allowed. Bots you train with it are yours to do with as you like, as long as you never
use them to cheat online. See [LICENSE](LICENSE) and [BOTS-LICENSE.md](BOTS-LICENSE.md) for the exact terms.

## Help and support

Questions, bugs, or a bot to share: NitroLearn is supported in the Vanta Discord server,
<https://discord.gg/bYP3kYtryg>. The dashboard's **Help** page (in its top band) has the link too.

## What you need
- Windows 10 or 11 (64-bit), or Linux on a 64-bit Intel or AMD computer.
- An NVIDIA graphics card with 6 GB or more is strongly recommended. Without one, NitroLearn still works, but
  training runs on the processor, many times slower.
- About 10 GB of free disk space (about 2 GB without an NVIDIA card) to install, plus room for your bots'
  saves, and an internet connection while installing.
- To play your bots: Rocket League (from Epic Games or Steam) and RLBot v5, which the installer can get for
  you.

That is all. The installer brings its own Python and everything else NitroLearn needs, and it downloads the
arena shapes (the "collision meshes") for you, so you do not need Rocket League's files to train.

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

Everything below explains each part in more detail.

## Install
**Windows, the easy way:**
1. Download **[NitroLearn-Setup.exe](https://github.com/AverageCoderx/NitroLearn/releases/latest/download/NitroLearn-Setup.exe)**
   (from the [latest release](https://github.com/AverageCoderx/NitroLearn/releases/latest)).
2. Double-click it. NitroLearn-Setup is not signed with a certificate yet, so Windows may say **"Windows
   protected your PC"**: click **More info**, then **Run anyway**.
3. Follow the steps: the welcome, the licence and fair-play rules (**I accept**), the folder (the default needs
   no administrator password; a drive with lots of room, like `D:`, is fine), and the parts to install (the
   list below, with the same defaults). Setup then downloads and sets up each part, ticking them off as they
   finish, and **Finish** opens NitroLearn.

Setup itself is a small download: it carries NitroLearn's own files and uv, and downloads the rest while it
installs (about 3.5 GB with an NVIDIA card, mostly PyTorch; about 10 GB on disk). Everything goes into
NitroLearn's folder, as with the zip below. NitroLearn then appears in Windows' **Settings > Apps**, for
uninstalling.

**From the zip, or on Linux:**
1. **Get NitroLearn.** On the [latest release](https://github.com/AverageCoderx/NitroLearn/releases/latest)
   page, under **Assets**, click **Source code (zip)**: that is the code of exactly this release, which its
   ready-made files fit. Extract the whole zip (right-click it, **Extract All**). Put the folder somewhere
   with a short path and room to spare, for example `C:\NitroLearn` or `D:\NitroLearn`: very long folder
   paths can break the install on Windows, and a folder OneDrive syncs (often Desktop and Documents) causes
   trouble too. (The green **Code** button on the main page gives the newest code instead, which may not match
   a release and then needs build tools.)
2. **Run the installer.**
   - **Windows:** open the folder and double-click **`Install NitroLearn.cmd`**. If Windows warns you
     about a downloaded file, choose **More info**, then **Run anyway** (or **Run**). A window shows each
     step. Before downloading, it shows the optional parts to choose (below). It ends with a check of your
     PC and then "Done" (or what to fix), and opens the dashboard.
   - **Linux:** open a terminal in the folder and run `bash install.sh`.
3. **Start it.** On Windows, use the Start menu's **NitroLearn**, or double-click **Open the dashboard** in
   the `Shortcuts` folder. On Linux, run `./nitro.sh dashboard`.

The first install takes a while, mostly to download PyTorch. You can run the installer again whenever you
like: steps that are already done are skipped. To update, see **Update** below.

### Choosing what to install
Python, NitroLearn's core and programs, and the arena files are always installed. The rest is a list like
this (Windows; on Linux only 1, 5, 6 and 7, numbered 1 to 4). NitroLearn-Setup shows the same parts as
tick boxes on its "Select Components" page:

```
  [x] 1  PyTorch for the graphics card, <your NVIDIA card> (9 GB on disk; off: for the processor, 1.5 GB)
  [x] 2  RLBot v5, to play your bots in Rocket League (its own installer opens; a few MB)
  [x] 3  Start menu shortcut
  [ ] 4  Desktop shortcut (with the Start menu one)
  [ ] 5  Tools to build your own Rust rewards (Rust in NitroLearn's folder, about 1.5 GB)
  [ ] 6  Developer tools: tests, rustfmt and clippy (builds NitroLearn from its code, with Rust)
  [ ] 7  RocketSim's game recordings now (119 MB; else downloaded the first time a feature needs them)
```

Type a number to switch a part on or off, then press Enter to install. Without an NVIDIA card, line 1 says
so and PyTorch for the processor is installed. When NitroLearn has to be built from its code on your PC, Rust
is shown as required. With RLBot v5 installed already, its line says so (NitroLearn-Setup shows it greyed
out). Run the installer again and the list shows what is installed ticked: switch a part off to remove it
(RLBot v5 stays; remove it from Windows' Apps list), or on to add it. Nothing else is redone.

Switches set parts without the list (in a PowerShell terminal in the folder, for example
`& ".\Install NitroLearn.cmd" -Desktop -Recordings`): `-CPU`, `-RLBot` / `-NoRLBot`, `-Desktop` (Start menu
and desktop) / `-NoShortcuts`, `-Rust`, `-Dev`, `-Recordings`, or `-Parts` with exactly the parts wanted (for
example `-Parts gpu,start_menu`; NitroLearn-Setup passes its ticks this way). On Linux: `--cpu`, `--rust`, `--dev`,
`--recordings`. Without a window to answer in (`Install NitroLearn.cmd < nul`), nothing is asked: parts stay
as they are, and a first install makes no shortcuts and installs no RLBot unless a switch says so.

### Update
In the dashboard, open **App settings** and press **Check for updates**. When a new version is out, **Update
now** brings it in and runs the installer again with the parts you chose before, without asking anything;
on Windows, NitroLearn closes, a window shows the installer working, and NitroLearn opens again when it is
done. From a terminal: `.\nitro update` (Linux: `./nitro.sh update`), or `.\nitro update --check` to only
look. Nothing is checked in the background, and training must be stopped first. If you have your own Rust
rewards and Rust in the folder, the installer builds them in again by itself.

Everything of yours stays: your bots and their saves (`runs`), exported and packed bots, replays,
`my-settings`, the arena meshes and the tool folders. Files NitroLearn itself no longer has in the new
version are removed; nothing of yours is. A file of NitroLearn's that you changed is replaced, but first
copied to `.tmp\update\replaced\<old version>`. A folder made with `git clone` is moved to the release
with git instead, and only while you have no changes to its code.

Installed with NitroLearn-Setup? Running a newer NitroLearn-Setup into the same folder updates it too.

### What the installer downloads
Everything goes **inside the NitroLearn folder**:

| What | From | Where in the folder | Download | On disk |
|---|---|---|---|---|
| uv, which fetches Python and the packages (pinned, checked) | NitroLearn's release (else PyPI) | `.tools\uv` | 28 MB | 70 MB |
| Python 3.11, NitroLearn's own copy | NitroLearn's release (else python-build-standalone) | `.tools\python` | 25 MB | 100 MB |
| PyTorch, the learning library, for your graphics card | PyTorch's own site | `.venv`, `.cache` | about 3 GB or 0.2 GB | about 9 GB or 1.5 GB |
| The dashboard's window, Weights & Biases, the replay reader, NumPy | NitroLearn's release (else PyPI) | `.venv` | 100 MB | 350 MB |
| NitroLearn's core and programs (RLBot bot program, replay converters) | NitroLearn's release | `python\nitro`, `.tools\prebuilt` | 30 MB | 60 MB |
| The arena meshes (checked by fingerprint) | RLGym's public package on PyPI | `collision_meshes` | 80 KB | 200 KB |

NitroLearn-Setup carries uv itself. PyTorch's two figures are with an NVIDIA card and without one.

In total about **3.5 GB to download and 10 GB on disk** with an NVIDIA card, or about 0.5 GB and 2 GB
without one. The downloads stay in `.cache`, so running the installer again does not download them again.

**NitroLearn's release carries its own copies** of everything an install needs that can be passed on: uv,
Python, every Python package except PyTorch, the source code of every Rust crate (for building from the code
and your own Rust rewards, so no crates.io is needed) and RocketSim's game recordings. The installer takes
them from the release first and only falls back to the original sites when a copy is missing; each is
checked by its fingerprint wherever it comes from. Two things are never in the release: **PyTorch** (its
graphics-card builds are too big for a release file, and on Linux they need NVIDIA's own libraries) and **the
arena meshes** (they are Rocket League's game data: NitroLearn never hosts them). `LICENSES\BUNDLED.md` lists
every copy and its licence.

**Installing without internet** (except PyTorch and the arena meshes): download the release's files for your
system (and `SHA256SUMS.txt`) from the release page into one folder with `NitroLearn-Setup.exe` and run
it from there, or into a folder named `release-files` inside the NitroLearn folder or next to
`NitroLearn-Setup.exe`, and run the installer. It then downloads nothing else
(`bash install.sh --bundles <folder>` or `-Bundles <folder>` for a folder elsewhere). With PyTorch already
installed in the folder and the meshes already there, it needs no internet at all.

**Which PyTorch?** The installer asks the NVIDIA driver which card you have: most cards get the CUDA 12.6
build, RTX 50 cards get CUDA 12.8, and a PC without an NVIDIA card gets the build for the processor. To pick
another build, run the installer from a terminal in the folder with `-Torch` and one of `cpu`, `cu126`,
`cu128` or `cu130`, for example `& ".\Install NitroLearn.cmd" -Torch cu128` (Linux:
`bash install.sh --torch cu128`). The choice is remembered.

### What it does not install
These change Windows itself, so the installer leaves them to you. `nitro doctor` (the **Check this PC**
shortcut) says when one is missing and where to get it.
- **Visual C++ Redistributable** (Windows): PyTorch needs it. Most PCs have it, because games install it
  too. If not, get it from <https://aka.ms/vs/17/release/vc_redist.x64.exe>.
- **Microsoft Edge WebView2** (Windows): draws the dashboard's window. Windows 11 has it. If not, get the
  "Evergreen Bootstrapper" from <https://developer.microsoft.com/microsoft-edge/webview2/>.
- **The NVIDIA driver**: keep it up to date with the NVIDIA app or <https://www.nvidia.com/drivers>.
- **RLBot v5**: to play your bots in Rocket League. It is a separate program with its own licence, not part
  of NitroLearn. On Windows, it is part 2 of the installer's list, ticked when it is not found: the
  installer then downloads **RLBot's own official v5 installer** from RLBot's releases on GitHub, checks it is
  a whole installer file for RLBot v5, and opens it, so you go through RLBot's installer windows. RLBot
  installs itself in `%LOCALAPPDATA%\RLBot5`, outside NitroLearn's folder. Later, `.\nitro install-rlbot` does
  the same. You can also get it yourself from <https://rlbot.org>. On Linux, see <https://rlbot.org>: its
  installer is for Windows. NitroLearn's bots are RLBot v5 bots: the older RLBot v4 (RLBotGUI) does not run
  them, so NitroLearn never takes it for RLBot. It can stay installed next to v5.

**Only for your own Rust rewards** (most people never need this): they are built with Rust, which the
installer puts in the folder (`.tools`, about 1.5 GB) when you tick part 5 of its list or run it with `-Rust`
(`& ".\Install NitroLearn.cmd" -Rust` in a PowerShell terminal in the folder; Linux:
`bash install.sh --rust`). On Windows, building also needs Microsoft's C++ build tools, which the
installer never installs: get **Build Tools for Visual Studio 2022** from
<https://visualstudio.microsoft.com/downloads/> and tick **Desktop development with C++**. On Linux,
install your system's build tools (on Ubuntu: `sudo apt install build-essential`). The same applies when no
ready-made NitroLearn exists for your version or computer: then the installer builds it from its code,
installs Rust in the folder by itself, and stops with these instructions if the build tools are missing.

### Uninstall
**Installed with NitroLearn-Setup:** open Windows' **Settings > Apps**, find **NitroLearn**, and choose
**Uninstall**. It first asks whether to keep your bots and settings (`runs`, `my-settings`, `packs`, `replays`
and exported `bots`; kept unless you say No), then removes NitroLearn's shortcuts and its folder. RLBot is
never touched. Kept bots stay in the folder with a note, `Kept by NitroLearn.txt`: install NitroLearn into
that same folder again and they are all there.

**Installed from the zip:** double-click **Uninstall NitroLearn** in the folder, then delete the folder.
That is all: the installer changes nothing outside it, with no system Python, PATH or registry entry; the
only things outside are the Start menu and desktop shortcuts you chose in its list (Windows), and Uninstall
NitroLearn removes exactly those.
RLBot v5, if you installed it (also from NitroLearn's installer), is its own program and stays: remove it
from Windows' **Settings > Apps** (it is listed as "RLBot v5").
Your bots live in `runs` inside the folder, so copy that first if you want to keep them (or use **Pack for
another computer**).

A few things can still end up outside the folder, and stay after you delete it:
- If you log in to Weights & Biases (`wandb login`), W&B keeps your key in your user folder (`_netrc` on
  Windows, `.netrc` on Linux), where other programs can use it too. Delete that file to remove the key.
- Windows' own records: the file dialogs remember recently used folders, and Windows may keep crash reports
  or a note that you allowed `Install NitroLearn.cmd` to run.
- What RLBot, Rocket League and the NVIDIA driver store is their own.

Do not move the folder after installing: its private Python environment remembers where it is. If you
moved it, run the installer again and it repairs that.

### Troubleshooting
- **"Windows protected your PC"** when you open NitroLearn-Setup or `Install NitroLearn.cmd`: NitroLearn is
  not signed with a certificate yet. Click **More info**, then **Run anyway**.
- **The browser holds back the download** ("not commonly downloaded" or similar): open the browser's
  downloads list and choose **Keep** (in Edge, under **...**; in Chrome, **Keep** or **Download unverified
  file**).
- **A download fails or stalls:** antivirus web shields (HTTPS scanning) and company proxies often block
  downloads like these. Allow NitroLearn's folder in your antivirus (or pause its web shield while
  installing), then run the installer again. What was already downloaded is kept.
- **Not enough disk space:** the installer checks before it downloads and says how much room it needs. Free
  up space, or install on another drive.
- **Something does not work after installing:** press **Check this PC** at the top of the dashboard (or
  double-click `Check this PC` in the `Shortcuts` folder, or run `.\nitro doctor`). It checks everything and
  says how to fix what is wrong.
- **Finding NitroLearn's folder:** NitroLearn-Setup puts it in `%LOCALAPPDATA%\Programs\NitroLearn`, inside
  AppData, which Windows hides. Press the Windows key and R together, paste
  `%LOCALAPPDATA%\Programs\NitroLearn` and press Enter. The full log of NitroLearn-Setup's install is
  `install-log.txt` in that folder; the zip's installer shows everything in its own window.
- **Still stuck?** Ask in the [Discord](https://discord.gg/bYP3kYtryg), with `install-log.txt` and what
  Check this PC says.

## The dashboard (no typing at all)
Double-click `Shortcuts\Open the dashboard` (or type `.\nitro dashboard`): a window for doing everything
without a terminal. Every button that has a matching command shows it, so you can learn the commands as you go.

**The Garage page** shows the bot that is training on top, with its steps, speed and saves and the two buttons
you need most (**Stop and save**, **Watch it play**). Under it a short **Getting set up** checklist (this PC,
the arena files, your first bot, where RLBot is) that goes away once everything is done, and then every other
bot as a row with the one button that fits it (Start training, Carry on, See progress, Fix settings). Each bot
has a race number and two colours, picked from its name, so you know it at a glance; click its number plate on
its page to change them.

**+ New bot** is four steps: the kind of bot (a 1v1, 2v2 or 3v3 starter, the
[GCO starter](#your-first-training-run) (an untested design that learns from goals alone), learning from your
replays first, or starting from one of your bots), its name with its race number and colours, its brain size
(four sizes with their parameter counts, or Custom for every detail), and a review that lists what is fixed
after its first save. The bot is made but not started, so you can set up its rewards first.

**The search box** in the top band (Ctrl+K) finds bots, pages, the open bot's settings (by their plain name,
by their name in the file such as `ppo.policy_lr`, or by a word of their explanation) and actions.

**Each bot has its page:**
- **The header:** start or carry on training, stop it (it saves first), **Watch it play** and **Make RLBot
  bot**. Under it four steps (set up, training, watch it play, play it in RLBot) show how far the bot is, and
  a **Next** board says the most useful next thing to do.
- **Overview:** charts of its reward, speed, randomness, value prediction, touches and game length, each with a
  line saying which way is good; point at a chart to read a value, and switch between all of its training, the
  last hour and the last 10 minutes.
- **Tools** (beside the charts and as a tab), each with one line saying what it does: test the newest save
  against an older one, a skill check (six short drills), compare two settings, make a wider copy of the bot
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

**Check this PC** (in the top band) runs the same checks as `nitro doctor` and says how to fix what is wrong.
**Open RLBot** starts RLBot, and the first time says what RLBot is for and asks where its program is. **App
settings** has the look, where RLBot is, and an About block with the version and the licences of what comes
with NitroLearn. **Help** says where to ask for help, report a bug or suggest an idea (the Vanta Discord
server). Nothing in the dashboard listens on a network port.

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
| `Open the dashboard` | The dashboard window (above): everything in one place. |

Keep your own settings in `my-settings\` rather than editing the presets: updates replace the presets and
never touch `my-settings`. It stays on your PC only, so copy it somewhere if you want a backup. Everything
below can still be typed in PowerShell instead.

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

**Playing against past versions:** with `enabled = true` under `[selfplay]` in the settings, one game in
five is played against a saved earlier version of the bot (a copy joins the pool every 20M steps; the
pool lives in `runs\my-first-bot\pool\`). The report then shows the bot's rating against its past
selves and how even the matches are. It is off in the presets: only full games could show whether it helps.
It is standard practice in self-play training.

## Starting from something you already have
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
Training has not started: watch the copy play, try rewards on it (below), then start training with
`.\nitro resume <name>` (the dashboard: Start training). `--then-train` on `nitro train` or `nitro resume`
trains straight after the copy instead. It has been shown to learn the players' choices; whether it makes a
better bot than training from scratch is not proven yet. A copy stopped part way keeps what it has done and carries
on at the next start.

**Trying rewards on the copied bot before training:** guessing rewards that suit a bot copied from
replays is hard, so try them first. `.\nitro reward-check <name>` plays the bot as last saved, learning
nothing (its policy and value networks stay as they are, as with a learning rate of 0 for both), for a
million agent steps, and prints each reward's share of all the reward paid, what it pays a step and a minute
of play, and its value before the weight; `--rewards some-set.toml` tries other rewards (a reward set or a
settings file), `--steps` plays longer. Nothing of the bot changes; the result is kept in its
`reward_checks` folder. In the dashboard: the Rewards tab, Try these rewards, which tries the rewards as
shown there, changes not yet applied included. Change the weights until no reward drowns out the others and
each one you care about pays something, then Apply and start training.

**Thousands of replays (the large replay mode):** the way above holds every replay in memory at once, so
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
`env.team_sizes` is set (above); otherwise change `team_size` in the run's settings between stop and resume. With a graphics card and
`precision = "fp16"`, the copy learns in FP16 too.

**Learning from more replays, without training:** `continue_from = "<run>"` carries on another run's copy
with new replays, and `new_only = true` leaves out the replays a copy on this PC used before. Nothing in it
needs the internet.

**Copying recorded games by hand:** `.\nitro pretrain my-settings.toml recordings\ --name cloned` teaches a
new bot to choose what the players in recorded games chose, then a normal run continues from
it: `.\nitro train my-settings.toml --from cloned` with `parts = "policy"`. Recordings are RocketSim's
format: replays converted ahead of time with `.\.venv\Scripts\python -I tools\replays.py` (into
`replays\1v1` and so on), or `--rocketsim` for the five real-game recordings RocketSim ships (about 25
minutes, 1v1 to 3v3, 119 MB downloaded the first time; use `obs_team_size = 3` to learn from all of them). Converted replays and RocketSim's
own recordings place the controls one row apart: inside `replays\` conversions are read as such by
themselves; for conversions kept anywhere else add `--rows replicar`. `nitro pretrain` says how many it read each way. You can
`nitro watch` and `nitro export` a cloned bot like any run.

## Training on a rented computer
A bot can move to another machine with a bigger graphics card (a rented cloud machine, say), train there and come
back, carrying on from its newest save. **Pack for another computer** on its page (or
`.\nitro pack <bot>`) puts it in one file; on the machine, `bash tools/cloud_setup.sh` sets NitroLearn up,
`nitro unpack <file>` adds the bot and `nitro resume <bot>` trains it; back home, **Add a packed bot…** on
the dashboard's Garage page. The file names no folder of your computer: a path in the bot's settings or saves
goes in relative to NitroLearn's folder or the bot's, or by its name alone; W&B's own files stay home. The
arena files stay out too (they are Rocket League's game files): the other machine downloads its own.

## Playing your bot in RLBot v5
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

## Will an update break my bot?
No, from version 1.0 on. A run you start on a release keeps resuming, exporting and playing the same in
every later version: its settings keep their meaning, new features are off unless you turn them on, and
tests built from real saves of each release check this for every change. Choose these
before a long run, because they shape the bot's networks: how many players a side it may ever play
(`env.obs_team_size`), the observation (the presets use `nitro_v2`), the network sizes and the precision.

## Before your first long run
These are fixed for the whole run (they shape the bot's networks), so choose them first. All are lines in
your copy of a preset.
- **Game sizes:** `obs_team_size = 3` under `[env]` if the bot may ever train at 2v2 or 3v3 (one bot for
  all three); leave it out for a bot that trains at one size. An attention network needs nothing
  here: left out, it gets room for the run's own team size, it plays any size anyway (below), and its
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
  plain network (with `obs_team_size`) or attention (below; the 2v2 and 3v3 presets hold its settings,
  commented out, at the recommended sizes: 2 blocks, 4 heads, width 128). Attention trains at about a
  fifth of the presets' speed. Its learning was compared only with no other car (slower) and in 1v1 (the
  same), never in 2v2 or 3v3, where it is meant to help; there the tests check that it works.
- **Playing past versions** (`[selfplay] enabled = true`) can be switched on or off at any time; it
  has not had a long-run check yet.

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

## All commands
Run these from this folder:

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
| `.\nitro convert-replays FOLDERS` | Converts replays ahead of a later copy of replays, learning nothing. |
| `.\nitro wandb NAME` | Sends a run's charts to Weights & Biases (its copy of replays and its training). |
| `.\nitro dashboard` | The dashboard window: make, train, change, watch and export bots without typing. |
| `.\nitro update` | Updates NitroLearn to its newest release, keeping your bots, settings, replays and saves (`--check`: only says whether one is out). |
| `.\nitro rebuild` | Builds NitroLearn again with your own Rust rewards (`my-settings\rust-files.txt`, docs/your_own_rewards.md). |
| `.\nitro watch NAME` | A game played by the run's newest save, in the 3D match viewer (`--follow`: a new game after every save; `--live`: played and shown live in a window of its own until you close it). |
| `.\nitro eval NAME` | Plays the run's newest save against its oldest (or `--vs-at STEPS`, `--vs OTHER`) and reports the goal share with an interval; `--both-ways` plays each colour once from the same starts (`--log` keeps the result in evals.csv). |
| `.\nitro skills NAME` | Scores the run's newest save on the practice skills against each skill's bar; appends to `runs\NAME\skills.csv`. |
| `.\nitro export NAME` | Writes an RLBot v5 bot folder (`bots\NAME\`) from the run's newest save. |
| `.\nitro pack NAME` / `.\nitro unpack FILE` | Puts a bot in one file to train it on another computer, and adds it there (or back home). |
| `.\nitro surgery NAME --widen 2` | Makes `NAME-x2`, the same bot with networks twice as wide, to resume with bigger networks. |
| `.\nitro ab A.toml B.toml --task TASK` (or `--speed`) | Compares two settings files. |

Add `--help` to any command for its options.

## For developers
`& ".\Install NitroLearn.cmd" -Dev` in PowerShell (Linux: `bash install.sh --dev`) builds NitroLearn from
its code with the test tools, Rust in `.tools` included (on Windows it also needs Microsoft's C++ build
tools, see above). `powershell -ExecutionPolicy Bypass -File tools\check.ps1` (Linux: `bash tools/check.sh`)
runs every code check and test. How the parts fit together: `ARCHITECTURE.md`; benchmarks: `benchmarks/`.

Without a graphics card, training runs on the processor (many times slower) with `device = "cpu"` or
`"auto"` in `[run]`.
The collision meshes are Rocket League assets and never travel with the repository: `nitro fetch-meshes` puts
the copy from the official RLGym package (PyPI) in `collision_meshes/` (checked by hash; git ignores it).
Without them every test that simulates a game skips itself, and the check scripts print how many did.

The release's ready-made files come from `.github/workflows/release.yml` on a version tag (`v<version>`):
the core as an abi3 wheel per platform and the programs (`nitro_bot`, `nitro-replicar`, `replicar`) with a
`prebuilt.json` that records the code each was built from. The installer uses them only while that matches
the folder's code; otherwise it builds from the code.
