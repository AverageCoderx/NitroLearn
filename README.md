# NitroLearn

A fast Rocket League bot training framework for Windows (and Linux), built to train bots quickly and to run
them in RLBot v5. Training works, and a trained bot plays real matches in RLBot v5. More features are coming
in later versions.

Website and downloads: <https://nitrolearn.net>

**New here?** Read the quick start below, then the [handbook](docs/guide/README.md): one page per topic, in
plain words. The same pages open inside the dashboard under **Help**.

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
When you ask, paste the report from the Help page's **Copy a report for Discord** button: it holds
NitroLearn's version, what Check this PC found, the bot's settings and the end of its console, with your
user name and folder names taken out.

## What's new in 1.0.7
- **Multi-model bots (new, off by default):** make one bot out of several of your bots. A selector learns
  which of them should drive, moment by moment. How well it plays is not measured yet: compare it with your
  best single bot. In the dashboard: + New bot, then A multi-model bot from my bots. See
  [Multi-model bots](docs/guide/multi-model.md).
- **A faster dashboard:** a bot's page opens much faster and uses far less memory when its training run is
  long, because the dashboard no longer reads the bot's whole numbers file each time.

The changes in 1.0.6 and earlier are in [What was new in earlier versions](docs/guide/whats-new.md).

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

The [handbook](docs/guide/README.md) explains each part in more detail.

## Install
**Windows:** download
**[NitroLearn-Setup.exe](https://github.com/AverageCoderx/NitroLearn/releases/latest/download/NitroLearn-Setup.exe)**
and double-click it. **From the zip, or on Linux:** download **Source code (zip)** from the
[latest release](https://github.com/AverageCoderx/NitroLearn/releases/latest), extract it to a short path
(like `C:\NitroLearn`), and double-click `Install NitroLearn.cmd` (Linux: `bash install.sh`).

Everything goes inside NitroLearn's own folder: its own Python, PyTorch and the arena files, with no
system-wide changes. In total about 3.5 GB to download and 10 GB on disk with an NVIDIA card. The parts to
choose, updating, uninstalling and what the installer downloads are in
[Getting started](docs/guide/getting-started.md).

## The handbook
- [Getting started](docs/guide/getting-started.md): what you need, installing, updating, uninstalling.
- [Your first bot](docs/guide/first-bot.md): make a bot, train it, stop it and carry on.
- [Reading the dashboard](docs/guide/dashboard.md) and [the training coach](docs/guide/coach.md).
- [Rewards](docs/guide/rewards.md), [learning from replays](docs/guide/replays.md),
  [exploiters](docs/guide/exploiters.md), [ball prediction](docs/guide/ball-prediction.md),
  [measuring your bot](docs/guide/measuring.md), [multi-model bots](docs/guide/multi-model.md) and
  [more settings](docs/guide/more-settings.md).
- [Playing your bot in RLBot](docs/guide/rlbot.md), [sharing bots](docs/guide/sharing.md),
  [bringing a GigaLearn bot over](docs/guide/gigalearn.md) and
  [bringing an RLGym-PPO bot over](docs/guide/rlgym.md).
- [Troubleshooting](docs/guide/troubleshooting.md), [FAQ](docs/guide/faq.md),
  [glossary](docs/guide/glossary.md) and [all commands](docs/guide/commands.md).
- [What was new in earlier versions](docs/guide/whats-new.md).

## For developers
`& ".\Install NitroLearn.cmd" -Dev` in PowerShell (Linux: `bash install.sh --dev`) builds NitroLearn from
its code with the test tools, Rust in `.tools` included (on Windows it also needs Microsoft's C++ build
tools, see [What it does not install](docs/guide/getting-started.md#what-it-does-not-install)). `powershell -ExecutionPolicy Bypass -File tools\check.ps1` (Linux: `bash tools/check.sh`)
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
