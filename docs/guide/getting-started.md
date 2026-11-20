# Getting started

What NitroLearn needs, how to install it, how to keep it up to date and how to remove it. To make
your first bot once it is installed, see [Your first bot](first-bot.md).

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
When a new version is out, the dashboard says so at the top of the Garage (and with a yellow dot on **App
settings**): it asks GitHub for the newest version number when it opens, at most once a day, and sends nothing
about you. **Not now** hides the notice until the next version; to stop the check, turn off **Check for new
versions when NitroLearn opens** in **App settings**. You can also check there any time with **Check for
updates**. **Update now** brings the new version in and runs the installer again with the parts you chose
before, without asking anything; on Windows, NitroLearn closes, a window shows the installer working, and
NitroLearn opens again when it is done. From a terminal: `.\nitro update` (Linux: `./nitro.sh update`), or
`.\nitro update --check` to only look. Training must be stopped first. If you have your own Rust rewards and
Rust in the folder, the installer builds them in again by itself.

Everything of yours stays: your bots and their saves (`runs`), exported and packed bots, replays,
`my-settings`, the arena meshes and the tool folders. Files NitroLearn itself no longer has in the new
version are removed; nothing of yours is. A file of NitroLearn's that you changed is replaced, but first
copied to `.tmp\update\replaced\<old version>`. A folder made with `git clone` is moved to the release
with git instead, and only while you have no changes to its code.

An update from a version up to 1.0.2 installed with NitroLearn-Setup may list nearly all of NitroLearn's
files as changed by you (only their line endings differ): the files listed are just backup copies, and if
you never edited NitroLearn's own files you can ignore the list.

Every update cleans up after itself: what the old version installed that the new one no longer needs (Python
packages, an older Python, old downloads and caches) is removed, so you never have to uninstall first.
Python packages you installed yourself stay.

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

**Which PyTorch?** The installer asks the NVIDIA driver which card you have (it finds the driver's
`nvidia-smi` even when it is not on the PATH): most cards get the CUDA 12.6 build (the one NitroLearn's speeds
were measured with), RTX 50 cards get CUDA 13.0 (CUDA 13.2 if PyTorch's site lacks that), and a PC without an
NVIDIA card gets the build for the processor. A driver too old for the build a card needs is named, with how to
update it; a card Windows lists without a working driver gets the processor build and a note to install the
driver first. If PyTorch's site has no build that fits, the installer tries the next one that does and, only
as a last resort, installs the processor build and says so. To pick another build, run the installer from a
terminal in the folder with `-Torch` and one of `cpu`, `cu126`, `cu130` or `cu132`, for example
`& ".\Install NitroLearn.cmd" -Torch cu130` (Linux: `bash install.sh --torch cu130`). The choice is
remembered.

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
See [Troubleshooting](troubleshooting.md#installing) for what to do when the install stops, a download
fails, or something does not work afterwards.
