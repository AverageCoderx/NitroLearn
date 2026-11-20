# What was new in earlier versions

The newest version's changes are at the top of NitroLearn's README.

## What's new in 1.0.4
- **Bring a GigaLearn bot over (new):** a bot you trained with GigaLearn can keep training here as the same
  bot: its networks, what it sees, its actions, rewards and learning settings, from the step it had reached.
  In the dashboard: New bot, then From a GigaLearn bot. NitroLearn first checks that the new bot chooses
  exactly as the GigaLearn one does, and says what you have to port by hand. See [Bringing a GigaLearn bot over](gigalearn.md).
- **More accurate replay conversion (on for new bots):** after re-simulating a replay, NitroLearn now works
  out on which tick each jump, each update from the game's server and each throttle, steering, powerslide and
  boost change took effect. In a check on 200 high-level games, the simulated cars ended up less than half as
  far from where the replay shows them. Converting takes about 2.5 to 3 times as long.
- **Your existing bots and replays are left as they are:** bots made before 1.0.4 carry on with the replays
  they already use, and replays converted with and without the new timing are kept apart. Nothing is
  converted again unless you ask: `nitro convert-replays --reconvert`, or the tick box on the Converted
  replays card. For a big collection that takes hours (roughly 12 hours for 40,000 replays on a fast PC).
- **Fixes:** updating from the dashboard now adds everything the installer prints, the clean-up summary
  included, to install-log.txt in NitroLearn's folder, so you can read it after the window closes. A Start
  menu or desktop shortcut called NitroLearn that you changed to open something else is no longer replaced by
  installing or updating, nor deleted by uninstalling. The warning that exploiters.max_steps is not shorter
  than exploiters.every_steps now appears once when training starts, not again at every exploiter phase.

## What's new in 1.0.3
- **Exploiters (new, off by default):** every so often NitroLearn trains a separate bot whose only job is to
  beat yours, then mixes games against it into your bot's training, so your bot learns to fix its weak spots.
  Each exploiter stops being played once your bot clearly beats it (a share of goals you choose). See
  [Exploiters](exploiters.md) and the Exploiters settings group.
- **Updates clean up after themselves:** an update now removes what the new version no longer needs (old
  packages, older Python copies, leftover downloads and caches), skips steps that are already done, and
  finishes with one click. Your bots, saves, settings and replays, and Python packages you added yourself,
  are never touched. (The update to 1.0.3 itself still runs the older version's updater, so it can still ask
  for one more click and list files as changed; those are only backups. The new behaviour starts with the
  next update.)
- **Fixes:** updates no longer list every NitroLearn file as one you changed (from the next update on); the update notice shows plain
  text; resuming a bot that trains against its past versions now carries on exactly as if it had not stopped.

## What's new in 1.0.2
- **NitroLearn tells you when an update is out.** When the dashboard opens (at most once a day) it asks GitHub
  for the newest version; when there is one, the Garage says so, with what is new and an **Update now**
  button. Nothing about you or your PC is sent. You can turn the check off in App settings.
- **Ball prediction (new setting, off by default):** a bot can see where the ball will be a moment ahead, as
  Rocket League's own prediction shows it. Turn it on for a new bot under Settings, Games (advanced). It costs some
  training speed (about 15% on one test PC); whether it makes a better bot is not measured yet.
- **Bot plates take letters too:** a bot's plate shows one or two letters or digits (like 07 or AB), in the
  colours you pick.

## What's new in 1.0.1
- **RTX 50 cards install again.** 1.0.0 asked PyTorch's site for a CUDA 12.8 build of PyTorch 2.14.1, which
  does not exist, so setup stopped. RTX 50 cards now get the CUDA 13.0 build.
- **The installer finds the graphics card more reliably:** it finds NVIDIA's `nvidia-smi` even when it is not
  on the PATH, reads every card and the driver's CUDA version, and says plainly when the driver is missing or
  too old instead of quietly installing PyTorch for the processor.
- **PyTorch's download is sturdier:** a download that is cut off or stalls is tried again (what was downloaded
  in full is kept), and a build PyTorch's site does not have is skipped for the next one that fits.
- **Clearer messages when setup stops:** the setup says the real reason (for example "the connection was cut
  off while downloading torch" or "there is no torch==... to download") with advice that fits it.
