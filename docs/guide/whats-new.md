# What was new in earlier versions

The newest version's changes are at the top of NitroLearn's README.

## What's new in 1.0.6
- **Rewards without code (new):** in a bot's Rewards tab, **Add reward**, **Build your own (no code)** puts a
  reward together from ready-made pieces: conditions such as "in the air" or "ball above 300", measures such
  as "speed toward the ball" or "a goal", and weights. Nothing to install or compile.
- **Formulas and memory, for when the pieces are not enough:** a reward can pay a formula you write, such as
  "the ball's speed towards their net while I'm touching it", and remember things from one moment to the
  next: how long the car has been in the air, how many touches in a row, or "a flip reset, then a shot within
  2 seconds". **Start from a template** gives ready-made rewards to begin from.
- **Checked before you train:** NitroLearn checks every built reward before training starts and names any
  mistake with its place; numbers too big for the game's maths and formulas long enough to crash it are
  refused with a plain message. **Test this reward on a recorded game** shows what it would have paid in a
  real game. See [Building a reward without code](../reward_builder.md).

## What's new in 1.0.5
- **Bring an RLGym-PPO bot over (new):** like a GigaLearn bot, a bot trained with RLGym-PPO can keep training
  here as the same bot. See [Bringing an RLGym-PPO bot over](rlgym.md).
- **See what each reward paid (new):** `nitro watch` with `--rewards` shows under the game what every reward
  paid, step by step. Pause on a moment the bot got wrong and press Practice this moment to add it to the
  bot's training starts. `nitro score-replay` pays the players of a real game with your bot's rewards, so
  you can see what your rewards make of good play.
- **Download replays (coming soon):** `nitro get-replays` (or Resources, Download replays) will download
  high-level 1v1 or 2v2 games, the most skilled first and only as many as you ask for, and convert them. The
  replays are not online yet; until they are, it says "Replay downloads aren't available yet".
- **Measuring your bot (new, off by default):** `nitro gauntlet` plays your bot against scripted players
  and gives it a report card; a strength chart follows it against fixed opponents as it trains; skill checks
  can run every so often during training, with a list of its weak spots and practice drills for them. See
  [Measuring your bot](measuring.md).
- **Training carries on by itself (new, off by default):** with **Start again by itself after a crash or
  restart** on (Settings, Saving), training starts again from its newest save after a crash or after the computer restarts. Two new limits keep long runs tidy:
  how much disk the saves may take, and how big the run's numbers file may grow.
- **A handbook:** the README's long explanations are now a handbook in `docs/guide`, one topic a page, which
  also opens in the dashboard's Help page. The Help page can copy a report for asking on Discord, with your
  user name and folder names taken out. A training coach in the bot's page gives a plan and hints when
  training goes wrong.

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
