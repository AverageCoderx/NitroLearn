# Troubleshooting

Find what you see, then follow what it says. If nothing here fits, press **Check this PC** in the dashboard's
top band first (it checks everything and names the fix), then ask in the Vanta Discord server
(<https://discord.gg/bYP3kYtryg>) and paste a report: the **Help** page's **Copy a report for Discord**
button puts NitroLearn's version, what Check this PC found, the bot's settings and the end of its console on
the clipboard, with your user name and folder names taken out.

## Installing

- **"Windows protected your PC"** when you open NitroLearn-Setup or `Install NitroLearn.cmd`: NitroLearn is
  not signed with a certificate yet. Click **More info**, then **Run anyway**.
- **The browser holds back the download** ("not commonly downloaded" or similar): open the browser's
  downloads list and choose **Keep** (in Edge, under **...**; in Chrome, **Keep** or **Download unverified
  file**).
- **A download fails or stalls:** antivirus web shields (HTTPS scanning) and company proxies often block
  downloads like these. Allow NitroLearn's folder in your antivirus (or pause its web shield while
  installing), then run the installer again. What was already downloaded is kept.
- **"Not enough disk space":** the installer checks before it downloads and says how much room it needs.
  Free up space, or install on another drive.
- **Setup stops with a message about PyTorch or the NVIDIA driver:** update the NVIDIA driver (the NVIDIA app
  or <https://www.nvidia.com/drivers>) and run the installer again. The message names the reason.
- **Finding NitroLearn's folder:** NitroLearn-Setup puts it in `%LOCALAPPDATA%\Programs\NitroLearn`, inside
  AppData, which Windows hides. Press the Windows key and R together, paste
  `%LOCALAPPDATA%\Programs\NitroLearn` and press Enter. The full log of the install is `install-log.txt` in
  that folder.

## Opening the dashboard

- **The window stays empty or does not open:** the window needs Microsoft Edge WebView2. Windows 11 has it;
  on Windows 10 get the "Evergreen Bootstrapper" from
  <https://developer.microsoft.com/microsoft-edge/webview2/>.
- **"The dashboard is already open":** only one dashboard runs at a time. Starting it again brings the open
  window to the front; look for it on the taskbar.
- **Something else after installing:** press **Check this PC** (or double-click `Check this PC` in the
  `Shortcuts` folder, or run `.\nitro doctor`).

## Training

- **It trains very slowly** (a few thousand steps a second): it is probably training on the processor.
  **Check this PC** says whether PyTorch can use the graphics card, and how to fix it if not (usually the
  NVIDIA driver, or PyTorch installed for the processor: run the installer again with the graphics card part
  ticked). Without an NVIDIA card, training on the processor is many times slower; that is expected.
- **The bot says "Settings problem" and will not start:** its settings file has a mistake. Its page shows
  which line and why; **Fix its settings** takes you there. From a terminal, `.\nitro check-settings` lists
  every mistake with its line number.
- **"The graphics card ran out of memory":** NitroLearn halves each learning step's data and carries on.
  To stop it happening, make the bot smaller or play fewer games at once (fewer arenas); the message names
  the settings. Close games and other programs that use the graphics card while training.
- **Training stopped by itself with "Training problem":** the maths broke (numbers that are not numbers,
  or updates that keep failing). The usual causes are a learning rate that is too high or a reward that is far
  too large. Undo your last change and start it again; it carries on from its last save.
- **It has trained for hours and still does nothing useful:** look at the bot's **Coach** tab
  ([The training coach](coach.md)) and run a skill check. The first hours are mostly chasing the ball; a bot
  needs many hours (a day or more) before it is fun to play against.
- **It drives in circles, or keeps the ball in a corner:** it is probably farming a reward. See
  [Rewards](rewards.md#farming-paid-without-playing).
- **Its randomness falls to almost nothing and it stops improving:** see the entropy hint in
  [The training coach](coach.md).
- **I changed a setting and it says it cannot change:** some settings shape the bot's networks (team size,
  network sizes, the observation, precision) and are fixed after its first save. Make a new bot from this
  one (**New bot from this one**) to change them.
- **The PC is noisy or hot while training:** that is the graphics card working hard. Stop the bot (**Stop and
  save**) whenever you need the PC; it carries on later exactly where it stopped.

## Watching and RLBot

- **Watch it play shows nothing new:** a new save arrives every 15 minutes while it trains; **Watch live**
  switches to each new save by itself.
- **My bot does not appear in RLBot:** NitroLearn's bots are RLBot v5 bots. The older RLBot v4 (RLBotGUI) does
  not run them. In RLBot v5, add the bot's folder (`bots\<bot>`) as a bot folder; see
  [Playing your bot in RLBot](rlbot.md).
- **RLBot's folder picker cannot find the folder:** NitroLearn-Setup's folder is inside AppData, which
  Windows hides in file pickers. Press **Copy folder path** after **Make RLBot bot** and paste the address
  into the picker's address bar.
- **The match does not start:** RLBot needs Rocket League (Epic Games or Steam) installed; it starts the game
  itself.
- **The bot plays worse in RLBot than in training:** export again after more training (the folder holds the
  save it was made from). An attention bot playing a team size it never trained at plays, but how well is not
  known.

## Updating

- **The update asks me to stop training first:** stop every bot (**Stop and save**), then update.
- **An update from an old version lists nearly every file as changed by me:** those files are only backup
  copies (their line endings differ). If you never edited NitroLearn's own files, ignore the list.
