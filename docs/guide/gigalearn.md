# Bringing a GigaLearn bot over
If you trained a bot with GigaLearn, NitroLearn can take it over so it keeps training here as the same bot,
not a new one copied from it. In the dashboard: **New bot**, then **From a GigaLearn bot**. Choose the bot's
saves (its checkpoints folder) and its project folder (the one with your main `.cpp` file), press **Check
what comes over** to see the report, give it a name, and press **Make the bot**. When it is made you can start
training it straight away. From a terminal:
```powershell
.\nitro import-gigalearn D:\MyGigaLearnBot\checkpoints --source D:\MyGigaLearnBot --into my-ggl-bot
.\nitro resume my-ggl-bot
```
What comes over: the networks with their trained weights and Adam's running averages, what the bot sees
(GigaLearn's `AdvancedObs`, `DefaultObs`, or the community `AdvancedObsPadderGGL`), its actions and their
masks, its rewards with their weights, its learning settings and how its rewards were scaled, starting at the
step count GigaLearn had reached. It acts with the lowest action delay (1 tick), with which a bot reacts
fastest in a real game, whatever delay it was trained with (`--keep-action-delay` keeps its own). Before anything is written, NitroLearn plays a few short games and checks
that the new bot chooses exactly as the GigaLearn bot does ("Parity check PASSED"); if it does not, nothing is
made.

The report says what came over exactly, what is close but not quite the same, and what you have to port by
hand: rewards, observations or action parsers you wrote yourself in C++ (with the file and line where they
are), and GigaLearn's shot, save and assist rewards, which need game events NitroLearn's simulator does not
have. A bot with its own observation or action parser, or with sigmoid networks, cannot be brought over: the
import says so and writes nothing. The report is kept in the bot's folder as `imported.txt`.
