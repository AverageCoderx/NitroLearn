# Bringing an RLGym-PPO bot over
If you trained a bot with RLGym-PPO (on rlgym-sim), NitroLearn can take it over so it keeps training here as
the same bot. From a terminal:
```powershell
.\nitro import-rlgym D:\MyRlgymBot\data\checkpoints --source D:\MyRlgymBot --into my-rlgym-bot
.\nitro resume my-rlgym-bot
```
The first path is the bot's checkpoints folder (or one save in it); `--source` is the folder with the `.py`
file that makes your `Learner`. NitroLearn reads that file without running it.

What comes over: both networks with their trained weights and Adam's running averages; what the bot sees
(rlgym-sim's `DefaultObs` or `AdvancedObs`, or rlgym-tools' `AdvancedObsPadder`), with RLGym-PPO's
observation standardisation kept exactly as the save holds it; rlgym-tools' `LookupAction` (90 actions, the
same as NitroLearn's, in the same order, with no masks, as RLGym-PPO trains); the rewards NitroLearn has (the
velocity, face-ball, touch, boost, demolition and Liu distance rewards, and `EventReward`'s goal, concede,
touch, demolition and boost parts) with their weights; the learning settings; how its rewards were scaled;
and the step count the save had reached. It acts with the lowest action delay (1 tick), with which a bot
reacts fastest in a real game (`--keep-action-delay` keeps RLGym's own timing).

Before anything is written, NitroLearn plays a few short games and runs RLGym's own observation code and your
original networks on the same moments: the new bot must choose exactly as the old one, and its critic must
give the old critic's values ("Parity check PASSED"), or nothing is made.

The report says what came over exactly, what is close but not quite the same (for example, episodes cut by a
timeout are treated as cut short rather than finished, and the observation's standardisation stays fixed from
now on), and what you have to port by hand: your own reward classes (with the file and line), and
`EventReward`'s shot and save parts, which need game events NitroLearn's simulator does not have. A bot with
its own observation builder or action parser, a continuous policy, or one made with rlgym-learn (RLGym 2)
cannot be brought over yet: the import says why and writes nothing. The report is kept in the bot's folder as
`imported.txt`.
