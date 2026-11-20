# Glossary

Words you will meet in the dashboard and in this handbook, in plain English.

- **Agent step:** one decision of one car. Training speed is counted in agent steps a second, and a bot's
  age in agent steps ("50M steps" is fifty million).
- **Arena:** one simulated game. NitroLearn plays hundreds or thousands of arenas at once.
- **Attention network:** a kind of network that looks at every other car the same way, meant for 2v2 and
  3v3. See [More settings](more-settings.md).
- **Bar (skill check):** the score a fresh bot trained on one drill alone reaches. Above the bar means the
  skill is there.
- **Checkpoint / save:** a copy of the bot's networks at one moment, kept in its `checkpoints` folder.
  Training saves every 15 minutes and when you stop it.
- **Clip fraction:** the share of samples where PPO limited an update. High means the steps are too big.
- **Collision meshes / arena files:** the shapes of the arena's floor, walls and goals, downloaded when
  NitroLearn installs. Without them nothing can train.
- **Copying replays (pretraining):** teaching a new bot to choose what players chose in your replays,
  before training. See [Learning from replays](replays.md).
- **Entropy (randomness):** how random the bot's choices are, from 1 (any allowed move alike) to 0 (always
  the same move). It falls as the bot learns; very low means it has stopped exploring.
- **Explained variance:** how well the value network predicts what comes next: 1 is perfect, 0 is no better
  than guessing the average.
- **Exploiter:** a bot trained only to beat yours, so yours learns to stop what beat it. See
  [Exploiters](exploiters.md).
- **Export:** turning a save into a folder RLBot can play ([Playing your bot in RLBot](rlbot.md)).
- **Farming:** a bot earning a reward without playing well, like circling the ball forever.
- **FP16:** a faster, half-size way for the graphics card to do the maths; used by bigger bots.
- **Iteration / learning update:** one round of playing and then learning from what was played. Each row of
  `metrics.csv` is one.
- **Learning rate:** how big a step each update takes. Too high and learning is jumpy; too low and it
  crawls.
- **Observation:** the numbers the bot sees each decision: where the ball and the cars are, how fast they
  move, its boost. It is fixed for a bot's whole life.
- **Pack:** one file holding a bot, to train it on another computer ([Sharing bots](sharing.md)).
- **Policy:** the part of the bot that chooses what to do.
- **PPO:** the learning method NitroLearn uses (Proximal Policy Optimization): it improves the policy a
  little at a time, never too far in one update.
- **Reward:** what the bot is paid for something it does; its **weight** is how much that is worth. See
  [Rewards](rewards.md).
- **RLBot:** the free program that runs bots in real Rocket League matches on your own PC, offline.
- **Self-play:** the bot playing against itself, or against its own earlier saves.
- **Skill check:** nine short drills that show which skills a save has (Tools tab).
- **Tick skip:** how many physics ticks pass between two decisions (4 for new bots).
- **Transfer learning:** starting a new bot from what another bot already learned.
- **Value network (critic):** the part of the bot that judges how good the moment is for its team. It helps
  the policy learn; it does not choose moves.
- **Zero-sum:** a reward shared so that what one team gains, the other loses.
