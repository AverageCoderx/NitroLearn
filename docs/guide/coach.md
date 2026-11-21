# The training coach

Each bot's page has a **Coach** tab. It reads the bot's skill checks and its training numbers and tells
you, in plain words, how far the bot has come and whether something in its training looks wrong. It only
makes suggestions: **it never changes a setting by itself.** You decide, and change things on the Settings
or Rewards tab as usual.

## The plan: four stages

A bot learns Rocket League in roughly this order, so the coach follows it:

1. **Basics:** touching the ball, landing on its wheels, collecting boost.
2. **Shooting:** scoring an open goal, saving a shot.
3. **Air play:** touching a ball in the air.
4. **Duels and reads:** winning kickoffs and 50/50s against a scripted opponent, reading a ball off the wall.

The stages come from the **Skill check** (Tools tab, or `.\nitro skills <bot>`): nine short drills, each
scored against a bar. For the first six the bar is the score a fresh bot trained on that one drill alone
reaches; for the last three it is what a simple scripted player reaches. A stage is done when
every one of its skills is above its bar, and the coach then says the bot is **ready for the next stage**.
Until you have run a skill check, the plan says to run one; run one again every few hours of training to
see the bot move through the stages.

Being above a bar means the skill is there, not that the bot is good at it. A bot that passes all four
stages still has a lot to learn; from there, longer training and matches against it show more.

## The hints

Under the plan, the coach lists anything in the bot's training that looks wrong, each with what it means
and what to try. It needs a few dozen learning updates first. The hints:

- **Training looks flat (a plateau).** Its reward and its touches have hardly changed over the last fifth
  of its training, or its last three skill checks did not get better. Plateaus often end by themselves.
  In self-play a flat reward can also mean it improves as fast as its opponent, so the coach asks for a
  skill check before you change anything. To try, one at a time: train longer, raise `ppo.entropy` a
  little, turn on playing against past versions, or add a reward for the next skill in the plan.
- **One reward makes up most of its pay (farming).** Over 90% of what it is paid comes from one reward, or
  its reward rose by a fifth while its touches fell by a fifth. It may be paid without playing well. Watch it
  play, then lower that reward's weight (see [Rewards](rewards.md)).
- **It has become almost certain of every choice (entropy collapse).** Its randomness has fallen under
  0.1, so it barely tries anything new. If it has also stalled, raise `ppo.entropy` a little.
- **Its learning rate looks stuck.** Three cases: the learning rate is 0 (it plays but learns nothing);
  PPO cuts back over 30% of every update (the learning rate is too high for where the bot is); or each
  update now moves it less than a twentieth as much as early on with the same learning rate (it may be
  stuck where it is).

Press **Look again** to read the newest numbers. The thresholds are rules of thumb, not proof: a hint is a
reason to look closer (watch it play, run a skill check), not a reason to change several settings at once.
Change one thing, give it a few hours, and look again.
