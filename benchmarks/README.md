# Benchmarks

Scripts that measure NitroLearn's speed and check that its learner learns. Each writes its raw rows to the
file given with `--out`, so a result can be checked again. Run them from NitroLearn's folder with its
environment active (`.\.venv\Scripts\python` on Windows, `.venv/bin/python` on Linux), on a machine doing
nothing else: other programs change speed measurements.

## Environment speed (`env/random_play.py`)
Plays games with random actions, once driven entirely by Rust and once by a Python loop as the trainer
drives them, and reports agent steps per second, the Python loop's cost and the share of time spent in
physics.

```
python -I benchmarks/env/random_play.py --out env_random_play.csv
```

## Learning proof (`proof/cartpole.py`)
Trains NitroLearn's learner on CartPole (plain, and with masked actions) next to a known-good PPO
(Stable-Baselines3, sb3-contrib's MaskablePPO) with the same settings, and reports how many steps each
needs. Needs the `dev` extra (`pip install -e .[dev]`).

```
python -I benchmarks/proof/cartpole.py --task masked --out cartpole_masked_proof.csv
```

## Practice tasks (`tasks/run.py`, `tasks/analyse.py`)
Trains fresh bots on one narrow practice task (`nitro.tasks`) for ten seeds per setup and evaluates them;
`analyse.py` applies the comparison rules (learner works, A-against-A noise floor, the smallest difference
that matters) to the rows. `nitro ab` runs the same comparison for two settings files.

```
python -I benchmarks/tasks/run.py --task ball_touch --out ball_touch.csv
python -I benchmarks/tasks/analyse.py ball_touch.csv
```

## Several graphics cards (`profile/cards_speed.py`)
Linux only. Trains a scratch copy of a bot on one card, then on several (`run.cards`), in turns, and
reports the speed-up on this machine. The bot itself is not touched.

```
python -I benchmarks/profile/cards_speed.py --bot my-bot --cards 2 --out my-bot_2cards.csv
```

## FP16 sums (`profile/fp16_accumulation.py`)
Measures how much faster a bot's FP16 passes run with their sums kept in FP16 (`run.fp16_sums`) and how far
its choices and gradients move. Needs a CUDA card and PyTorch 2.7 or newer.

```
python -I benchmarks/profile/fp16_accumulation.py --bot my-bot --out my-bot_fp16_accumulation.txt
```
