"""CartPole proofs: NitroLearn's learner against a known-good PPO on the same task.

Two tasks (`--task`):
- `cartpole`: Gymnasium's CartPole-v1; reference Stable-Baselines3's PPO.
- `masked`: the same task with nitro.cartpole's masked action layout (8 actions, one random copy each of
  push-left and push-right allowed at every step, two actions never allowed); reference sb3-contrib's
  MaskablePPO on an identical Gymnasium version of it. This exercises action masking, which plain
  CartPole doesn't.
Both use CleanRL's classic CartPole settings: 4 environments, 128 steps each per iteration, 4 epochs of 4
minibatches, learning rate 2.5e-4, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01,
gradient clipping 0.5, 64x64 tanh networks. NitroLearn runs its full learner (reward scaling,
PopArt, CUDA-graph inference) through `nitro.cartpole`; its entropy is normalised by log(number of
allowed actions) = log(2) in both tasks, so its coefficient is 0.01 * log(2) to match.

Every finished episode's return is written out; the summary reports steps until the 20-episode average
first reaches 475. Every NitroLearn run is also checked for learning (nitro.health.learning_problems:
finite losses and weights, applied updates, moving weights, KL and entropy); a run that fails is invalid.

    python -I benchmarks/proof/cartpole.py --task masked --out cartpole_masked_proof.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from nitro.cartpole import MASKED_ACTIONS, CartPoleEnvs
from nitro.config.schema import ModelConfig, PPOConfig
from nitro.health import learning_problems
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.trainer import Trainer

NUM_ENVS, STEPS_PER_ENV, EPOCHS, MINIBATCHES = 4, 128, 4, 4
LR, GAMMA, LAMBDA, CLIP, ENT, GRAD = 2.5e-4, 0.99, 0.95, 0.2, 0.01, 0.5
SOLVED, WINDOW = 475.0, 20

Episode = tuple[int, float]  # (environment steps so far, episode return)


def run_nitro(seed: int, total_steps: int, fp16: bool, masked: bool) -> tuple[list[Episode], list[str]]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    envs = CartPoleEnvs(NUM_ENVS, num_groups=2, seed=seed, pin_memory=True, masked=masked)
    model_config = ModelConfig(policy=(64, 64), value=(64, 64), layer_norm=False, activation="tanh")
    ppo = PPOConfig(
        steps_per_iteration=NUM_ENVS * STEPS_PER_ENV,
        epochs=EPOCHS,
        minibatches=MINIBATCHES,
        policy_lr=LR,
        value_lr=LR,
        gae_lambda=LAMBDA,
        clip_range=CLIP,
        entropy=ENT * math.log(2),
        max_grad_norm=GRAD,
    )
    model = ActorCritic(4, envs.groups[0].mask.shape[1], model_config).to(device)
    learner = Learner(model, ppo, GAMMA, device, fp16)
    trainer = Trainer(envs, model, learner, ppo.steps_per_iteration, device, fp16, cuda_graphs=True)
    episodes: list[Episode] = []
    stats = []
    steps = 0
    while steps < total_steps:
        report = trainer.iteration()
        stats.append(report.learn)
        steps += report.agent_steps
        episodes.extend((steps, float(length)) for length in envs.take_episode_lengths())
    return episodes, learning_problems(stats)


def masked_gym_env() -> Any:
    """Gymnasium CartPole-v1 with nitro.cartpole's masked action layout, for MaskablePPO."""
    import gymnasium
    import numpy as np
    from sb3_contrib.common.wrappers import ActionMasker

    class MaskedCartPole(gymnasium.Wrapper[Any, Any, Any, Any]):
        def __init__(self) -> None:
            super().__init__(gymnasium.make("CartPole-v1"))
            self.action_space = gymnasium.spaces.Discrete(MASKED_ACTIONS)
            self.mask = np.zeros(MASKED_ACTIONS, dtype=bool)

        def _new_mask(self) -> None:
            self.mask[:] = False
            self.mask[self.np_random.integers(3)] = True
            self.mask[3 + self.np_random.integers(3)] = True

        def reset(self, **kwargs: Any) -> Any:
            result = self.env.reset(**kwargs)
            self._new_mask()
            return result

        def step(self, action: Any) -> Any:
            if not self.mask[action]:
                raise ValueError("an action that was not allowed was taken")
            result = self.env.step(int(action >= 3))
            self._new_mask()
            return result

        def action_masks(self) -> Any:
            return self.mask.copy()

    return ActionMasker(MaskedCartPole(), lambda env: env.action_masks())


def run_reference(seed: int, total_steps: int, masked: bool) -> tuple[list[Episode], list[str]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    episodes: list[Episode] = []

    class Record(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episodes.append((self.num_timesteps, float(info["episode"]["r"])))
            return True

    settings: dict[str, Any] = {
        "n_steps": STEPS_PER_ENV,
        "batch_size": NUM_ENVS * STEPS_PER_ENV // MINIBATCHES,
        "n_epochs": EPOCHS,
        "learning_rate": LR,
        "gamma": GAMMA,
        "gae_lambda": LAMBDA,
        "clip_range": CLIP,
        "ent_coef": ENT,
        "vf_coef": 0.5,
        "max_grad_norm": GRAD,
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh},
        "device": "cpu",
        "seed": seed,
    }
    if masked:
        from sb3_contrib import MaskablePPO

        env = make_vec_env(masked_gym_env, n_envs=NUM_ENVS, seed=seed)
        MaskablePPO("MlpPolicy", env, **settings).learn(total_steps, callback=Record())
    else:
        env = make_vec_env("CartPole-v1", n_envs=NUM_ENVS, seed=seed)
        PPO("MlpPolicy", env, **settings).learn(total_steps, callback=Record())
    return episodes, []


def steps_to_solve(episodes: list[Episode]) -> int | None:
    returns = [r for _, r in episodes]
    for i in range(WINDOW, len(returns) + 1):
        if statistics.mean(returns[i - WINDOW : i]) >= SOLVED:
            return episodes[i - 1][0]
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["cartpole", "masked"], default="cartpole")
    parser.add_argument("--learners", nargs="+", default=["reference", "nitro_fp32", "nitro_fp16"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    masked = args.task == "masked"
    learners: dict[str, Callable[[int, int], tuple[list[Episode], list[str]]]] = {
        "reference": lambda seed, steps: run_reference(seed, steps, masked),
        "nitro_fp32": lambda seed, steps: run_nitro(seed, steps, fp16=False, masked=masked),
        "nitro_fp16": lambda seed, steps: run_nitro(seed, steps, fp16=True, masked=masked),
    }
    reference_name = "maskable_ppo" if masked else "sb3_ppo"
    rows, summary = [], []
    for name in args.learners:
        label = reference_name if name == "reference" else name
        for seed in args.seeds:
            start = time.perf_counter()
            episodes, problems = learners[name](seed, args.steps)
            seconds = time.perf_counter() - start
            solved = steps_to_solve(episodes)
            if problems:
                print(f"INVALID RUN {label} seed {seed} (did not learn): {'; '.join(problems)}", flush=True)
            print(
                f"{label} seed {seed}: {len(episodes)} episodes, solved at {solved}, {seconds:.0f} s",
                flush=True,
            )
            rows.extend({"learner": label, "seed": seed, "env_steps": s, "return": r} for s, r in episodes)
            summary.append(
                {
                    "task": args.task,
                    "learner": label,
                    "seed": seed,
                    "learned": not problems,
                    "steps_to_solve": solved,
                    "final_average_return": round(statistics.mean(r for _, r in episodes[-WINDOW:]), 1),
                    "episodes": len(episodes),
                    "problems": "; ".join(problems),
                }
            )
    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
