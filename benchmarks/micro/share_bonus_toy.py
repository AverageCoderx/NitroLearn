"""The multi-model bot's minimum-share controller on a toy, without games or a GPU.

A PPO selector (nitro's own `SelectorEnvs`, `Trainer` and `Learner`) chooses between two models that always
play the same action. Each step every car sees u ~ U(0, 1); model A earns a0, model B earns a0 - k (u - 0.2),
so a selector left alone gives B the 20% of states with u < 0.2. B's target is 40%. Plants: a0 0.4, 0.08 and
0.025 (with k = 1) need a bonus of about 0.5, 2 and 3.5 typical step rewards to reach it; selector learning
rates 3e-4, 1e-3 and 3e-3 set how fast the selector answers. 64 cars x 32 steps a collection, 200
collections.

Controllers: `integral` is a plain integral controller (the bonus grows by rate x typical x shortfall a
collection, the share averaged over collections with weight 0.3), at rates 0.05 and 0.5; `pi` is
nitro.selector.ShareBonus as it is now (the integral plus PUSH x rate x typical x shortfall, the
shortfall of the last collection) at rates 0.25, 0.5 (the default) and 1.

Reported per controller and plant, every seed: the first collection after the selector's first fall below
30% at which the share (mean of 5 collections) reaches 95% of the target, the share over the last tenth,
the largest gap of the 5-collection share from the target over the second half, and the bonus at the end
in typical step rewards.

    python -I benchmarks/micro/share_bonus_toy.py --out benchmarks/results/share_bonus_toy
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from nitro.config.schema import ModelConfig, PPOConfig, SelectorConfig
from nitro.envs import GroupTensors
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.selector import SelectorEnvs, ShareBonus, extra_size
from nitro.trainer import Trainer

CPU = torch.device("cpu")
TARGET = 0.40
PLANTS = [(a0, lr) for a0 in (0.4, 0.08, 0.025) for lr in (3e-4, 1e-3, 3e-3)]
CONTROLLERS = {"integral": (0.05, 0.5), "pi": (0.25, 0.5, 1.0)}


class Always(torch.nn.Module):
    """A model that always plays `action` (of 2) and values every state at 0."""

    def __init__(self, action: int) -> None:
        super().__init__()
        self.action = action

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full((obs.shape[0], 2), -30.0)
        logits[:, self.action] = 30.0
        return logits, self.normalised_value(obs)

    def normalised_value(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0])


class Contexts:
    """The toy's games (see the module docstring); the observation is 4u - 2."""

    def __init__(self, agents: int, a0: float, k: float = 1.0) -> None:
        self.groups = (GroupTensors.allocate(agents, 1, 2, pin_memory=False),)
        self.groups[0].mask.fill_(True)
        self.a0, self.k = a0, k
        self.u = torch.zeros(agents)
        self._draw()

    def _draw(self) -> None:
        self.u = torch.rand(self.u.shape[0])
        self.groups[0].obs[:, 0] = 4 * (self.u - 0.5)

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        second = tensors.actions == 1
        tensors.reward.copy_(torch.where(second, self.a0 - self.k * (self.u - 0.2), self.a0))
        self._draw()

    def wait(self, group: int) -> None:
        pass


class Integral:
    """A plain integral controller, to compare against: an integral alone on an averaged share."""

    AVERAGE = 0.3

    def __init__(self, targets: Sequence[float], rate: float) -> None:
        self.targets = torch.tensor([t / 100 for t in targets])
        self.rate = rate
        self.bonus = torch.zeros(len(targets))
        self.share: torch.Tensor | None = None

    def update(self, driven: Sequence[int], typical: float) -> None:
        share = torch.tensor([n / sum(driven) for n in driven])
        self.share = share if self.share is None else self.share + self.AVERAGE * (share - self.share)
        wanted = self.targets > 0
        shortfall = torch.where(wanted, (self.targets - self.share) / self.targets.clamp_min(1e-9), 0.0)
        typical = max(typical, 1e-6)
        self.bonus = (self.bonus + self.rate * typical * shortfall).clamp(0.0, ShareBonus.CAP * typical)
        self.bonus = torch.where(wanted, self.bonus, 0.0)


def run(
    controller: str = "pi",
    rate: float | None = None,
    a0: float = 0.08,
    lr: float = 1e-3,
    collections: int = 200,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Model B's share and its bonus in typical step rewards, per collection. `rate` None: the default."""
    torch.manual_seed(seed)
    games = Contexts(64, a0)
    targets = (0.0, TARGET * 100)
    settings = SelectorConfig(specialists=("a", "b"), min_hold_seconds=0.0, min_share=targets)
    if rate is not None and controller == "pi":
        settings = SelectorConfig(
            specialists=("a", "b"), min_hold_seconds=0.0, min_share=targets, share_bonus_rate=rate
        )
    chooser = SelectorEnvs(
        games,
        [Always(0), Always(1)],
        settings,
        8,
        CPU,
        False,
        seed,
        pin_memory=False,
    )
    if controller == "integral":
        chooser.bonus = Integral(targets, rate or 0.05)  # type: ignore[assignment]
    networks = ModelConfig(policy=(32,), value=(32,), activation="relu")
    model = ActorCritic(1 + extra_size(2, False), 2, networks)
    ppo = PPOConfig(epochs=2, minibatches=4, policy_lr=lr, value_lr=lr, entropy=0.01)
    trainer = Trainer(chooser, model, Learner(model, ppo, 0.9, CPU, False), 64 * 32, CPU, False, False)
    shares, bonuses = [], []
    for _ in range(collections):
        trainer.iteration()
        driven = chooser.take_stats().driven
        typical = float(chooser.reward_sum) / sum(driven)
        chooser.end_collection(driven)
        assert chooser.bonus is not None
        shares.append(driven[1] / sum(driven))
        bonuses.append(float(chooser.bonus.bonus[1]) / typical)
    return shares, bonuses


def reading(shares: Sequence[float]) -> dict[str, Any]:
    n = len(shares)
    mean5 = [sum(shares[max(0, i - 4) : i + 1]) / len(shares[max(0, i - 4) : i + 1]) for i in range(n)]
    fell = next((i for i in range(n) if shares[i] < 0.3), 0)
    reached = next((i for i in range(fell + 5, n) if mean5[i] >= 0.95 * TARGET), None)
    return {
        "reached_at": "" if reached is None else reached,
        "last_tenth": round(sum(shares[int(0.9 * n) :]) / (n - int(0.9 * n)), 3),
        "largest_gap": round(max(abs(m - TARGET) for m in mean5[n // 2 :]), 3),
    }


def _one(job: tuple[str, float, float, float, int, int]) -> dict[str, Any]:
    torch.set_num_threads(1)
    controller, rate, a0, lr, seed, collections = job
    shares, bonuses = run(controller, rate, a0, lr, collections, seed)
    return {
        "controller": controller, "rate": rate, "a0": a0, "lr": lr, "seed": seed, **reading(shares),
        "bonus_end": round(bonuses[-1], 2), "shares": " ".join(f"{s:.3f}" for s in shares),
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/share_bonus_toy"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--collections", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [
        (controller, rate, a0, lr, seed, args.collections)
        for controller, rates in CONTROLLERS.items()
        for rate in rates
        for a0, lr in PLANTS
        for seed in range(args.seeds)
    ]
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(_one, jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"Share bonus toy: target {TARGET:.0%}, {args.collections} collections, {args.seeds} seeds.",
        "reached = first collection at 95% of the target (mean of 5; '-' never); last = share over the last "
        "tenth; gap = largest gap of the 5-collection share from the target over the second half; "
        "bonus = at the end, in typical step rewards.",
    ]
    for controller, rates in CONTROLLERS.items():
        for rate in rates:
            for a0, lr in PLANTS:
                cell = [
                    r
                    for r in rows
                    if (r["controller"], r["rate"], r["a0"], r["lr"]) == (controller, rate, a0, lr)
                ]
                lines.append(
                    f"{controller} {rate:<4} a0 {a0:<5} lr {lr:.0e}: reached "
                    f"{', '.join(str(r['reached_at']) or '-' for r in cell)}; last "
                    f"{', '.join(str(r['last_tenth']) for r in cell)}; gap "
                    f"{max(r['largest_gap'] for r in cell)}; bonus "
                    f"{', '.join(str(r['bonus_end']) for r in cell)}"
                )
    (args.out / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
