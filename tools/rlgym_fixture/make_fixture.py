"""Writes small RLGym-PPO and rlgym-learn saves for the import's tests, as those frameworks save them:
rlgym-ppo's DiscreteFF / ValueEstimator (`model` = Linear, ReLU, ..., Linear(, Softmax)) as PPO_POLICY.pt and
PPO_VALUE_NET.pt with their Adam states and BOOK_KEEPING_VARS.json (rlgym_ppo/ppo/ppo_learner.py, learner.py);
rlgym-learn's SeparateActorCritic (`actor.model`, `critic.model`) as ppo_learner/actor_critic.pt with
optimizers.pt, misc.json, ppo_agent.json and trajectory_processor.json (rlgym_learn_algos/ppo/...).

    python tools/rlgym_fixture/make_fixture.py <out folder> <obs size> [--learn] [--no-obs-stats]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import nn


def network(inputs: int, hidden: list[int], outputs: int, softmax: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(inputs, hidden[0]), nn.ReLU()]
    for a, b in itertools.pairwise(hidden):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(hidden[-1], outputs))
    if softmax:
        layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


class Holder(nn.Module):
    """rlgym-ppo's DiscreteFF / ValueEstimator keep their layers as `model`."""

    def __init__(self, model: nn.Sequential) -> None:
        super().__init__()
        self.model = model


def write(out: Path, obs: int, learn: bool = False, obs_stats_too: bool = True) -> Path:
    """Writes one save under `out`/checkpoints and returns its folder."""
    folder: Path
    args = argparse.Namespace(out=out, obs=obs, learn=learn, no_obs_stats=not obs_stats_too)
    torch.manual_seed(5)
    policy = Holder(network(args.obs, [32, 24], 90, softmax=True))
    critic = Holder(network(args.obs, [24], 1, softmax=False))
    optims = [torch.optim.Adam(policy.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)]
    for _ in range(3):  # some steps, so the optimizers hold moments
        x = torch.randn(64, args.obs)
        loss = -policy.model(x).clamp(1e-11, 1).log().mean() + critic.model(x).pow(2).mean()
        for o in optims:
            o.zero_grad()
        loss.backward()
        for o in optims:
            o.step()
    generator = torch.Generator().manual_seed(6)
    obs_stats = {
        "mean": (torch.randn(args.obs, generator=generator) * 0.3).tolist(),
        "var": (torch.rand(args.obs, generator=generator) * 400 + 1).tolist(),
        "shape": [args.obs],
        "count": 2000,
    }
    returns = {"mean": [0.1], "var": [3200.0], "shape": [1], "count": 801}
    if args.learn:
        folder = args.out / "checkpoints" / "1763400000000000000"
        (folder / "ppo_learner").mkdir(parents=True, exist_ok=True)
        both = nn.Module()
        both.actor, both.critic = policy, critic
        torch.save(both.state_dict(), folder / "ppo_learner" / "actor_critic.pt")
        torch.save([o.state_dict() for o in optims], folder / "ppo_learner" / "optimizers.pt")
        (folder / "ppo_learner" / "misc.json").write_text(json.dumps({"cumulative_model_updates": 30}))
        agent = {"cur_iteration": 12, "iteration_timesteps": 0, "cumulative_timesteps": 654321}
        (folder / "ppo_agent.json").write_text(json.dumps(agent))
        (folder / "trajectory_processor.json").write_text(
            json.dumps({"return_running_stats": {"mean": 0.1, "m2": 3200.0, "count": 801}})
        )
    else:
        folder = args.out / "checkpoints" / "rlgym-ppo-run-1763400000" / "654321"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), folder / "PPO_POLICY.pt")
        torch.save(critic.state_dict(), folder / "PPO_VALUE_NET.pt")
        torch.save(optims[0].state_dict(), folder / "PPO_POLICY_OPTIMIZER.pt")
        torch.save(optims[1].state_dict(), folder / "PPO_VALUE_NET_OPTIMIZER.pt")
        book = {
            "cumulative_timesteps": 654321,
            "cumulative_model_updates": 30,
            "policy_average_reward": 0.5,
            "epoch": 12,
            "ts_since_last_save": 0,
            "reward_running_stats": returns,
        }
        if not args.no_obs_stats:
            book["obs_running_stats"] = obs_stats
        (folder / "BOOK_KEEPING_VARS.json").write_text(json.dumps(book, indent=4))
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("obs", type=int)
    parser.add_argument("--learn", action="store_true", help="rlgym-learn's layout instead of rlgym-ppo's")
    parser.add_argument("--no-obs-stats", action="store_true")
    args = parser.parse_args()
    print(f"wrote {write(args.out, args.obs, args.learn, not args.no_obs_stats)}")


if __name__ == "__main__":
    main()
