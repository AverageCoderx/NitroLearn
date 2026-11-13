"""The concurrent passes of the collection graph (`env.concurrent_passes`) choose exactly
what the passes one after another do: the same actions, likelihoods, values and rollout rows, to the bit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import GroupTensors
from nitro.inference import GroupInference, RolloutColumns
from nitro.model import ActorCritic, build_networks, fp16_inference_copy, observation_layout
from nitro.rollout import Rollout
from nitro.selfplay import OpponentBlock

AGENTS, ACTIONS, STEPS = 64, 90, 4
PLAIN = "policy = [48, 32]\nvalue = [48, 32]\n"
MODELS = {
    "plain, shared trunk, fused norm": PLAIN + "shared = [64]\nfused_norm = true\n",
    "plain, no trunk": PLAIN,
    "transformer, lean, shared trunk": PLAIN
    + 'shared = [64]\nencoder = "transformer"\ntransformer_width = 32\ntransformer_heads = 4\n'
    + "lean_attention = true\n",
    "attention, no trunk": PLAIN + 'encoder = "attention"\nattention_width = 32\nattention_heads = 4\n',
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def networks(tmp_path: Path, name: str, device: str) -> tuple[ActorCritic, list[ActorCritic]]:
    """The learner's networks (as collection runs them) and three past versions with other weights."""
    config = parse_toml(
        f"[env]\nteam_size = 2\nobs_team_size = 3\n[model]\n{MODELS[name]}",
        NitroConfig,
        file=tmp_path / "s.toml",
        base_dir=tmp_path,
    )
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, ACTIONS).eval().to(device)
    past = []
    for _ in range(3):
        other = build_networks(config, model.obs_size, ACTIONS).eval().to(device)
        past.append(fp16_inference_copy(other) if device == "cuda" else other)
    return (fp16_inference_copy(model) if device == "cuda" else model), past


def blocks(past: list[ActorCritic], device: str) -> list[OpponentBlock]:
    """Three slots of 8 rows each at the end of the group, their Orange rows driven by the past version."""
    found = []
    for slot, model in enumerate(past):
        rows = slice(AGENTS - 24 + 8 * slot, AGENTS - 16 + 8 * slot)
        select = torch.arange(8, device=device) % 2 == 1
        found.append(OpponentBlock(rows, select, model))
    return found


def fill(tensors: GroupTensors, step: int) -> None:
    generator = torch.Generator().manual_seed(100 + step)
    tensors.obs.copy_(torch.randn(tensors.obs.shape, generator=generator))
    tensors.mask.copy_(torch.rand(tensors.mask.shape, generator=generator) < 0.6)
    tensors.mask[:, 0] = True  # some action is always allowed
    tensors.reward.copy_(torch.randn(AGENTS, generator=generator))
    tensors.terminal.copy_(torch.rand(AGENTS, generator=generator) < 0.1)
    tensors.episode.fill_(step)
    tensors.step.fill_(step)


def columns_of(rollout: Rollout) -> RolloutColumns:
    r = rollout
    return RolloutColumns(
        r.obs, r.mask, r.actions, r.log_probs, r.values, r.rewards, r.terminal, r.truncated, r.episode,
        r.episode_step,
    )  # fmt: skip


def play(engine: GroupInference, tensors: GroupTensors, seed: int) -> list[torch.Tensor]:
    """Runs STEPS steps from the same random numbers; returns the actions handed to the games each step."""
    torch.manual_seed(seed)
    engine.restart()
    chosen = []
    for step in range(STEPS):
        fill(tensors, step)
        engine.run(record_previous=step > 0)
        engine.wait()
        chosen.append(tensors.actions.clone())
    return chosen


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("name", list(MODELS))
def test_concurrent_passes_choose_exactly_what_the_passes_one_after_another_do(
    tmp_path: Path, name: str, graph: bool, device: str
) -> None:
    if graph and device == "cpu":
        pytest.skip("CUDA graphs need a graphics card")
    model, past = networks(tmp_path, name, device)
    tensors = GroupTensors.allocate(AGENTS, model.obs_size, ACTIONS, pin_memory=device == "cuda")
    results = []
    for concurrent in (False, True):
        rollout = Rollout.allocate(STEPS, AGENTS, model.obs_size, ACTIONS, torch.device(device))
        engine = GroupInference(
            model,
            tensors,
            torch.device(device),
            fp16=device == "cuda",
            cuda_graph=graph,
            columns=columns_of(rollout),
            opponents=blocks(past, device),
            concurrent=concurrent,
        )
        assert bool(engine.side) == (concurrent and device == "cuda")
        results.append((play(engine, tensors, seed=7), rollout))
    (plain_actions, plain), (concurrent_actions, concurrent) = results
    for a, b in zip(plain_actions, concurrent_actions, strict=True):
        assert torch.equal(a, b)
    for field in ("obs", "mask", "actions", "log_probs", "values", "rewards", "terminal", "episode"):
        assert torch.equal(getattr(plain, field), getattr(concurrent, field)), field
    # The past versions really drove their Orange rows: alone, the learner chose otherwise somewhere there.
    rollout = Rollout.allocate(STEPS, AGENTS, model.obs_size, ACTIONS, torch.device(device))
    alone = GroupInference(model, tensors, torch.device(device), device == "cuda", graph, columns_of(rollout))
    learner_only = play(alone, tensors, seed=7)
    orange = torch.cat(
        [torch.arange(b.rows.start, b.rows.stop)[b.select.cpu()] for b in blocks(past, device)]
    )
    assert any(
        not torch.equal(a[orange], b[orange]) for a, b in zip(plain_actions, learner_only, strict=True)
    )
