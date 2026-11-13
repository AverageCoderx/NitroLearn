from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import EnvPool, GroupTensors, core_settings

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
pytestmark = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

FIELDS = dataclasses.fields(GroupTensors)
SETTINGS = """
[run]
seed = 3
[env]
num_arenas = 6
groups = 2
threads = 3
action_delay = 1
[terminals]
no_touch_seconds = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""


def config(text: str = SETTINGS) -> NitroConfig:
    parsed = parse_toml(text, NitroConfig, file=Path("test.toml"), base_dir=Path.cwd())
    return dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))


def play(pool: EnvPool, steps: int, seed: int) -> list[dict[str, torch.Tensor]]:
    """Steps every group with seeded random actions and records each group's tensors after each step."""
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(steps):
        for group, tensors in enumerate(pool.groups):
            tensors.actions.copy_(torch.randint(0, 90, tensors.actions.shape, generator=generator))
            pool.step(group)
        for group in range(pool.num_groups):
            pool.wait(group)
        history.append(
            {f"{g}.{f.name}": getattr(t, f.name).clone() for g, t in enumerate(pool.groups) for f in FIELDS}
        )
    return history


def test_shapes_and_layout() -> None:
    pool = EnvPool(config(), pin_memory=False)
    assert pool.num_groups == 2 and pool.num_workers == 3
    assert [t.obs.shape for t in pool.groups] == [torch.Size([6, 148])] * 2
    assert pool.obs_layout.size == 148 and pool.obs_layout.opponents == 1
    assert pool.reward_labels[0] == "air" and len(pool.reward_labels) == 10
    assert all(t.mask.any(dim=1).all() for t in pool.groups), "every car needs an allowed action at the start"


def test_python_driven_runs_are_reproducible() -> None:
    first = play(EnvPool(config(), pin_memory=False), 60, seed=1)
    second = play(EnvPool(config(), pin_memory=False), 60, seed=1)
    for a, b in zip(first, second, strict=True):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_episodes_never_splice() -> None:
    pool = EnvPool(config(), pin_memory=False)
    before = [(t.episode.clone(), t.step.clone()) for t in pool.groups]
    endings = 0
    for snapshot in play(pool, 200, seed=2):
        for group, (episode, step) in enumerate(before):
            ended = snapshot[f"{group}.terminal"] | snapshot[f"{group}.truncated"]
            endings += int(ended.sum())
            same = snapshot[f"{group}.episode"] == episode
            assert torch.equal(same, ~ended), "an episode changed without ending, or ended without changing"
            assert torch.equal(snapshot[f"{group}.step"], torch.where(ended, 0, step + 1))
            before[group] = (snapshot[f"{group}.episode"], snapshot[f"{group}.step"])
    assert endings > 20


def test_stats_count_every_agent_step() -> None:
    pool = EnvPool(config(), pin_memory=False)
    play(pool, 10, seed=3)
    stats = pool.take_stats()
    assert stats.agent_steps == 10 * 12 and stats.env_steps == 10 * 6
    assert 0 < stats.physics_seconds < stats.step_seconds
    assert len(stats.raw) == len(pool.reward_labels)
    assert pool.take_stats().agent_steps == 0


def test_wrong_tensors_are_refused_with_a_clear_message() -> None:
    core = _core.EnvPool(core_settings(config()))
    good = GroupTensors.allocate(core.group_agents(0), core.obs_size, core.num_actions, pin_memory=False)
    with pytest.raises(
        ValueError, match=re.escape("'reward' must be a contiguous CPU tensor of torch.float32")
    ):
        core.attach(0, dataclasses.replace(good, reward=good.reward.double()))
    with pytest.raises(ValueError, match="'obs' must be"):
        core.attach(0, dataclasses.replace(good, obs=good.obs.t()))
    with pytest.raises(ValueError, match="group 2 does not exist"):
        core.attach(2, good)
    with pytest.raises(_core.EnvError, match="attach tensors to group 1 first"):
        core.step(1)
    core.attach(0, good)
    good.actions.resize_(1)
    with pytest.raises(_core.EnvError, match="resized or replaced"):
        core.step(0)


def test_bad_core_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="invalid pool settings"):
        _core.EnvPool('{"seed": 1}')


def test_uncounted_cars_are_left_out_of_the_statistics() -> None:
    pool = EnvPool(config(), pin_memory=False)
    pool.take_stats()
    play(pool, 3, seed=1)
    everyone = pool.take_stats()
    assert everyone.agent_steps == 6 * 2 * 3, "6 arenas x 2 cars x 3 steps"
    for arena in range(6):
        pool.set_counted(arena, [True, False])
    play(pool, 3, seed=2)
    blue_only = pool.take_stats()
    assert blue_only.agent_steps == 6 * 3 and blue_only.env_steps == everyone.env_steps
    with pytest.raises(ValueError, match="flags"):
        pool.set_counted(0, [True])
    with pytest.raises(ValueError, match="arena"):
        pool.set_counted(99, [True, True])
