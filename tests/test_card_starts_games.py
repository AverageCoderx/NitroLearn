"""`env.card_starts_games`: the graphics card starts each group's games itself, through the
C function `EnvPool.card_start` hands out. Called as the card's host node calls it, it steps a group exactly
as `step` does, reports a failure at the group's next `wait`, and refuses once the group's tensors are
attached again; on a card, training with it gives the same rollouts and losses to the bit."""

from __future__ import annotations

import ctypes
import dataclasses
import os
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import EnvPool
from nitro.training import build, configure_determinism

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
pytestmark = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")

SETTINGS = """
[run]
seed = 7
deterministic = true
[env]
num_arenas = 8
groups = 2
threads = 3
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""
HOST_FUNCTION = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def config(**changes: dict[str, object]) -> NitroConfig:
    parsed = parse_toml(SETTINGS, NitroConfig, file=Path("test.toml"), base_dir=Path.cwd())
    parsed = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    for section, values in changes.items():
        parsed = dataclasses.replace(
            parsed, **{section: dataclasses.replace(getattr(parsed, section), **values)}
        )
    return parsed


def start_as_the_card_does(pool: EnvPool, group: int) -> None:
    function, argument = pool.card_start(group)
    HOST_FUNCTION(function)(argument)


def play(pool: EnvPool, steps: int, card: bool) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    seen = []
    for _ in range(steps):
        for group, tensors in enumerate(pool.groups):
            tensors.actions.copy_(torch.randint(0, 90, tensors.actions.shape, generator=generator))
            if card:
                start_as_the_card_does(pool, group)
            else:
                pool.step(group)
        for group, tensors in enumerate(pool.groups):
            pool.wait(group)
            seen += [
                tensors.obs.clone(),
                tensors.reward.clone(),
                tensors.terminal.clone(),
                tensors.step.clone(),
            ]
    return seen


def test_a_card_start_steps_the_group_as_step_does() -> None:
    by_python = play(EnvPool(config(), pin_memory=False), 30, card=False)
    by_card = play(EnvPool(config(), pin_memory=False), 30, card=True)
    for a, b in zip(by_python, by_card, strict=True):
        assert torch.equal(a, b)


def test_a_failed_card_start_is_reported_by_the_next_wait() -> None:
    pool = EnvPool(config(), pin_memory=False)
    pool.groups[0].actions.fill_(999)
    start_as_the_card_does(pool, 0)
    with pytest.raises(_core.EnvError, match="actions must be"):
        pool.wait(0)
    pool.groups[0].actions.zero_()
    start_as_the_card_does(pool, 0)
    pool.wait(0)  # the failure is reported once


def test_a_card_start_refuses_once_the_tensors_are_attached_again() -> None:
    pool = EnvPool(config(), pin_memory=False)
    function, argument = pool.card_start(1)
    pool._core.attach(1, pool.groups[1])
    HOST_FUNCTION(function)(argument)
    with pytest.raises(_core.EnvError, match="attached again"):
        pool.wait(1)


def collect(card: bool, precision: str) -> list[torch.Tensor]:
    setup = build(config(run={"device": "cuda", "precision": precision}, env={"card_starts_games": card}))
    trainer = setup.trainer
    assert all(engine.starts_games for engine in trainer.engines) is card
    seen = []
    for _ in range(2):
        report = trainer.iteration()
        r = trainer.rollout
        seen += [r.obs.cpu(), r.actions.cpu(), r.log_probs.cpu(), r.rewards.cpu(), r.terminal.cpu()]
        seen.append(torch.tensor([report.learn.policy_loss, report.learn.value_loss, report.learn.entropy]))
    configure_determinism(False)
    return seen


@NEEDS_GPU
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_training_with_the_card_starting_the_games_is_the_same_to_the_bit(precision: str) -> None:
    for a, b in zip(collect(False, precision), collect(True, precision), strict=True):
        assert torch.equal(a, b)
