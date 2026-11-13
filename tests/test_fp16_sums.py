"""`run.fp16_sums`: FP16 products keep their running sums in FP16 in the passes it names.

On any machine: the switch is on only inside `kept(True)` and back as it was after (an error too); the
learner turns it on for its forward passes with 'forward' and for the backward pass too with 'all'; a playing
pass with the setting on, past versions' included; the settings refuse it without FP16. On a graphics card:
the playing passes captured with the sums in FP16 choose nearly every action the plain ones do, and a learning
step's gradients point where FP32 sums' do."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro import fp16_sums
from nitro.config import default_config
from nitro.envs import GroupTensors
from nitro.inference import GroupInference
from nitro.rollout import Rollout
from test_concurrent_passes import ACTIONS, AGENTS, STEPS, blocks, columns_of, networks, play
from test_graphed_steps import batch, learner

SWITCH = torch.backends.cuda.matmul
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def test_the_switch_is_on_only_inside() -> None:
    assert fp16_sums.available()
    assert not SWITCH.allow_fp16_accumulation
    with fp16_sums.kept(False):
        assert not SWITCH.allow_fp16_accumulation
    with fp16_sums.kept(True):
        assert SWITCH.allow_fp16_accumulation
        with fp16_sums.kept(False):  # an inner pass without it leaves it as it is
            assert SWITCH.allow_fp16_accumulation
    assert not SWITCH.allow_fp16_accumulation
    with pytest.raises(RuntimeError), fp16_sums.kept(True):
        raise RuntimeError
    assert not SWITCH.allow_fp16_accumulation


@pytest.mark.parametrize(
    ("mode", "forward", "backward"),
    [("off", False, False), ("playing", False, False), ("forward", True, False), ("all", True, True)],
)
def test_the_learner_keeps_fp16_sums_in_the_passes_its_mode_names(
    mode: str, forward: bool, backward: bool
) -> None:
    learn = learner(torch.device("cpu"))
    learn.fp16_sums = mode
    seen: dict[str, list[bool]] = {"forward": [], "backward": []}
    model = learn.model
    plain_forward = model.forward

    def recording_forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seen["forward"].append(SWITCH.allow_fp16_accumulation)
        return plain_forward(obs)

    model.forward = recording_forward  # type: ignore[method-assign]
    weight = next(model.parameters())
    weight.register_hook(lambda grad: seen["backward"].append(SWITCH.allow_fp16_accumulation))
    learn._step(*batch(torch.float32), value_only=False)
    assert seen["forward"] and set(seen["forward"]) == {forward}
    assert seen["backward"] and set(seen["backward"]) == {backward}
    assert not SWITCH.allow_fp16_accumulation


@pytest.mark.parametrize("on", [False, True])
def test_a_playing_pass_keeps_fp16_sums_with_the_setting_on(tmp_path: Path, on: bool) -> None:
    model, past = networks(tmp_path, "plain, no trunk", "cpu")
    seen: list[bool] = []
    for network in (model, *past):
        network.policy[0].register_forward_pre_hook(lambda *_: seen.append(SWITCH.allow_fp16_accumulation))
    tensors = GroupTensors.allocate(AGENTS, model.obs_size, ACTIONS, pin_memory=False)
    rollout = Rollout.allocate(STEPS, AGENTS, model.obs_size, ACTIONS, torch.device("cpu"))
    engine = GroupInference(
        model,
        tensors,
        torch.device("cpu"),
        False,
        False,
        columns_of(rollout),
        opponents=blocks(past, "cpu"),
        fp16_sums=on,
    )
    play(engine, tensors, seed=7)
    assert len(seen) == STEPS * (1 + len(past)) and set(seen) == {on}
    assert not SWITCH.allow_fp16_accumulation


def test_the_settings_refuse_fp16_sums_without_fp16(tmp_path: Path) -> None:
    config = default_config(tmp_path)
    fp16 = dataclasses.replace(config.run, precision="fp16", fp16_sums="all")
    assert not [p for p in dataclasses.replace(config, run=fp16).problems() if p[0] == "run.fp16_sums"]
    fp32 = dataclasses.replace(fp16, precision="fp32")
    assert ("run.fp16_sums", "needs run.precision = 'fp16'") in dataclasses.replace(
        config, run=fp32
    ).problems()


@NEEDS_GPU
def test_playing_with_fp16_sums_chooses_nearly_the_same_actions(tmp_path: Path) -> None:
    model, past = networks(tmp_path, "transformer, lean, shared trunk", "cuda")
    tensors = GroupTensors.allocate(AGENTS, model.obs_size, ACTIONS, pin_memory=True)
    chosen = []
    for on in (False, True):
        rollout = Rollout.allocate(STEPS, AGENTS, model.obs_size, ACTIONS, torch.device("cuda"))
        engine = GroupInference(
            model,
            tensors,
            torch.device("cuda"),
            True,
            True,
            columns_of(rollout),
            blocks(past, "cuda"),
            fp16_sums=on,
        )
        chosen.append(torch.stack(play(engine, tensors, seed=7)))
    assert not SWITCH.allow_fp16_accumulation
    assert (chosen[0] == chosen[1]).float().mean().item() > 0.95


@NEEDS_GPU
@pytest.mark.parametrize("mode", ["forward", "all"])
def test_a_learning_step_with_fp16_sums_points_where_fp32_sums_do(mode: str) -> None:
    gradients = []
    for sums in ("off", mode):
        learn = learner(torch.device("cuda"), fp16=True)
        learn.fp16_sums = sums
        inputs = [t.cuda() for t in batch(torch.float32)]
        learn.optimizer.zero_grad(set_to_none=True)
        parts = learn._step_parts(*inputs, False)
        next(parts)
        scale = learn.scaler.get_scale()
        gradients.append(
            torch.cat([p.grad.flatten() / scale for p in learn.model.parameters() if p.grad is not None])
        )
        parts.close()
    cosine = torch.nn.functional.cosine_similarity(*gradients, dim=0).item()
    assert cosine > 0.999
