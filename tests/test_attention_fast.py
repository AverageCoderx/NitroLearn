"""`model.attention_fast`: the attention encoder's same maths in fewer passes, so outputs,
gradients and saves match the usual path; and the attention it keeps for `nitro watch` adds up as it should.
The FP16 and inference-graph test runs on a machine with a graphics card."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro.model import ActorCritic, EntityEncoder, build_networks, observation_layout
from test_attention import NEEDS_GPU, SMALL, answers, observations, settings

FAST = SMALL + "attention_fast = true\n"


def pair(tmp_path: Path, model: str = SMALL) -> tuple[ActorCritic, ActorCritic, tuple[int, int, int, int]]:
    plain_config = settings(tmp_path, model=model)
    fast_config = dataclasses.replace(
        plain_config, model=dataclasses.replace(plain_config.model, attention_fast=True)
    )
    layout = observation_layout(plain_config)
    own, block, teammates, opponents = layout
    size = own + (teammates + opponents) * block
    torch.manual_seed(0)
    plain = build_networks(plain_config, size, 90)
    with torch.no_grad():  # LayerNorms off their identity start and the null car off zero, so the
        for p in plain.parameters():  # order of every weight in the bot file is tested
            p.add_(torch.randn_like(p) * 0.3)
    fast = build_networks(fast_config, size, 90)
    fast.load_state_dict(plain.state_dict())
    return plain, fast, layout


@pytest.mark.parametrize("filled", [(0, 0), (1, 2), (2, 3), (0, 1)])
def test_the_fast_path_gives_the_same_answers_and_gradients(tmp_path: Path, filled: tuple[int, int]) -> None:
    plain, fast, layout = pair(tmp_path)
    obs = observations(layout, 16, filled)
    for model in (plain, fast):
        logits, value = model(obs)
        (logits.square().mean() + value.square().mean()).backward()
    for a, b in zip(answers(plain, obs), answers(fast, obs), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
    for (name, a), b in zip(plain.named_parameters(), fast.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_a_save_is_the_same_with_it_on_or_off(tmp_path: Path) -> None:
    plain, fast, _ = pair(tmp_path)
    assert list(plain.state_dict()) == list(fast.state_dict())
    encoder = fast.policy[0]
    assert isinstance(encoder, EntityEncoder) and encoder.fast
    pairs = encoder.fresh_parameters()
    assert [mine for mine, _ in pairs] == list(encoder.parameters()), "fresh weights for every parameter"
    assert all(mine.shape == new.shape for mine, new in pairs)


def test_the_kept_attention_adds_up_and_ignores_empty_places(tmp_path: Path) -> None:
    for model in pair(tmp_path)[:2]:
        encoder = model.policy[0]
        assert isinstance(encoder, EntityEncoder)
        encoder.keep_weights = True
        layout = model.layout
        assert layout is not None
        _, _, teammates, opponents = layout
        obs = observations(layout, 8, (1, 2))
        answers(model, obs)
        weights = encoder.attention_weights
        assert weights is not None and weights.shape == (8, encoder.heads, 1 + teammates + opponents)
        torch.testing.assert_close(weights.sum(-1), torch.ones(8, encoder.heads))
        taken = [0, 1, 1 + teammates, 1 + teammates + 1]  # the null car, one teammate, two opponents
        empty = [i for i in range(1 + teammates + opponents) if i not in taken]
        assert torch.all(weights[..., empty] == 0)
        answers(model, observations(layout, 8, (0, 0)))
        alone = encoder.attention_weights
        assert alone is not None and torch.all(alone[..., 0] == 1), "with no other car, all on the null car"
        encoder.keep_weights = False


@NEEDS_GPU
def test_fp16_and_the_inference_graph_with_the_fast_path(tmp_path: Path) -> None:
    from nitro.envs import GroupTensors
    from nitro.inference import GroupInference

    plain, fast, layout = pair(tmp_path)
    plain, fast = plain.eval().cuda(), fast.eval().cuda()
    obs = observations(layout, 64, (2, 3)).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        expected, got = answers(plain, obs), answers(fast, obs)
    for a, b in zip(expected, got, strict=True):
        torch.testing.assert_close(b.float(), a.float(), rtol=0.02, atol=0.02)
    tensors = GroupTensors.allocate(64, fast.obs_size, 90, pin_memory=True)
    tensors.obs.copy_(obs.cpu())
    tensors.mask.fill_(True)
    engine = GroupInference(fast, tensors, torch.device("cuda"), fp16=True, cuda_graph=True)
    engine.run()
    engine.wait()
    assert ((tensors.actions >= 0) & (tensors.actions < 90)).all()
