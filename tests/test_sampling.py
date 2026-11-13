"""Choosing actions in fewer GPU jobs: `masked_log_probs` as one `where` gives the old
function's numbers and gradients to the bit; the small attention routine reads the same present flags without
copying them; the sampling gives the old code's actions, log-probabilities and random stream to the bit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from nitro import small_attention
from nitro.envs import GroupTensors
from nitro.inference import GroupInference
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.selfplay import OpponentBlock
from test_concurrent_passes import ACTIONS, AGENTS, STEPS, blocks, fill, networks

BOUNDS = (1e-20, 1.0 - 1e-7)  # nitro.inference's clamp of the uniform numbers


def old_masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`ppo.masked_log_probs` in an earlier version."""
    return torch.log_softmax(logits.float().masked_fill(~mask, -math.inf), dim=-1)


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.float().view(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_log_probs_is_the_old_function_to_the_bit(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    logits = (torch.randn(300, 90, generator=generator) * 4).to(dtype)
    mask = torch.rand(300, 90, generator=generator) < 0.6
    mask[:, 0] = True
    mask[1] = False  # nothing allowed: NaN everywhere, as before
    logits[2, 5] = math.nan
    logits[3, 7] = math.inf
    logits[4, :] = 1.5  # ties
    for check_mask in (mask, mask[:1]):  # a mask broadcast over the rows as well
        new, old = masked_log_probs(logits, check_mask), old_masked_log_probs(logits, check_mask)
        assert new.dtype == old.dtype == torch.float32 and new.shape == old.shape
        assert torch.equal(bits(new), bits(old))
    grads = []
    for function in (masked_log_probs, old_masked_log_probs):
        leaf = logits[5:].float().clone().requires_grad_()
        log_probs = function(leaf, mask[5:])
        picked = log_probs.gather(1, torch.zeros(len(leaf), 1, dtype=torch.long))
        probs = log_probs.masked_fill(~mask[5:], 0.0).exp().masked_fill(~mask[5:], 0.0)
        (picked.sum() + (probs * log_probs.masked_fill(~mask[5:], 0.0)).sum()).backward()
        assert leaf.grad is not None
        grads.append(leaf.grad)
    assert torch.equal(bits(grads[0]), bits(grads[1]))


def test_the_small_attention_flags_are_the_same_bytes_without_a_copy() -> None:
    present = torch.rand(64, 7, generator=torch.Generator().manual_seed(1)) < 0.6
    flags = small_attention._flags(present)
    assert flags.data_ptr() == present.data_ptr(), "a view, no copy"
    assert flags.dtype == torch.uint8 and flags.stride() == present.stride()
    assert torch.equal(flags, present.to(torch.uint8))
    sliced = present[:, ::2]  # not contiguous: copied, so the routine reads the right places
    assert torch.equal(small_attention._flags(sliced), sliced.to(torch.uint8))
    assert small_attention._flags(sliced).is_contiguous()


def old_step(
    model: ActorCritic, opponents: list[OpponentBlock], obs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """One step's choice as `GroupInference` made it in an earlier version (one pass after another, on the
    processor):
    the actions, the learner's log-probabilities of them and its values."""
    with torch.no_grad():
        logits, values = model(obs)
        log_probs = old_masked_log_probs(logits, mask)
        uniform = torch.rand_like(log_probs).clamp_(*BOUNDS)
        actions = torch.argmax(log_probs + -torch.log(-torch.log(uniform)), dim=-1)
        chosen = []
        for block in opponents:
            theirs = old_masked_log_probs(block.model.logits(obs[block.rows]), mask[block.rows])
            noise = -torch.log(-torch.log(torch.rand_like(theirs).clamp_(*BOUNDS)))
            chosen.append(torch.argmax(theirs + noise, dim=-1))
        for block, choice in zip(opponents, chosen, strict=True):
            actions[block.rows].copy_(torch.where(block.select, choice, actions[block.rows]))
        return actions, log_probs.gather(1, actions[:, None]).squeeze(1), values


@pytest.mark.parametrize("name", ["plain, no trunk", "transformer, lean, shared trunk"])
def test_on_the_processor_the_choice_is_the_old_codes_to_the_bit(tmp_path: Path, name: str) -> None:
    """The sampling as it is now gives the old code's actions, log-probabilities and values, and draws the
    same random numbers."""
    model, past = networks(tmp_path, name, "cpu")
    opponents = blocks(past, "cpu")
    tensors = GroupTensors.allocate(AGENTS, model.obs_size, ACTIONS, pin_memory=False)
    engine = GroupInference(model, tensors, torch.device("cpu"), False, False, opponents=opponents)
    for step in range(STEPS):
        fill(tensors, step)
        torch.manual_seed(step)
        engine.run()
        after = torch.rand(3)
        torch.manual_seed(step)
        actions, log_probs, values = old_step(model, opponents, tensors.obs, tensors.mask)
        assert torch.equal(after, torch.rand(3)), "the same random numbers drawn"
        assert torch.equal(engine.actions, actions)
        assert torch.equal(tensors.actions, actions.int())
        assert torch.equal(bits(engine.log_probs), bits(log_probs))
        assert torch.equal(bits(engine.values), bits(values))
