"""`ppo.microbatch`: a learning step worked out in parts learns as the whole step does."""

from __future__ import annotations

import pytest
import torch

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout

CPU = torch.device("cpu")
OBS, ACTIONS, STEPS, AGENTS = 6, 4, 8, 6  # 48 samples, 24 a minibatch


def rollout_for(model: ActorCritic) -> Rollout:
    generator = torch.Generator().manual_seed(0)
    r = Rollout.allocate(STEPS, AGENTS, OBS, ACTIONS, CPU)
    r.obs.copy_(torch.randn(STEPS, AGENTS, OBS, generator=generator))
    r.mask.copy_(torch.rand(STEPS, AGENTS, ACTIONS, generator=generator) > 0.3)
    r.mask[..., 0] = True
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    r.actions.copy_(torch.distributions.Categorical(logits=log_probs).sample())
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(STEPS, AGENTS, generator=generator))
    r.terminal[5, :] = True
    r.truncated[2, :] = True
    return r


def learned(microbatch: int, shared: tuple[int, ...], frozen: bool) -> tuple[list[torch.Tensor], object]:
    torch.manual_seed(1)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(shared=shared, policy=(16,), value=(16,)))
    config = PPOConfig(epochs=2, minibatches=2, microbatch=microbatch)
    learner = Learner(model, config, gamma=0.9, device=CPU, fp16=False)
    learner.policy_frozen = frozen
    torch.manual_seed(2)  # the same minibatch order for every learner
    stats = learner.learn(rollout_for(model))
    return [p.detach().clone() for p in model.parameters()], stats


@pytest.mark.parametrize("shared", [(), (16,)])
@pytest.mark.parametrize("frozen", [False, True])
def test_a_step_in_parts_learns_as_the_whole_step(shared: tuple[int, ...], frozen: bool) -> None:
    whole, whole_stats = learned(0, shared, frozen)
    parts, part_stats = learned(5, shared, frozen)  # 24 samples a step: parts of 5, 5, 5, 5 and 4
    assert any(not torch.equal(a, b) for a, b in zip(whole, parts, strict=True)), "rounding differs a little"
    for a, b in zip(whole, parts, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert getattr(part_stats, name) == pytest.approx(getattr(whole_stats, name), rel=1e-5, abs=1e-7)


def test_parts_as_big_as_the_step_change_nothing() -> None:
    whole, _ = learned(0, (16,), False)
    same, _ = learned(24, (16,), False)
    assert all(torch.equal(a, b) for a, b in zip(whole, same, strict=True))
