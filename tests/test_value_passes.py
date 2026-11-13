"""Value-only passes: a value warm-up trains the value side alone."""

from __future__ import annotations

import pytest
import torch

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout

CPU = torch.device("cpu")
OBS, ACTIONS, STEPS, AGENTS = 4, 3, 6, 4


def rollout_for(model: ActorCritic, device: torch.device = CPU) -> Rollout:
    generator = torch.Generator().manual_seed(0)
    r = Rollout.allocate(STEPS, AGENTS, OBS, ACTIONS, device)
    r.obs.copy_(torch.randn(STEPS, AGENTS, OBS, generator=generator))
    r.mask.copy_(torch.rand(STEPS, AGENTS, ACTIONS, generator=generator) > 0.3)
    r.mask[..., 2] = True
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    r.actions.copy_(torch.distributions.Categorical(logits=log_probs).sample())
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(STEPS, AGENTS, generator=generator))
    r.terminal[4, :] = True
    r.truncated[1, :] = True
    return r


def learner_for(**ppo: object) -> Learner:
    torch.manual_seed(3)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(16,), value=(16,)))
    config = PPOConfig(**({"epochs": 1, "minibatches": 2} | ppo))  # type: ignore[arg-type]
    return Learner(model, config, gamma=0.9, device=CPU, fp16=False)


def params(model: ActorCritic, head: str) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in getattr(model, head).parameters()]).clone()


def test_a_value_only_step_moves_the_value_network_and_nothing_else() -> None:
    learner = learner_for()
    r = rollout_for(learner.model)
    n = STEPS * AGENTS
    policy_before, value_before = params(learner.model, "policy"), params(learner.model, "value")
    measured, norms = learner._step(
        r.obs.reshape(n, OBS),
        r.mask.reshape(n, ACTIONS),
        r.actions.reshape(n),
        r.log_probs.reshape(n),
        torch.randn(n),
        torch.randn(n),
        value_only=True,
    )
    assert torch.equal(params(learner.model, "policy"), policy_before)
    assert not torch.equal(params(learner.model, "value"), value_before)
    assert measured[1] > 0 and measured[[0, 2, 3, 4]].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert norms["policy"] == 0 and norms["value"] > 0
    assert all(norm.device == learner.device for norm in norms.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_value_only_passes_run_on_the_graphics_card() -> None:
    """In a value-only step the policy's empty gradient norm came back on the CPU, and
    stacking it with the value network's norm on the card failed in every run."""
    cuda = torch.device("cuda")
    torch.manual_seed(3)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(16,), value=(16,))).to(cuda)
    config = PPOConfig(epochs=1, minibatches=2)
    learner = Learner(model, config, gamma=0.9, device=cuda, fp16=False)
    learner.policy_frozen = True
    stats = learner.learn(rollout_for(model, cuda))
    assert stats.value_only_steps == 2 and stats.skipped_updates == 0


def test_a_frozen_policy_trains_the_value_side_alone_in_every_pass() -> None:
    learner = learner_for(epochs=2)
    learner.policy_frozen = True
    policy_before = params(learner.model, "policy")
    stats = learner.learn(rollout_for(learner.model))
    assert stats.gradient_steps == 2 * 2 and stats.value_only_steps == 2 * 2
    assert stats.losses_finite and stats.params_finite
    assert torch.equal(params(learner.model, "policy"), policy_before)
    plain = learner_for()
    assert plain.learn(rollout_for(plain.model)).value_only_steps == 0
