"""Fresh weights drawn as a new network would be, and Adam's state on a resume."""

from __future__ import annotations

import torch

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout

CPU = torch.device("cpu")
OBS, ACTIONS, STEPS, AGENTS = 4, 3, 6, 4


def rollout_for(model: ActorCritic, seed: int = 0) -> Rollout:
    generator = torch.Generator().manual_seed(seed)
    r = Rollout.allocate(STEPS, AGENTS, OBS, ACTIONS, CPU)
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
    return r


def learner_for(**ppo: object) -> Learner:
    torch.manual_seed(5)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(16,), value=(16,), shared=(8,)))
    config = PPOConfig(**({"epochs": 1, "minibatches": 2} | ppo))  # type: ignore[arg-type]
    return Learner(model, config, gamma=0.9, device=CPU, fp16=False)


def test_fresh_parameters_match_the_networks_initialisation_rules() -> None:
    torch.manual_seed(1)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(16,), value=(16,), shared=(8,)))
    pairs = model.fresh_parameters()
    assert [p.shape for p, _ in pairs] == [p.shape for p in model.parameters()]
    assert all(f.shape == p.shape and f.dtype == p.dtype for p, f in pairs)
    by_param = {id(p): f for p, f in pairs}
    # Orthogonal hidden weights with gain sqrt(2): rows of unit-ish norm scaled; biases zero; LayerNorm 1/0.
    for head in (model.shared, model.policy, model.value):
        for module in head:
            if isinstance(module, torch.nn.Linear):
                assert torch.equal(by_param[id(module.bias)], torch.zeros_like(module.bias))
            if isinstance(module, torch.nn.LayerNorm):
                assert torch.equal(by_param[id(module.weight)], torch.ones_like(module.weight))
    policy_out = by_param[id(model.value_output.weight)]
    assert policy_out.norm() > 0.5, "the value output layer starts with gain 1"
    tiny = by_param[id(model.policy[-1].weight)]
    assert tiny.abs().max() < 0.05, "the policy output layer starts near zero (gain 0.01)"


def test_a_save_does_not_choose_how_adam_runs_here() -> None:
    """A save made where Adam ran one way (another machine, or `nitro surgery` on the CPU) is loaded with
    this machine's way; with the fused kernel the step counters are placed where that kernel needs them.
    Fused Adam also runs on the CPU, which stands in for a graphics card here."""
    saved = learner_for()
    saved.learn(rollout_for(saved.model))
    state = saved.state_dict()
    assert state["optimizer"]["param_groups"][0]["fused"] is False
    here = learner_for()
    here.optimizer = torch.optim.Adam(
        [{"params": p, "lr": 1e-3} for p in here.model.parameter_groups().values()], fused=True
    )
    here.load_state_dict(state)
    assert all(group["fused"] is True for group in here.optimizer.param_groups)
    step = next(iter(here.optimizer.state.values()))["step"]
    assert step.dtype == torch.float32 and step.item() > 0, "the counter carried over, as fused Adam keeps it"
    here.learn(rollout_for(here.model, seed=1))  # the fused kernel runs on the loaded state
    back = learner_for()
    back.load_state_dict(here.state_dict())
    assert all(group["fused"] is False for group in back.optimizer.param_groups)
