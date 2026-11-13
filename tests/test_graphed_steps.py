"""`ppo.graphed_steps`: learning steps padded to a fixed size and replayed as CUDA graphs.

A padded step (its padding weighted 0) is the plain step's maths: the weights after it agree to FP32's
rounding (the output layers run in FP32 by design, so FP64 cannot be used), also with microbatches and in a
value-only pass; counting the padding as samples fails the check. On a graphics card a graphed update is the
eager one to the bit, also for a network some of whose weights never get a gradient (lean attention)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic, build_networks, observation_layout
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from test_attention import SMALL, observations, settings

OBS, ACTIONS, ROWS = 6, 90, 40
CUDA = torch.device("cuda")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def learner(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    fp16: bool = False,
    model: ActorCritic | None = None,
    **ppo: object,
) -> Learner:
    torch.manual_seed(3)
    if model is None:
        model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(32,), value=(16,), shared=(24,)))
    config = PPOConfig(epochs=2, minibatches=2, entropy=0.01, policy_lr=0.01, **ppo)  # type: ignore[arg-type]
    return Learner(model.to(device, dtype), config, gamma=0.9, device=device, fp16=fp16)


def lean_attention_model(tmp_path: Path) -> tuple[ActorCritic, torch.Tensor]:
    """An attention encoder with lean_attention, whose key biases never get a gradient, and observations
    for a rollout (8 steps of 4 agents)."""
    config = settings(tmp_path, model=SMALL + "lean_attention = true\n")
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, ACTIONS)
    return model, observations(layout, 8 * 4, (1, 2)).view(8, 4, -1)


def batch(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(ROWS, OBS, generator=g, dtype=dtype)
    mask = torch.rand(ROWS, ACTIONS, generator=g) < 0.8
    mask[:, 0] = True
    actions = (torch.rand(ROWS, ACTIONS, generator=g) * mask).argmax(1)
    old = -torch.rand(ROWS, generator=g, dtype=dtype) - 2.0
    advantages = torch.randn(ROWS, generator=g, dtype=dtype)
    targets = torch.randn(ROWS, generator=g, dtype=dtype)
    return obs, mask, actions, old, advantages, targets


def pad(t: torch.Tensor, extra: int) -> torch.Tensor:
    """`t` and `extra` copies of its first row (the graphed steps' padding)."""
    return torch.cat([t, t[:1].expand(extra, *t.shape[1:])])


@pytest.mark.parametrize("case", ["plain", "microbatch", "value only"])
def test_a_padded_step_is_the_plain_steps_maths(case: str) -> None:
    dtype, extra = torch.float32, 13
    ppo = {"microbatch": 16} if case == "microbatch" else {}
    plain, padded = learner(torch.device("cpu"), dtype, **ppo), learner(torch.device("cpu"), dtype, **ppo)
    inputs = batch(dtype)
    samples = torch.cat([torch.ones(ROWS, dtype=dtype), torch.zeros(extra, dtype=dtype)])
    padded_inputs = [pad(t, extra) for t in inputs]
    kwargs: dict[str, object] = {}
    padded_kwargs: dict[str, object] = {"samples": samples}
    value_only = case == "value only"
    for _ in range(2):
        a, _ = plain._step(*inputs, value_only, **kwargs)  # type: ignore[arg-type]
        b, _ = padded._step(*padded_inputs, value_only, **padded_kwargs)  # type: ignore[arg-type]
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
    for p, q in zip(plain.model.parameters(), padded.model.parameters(), strict=True):
        torch.testing.assert_close(q, p, rtol=1e-5, atol=1e-6)


def weights(learn: Learner) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in learn.model.parameters()])


def apart(a: Learner, b: Learner) -> float:
    return float((weights(b) - weights(a)).norm())


def moved(learn: Learner, start: torch.Tensor) -> float:
    return float((weights(learn) - start).norm())


def rollout(
    model: ActorCritic, seed: int, steps: int = 8, agents: int = 4, obs: torch.Tensor | None = None
) -> Rollout:
    g = torch.Generator().manual_seed(seed)
    size = OBS if obs is None else obs.shape[-1]
    r = Rollout.allocate(steps, agents, size, ACTIONS, next(model.parameters()).device)
    r.obs.copy_(torch.randn(steps, agents, OBS, generator=g) if obs is None else obs)
    r.rewards.copy_(torch.randn(steps, agents, generator=g))
    r.terminal.copy_(torch.rand(steps, agents, generator=g) < 0.05)
    r.mask.fill_(True)
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    r.actions.copy_(log_probs.argmax(-1))
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    return r


@NEEDS_GPU
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphed_learning_gives_the_eager_updates(fp16: bool) -> None:
    eager, graphed = (learner(CUDA, fp16=fp16, microbatch=0, graphed_steps=on) for on in (False, True))
    for seed in range(4):  # the first update runs eagerly (Adam has no state yet), then capture and replays
        stats = []
        for learn in (eager, graphed):
            torch.manual_seed(100 + seed)  # the same minibatches (drawn on the card)
            stats.append(learn.learn(rollout(learn.model, seed)))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b)
        assert repr(stats[0]) == repr(stats[1])
    assert graphed.graphed is not None and graphed.graphed.graphs, "the steps ran graphed"


@NEEDS_GPU
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphs_engage_when_some_weights_never_get_a_gradient(tmp_path: Path, fp16: bool) -> None:
    """lean_attention's key biases never get a gradient (nor Adam's state): the graphed steps still run, over
    the other weights, and give the eager updates to the bit."""
    model, obs = lean_attention_model(tmp_path)
    eager, graphed = (
        learner(CUDA, fp16=fp16, model=copy.deepcopy(model), microbatch=0, graphed_steps=on)
        for on in (False, True)
    )
    for seed in range(4):
        stats = []
        for learn in (eager, graphed):
            torch.manual_seed(100 + seed)
            stats.append(learn.learn(rollout(learn.model, seed, obs=obs)))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b)
        assert repr(stats[0]) == repr(stats[1])
    unused = [name for name, p in graphed.model.named_parameters() if p not in graphed.optimizer.state]
    assert unused and all(name.endswith("k.bias") for name in unused), unused
    assert graphed.graphed is not None and graphed.graphed.graphs, "the steps ran graphed"
