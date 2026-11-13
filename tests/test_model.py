"""The networks' memory-saving ReLU gives exactly the numbers of a plain ReLU, in every precision."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from nitro.config.schema import ModelConfig
from nitro.model import ActorCritic, _ReLU

NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
CONFIGS = [
    pytest.param(ModelConfig(policy=(64, 64), value=(64, 64)), id="separate"),
    pytest.param(ModelConfig(shared=(64, 64), policy=(64,), value=(64,)), id="shared-trunk"),
    pytest.param(ModelConfig(shared=(64,), policy=(), value=(64,)), id="trunk-into-output"),
    pytest.param(ModelConfig(policy=(64, 64), value=(64, 64), layer_norm=False), id="no-layer-norm"),
]
PATHS = [
    pytest.param("cpu", False, id="cpu-fp32"),
    pytest.param("cuda", False, id="cuda-fp32", marks=NEEDS_GPU),
    pytest.param("cuda", True, id="cuda-fp16", marks=NEEDS_GPU),
]


def plain(model: ActorCritic) -> ActorCritic:
    reference = copy.deepcopy(model)
    for network in (reference.shared, reference.policy, reference.value):
        for i, layer in enumerate(network):
            if isinstance(layer, _ReLU):
                network[i] = nn.ReLU()
    return reference


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(("device", "fp16"), PATHS)
def test_outputs_and_gradients_match_a_plain_relu_bit_for_bit(
    config: ModelConfig, device: str, fp16: bool
) -> None:
    torch.manual_seed(0)
    model = ActorCritic(37, 90, config).to(device)
    reference = plain(model)
    obs = torch.randn(512, 37, device=device) * 3
    results = []
    for network in (model, reference):
        with torch.autocast(device, dtype=torch.float16, enabled=fp16):
            logits, value = network(obs)
        (logits.logsumexp(1).mean() + value.pow(2).mean()).backward()
        results.append((logits, value, [p.grad for p in network.parameters()]))
    (logits, value, grads), (ref_logits, ref_value, ref_grads) = results
    assert torch.equal(logits, ref_logits) and torch.equal(value, ref_value)
    assert all(torch.equal(g, r) for g, r in zip(grads, ref_grads, strict=True))
