"""A collection's start: the FP16 copy brought up to date in one copy call per pair of dtypes, to the
bit as the copy per tensor did."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from nitro import model as model_module
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.model import ActorCritic, build_networks, copy_inference_weights, fp16_inference_copy
from test_speed_paths import MODELS
from test_trainer import NEEDS_GPU

# A card's copy tensor by tensor, and CUDA's one call per dtype pair (`_ONE_CALL_ON_CARDS`).
WEIGHT_COPIES = [
    pytest.param("cpu", False, id="cpu"),
    pytest.param("cuda", False, id="cuda-per-tensor", marks=NEEDS_GPU),
    pytest.param("cuda", True, id="cuda-one-call", marks=NEEDS_GPU),
]


def old_copy_inference_weights(copy: ActorCritic, model: ActorCritic) -> None:
    """`copy_inference_weights` in an earlier version: two state_dict walks and one copy per tensor."""
    with torch.no_grad():
        for (name, target), source in zip(
            copy.state_dict().items(), model.state_dict().values(), strict=True
        ):
            assert target.shape == source.shape, name
            target.copy_(source)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
    )


def same_weights(a: ActorCritic, b: ActorCritic) -> bool:
    return all(same_bits(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True))


def networks(tmp_path: Path, name: str, device: str) -> ActorCritic:
    config = parse_toml(
        f"[env]\nteam_size = 2\nobs_team_size = 3\n[model]\n{MODELS[name]}",
        NitroConfig,
        file=tmp_path / "s.toml",
        base_dir=tmp_path,
    )
    from nitro.model import observation_layout

    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    return build_networks(config, own + (teammates + opponents) * block, 90).to(device)


def nudged(model: ActorCritic, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_((torch.randn(parameter.shape, generator=g) * 0.05).to(parameter.device))


@pytest.mark.parametrize(("device", "one_call"), WEIGHT_COPIES)
@pytest.mark.parametrize(
    "name", ["plain, FP32 norm", "plain, fused norm", "attention, lean", "transformer, lean"]
)
@pytest.mark.parametrize("fp16", [True, False], ids=["fp16 copy", "fp32 copy"])
def test_the_weight_copy_in_one_call_per_dtype_gives_the_copy_per_tensors_bits(
    tmp_path: Path, name: str, device: str, one_call: bool, fp16: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both copies (the FP16 one that plays, and a plain FP32 one) stay what the per-tensor copies made of
    them, to the bit: after the weights change in place (a learning update, rows copied
    within a layer, a save loaded) the kept lists are reused; after the tensors move, or for another model,
    they are listed again. On a card, cuda-one-call is the check `_ONE_CALL_ON_CARDS` rests on."""
    monkeypatch.setattr(model_module, "_ONE_CALL_ON_CARDS", one_call)
    model = networks(tmp_path, name, device)

    def made() -> ActorCritic:
        return fp16_inference_copy(model) if fp16 else copy.deepcopy(model).requires_grad_(False)

    new, old = made(), made()

    def check() -> None:
        copy_inference_weights(new, model)
        old_copy_inference_weights(old, model)
        assert same_weights(new, old)

    nudged(model, 1)
    check()
    kept = model_module._WEIGHT_PAIRS[new]
    nudged(model, 2)
    with torch.no_grad():  # rows of a layer copied from others in place
        output = model.policy[-1]
        assert isinstance(output, torch.nn.Linear)
        output.weight[5:9] = output.weight[torch.tensor([0, 1, 2, 3], device=device)]
        output.bias[5:9] = output.bias[:4] - 1.0
    check()
    other = networks(tmp_path, name, device)
    nudged(other, 3)
    model.load_state_dict(other.state_dict())  # a save loaded, or the out-of-memory retry
    check()
    assert model_module._WEIGHT_PAIRS[new] is kept, "loads and in-place copies write into the same tensors"
    model.double().float()  # every tensor of the model moved
    nudged(model, 4)
    check()
    assert model_module._WEIGHT_PAIRS[new] is not kept
    kept = model_module._WEIGHT_PAIRS[new]
    with torch.no_grad():
        for parameter in new.parameters():  # the copy's tensors moved
            parameter.data = parameter.data.clone()
    check()
    assert model_module._WEIGHT_PAIRS[new] is not kept
    kept = model_module._WEIGHT_PAIRS[new]
    model = other  # another model into the same copy
    check()
    assert model_module._WEIGHT_PAIRS[new] is not kept
    if fp16:
        assert len(model_module._WEIGHT_PAIRS[new].lists) == 2, "FP32 into FP16, FP32 into FP32"
