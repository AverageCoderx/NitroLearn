"""`model.lean_attention`: the three attention encoders' same maths in fewer steps. Its
answers and every gradient equal the usual paths' (FP64 to its rounding, FP32 to FP32's) in every game size,
with places filled, partly empty and empty, beside or under a shared trunk, with attention_skip; the key
layers' biases of the one-query looks get no gradient (they have none: the softmax ignores them); the
attention kept for `nitro watch`, saves, the bot file and deterministic mode are as before. FP16 and the
inference graph run on a machine with a graphics card."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import torch
from torch import nn

from nitro.config.schema import NitroConfig
from nitro.model import ActorCritic, attention_encoder, build_networks, observation_layout
from nitro.runs import FIXED_SETTINGS
from test_attention import NEEDS_GPU, observations, settings

BASE = "policy = [32]\nvalue = [24]\nattention_width = 32\nattention_heads = 4\n"
TRANSFORMER = 'encoder = "transformer"\ntransformer_width = 32\ntransformer_heads = 4\ntransformer_ff = 64\n'
ENCODERS = {
    "attention": 'encoder = "attention"\n',
    "attention, fast": 'encoder = "attention"\nattention_fast = true\n',
    "attention_communication": 'encoder = "attention_communication"\n',
    "transformer": TRANSFORMER,
    "transformer, pads, summary, 3 layers": TRANSFORMER
    + 'transformer_pads = true\ntransformer_readout = "summary"\ntransformer_layers = 3\n',
}
# FP64 needs a usual path without FP32 casts of its own: attention_fast's null car score has one.
EXACT = [name for name in ENCODERS if name != "attention, fast"]
FILLED = [(0, 0), (1, 2), (2, 3), (0, 1)]


def config_for(tmp_path: Path, name: str, shared: bool = False, skip: bool = False) -> NitroConfig:
    text = (
        BASE
        + ENCODERS[name]
        + ("shared = [40]\n" if shared else "")
        + ("attention_skip = true\n" if skip else "")
    )
    return settings(tmp_path, model=text)


def pair(config: NitroConfig, dtype: torch.dtype = torch.float32) -> tuple[ActorCritic, ActorCritic]:
    """The usual networks and lean ones with the same weights, every weight off its start (the norms off
    their identity, the null car off zero), so a misplaced one shows."""
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    plain = build_networks(config, own + (teammates + opponents) * block, 90)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    lean_config = dataclasses.replace(config, model=dataclasses.replace(config.model, lean_attention=True))
    lean = build_networks(lean_config, plain.obs_size, 90)
    lean.load_state_dict(plain.state_dict())
    return plain.to(dtype), lean.to(dtype)


def no_effect(name: str) -> bool:
    """The key layer biases of the one-query looks: no effect on any output, so no gradient with lean."""
    return re.search(r"(^|\.)(attention|listen)\.\d+\.k\.bias$", name) is not None


def mixed(rows: int, layout: tuple[int, int, int, int], filled: tuple[int, int]) -> torch.Tensor:
    """`observations` with every row's places filled differently in the first quarter (all empty there)."""
    obs = observations(layout, rows, filled)
    obs[: rows // 4, layout[0] :] = 0.0
    return obs


def gradients(model: nn.Module, outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor | None]:
    model.zero_grad(set_to_none=True)
    sum(o.square().mean() for o in outputs).backward()  # type: ignore[union-attr]
    return {name: p.grad for name, p in model.named_parameters()}


def check_gradients(
    plain: dict[str, torch.Tensor | None], lean: dict[str, torch.Tensor | None], rtol: float, atol: float
) -> None:
    largest = max(float(g.abs().max()) for g in plain.values() if g is not None)
    for name, a in plain.items():
        b = lean[name]
        if no_effect(name):
            assert b is None, name
            assert a is None or float(a.abs().max()) <= 1e-6 * largest, f"{name}: rounding noise only"
            continue
        assert a is not None and b is not None, name
        torch.testing.assert_close(b, a, rtol=rtol, atol=atol * largest, msg=name)


@pytest.mark.parametrize("filled", FILLED)
@pytest.mark.parametrize("skip", [False, True], ids=["", "skip"])
@pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
@pytest.mark.parametrize("name", list(ENCODERS))
def test_lean_networks_give_the_same_answers_and_gradients(
    tmp_path: Path, name: str, shared: bool, skip: bool, filled: tuple[int, int]
) -> None:
    config = config_for(tmp_path, name, shared, skip)
    plain, lean = pair(config)
    obs = mixed(16, observation_layout(config), filled)
    plain_out, lean_out = plain(obs), lean(obs)
    for a, b in zip(plain_out, lean_out, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
    check_gradients(gradients(plain, plain_out), gradients(lean, lean_out), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("skip", [False, True], ids=["", "skip"])
@pytest.mark.parametrize("name", EXACT)
def test_lean_encoders_are_the_same_maths_in_fp64_in_every_game_size(
    tmp_path: Path, name: str, skip: bool
) -> None:
    """The encoder alone in FP64 (the output layers run in FP32 by design): its output and every gradient
    agree to FP64's rounding, in the trained room, other rooms, and given places (the bot beyond its room)."""
    config = config_for(tmp_path, name, skip=skip)
    plain_net, lean_net = pair(config, torch.float64)
    plain, lean = attention_encoder(plain_net), attention_encoder(lean_net)
    assert plain is not None and lean is not None and getattr(lean, "lean", False)
    own, block, teammates, opponents = observation_layout(config)
    cases: list[tuple[torch.Tensor, tuple[int, int] | None]] = [
        (mixed(24, (own, block, teammates, opponents), filled), None) for filled in FILLED
    ]
    for room in ((0, 1), (3, 4)):  # 1v1's room and 4v4's
        cases.append((mixed(24, (own, block, *room), room), None))
    cases.append((mixed(24, (own, block, 1, 1), (1, 1)), (1, 1)))  # one block per car present
    cases.append((mixed(24, (own, block, 0, 0), (0, 0)), (0, 0)))  # no other car at all
    for obs, places in cases:
        obs = obs.double()
        a, b = plain(obs, places), lean(obs, places)
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10)
        check_gradients(gradients(plain, (a,)), gradients(lean, (b,)), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_the_kept_attention_is_the_same(tmp_path: Path, name: str) -> None:
    config = config_for(tmp_path, name)
    plain_net, lean_net = pair(config)
    obs = mixed(16, observation_layout(config), (1, 2))
    kept = []
    for net in (plain_net, lean_net):
        encoder = attention_encoder(net)
        assert encoder is not None
        encoder.keep_weights = True
        with torch.no_grad():
            net(obs)
        encoder.keep_weights = False
        kept.append(
            [getattr(encoder, key) for key in ("attention_weights", "talk_weights") if hasattr(encoder, key)]
        )
    for a, b in zip(*kept, strict=True):
        assert a is not None and b is not None and a.shape == b.shape
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_saves_the_bot_file_and_resumes_are_the_same_with_it_on_or_off(tmp_path: Path, name: str) -> None:
    from nitro.export import _layer_records

    plain, lean = pair(config_for(tmp_path, name, shared=True, skip=True))
    assert list(plain.state_dict()) == list(lean.state_dict()), "the same parameters, the same names"
    assert "model.lean_attention" not in FIXED_SETTINGS, "a run may switch it between resumes"
    layers, weights = _layer_records(plain)
    lean_layers, lean_weights = _layer_records(lean)
    assert layers == lean_layers and len(weights) == len(lean_weights)
    assert all(torch.equal(a, b) for a, b in zip(weights, lean_weights, strict=True)), "the same bot file"
    encoder = attention_encoder(lean)
    assert encoder is not None
    pairs = encoder.fresh_parameters()  # type: ignore[operator]
    assert [mine for mine, _ in pairs] == list(encoder.parameters()), "fresh weights for every parameter"


@pytest.mark.parametrize("name", list(ENCODERS))
def test_deterministic_mode_repeats_itself(tmp_path: Path, name: str) -> None:
    config = config_for(tmp_path, name, shared=True)
    plain, lean = pair(config)
    obs = mixed(16, observation_layout(config), (2, 3))
    torch.use_deterministic_algorithms(True)
    try:
        first, again = lean(obs), lean(obs)
        grads = gradients(lean, first)
        for a, b in zip(first, again, strict=True):
            assert torch.equal(a, b)
    finally:
        torch.use_deterministic_algorithms(False)
    check_gradients(gradients(plain, plain(obs)), grads, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_fp16_autocast_on_the_cpu_runs_and_stays_close(tmp_path: Path, name: str) -> None:
    config = config_for(tmp_path, name, shared=True)
    plain, lean = pair(config)
    obs = mixed(32, observation_layout(config), (2, 3))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        expected, got = plain(obs), lean(obs)
    for a, b in zip(expected, got, strict=True):
        torch.testing.assert_close(b.float(), a.float(), rtol=0.03, atol=0.03)


@NEEDS_GPU
@pytest.mark.parametrize("name", list(ENCODERS))
def test_fp16_and_the_inference_graph_with_lean_attention(tmp_path: Path, name: str) -> None:
    from nitro.envs import GroupTensors
    from nitro.inference import GroupInference

    config = config_for(tmp_path, name, shared=True)
    plain, lean = pair(config)
    plain, lean = plain.eval().cuda(), lean.eval().cuda()
    obs = mixed(64, observation_layout(config), (2, 3)).cuda()
    with torch.no_grad():
        for a, b in zip(plain(obs), lean(obs), strict=True):
            torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
        with torch.autocast("cuda", dtype=torch.float16):
            expected, got = plain(obs), lean(obs)
    for a, b in zip(expected, got, strict=True):
        torch.testing.assert_close(b.float(), a.float(), rtol=0.02, atol=0.02)
    tensors = GroupTensors.allocate(64, lean.obs_size, 90, pin_memory=True)
    tensors.obs.copy_(obs.cpu())
    tensors.mask.fill_(True)
    engine = GroupInference(lean, tensors, torch.device("cuda"), fp16=True, cuda_graph=True)
    engine.run()
    engine.wait()
    assert ((tensors.actions >= 0) & (tensors.actions < 90)).all()
