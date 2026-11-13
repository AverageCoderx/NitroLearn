"""`model.fused_norm`: LayerNorm and ReLU as one GPU routine. Off the graphics card it is
exactly the plain network; on it, the same maths to rounding, forward and backward, inside CUDA graphs too,
and the same answers every time. The GPU tests run on a machine with a graphics card."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from nitro import fused_norm
from nitro.config.loader import parse_toml
from nitro.config.schema import ModelConfig, NitroConfig
from nitro.export import _layer_records
from nitro.model import build_networks

NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
OBS, ACTIONS = 37, 90


def config(text: str = "", fused: bool = True) -> NitroConfig:
    parsed = parse_toml(
        "[model]\npolicy = [64, 48]\nvalue = [64, 48]\n" + text,
        NitroConfig,
        file=Path("x"),
        base_dir=Path("."),
    )
    return dataclasses.replace(parsed, model=dataclasses.replace(parsed.model, fused_norm=fused))


def pair(text: str = "") -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(3)
    plain = build_networks(config(text, fused=False), OBS, ACTIONS)
    torch.manual_seed(3)
    fused = build_networks(config(text), OBS, ACTIONS)
    return plain, fused


def test_the_kernels_compile_for_capability_7_5() -> None:
    try:
        cubin = fused_norm.compile_to_cubin("75")
    except OSError:
        pytest.skip("no NVRTC library here")
    assert all(name.encode() in cubin for name in fused_norm.KERNELS)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu"])
def test_off_the_graphics_card_a_fused_network_is_the_plain_network(activation: str) -> None:
    plain, fused = pair(f'activation = "{activation}"\n')
    assert list(plain.state_dict()) == list(fused.state_dict()), "the same layer numbering, so the same saves"
    pairs = zip(plain.state_dict().values(), fused.state_dict().values(), strict=True)
    assert all(torch.equal(a, b) for a, b in pairs)
    obs = torch.randn(32, OBS)
    for net in (plain, fused):
        logits, value = net(obs)
        (logits.square().mean() + value.square().mean()).backward()
    assert torch.equal(plain(obs)[0], fused(obs)[0]) and torch.equal(plain(obs)[1], fused(obs)[1])
    grads = [(a.grad, b.grad) for a, b in zip(plain.parameters(), fused.parameters(), strict=True)]
    assert all(a is not None and b is not None and torch.equal(a, b) for a, b in grads)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu"])
def test_saves_load_both_ways_and_the_bot_export_is_the_same(activation: str) -> None:
    text = f'activation = "{activation}"\n'
    plain, fused = pair(text)
    torch.manual_seed(9)
    other = build_networks(config(text, fused=False), OBS, ACTIONS)
    fused.load_state_dict(other.state_dict())
    plain.load_state_dict(fused.state_dict())
    (records_plain, weights_plain), (records_fused, weights_fused) = (
        _layer_records(plain),
        _layer_records(fused),
    )
    assert records_plain == records_fused
    assert {"kind": activation} in records_fused
    assert all(torch.equal(a, b) for a, b in zip(weights_plain, weights_fused, strict=True))


def test_the_settings_say_what_it_needs() -> None:
    assert ModelConfig().fused_norm is False, "off by default"
    assert config().problems() == []

    def problems(**model: object) -> list[str]:
        base = config()
        return [
            path
            for path, _ in dataclasses.replace(
                base, model=dataclasses.replace(base.model, **model)
            ).problems()
        ]

    assert problems(layer_norm=False) == ["model.fused_norm"]
    assert problems(activation="gelu") == ["model.fused_norm"]
    assert problems(activation="leaky_relu") == []


def test_without_a_graphics_card_the_routine_is_simply_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    assert not fused_norm.available(torch.device("cpu"))


def _stand_in(
    x: Any, weight: Any, bias: Any, y: Any, mean: Any, rstd: Any, n: int, eps: float, slope: float = 0.0
) -> None:
    """The forward kernel's maths on the CPU."""
    xf = x.float()
    m = xf.mean(1)
    r = torch.rsqrt(((xf - m[:, None]) ** 2).mean(1) + eps)
    y.copy_(F.leaky_relu((xf - m[:, None]) * r[:, None] * weight + bias, slope).to(y.dtype))
    mean.copy_(m)
    rstd.copy_(r)


def _stand_in_backward(
    dy: Any, x: Any, weight: Any, bias: Any, mean: Any, rstd: Any, dx: Any, *rest: Any
) -> None:
    """The backward kernel's maths on the CPU."""
    dw_part, db_part, slope = rest[0], rest[1], rest[-1]
    xh = (x.float() - mean[:, None]) * rstd[:, None]
    g = dy.float() * torch.where(xh * weight + bias > 0, 1.0, slope)
    gw = g * weight
    dx.copy_(
        (rstd[:, None] * (gw - gw.mean(1, keepdim=True) - xh * (gw * xh).mean(1, keepdim=True))).to(dx.dtype)
    )
    dw_part.zero_()
    db_part.zero_()
    dw_part[0] = (g * xh).sum(0)
    db_part[0] = g.sum(0)


class _Launcher:
    def __init__(self, maths: Any) -> None:
        self.maths = maths

    def __call__(self, grid: Any, block: Any, args: list[Any], stream: Any = None) -> None:
        self.maths(*args)


class _NoDevice(contextlib.nullcontext[None]):
    """torch.cuda.device's stand-in. A class, as torch.cuda.device is: on a machine with a card torch checks
    devices against it (`isinstance` in torch.cuda._utils._get_device_index), as torch.compile does while it
    reads a network; a function there fails every compile (the card tests)."""

    def __init__(self, device: Any) -> None:
        super().__init__()


@pytest.fixture
def no_cuda_plumbing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # torch.compile's modules keep the torch.cuda.device and current_stream they are first imported with,
    # for the rest of the process: imported first, they keep the real ones.
    import torch._dynamo

    monkeypatch.setattr(torch.cuda, "device", _NoDevice)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    yield


@pytest.mark.usefixtures("no_cuda_plumbing")
def test_the_self_test_passes_right_maths_and_refuses_wrong_maths() -> None:
    def module() -> dict[str, Any]:
        return {
            k: _Launcher(_stand_in if k.startswith("fwd") else _stand_in_backward) for k in fused_norm.KERNELS
        }

    fused_norm._self_test(module(), torch.device("cpu"))
    wrong_eps = module()
    wrong_eps["fwd_f16_f32"] = _Launcher(lambda *args: _stand_in(*args[:7], 1.0, args[8]))
    with pytest.raises(RuntimeError, match="f16->f32"):
        fused_norm._self_test(wrong_eps, torch.device("cpu"))
    wrong_gradient = module()
    wrong_gradient["bwd_f32_f32"] = _Launcher(lambda *args: (_stand_in_backward(*args), args[6].mul_(1.1)))
    with pytest.raises(RuntimeError, match="f32->f32"):
        fused_norm._self_test(wrong_gradient, torch.device("cpu"))
    no_leak = module()  # a routine that ignores the slope fails the leaky half of the test
    no_leak["fwd_f32_f32"] = _Launcher(lambda *args: _stand_in(*args[:8]))
    with pytest.raises(RuntimeError, match=r"f32->f32 slope 0\.01"):
        fused_norm._self_test(no_leak, torch.device("cpu"))


@pytest.mark.usefixtures("no_cuda_plumbing")
def test_any_failure_while_building_leaves_the_usual_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.cuda._utils as utils

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 5))
    monkeypatch.setattr(fused_norm, "compile_to_cubin", lambda architecture: b"")
    monkeypatch.setattr(utils, "_cuda_load_module", lambda cubin, names: ["not", "a", "dict"])
    monkeypatch.setattr(fused_norm, "_failed", {})
    assert fused_norm.kernels(torch.device("cuda", 5)) is None
    assert fused_norm._failed[5].startswith("TypeError")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert fused_norm.kernels(torch.device("cuda", 6)) is None
    assert 6 not in fused_norm._failed, "a capture only postpones the build"


def reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.dtype, slope: float = 0.0
) -> torch.Tensor:
    return F.leaky_relu(F.layer_norm(x.float(), weight.shape, weight, bias, 1e-5), slope).to(out)


@NEEDS_GPU
@pytest.mark.parametrize(
    ("dtype_in", "dtype_out"),
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
    ],
)
@pytest.mark.parametrize("shape", [(1, 256), (7, 1000), (12_500, 256), (3, 5, 128), (2, 33), (4, 1), (6, 20)])
@pytest.mark.parametrize("slope", [0.0, fused_norm.LEAKY_SLOPE])
def test_the_routine_matches_layer_norm_and_relu_forward_and_backward(
    dtype_in: torch.dtype, dtype_out: torch.dtype, shape: tuple[int, ...], slope: float
) -> None:
    assert fused_norm.available(torch.device("cuda")), fused_norm._failed
    torch.manual_seed(0)
    n = shape[-1]
    x = (torch.randn(shape, device="cuda") * 3 + 1).to(dtype_in)
    x.view(-1, n)[0] = 2.5  # a row with no spread
    weight = (torch.rand(n, device="cuda") + 0.5).requires_grad_()
    bias = (torch.randn(n, device="cuda") * 0.1).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_fused = x.detach().clone().requires_grad_()
    w_ref, b_ref = weight.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
    y_ref = reference(x_ref, w_ref, b_ref, dtype_out, slope)
    y_fused = fused_norm._Function.apply(x_fused, weight, bias, 1e-5, dtype_out, slope)
    assert y_fused.dtype == dtype_out and y_fused.shape == x.shape
    half = torch.float16 in (dtype_in, dtype_out)
    tolerance = {"rtol": 2e-3, "atol": 2e-3} if half else {"rtol": 1e-5, "atol": 1e-5}
    torch.testing.assert_close(y_fused, y_ref, **tolerance)
    z = F.layer_norm(x.float(), (n,), weight.detach(), bias.detach(), 1e-5)
    # No gradient where rounding could flip the ReLU's on/off between the two.
    dy = torch.randn(shape, device="cuda").masked_fill(z.abs() < 1e-4, 0).to(dtype_out)
    y_ref.backward(dy)
    y_fused.backward(dy)
    grad_tolerance = {"rtol": 1e-2, "atol": 1e-2} if half else {"rtol": 1e-4, "atol": 1e-4}
    assert x_fused.grad is not None and x_ref.grad is not None and x_fused.grad.dtype == dtype_in
    torch.testing.assert_close(x_fused.grad, x_ref.grad, **grad_tolerance)
    scale = max(1.0, float(w_ref.grad.abs().max()))  # type: ignore[union-attr]
    torch.testing.assert_close(
        weight.grad, w_ref.grad, rtol=grad_tolerance["rtol"], atol=grad_tolerance["atol"] * scale
    )
    torch.testing.assert_close(
        bias.grad, b_ref.grad, rtol=grad_tolerance["rtol"], atol=grad_tolerance["atol"] * scale
    )


@NEEDS_GPU
def test_the_same_answers_every_time() -> None:
    torch.manual_seed(1)
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)
    weight = torch.ones(256, device="cuda", requires_grad=True)
    bias = torch.zeros(256, device="cuda", requires_grad=True)
    runs = []
    for _ in range(2):
        for t in (x, weight, bias):
            t.grad = None
        fused_norm._Function.apply(x, weight, bias, 1e-5, torch.float32).square().sum().backward()
        runs.append([t.grad.clone() for t in (x, weight, bias)])  # type: ignore[union-attr]
    assert all(torch.equal(a, b) for a, b in zip(*runs, strict=True))


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Counts the routine's forward and backward launches, so a test cannot pass on the fallback."""
    assert fused_norm.available(torch.device("cuda")), fused_norm._failed
    counts = {"forward": 0, "backward": 0}
    forward, backward = fused_norm._forward, fused_norm._backward

    def counted_forward(*args: Any) -> Any:
        counts["forward"] += 1
        return forward(*args)

    def counted_backward(*args: Any) -> Any:
        counts["backward"] += 1
        return backward(*args)

    monkeypatch.setattr(fused_norm, "_forward", counted_forward)
    monkeypatch.setattr(fused_norm, "_backward", counted_backward)
    return counts


@NEEDS_GPU
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_a_fused_network_on_the_graphics_card_learns_as_the_plain_one(
    precision: str, launches: dict[str, int]
) -> None:
    plain, fused = pair()
    plain, fused = plain.cuda(), fused.cuda()
    obs = torch.randn(512, OBS, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        outputs = [net(obs) for net in (plain, fused)]
    tolerance = {"rtol": 2e-2, "atol": 2e-2} if precision == "fp16" else {"rtol": 1e-5, "atol": 1e-5}
    for a, b in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(b.float(), a.float(), **tolerance)
    for logits, value in outputs:
        (logits.float().square().mean() + value.float().square().mean()).backward()
    for a, b in zip(plain.parameters(), fused.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        scale = max(1.0, float(a.grad.abs().max()))
        torch.testing.assert_close(b.grad, a.grad, rtol=5e-2, atol=5e-3 * scale)
    assert launches["forward"] == 4 and launches["backward"] == 4, "two hidden layers in each of two networks"


@NEEDS_GPU
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_a_cuda_graph_captures_the_routine(precision: str, launches: dict[str, int]) -> None:
    _, fused = pair()
    fused = fused.cuda().eval()
    obs = torch.randn(256, OBS, device="cuda")
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16", cache_enabled=False)
    with torch.no_grad(), autocast:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.logits(obs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused.logits(obs)
        assert launches["forward"] == 8, "launched while capturing, not left to the usual layers"
        obs.copy_(torch.randn_like(obs))  # a replay must recompute from the new input
        graph.replay()
        eager = fused.logits(obs)
    tolerance = 1e-3 if precision == "fp16" else 1e-6
    torch.testing.assert_close(captured, eager, rtol=tolerance, atol=tolerance)
