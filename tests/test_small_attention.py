"""`model.small_attention`: the transformer's attention over a few tokens as NitroLearn's own
GPU routine. On the card its output and gradients are PyTorch's attention to FP64's rounding (every count of
tokens it takes, narrow and full heads, empty places), FP32 and FP16 within their rounding; whole transformer
networks with it on give the usual answers; it runs inside a CUDA graph and repeats itself bit for bit. On the
processor the setting changes nothing."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro import small_attention
from nitro.model import ActorCritic, attention_encoder, build_networks, observation_layout
from test_attention import NEEDS_GPU, observations, settings

TRANSFORMER = (
    'policy = [32]\nvalue = [24]\nencoder = "transformer"\ntransformer_width = 32\ntransformer_heads = 4\n'
    "transformer_ff = 64\n"
)
FILLED = [(0, 0), (1, 2), (2, 3)]


def pair(
    tmp_path: Path, extra: str = "", dtype: torch.dtype = torch.float32
) -> tuple[ActorCritic, ActorCritic]:
    """Transformer networks without and with small_attention, the same weights (moved off their start)."""
    config = settings(tmp_path, model=TRANSFORMER + extra)
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    plain = build_networks(config, own + (teammates + opponents) * block, 90)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    small_config = dataclasses.replace(config, model=dataclasses.replace(config.model, small_attention=True))
    small = build_networks(small_config, plain.obs_size, 90)
    small.load_state_dict(plain.state_dict())
    return plain.to(dtype), small.to(dtype)


def inputs(rows: int, tokens: int, width: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(tokens * 100 + width)
    qkv = torch.randn(rows, tokens, 3 * width, generator=generator).to(device, dtype)
    present = torch.rand(rows, tokens, generator=generator) < 0.7
    present[:, 0] = True  # a token always looks at itself at least
    grad = torch.randn(rows, tokens, width, generator=generator).to(device, dtype)
    return qkv, present.to(device), grad


def test_on_the_processor_nothing_changes(tmp_path: Path) -> None:
    plain, small = pair(tmp_path)
    obs = observations(observation_layout(settings(tmp_path, model=TRANSFORMER)), 16, (1, 2))
    for a, b in zip(plain(obs), small(obs), strict=True):
        assert torch.equal(a, b)
    assert not small_attention.usable(torch.zeros(2, 5, 96), 4)


@NEEDS_GPU
@pytest.mark.parametrize("tokens", list(range(1, small_attention.MAX_TOKENS + 1)))
@pytest.mark.parametrize(("width", "heads"), [(32, 4), (256, 8), (12, 3)])
def test_the_routine_is_pytorchs_attention_to_fp64s_rounding(tokens: int, width: int, heads: int) -> None:
    qkv, present, grad = inputs(37, tokens, width, "cuda", torch.float64)
    assert small_attention.usable(qkv, heads)
    exact_in, mine_in = qkv.clone().requires_grad_(), qkv.clone().requires_grad_()
    exact = small_attention.reference(exact_in, present, heads)
    mine = small_attention.attention(mine_in, present, heads)
    exact.backward(grad)
    mine.backward(grad)
    torch.testing.assert_close(mine, exact, rtol=1e-11, atol=1e-11)
    assert mine_in.grad is not None and exact_in.grad is not None
    torch.testing.assert_close(mine_in.grad, exact_in.grad, rtol=1e-10, atol=1e-10)


@NEEDS_GPU
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-5), (torch.float16, 1e-2)])
def test_fp32_and_fp16_stay_within_their_rounding(dtype: torch.dtype, tolerance: float) -> None:
    qkv, present, grad = inputs(4096, 5, 256, "cuda", dtype)
    exact_in = qkv.double().requires_grad_()
    exact = small_attention.reference(exact_in, present, 8)
    exact.backward(grad.double())
    mine_in = qkv.clone().requires_grad_()
    mine = small_attention.attention(mine_in, present, 8)
    mine.backward(grad)
    assert mine.dtype == dtype and mine_in.grad is not None and mine_in.grad.dtype == dtype
    assert exact_in.grad is not None
    assert float((mine.double() - exact).abs().max()) <= tolerance * float(exact.abs().max())
    assert float((mine_in.grad.double() - exact_in.grad).abs().max()) <= tolerance * float(
        exact_in.grad.abs().max()
    )


@NEEDS_GPU
def test_it_repeats_itself_bit_for_bit_and_runs_in_a_cuda_graph() -> None:
    qkv, present, _ = inputs(512, 5, 256, "cuda", torch.float16)
    first = small_attention.attention(qkv, present, 8)
    assert torch.equal(first, small_attention.attention(qkv, present, 8))
    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        small_attention.attention(qkv, present, 8)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = small_attention.attention(qkv, present, 8)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, first)


@NEEDS_GPU
@pytest.mark.parametrize("filled", FILLED)
@pytest.mark.parametrize(
    "extra", ["", "lean_attention = true\n", "shared = [40]\n"], ids=["", "lean", "shared"]
)
def test_transformer_networks_with_it_give_the_usual_answers(
    tmp_path: Path, extra: str, filled: tuple[int, int]
) -> None:
    """The whole encoder in FP64 on the card (its output layers stay FP32 by design): the same maths."""
    plain_net, small_net = pair(tmp_path, extra, torch.float64)
    plain, small = attention_encoder(plain_net.cuda()), attention_encoder(small_net.cuda())
    assert plain is not None and small is not None and getattr(small, "small", False)
    obs = observations(observation_layout(settings(tmp_path, model=TRANSFORMER)), 24, filled).double().cuda()
    a, b = plain(obs), small(obs)
    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), q in zip(plain.named_parameters(), small.parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None, name
            continue
        assert q.grad is not None, name
        torch.testing.assert_close(q.grad, p.grad, rtol=1e-9, atol=1e-12, msg=name)
