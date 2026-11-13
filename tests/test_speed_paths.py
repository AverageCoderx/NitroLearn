"""Faster paths that must compute exactly what the plain ones did."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.model import build_networks, copy_inference_weights, fp16_inference_copy

PLAIN = "policy = [48, 32]\nvalue = [48, 32]\nshared = [64]\n"
ATTENTION = 'encoder = "attention"\nattention_width = 32\nattention_heads = 4\n'
TRANSFORMER = 'encoder = "transformer"\ntransformer_width = 32\ntransformer_heads = 4\n'
MODELS = {
    "plain, FP32 norm": PLAIN,
    "plain, fused norm": PLAIN + "fused_norm = true\n",
    "attention": PLAIN + ATTENTION,
    "attention, fast, skip": PLAIN + ATTENTION + "attention_fast = true\nattention_skip = true\n",
    "attention_communication": PLAIN + ATTENTION.replace('"attention"', '"attention_communication"'),
    "transformer": PLAIN + TRANSFORMER,
    # model.lean_attention: the copy keeps in FP32 the layers whose weights the lean looks fold themselves.
    "attention, lean": PLAIN + ATTENTION + "lean_attention = true\n",
    "attention, fast, lean, skip": PLAIN
    + ATTENTION
    + "attention_fast = true\nlean_attention = true\nattention_skip = true\n",
    "attention_communication, lean": PLAIN
    + ATTENTION.replace('"attention"', '"attention_communication"')
    + "lean_attention = true\n",
    "transformer, lean": PLAIN + TRANSFORMER + "lean_attention = true\n",
    "transformer, pads, summary, lean": PLAIN
    + TRANSFORMER
    + 'transformer_pads = true\ntransformer_readout = "summary"\nlean_attention = true\n',
}


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(MODELS))
def test_the_fp16_inference_copy_answers_to_the_bit_and_follows_the_model(
    tmp_path: Path, name: str, device: str
) -> None:
    """The copy that chooses actions holds in FP16 what FP16 autocast would cast on every pass, and
    gives the model's logits and values to the bit, before and after the model's weights change."""
    config = parse_toml(
        f"[env]\nteam_size = 2\nobs_team_size = 3\n[model]\n{MODELS[name]}",
        NitroConfig,
        file=tmp_path / "s.toml",
        base_dir=tmp_path,
    )
    from nitro.model import observation_layout

    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90).eval().to(device)
    copy = fp16_inference_copy(model)
    assert any(p.dtype == torch.float16 for p in copy.parameters()), "something is held in FP16"
    obs = torch.randn(64, model.obs_size, generator=torch.Generator().manual_seed(1)).to(device)

    def same() -> None:
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            logits, value = model(obs)
            copy_logits, copy_value = copy(obs)
            assert torch.equal(logits, copy_logits) and torch.equal(value, copy_value)
            assert torch.equal(model.logits(obs), copy.logits(obs))

    same()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    copy_inference_weights(copy, model)
    same()


class _OldReLU(torch.nn.Module):
    """The ReLU in an earlier version: an FP16 input without `to_half` was made FP32 first."""

    def __init__(self, to_half: bool) -> None:
        super().__init__()
        self.to_half = to_half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.to_half and x.dtype == torch.float32 and torch.is_autocast_enabled(x.device.type):
            x = x.half()
        elif not self.to_half and x.dtype == torch.float16:
            x = x.float()
        return torch.relu_(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["plain, FP32 norm", "attention"])
@pytest.mark.parametrize("shared", [True, False])
def test_the_trunk_kept_in_fp16_gives_the_same_bits_and_gradients(
    tmp_path: Path, name: str, shared: bool, device: str
) -> None:
    """The shared trunk's output stays in FP16 into both heads and each head's last ReLU runs before
    its FP32 cast; outputs and every gradient are the bits of the FP32 hand-over before."""
    import copy as copying

    from nitro.model import _ReLU, observation_layout
    from nitro.ppo import masked_log_probs

    model_text = MODELS[name]
    if not shared:
        model_text = model_text.replace("shared = [64]\n", "")
    config = parse_toml(
        f"[env]\nteam_size = 2\nobs_team_size = 3\n[model]\n{model_text}",
        NitroConfig,
        file=tmp_path / "s.toml",
        base_dir=tmp_path,
    )
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90).to(device)
    old = copying.deepcopy(model)
    for module in list(old.modules()):
        for key, child in list(module.named_children()):
            if isinstance(child, _ReLU):
                setattr(module, key, _OldReLU(child.to_half))
    generator = torch.Generator().manual_seed(2)
    obs = torch.randn(256, model.obs_size, generator=generator).to(device)
    mask = (torch.rand(256, 90, generator=generator) > 0.3).to(device)
    mask[:, 0] = True
    targets = torch.randn(256, generator=generator).to(device)

    def run(net: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        net.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16):
            logits, value = net(obs)
        log_probs = masked_log_probs(logits, mask)
        loss = -log_probs[:, 0].mean() + (value - targets).pow(2).mean()
        (loss * 1024).backward()
        grads = [p.grad.clone() for p in net.parameters() if p.grad is not None]
        return logits.detach(), value.detach(), grads

    new_logits, new_value, new_grads = run(model)
    old_logits, old_value, old_grads = run(old)
    assert torch.equal(new_logits, old_logits) and torch.equal(new_value, old_value)
    assert len(new_grads) == len(old_grads)
    assert all(torch.equal(a, b) for a, b in zip(new_grads, old_grads, strict=True))


def _old_step_forward(
    self: Any, query: torch.Tensor, cars: torch.Tensor, present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_AttentionStep.forward` in an earlier version: its products' operands made FP32 first."""
    b, t, w = cars.shape
    h, d = self.heads, w // self.heads
    q = self.q(self.norm_query(query)).view(b, 1, h, d).transpose(1, 2)
    normed = self.norm_cars(cars)
    k = self.k(normed).view(b, t, h, d).transpose(1, 2)
    v = self.v(normed).view(b, t, h, d).transpose(1, 2)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    weights = scores.masked_fill(~present[:, None, None, :], float("-inf")).softmax(-1)
    mixed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, 1, w)
    return self._finish(query, mixed), weights[:, :, 0]


def _old_forward_kv(
    self: Any,
    query: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    null_kv: tuple[torch.Tensor, torch.Tensor],
    present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_AttentionStep.forward_kv` in an earlier version."""
    b, t, w = k.shape
    h, d = self.heads, w // self.heads
    q = self.q(self.norm_query(query)).view(b, 1, h, d).transpose(1, 2).float()
    k = k.view(b, t, h, d).transpose(1, 2)
    v = v.view(b, t, h, d).transpose(1, 2)
    k_null, v_null = (x.view(1, h, 1, d) for x in null_kv)
    scores = (q @ k.float().transpose(-1, -2)) / math.sqrt(d)
    scores = scores.masked_fill(~present[:, None, None, :], float("-inf"))
    null = (q * k_null.float()).sum(-1, keepdim=True) / math.sqrt(d)
    weights = torch.cat([null, scores], dim=-1).softmax(-1)
    mixed = weights[..., :1].to(v.dtype) * v_null.to(v.dtype) + weights[..., 1:].to(v.dtype) @ v
    return self._finish(query, mixed.transpose(1, 2).reshape(b, 1, w)), weights[:, :, 0]


def _old_talk_forward(
    self: Any, tokens: torch.Tensor, present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_TalkStep.forward` in an earlier version."""
    b, t, w = tokens.shape
    h, d = self.heads, w // self.heads
    normed = self.norm(tokens)
    q, k, v = (layer(normed).view(b, t, h, d).transpose(1, 2) for layer in (self.q, self.k, self.v))
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    weights = scores.masked_fill(~present[:, None, None, :], float("-inf")).softmax(-1)
    mixed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, t, w)
    tokens = tokens + self.out(mixed)
    result: torch.Tensor = tokens + self.ff(self.norm_ff(tokens))
    return result, weights


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16-autocast"])
@pytest.mark.parametrize("name", ["attention", "attention, fast, skip", "attention_communication"])
def test_attention_products_without_their_fp32_casts_give_the_same_bits(
    tmp_path: Path, name: str, fp16: bool, device: str
) -> None:
    """Autocast casts a matrix product's operands to FP16 whatever they were, so the `.float()` the
    attention steps put on them only cost two casts each; without it outputs and every gradient are the
    same bits, in FP32 and under FP16 autocast (the CPU's here, CUDA's on a card)."""
    from nitro.model import _AttentionStep, _TalkStep, observation_layout

    config = parse_toml(
        f"[env]\nteam_size = 2\nobs_team_size = 3\n[model]\n{MODELS[name]}",
        NitroConfig,
        file=tmp_path / "s.toml",
        base_dir=tmp_path,
    )
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90).to(device)
    with torch.no_grad():  # norms off their identity start and the null car off zero
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)
    obs = torch.randn(64, model.obs_size, generator=torch.Generator().manual_seed(1))
    cars = obs[:, own:].view(64, teammates + opponents, block)
    cars[..., -1] = (
        torch.rand(64, teammates + opponents, generator=torch.Generator().manual_seed(2)) > 0.4
    ).float()
    cars[cars[..., -1] == 0] = 0.0
    obs = obs.to(device)

    def run() -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16, enabled=fp16):
            logits, value = model(obs)
        (logits.float().square().mean() + value.float().square().mean()).backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return [logits.detach(), value.detach(), *grads]

    new = run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_AttentionStep, "forward", _old_step_forward)
        patch.setattr(_AttentionStep, "forward_kv", _old_forward_kv)
        patch.setattr(_TalkStep, "forward", _old_talk_forward)
        old = run()
    assert len(new) == len(old) == 2 + len(list(model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(new, old, strict=True))
