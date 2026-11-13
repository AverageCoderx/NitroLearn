"""A shared trunk with an attention encoder (`model.shared` with `model.encoder`): one
encoder and the shared layers serve both the policy and the value network (a shared head); the
order of the cars still changes nothing, the encoder's own learning rate and the attention view find it, a
start from the policy takes the trunk, and the RLBot bot runs it exactly as PyTorch does."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config.schema import PPOConfig
from nitro.export import write_policy
from nitro.learner import Learner
from nitro.model import EntityEncoder, TalkingEncoder, attention_encoder, build_networks, observation_layout
from test_attention import observations, settings

SHARED = "shared = [48, 40]\npolicy = [32]\nvalue = [24]\nattention_width = 32\nattention_heads = 4\n"


def shared_networks(tmp_path: Path, extra: str = "", encoder: str = "attention") -> tuple:
    config = settings(tmp_path, model=SHARED + f'encoder = "{encoder}"\n' + extra)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90).eval()
    return config, model, layout


@pytest.mark.parametrize("encoder", ["attention", "attention_communication"])
def test_one_encoder_in_the_trunk_serves_both_networks(tmp_path: Path, encoder: str) -> None:
    _, model, layout = shared_networks(tmp_path, encoder=encoder)
    first = model.shared[0]
    assert isinstance(first, EntityEncoder | TalkingEncoder) and attention_encoder(model) is first
    assert not any(isinstance(m, EntityEncoder | TalkingEncoder) for m in [*model.policy, *model.value])
    assert model.policy[0].in_features == 40 and model.value[0].in_features == 40
    obs = observations(layout, 5, (2, 3))
    own, block, teammates, opponents = layout
    swapped = obs.clone()
    cars = swapped[:, own:].view(5, teammates + opponents, block)
    cars[:, [teammates, teammates + 1]] = cars[:, [teammates + 1, teammates]]  # two opponents change places
    with torch.no_grad():
        logits, value = model(obs)
        torch.testing.assert_close(model.logits(swapped), logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.normalised_value(obs), value)
    assert set(model.policy_state_dict()) >= {k for k in model.state_dict() if k.startswith("shared.0.")}


def test_the_encoder_in_the_trunk_learns_at_its_own_rate(tmp_path: Path) -> None:
    _, model, _ = shared_networks(tmp_path)
    learner = Learner(model, PPOConfig(encoder_lr_scale=0.5), 0.99, torch.device("cpu"), fp16=False)
    rates = {(name, factor) for name, factor, _ in learner.rate_groups}
    assert rates == {("shared", 0.5), ("shared", 1.0), ("policy", 1.0), ("value", 1.0)}
    encoder = {id(p) for p in model.shared[0].parameters()}
    halved = [g for g in learner.optimizer.param_groups if g["lr"] == PPOConfig().policy_lr * 0.5]
    assert len(halved) == 1 and {id(p) for p in halved[0]["params"]} == encoder


@pytest.mark.parametrize(
    ("encoder", "extra", "file_format"),
    [
        ("attention", "", 2),
        ("attention", "attention_skip = true\n", 3),
        ("attention_communication", "", 4),
        ("attention", "lean_attention = true\nattention_skip = true\n", 3),
        ("attention_communication", "lean_attention = true\n", 4),
    ],
)
def test_the_rlbot_bot_runs_a_shared_attention_trunk_exactly_as_pytorch_does(
    tmp_path: Path, encoder: str, extra: str, file_format: int
) -> None:
    config, model, layout = shared_networks(tmp_path, extra, encoder)
    with torch.no_grad():  # every weight off its starting value, so their order in the file is tested
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "shared-attention", 0)
    bot = _core.Policy(path)
    assert bot.header()["format"] == file_format
    for filled in ((2, 3), (0, 1), (0, 0)):
        obs = observations(layout, 6, filled)
        with torch.no_grad():
            expected = model.logits(obs)
        flat = bot.logits_batch(obs.flatten().tolist(), 6)
        torch.testing.assert_close(torch.tensor(flat).view(6, 90), expected, rtol=1e-5, atol=1e-5)
