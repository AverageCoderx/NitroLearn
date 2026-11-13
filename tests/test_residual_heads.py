"""`model.residual` and the value side with several heads: residual blocks where a layer
keeps the width, a critic trunk with named heads that are saved and restored but learn nothing without the
signals that give them a loss (test_gco_signals.py), the parameter count `nitro check-settings` prints,
and the settings refused. The bot parity of residual networks is in test_export.py."""

from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path

import pytest
import torch
from torch import nn

from nitro.cli import main
from nitro.config import ConfigError, load_config
from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic, ResidualBlock, fp16_inference_copy, parameter_counts
from nitro.runs import FIXED_SETTINGS
from test_optimizer_options import rollout_for

ROOT = Path(__file__).resolve().parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
OBS, ACTIONS = 4, 3
HEADS = ModelConfig(
    shared=(8, 8),
    policy=(16, 16),
    value=(12, 12),
    value_head_layers=(12,),
    value_heads=("main", "goal", "vdag1", "vdag2"),
    residual=True,
    activation="leaky_relu",
)


def test_residual_blocks_add_their_layer_to_their_input_where_the_width_stays() -> None:
    torch.manual_seed(0)
    model = ActorCritic(6, ACTIONS, dataclasses.replace(ModelConfig(), residual=True, policy=(6, 10, 10)))
    blocks = [m for m in model.policy if isinstance(m, ResidualBlock)]
    assert len(blocks) == 2, "6 -> 6 and 10 -> 10 are blocks; 6 -> 10 is a plain layer"
    assert isinstance(model.policy[0], ResidualBlock) and isinstance(model.policy[-1], nn.Linear)
    x = torch.randn(5, 6)
    block = blocks[0]
    assert torch.equal(block(x), x + block.branch(x))
    branch = list(block.branch)
    assert [type(m).__name__ for m in branch] == ["Linear", "LayerNorm", "LeakyReLU"]
    assert "policy.0.branch.0.weight" in model.state_dict()


def test_with_the_settings_off_the_networks_have_no_blocks_or_heads() -> None:
    """No block, no head, no keys for them; the settings that shape them are fixed for a run."""
    model = ActorCritic(OBS, ACTIONS, ModelConfig(shared=(8, 8), policy=(8, 8), value=(8,)))
    assert not any(isinstance(m, ResidualBlock) for m in model.modules())
    assert len(model.value_heads) == 0 and not any(k.startswith("value_heads") for k in model.state_dict())
    assert model.value_output is model.value[-1]
    assert not ModelConfig().value_heads_apart
    assert {"model.residual", "model.value_heads", "model.value_head_layers"} <= set(FIXED_SETTINGS)


def test_fp16_keeps_the_residual_stream_in_fp32() -> None:
    torch.manual_seed(1)
    model = ActorCritic(OBS, ACTIONS, HEADS)
    x = torch.randn(7, OBS)
    with torch.autocast("cpu", dtype=torch.float16):
        trunk = model.shared(x)
        logits, heads = model.forward_heads(x)
    assert trunk.dtype == torch.float32 and logits.dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in heads.values())
    copy_ = fp16_inference_copy(model)
    assert all(layer.weight.dtype == torch.float32 for layer in copy_.output_layers())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        assert torch.equal(copy_(x)[1], model(x)[1].detach())


def test_value_heads_sit_on_the_critic_trunk_and_main_is_the_value() -> None:
    torch.manual_seed(2)
    model = ActorCritic(OBS, ACTIONS, HEADS)
    assert list(model.value_heads) == ["main", "goal", "vdag1", "vdag2"]
    assert not isinstance(model.value[-1], nn.Linear), "the critic trunk has no output layer"
    x = torch.randn(5, OBS)
    logits, heads = model.forward_heads(x)
    value = model(x)[1]
    assert torch.equal(heads["main"], value) and torch.equal(logits, model(x)[0])
    assert all(v.shape == (5,) for v in heads.values())
    assert len({float(v.detach().sum()) for v in heads.values()}) == 4, "each head its own weights"
    assert torch.equal(model.head_values(x)["goal"], heads["goal"])
    # The value group holds the trunk and every head; the policy's frozen copy has none of them.
    value_group = {id(p) for p in model.parameter_groups()["value"]}
    assert {id(p) for p in model.value_heads.parameters()} <= value_group
    assert not any(k.startswith("value") for k in model.policy_state_dict())
    # Without heads the same call gives the one value.
    plain = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(8,), value=(8,)))
    assert list(plain.head_values(x)) == ["main"]
    assert torch.equal(plain.head_values(x)["main"], plain(x)[1])


def test_extra_heads_learn_nothing_without_their_signals_and_are_saved_and_restored() -> None:
    torch.manual_seed(3)
    model = ActorCritic(OBS, ACTIONS, HEADS)
    learner = Learner(
        model, PPOConfig(epochs=1, minibatches=2), gamma=0.9, device=torch.device("cpu"), fp16=False
    )
    optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}, "the optimizer holds every head"
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = learner.learn(rollout_for(model))
    after = model.state_dict()
    moved = {key for key in after if not torch.equal(before[key], after[key])}
    extra = {key for key in after if key.startswith(("value_heads.goal", "value_heads.vdag"))}
    assert extra and not moved & extra, "no loss reaches the extra heads"
    assert moved == set(after) - extra, "everything else learned"
    assert stats.losses_finite and set(stats.update_norms) == {"policy", "value", "shared"}
    # A save restores every head.
    with torch.no_grad():
        for p in model.value_heads.parameters():
            p.add_(0.5)
    state = copy.deepcopy(learner.state_dict())
    other = Learner(
        ActorCritic(OBS, ACTIONS, HEADS), PPOConfig(epochs=1, minibatches=2), 0.9, torch.device("cpu"), False
    )
    other.load_state_dict(state)
    restored = other.model.state_dict()
    assert all(torch.equal(restored[k], v) for k, v in model.state_dict().items())
    x = torch.randn(3, OBS)
    assert all(
        torch.equal(a, b)
        for a, b in zip(other.head_values(x).values(), learner.head_values(x).values(), strict=True)
    )


def test_the_update_norms_are_each_networks_change() -> None:
    """The learning steps' change (PopArt's rescaling of the value output comes before, so it is off here)."""
    torch.manual_seed(4)
    model = ActorCritic(OBS, ACTIONS, HEADS)
    config = PPOConfig(epochs=1, minibatches=2, popart=False)
    learner = Learner(model, config, gamma=0.9, device=torch.device("cpu"), fp16=False)
    before = {name: [p.detach().clone() for p in params] for name, params in learner.groups.items()}
    stats = learner.learn(rollout_for(model))
    for name, params in learner.groups.items():
        change = torch.sqrt(
            sum(((p.detach() - b) ** 2).sum() for p, b in zip(params, before[name], strict=True))
        )
        assert stats.update_norms[name] == pytest.approx(float(change), rel=1e-5)


def test_check_settings_prints_the_gco_starters_parameter_count(capsys: pytest.CaptureFixture[str]) -> None:
    settings = GCO_STARTER
    counts = parameter_counts(load_config(settings))
    # Shared 148 -> 520 then two blocks; policy 520 -> 352, two blocks, 90 outputs; critic trunk 520 ->
    # 600, two blocks; each of four heads two blocks and one output.
    assert counts["shared"] == 148 * 520 + 520 + 2 * (520 * 520 + 520) + 3 * 2 * 520
    assert counts["playing"] == counts["shared"] + counts["policy"] == 1_088_226
    assert counts["head_goal"] == 2 * (600 * 600 + 600 + 2 * 600) + 601
    assert counts["total"] == 5_022_430
    meshes = (ROOT / "collision_meshes" / "soccar").is_dir()  # without them the check says so
    assert main(["check-settings", str(settings)]) == (0 if meshes else 1)
    out = capsys.readouterr().out
    assert "Networks: 5,022,430 parameters" in out and "The bot runs 1,088,226" in out


def test_the_gco_starters_full_size_is_the_commented_one(tmp_path: Path) -> None:
    """The four commented lines under [model] give the design's full size, as the comments say."""
    text = GCO_STARTER.read_text(encoding="utf-8")
    full = re.findall(r"^# ((?:shared|policy|value|value_head_layers) = \[[\d, ]+\])$", text, re.MULTILINE)
    assert len(full) == 4
    for line in full:
        name = line.split(" = ")[0]
        text = re.sub(rf"^{name} = \[[\d, ]+\]", line, text, count=1, flags=re.MULTILINE)
    file = tmp_path / "full.toml"
    file.write_text(text, encoding="utf-8")
    counts = parameter_counts(load_config(file))
    assert counts["total"] == 22_881_502 and counts["playing"] == 4_975_578
    assert "about 23 million numbers" in text and "About 5 million numbers" in text


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[model]\nresidual = true\nlayer_norm = false\n", "model.residual: needs model.layer_norm"),
        ('[model]\nvalue_heads = ["goal", "main"]\n', 'model.value_heads: must start with "main"'),
        ('[model]\nvalue_heads = ["main", "goal", "goal"]\n', "model.value_heads: names a head twice"),
        ('[model]\nvalue_heads = ["main", "tdag"]\n', r"model.value_heads\[1\]: must be one of"),
    ],
)
def test_settings_that_cannot_work_are_refused(tmp_path: Path, text: str, problem: str) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert re.search(problem, str(caught.value)), str(caught.value)
