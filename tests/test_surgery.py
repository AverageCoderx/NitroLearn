"""`nitro surgery`: widening a run's networks with identical outputs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from nitro import __version__
from nitro.config import load_config
from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder, all_settings, fixed_settings
from nitro.surgery import (
    SurgeryError,
    allowed,
    compare,
    copy_shares,
    model_shapes,
    widen_model,
    widen_run,
    widen_settings_text,
    widened_config,
)

OBS, ACTIONS = 23, 7


def random_model(config: ModelConfig, seed: int = 1) -> ActorCritic:
    torch.manual_seed(seed)
    model = ActorCritic(OBS, ACTIONS, config)
    with torch.no_grad():
        for p in model.parameters():  # off the orthogonal start, so nothing is special about the weights
            p.add_(torch.randn_like(p) * 0.3)
    return model


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(shared=(12, 10), policy=(8,), value=(6, 6)),
        ModelConfig(policy=(16, 8), value=(8,), layer_norm=False),
        ModelConfig(shared=(9,), policy=(8, 8), value=(8,), activation="gelu"),
        ModelConfig(policy=(8,), value=(8,), activation="tanh", init="pytorch"),
    ],
    ids=["shared-layernorm-relu", "no-layernorm", "gelu", "tanh-pytorch-init"],
)
@pytest.mark.parametrize("factor", [2, 3])
def test_a_widened_network_computes_exactly_what_the_old_one_did(config: ModelConfig, factor: int) -> None:
    old = random_model(config)
    new = widen_model(old, factor)
    assert new.config == widened_config(config, factor)
    assert tuple(new.config.policy) == tuple(w * factor for w in config.policy)
    difference, scale = compare(old, new, OBS)
    assert difference < allowed(scale) and difference < 1e-4
    assert sum(p.numel() for p in new.parameters()) > sum(p.numel() for p in old.parameters())


def copies_apart(model: ActorCritic, factor: int) -> float:
    """How far apart the copies of a unit are: the largest difference between the incoming weights of two
    copies of one unit, over every hidden layer, relative to the layer's mean weight."""
    apart = 0.0
    for stack in (model.shared, model.policy, model.value):
        hidden = [m for m in stack if isinstance(m, torch.nn.Linear)]
        if stack is not model.shared:
            hidden = hidden[:-1]  # the output layer has no copies
        for layer in hidden:
            rows = layer.weight.detach().reshape(-1, factor, layer.in_features)
            spread = (rows - rows[:, :1]).abs().max() / layer.weight.detach().abs().mean()
            apart = max(apart, float(spread))
    return apart


def train_briefly(model: ActorCritic, optimizer: str, steps: int = 30) -> None:
    """A few of the learner's own optimizer steps on a made-up task."""
    learner = Learner(
        model, dataclasses.replace(PPOConfig(), optimizer=optimizer), 0.99, torch.device("cpu"), fp16=False
    )
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(OBS, ACTIONS, generator=generator)
    for _ in range(steps):
        obs = torch.randn(256, OBS, generator=generator)
        logits, values = model(obs)
        loss = (logits - torch.tanh(obs @ target)).pow(2).mean() + (values - obs[:, 0].sin()).pow(2).mean()
        learner.optimizer.zero_grad()
        loss.backward()
        learner.optimizer.step()


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(shared=(12, 10), policy=(8,), value=(6, 6)),
        ModelConfig(policy=(16, 8), value=(8,), layer_norm=False),
        ModelConfig(policy=(8, 8), value=(8,), activation="tanh"),
    ],
    ids=["shared-layernorm-relu", "no-layernorm", "tanh"],
)
@pytest.mark.parametrize("optimizer", ["adam", "muon"])
@pytest.mark.parametrize("factor", [2, 3])
def test_the_copies_of_a_unit_learn_apart(config: ModelConfig, optimizer: str, factor: int) -> None:
    """Copies that read on with equal shares get the same updates and stay the same for good (no new
    capacity); the unequal shares part them within a few steps, while the widened network still computes
    what the old one did at the moment of the surgery."""
    old = random_model(config)
    new = widen_model(old, factor)
    assert copies_apart(new, factor) == 0.0, "the copies start alike"
    difference, scale = compare(old, new, OBS)
    assert difference < allowed(scale) and difference < 1e-4
    train_briefly(new, optimizer)
    assert copies_apart(new, factor) > 1e-3, "after 30 steps the copies differ"


def test_copies_with_equal_shares_would_never_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why the shares are unequal: with equal ones (the widening as it was) the copies stay identical."""
    from nitro import surgery

    monkeypatch.setattr(surgery, "SPREAD", 0.0)
    new = widen_model(random_model(ModelConfig(shared=(12, 10), policy=(8,), value=(6, 6))), 2)
    train_briefly(new, "adam")
    assert copies_apart(new, 2) == 0.0


def test_the_copies_shares_are_unequal_and_add_up_to_the_original() -> None:
    for factor in (2, 3, 4):
        shares = copy_shares(factor)
        assert float(shares.sum()) == pytest.approx(1.0, abs=1e-12)
        assert bool((shares > 0).all()) and len(set(shares.tolist())) == factor
    assert copy_shares(2).tolist() == pytest.approx([0.25, 0.75])


def test_the_widening_factor_must_be_two_or_more() -> None:
    with pytest.raises(SurgeryError, match="2 or more"):
        widen_model(random_model(ModelConfig(policy=(8,), value=(8,))), 1)


def test_the_settings_text_gets_the_new_widths_and_keeps_everything_else() -> None:
    model = ModelConfig(shared=(), policy=(256, 128), value=(64,))
    text = (
        "[run]\nseed = 3\n\n[model]\npolicy = [256, 128]   # the policy\nvalue = [64]\nlayer_norm = true\n\n"
        "[ppo]\nepochs = 2\n"
    )
    wide = widen_settings_text(text, model, 2)
    assert "policy = [512, 256]   # the policy" in wide and "value = [128]" in wide
    assert "shared = []  # set by nitro surgery" in wide, (
        "a width the file left to its default is written out"
    )
    assert wide.startswith("[run]\nseed = 3\n") and wide.rstrip().endswith("[ppo]\nepochs = 2")
    # No [model] section at all: one is added with every width.
    plain = widen_settings_text("[run]\nseed = 3\n", ModelConfig(), 2)
    assert "[model]" in plain and "policy = [512, 512, 512]" in plain and "value = [512, 512, 512]" in plain


def test_the_check_allows_rounding_in_proportion_to_the_outputs() -> None:
    assert allowed(0.3) == pytest.approx(1e-4), "small outputs: an absolute 1e-4"
    assert allowed(30.0) == pytest.approx(3e-3), "a trained policy's large logits: 1e-4 of them"
    old = random_model(ModelConfig(policy=(16, 16), value=(16,)))
    with torch.no_grad():
        old.policy[-1].weight.mul_(40.0)  # logits in the tens, as a trained policy's can be
    difference, scale = compare(old, widen_model(old, 3), OBS)
    assert scale > 10 and difference < allowed(scale)


def test_unusual_but_valid_settings_layouts_are_rewritten() -> None:
    model = ModelConfig(policy=(256, 256), value=(256,))
    spread = (
        '[ model ]\npolicy = [\n  256,  # first\n  256,\n]   # the policy\n"value" = [256]\n'
        "[ppo]\nepochs = 2\n"
    )
    wide = widen_settings_text(spread, model, 2)
    assert "policy = [512, 512]   # the policy" in wide and '"value" = [512]' in wide
    assert "256" not in wide and wide.count("policy =") == 1 and wide.rstrip().endswith("epochs = 2")


def test_a_surgery_that_cannot_rewrite_the_settings_leaves_nothing_behind(tmp_path: Path) -> None:
    folder, _ = saved_run(tmp_path, "model.policy = [8, 8]\nmodel.value = [8]\n[run]\nseed = 4\n")
    with pytest.raises(SurgeryError, match="could not be written into the settings file") as info:
        widen_run(tmp_path / "runs", "small", 2)
    assert "Nothing was written" in str(info.value)
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["small"], "no new or half-made folder"
    assert folder.load_settings().model.policy == (8, 8)


def test_a_run_without_hidden_layers_has_nothing_to_widen(tmp_path: Path) -> None:
    saved_run(tmp_path, "[model]\npolicy = []\nvalue = []\n")
    with pytest.raises(SurgeryError, match="no hidden layers to widen"):
        widen_run(tmp_path / "runs", "small", 2)


def saved_run(tmp_path: Path, text: str) -> tuple[RunFolder, ActorCritic]:
    """A run folder called 'small' with one save at 1,000,000 agent steps, made from settings `text`."""
    settings = tmp_path / "small.toml"
    settings.write_text(text, encoding="utf-8")
    config = load_config(settings)
    folder = RunFolder.create(tmp_path / "runs", "small", settings)
    model = random_model(config.model)
    learner = Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False)
    state = {
        "learner": learner.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 3,
        "agent_steps": 1_000_000,
        "iterations": 20,
        "settings": fixed_settings(config),
        "all_settings": all_settings(config),
        "device": "cpu",
        "fp16": False,
    }
    folder.save_checkpoint(state, 1_000_000, keep=5, milestone_steps=0)
    return folder, model


def test_model_shapes_are_read_off_the_saved_weights() -> None:
    model = random_model(ModelConfig(shared=(5,), policy=(8, 8), value=(8,)))
    assert model_shapes(model.state_dict()) == (OBS, ACTIONS)
    assert model_shapes(random_model(ModelConfig(policy=(8,), value=(8,))).state_dict()) == (OBS, ACTIONS)


def test_widening_a_run_makes_a_new_run_that_resumes_from_the_same_point(tmp_path: Path) -> None:
    settings = tmp_path / "small.toml"
    settings.write_text(
        "[run]\nseed = 4\n[model]\nshared = [6]\npolicy = [8, 8]\nvalue = [8]\n[ppo]\nminibatches = 2\n",
        encoding="utf-8",
    )
    config = load_config(settings)
    folder = RunFolder.create(tmp_path / "runs", "small", settings)
    model = random_model(config.model)
    learner = Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False)
    learner.reward_scaler.moments.count.fill_(1.0)
    learner.reward_scaler.moments.m2.fill_(49.0)
    state = {
        "learner": learner.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 3,
        "agent_steps": 1_000_000,
        "iterations": 20,
        "settings": fixed_settings(config),
        "all_settings": all_settings(config),
        "device": "cpu",
        "fp16": False,
    }
    folder.save_checkpoint(state, 1_000_000, keep=5, milestone_steps=0)
    folder.metrics.write_text("agent_steps,iteration\n50000,1\n1000000,20\n", encoding="utf-8")
    pool = folder.root / "pool"
    pool.mkdir()
    torch.save(model.policy_state_dict(), pool / "000000500000.pt")
    (pool / "games.csv").write_text("a,b\n", encoding="utf-8")
    (pool / "000000600000.pt").write_bytes(b"")  # cut short by a power cut: copied as it is

    report = widen_run(tmp_path / "runs", "small", 2)
    assert report.target == "small-x2" and report.largest_difference < 1e-4 and report.pool_versions == 1
    new_folder = RunFolder(tmp_path / "runs" / "small-x2")
    new_state = new_folder.load_latest()
    assert new_state is not None and new_state["agent_steps"] == 1_000_000 and new_state["checkpoint"] == 3
    assert new_state["nitro_version"] == __version__, "the release that widened it"
    new_config = new_folder.load_settings(new_state)  # the widened save fits the new settings file
    assert (new_config.model.shared, new_config.model.policy) == ((12,), (16, 16))
    assert new_config.run.seed == 4 and new_config.ppo.minibatches == 2
    # The learner continues: the wider model loads, Adam is fresh, the other state carried over.
    wide = ActorCritic(OBS, ACTIONS, new_config.model)
    resumed = Learner(wide, new_config.ppo, 0.99, torch.device("cpu"), fp16=False)
    resumed.load_state_dict(new_state["learner"])
    assert float(resumed.reward_scaler.moments.std) == 7.0 and not resumed.optimizer.state_dict()["state"]
    assert compare(model, wide, OBS)[0] < 1e-4
    assert new_folder.metrics.read_text(encoding="utf-8") == folder.metrics.read_text(encoding="utf-8")
    widened = torch.load(new_folder.root / "pool" / "000000500000.pt", weights_only=True)
    carrier = ActorCritic(OBS, ACTIONS, new_config.model)
    carrier.load_policy_state_dict(widened)
    obs = torch.randn(5, OBS)
    assert torch.allclose(carrier.logits(obs), model.logits(obs), atol=1e-4), (
        "the past version plays the same"
    )
    assert (new_folder.root / "pool" / "games.csv").is_file()
    assert (new_folder.root / "pool" / "000000600000.pt").read_bytes() == b""
    assert (
        json.loads(new_folder.info.read_text())["settings_dir"]
        == json.loads(folder.info.read_text())["settings_dir"]
    )
    # The old run is untouched; a second surgery under the same name is refused.
    assert folder.load_settings().model.policy == (8, 8)
    with pytest.raises(SurgeryError, match="already exists"):
        widen_run(tmp_path / "runs", "small", 2)
    with pytest.raises(SurgeryError, match="no save at"):
        widen_run(tmp_path / "runs", "small", 2, target="other", at=7)
    with pytest.raises(SurgeryError, match="no run called"):
        widen_run(tmp_path / "runs", "missing", 2)
