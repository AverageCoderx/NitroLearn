"""start_from.parts = "matching": a new bot takes another's policy and shared trunk whole and
every value tensor that fits, so it decides exactly as that bot with a lighter value side; 'all' and 'policy'
take exactly what they took before."""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
import torch

from nitro.config.loader import parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig, StartFromConfig
from nitro.dashboard.app import page_source
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder, fixed_settings
from nitro.training import TrainingRun
from nitro.transfer import Source, TransferError, apply, matching, problems

ROOT = Path(__file__).parents[1]
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
CPU = torch.device("cpu")
OBS, ACTIONS = 6, 5
# V2's shape in small: a shared trunk, a critic trunk and four heads of two layers; V3 keeps two heads of one.
SOURCE = ModelConfig(
    shared=(32,),
    policy=(16,),
    value=(24,),
    value_heads=("main", "goal", "vdag1", "vdag2"),
    value_head_layers=(32, 32),
)
LIGHT = dataclasses.replace(SOURCE, value_heads=("main", "goal"), value_head_layers=(16,))
MATCHING = 'run = "v2"\nparts = "matching"'


def settings(model: ModelConfig, start: str = "", env: str = "") -> NitroConfig:
    text = f"[env]\n{env}\n[start_from]\n{start}\n"
    parsed = parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())
    return dataclasses.replace(parsed, model=model)


def learner(config: NitroConfig, seed: int, actions: int = ACTIONS) -> Learner:
    torch.manual_seed(seed)
    return Learner(ActorCritic(OBS, actions, config.model), config.ppo, gamma=0.9, device=CPU, fp16=False)


def source_of(config: NitroConfig, actions: int = ACTIONS) -> tuple[Source, Learner]:
    trained = learner(config, 1, actions)
    trained.reward_scaler.moments.count.fill_(1.0)
    trained.reward_scaler.moments.m2.fill_(49.0)
    trained.value_norm.load_state_dict(
        {"mean": torch.tensor(3.0), "mean_sq": torch.tensor(13.0), "initialised": torch.tensor(True)}
    )
    state = {"learner": trained.state_dict(), "agent_steps": 5, "settings": fixed_settings(config)}
    return Source(state, "v2 at 5 agent steps", "v2"), trained


def scaling_taken(fresh: Learner) -> bool:
    norm = fresh.value_norm.state_dict()
    return (
        float(norm["mean"]) == 3.0
        and float(norm["mean_sq"]) == 13.0
        and float(fresh.reward_scaler.moments.std) == 7.0
    )


# --- what 'matching' takes ------------------------------------------------------------------------------


def test_matching_takes_the_policy_and_both_trunks_and_starts_other_heads_fresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(a): heads of other layers start exactly as a fresh build with the same seed; the policy, the
    shared trunk and the critic trunk are the source's; the value and reward scaling are taken."""
    source, trained = source_of(settings(SOURCE))
    light = settings(LIGHT, MATCHING)
    built = learner(light, 2).model.state_dict()
    fresh = learner(light, 2)
    with caplog.at_level(logging.INFO, logger="nitro"):
        apply(source, fresh, light)
    state, theirs = fresh.model.state_dict(), trained.model.state_dict()
    for key, value in state.items():
        took = key.startswith(("shared.", "policy.", "value."))
        assert torch.equal(value, theirs[key] if took else built[key]), key
    assert any(k.startswith("value_heads.goal.") for k in state)
    assert scaling_taken(fresh)
    said = caplog.text
    assert "the policy and shared trunk whole" in said
    assert "starting fresh: value_heads.main (all layers), value_heads.goal (all layers)" in said
    assert "left out: value_heads.vdag1 (all layers), value_heads.vdag2 (all layers)" in said


def test_matching_takes_every_head_whose_layers_are_the_same() -> None:
    source, trained = source_of(settings(SOURCE))
    fewer = settings(dataclasses.replace(SOURCE, value_heads=("main", "goal")), MATCHING)
    fresh = learner(fewer, 2)
    apply(source, fresh, fewer)
    theirs = trained.model.state_dict()
    for key, value in fresh.model.state_dict().items():
        assert torch.equal(value, theirs[key]), key
    taken = matching(fresh.model.state_dict(), theirs, "v2")
    assert taken.fresh == () and taken.left_out == (
        "value_heads.vdag1 (all layers)",
        "value_heads.vdag2 (all layers)",
    )


def test_matching_names_the_layers_it_starts_fresh_within_a_network() -> None:
    """Another critic trunk with the same first layer: that layer is taken, the rest named."""
    source, trained = source_of(settings(SOURCE))
    deeper = settings(dataclasses.replace(SOURCE, value=(24, 24)), MATCHING)
    fresh = learner(deeper, 2)
    apply(source, fresh, deeper)
    state = fresh.model.state_dict()
    assert torch.equal(state["value.0.weight"], trained.model.state_dict()["value.0.weight"])
    taken = matching(learner(deeper, 2).model.state_dict(), trained.model.state_dict(), "v2")
    assert taken.fresh == ("value (3.weight, 3.bias, 4.weight, 4.bias)",)  # the second layer and its norm


def test_matching_refuses_another_policy_observation_or_action_count() -> None:
    """(c): the decision-making is copied whole or not at all; the value side may differ."""
    source, _ = source_of(settings(SOURCE))
    for model, said in (
        (dataclasses.replace(LIGHT, policy=(32,)), r"model\.policy is \[16\] there and \[32\] here"),
        (dataclasses.replace(LIGHT, shared=(48,)), r"model\.shared is \[32\] there and \[48\] here"),
    ):
        config = settings(model, MATCHING)
        with pytest.raises(TransferError, match=said):
            apply(source, learner(config, 2), config)
    observed = settings(LIGHT, MATCHING, 'observation = "native"')
    assert any("env.observation" in p for p in problems(source, observed))
    config = settings(LIGHT, MATCHING)
    with pytest.raises(
        TransferError, match=r"policy\.\d\.weight: \(5, 16\) there, \(6, 16\) here.*never takes"
    ):
        apply(source, learner(config, 2, ACTIONS + 1), config)  # another action count
    no_popart = dataclasses.replace(config, ppo=PPOConfig(popart=False))
    assert any("ppo.popart" in p for p in problems(source, no_popart))
    other_value = settings(dataclasses.replace(LIGHT, value=(40,)), MATCHING)
    assert problems(source, other_value) == [], "the value side may differ"


# --- 'all' and 'policy' exactly as before ---------------------------------------------------------------


PLAIN = ModelConfig(policy=(16,), value=(16,))


@pytest.mark.parametrize("model", [PLAIN, SOURCE], ids=["one-value-network", "value-heads"])
def test_all_and_policy_take_exactly_what_they_took_before(model: ModelConfig) -> None:
    """(d): 'all' gives every tensor and the scaling of the source; 'policy' the policy and shared trunk
    with the value side as built and no scaling: bit for bit."""
    source, trained = source_of(settings(model))
    theirs = trained.model.state_dict()
    every = settings(model, 'run = "v2"')
    fresh = learner(every, 2)
    apply(source, fresh, every)
    assert fresh.model.state_dict().keys() == theirs.keys()
    assert all(torch.equal(v, theirs[k]) for k, v in fresh.model.state_dict().items())
    assert scaling_taken(fresh)

    policy = settings(model, 'run = "v2"\nparts = "policy"')
    built = learner(policy, 2).model.state_dict()
    fresh = learner(policy, 2)
    apply(source, fresh, policy)
    for key, value in fresh.model.state_dict().items():
        assert torch.equal(value, theirs[key] if key.startswith(("shared.", "policy.")) else built[key]), key
    assert float(fresh.value_norm.state_dict()["mean"]) == 0.0


def test_all_and_policy_refuse_what_they_refused_before() -> None:
    source, _ = source_of(settings(SOURCE))
    lighter = dataclasses.replace(LIGHT, value=(40,))
    said = {
        parts: problems(source, settings(lighter, f'run = "v2"\nparts = "{parts}"'))
        for parts in ("all", "policy")
    }
    assert said["all"] == [
        "model.value is [24] there and [40] here",
        'model.value_heads is ["main", "goal", "vdag1", "vdag2"] there and ["main", "goal"] here',
        "model.value_head_layers is [32, 32] there and [16] here",
    ]
    assert said["policy"] == ["model.value is [24] there and [40] here"]


# --- real runs and the check tool ---------------------------------------------------------------------


RUN = """
[run]
seed = 7
[env]
num_arenas = 4
threads = 2
[terminals]
no_touch_seconds = 2
[model]
shared = [24]
policy = [32, 32]
value = [32]
residual = true
layer_norm = true
{heads}
[ppo]
steps_per_iteration = 320
minibatches = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""
V2_HEADS = 'value_heads = ["main", "goal", "vdag1", "vdag2"]\nvalue_head_layers = [32, 32]'
V3_HEADS = 'value_heads = ["main", "goal"]\nvalue_head_layers = [16]'
V3_START = '[start_from]\nrun = "v2"\nparts = "matching"\n'


def _config(text: str, file: Path) -> NitroConfig:
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=file.parent)
    return dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))


def _v2(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """V2 in small: one iteration, saved."""
    file = tmp_path / "v2.toml"
    first = TrainingRun(
        _config(RUN.format(heads=V2_HEADS), file), RunFolder.create(tmp_path / "runs", "v2", file)
    )
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    return file, {k: v.clone() for k, v in first.model.state_dict().items()}


@NEEDS_MESHES
def test_a_matching_run_trains_saves_and_resumes(tmp_path: Path) -> None:
    """(e): V3 in small starts from V2 in small with V2's policy and trunks and its own fresh heads, learns,
    saves and resumes exactly where it stopped."""
    _, trained = _v2(tmp_path)
    file = tmp_path / "v3.toml"
    settings = _config(RUN.format(heads=V3_HEADS) + V3_START, file)
    v3 = TrainingRun(settings, RunFolder.create(tmp_path / "runs", "v3", file))
    assert v3.started_from is not None and v3.started_from.startswith("v2 at ")
    before = {k: v.clone() for k, v in v3.model.state_dict().items()}
    taken = [k for k in before if k.startswith(("shared.", "policy.", "value."))]
    assert taken and all(torch.equal(before[k], trained[k]) for k in taken)
    v3.iteration()
    after = {k: v.clone() for k, v in v3.model.state_dict().items()}
    assert any(not torch.equal(after[k], before[k]) for k in after if k.startswith("policy.")), "it learns"
    assert any(not torch.equal(after[k], before[k]) for k in after if k.startswith("value_heads.main."))
    path = v3.checkpoint(continue_after=False)
    v3.metrics.close()
    saved = torch.load(path / "state.pt", weights_only=True)
    resumed = TrainingRun(settings, RunFolder(tmp_path / "runs" / "v3"), saved)
    assert all(torch.equal(v, after[k]) for k, v in resumed.model.state_dict().items()), "where it stopped"
    row = resumed.iteration()
    resumed.metrics.close()
    assert float(row["entropy"]) > 0


def test_the_dashboard_says_what_matching_takes_and_new_rewards_switch_it_to_the_policy(
    tmp_path: Path,
) -> None:
    """Like 'all', 'matching' takes the value scaling, so new rewards on a bot not started yet switch it to
    the policy only, as the dashboard does for 'all'."""
    from nitro.dashboard.api import Api
    from test_dashboard import TINY, make_bot
    from test_dashboard_transfer import change_a_reward

    runs = tmp_path / "runs"
    make_bot(runs, "first", TINY, saves=1)
    api = Api(ROOT, runs)
    api.copy_bot("first", "light", True)
    api.stage("light", "start_from.parts", "matching")
    api.apply("light")
    line = api.bot("light")["start"]["line"]
    assert "first's policy whole and every part of its value network that fits, and its scaling" in line
    change_a_reward(api, "light")
    assert "will take only first's policy" in api.apply("light")["switched"]
    assert RunFolder(runs / "light").load_settings().start_from.parts == "policy"


def test_new_rewards_keep_matching_when_the_policy_alone_would_not_fit(tmp_path: Path) -> None:
    """'policy' refuses another model.value, so a 'matching' bot with a lighter critic trunk keeps
    'matching' on new rewards and can still start."""
    from nitro import transfer
    from nitro.dashboard.api import Api
    from test_dashboard import TINY, make_bot
    from test_dashboard_transfer import change_a_reward

    runs = tmp_path / "runs"
    make_bot(runs, "first", TINY, saves=1)
    save = runs / "first" / "checkpoints" / f"{1000:012d}" / "state.pt"
    first = RunFolder(runs / "first").load_settings()
    trained = Learner(ActorCritic(OBS, ACTIONS, first.model), first.ppo, gamma=0.9, device=CPU, fp16=False)
    saved = {"learner": trained.state_dict(), "agent_steps": 1000, "settings": fixed_settings(first)}
    torch.save(saved, save)
    api = Api(ROOT, runs)
    api.copy_bot("first", "light", True)
    api.stage("light", "start_from.parts", "matching")
    api.stage("light", "model.value", [8])
    api.apply("light")
    change_a_reward(api, "light")
    assert "switched" not in api.apply("light")
    light = RunFolder(runs / "light").load_settings()
    assert light.start_from.parts == "matching"
    source = transfer.find_source(light.start_from, runs)
    assert source is not None and problems(source, light) == []
    policy = dataclasses.replace(light, start_from=dataclasses.replace(light.start_from, parts="policy"))
    assert problems(source, policy) == ["model.value is [16] there and [8] here"]


def test_the_dashboard_offers_every_choice_of_parts() -> None:
    """The Transfer learning card's own dropdown (not the settings list) offers each start_from.parts choice
    and shows the one written as chosen."""
    page = page_source()
    control = page[page.index('if (item.path === "start_from.parts")') :]
    control = control[: control.index("\n  }")]
    offered = re.findall(r'opt\("(\w+)", "[^"]*", v === "(\w+)"\)', control)
    choices = get_args(get_type_hints(StartFromConfig)["parts"])
    assert [value for value, _ in offered] == list(choices)
    assert all(value == chosen for value, chosen in offered)
