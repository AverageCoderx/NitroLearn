"""Saves and exported bots from released versions keep working.

Each folder under tests/fixtures/saves/ is a tiny real run made by a released version
(`tools/make_fixtures.py`): its settings, its last save, the bot exported from it and what
its networks answered then on fixed observations. Every later version must load the save with the same
settings, rebuild networks that answer exactly the same, read the old bot file, export the save again,
and (with the collision meshes) resume and train it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import _core
from nitro.eval import Side
from nitro.eval import _model as eval_model
from nitro.export import export_run, model_from_state
from nitro.learner import Learner
from nitro.ppo import discount_from_half_life
from nitro.runs import RunFolder
from nitro.transfer import Source, apply, problems

FIXTURES = Path(__file__).parent / "fixtures" / "saves"
SAVES = sorted(p for p in FIXTURES.iterdir() if (p / "expected.json").is_file()) if FIXTURES.is_dir() else []
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
CPU = torch.device("cpu")


def copy(fixture: Path, tmp_path: Path) -> RunFolder:
    """The fixture as a run folder of its own (the committed one never changes)."""
    root = tmp_path / fixture.name
    shutil.copytree(fixture, root)
    (root / "run.json").write_text(json.dumps({"settings_dir": str(root)}), encoding="utf-8")
    return RunFolder(root)


def bot_answers(path: Path, obs: torch.Tensor, actions: int) -> torch.Tensor:
    flat = _core.Policy(path).logits_batch(obs.flatten().tolist(), obs.shape[0])
    return torch.tensor(flat).reshape(obs.shape[0], actions)


@pytest.mark.parametrize("fixture", SAVES, ids=[p.name for p in SAVES])
def test_a_released_save_loads_and_its_networks_and_bot_answer_the_same(
    fixture: Path, tmp_path: Path
) -> None:
    check_save(fixture, tmp_path)


def fixture_tool() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_fixtures", Path(__file__).parents[1] / "tools" / "make_fixtures.py"
    )
    assert spec and spec.loader
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_every_kind_of_fixture_has_settings_a_run_accepts(tmp_path: Path) -> None:
    """The PC makes the fixtures in one go; a kind whose settings are refused stops it part way (it did
    for self-play once)."""
    from nitro.config import load_config

    tool = fixture_tool()
    for kind in tool.KINDS:
        file = tmp_path / f"{kind}.toml"
        file.write_text(tool.settings_text(tmp_path / "meshes", kind), encoding="utf-8")
        load_config(file)


@pytest.mark.parametrize("kind", ["native", "attention"])
def test_the_fixture_tool_and_these_checks_agree_on_a_made_up_save(kind: str, tmp_path: Path) -> None:
    """Without the meshes nothing trains here, so a save is made by hand the way a run makes one, and goes
    through the fixture tool's answers and the checks above."""
    from nitro import __version__
    from nitro.config import load_config
    from nitro.envs import core_settings
    from nitro.model import build_networks
    from nitro.runs import all_settings, fixed_settings

    tool: Any = fixture_tool()
    work = tmp_path / "work"
    work.mkdir()
    file = work / f"{kind}.toml"
    file.write_text(tool.settings_text(tmp_path / "meshes", kind), encoding="utf-8")
    config = load_config(file)
    obs_size = _core.observation_layout(core_settings(config))[0]
    torch.manual_seed(3)
    model = build_networks(config, obs_size, _core.num_actions())
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(model, config.ppo, gamma, torch.device("cpu"), fp16=False)
    fixture = tmp_path / f"v0-{kind}"
    folder = RunFolder.create(tmp_path, fixture.name, file)
    state = {
        "learner": learner.state_dict(),
        "agent_steps": 1280,
        "settings": fixed_settings(config),
        "all_settings": all_settings(config),
        "nitro_version": __version__,
    }
    folder.save_checkpoint(state, 1280, keep=1, milestone_steps=0)
    export_run(folder, fixture / "bot", None, build=False)
    saved = folder.load_latest()
    assert saved is not None
    expected = tool.answers(saved, folder.load_settings(saved), fixture)
    (fixture / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    check_save(fixture, tmp_path / "checked")


def check_save(fixture: Path, tmp_path: Path) -> None:
    folder = copy(fixture, tmp_path)
    state = folder.load_latest()
    assert state is not None
    config = folder.load_settings(state)  # its fixed settings still fit its networks
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    obs = torch.tensor(expected["obs"])
    model, obs_size, actions = model_from_state(state, config)
    shape = (obs_size, actions)
    with torch.no_grad():
        logits, values = model(obs)
    torch.testing.assert_close(logits, torch.tensor(expected["logits"]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(values, torch.tensor(expected["values"]), rtol=1e-5, atol=1e-5)

    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(model, config.ppo, gamma, torch.device("cpu"), fp16=False)
    learner.load_state_dict(state["learner"], load_scaler=False)  # optimizer, PopArt, reward scale

    # Everywhere else a save is used: the self-play pool's frozen copy of a past version,
    # `start_from` in a new run with the same settings, and `nitro eval`.
    pool_copy = model.fresh_copy()
    pool_copy.load_policy_state_dict(model.policy_state_dict())
    source = Source(state, fixture.name)
    assert problems(source, config) == []
    new_run = Learner(model.fresh_copy(), config.ppo, gamma, torch.device("cpu"), fp16=False)
    apply(source, new_run, config)
    evaluated = eval_model(Side(fixture.name, int(state["agent_steps"]), state, config), *shape, CPU)
    with torch.no_grad():
        for other in (pool_copy, new_run.model, evaluated):
            torch.testing.assert_close(other.logits(obs), logits, rtol=1e-5, atol=1e-5)

    released_bot = bot_answers(fixture / "bot" / "policy.nwt", obs, actions)
    torch.testing.assert_close(released_bot, torch.tensor(expected["bot_logits"]), rtol=1e-5, atol=1e-5)
    exported = export_run(folder, tmp_path / "bot", None, build=False)
    assert exported.folder.is_dir()
    torch.testing.assert_close(
        bot_answers(tmp_path / "bot" / "policy.nwt", obs, actions), released_bot, rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@pytest.mark.parametrize("fixture", SAVES, ids=[p.name for p in SAVES])
def test_a_released_save_resumes_and_trains(fixture: Path, tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    if "fp16" in fixture.name and not torch.cuda.is_available():
        pytest.skip("an FP16 run resumes on a graphics card")
    folder = copy(fixture, tmp_path)
    state = folder.load_latest()
    assert state is not None
    config = folder.load_settings(state)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    run = TrainingRun(config, folder, state)
    row = run.iteration()
    run.metrics.close()
    assert row["agent_steps"] > state["agent_steps"]
