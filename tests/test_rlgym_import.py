"""`nitro import-rlgym`: RLGym-PPO and rlgym-learn saves as those frameworks write them
(tools/rlgym_fixture), the project reader, the plan and its refusals, the input map, and the parity of the
imported bot with RLGym's own observation code and the original networks on real game states."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
from pathlib import Path

import pytest
import torch

from nitro import _core, rlgym_source
from nitro import rlgym_import as ri
from nitro.cli import main
from nitro.export import write_policy
from nitro.gigalearn_import import checked_settings
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
PROJECT = Path(__file__).parent / "fixtures" / "rlgym" / "project"
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason="no collision meshes")

_spec = importlib.util.spec_from_file_location(
    "make_fixture", ROOT / "tools" / "rlgym_fixture" / "make_fixture.py"
)
assert _spec is not None and _spec.loader is not None
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)

DEFAULT_OBS = (
    "DefaultObs(pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, "
    "1 / common_values.CEILING_Z]), ang_coef=1 / np.pi, lin_vel_coef=1 / common_values.CAR_MAX_SPEED, "
    "ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)"
)
IMPORTS = (
    "from rlgym_sim.utils.obs_builders import AdvancedObs\n"
    "from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder\n"
)


def project(tmp_path: Path, obs: str | None = None, team_size: int = 1, **text: str) -> Path:
    """The fixture project with another observation builder, team size or other lines."""
    folder = tmp_path / "project"
    shutil.copytree(PROJECT, folder)
    bot = folder / "bot.py"
    code = bot.read_text()
    obs_call = re.search(r"obs_builder = DefaultObs\((.|\n)*?\n    \)\n", code)
    assert obs_call
    if obs is not None:
        code = code.replace(obs_call.group(0), f"obs_builder = {obs}\n")
        code = IMPORTS + code
    code = code.replace("team_size=1", f"team_size={team_size}")
    for old, new in text.items():
        assert old in code, old
        code = code.replace(old, new)
    bot.write_text(code)
    return folder


def obs_size(obs: str, team_size: int) -> int:
    cars = 2 * team_size
    if obs.startswith("DefaultObs"):
        return 51 + 19 * cars
    if obs.startswith("AdvancedObs("):
        return 51 + 25 + 31 * (cars - 1)
    places = int(obs.split("=")[1].rstrip(")")) if "=" in obs else 3
    return 51 + 26 + 32 * (2 * places - 1)


def plan_for(
    tmp_path: Path, obs: str = DEFAULT_OBS, team_size: int = 1, **kw: object
) -> tuple[ri.RlgymSave, ri.RlgymPlan]:
    save = ri.read_save(fixture.write(tmp_path / "save", obs_size(obs, team_size), **kw))
    source = rlgym_source.read(project(tmp_path, None if obs == DEFAULT_OBS else obs, team_size))
    return save, ri.build_plan(save, source)


def test_both_save_layouts_are_read(tmp_path: Path) -> None:
    ppo = ri.read_save(fixture.write(tmp_path / "ppo", 89).parents[1].parent)  # the checkpoints folder
    assert ppo.kind == "RLGym-PPO" and ppo.folder.name == "654321"
    assert [w.shape for w, _ in ppo.policy] == [(32, 89), (24, 32), (90, 24)]
    assert [w.shape for w, _ in ppo.critic] == [(24, 89), (1, 24)]
    assert ppo.steps == 654321 and ppo.iterations == 12 and ppo.returns == (0.1, 3200.0, 801)
    assert ppo.obs_stats is not None and ppo.obs_stats[2] == 2000
    assert ppo.policy_optim is not None and len(ppo.policy_optim["state"]) == 6
    learn = ri.read_save(fixture.write(tmp_path / "learn", 89, learn=True).parent)
    assert learn.kind == "rlgym-learn" and learn.steps == 654321 and learn.iterations == 12
    assert [w.shape for w, _ in learn.policy] == [(32, 89), (24, 32), (90, 24)]
    assert learn.returns == (0.1, 3200.0, 801) and learn.critic_optim is not None
    with pytest.raises(ri.RlgymImportError, match="no RLGym-PPO save"):
        ri.read_save(tmp_path / "project")


def test_the_project_reader_reads_the_learner_and_the_environment() -> None:
    source = rlgym_source.read(PROJECT)
    assert source.file.name == "bot.py" and not source.unread
    assert source.learner["ppo_ent_coef"] == 0.01 and source.learner["policy_layer_sizes"] == [32, 24]
    assert source.make is not None and source.make.kwargs["tick_skip"] == 8
    assert source.obs.name == "DefaultObs" and source.obs.module == "rlgym_sim.utils.obs_builders"
    assert source.obs.kwargs["pos_coef"] == [1 / 4096, 1 / 6000, 1 / 2044]
    assert source.action.name == "LookupAction"
    assert [c.name for c in source.terminals] == ["NoTouchTimeoutCondition", "GoalScoredCondition"]
    assert source.terminals[0].args == (150,)
    assert source.classes["SpeedFlipReward"] == "my_rewards.py:5"


def test_the_plan_maps_settings_rewards_and_the_action_delay(tmp_path: Path) -> None:
    save, plan = plan_for(tmp_path)
    assert not plan.refused
    config = checked_settings(plan.toml, tmp_path, MESHES)[1]
    assert config.env.observation == "gigalearn_default" and config.env.action_masks == "none"
    assert config.env.action_delay == 1 and config.env.tick_skip == 8
    assert config.ppo.entropy == pytest.approx(0.01 * math.log(90))
    assert config.ppo.epochs == 2 and config.ppo.minibatches == 1 and not config.ppo.normalize_advantages
    assert config.model.input_map == 89 and config.model.policy == (32, 24) and config.model.value == (24,)
    assert config.terminals.no_touch_seconds == 10
    weights = {r["name"]: r["weight"] for r in plan.rewards}
    assert weights == {
        "velocity_player_to_ball": 0.01,
        "velocity_ball_to_goal": 0.1,
        "face_ball": 0.02,
        "rlgym_liu_distance_player_to_ball": 0.05,
        "rlgym_goal": 10.0,
        "rlgym_touch_start": 5.0,
        "demo": 3.0,
        "rlgym_boost_gain": 1.0,
    }
    assert any("shot part" in line for line in plan.to_port)
    assert any("SpeedFlipReward" in line and "my_rewards.py:5" in line for line in plan.to_port)
    kept = ri.build_plan(save, rlgym_source.read(PROJECT), keep_action_delay=True)
    assert checked_settings(kept.toml, tmp_path, MESHES)[1].env.action_delay == 7


def test_every_step_rewards_are_rescaled_for_the_tick_skip(tmp_path: Path) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    source = rlgym_source.read(project(tmp_path, None, 1, **{"tick_skip = 8": "tick_skip = 4"}))
    weights = {r["name"]: r["weight"] for r in ri.build_plan(save, source).rewards}
    assert weights["face_ball"] == pytest.approx(0.04) and weights[
        "rlgym_liu_distance_player_to_ball"
    ] == pytest.approx(0.1)
    assert weights["rlgym_goal"] == 10.0, "an event pays the same at any tick skip"


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"obs_builder=obs_builder": "obs_builder=MyObs()"}, "not one of RLGym's built-in ones"),
        ({"action_parser=LookupAction()": "action_parser=LookupAction(bins=[[-1, 1]] * 5)"}, "its own bins"),
        ({"action_parser=LookupAction()": "action_parser=MyParser()"}, "not rlgym-tools' LookupAction"),
        ({"action_parser=LookupAction()": "action_parser=ContinuousAction()"}, "continuous"),
        ({"spawn_opponents=True": "spawn_opponents=False"}, "spawn_opponents"),
        ({"team_size=1": "team_size=2"}, "do not belong together"),
        ({"rlgym_sim.make(": "RLGym("}, "RLGym 2 bot"),
    ],
)
def test_what_cannot_be_imported_is_refused_in_plain_words(
    tmp_path: Path, change: dict[str, str], says: str
) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    plan = ri.build_plan(save, rlgym_source.read(project(tmp_path, None, 1, **change)))
    assert any(says in line for line in plan.refused), plan.refused


def test_rlgym_2_observations_are_refused_with_what_is_missing(tmp_path: Path) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    code = {
        "from rlgym_ppo import Learner": "from rlgym_ppo import Learner\nfrom "
        "rlgym.rocket_league.obs_builders import DefaultObs as V2"
    }
    source = rlgym_source.read(project(tmp_path, "V2()", 1, **code))
    plan = ri.build_plan(save, source)
    assert any("RLGym 2's" in line and "jumping or flipping" in line for line in plan.refused)
    learn = ri.read_save(fixture.write(tmp_path / "learn", 89, learn=True))
    assert any("RLGym 2 bot" in line for line in ri.build_plan(learn, rlgym_source.read(PROJECT)).refused)


def test_standardisation_needs_the_saved_statistics(tmp_path: Path) -> None:
    save, plan = plan_for(tmp_path, obs_stats_too=False)
    ri.standardisation(save, plan)
    assert any("no obs_running_stats" in line for line in plan.refused)


OBSERVATIONS = [
    (DEFAULT_OBS, 1, "gigalearn_default"),
    (DEFAULT_OBS, 2, "gigalearn_default"),
    ("AdvancedObs()", 1, "gigalearn_padder"),
    ("AdvancedObs()", 2, "gigalearn_padder"),
    ("AdvancedObsPadder()", 1, "rlgym_padder"),
    ("AdvancedObsPadder(team_size=2)", 2, "rlgym_padder"),
]


@needs_meshes
@pytest.mark.parametrize(("obs", "team_size", "kind"), OBSERVATIONS)
def test_each_observation_matches_rlgyms_own_code_on_real_states(
    tmp_path: Path, obs: str, team_size: int, kind: str
) -> None:
    save, plan = plan_for(tmp_path, obs, team_size)
    assert not plan.refused, plan.refused
    assert plan.observation == kind
    config = checked_settings(plan.toml, tmp_path, MESHES)[1]
    width = obs_size(obs, team_size) + (0 if kind != "gigalearn_padder" else 2 * team_size)
    model = ri.build_model(config, plan, save, width)
    observed, states = ri.game_states(config, model, 25)
    parity = ri.check(model, save, plan, observed, states)
    assert parity.passed, parity.line()
    assert parity.largest_obs_difference < 1e-5
    # A wrong scale is caught.
    with torch.no_grad():
        model.input_map.scale[2] *= 1.5  # the ball's height
    assert ri.check(model, save, plan, observed, states).largest_obs_difference > 1e-3
    with torch.no_grad():
        model.input_map.scale[2] /= 1.5
    # So is a critic that does not give the original's values: the value is checked too.
    assert ri.check(model, save, plan, observed, states).passed
    with torch.no_grad():
        model.value_output.bias += 0.01
    assert not ri.check(model, save, plan, observed, states).passed


@needs_meshes
def test_import_writes_a_run_that_resumes_and_exports(tmp_path: Path) -> None:
    checkpoint = fixture.write(tmp_path / "save", 89).parents[1]
    report = ri.import_bot(
        checkpoint, PROJECT, "ported", tmp_path / "runs", meshes=MESHES, check_steps=30, say=lambda _: None
    )
    assert report.run is not None and report.parity is not None and report.parity.passed, report.lines()
    folder = RunFolder(tmp_path / "runs" / "ported")
    state = folder.load_latest()
    assert state is not None and state["agent_steps"] == 654321 and state["iterations"] == 12
    config = folder.load_settings(state)
    # Adam's moments are the save's, parameter for parameter.
    saved = torch.load(next(checkpoint.rglob("PPO_POLICY_OPTIMIZER.pt")), weights_only=True)
    moments = state["learner"]["optimizer"]["state"]
    assert any(torch.equal(m["exp_avg"], saved["state"][0]["exp_avg"]) for m in moments.values())
    scaler = state["learner"]["reward_scaler"]
    assert float((scaler["m2"] / scaler["count"]).sqrt()) == pytest.approx(2.0)
    assert "PASSED" in (folder.root / "imported.txt").read_text()
    # The bot file carries the input map: the core's policy gives the PyTorch model's logits.
    from nitro.model import ActorCritic

    model = ActorCritic(config.model.input_map and 89, 90, config.model)
    model.load_state_dict(state["learner"]["model"])
    model.eval()
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, 89, 90, "ported", 654321)
    policy = _core.Policy(path)
    assert policy.header()["format"] == 9 and policy.obs_size == 89
    obs = torch.randn(64, 89) * 0.5
    with torch.no_grad():
        expected = model(obs)[0]
    got = torch.tensor(policy.logits_batch(obs.reshape(-1).tolist(), 64)).reshape(64, 90)
    assert (got - expected).abs().max().item() < 1e-4


@needs_meshes
def test_a_refused_import_writes_nothing(tmp_path: Path) -> None:
    checkpoint = fixture.write(tmp_path / "save", 89)
    source = project(tmp_path, None, 1, **{"action_parser=LookupAction()": "action_parser=MyParser()"})
    report = ri.import_bot(checkpoint, source, "ported", tmp_path / "runs", meshes=MESHES)
    assert report.run is None and report.plan.refused
    assert not (tmp_path / "runs").exists()


def test_the_command_prints_why_it_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = fixture.write(tmp_path / "save", 89)
    source = project(tmp_path, None, 1, **{"action_parser=LookupAction()": "action_parser=MyParser()"})
    code = main(
        [
            "import-rlgym",
            str(checkpoint),
            "--source",
            str(source),
            "--into",
            "x",
            "--runs",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 1 and "not rlgym-tools' LookupAction" in capsys.readouterr().out


def test_lookup_action_is_nitrolearns_table_in_order() -> None:
    from nitro.rlgym_reference.lookup_act import LookupAction

    table = LookupAction.make_lookup_table(LookupAction().bins)
    ours = [list(_core.action_controls(i)) for i in range(90)]
    assert ours == [list(map(float, row)) for row in table]


@pytest.mark.parametrize(
    "weights", ["reward_weights=W", "reward_weights=tuple(W)", "reward_weights=(0.01, 0.1)"]
)
def test_reward_weights_that_cannot_be_read_are_refused_not_taken_as_one(
    tmp_path: Path, weights: str
) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    change = {"reward_weights=(0.01, 0.1, 0.02, 0.05, 10.0, 1.0)": weights}
    plan = ri.build_plan(save, rlgym_source.read(project(tmp_path, None, 1, **change)))
    assert any("weights could not be read" in line for line in plan.refused), plan.refused


def test_no_reward_weights_are_rlgyms_ones(tmp_path: Path) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    change = {"reward_weights=(0.01, 0.1, 0.02, 0.05, 10.0, 1.0),": ""}
    plan = ri.build_plan(save, rlgym_source.read(project(tmp_path, None, 1, **change)))
    assert not plan.refused, plan.refused
    assert {r["name"]: r["weight"] for r in plan.rewards}["rlgym_goal"] == 1.0


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"policy_lr=2e-4": "policy_lr=get_lr()"}, "policy_lr"),
        ({"ppo_ent_coef=0.01": "ppo_ent_coef=ENT"}, "ppo_ent_coef"),
        ({"ts_per_iteration=50_000": "ts_per_iteration='50000'"}, "ts_per_iteration"),
        ({"ppo_batch_size=50_000": "ppo_batch_size=0"}, "ppo_batch_size"),
        ({"gae_gamma=0.995": "gae_gamma=1.0"}, "gae_gamma"),
        ({"tick_skip = 8\n": "tick_skip = 0\n"}, "tick_skip"),
    ],
)
def test_learner_settings_that_cannot_be_used_are_refused_not_a_crash(
    tmp_path: Path, change: dict[str, str], says: str
) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    plan = ri.build_plan(save, rlgym_source.read(project(tmp_path, None, 1, **change)))
    assert any(says in line for line in plan.refused), plan.refused


def test_the_reader_leaves_arithmetic_it_cannot_do_unread(tmp_path: Path) -> None:
    bot = tmp_path / "bot.py"
    bot.write_text(
        "import rlgym_sim\nfrom rlgym_ppo import Learner\n"
        "a = 1 / 0\nb = 9 ** 9 ** 9\nc = (-1) ** 0.5\n"
        "env = rlgym_sim.make(tick_skip=a)\nLearner(env, policy_lr=b, critic_lr=c, ppo_epochs=2 ** 3)\n"
    )
    source = rlgym_source.read(bot)
    assert source.make is not None and isinstance(source.make.kwargs["tick_skip"], rlgym_source.Unread)
    assert isinstance(source.learner["policy_lr"], rlgym_source.Unread)
    assert isinstance(source.learner["critic_lr"], rlgym_source.Unread)
    assert source.learner["ppo_epochs"] == 8


class _Planted:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (print, ("planted code ran",))


def _damaged(tmp_path: Path, what: str) -> Path:
    folder = fixture.write(tmp_path / "save", 89)
    book = folder / "BOOK_KEEPING_VARS.json"
    if what == "garbage":
        (folder / "PPO_POLICY.pt").write_bytes(b"not a save")
    elif what == "code":
        torch.save({"x": _Planted()}, folder / "PPO_VALUE_NET.pt")
    elif what == "code in the optimizer":
        torch.save({"x": _Planted()}, folder / "PPO_POLICY_OPTIMIZER.pt")
    elif what == "a list":
        torch.save([1, 2], folder / "PPO_POLICY.pt")
    elif what == "broken json":
        book.write_text("{oops")
    elif what == "no count":
        book.write_text(book.read_text().replace('"count": 801', '"counted": 801'))
    elif what == "short variances":
        data = json.loads(book.read_text())
        data["obs_running_stats"]["var"] = [1.0] * 5
        book.write_text(json.dumps(data))
    elif what == "no steps":
        book.write_text(
            book.read_text().replace('"cumulative_timesteps": 654321', '"cumulative_timesteps": null')
        )
    return folder


@pytest.mark.parametrize(
    "what",
    [
        "garbage",
        "code",
        "code in the optimizer",
        "a list",
        "broken json",
        "no count",
        "short variances",
        "no steps",
    ],
)
def test_a_damaged_or_planted_save_is_refused_in_plain_words(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], what: str
) -> None:
    folder = _damaged(tmp_path, what)
    with pytest.raises(ri.RlgymImportError) as refused:
        ri.read_save(folder)
    assert "weights_only" not in str(refused.value)
    assert "planted code ran" not in capsys.readouterr().out


def test_game_settings_nitrolearn_does_not_have_are_listed(tmp_path: Path) -> None:
    save = ri.read_save(fixture.write(tmp_path / "save", 89))
    change = {"spawn_opponents=True,": "spawn_opponents=True, gravity=0.5, boost_consumption=0,"}
    plan = ri.build_plan(save, rlgym_source.read(project(tmp_path, None, 1, **change)))
    listed = " ".join(plan.changed)
    assert "gravity" in listed and "boost_consumption" in listed, plan.changed
    plain = ri.build_plan(save, rlgym_source.read(PROJECT))
    assert not any("gravity" in line for line in plain.changed)


def test_a_folder_with_several_learners_asks_which_one(tmp_path: Path) -> None:
    folder = project(tmp_path)
    (folder / "bot.py").rename(folder / "my_bot.py")
    (folder / "example.py").write_text(
        "from rlgym_ppo import Learner\n\ndef env():\n    pass\n\nLearner(env, ppo_ent_coef=0.5)\n"
    )
    with pytest.raises(rlgym_source.SourceError, match=r"example\.py.*my_bot\.py"):
        rlgym_source.read(folder)
    assert rlgym_source.read(folder / "my_bot.py").learner["ppo_ent_coef"] == 0.01
    # A copy of RLGym-PPO's own code (its `class Learner(...)`) is not a project file that makes one.
    (folder / "example.py").write_text("class Learner(object):\n    pass\n")
    assert rlgym_source.read(folder).file.name == "my_bot.py"


@pytest.mark.parametrize("name", ["../escaped", "a/b", "a\\b", "", ".hidden"])
def test_a_run_name_that_is_not_a_plain_folder_name_is_refused_first(tmp_path: Path, name: str) -> None:
    checkpoint = fixture.write(tmp_path / "save", 89)
    with pytest.raises(ri.RlgymImportError, match="name"):
        ri.import_bot(checkpoint, PROJECT, name, tmp_path / "runs", meshes=MESHES, say=lambda _: None)
    assert not (tmp_path / "runs").exists()
