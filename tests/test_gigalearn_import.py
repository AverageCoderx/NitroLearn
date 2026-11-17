"""`nitro import-gigalearn`: GigaLearn saves made by LibTorch itself, the
project reader, the plan, and the networks' parity with LibTorch's own outputs and on real game states."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from nitro import gigalearn_import as gi
from nitro import gigalearn_source
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings
from nitro.gigalearn_source import Call, Source
from nitro.runs import RunFolder

HERE = Path(__file__).parent / "fixtures" / "gigalearn"
PROJECT = HERE / "project"
SAVES = HERE / "saves"
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason="no collision meshes")


def reference(save: str) -> dict[str, torch.Tensor]:
    """The inputs and outputs LibTorch computed when it wrote the save (make_fixture.cpp)."""
    out = {}
    for line in (SAVES / save / "reference.txt").read_text().splitlines():
        name, count, *values = line.split()
        assert len(values) == int(count)
        out[name] = torch.tensor([float(v) for v in values])
    return out


def source_with(**settings: object) -> Source:
    source = gigalearn_source.read(PROJECT)
    source.settings.update(settings)
    return source


def plain_source(obs: str, **settings: object) -> Source:
    """A project with only an observation and settings, its one reward an air reward."""
    entry = gigalearn_source.WeightedEntry(Call("AirReward", (), 1, "new AirReward()"), 1.0, 1)
    return Source(
        PROJECT / "x.cpp", rewards=[entry], obs=Call(obs, (), 1, f"new {obs}()"), settings=dict(settings)
    )


def config_of(plan: gi.Plan) -> NitroConfig:
    text = f"[paths]\ncollision_meshes = {json.dumps(MESHES.resolve().as_posix())}\n" + plan.toml
    return parse_toml(text, NitroConfig, file=Path("settings.toml"), base_dir=Path.cwd())


def test_the_reader_finds_the_layers_from_the_weights() -> None:
    save = gi.read_save(SAVES / "advanced_shared")
    assert save.folder.name == "123456"
    assert save.shared is not None and save.shared.widths == ((109, 32),) and save.shared.norm
    assert save.policy.widths == ((32, 32), (32, 24), (24, 90))
    assert save.critic.widths == ((32, 24), (24, 1))
    assert set(save.optimizers) == {"shared", "policy", "critic"}
    assert save.stats["total_timesteps"] == 123456
    plain = gi.read_save(SAVES / "default_plain" / "checkpoints" / "123456")
    assert plain.shared is None and not plain.policy.norm and plain.policy.inputs == 89


def test_the_project_reader_reads_rewards_settings_and_parts() -> None:
    source = gigalearn_source.read(PROJECT)
    assert source.file.name == "BotMain.cpp" and not source.unread
    names = [e.reward.name for e in source.rewards]
    assert names[:3] == ["AirReward", "FaceBallReward", "VelocityPlayerToBallReward"]
    assert "ShotReward" not in names, "a commented-out line is not read"
    zero_sum = source.rewards[8]
    assert (
        zero_sum.reward.name == "ZeroSumReward"
        and zero_sum.reward.args[1:] == (0.5, 0.8)
        and zero_sum.weight == 15
    )
    assert source.settings["ppo.tsPerItr"] == 100_000 and source.settings["ppo.batchSize"] == 50_000
    assert source.settings["actionDelay"] == 7 and source.settings["ppo.policy.activationType"] == "RELU"
    assert source.obs is not None and source.obs.name == "AdvancedObs"
    assert [t.name for t in source.terminals] == ["NoTouchCondition", "GoalScoreCondition"]
    assert source.classes["MyAerialReward"] == "src/MyRewards.h:6"
    # Rewards added to the project's own copy of GigaLearn are found too, with their code.
    where = source.classes["AddedInGigaLearnReward"]
    assert where.startswith("GigaLearnCPP/RLGymCPP/") and where.endswith(":6")
    assert "return player.boost > 50;" in source.class_code["AddedInGigaLearnReward"]


def test_the_plan_maps_every_built_in_and_lists_the_rest() -> None:
    save = gi.read_save(SAVES / "advanced_shared")
    plan = gi.build_plan(save, gigalearn_source.read(PROJECT))
    assert not plan.refused
    assert (plan.observation, plan.team_size, plan.activation) == ("gigalearn_advanced", 1, "relu")
    by_name = {block["name"]: block for block in plan.rewards}
    assert by_name["gigalearn_strong_touch"] == {
        "name": "gigalearn_strong_touch",
        "min_speed_kph": 25.0,
        "max_speed_kph": 90.0,
        "weight": 40.0,
    }
    assert by_name["bump"]["zero_sum"] == {"team_spirit": 0.5, "opponent_scale": 0.8}
    assert by_name["gigalearn_velocity_ball_to_goal"] == {
        "name": "gigalearn_velocity_ball_to_goal",
        "own_goal": True,
        "weight": -0.5,
    }
    assert by_name["goal"]["concede_scale"] == -0.8
    assert len(plan.rewards) == 20 == len(plan.reward_origins)
    # The community KickoffProximityReward and SpeedTowardsOpponentReward, recognised by their code.
    assert by_name["kickoff_proximity"] == {
        "name": "kickoff_proximity",
        "zero_within_200": False,
        "weight": 5.0,
    }
    assert by_name["speed_towards_opponent"] == {
        "name": "speed_towards_opponent",
        "scale": 1.5,
        "weight": 2.0,
    }
    assert not any("KickoffProximity" in line or "SpeedTowards" in line for line in plan.to_port)
    assert any("SaveReward" in line and "event tracker" in line for line in plan.to_port)
    assert any("MyAerialReward" in line and "src/MyRewards.h:6" in line for line in plan.to_port)
    assert [s["name"] for s in plan.setters] == ["kickoff", "random"]
    config = config_of(plan)
    assert config.ppo.minibatches == 2 and config.ppo.microbatch == 25_000
    assert config.ppo.normalize_advantages is False and config.ppo.popart is False
    assert config.env.action_delay == 1, "the lowest delay, not GigaLearn's 7"
    assert config.terminals.no_touch_seconds == 12
    assert any(line.startswith("Action delay: GigaLearn's actionDelay 7 set to 1") for line in plan.changed)
    kept = gi.build_plan(save, gigalearn_source.read(PROJECT), keep_action_delay=True)
    assert config_of(kept).env.action_delay == 7


def test_every_step_rewards_are_rescaled_for_the_tick_skip() -> None:
    save = gi.read_save(SAVES / "advanced_shared")
    plan = gi.build_plan(save, source_with(tickSkip=4, actionDelay=3))
    by_name = {block["name"]: block["weight"] for block in plan.rewards}
    assert by_name["air"] == pytest.approx(0.6), "NitroLearn halves every-step pay at tick skip 4"
    assert by_name["pickup_boost"] == 8.0, "event rewards are not scaled"
    gamma = 0.995
    half = config_of(plan).ppo.discount_half_life
    assert 0.5 ** (4 / (120 * half)) == pytest.approx(gamma, abs=1e-12)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"obs": Call("MyObs", (), 50, "new MyObs()")}, "MyObs"),
        ({"obs": Call("DefaultObsPadded", (3,), 50, "new DefaultObsPadded(3)")}, "pads"),
        ({"action": Call("MyParser", (), 49, "new MyParser()")}, "MyParser"),
        (
            {
                "settings": {
                    "ppo.policy.activationType": "SIGMOID",
                    "ppo.critic.activationType": "SIGMOID",
                    "ppo.sharedHead.activationType": "SIGMOID",
                }
            },
            "SIGMOID",
        ),
        ({"settings": {"ppo.policy.activationType": "TANH"}}, "different activations"),
    ],
)
def test_what_cannot_be_converted_is_refused_in_plain_words(change: dict[str, object], says: str) -> None:
    save = gi.read_save(SAVES / "advanced_shared")
    source = gigalearn_source.read(PROJECT)
    for key, value in change.items():
        if key == "settings":
            assert isinstance(value, dict)
            source.settings.update(value)
        else:
            setattr(source, key, value)
    plan = gi.build_plan(save, source)
    assert any(says in line for line in plan.refused), plan.refused


def test_another_action_list_is_refused_without_a_setting_this_version_lacks() -> None:
    """A policy with another number of outputs, or its own action parser, is refused plainly; the reason
    names no settings section this version refuses ([actions] is not a known setting)."""
    save = gi.read_save(SAVES / "advanced_shared")
    last = save.policy.widths[-1]
    save.policy = dataclasses.replace(save.policy, widths=(*save.policy.widths[:-1], (last[0], 120)))
    source = gigalearn_source.read(PROJECT)
    refused = gi.build_plan(save, source).refused
    assert any("120 outputs" in line for line in refused), refused
    source.action = Call("MyParser", (), 49, "new MyParser()")
    refused += gi.build_plan(gi.read_save(SAVES / "advanced_shared"), source).refused
    assert any("MyParser" in line for line in refused), refused
    assert not any("[actions]" in line for line in refused), refused


def test_no_source_needs_the_activation_and_still_refuses_without_rewards() -> None:
    save = gi.read_save(SAVES / "advanced_shared")
    plan = gi.build_plan(save, None)
    assert any("--activation" in line for line in plan.refused)
    plan = gi.build_plan(gi.read_save(SAVES / "default_plain"), None, activation="relu")
    assert plan.observation == "gigalearn_default" and any("rewards" in line for line in plan.refused)


@pytest.mark.parametrize(
    ("save", "source"),
    [
        ("advanced_shared", lambda: gigalearn_source.read(PROJECT)),
        (
            "default_plain",
            lambda: plain_source(
                "DefaultObs",
                **{f"ppo.{n}.activationType": "LEAKY_RELU" for n in ("policy", "critic", "sharedHead")},
                **{"ppo.policyTemperature": 0.8},
            ),
        ),
    ],
)
def test_the_imported_networks_give_libtorchs_own_outputs(save: str, source: object) -> None:
    """The policy and value as LibTorch computed them with GigaLearn's rule, against NitroLearn's networks."""
    loaded = gi.read_save(SAVES / save)
    plan = gi.build_plan(loaded, source())  # type: ignore[operator]
    assert not plan.refused, plan.refused
    model = gi.build_model(config_of(plan), plan, loaded)
    ref = reference(save)
    obs = ref["obs"].view(16, -1)
    mask = ref["mask"].view(16, 90).bool()
    probs, values = gi.nitro_probs(model, obs, mask)
    probs = torch.where(mask, probs, torch.full_like(probs, gi.MIN_PROB))
    assert (probs - ref["probs"].view(16, 90)).abs().max() < 1e-6
    assert (values - ref["values"]).abs().max() < 1e-5
    # The rebuilt GigaLearn networks (the import's own check) agree with LibTorch too.
    nets = gi.gigalearn_networks(loaded, plan.activation)
    ref_probs, ref_values = gi.gigalearn_probs(nets, obs, mask, plan.temperature, None)
    assert (ref_probs - ref["probs"].view(16, 90)).abs().max() < 1e-6
    assert (ref_values - ref["values"]).abs().max() < 1e-5


def test_folded_observation_standardisation_gives_the_same_policy() -> None:
    loaded = gi.read_save(SAVES / "advanced_shared")
    source = source_with(standardizeObs=True)
    plan = gi.build_plan(loaded, source)
    generator = torch.Generator().manual_seed(1)
    loaded.stats["obs_stat"] = {
        "means": (torch.randn(109, generator=generator) * 2).tolist(),
        "vars": (torch.rand(109, generator=generator) * 500).tolist(),
        "count": 300,
    }
    plan.obs_norm = gi.obs_normalisation(loaded, plan)
    assert plan.obs_norm is not None and float(plan.obs_norm[0].abs().max()) <= 3
    model = gi.build_model(config_of(plan), plan, loaded)
    obs = torch.randn(64, 109, generator=generator) * 3
    mask = torch.rand(64, 90, generator=generator) > 0.3
    parity = gi.compare(model, loaded, plan, obs, mask)
    assert parity.passed, parity.line()


def test_a_wrong_mapping_fails_the_parity_check() -> None:
    loaded = gi.read_save(SAVES / "advanced_shared")
    plan = gi.build_plan(loaded, gigalearn_source.read(PROJECT))
    model = gi.build_model(config_of(plan), plan, loaded)
    first = model.policy[0]
    assert isinstance(first, torch.nn.Linear)
    with torch.no_grad():
        first.weight[0, 0] += 0.5
    generator = torch.Generator().manual_seed(2)
    obs, mask = torch.randn(64, 109, generator=generator), torch.ones(64, 90, dtype=torch.bool)
    assert not gi.compare(model, loaded, plan, obs, mask).passed


def test_the_reward_fixture_of_the_core_test_is_what_the_import_makes() -> None:
    """crates/nitro_core/src/gigalearn_rewards.rs holds these rewards to GigaLearn's formulas."""
    save = gi.read_save(SAVES / "advanced_shared")
    for tick_skip in (8, 4):
        plan = gi.build_plan(save, source_with(tickSkip=tick_skip, actionDelay=tick_skip - 1))
        made = {
            "tick_skip": tick_skip,
            "gigalearn": plan.reward_origins,
            "terms": json.loads(core_settings(config_of(plan)))["rewards"],
        }
        path = HERE / f"rewards_tick{tick_skip}.json"
        if os.environ.get("NITRO_WRITE_REFERENCE"):
            path.write_text(json.dumps(made, indent=1) + "\n")
        assert json.loads(path.read_text()) == json.loads(json.dumps(made))


@needs_meshes
def test_import_writes_a_run_that_resumes_from_the_gigalearn_step(tmp_path: Path) -> None:
    report = gi.import_bot(
        SAVES / "advanced_shared",
        PROJECT,
        "ported",
        tmp_path,
        meshes=MESHES,
        check_steps=40,
        say=lambda _: None,
    )
    assert report.run is not None and report.parity is not None and report.parity.passed
    assert report.parity.states == 40 * 4 * 2, "4 arenas of 2 cars"
    folder = RunFolder(tmp_path / "ported")
    state = folder.load_latest()
    assert state is not None and state["agent_steps"] == 123456 and state["iterations"] == 42
    config = folder.load_settings(state)
    assert config.env.observation == "gigalearn_advanced" and config.model.shared == (32,)
    # Adam's moments are LibTorch's, parameter for parameter.
    adam = gi._adam_state(SAVES / "advanced_shared" / "checkpoints" / "123456" / "POLICY_OPTIM.lt")
    assert adam is not None
    saved = state["learner"]["optimizer"]["state"]
    groups = state["learner"]["optimizer"]["param_groups"]
    policy_group = next(g for g in groups if len(g["params"]) == len(adam))
    first = saved[policy_group["params"][0]]
    assert torch.equal(first["exp_avg"], adam[0]["exp_avg"]) and float(first["step"]) == 3.0
    scaler = state["learner"]["reward_scaler"]
    assert float((scaler["m2"] / scaler["count"]).sqrt()) == pytest.approx((9000.0 / 4499) ** 0.5)
    assert "PASSED" in (folder.root / "imported.txt").read_text()
    assert not list(tmp_path.glob(".*")), "nothing half-written is left"


@needs_meshes
def test_a_refused_import_writes_nothing(tmp_path: Path) -> None:
    report = gi.import_bot(SAVES / "advanced_shared", None, "ported", tmp_path, meshes=MESHES)
    assert report.run is None and report.plan.refused
    assert not any(tmp_path.iterdir())


def test_an_existing_name_is_refused(tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    with pytest.raises(gi.GigaLearnImportError, match="already exists"):
        gi.import_bot(SAVES / "advanced_shared", PROJECT, "taken", tmp_path)


def test_the_command_prints_why_it_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from nitro.cli import main

    code = main(["import-gigalearn", str(SAVES / "advanced_shared"), "--into", "x", "--runs", str(tmp_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "Cannot import:" in out and "--activation" in out
    assert main(["import-gigalearn", str(tmp_path), "--into", "x", "--runs", str(tmp_path)]) == 1
    assert "holds no GigaLearn save" in capsys.readouterr().err


def test_an_own_reward_with_other_code_is_not_mapped_by_its_name() -> None:
    source = gigalearn_source.read(PROJECT)
    source.class_code["KickoffProximityReward"] = source.class_code["KickoffProximityReward"].replace(
        "1.0f", "2.0f"
    )
    plan = gi.build_plan(gi.read_save(SAVES / "advanced_shared"), source)
    assert "kickoff_proximity" not in {block["name"] for block in plan.rewards}
    assert any("KickoffProximityReward" in line and "your own reward" in line for line in plan.to_port)


def padder_project(tmp_path: Path, team_size: int) -> Path:
    """The test project with AdvancedObsPadderGGL(team_size) as its observation."""
    project = tmp_path / "project"
    shutil.copytree(PROJECT, project)
    main = project / "src" / "BotMain.cpp"
    text = main.read_text().replace("new AdvancedObs()", f"new AdvancedObsPadderGGL({team_size})")
    main.write_text(text)
    return project


def test_advanced_obs_padder_maps_to_gigalearn_padder(tmp_path: Path) -> None:
    save = gi.read_save(
        SAVES / "advanced_shared"
    )  # 109 inputs: AdvancedObs or AdvancedObsPadderGGL(1) in 1v1
    plan = gi.build_plan(save, gigalearn_source.read(padder_project(tmp_path, 1)))
    assert not plan.refused, plan.refused
    assert (plan.observation, plan.team_size, plan.obs_room) == ("gigalearn_padder", 1, 0)
    plan = gi.build_plan(save, gigalearn_source.read(padder_project(tmp_path / "3", 3)))
    assert any("gives 237 numbers" in line for line in plan.refused)
    # With room for 3 a side in a 1v1 project: env.obs_team_size = 3.
    room = gi.build_plan(save, gigalearn_source.read(PROJECT))
    room.refused.clear()
    gi._plan_padder(
        Call("AdvancedObsPadderGGL", (3,), 72, "new AdvancedObsPadderGGL(3)"),
        237,
        gigalearn_source.read(PROJECT),
        room,
    )
    assert (room.observation, room.team_size, room.obs_room) == (
        "gigalearn_padder",
        1,
        3,
    ) and not room.refused
    no_source = gi.build_plan(save, None, activation="relu")
    assert any("AdvancedObsPadderGGL(1)" in line for line in no_source.refused), "109 numbers are ambiguous"


@needs_meshes
def test_a_padder_bot_imports_and_passes_on_real_states(tmp_path: Path) -> None:
    report = gi.import_bot(
        SAVES / "advanced_shared",
        padder_project(tmp_path, 1),
        "padded",
        tmp_path / "runs",
        meshes=MESHES,
        check_steps=40,
        say=lambda _: None,
    )
    assert report.run is not None and report.parity is not None and report.parity.passed, report.lines()
    config = RunFolder(report.run).load_settings()
    assert config.env.observation == "gigalearn_padder"


def _scripted_save_with_code(folder: Path) -> Path:
    """A GigaLearn save whose POLICY.lt also holds TorchScript code that runs when torch.jit.load reads it."""
    from typing import Tuple  # noqa: UP035 - TorchScript reads the annotation

    class Planted(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.layer(x)
            return out

        @torch.jit.export
        def __getstate__(self) -> Tuple[int, bool]:  # noqa: UP006
            return (1, self.training)

        @torch.jit.export
        def __setstate__(self, state: Tuple[int, bool]) -> None:  # noqa: UP006
            self.training = state[1]
            print("planted code ran")

    shutil.copytree(SAVES / "default_plain", folder)
    policy = folder / "checkpoints" / "123456" / "POLICY.lt"
    torch.jit.save(torch.jit.script(Planted()), str(policy))
    return folder


def test_a_saves_own_code_never_runs(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """A GigaLearn save is read as data: TorchScript code in it is never run (torch.jit.load would)."""
    folder = _scripted_save_with_code(tmp_path / "save")
    with pytest.raises(gi.GigaLearnImportError, match=r"POLICY\.lt"):
        gi.read_save(folder)
    assert "planted code ran" not in capfd.readouterr().out


def test_the_weights_read_are_libtorchs_own() -> None:
    import warnings

    for file in sorted(SAVES.glob("*/checkpoints/*/*.lt")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            module = torch.jit.load(str(file))  # type: ignore[no-untyped-call]  # the trusted fixture
        if file.stem.endswith("_OPTIM"):
            adam = gi._adam_state(file)
            group = getattr(module.param_groups, "param_groups/0")
            keys = [getattr(group, f"params/{i}") for i in range(int(getattr(group, "params/size")))]
            assert adam is not None and len(adam) == len(keys), file
            for moment, key in zip(adam, keys, strict=True):
                state = getattr(module.state, key)
                assert float(moment["step"]) == float(state.step)
                assert torch.equal(moment["exp_avg"], state.exp_avg)
                assert torch.equal(moment["exp_avg_sq"], state.exp_avg_sq)
            continue
        theirs = module.state_dict()
        ours = gi._load_weights(file)
        assert set(ours) == set(theirs), file
        assert all(torch.equal(ours[key], theirs[key]) for key in theirs), file


def test_unreadable_files_are_refused_in_plain_words(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken save or project stops the import with a reason, not a crash."""
    from nitro.cli import main

    save = tmp_path / "save"
    shutil.copytree(SAVES / "default_plain", save)
    stats = save / "checkpoints" / "123456" / "RUNNING_STATS.json"
    stats.write_text("{", encoding="utf-8")
    with pytest.raises(gi.GigaLearnImportError, match="RUNNING_STATS"):
        gi.read_save(save)
    stats.write_text("{}", encoding="utf-8")
    (save / "checkpoints" / "123456" / "CRITIC.lt").write_bytes(b"not an archive")
    with pytest.raises(gi.GigaLearnImportError, match="CRITIC"):
        gi.read_save(save)
    args = ["import-gigalearn", str(save), "--into", "x", "--runs", str(tmp_path / "runs")]
    assert main([*args, "--activation", "relu"]) == 1
    assert "Cannot import" in capsys.readouterr().err

    project = tmp_path / "project"
    shutil.copytree(PROJECT, project)
    main_cpp = project / "src" / "BotMain.cpp"
    text = main_cpp.read_text(encoding="utf-8")
    main_cpp.write_text(text.replace("int tsPerItr", "int unused = 4 / 0;\n\tint tsPerItr"), encoding="utf-8")
    assert gigalearn_source.read(project).settings["ppo.tsPerItr"] == 100_000, "1 / 0 is just not read"
    main_cpp.write_text(text + "\nstd::vector<WeightedReward> unfinished = { { new AirReward(), 1 },\n")
    with pytest.raises(gi.GigaLearnImportError, match=r"BotMain\.cpp"):
        gi.plan_import(SAVES / "advanced_shared", project)


def test_a_header_the_reader_cannot_follow_does_not_stop_the_import(tmp_path: Path) -> None:
    """Preprocessor branches (two openings, one end) in some other header of the project."""
    project = tmp_path / "project"
    shutil.copytree(PROJECT, project)
    (project / "src" / "Branches.h").write_text(
        "#ifdef A\nclass Twice : public Reward {\n#else\nclass Twice : public Reward {\n#endif\n};\n",
        encoding="utf-8",
    )
    source = gigalearn_source.read(project)
    assert source.classes["Twice"].startswith("src/Branches.h:") and "MyAerialReward" in source.classes


def test_a_name_that_is_not_one_folder_is_refused_first(tmp_path: Path) -> None:
    for name in ("../out", "a/b", ""):
        with pytest.raises(gi.GigaLearnImportError, match="name"):
            gi.import_bot(SAVES / "advanced_shared", PROJECT, name, tmp_path / "runs")
    assert not (tmp_path / "out").exists()


def test_without_a_shared_head_its_activation_setting_does_not_count() -> None:
    """A project that sets the policy's and critic's activation only, its save without a shared head."""
    leaky = {f"ppo.{n}.activationType": "LEAKY_RELU" for n in ("policy", "critic")}
    plan = gi.build_plan(gi.read_save(SAVES / "default_plain"), plain_source("DefaultObs", **leaky))
    assert not plan.refused and plan.activation == "leaky_relu", plan.refused
    shared = gi.build_plan(gi.read_save(SAVES / "advanced_shared"), plain_source("AdvancedObs", **leaky))
    assert any("different activations" in line for line in shared.refused), "its head is RELU, by default"


def test_a_wrong_critic_fails_the_parity_check() -> None:
    """The value is held to GigaLearn's critic as the policy is to its policy."""
    loaded = gi.read_save(SAVES / "advanced_shared")
    plan = gi.build_plan(loaded, gigalearn_source.read(PROJECT))
    model = gi.build_model(config_of(plan), plan, loaded)
    generator = torch.Generator().manual_seed(3)
    obs, mask = torch.randn(64, 109, generator=generator), torch.ones(64, 90, dtype=torch.bool)
    assert gi.compare(model, loaded, plan, obs, mask).passed
    last = model.value[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.bias.add_(0.05)
    parity = gi.compare(model, loaded, plan, obs, mask)
    assert not parity.passed and "FAILED" in parity.line()
