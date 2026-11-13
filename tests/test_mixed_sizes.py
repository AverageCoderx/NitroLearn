"""Several game sizes in one run (env.team_sizes) and shuffled car places (env.shuffle_slots)."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import EnvPool, core_settings, probe_config
from nitro.export import _readme
from nitro.model import play_settings
from nitro.runs import RunFolder
from nitro.selfplay import Block, group_rows
from nitro.training import SIZE_COLUMNS, TrainingRun, size_column_names, sizes_note

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

MIXED = "team_size = 3\nteam_sizes = [1, 2, 3]\nobs_team_size = 3\nnum_arenas = 12\nthreads = 3\n"
TRAINING = """
[run]
seed = 5
device = "cpu"
[env]
{env}
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 960
minibatches = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
{extra}
"""


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())


def issues(text: str) -> dict[str, str]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return {issue.path: issue.message for issue in info.value.issues}


TWO = "[env]\nteam_size = 2\nteam_sizes = [1, 2]\nobs_team_size = 2\n"  # a mixed run that is fine
THREE = "[env]\nteam_size = 3\nteam_sizes = [1, 3]\nobs_team_size = 3\n"


@pytest.mark.parametrize(
    ("text", "path", "message"),
    [
        ("[env]\nteam_sizes = [1, 2, 3]\nobs_team_size = 3", "env.team_size", "must be 3, the largest"),
        ("[env]\nteam_size = 2\nteam_sizes = [2]\nobs_team_size = 2", "env.team_sizes", "two or more"),
        ("[env]\nteam_size = 2\nteam_sizes = [2, 2, 1]\nobs_team_size = 2", "env.team_sizes", "twice"),
        ("[env]\nteam_size = 3\nteam_sizes = [1, 3]", "env.team_sizes", "set env.obs_team_size = 3"),
        (THREE + "team_size_shares = [1.0]", "env.team_size_shares", "one share for each"),
        (THREE + "team_size_shares = [1.0, 0.0]", "env.team_size_shares", "above 0"),
        (THREE.replace("[1, 3]", "[1, 2, 3]") + "num_arenas = 4", "env.num_arenas", "at least 6"),
        (TWO + "opponents = false", "env.team_sizes", "needs opponents"),
        (TWO + "[[state_setters]]\nname = 'replay'\nbundled = true", "state_setters", "replay start states"),
        ("[env]\nshuffle_slots = true", "env.shuffle_slots", "padded observation"),
    ],
)
def test_settings_that_cannot_work_are_refused_with_what_to_change(
    text: str, path: str, message: str
) -> None:
    found = issues(text)
    assert message in found.get(path, ""), found


def test_an_attention_encoder_needs_no_padding_and_the_setting_is_off_by_default() -> None:
    default = parse("")
    assert default.env.team_sizes == () and not default.env.shuffle_slots
    assert default.env.sizes_label == "1v1" and not default.env.mixed_sizes
    text = '[env]\nteam_size = 3\nteam_sizes = [1, 3]\n[model]\nencoder = "transformer"'
    config = parse(text)
    assert config.env.sizes_label == "1v1 and 3v3"
    settings = json.loads(core_settings(config))
    assert (settings["team_sizes"], settings["obs_team_size"]) == ([1, 3], 3), (
        "the encoder's room: the largest"
    )
    off = json.loads(core_settings(default))
    assert not {"team_sizes", "team_size_shares", "shuffle_slots"} & set(off), (
        "off: the same settings as before"
    )
    # A look at the bot (watch, eval) plays one game size: the largest, or the one asked for.
    assert probe_config(config).env.team_sizes == ()
    two = play_settings(config, 2)
    assert (two.env.team_size, two.env.team_sizes) == (2, ())


def test_the_core_refuses_a_mixed_run_without_room_before_loading_anything() -> None:
    settings = json.loads(core_settings(parse(f"[env]\n{MIXED}")))
    settings["obs_team_size"] = 0
    with pytest.raises(ValueError, match="room for the largest size"):
        _core.EnvPool(json.dumps(settings))
    settings["obs_team_size"] = 3
    settings["team_size_shares"] = [100.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="gets no arena"):
        _core.EnvPool(json.dumps(settings))


def test_self_play_rows_follow_each_arenas_car_count() -> None:
    cars = [2, 2, 4, 4, 6, 6, 2, 2, 4, 4, 6, 6]  # arena i in group i % 2 at place i // 2
    assert group_rows(Block(0, 6, 12), 2, 0, cars) == slice(12, 24)  # arenas 6, 8, 10
    assert group_rows(Block(0, 8, 10), 2, 1, cars) == slice(14, 18)  # arena 9
    assert group_rows(Block(0, 6, 8), 2, 0, 2) == group_rows(Block(0, 6, 8), 2, 0, [2] * 12)


def test_the_log_names_each_sizes_share_of_the_player_steps() -> None:
    note = sizes_note([1, 1, 2, 2, 3, 3], 2)
    assert "1v1 in 2 arenas (17% of the player-steps)" in note and "3v3 in 2 arenas (50%" in note
    config = parse(f"[env]\n{MIXED}")
    readme = _readme("bot", "run", 10, config, built=True)
    assert "1v1, 2v2 and 3v3 side by side" in readme


@NEEDS_MESHES
def test_a_mixed_pool_builds_each_arena_at_its_size() -> None:
    config = parse(f"[env]\n{MIXED}")
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    envs = EnvPool(config, pin_memory=False)
    assert envs.arena_team_sizes == [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]
    assert [tensors.reward.numel() for tensors in envs.groups] == [24, 24]
    for _ in range(20):
        for group in range(envs.num_groups):
            envs.step(group)
            envs.wait(group)
    stats = envs.take_stats()
    assert sorted(stats.play_by_size) == [1, 2, 3]
    assert [stats.play_by_size[n]["car_steps"] for n in (1, 2, 3)] == [4 * 2 * 20, 4 * 4 * 20, 4 * 6 * 20]


@NEEDS_MESHES
def test_the_cores_rows_follow_the_groups_arenas_in_turn() -> None:
    """Every row's episode id carries its arena in the high 32 bits (the core's `episode_id`): the groups'
    rows follow one another, and group g holds arenas g, g + groups, ... in turn, each car's row in it."""
    config = parse(f"[env]\n{MIXED}")
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    envs = EnvPool(config, pin_memory=False)
    episodes = torch.cat([tensors.episode.flatten() for tensors in envs.groups])
    cars, groups = [2 * size for size in envs.arena_team_sizes], envs.num_groups
    expected = [
        arena
        for group in range(groups)
        for arena in range(group, len(cars), groups)
        for _ in range(cars[arena])
    ]
    assert (episodes >> 32).tolist() == expected


def _run(
    tmp_path: Path, env: str, extra: str = "", name: str = "mixed", ppo: str = ""
) -> tuple[TrainingRun, RunFolder]:
    text = TRAINING.format(env=env, extra=extra).replace("minibatches = 2\n", f"minibatches = 2\n{ppo}", 1)
    file = tmp_path / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, name, file)
    return TrainingRun(settings, folder), folder


@NEEDS_MESHES
def test_a_short_mixed_run_with_self_play_trains_saves_and_reports_each_size(tmp_path: Path) -> None:
    selfplay = (
        "[selfplay]\nenabled = true\npast_share = 0.5\nactive_opponents = 1\nsnapshot_steps = 1000000\n"
    )
    run, folder = _run(tmp_path, MIXED + "shuffle_slots = true", selfplay)
    run.selfplay.next_snapshot_steps = 1  # a past version joins at the first iteration
    rows = [run.iteration()]
    run.checkpoint()  # rotates it in: it drives Orange in the last half of the arenas, every size
    assert run.selfplay.active[0] is not None
    learner = run.selfplay.learner_rows
    assert learner is not None
    # Group 0 holds arenas 0, 2, ..., 10 (sizes 1, 2, 3, 1, 2, 3): the past half is arenas 6, 8, 10, rows
    # 12 to 24, whose Orange (odd) rows the past version drives; group 1 alike from row 24.
    expected = torch.ones(48, dtype=torch.bool)
    for start in (0, 24):
        expected[start + 13 : start + 24 : 2] = False
    assert torch.equal(learner, expected)
    rows += [run.iteration() for _ in range(3)]
    run.metrics.close()
    with folder.metrics.open(encoding="utf-8") as metrics:
        header = next(csv.reader(metrics))
    assert set(size_column_names((1, 2, 3))) <= set(header)
    assert len(size_column_names((1, 2, 3))) == 3 * len(SIZE_COLUMNS)
    for row in rows:
        assert math.isfinite(float(row["entropy"])) and math.isfinite(float(row["value_loss"]))
        shares = [float(row[f"{n}v{n}_car_step_share"]) for n in (1, 2, 3)]
        assert sum(shares) == pytest.approx(1.0)
    # With a past version on Orange in half the arenas the counted share is no longer exactly 1:2:3; the
    # first iteration (every car counted) is.
    assert [float(rows[0][f"{n}v{n}_car_step_share"]) for n in (1, 2, 3)] == pytest.approx(
        [1 / 6, 2 / 6, 3 / 6]
    )
    episodes = [
        float(r[f"{n}v{n}_episode_seconds"])
        for r in rows
        for n in (1, 2, 3)
        if r[f"{n}v{n}_episode_seconds"] != ""
    ]
    assert episodes and all(0 < s < 60 for s in episodes)
    assert all(isinstance(r[f"{n}v{n}_touches_per_minute"], float) for r in rows for n in (1, 2, 3))
    # A resume from the save carries on with the same sizes.
    resumed = TrainingRun(run.config, folder, folder.load_latest())
    assert resumed.envs.arena_team_sizes == run.envs.arena_team_sizes
    resumed.iteration()
    resumed.metrics.close()


@NEEDS_MESHES
def test_a_1v1_run_resumes_with_every_size_added(tmp_path: Path) -> None:
    """The recommended way: 1v1 first with room for 3, then 1v1 to 3v3 on a resume."""
    run, folder = _run(tmp_path, "obs_team_size = 3\nnum_arenas = 12\nthreads = 3", name="growing")
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    width = run.envs.groups[0].obs.shape[1]
    text = folder.settings.read_text(encoding="utf-8").replace(
        "[env]\n", "[env]\nteam_size = 3\nteam_sizes = [1, 2, 3]\nteam_size_shares = [2, 1, 1]\n", 1
    )
    folder.settings.write_text(text, encoding="utf-8")
    state = folder.load_latest()
    bigger = folder.load_settings(state)
    bigger = dataclasses.replace(bigger, paths=dataclasses.replace(bigger.paths, collision_meshes=MESHES))
    resumed = TrainingRun(bigger, folder, state)
    assert resumed.envs.groups[0].obs.shape[1] == width
    assert sorted(set(resumed.envs.arena_team_sizes)) == [1, 2, 3]
    assert resumed.envs.arena_team_sizes.count(1) == 6, "half the arenas stay 1v1"
    row = resumed.iteration()
    assert math.isfinite(float(row["entropy"]))
    resumed.metrics.close()
