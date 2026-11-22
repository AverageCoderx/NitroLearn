"""The drills against scripted players and the weakness map : the measures without a simulator, then
known answers with the collision meshes: an idle player scores at the bottom, the reference script clearly
above it, the bars come back from `reference_score`, and the same seed gives the same scores."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nitro import weakness
from nitro.config.schema import PPOConfig
from nitro.config.state_setters import FiftyFifty, WallRead
from nitro.tasks import EVAL_SEEDS, TASKS, _scripted_orange, practise, reference_score
from nitro.training import build

ROOT = Path(__file__).parents[1]
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
DRILLS = ("kickoff_duel", "fifty_fifty", "wall_read")


def record(**fields: object) -> Any:
    base = {
        "env": 0,
        "car": 0,
        "seconds": 5.0,
        "terminal": False,
        "first_touch": math.nan,
        "landed": math.nan,
        "boost_collected": 0.0,
        "ball_y_after_touch": math.nan,
        "goal": 0,
        "setter": 0,
        "start_ball_x": 0.0,
        "start_ball_y": 0.0,
        "start_ball_z": 93.0,
        "start_ball_speed": 0.0,
        "start_car_speed": 0.0,
        "start_car_z": 17.0,
        "start_facing": 1.0,
        "start_distance": 2000.0,
    }
    return SimpleNamespace(**(base | fields))


def test_a_kickoff_touched_on_the_same_tick_counts_half() -> None:
    pairs = [
        (record(env=0, first_touch=1.0, ball_y_after_touch=800.0), record(env=0, car=1, first_touch=1.5)),
        (record(env=1, first_touch=1.2), record(env=1, car=1, first_touch=1.2)),
        (record(env=2), record(env=2, car=1, first_touch=1.0)),
    ]
    measured = TASKS["kickoff_duel"].measure([r for pair in pairs for r in pair])
    assert measured["first_touch_share"] == pytest.approx(0.5)  # 1 + 0.5 + 0 of 3
    assert measured["time_to_first_touch"] == pytest.approx((1.0 + 1.2 + 6.0) / 3)
    assert measured["ball_y_after_touch"] == 800.0


def test_a_fifty_fifty_is_won_by_moving_the_ball_500_toward_their_goal() -> None:
    def pair(env: int, start: float, after: float) -> list[Any]:
        touch = math.nan if math.isnan(after) else 1.0
        return [
            record(env=env, start_ball_y=start, ball_y_after_touch=after, first_touch=touch),
            record(env=env, car=1, first_touch=2.0),
        ]

    records = pair(0, -1000.0, 0.0) + pair(1, 0.0, -600.0) + pair(2, 0.0, 300.0) + pair(3, 0.0, math.nan)
    measured = TASKS["fifty_fifty"].measure(records)
    assert measured["win_share"] == pytest.approx((1 + 0 + 0.5 + 0.5) / 4)
    assert measured["first_touch_share"] == pytest.approx(3 / 4)
    assert measured["ball_progress"] == pytest.approx((1000 - 600 + 300) / 3)


def test_the_new_starts_check_their_ranges() -> None:
    assert FiftyFifty().problems() == [] and WallRead().problems() == []
    assert FiftyFifty(min_distance_uu=2000.0, max_distance_uu=1000.0).problems()
    assert WallRead(min_speed=2000.0, max_speed=1000.0).problems()
    config = TASKS["fifty_fifty"].settings()
    lone = dataclasses.replace(config, env=dataclasses.replace(config.env, opponents=False))
    assert any("needs an opponent" in why for _, why in lone.problems())


def test_the_weakness_map_finds_the_group_that_does_worst() -> None:
    task = TASKS["ball_touch"]
    # Facing the ball: touched in 2 s; facing away: missed (10 s). Only 5 situations start in the air.
    situations = [[record(env=i, first_touch=2.0, start_facing=1.0)] for i in range(60)]
    situations += [[record(env=60 + i, start_facing=-1.0)] for i in range(30)]
    situations += [[record(env=90 + i, first_touch=2.0, start_ball_z=400.0)] for i in range(5)]
    entry = weakness.analyse(task, situations)
    assert entry["situations"] == 95 and entry["cutoff"] == 10.0
    assert entry["worst_share"] == pytest.approx(30 / 95, abs=1e-3), "ties at the cut-off all count"
    worst = entry["groups"][0]
    assert (worst["feature"], worst["group"], worst["count"]) == (
        "car heading",
        "car facing away from the ball",
        30,
    )
    assert worst["value"] == 10.0 and worst["worst_share"] == 1.0 and not worst["few"]
    air = next(g for g in entry["groups"] if g["group"] == "ball in the air")
    assert air["few"] and air["count"] == 5
    assert {g["feature"] for g in entry["groups"]} == {"car heading", "ball height"}, (
        "one-group features left out"
    )
    lines = weakness.describe(entry)
    assert "car facing away from the ball: 10s in 30 situations, 100% of them among the worst" in lines[1]
    json.dumps(entry)  # no NaN: the dashboard reads it


def test_the_weakness_report_keeps_one_map_per_save(tmp_path: Path) -> None:
    path = tmp_path / weakness.REPORT
    weakness.append_report(path, 200, [{"task": "a"}], 1, 128)
    weakness.append_report(path, 100, [{"task": "b"}], 1, 128)
    weakness.append_report(path, 200, [{"task": "c"}], 1, 128)
    maps = weakness.read_report(path)
    assert [(m["agent_steps"], m["tasks"][0]["task"]) for m in maps] == [(100, "b"), (200, "c")]
    assert weakness.read_report(tmp_path / "none.json") == []


def test_a_weakness_report_cut_off_while_written_keeps_the_earlier_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training stopped while the report is written (it is written after every skill check) keeps its
    history; a list holding something other than maps is read without failing."""
    path = tmp_path / weakness.REPORT
    weakness.append_report(path, 100, [{"task": "a"}], 1, 128)
    before = path.read_text(encoding="utf-8")
    write = Path.write_text

    def cut_off(self: Path, data: str, *args: object, **kwargs: object) -> int:
        write(self, data[: len(data) // 2], encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", cut_off)
    with pytest.raises(KeyboardInterrupt):
        weakness.append_report(path, 200, [{"task": "b"}], 1, 128)
    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == before
    path.write_text(before.replace("[", "[3, ", 1), encoding="utf-8")
    weakness.append_report(path, 200, [{"task": "b"}], 1, 128)
    assert [m["agent_steps"] for m in weakness.read_report(path)] == [100, 200]


@needs_meshes
@pytest.mark.parametrize("name", DRILLS)
def test_idle_scores_at_the_bottom_and_the_reference_script_clearly_above(name: str) -> None:
    task = TASKS[name]
    idle = reference_score(task, MESHES, games=64, script="idle")[task.metric]
    good = reference_score(task, MESHES, games=64)[task.metric]
    assert task.better(task.bar, idle) > 0, f"idle {idle} must be below the bar {task.bar}"
    # At least 5 of the task's smallest differences: a fair 50/50 split (0.5) is only 10 of them above idle.
    assert task.better(good, idle) >= 5 * task.smallest_difference, (good, idle)
    assert reference_score(task, MESHES, games=64) == reference_score(task, MESHES, games=64)


@needs_meshes
@pytest.mark.parametrize("name", DRILLS)
def test_the_bars_come_back_from_the_reference_scripts(name: str) -> None:
    """How the bars were made (512 situations, the first evaluation seed); re-run to remake them."""
    task = TASKS[name]
    measured = reference_score(task, MESHES, games=512, seed=EVAL_SEEDS[0])[task.metric]
    assert measured == pytest.approx(task.bar, abs=0.05 if task.unit == "" else 0.1), measured


@needs_meshes
def test_recovery_known_answers() -> None:
    """The recovery drill (tumbling starts at random turns): the chaser's levelling out lands well before an
    idle car, which mostly falls onto its wheels or not at all."""
    task = TASKS["recovery"]
    chaser = reference_score(task, MESHES, games=64, script="chaser")["time_to_land"]
    idle = reference_score(task, MESHES, games=64, script="idle")["time_to_land"]
    assert chaser < task.bar < idle


@needs_meshes
def test_a_drill_against_a_script_trains_on_the_bots_cars_only() -> None:
    task = TASKS["fifty_fifty"]
    base = task.settings()
    arenas = 4
    config = dataclasses.replace(
        base,
        paths=dataclasses.replace(base.paths, collision_meshes=MESHES),
        run=dataclasses.replace(base.run, device="cpu"),
        env=dataclasses.replace(base.env, num_arenas=arenas, threads=2),
        ppo=PPOConfig(steps_per_iteration=arenas * 2 * 16, minibatches=2),
    )
    setup = build(config)
    _scripted_orange(setup, config, "chaser", torch.device("cpu"))
    learner = setup.trainer.learner_rows
    assert learner is not None and learner.tolist() == [True, False] * arenas, "Blue learns, the script not"
    rows = practise(task, config, steps=1)
    assert [r["stage"] for r in rows] == ["untrained", "trained", "trained"]
    assert all(r["episodes"] == 2 * arenas and 0 <= r["win_share"] <= 1 for r in rows)
