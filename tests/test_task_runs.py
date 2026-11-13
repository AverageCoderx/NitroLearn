"""The practice-task runner and its analysis (benchmarks/tasks/run.py, analyse.py) with made-up rows: a
crashed run is run again on --resume, and the analysis says it crashed once. Evaluating part way leaves the
training exactly as it was."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from nitro import tasks
from nitro.config import NitroConfig
from nitro.tasks import EVAL_SEEDS, TASKS

BENCHMARKS = Path(__file__).parents[1] / "benchmarks" / "tasks"
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


def script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tasks_{name}", BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["task", "precision", "seed", "stage", "eval_seed", "time_to_touch", "learned", "problems"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def results(crash_first: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if crash_first:
        rows.append({"task": "ball_touch", "precision": "fp16", "seed": 3, "stage": "crashed"})
    for precision in ("fp32", "fp16"):
        for seed in range(10):
            base = {"task": "ball_touch", "precision": precision, "seed": seed, "learned": True}
            rows.append(base | {"stage": "untrained", "eval_seed": EVAL_SEEDS[0], "time_to_touch": 9.0})
            for eval_seed in EVAL_SEEDS:
                rows.append(
                    base | {"stage": "trained", "eval_seed": eval_seed, "time_to_touch": 2.0 + seed / 100}
                )
    return rows


def test_resume_runs_a_crashed_run_again_and_keeps_its_crash_row(tmp_path: Path) -> None:
    finished = script("run").finished
    out = tmp_path / "runs.csv"
    assert finished(out) == (set(), False)
    write(out, [])
    assert finished(out) == (set(), False), "only a header: start the file afresh"
    write(out, results(crash_first=True)[:1])
    assert finished(out) == (set(), True), "the crashed run is not done, but its row is kept"
    write(out, results(crash_first=True))
    done, kept = finished(out)
    assert kept and len(done) == 20 and ("fp16", 3) in done


def test_the_analysis_names_a_run_that_crashed_and_was_run_again(tmp_path: Path) -> None:
    analyse = script("analyse").analyse
    out = tmp_path / "runs.csv"
    write(out, results(crash_first=False))
    assert "crashed runs: none; runs failing the learning check: none" in analyse(out)
    write(out, results(crash_first=True))
    lines = analyse(out)
    assert (
        "crashed runs: none (crashed, then run again: ['fp16 seed 3']); runs failing the learning check: none"
        in lines
    )
    write(
        out,
        results(crash_first=True)[:1]
        + [r for r in results(False) if r["seed"] != 3 or r["precision"] != "fp16"],
    )
    assert any(line.startswith("crashed runs: ['fp16 seed 3']; ") for line in analyse(out))


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_evaluating_part_way_leaves_the_training_bit_for_bit_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`nitro ab --checkpoints` evaluates after the first iteration; the trained policy and its
    results are exactly those of the same run without it."""
    built: list[Any] = []
    build = tasks.build

    def building(config: NitroConfig) -> Any:
        built.append(build(config))
        return built[-1]

    monkeypatch.setattr(tasks, "build", building)
    task = TASKS["ball_touch"]
    config = task.settings()
    config = dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES.resolve()),
        run=dataclasses.replace(config.run, device="cpu", seed=3),
        env=dataclasses.replace(config.env, num_arenas=8, threads=2),
        model=dataclasses.replace(config.model, policy=(32,), value=(32,)),
        ppo=dataclasses.replace(config.ppo, steps_per_iteration=320, minibatches=2),
    )
    plain = tasks.practise(task, config, 3 * 320)
    checked = tasks.practise(task, config, 3 * 320, checkpoints=[320])
    assert [r["stage"] for r in checked] == ["untrained", "checkpoint", "trained", "trained"]
    assert checked[1]["agent_steps"] == 320, "evaluated after learning the first batch, which it records"

    def results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in r.items() if k != "train_minutes"} for r in rows if r["stage"] != "checkpoint"
        ]

    assert results(checked) == results(plain)
    # With the likeliest-action row as well: the same curve point, and that row as without the curve.
    likeliest = tasks.practise(task, config, 3 * 320, also_likeliest=True)
    both = tasks.practise(task, config, 3 * 320, checkpoints=[320], also_likeliest=True)
    assert [r["stage"] for r in both] == [*(r["stage"] for r in checked), "trained_likeliest"]
    assert results(both) == results(likeliest) == [*results(plain), results(likeliest)[-1]]
    assert results([{**both[1], "stage": "point"}]) == results([{**checked[1], "stage": "point"}])
    weights = [dict(setup.model.state_dict()) for setup in built]
    assert all(other.keys() == weights[0].keys() for other in weights)
    assert all(torch.equal(weights[0][name], other[name]) for other in weights[1:] for name in weights[0]), (
        "every parameter the same"
    )
