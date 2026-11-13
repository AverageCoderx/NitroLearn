"""The A/B tool: verdicts on made-up seed results with and without a planted difference; a tiny run."""

from __future__ import annotations

import math
import os
import random
import shutil
import statistics
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from nitro.ab import (
    Comparison,
    Experiment,
    compare_on_task,
    compare_speed,
    curve_marks,
    differences,
    report,
    run_problems,
)
from nitro.config import NitroConfig, load_config
from nitro.tasks import EVAL_SEEDS, TASKS, Task

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


def seeds(values: list[float]) -> dict[int, float]:
    return dict(enumerate(values))


def test_clearly_separated_results_get_a_verdict_in_the_right_direction() -> None:
    fast, slow = seeds([2.0, 2.1, 2.05, 1.95, 2.0, 2.1]), seeds([2.6, 2.7, 2.65, 2.55, 2.6, 2.7])
    assert Comparison(slow, fast, 0.1, higher_is_better=False).verdict() == "B is better"
    assert Comparison(fast, slow, 0.1, higher_is_better=False).verdict() == "B is worse"
    assert Comparison(fast, slow, 0.1, higher_is_better=True).verdict() == "B is better"


def test_a_difference_smaller_than_what_matters_is_no_measurable_difference() -> None:
    a, b = seeds([2.0, 2.01, 2.0, 2.01]), seeds([2.05, 2.06, 2.05, 2.06])
    assert Comparison(a, b, 0.1, higher_is_better=True).verdict() == "no measurable difference"


def test_overlapping_seeds_with_a_large_mean_difference_are_borderline() -> None:
    a = seeds([1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 3.0])  # one wild seed stretches A's range
    b = seeds([1.7] * 10)
    assert Comparison(a, b, 0.1, higher_is_better=True).verdict() == "borderline"


def test_the_rule_rarely_calls_a_winner_between_identical_setups_and_finds_a_planted_one() -> None:
    rng = random.Random(0)

    def draw(shift: float) -> dict[int, float]:
        return seeds([rng.gauss(shift, 1.0) for _ in range(10)])

    trials = 2000
    false_alarms = sum(
        Comparison(draw(0), draw(0), 0.5, higher_is_better=True).verdict() in ("B is better", "B is worse")
        for _ in range(trials)
    )
    found = sum(Comparison(draw(0), draw(3), 0.5, True).verdict() == "B is better" for _ in range(trials))
    assert false_alarms / trials < 0.01
    assert found / trials > 0.8


def test_the_detectable_difference_is_the_one_the_rule_really_finds_8_times_in_10() -> None:
    rng = random.Random(1)
    a, b = seeds([rng.gauss(0, 1) for _ in range(10)]), seeds([rng.gauss(0, 1) for _ in range(10)])
    comparison = Comparison(a, b, 0.5, higher_is_better=True)
    detectable = comparison.detectable()
    pooled = (comparison.a and comparison.b) and 1.0  # the samples' pooled sd is about 1
    assert pooled and 1.8 < detectable < 3.2, f"{detectable} (a t-test figure would be about 1.2)"
    # Planting that difference, the rule finds it at least 7 times in 10 and misses half the size often.
    hits = sum(
        Comparison(
            seeds([rng.gauss(0, 1) for _ in range(10)]),
            seeds([rng.gauss(detectable, 1) for _ in range(10)]),
            0.5,
            True,
        ).verdict()
        == "B is better"
        for _ in range(400)
    )
    small = sum(
        Comparison(
            seeds([rng.gauss(0, 1) for _ in range(10)]),
            seeds([rng.gauss(detectable / 2, 1) for _ in range(10)]),
            0.5,
            True,
        ).verdict()
        == "B is better"
        for _ in range(400)
    )
    assert hits / 400 > 0.7 and small / 400 < 0.5, f"{hits} {small}"


def test_a_comparison_refuses_to_continue_with_another_plan(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("[run]\nseed = 1\n", encoding="utf-8")
    (tmp_path / "b.toml").write_text("[run]\nseed = 2\n", encoding="utf-8")
    experiment = Experiment(tmp_path / "ab", {"A": tmp_path / "a.toml", "B": tmp_path / "b.toml"})
    experiment.check_plan({"task": "ball_touch", "steps": 10_000_000, "seeds": 10})
    experiment.check_plan({"task": "ball_touch", "steps": 10_000_000, "seeds": 10})
    with pytest.raises(ValueError, match="steps: 10000000 -> 5000000"):
        experiment.check_plan({"task": "ball_touch", "steps": 5_000_000, "seeds": 10})
    # An option saved in the plan and left off now is a change too, e.g. --with-rewards.
    rewards = Experiment(tmp_path / "rewards", {"A": tmp_path / "a.toml", "B": tmp_path / "b.toml"})
    rewards.check_plan({"task": "shot", "rewards": "from the settings files"})
    with pytest.raises(ValueError, match="rewards: from the settings files -> not set"):
        rewards.check_plan({"task": "shot"})


def test_a_comparison_carries_on_across_commits_that_leave_the_code_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nitro.ab as ab

    (tmp_path / "a.toml").write_text("[run]\nseed = 1\n", encoding="utf-8")
    experiment = Experiment(tmp_path / "ab", {"A": tmp_path / "a.toml"})
    monkeypatch.setattr(ab, "_commit", lambda: "aaaaaaa")
    monkeypatch.setattr(ab, "_code", lambda: "code-1")
    experiment.check_plan({"seeds": 10})
    monkeypatch.setattr(ab, "_commit", lambda: "bbbbbbb")  # a results-only commit came in
    experiment.check_plan({"seeds": 10})
    monkeypatch.setattr(ab, "_code", lambda: "code-2")
    with pytest.raises(ValueError, match="code: code-1 -> code-2"):
        experiment.check_plan({"seeds": 10})


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_fingerprint_follows_the_code_runs_depend_on_and_nothing_else(tmp_path: Path) -> None:
    """A commit to the tools, the presets or a benchmark leaves a comparison able to carry
    on; a commit to the package, the core or a learning script stops it."""
    import nitro.ab as ab

    def commit(path: str, text: str) -> str:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        for command in (["add", "-A"], ["commit", "-q", "-m", path]):
            subprocess.run(["git", *command], cwd=tmp_path, check=True, capture_output=True)
        return ab._code(tmp_path)

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for key, value in (("user.email", "test@example.com"), ("user.name", "test")):
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    first = commit("python/nitro/core.py", "x = 1\n")
    assert first != "unknown"
    for path in ("tools/check.sh", "configs/presets/1v1.toml", "benchmarks/tasks/b.toml", "README.md",
                 "benchmarks/results/ab/runs.csv", "Shortcuts/shortcuts.ps1", "tests/test_x.py"):  # fmt: skip
        assert commit(path, "changed\n") == first, path
    assert commit("crates/nitro_core/src/x.rs", "y = 2\n") != first
    second = ab._code(tmp_path)
    assert commit("python/nitro/core.py", "x = 2\n") != second


def test_the_report_lists_every_seed_and_the_numbers_that_help_read_it() -> None:
    comparison = Comparison(seeds([2.0, 2.2, 2.1, 2.3]), seeds([1.5, 1.6, 1.4, 1.55]), 0.1, False)
    text = "\n".join(report(comparison, "time_to_touch", "s", ["seed 9 crashed"]))
    for value in ("2.3s", "1.4s", "95% interval", "would reliably show up", "Caution: seed 9 crashed"):
        assert value in text
    low, high = comparison.interval()
    assert 0 < low < high, "B is clearly faster, so the whole interval says B is better"


def test_differences_name_each_changed_setting_but_not_paths_or_seeds(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("[run]\nseed = 1\n[ppo]\npolicy_lr = 1e-4\n", encoding="utf-8")
    (tmp_path / "b.toml").write_text("[run]\nseed = 2\n[ppo]\npolicy_lr = 3e-4\n", encoding="utf-8")
    a, b = load_config(tmp_path / "a.toml"), load_config(tmp_path / "b.toml")
    assert differences(a, b) == ["ppo.policy_lr: 0.0001 -> 0.0003"]


def test_a_speed_run_plays_self_play_when_the_settings_have_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A speed run gives self-play a scratch pool, so its cost is measured; without self-play none."""
    from nitro import ab

    seen: list[Path | None] = []

    def fake(config: NitroConfig, pool_folder: Path | None) -> dict[str, object]:
        seen.append(pool_folder)
        assert pool_folder is None or pool_folder.is_dir()
        return {}

    monkeypatch.setattr(ab, "_speed_with", fake)
    (tmp_path / "s.toml").write_text("[selfplay]\nenabled = true\n", encoding="utf-8")
    ab._speed(load_config(tmp_path / "s.toml"))
    (tmp_path / "p.toml").write_text("[run]\nseed = 1\n", encoding="utf-8")
    ab._speed(load_config(tmp_path / "p.toml"))
    assert seen[0] is not None and seen[1] is None


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_speed_comparison_runs_saves_every_run_and_carries_on_where_it_stopped(tmp_path: Path) -> None:
    common = (
        f"[paths]\ncollision_meshes = '{MESHES.resolve()}'\n[run]\ndevice = 'cpu'\n"
        "[model]\npolicy = [32]\nvalue = [32]\n[ppo]\nsteps_per_iteration = 320\nminibatches = 2\n"
    )
    (tmp_path / "a.toml").write_text(common + "[env]\nnum_arenas = 8\nthreads = 2\n", encoding="utf-8")
    (tmp_path / "b.toml").write_text(common + "[env]\nnum_arenas = 16\nthreads = 2\n", encoding="utf-8")
    settings = {name: load_config(tmp_path / f"{name.lower()}.toml") for name in ("A", "B")}
    experiment = Experiment(tmp_path / "ab", {"A": tmp_path / "a.toml", "B": tmp_path / "b.toml"})
    lines = compare_speed(experiment, settings, repeats=2)
    rows = experiment.rows()
    assert [(r["setup"], r["key"]) for r in rows] == [("A", "0"), ("B", "0"), ("B", "1"), ("A", "1")]
    assert all(float(r["agent_sps"]) > 0 for r in rows)
    assert any(line.startswith("Verdict: ") for line in lines)
    assert compare_speed(experiment, settings, repeats=2) == lines, "nothing left to run: the same result"
    (tmp_path / "b.toml").write_text(common + "[env]\nnum_arenas = 32\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since this comparison started"):
        Experiment(tmp_path / "ab", {"A": tmp_path / "a.toml", "B": tmp_path / "b.toml"})


def test_a_crashed_run_is_run_again_and_the_report_says_so(tmp_path: Path) -> None:
    experiment = Experiment(tmp_path / "ab", {})
    calls: list[tuple[str, int]] = []

    def one(setup: str, key: int) -> list[dict[str, object]]:
        calls.append((setup, key))
        if len(calls) == 1:
            raise RuntimeError("unknown field `physics`")  # e.g. a stale core
        return [{"stage": "trained", "learned": True, "problems": ""}]

    jobs, fields = [("A", 0), ("B", 0)], ["stage", "learned", "problems"]
    experiment.run(jobs, "seed", fields, one)
    assert run_problems(experiment.rows(), "seed") == [
        "these runs crashed and are left out (see runs.csv): A seed 0"
    ]
    experiment.run(jobs, "seed", fields, one)
    assert calls == [("A", 0), ("B", 0), ("A", 0)], "only the crashed run is run again"
    assert [r["stage"] for r in experiment.rows()] == ["crashed", "trained", "trained"], "the crash stays"
    assert run_problems(experiment.rows(), "seed") == [
        "these runs crashed once and were run again (see runs.csv): A seed 0"
    ]
    experiment.run(jobs, "seed", fields, one)
    assert len(calls) == 3, "nothing left to run"


def test_replay_mix_is_said_to_be_left_out_and_alone_is_nothing_to_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """[replay_mix]'s copy steps run only in nitro train: nitro ab says so, not comparing them."""
    from nitro.ab import not_measured
    from nitro.cli import main

    preset = (Path(__file__).parents[1] / "configs/presets/1v1_starter.toml").read_text(encoding="utf-8")
    (tmp_path / "a.toml").write_text(preset, encoding="utf-8")
    (tmp_path / "b.toml").write_text(preset + "\n[replay_mix]\nenabled = true\n", encoding="utf-8")
    a, b = (load_config(tmp_path / f"{s}.toml") for s in "ab")
    assert not_measured([a, a], task=False) == []
    assert "cost does not show" in not_measured([a, b], task=False)[0]
    assert "only in nitro train" in not_measured([a, b], task=True)[0]
    for kind in (["--speed"], ["--task", "ball_touch"]):
        code = main(
            ["ab", str(tmp_path / "a.toml"), str(tmp_path / "b.toml"), *kind, "--folder", str(tmp_path)]
        )
        err = capsys.readouterr().err
        assert code == 1 and "Not used here: [replay_mix]" in err and "exactly the same way" in err, kind


# --- learning curves on a task: made-up runs instead of training ---------------------------------

STEPS = 1_000_000
# What `compare_on_task` prints and saves for the made-up runs below (default options).
GOLDEN_LINES = [
    "Practice task 'ball_touch', 1,000,000 agent steps per run; the untrained policy scored 9.12s on "
    "average.",
    "Measured: time_to_touch (lower is better).",
    "  A |                                 o   o   o   o   o| mean 4.12s, range 4s to 4.28s",
    "      every seed: 0: 4s, 1: 4.14s, 2: 4.28s, 3: 4.07s, 4: 4.21s, 5: 4s",
    "  B |o   o   o   o   o                                 | mean 3.52s, range 3.4s to 3.68s",
    "      every seed: 0: 3.4s, 1: 3.54s, 2: 3.68s, 3: 3.47s, 4: 3.61s, 5: 3.4s",
    "     3.4s                                           4.28s",
    "Verdict: B IS BETTER - B's results are clearly apart from A's, by more than the noise and the smallest "
    "difference that matters.",
    "  Smallest difference that matters: 0.1s. Noise floor (A vs A, B vs B), averaged over every "
    "half-and-half split of the seeds: 0.0747s, 0.0747s (the pre-registered first-half vs second-half split: "
    "0.0467s, 0.0467s).",
    "  To help read it (never decides): B better than A by 0.453s to 0.747s (95% interval); seed by seed, B "
    "better 6, about the same 0, A better 0.",
    "  With 6 seeds, only differences of about 0.248s or more would reliably show up (this rule calls them 8 "
    "times in 10); a smaller real difference can still read 'no measurable difference'.",
]
GOLDEN_PLAN = (
    '{\n "task": "ball_touch",\n "steps": 1000000,\n "seeds": 6,\n "eval_seeds": [\n  1000001,\n  1000002'
    '\n ],\n "commit": "abc1234",\n "code": "code-1"\n}'
)


def curve_value(setup: str, seed: int, at: int, eval_seed: int) -> float:
    """Made-up time to touch: falls with training, faster for B; a per-seed offset, an evaluation wobble."""
    gain = 5.0 if setup == "A" else 5.6
    wobble = 0.01 * (seed % 3) if eval_seed == EVAL_SEEDS[1] else 0.0
    return round(9.0 - gain * at / STEPS + 0.07 * (seed * 7 % 5) + wobble, 5)


def fake_practise(
    late: int = 0, missing: tuple[str, int] | None = None
) -> Callable[..., list[dict[str, Any]]]:
    """`nitro.tasks.practise` without training: B's points come `late` agent steps after the marks, and
    `missing` (setup, seed) lacks its second checkpoint row. Picking the likeliest action, B is worse:
    A's and B's trained values swapped."""

    def practise(
        task: Task,
        config: NitroConfig,
        steps: int,
        checkpoints: Sequence[int] = (),
        also_likeliest: bool = False,
    ) -> list[Any]:
        setup, seed = "A" if config.ppo.policy_lr == 1e-4 else "B", config.run.seed

        def row(stage: str, at: int, eval_seed: int) -> dict[str, Any]:
            at += late if setup == "B" and stage == "checkpoint" else 0
            value = curve_value(setup, seed, at, eval_seed)
            return {"stage": stage, "agent_steps": at, "eval_seed": eval_seed, "episodes": 512,
                    "time_to_touch": value, "touch_rate": 0.5}  # fmt: skip

        rows = [row("untrained", 0, EVAL_SEEDS[0])]
        rows += [
            row("checkpoint", mark, EVAL_SEEDS[0])
            for i, mark in enumerate(checkpoints)
            if (setup, seed) != missing or i != 1
        ]
        rows += [row("trained", steps, eval_seed) for eval_seed in EVAL_SEEDS]
        if also_likeliest:
            swapped = curve_value("B" if setup == "A" else "A", seed, steps, EVAL_SEEDS[0])
            rows.append(row("trained_likeliest", steps, EVAL_SEEDS[0]) | {"time_to_touch": swapped})
        return [r | {"train_minutes": 0.1, "learned": True, "problems": ""} for r in rows]

    return practise


def on_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    practise: Callable[..., list[dict[str, Any]]] | None = None,
    **options: Any,
) -> tuple[Experiment, list[str]]:
    import nitro.ab as ab

    monkeypatch.setattr(ab, "practise", practise or fake_practise())
    monkeypatch.setattr(ab, "_commit", lambda: "abc1234")
    monkeypatch.setattr(ab, "_code", lambda: "code-1")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a.toml").write_text("[ppo]\npolicy_lr = 1e-4\n", encoding="utf-8")
    (tmp_path / "b.toml").write_text("[ppo]\npolicy_lr = 3e-4\n", encoding="utf-8")
    settings = {s: load_config(tmp_path / f"{s.lower()}.toml") for s in "AB"}
    experiment = Experiment(tmp_path / "ab", {"A": tmp_path / "a.toml", "B": tmp_path / "b.toml"})
    lines = compare_on_task(experiment, TASKS["ball_touch"], settings, list(range(6)), STEPS, **options)
    return experiment, lines


def test_by_default_a_task_comparison_saves_and_says_exactly_this(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, lines = on_task(tmp_path, monkeypatch)
    assert lines == GOLDEN_LINES
    assert (experiment.folder / "plan.json").read_text(encoding="utf-8") == GOLDEN_PLAN
    assert {r["stage"] for r in experiment.rows()} == {"untrained", "trained"}


def test_a_comparison_without_a_curve_carries_on_by_default_and_refuses_other_curve_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / "plan.json").write_text(GOLDEN_PLAN, encoding="utf-8")  # started without a curve
    assert on_task(tmp_path, monkeypatch)[1] == GOLDEN_LINES
    with pytest.raises(ValueError, match=r"checkpoints: not set -> \[250000, 500000, 750000\]"):
        on_task(tmp_path, monkeypatch, checkpoints=4)
    # Started with a curve, it cannot carry on without it, nor with the area deciding afterwards.
    curved = tmp_path / "curved"
    on_task(curved, monkeypatch, checkpoints=4)
    with pytest.raises(ValueError, match=r"checkpoints: \[250000, 500000, 750000\] -> not set"):
        on_task(curved, monkeypatch)
    with pytest.raises(ValueError, match="decide: not set -> auc"):
        on_task(curved, monkeypatch, checkpoints=4, decide="auc", auc_smallest=0.05)
    deciding = tmp_path / "deciding"
    on_task(deciding, monkeypatch, checkpoints=4, decide="auc", auc_smallest=0.05)
    with pytest.raises(ValueError, match="decide: auc -> not set"):
        on_task(deciding, monkeypatch, checkpoints=4)


def test_the_marks_are_evenly_spaced_before_the_end_and_odd_counts_are_refused() -> None:
    assert curve_marks(10_000_000, 4) == [2_500_000, 5_000_000, 7_500_000]
    assert curve_marks(10_000_000, 0) == []
    assert curve_marks(10_000_000, 2) == [5_000_000]
    for count in (1, 11, -1):
        with pytest.raises(ValueError, match="0 \\(off\\) or 2 to 10"):
            curve_marks(10_000_000, count)
    with pytest.raises(ValueError, match="too few"):
        curve_marks(3, 4)


def test_the_area_is_each_seeds_mean_over_the_points_and_a_seed_missing_one_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, lines = on_task(tmp_path, monkeypatch, fake_practise(missing=("A", 2)), checkpoints=4)
    points = [250_000, 500_000, 750_000, STEPS]
    areas = {
        setup: {
            seed: statistics.mean(curve_value(setup, seed, at, EVAL_SEEDS[0]) for at in points)
            for seed in range(6)
            if (setup, seed) != ("A", 2)
        }
        for setup in "AB"
    }
    area = Comparison(areas["A"], areas["B"], 0.1, higher_is_better=False)
    shown = report(area, "time_to_touch, mean over the 4 points", "s", [])
    verdict = next(line for line in shown if line.startswith("Verdict: "))
    assert verdict.startswith("Verdict: B IS BETTER")
    start = lines.index("Area under the curve, 4 points - information only, never decides:")
    assert lines[start + 1 : start + 1 + len(shown)] == [
        "Information only, not the verdict: " + line[len("Verdict: ") :] if line == verdict else line
        for line in shown
    ]
    # The final point still decides, exactly as without the curve (and with seed 2 of A in it).
    assert lines[: len(GOLDEN_LINES)] == GOLDEN_LINES
    assert sum(line.startswith("Verdict: ") for line in lines) == 1
    assert (
        "Caution: these runs miss a point of the curve and are left out of it and its area: A seed 2" in lines
    )
    mean_a = statistics.mean(curve_value("A", s, 250_000, EVAL_SEEDS[0]) for s in range(6) if s != 2)
    mean_b = statistics.mean(curve_value("B", s, 250_000, EVAL_SEEDS[0]) for s in range(6))
    assert (
        f"  at 250,000 steps: A {mean_a:.3g}s, B {mean_b:.3g}s, B better by {mean_a - mean_b:.3g}s "
        "(evaluated after A 250,000, B 250,000 agent steps)"
    ) in lines
    assert sum(r["stage"] == "checkpoint" for r in experiment.rows()) == 6 * 3 * 2 - 1
    # The evaluation noise: the trained policies' two evaluations, against the seeds' spread.
    pairs = [
        (curve_value(s, k, STEPS, EVAL_SEEDS[0]), curve_value(s, k, STEPS, EVAL_SEEDS[1]))
        for s in "AB"
        for k in range(6)
    ]
    sd = math.sqrt(statistics.mean((x - y) ** 2 for x, y in pairs) / 2)
    noise = next(line for line in lines if line.startswith("Evaluation noise"))
    assert f"about {sd:.3g}s (SD" in noise and "% of the seed-to-seed variance" in noise


def test_the_area_decides_only_when_planned_and_the_final_point_is_then_information_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lines = on_task(tmp_path, monkeypatch, checkpoints=4, decide="auc", auc_smallest=0.05)
    assert lines[1] == "Area under the curve, 4 points - decides (planned with --decide auc):"
    assert lines[2] == "Measured: time_to_touch, mean over the 4 points (lower is better)."
    assert "  Smallest difference that matters: 0.05s." in "\n".join(lines)
    final = lines.index("Final point (1,000,000 steps) - information only, never decides:")
    verdicts = [i for i, line in enumerate(lines) if line.startswith("Verdict: ")]
    assert len(verdicts) == 1 and verdicts[0] < final
    assert lines[final + 1 : final + len(GOLDEN_LINES)] == [
        "Information only, not the verdict: " + line[len("Verdict: ") :]
        if line.startswith("Verdict: ")
        else line
        for line in GOLDEN_LINES[1:]
    ]


def test_the_curve_and_the_likeliest_score_go_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The curve, its area and the evaluation noise are of the sampled score; the trained
    policy is also scored picking its likeliest action, which decides only with --likeliest-decides."""
    import json

    asked: list[tuple[list[int], bool]] = []
    fake = fake_practise()

    def practise(task: Task, config: NitroConfig, steps: int, **options: Any) -> list[Any]:
        asked.append((list(options["checkpoints"]), options["also_likeliest"]))
        return fake(task, config, steps, **options)

    def plain(lines: list[str]) -> list[str]:
        """The lines as the curve alone says them."""
        unmeasured = (
            "; the likeliest-action score is evaluated on one seed only, so its noise is not measured"
        )
        return [
            line.replace(", sampled as in training (", " (")
            .replace("one sampled evaluation", "one evaluation")
            .replace("the sampled final results", "the final results")
            .replace(unmeasured, "")
            for line in lines
        ]

    _, curve = on_task(tmp_path / "curve", monkeypatch, practise, checkpoints=4)
    _, area = on_task(
        tmp_path / "area", monkeypatch, practise, checkpoints=4, decide="auc", auc_smallest=0.05
    )
    marks = [250_000, 500_000, 750_000]
    assert asked == [(marks, False)] * 24
    asked.clear()
    both, lines = on_task(tmp_path / "both", monkeypatch, practise, checkpoints=4, also_likeliest=True)
    assert asked == [(marks, True)] * 12
    plan = json.loads((both.folder / "plan.json").read_text(encoding="utf-8"))
    assert plan["also_likeliest"] is True and plan["checkpoints"] == marks and "decides" not in plan
    assert {r["stage"] for r in both.rows()} == {"untrained", "checkpoint", "trained", "trained_likeliest"}
    assert lines[1].startswith("Judged sampled as in training") and "A 3.52s, B 4.12s" in lines[1]
    assert lines[0] == curve[0]
    assert plain(lines[2:]) == curve[1:], "the curve, its area and the sampled verdict as without the row"
    assert any(
        line.startswith("Learning curve at evaluation seed 1,000,001, sampled as in training")
        for line in lines
    )
    assert sum(line.startswith("Verdict: B IS BETTER") for line in lines) == 1

    _, decided = on_task(tmp_path / "decides", monkeypatch, practise, checkpoints=4, likeliest_decides=True)
    assert decided[1].startswith("Judged picking the likeliest action") and "A 4.12s, B 3.52s" in decided[1]
    assert [line for line in decided if line.startswith("Verdict: ")] == [
        next(line for line in decided if line.startswith("Verdict: B IS WORSE"))
    ], "the likeliest score decides the final point"
    start = next(i for i, line in enumerate(lines) if line.startswith("Learning curve at evaluation seed"))
    shown = decided.index(lines[start])
    noise = next(line for line in lines if line.startswith("Evaluation noise"))
    assert decided[shown:] == lines[start:], "curve, area and noise stay the sampled score's"
    assert noise in decided
    assert "of the sampled final results" in noise and "its noise is not measured" in noise, noise

    _, auc = on_task(
        tmp_path / "auc",
        monkeypatch,
        practise,
        checkpoints=4,
        decide="auc",
        auc_smallest=0.05,
        also_likeliest=True,
    )
    assert auc[1].startswith("Judged sampled as in training")
    assert plain([auc[0], *auc[2:]]) == area, "the area decides as without the row"
    asked.clear()
    with pytest.raises(ValueError, match="each name the score that decides"):
        on_task(
            tmp_path / "two", monkeypatch, practise, checkpoints=4, decide="auc", auc_smallest=0.05,
            likeliest_decides=True,
        )  # fmt: skip
    assert not asked and not (tmp_path / "two" / "ab" / "plan.json").exists(), "refused before anything ran"

    # A folder started with both carries on only with both; one started with one refuses the other added.
    for folder, options, change in (
        ("both", {"checkpoints": 4}, "also_likeliest: True -> not set"),
        ("both", {"also_likeliest": True}, r"checkpoints: \[250000, 500000, 750000\] -> not set"),
        ("both", {"checkpoints": 4, "likeliest_decides": True}, "decides: not set -> trained_likeliest"),
        ("curve", {"checkpoints": 4, "also_likeliest": True}, "also_likeliest: not set -> True"),
        ("decides", {"likeliest_decides": True}, r"checkpoints: \[250000, 500000, 750000\] -> not set"),
    ):
        asked.clear()
        with pytest.raises(ValueError, match=change):
            on_task(tmp_path / folder, monkeypatch, practise, **options)
        assert not asked, "refused before any run"
    assert on_task(tmp_path / "both", monkeypatch, practise, checkpoints=4, also_likeliest=True)[1] == lines


def test_points_evaluated_far_apart_are_said_to_compare_unequal_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lines = on_task(tmp_path / "far", monkeypatch, fake_practise(late=30_000), checkpoints=4)
    cautions = [line for line in lines if "agent steps apart" in line]
    assert len(cautions) == 3 and cautions[0].startswith(
        "Caution: at 250,000 steps A and B were evaluated 30,000"
    )
    _, lines = on_task(tmp_path / "near", monkeypatch, fake_practise(late=10_000), checkpoints=4)
    assert not any("agent steps apart" in line for line in lines)


@pytest.mark.parametrize(
    ("options", "said"),
    [
        (["--task", "ball_touch", "--checkpoints", "4", "--decide", "auc"], "--decide auc needs"),
        (["--task", "ball_touch", "--decide", "auc", "--auc-smallest", "0.1"], "--decide auc needs"),
        (["--task", "ball_touch", "--checkpoints", "4", "--auc-smallest", "0.1"], "only for --decide auc"),
        (["--task", "ball_touch", "--checkpoints", "11"], "0 (off) or 2 to 10"),
        (["--speed", "--checkpoints", "4"], "add --task TASK"),
        (
            [
                "--task",
                "ball_touch",
                "--checkpoints",
                "4",
                "--decide",
                "auc",
                "--auc-smallest",
                "0.1",
                "--likeliest-decides",
            ],
            "each name the score that decides",
        ),
    ],
)
def test_the_command_refuses_curve_options_that_do_not_fit_before_starting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str], said: str
) -> None:
    from nitro.cli import main

    (tmp_path / "a.toml").write_text("[ppo]\npolicy_lr = 1e-4\n", encoding="utf-8")
    (tmp_path / "b.toml").write_text("[ppo]\npolicy_lr = 3e-4\n", encoding="utf-8")
    folder = tmp_path / "results"
    code = main(["ab", str(tmp_path / "a.toml"), str(tmp_path / "b.toml"), *options, "--folder", str(folder)])
    assert code == 1 and said in capsys.readouterr().err
    assert not folder.exists(), "nothing started"
