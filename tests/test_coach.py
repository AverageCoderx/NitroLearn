"""The training coach's rules, on made-up metrics.csv and skills.csv files: the stages and "ready for the next
stage" from the skill checks, and each hint (plateau, reward farming, entropy collapse, a stuck learning rate)
when its sign is there and not otherwise. Nothing is trained."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nitro.dashboard import coach
from nitro.runs import RunFolder

Row = dict[str, float]


def healthy(i: int, n: int) -> Row:
    """A run that learns: reward and touches rising, randomness falling slowly, steady updates."""
    t = i / max(n - 1, 1)
    return {
        "agent_steps": float((i + 1) * 100_000),
        "iteration": float(i + 1),
        "reward_per_step": 0.01 + 0.02 * t,
        "touches_per_minute": 2.0 + 6.0 * t,
        "entropy": 0.9 - 0.4 * t,
        "policy_change": 0.01,
        "clip_fraction": 0.1,
        "policy_lr": 2e-4,
        "paid_abs_velocity_player_to_ball": 0.006,
        "paid_abs_strong_touch": 0.004,
        "paid_abs_goal": 0.002,
    }


def run(tmp_path: Path, n: int = 300, change: Callable[[int, Row], None] | None = None) -> RunFolder:
    rows = [healthy(i, n) for i in range(n)]
    if change:
        for i, row in enumerate(rows):
            change(i, row)
    folder = RunFolder(tmp_path / "bot")
    folder.root.mkdir(parents=True)
    header = list(rows[0])
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return folder


def skills(folder: RunFolder, checks: list[tuple[int, dict[str, float]]]) -> None:
    """Writes skills.csv as `nitro skills` does: one row per task and save."""
    bars = {
        "ball_touch": 3.0,
        "recovery": 1.6,
        "boost_pickup": 40.0,
        "shot": 0.45,
        "save": 0.8,
        "aerial_touch": 5.6,
        "kickoff_duel": 0.5,
        "fifty_fifty": 0.5,
        "wall_read": 3.0,
    }
    with (folder.root / "skills.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "when",
            "agent_steps",
            "task",
            "metric",
            "value",
            "unit",
            "bar",
            "above_bar",
            "games",
            "seed",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for steps, values in checks:
            for task, value in values.items():
                lower = not coach.SKILLS[task].higher_is_better
                above = value <= bars[task] if lower else value >= bars[task]
                row = {"when": "2026-01-01 00:00", "agent_steps": steps, "task": task, "metric": "m"}
                row |= {"value": value, "unit": "s" if lower else "", "bar": bars[task], "above_bar": above}
                writer.writerow(row | {"games": 512, "seed": 0})


def kinds(advice: dict[str, Any]) -> list[str]:
    return [hint["kind"] for hint in advice["hints"]]


BASICS = {"ball_touch": 2.0, "recovery": 1.0, "boost_pickup": 50.0}
SHOOTING = {"shot": 0.6, "save": 0.9}


def test_the_coach_agrees_with_the_skill_checks_tasks() -> None:
    from nitro.skills import DEFAULT_TASKS
    from nitro.tasks import TASKS

    assert set(coach.SKILLS) == set(DEFAULT_TASKS)
    for name, skill in coach.SKILLS.items():
        assert skill.higher_is_better == TASKS[name].higher_is_better, name
        assert skill.smallest_difference == TASKS[name].smallest_difference, name
    staged = [task for _, _, tasks in coach.STAGES for task in tasks]
    assert sorted(staged) == sorted(coach.SKILLS), "every skill in exactly one stage"


def test_a_healthy_run_gets_no_hints_and_without_checks_the_plan_asks_for_one(tmp_path: Path) -> None:
    advice = coach.advise(run(tmp_path))
    assert advice["hints"] == [] and not advice["too_early"]
    assert not advice["checked"] and advice["current"] == "basics"
    assert "Run a skill check" in advice["verdict"]


def test_too_few_updates_say_nothing(tmp_path: Path) -> None:
    advice = coach.advise(run(tmp_path, n=10, change=lambda i, r: r.update(entropy=0.01)))
    assert advice["too_early"] and advice["hints"] == []


def test_the_stages_follow_the_newest_skill_check(tmp_path: Path) -> None:
    folder = run(tmp_path)
    skills(folder, [(1_000_000, {"ball_touch": 4.0, "recovery": 2.0, "boost_pickup": 20.0, "shot": 0.1})])
    advice = coach.advise(folder)
    assert advice["current"] == "basics" and "Working on: basics" in advice["verdict"]
    assert "touching the ball" in advice["verdict"] and "Ready for" not in advice["verdict"]

    skills(folder, [(1_000_000, {"ball_touch": 4.0}), (5_000_000, {**BASICS, "shot": 0.2, "save": 0.5})])
    advice = coach.advise(folder)
    assert advice["current"] == "shooting" and advice["checked_at"] == 5_000_000
    assert advice["verdict"].startswith("Ready for shooting: basics is done.")
    basics = advice["stages"][0]
    assert basics["passed"] and all(s["above"] for s in basics["skills"])
    shot = advice["stages"][1]["skills"][0]
    assert shot["value"] == 0.2 and shot["bar"] == 0.45 and not shot["lower_is_better"]

    skills(folder, [(9_000_000, {**BASICS, **SHOOTING, "aerial_touch": 9.0})])
    assert coach.advise(folder)["verdict"].startswith("Ready for air play: shooting is done.")
    skills(folder, [(9_000_000, {**BASICS, **SHOOTING, "aerial_touch": 4.0})])
    assert coach.advise(folder)["verdict"].startswith("Ready for duels and reads: air play is done.")
    duels = {"kickoff_duel": 0.6, "fifty_fifty": 0.55, "wall_read": 2.5}
    skills(folder, [(9_000_000, {**BASICS, **SHOOTING, "aerial_touch": 4.0, **duels})])
    done = coach.advise(folder)
    assert done["current"] is None and "Every skill is above its bar" in done["verdict"]


def test_a_skill_never_checked_is_not_called_under_its_bar(tmp_path: Path) -> None:
    """A check from before the drills existed, or of a bot without opponents (the duels are left out), has no
    row for them: the plan says they are not checked yet, not that the bot is under their bars."""
    folder = run(tmp_path)
    skills(folder, [(9_000_000, {**BASICS, **SHOOTING, "aerial_touch": 4.0, "wall_read": 4.0})])
    verdict = coach.advise(folder)["verdict"]
    assert verdict.startswith("Ready for duels and reads: air play is done."), verdict
    assert "Still under the bar: reading a ball off the wall." in verdict, verdict
    assert "Not checked yet: winning kickoffs, winning 50/50s." in verdict, verdict
    assert "under the bar: winning" not in verdict and "wall, winning" not in verdict, verdict


def test_entropy_collapse(tmp_path: Path) -> None:
    folder = run(tmp_path, change=lambda i, r: r.update(entropy=0.05) if i > 270 else None)
    hint = next(h for h in coach.advise(folder)["hints"] if h["kind"] == "entropy")
    assert "0.05" in hint["text"] and "ppo.entropy" in hint["try"]
    assert "entropy" not in kinds(
        coach.advise(run(tmp_path / "b", change=lambda i, r: r.update(entropy=0.12)))
    )


def test_one_reward_paying_almost_everything_is_farming(tmp_path: Path) -> None:
    def farm(i: int, r: Row) -> None:
        if i > 250:
            r.update(paid_abs_velocity_player_to_ball=0.2, paid_abs_strong_touch=0.001, paid_abs_goal=0.001)

    hint = next(h for h in coach.advise(run(tmp_path, change=farm))["hints"] if h["kind"] == "farming")
    assert "velocity_player_to_ball" in hint["text"] and hint["title"].startswith("One reward makes up 99%")

    def alone(i: int, r: Row) -> None:  # the only reward paid: a choice, not farming
        r.update(paid_abs_strong_touch=0.0, paid_abs_goal=0.0)

    assert "farming" not in kinds(coach.advise(run(tmp_path / "b", change=alone)))


def test_more_pay_for_fewer_touches_is_farming(tmp_path: Path) -> None:
    def farm(i: int, r: Row) -> None:
        if i >= 240:
            r.update(reward_per_step=0.05, touches_per_minute=3.0)
        elif i >= 180:
            r.update(reward_per_step=0.03, touches_per_minute=6.0)

    hint = next(h for h in coach.advise(run(tmp_path, change=farm))["hints"] if h["kind"] == "farming")
    assert hint["title"] == "It is paid more while it touches the ball less"
    assert "rose 67%" in hint["text"] and "fell 50%" in hint["text"]


def test_a_stuck_learning_rate(tmp_path: Path) -> None:
    zero = coach.advise(run(tmp_path / "a", change=lambda i, r: r.update(policy_lr=0.0) if i > 260 else None))
    assert [h["title"] for h in zero["hints"] if h["kind"] == "learning_rate"] == [
        "Its learning rate is at zero"
    ]

    clipped = coach.advise(
        run(tmp_path / "b", change=lambda i, r: r.update(clip_fraction=0.45) if i > 260 else None)
    )
    assert any("too much" in h["title"] for h in clipped["hints"])

    def still(i: int, r: Row) -> None:
        r["policy_change"] = 0.01 if i < 100 else 0.0002

    hint = next(
        h for h in coach.advise(run(tmp_path / "c", change=still))["hints"] if h["kind"] == "learning_rate"
    )
    assert hint["title"] == "Its updates barely move it any more" and "2%" in hint["text"]

    def scheduled(i: int, r: Row) -> None:  # the rate was lowered on purpose: small updates are expected
        r.update(policy_change=0.01 if i < 100 else 0.0002, policy_lr=2e-4 if i < 100 else 1e-5)

    assert "learning_rate" not in kinds(coach.advise(run(tmp_path / "d", change=scheduled)))


def test_a_plateau_in_the_metrics_and_in_the_skill_checks(tmp_path: Path) -> None:
    def flat(i: int, r: Row) -> None:
        if i >= 180:
            r.update(reward_per_step=0.025, touches_per_minute=6.5 + 0.01 * math.sin(i))

    folder = run(tmp_path / "a", change=flat)
    hint = next(h for h in coach.advise(folder)["hints"] if h["kind"] == "plateau")
    assert hint["title"] == "Training looks flat" and "skill check" in hint["try"]
    assert "plateau" not in kinds(coach.advise(run(tmp_path / "a2", n=150, change=flat))), "too short to say"

    folder = run(tmp_path / "b")
    skills(
        folder,
        [(s, {"shot": 0.2 + 0.01 * k, **BASICS}) for k, s in enumerate((1_000_000, 2_000_000, 3_000_000))],
    )
    hint = next(h for h in coach.advise(folder)["hints"] if h["kind"] == "plateau")
    assert hint["title"] == "Its skills have stopped improving" and "scoring an open goal" in hint["text"]
    assert "touching the ball" not in hint["text"], "a skill above its bar is not a plateau"

    skills(folder, [(s, {"shot": 0.1 + 0.1 * k}) for k, s in enumerate((1_000_000, 2_000_000, 3_000_000))])
    assert "plateau" not in kinds(coach.advise(folder)), "improving by more than the smallest difference"


def test_the_coach_reads_a_long_file_in_pieces(tmp_path: Path) -> None:
    folder = run(tmp_path, n=3000)
    rows = coach.metric_rows(folder)
    assert 300 <= len(rows) <= 450, "spread rows plus the newest, never the whole file"
    steps = [r["agent_steps"] for r in rows]
    assert steps == sorted(steps) and steps[-1] == 3000 * 100_000 and steps[0] == 100_000
