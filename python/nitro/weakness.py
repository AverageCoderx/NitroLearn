"""The weakness map : where in a practice task's situations a bot does worst.

Each situation of a skills battery is sorted into coarse groups by how it started, as the bot's car saw it
(where the ball was, how high and fast, the car's speed, height, heading and distance to the ball; read
from `EpisodeRecord`'s start fields). Per group: how many situations, the task's own measure on them, and
how many of them are among the worst 10% of all the task's situations. Groups doing worse than the task as
a whole come first; a group of fewer than FEW situations is flagged, as its number is mostly luck. Printed
by `nitro skills --weak` and kept in runs/<name>/weakness.json for the dashboard. For a player's reading,
never evidence for a setting.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nitro import _core

if TYPE_CHECKING:
    from nitro.tasks import Task

REPORT = "weakness.json"
WORST_SHARE = 0.1
FEW = 20  # a group with fewer situations is flagged as too small to trust
THIRD_Y = 5120.0 / 3  # the field's thirds along its length

Situation = list[_core.EpisodeRecord]  # car 0's episode, with Orange's first car's in a duel


def _ball_area(r: _core.EpisodeRecord) -> str:
    if r.start_ball_y < -THIRD_Y:
        return "ball in the bot's own third"
    return "ball in the opponent's third" if r.start_ball_y > THIRD_Y else "ball in the middle third"


def _facing(r: _core.EpisodeRecord) -> str:
    if r.start_facing > 0.5:
        return "car facing the ball"
    return "car facing away from the ball" if r.start_facing < -0.5 else "car side-on to the ball"


def _distance(r: _core.EpisodeRecord) -> str:
    if r.start_distance < 1500:
        return "ball near (under 1500)"
    return "ball far (over 3000)" if r.start_distance > 3000 else "ball at middle distance"


FEATURES: tuple[tuple[str, Callable[[_core.EpisodeRecord], str]], ...] = (
    ("ball area", _ball_area),
    ("ball height", lambda r: "ball on the ground" if r.start_ball_z < 150 else "ball in the air"),
    ("ball speed", lambda r: "ball still or slow" if r.start_ball_speed < 500 else "ball moving fast"),
    ("car speed", lambda r: "car slow" if r.start_car_speed < 500 else "car fast"),
    ("car height", lambda r: "car on the ground" if r.start_car_z < 60 else "car in the air"),
    ("car heading", _facing),
    ("distance", _distance),
)


def _value(task: Task, situations: list[Situation]) -> float:
    return task.measure([r for s in situations for r in s])[task.metric]


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 4)


def analyse(task: Task, situations: list[Situation]) -> dict[str, Any]:
    """The weakness map of one task's situations (each a list starting with the bot's own episode)."""
    scored = [(s, _value(task, [s])) for s in situations]
    scored = [(s, v) for s, v in scored if not math.isnan(v)]
    if not scored:
        return {"task": task.name, "metric": task.metric, "situations": 0, "groups": []}
    # Worst first; ties at the cut-off all count, so the worst share can be above 10%.
    ordered = sorted((v for _, v in scored), reverse=not task.higher_is_better)
    cutoff = ordered[max(0, math.ceil(WORST_SHARE * len(ordered)) - 1)]
    worst = [task.better(v, cutoff) <= 0 for _, v in scored]
    overall = _value(task, [s for s, _ in scored])
    groups: list[dict[str, Any]] = []
    for feature, label_of in FEATURES:
        members: dict[str, list[int]] = {}
        for index, (situation, _) in enumerate(scored):
            members.setdefault(label_of(situation[0]), []).append(index)
        if len(members) < 2:
            continue  # every situation in one group: says nothing
        for label, indices in members.items():
            value = _value(task, [scored[i][0] for i in indices])
            groups.append(
                {
                    "feature": feature,
                    "group": label,
                    "count": len(indices),
                    "value": _clean(value),
                    "gap": _clean(task.better(value, overall)),
                    "worst_share": round(sum(worst[i] for i in indices) / len(indices), 4),
                    "few": len(indices) < FEW,
                }
            )
    groups.sort(key=lambda g: math.inf if g["gap"] is None else g["gap"])
    return {
        "task": task.name,
        "metric": task.metric,
        "unit": task.unit,
        "higher_is_better": task.higher_is_better,
        "situations": len(scored),
        "overall": _clean(overall),
        "cutoff": _clean(cutoff),
        "worst_share": round(sum(worst) / len(worst), 4),
        "groups": groups,
    }


def describe(entry: dict[str, Any], top: int = 3) -> list[str]:
    """A few plain lines: the groups doing worse than the task overall, worst first."""
    if not entry.get("situations"):
        return [f"{entry['task']}: no situations to map"]
    unit = entry.get("unit", "")
    lines = [
        f"{entry['task']}: {entry['metric']} {entry['overall']:.3g}{unit} overall; the worst "
        f"{entry['worst_share']:.0%} of the {entry['situations']} situations scored "
        f"{entry['cutoff']:.3g}{unit} or worse"
        + (" (many tie at that score, so more than 10%)." if entry["worst_share"] > 2 * WORST_SHARE else ".")
    ]
    weak = [g for g in entry["groups"] if g["gap"] is not None and g["gap"] < 0][:top]
    if not weak:
        lines.append("  no group does clearly worse than the rest")
    for g in weak:
        note = " (few situations: mostly luck)" if g["few"] else ""
        lines.append(
            f"  {g['group']}: {g['value']:.3g}{unit} in {g['count']} situations, "
            f"{g['worst_share']:.0%} of them among the worst{note}"
        )
    return lines


def append_report(path: Path, agent_steps: int, entries: list[dict[str, Any]], seed: int, games: int) -> None:
    """Adds this save's map to the run's weakness.json (oldest save first), replacing an earlier map of
    the same save, seed and games."""
    try:
        maps = json.loads(path.read_text(encoding="utf-8"))
        maps = [x for x in maps if isinstance(x, dict)] if isinstance(maps, list) else []
    except (OSError, ValueError):
        maps = []
    maps = [
        m
        for m in maps
        if not (m.get("agent_steps") == agent_steps and m.get("seed") == seed and m.get("games") == games)
    ]
    maps.append(
        {
            "when": time.strftime("%Y-%m-%d %H:%M"),
            "agent_steps": agent_steps,
            "seed": seed,
            "games": games,
            "tasks": entries,
        }
    )
    maps.sort(key=lambda m: int(m.get("agent_steps", 0)))
    partial = path.with_name(path.name + ".partial")  # never a half-written report
    partial.write_text(json.dumps(maps, indent=1), encoding="utf-8")
    partial.replace(path)


def read_report(path: Path) -> list[dict[str, Any]]:
    """The run's maps, oldest save first ([] when none); for the dashboard, without PyTorch."""
    try:
        maps = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in maps if isinstance(m, dict)] if isinstance(maps, list) else []
