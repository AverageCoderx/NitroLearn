# ruff: noqa: E501  (plain-English hint texts, kept one sentence a line)
"""The training coach: a plan in four stages (basics, shooting, air play, duels and reads) read from the bot's skill checks
(skills.csv), and hints read from its metrics.csv when training looks stalled, farmed, too certain or held
back by its learning rate. It only reads and suggests: it never changes a setting.

Every rule is a plain function of the rows, so tests can feed it made-up files."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitro.dashboard import bots
from nitro.runs import RunFolder


@dataclass(frozen=True)
class Skill:
    label: str
    higher_is_better: bool
    smallest_difference: float  # a change smaller than this is no change (as in nitro.tasks)


# The skill check's tasks (nitro.tasks.TASKS, which imports PyTorch: tests check these agree).
SKILLS: dict[str, Skill] = {
    "ball_touch": Skill("Touching the ball", False, 0.1),
    "recovery": Skill("Landing on its wheels", False, 0.1),
    "boost_pickup": Skill("Collecting boost", True, 12.0),
    "shot": Skill("Scoring an open goal", True, 0.05),
    "save": Skill("Saving a shot", True, 0.05),
    "aerial_touch": Skill("Touching a ball in the air", False, 0.1),
    "kickoff_duel": Skill("Winning kickoffs", True, 0.05),
    "fifty_fifty": Skill("Winning 50/50s", True, 0.05),
    "wall_read": Skill("Reading a ball off the wall", False, 0.1),
}

STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("basics", "Basics", ("ball_touch", "recovery", "boost_pickup")),
    ("shooting", "Shooting", ("shot", "save")),
    ("air", "Air play", ("aerial_touch",)),
    ("duels", "Duels and reads", ("kickoff_duel", "fifty_fifty", "wall_read")),
)

STAGE_ADVICE = {
    "basics": (
        "Driving to the ball, touching it, landing on its wheels and picking up boost. Keep training with "
        "the starter's rewards; a bot usually touches the ball within its first hours."
    ),
    "shooting": (
        "Putting the ball in the net and keeping it out of its own. If it stays under these bars, rewards "
        "for the ball moving towards the opponents' net and for goals can help (the Rewards tab, + Add reward)."
    ),
    "air": (
        "Hitting the ball in the air. This comes late. Rewards for air touches can help once shooting is there; "
        "add them with small weights, so they never drown out goals."
    ),
    "duels": (
        "Winning kickoffs and 50/50s against a scripted opponent and reading a ball off the wall. Their bars "
        "are what a simple scripted player reaches, so passing them means the bot beats that player. Training "
        "against opponents (env.opponents) and the matching practice starts help."
    ),
}

# Thresholds of the hints.
MIN_ROWS = 40  # fewer learning updates than this: too early to say anything
RECENT_ROWS = 20  # the newest updates the hints read
ENTROPY_FLOOR = 0.1  # as the console's warning (health.TrendWarnings)
DOMINANT_SHARE = 0.9  # one reward over this share of all paid (as the console and the Rewards tab)
PLATEAU_ROWS = 200  # updates before a plateau is looked for
PLATEAU_CHANGE = 0.03  # under 3% change between the two newest fifths of the run: flat
FARM_RISE = 0.2  # reward per step up at least 20% ...
FARM_FALL = 0.2  # ... while touches per minute fell at least 20%
STILL_SHARE = 0.05  # updates moving the policy less than 5% as much as early on: barely learning
CLIP_HIGH = 0.3  # PPO cutting back over 30% of the samples: the steps are too big
LR_ZERO = 1e-8  # a learning rate this low learns nothing


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(clean) if clean else None


def _column(rows: list[dict[str, float]], name: str) -> list[float]:
    return [row[name] for row in rows if name in row and math.isfinite(row[name])]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stage_view(skills_rows: list[dict[str, str]]) -> dict[str, Any]:
    """The plan: each stage with its skills' newest check, which stage the bot is at, and whether it is ready
    for the next. Without skill checks, the plan only says to run one."""
    newest: dict[str, dict[str, str]] = {}
    for row in skills_rows:
        task = row.get("task", "")
        if task in SKILLS and _steps(row) >= _steps(newest.get(task, {})):
            newest[task] = row
    stages: list[dict[str, Any]] = []
    current: str | None = None
    for key, name, tasks in STAGES:
        skills: list[dict[str, Any]] = []
        for task in tasks:
            check = newest.get(task)
            skills.append(
                {
                    "task": task,
                    "label": SKILLS[task].label,
                    "value": _float(check.get("value")) if check else None,
                    "bar": _float(check.get("bar")) if check else None,
                    "unit": check.get("unit", "") if check else "",
                    "lower_is_better": not SKILLS[task].higher_is_better,
                    "above": check is not None and check.get("above_bar", "").strip().lower() == "true",
                    "checked": check is not None,
                }
            )
        passed = all(skill["above"] for skill in skills)
        if not passed and current is None:
            current = key
        stages.append(
            {"key": key, "name": name, "about": STAGE_ADVICE[key], "skills": skills, "passed": passed}
        )
    checked = bool(newest)
    if not checked:
        verdict = (
            "Run a skill check (the Tools tab) after a few hours of training: the plan reads its results to say "
            "which stage the bot is at."
        )
    elif current is None:
        verdict = "Every skill is above its bar. From here, longer training and matches against it show more."
    else:
        index = [s["key"] for s in stages].index(current)
        stage = stages[index]
        under = [str(s["label"]).lower() for s in stage["skills"] if s["checked"] and not s["above"]]
        unchecked = [str(s["label"]).lower() for s in stage["skills"] if not s["checked"]]
        verdict = f"Working on: {str(stage['name']).lower()}."
        if under:
            verdict += f" Still under the bar: {', '.join(under)}."
        if unchecked:
            verdict += f" Not checked yet: {', '.join(unchecked)}."
        if index > 0:
            verdict = (
                f"Ready for {stage['name'].lower()}: {stages[index - 1]['name'].lower()} is done. " + verdict
            )
    steps = max((_steps(row) for row in newest.values()), default=0)
    return {"stages": stages, "current": current, "checked": checked, "verdict": verdict, "checked_at": steps}


def hints(
    rows: list[dict[str, float]], skills_rows: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Hints from the bot's metrics rows (oldest first), each a title, what it means and what to try. Empty
    when nothing looks wrong, or too early to tell."""
    if len(rows) < MIN_ROWS:
        return []
    found: list[dict[str, str]] = []
    recent = rows[-RECENT_ROWS:]
    for rule in (_entropy, _farming, _learning_rate, _plateau):
        hint = rule(rows, recent, skills_rows or [])
        if hint is not None:
            found.append(hint)
    return found


def _entropy(rows: list[dict[str, float]], recent: list[dict[str, float]], _: Any) -> dict[str, str] | None:
    entropy = _median(_column(recent, "entropy"))
    if entropy is None or entropy >= ENTROPY_FLOOR:
        return None
    return {
        "kind": "entropy",
        "title": "It has become almost certain of every choice",
        "text": (
            f"Its randomness has fallen to {entropy:.2f} (1 is picking at random, 0 is always the same move), so "
            "it barely tries anything new. That is fine if it keeps improving; if it has stalled, it may be stuck "
            "in a habit."
        ),
        "try": "Raise ppo.entropy a little (Settings, Learning) and Apply. Small steps: a third more at a time.",
    }


def _shares(row: dict[str, float]) -> dict[str, float]:
    paid = {
        k[len("paid_abs_") :]: v for k, v in row.items() if k.startswith("paid_abs_") and math.isfinite(v)
    }
    total = sum(abs(v) for v in paid.values())
    return {name: abs(v) / total for name, v in paid.items()} if total > 0 else {}


def _farming(rows: list[dict[str, float]], recent: list[dict[str, float]], _: Any) -> dict[str, str] | None:
    sums: dict[str, float] = {}
    for row in recent:
        for name, share in _shares(row).items():
            sums[name] = sums.get(name, 0.0) + share / len(recent)
    paying = {name: share for name, share in sums.items() if share > 0.001}
    if len(paying) > 1:
        name, share = max(paying.items(), key=lambda item: item[1])
        if share > DOMINANT_SHARE:
            return {
                "kind": "farming",
                "title": f"One reward makes up {share:.0%} of its pay",
                "text": (
                    f"{name} pays almost everything it earns, so it learns to chase that one thing. It may be "
                    "farming it: being paid without playing well."
                ),
                "try": (
                    f"Watch it play to see what it does for {name}. If that is not good play, lower its weight on "
                    "the Rewards tab, or Try these rewards first."
                ),
            }
    fifth = max(len(rows) // 5, RECENT_ROWS)
    early, late = rows[-2 * fifth : -fifth], rows[-fifth:]
    reward = (_mean(_column(early, "reward_per_step")), _mean(_column(late, "reward_per_step")))
    touches = (_mean(_column(early, "touches_per_minute")), _mean(_column(late, "touches_per_minute")))
    if None in reward or None in touches or not early:
        return None
    (r0, r1), (t0, t1) = reward, touches
    assert r0 is not None and r1 is not None and t0 is not None and t1 is not None
    if r0 > 0 and t0 > 0 and r1 >= r0 * (1 + FARM_RISE) and t1 <= t0 * (1 - FARM_FALL):
        return {
            "kind": "farming",
            "title": "It is paid more while it touches the ball less",
            "text": (
                f"Its reward per step rose {r1 / r0 - 1:.0%} while its touches per minute fell {1 - t1 / t0:.0%}. "
                "It may have found a way to be paid without playing."
            ),
            "try": (
                "Watch it play. Then look at what each reward pays on the Rewards tab: the reward whose share grew "
                "is the likely one to lower."
            ),
        }
    return None


def _learning_rate(
    rows: list[dict[str, float]], recent: list[dict[str, float]], _: Any
) -> dict[str, str] | None:
    lr = _median(_column(recent, "policy_lr"))
    if lr is not None and lr < LR_ZERO:
        return {
            "kind": "learning_rate",
            "title": "Its learning rate is at zero",
            "text": "Its policy's learning rate is 0, so it plays but learns nothing.",
            "try": "Set ppo.policy_lr back to a normal value (Settings, Learning; Back to default) and Apply.",
        }
    clip = _median(_column(recent, "clip_fraction"))
    if clip is not None and clip > CLIP_HIGH:
        return {
            "kind": "learning_rate",
            "title": "Each update tries to change it too much",
            "text": (
                f"PPO is cutting back {clip:.0%} of every update, a sign the learning rate is too high for where "
                "the bot is now. Learning can become jumpy or stall."
            ),
            "try": "Lower ppo.policy_lr a little (a third less) and Apply.",
        }
    early = _median(_column(rows[: max(len(rows) // 10, RECENT_ROWS)], "policy_change"))
    now = _median(_column(recent, "policy_change"))
    first_lr = _median(_column(rows[:RECENT_ROWS], "policy_lr"))
    same_lr = lr is None or first_lr is None or abs(lr - first_lr) <= 1e-3 * first_lr
    if early and now is not None and now < STILL_SHARE * early and same_lr:
        return {
            "kind": "learning_rate",
            "title": "Its updates barely move it any more",
            "text": (
                f"Each update now changes its play about {now / early:.0%} as much as early on, with the same "
                "learning rate. It may be stuck where it is."
            ),
            "try": (
                "If it has also stopped improving, raise ppo.entropy a little first; a slightly higher "
                "ppo.policy_lr is the other thing to try. Change one at a time."
            ),
        }
    return None


def _plateau(
    rows: list[dict[str, float]], recent: list[dict[str, float]], skills_rows: list[dict[str, str]]
) -> dict[str, str] | None:
    flat_skills = _flat_skills(skills_rows)
    if flat_skills:
        return {
            "kind": "plateau",
            "title": "Its skills have stopped improving",
            "text": f"Over its last three skill checks, {', '.join(flat_skills)} did not get better.",
            "try": _PLATEAU_TRY,
        }
    if len(rows) < PLATEAU_ROWS:
        return None
    fifth = len(rows) // 5
    early, late = rows[-2 * fifth : -fifth], rows[-fifth:]
    for column in ("reward_per_step", "touches_per_minute"):
        a, b = _mean(_column(early, column)), _mean(_column(late, column))
        if a is None or b is None or a == 0 or abs(b - a) > PLATEAU_CHANGE * abs(a):
            return None
    return {
        "kind": "plateau",
        "title": "Training looks flat",
        "text": (
            "Its reward and its touches have hardly changed over the last fifth of its training. In self-play "
            "that can also mean it is improving as fast as its opponent, so check its skills before changing "
            "anything."
        ),
        "try": "Run a skill check (Tools) and compare it with the last one. " + _PLATEAU_TRY,
    }


_PLATEAU_TRY = (
    "Things to try, one at a time: train longer (plateaus often end by themselves), raise ppo.entropy a little, "
    "turn on playing against past versions (Self-play), or add a reward for the next skill in the plan."
)


def _flat_skills(skills_rows: list[dict[str, str]]) -> list[str]:
    """The skills whose last three checks (at different saves) show no gain beyond the skill's smallest
    difference, among those still under their bar."""
    by_task: dict[str, dict[int, dict[str, str]]] = {}
    for row in skills_rows:
        if row.get("task") in SKILLS:
            by_task.setdefault(row["task"], {})[_steps(row)] = row
    flat = []
    for task, checks in by_task.items():
        ordered = [checks[s] for s in sorted(checks)][-3:]
        if len(ordered) < 3 or ordered[-1].get("above_bar", "").strip().lower() == "true":
            continue
        values = [_float(r.get("value")) for r in ordered]
        if any(v is None for v in values):
            continue
        skill = SKILLS[task]
        first, last = values[0], values[-1]
        assert first is not None and last is not None
        gain = last - first if skill.higher_is_better else first - last
        if gain < skill.smallest_difference:
            flat.append(skill.label.lower())
    return flat


def _steps(row: dict[str, str]) -> int:
    value = _float(row.get("agent_steps"))
    return int(value) if value is not None else -1


def _float(text: str | None) -> float | None:
    try:
        value = float(text) if text not in (None, "") else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def read_skills(file: Path) -> list[dict[str, str]]:
    if not file.is_file():
        return []
    with file.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_rows(
    folder: RunFolder, points: int = 400, newest: int = 2 * RECENT_ROWS
) -> list[dict[str, float]]:
    """About `points` rows spread through the bot's metrics file and its newest `newest` rows, oldest first, as
    numbers (a long file is never read whole)."""
    found = bots.ends(folder)
    if not found.last:
        return []
    header = found.header
    spread = bots.spread_rows(folder, points)
    tail = bots.newest_rows(folder, newest)
    seen, out = set(), []
    for row in [*spread, *tail]:
        key = tuple(row[:2])
        if key in seen:
            continue
        seen.add(key)
        values = {}
        for name, cell in zip(header, row, strict=False):
            value = bots.number(cell)
            if value is not None:
                values[name] = value
        out.append(values)
    out.sort(key=lambda r: r.get("agent_steps", 0.0))
    return out


def advise(folder: RunFolder) -> dict[str, Any]:
    """What the coach page shows for the bot."""
    skills_rows = read_skills(folder.root / "skills.csv")
    rows = metric_rows(folder)
    plan = stage_view(skills_rows)
    return {
        **plan,
        "hints": hints(rows, skills_rows),
        "updates": len(rows),
        "too_early": len(rows) < MIN_ROWS,
    }
