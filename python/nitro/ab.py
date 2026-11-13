"""`nitro ab`: compares two settings files on the same seeds, with the noise rules built in.

Two kinds of comparison:
- On a practice task (`--task`): for every seed, a fresh policy is trained with each settings file on
  the task and judged by the task's metric (`nitro.tasks`). The task decides what is practised and how
  it is scored; the settings files decide how the bot learns (networks, learner, precision,
  observation and actions). Never judged by the reward.
- On speed (`--speed`): each settings file is trained as it is for a few iterations, taking turns, and
  judged by agent steps per second.

The verdict: B is better or worse only if the difference of the means is at least the smallest
difference that matters, larger than the noise floor (the gap between the two halves of each setup's
seeds: A against A, B against B), and each mean lies outside the other's range. If only the first two
hold, it is borderline; otherwise there is no measurable difference. The 95% interval, the seed-by-seed
wins and the difference these seeds could reliably detect are shown to help read the result, never to
decide it. Every run is listed, including crashed ones and ones that failed the learning check.

On a task, `--checkpoints K` also evaluates each run at K-1 evenly spaced points before the end and shows
the learning curve and its area (the mean over the K points), which notices a setting that learns faster.
The area decides only with `--decide auc` and its own smallest difference, set in a plan written before
the runs; otherwise the final point decides, as without it. The curve is scored sampled, as in
training; `--also-likeliest` and `--likeliest-decides` add a likeliest-action score of the trained
policy only, so the latter cannot go with `--decide auc`.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import hashlib
import itertools
import json
import math
import random
import statistics
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nitro.config import NitroConfig
from nitro.health import learning_problems
from nitro.tasks import EVAL_SEEDS, Task, practice_config, practise
from nitro.training import build

SETUPS = ("A", "B")
SPEED_SMALLEST = 0.03  # speed differences below 3% of A's mean don't matter
SPEED_WARMUP, SPEED_ITERATIONS = 3, 10
# Two-sided 95% quantiles of Student's t; between rows the smaller degrees of freedom is used (wider).
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04, 60: 2.00}  # fmt: skip
_STRIP = 50  # characters in the console dot plot
_POWER_TRIALS = 400  # simulated comparisons per step of `Comparison.detectable`
CHECKPOINTS = range(2, 11)  # K, the points of a learning curve, the end included
DECIDE = ("final", "auc")
_STEPS_APART = 0.02  # A and B evaluated further apart than this share of the run at a point: a caution


def not_measured(settings: Iterable[NitroConfig], task: bool) -> list[str]:
    """Settings switched on in A or B that this comparison leaves out, said plainly."""
    if not any(config.replay_mix.active for config in settings):
        return []
    if task:  # practice_config takes only how the bot learns: [replay_mix] is not part of it
        return ["Not used here: [replay_mix] (copying replays while training runs only in nitro train)."]
    return [
        "Not used here: [replay_mix]. Its copy steps run only in nitro train, so their cost does not show in "
        "this speed comparison."
    ]


def differences(a: NitroConfig, b: NitroConfig) -> list[str]:
    """Settings that differ between `a` and `b` as 'path: A value -> B value' (paths and seeds aside)."""

    def flat(value: Any, prefix: str) -> dict[str, Any]:
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in flat(item, f"{prefix}{key}.").items()}
        return {prefix.rstrip("."): value}

    first = flat(dataclasses.asdict(a), "")
    second = flat(dataclasses.asdict(b), "")
    changed = [
        f"{key}: {first[key]} -> {second[key]}"
        for key in first
        if first[key] != second[key] and not key.startswith(("paths.", "run.seed", "rewards"))
    ]
    return changed + reward_differences(a.active_rewards, b.active_rewards)


def reward_text(reward: Any) -> str:
    """A reward in a few words: its name, its own parameters, its weight and any zero-sum sharing."""

    def kind(item: Any) -> str:
        if item.name == "all_of":
            return f"all_of({', '.join(kind(part) for part in item.parts)})"
        if hasattr(item, "settings"):  # a reward of one's own
            listed = ", ".join(f"{key}={value}" for key, value in item.settings)
            return f"{item.name}({listed})" if listed else item.name
        params = {
            f.name: getattr(item, f.name)
            for f in dataclasses.fields(item)
            if f.name not in ("name", "weight", "zero_sum", "parts", "enabled")
        }
        listed = ", ".join(f"{name}={value:g}" for name, value in params.items())
        return f"{item.name}({listed})" if params else item.name

    text = f"{kind(reward)} weight {reward.weight:g}"
    if not reward.enabled:
        text += " (switched off)"
    if reward.zero_sum is not None:
        zs = reward.zero_sum
        text += f" zero-sum (team spirit {zs.team_spirit:g}, opponent scale {zs.opponent_scale:g})"
    return text


def reward_differences(a: Sequence[Any], b: Sequence[Any]) -> list[str]:
    """'rewards[i]: A reward -> B reward' for every position where the two reward lists differ."""
    return [
        f"rewards[{i}]: {reward_text(a[i]) if i < len(a) else '(none)'} -> "
        f"{reward_text(b[i]) if i < len(b) else '(none)'}"
        for i in range(max(len(a), len(b)))
        if i >= len(a) or i >= len(b) or a[i] != b[i]
    ]


@dataclass(frozen=True)
class Comparison:
    a: dict[int, float]  # seed (or repeat) -> value
    b: dict[int, float]
    smallest: float
    higher_is_better: bool

    def better(self, x: float, y: float) -> float:
        return x - y if self.higher_is_better else y - x

    @staticmethod
    def registered_floor(values: dict[int, float]) -> float:
        """A against A as pre-registered: the gap between the means of the first and second half."""
        ordered = [values[key] for key in sorted(values)]
        half = len(ordered) // 2
        return abs(statistics.mean(ordered[:half]) - statistics.mean(ordered[half:])) if half else math.inf

    @staticmethod
    def noise_floor(values: dict[int, float]) -> float:
        """A against A over every balanced split: the mean gap between the two halves' means (1).

        One chosen split is a single noisy draw of the noise; averaging over all of them (252 for 10 seeds)
        gives the expected gap. Splits are paired with their complements, so each pair counts once.
        """
        ordered = [values[key] for key in sorted(values)]
        half = len(ordered) // 2
        if half == 0:
            return math.inf
        total = sum(ordered)
        gaps = [
            abs(sum(first) / half - (total - sum(first)) / (len(ordered) - half))
            for first in itertools.combinations(ordered, half)
        ]
        return statistics.mean(gaps)

    def verdict(self) -> str:
        mean_a, mean_b = statistics.mean(self.a.values()), statistics.mean(self.b.values())
        difference = self.better(mean_b, mean_a)
        floor = max(self.noise_floor(self.a), self.noise_floor(self.b))
        beyond_noise = abs(difference) >= self.smallest and abs(difference) > floor
        apart = not min(self.a.values()) <= mean_b <= max(self.a.values()) and not (
            min(self.b.values()) <= mean_a <= max(self.b.values())
        )
        if beyond_noise and apart:
            return "B is better" if difference > 0 else "B is worse"
        return "borderline" if beyond_noise else "no measurable difference"

    def interval(self) -> tuple[float, float]:
        """Welch's 95% interval for how much better B is than A."""
        va, vb = statistics.variance(self.a.values()), statistics.variance(self.b.values())
        na, nb = len(self.a), len(self.b)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            centre = self.better(statistics.mean(self.b.values()), statistics.mean(self.a.values()))
            return centre, centre
        df = se**4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        t = _T95[max(k for k in _T95 if k <= max(1, math.floor(df)))]
        centre = self.better(statistics.mean(self.b.values()), statistics.mean(self.a.values()))
        return centre - t * se, centre + t * se

    def detectable(self) -> float:
        """The real difference the verdict rule would call "better" 8 times in 10, with these seed counts
        and this spread.

        Found by simulating the rule itself on normal noise at the pooled standard deviation (a bisection
        over the true difference), because the rule is far stricter than a t-test: the textbook figure
        2.8 * sd * sqrt(2/n) understated it by about half.
        """
        pooled = math.sqrt((statistics.variance(self.a.values()) + statistics.variance(self.b.values())) / 2)
        if pooled == 0:
            return self.smallest
        rng = random.Random(0)
        counts = (len(self.a), len(self.b))

        def power(shift: float) -> float:
            hits = 0
            for _ in range(_POWER_TRIALS):
                a = {i: rng.gauss(0.0, pooled) for i in range(counts[0])}
                b = {i: rng.gauss(shift, pooled) for i in range(counts[1])}
                hits += Comparison(a, b, self.smallest, higher_is_better=True).verdict() == "B is better"
            return hits / _POWER_TRIALS

        low, high = 0.0, 8.0 * pooled + self.smallest
        for _ in range(12):
            mid = (low + high) / 2
            if power(mid) >= 0.8:
                high = mid
            else:
                low = mid
        return high

    def wins(self) -> tuple[int, int, int]:
        """Seeds where B beat A, where they were within the smallest difference, where A beat B."""
        gaps = [self.better(self.b[key], self.a[key]) for key in self.a if key in self.b]
        return (
            sum(gap >= self.smallest for gap in gaps),
            sum(abs(gap) < self.smallest for gap in gaps),
            sum(gap <= -self.smallest for gap in gaps),
        )


def _strip(values: Iterable[float], low: float, high: float) -> str:
    cells = [" "] * _STRIP
    for value in values:
        cells[round((value - low) / (high - low or 1) * (_STRIP - 1))] = "o"
    return "".join(cells)


def _fmt(value: float, unit: str) -> str:
    return (f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}") + unit


def report(
    comparison: Comparison, measured: str, unit: str, problems: list[str], per: str = "seed"
) -> list[str]:
    """The comparison in plain English, every run included; `per` names what a run is (seed, repeat)."""
    c = comparison

    def fmt(value: float) -> str:
        return _fmt(value, unit)

    lines = [f"Measured: {measured} ({'higher' if c.higher_is_better else 'lower'} is better)."]
    low = min(*c.a.values(), *c.b.values())
    high = max(*c.a.values(), *c.b.values())
    for name, values in (("A", c.a), ("B", c.b)):
        listed = ", ".join(f"{key}: {fmt(value)}" for key, value in sorted(values.items()))
        lines += [
            f"  {name} |{_strip(values.values(), low, high)}| mean {fmt(statistics.mean(values.values()))}, "
            f"range {fmt(min(values.values()))} to {fmt(max(values.values()))}",
            f"      every {per}: {listed}",
        ]
    lines.append(f"     {fmt(low):<{_STRIP // 2 + 1}}{fmt(high):>{_STRIP // 2 + 1}}")
    verdict = c.verdict()
    apart = (
        "B's results are clearly apart from A's, by more than the noise and the smallest difference that "
        "matters."
    )
    meaning = {
        "B is better": apart,
        "B is worse": apart,
        "borderline": f"the means differ by more than the noise, but the two sets of {per}s overlap. "
        f"Treat it as a hint, not a result; more {per}s would settle it.",
        "no measurable difference": "the difference is within the noise or too small to matter.",
    }[verdict]
    low_ci, high_ci = c.interval()
    better_b, same, better_a = c.wins()
    too_small = verdict == "no measurable difference" and -c.smallest < low_ci and high_ci < c.smallest
    lines += [
        f"Verdict: {verdict.upper()} - {meaning}"
        + (" The difference, if there is one, is too small to matter." if too_small else ""),
        f"  Smallest difference that matters: {fmt(c.smallest)}. Noise floor (A vs A, B vs B), averaged over "
        f"every half-and-half split of the {per}s: {fmt(Comparison.noise_floor(c.a))}, "
        f"{fmt(Comparison.noise_floor(c.b))} (the pre-registered first-half vs second-half split: "
        f"{fmt(Comparison.registered_floor(c.a))}, {fmt(Comparison.registered_floor(c.b))}).",
        f"  To help read it (never decides): B better than A by {fmt(low_ci)} to {fmt(high_ci)} "
        f"(95% interval); {per} by {per}, B better {better_b}, about the same {same}, A better {better_a}.",
        f"  With {min(len(c.a), len(c.b))} {per}s, only differences of about {fmt(c.detectable())} or more "
        "would reliably show up (this rule calls them 8 times in 10); a smaller real difference can still "
        "read 'no measurable difference'.",
    ]
    return lines + [f"Caution: {problem}" for problem in problems]


class Experiment:
    """A comparison's folder: the settings files as they were (by setup name), every run's rows, the
    result."""

    def __init__(self, folder: Path, files: Mapping[str, Path]) -> None:
        self.folder = folder
        self.rows_file = folder / "runs.csv"
        folder.mkdir(parents=True, exist_ok=True)
        for name, file in files.items():
            copy, text = folder / f"{name}.toml", file.read_text(encoding="utf-8-sig")
            if copy.exists() and copy.read_text(encoding="utf-8") != text:
                raise ValueError(
                    f"{file} changed since this comparison started in {folder}. "
                    "Use a new --name to compare the new version."
                )
            copy.write_text(text, encoding="utf-8")

    def rows(self) -> list[dict[str, str]]:
        """Every saved run row; a row cut short by a crash (missing columns) does not count as done."""
        if not self.rows_file.exists():
            return []
        with self.rows_file.open(encoding="utf-8") as file:
            return [row for row in csv.DictReader(file) if None not in row.values()]

    def check_plan(self, plan: dict[str, Any]) -> None:
        """Refuses to carry on a comparison started with another task, length, seed count or code
        version; the first call stores the plan in the folder. The code version is what the code holds
        (`_code`), so commits of results or documents in between do not stop a comparison. A key in either
        plan and not the other (an option such as --with-rewards, --likeliest-decides or --decide auc added
        or left out) is a change too."""
        file = self.folder / "plan.json"
        plan = plan | {"commit": _commit(), "code": _code()}
        if file.exists():
            saved = json.loads(file.read_text(encoding="utf-8"))
            keys = [*plan, *(k for k in saved if k not in plan)]
            changed = [
                f"{k}: {saved.get(k, 'not set')} -> {plan.get(k, 'not set')}"
                for k in keys
                if k != "commit" and (k in saved, saved.get(k)) != (k in plan, plan.get(k))
            ]
            if changed:
                raise ValueError(
                    f"this comparison in {self.folder} was started with other settings "
                    f"({'; '.join(changed)}). Use another folder (--name or --folder), or the same task, "
                    "steps, seeds, options and code."
                )
        else:
            file.write_text(json.dumps(plan, indent=1), encoding="utf-8")

    def run(
        self,
        jobs: list[tuple[str, int]],
        label: str,
        fields: list[str],
        one: Callable[[str, int], list[dict[str, Any]]],
    ) -> None:
        """Runs `one(setup, key)` for every (setup, key) not run yet, saving its rows as it goes.

        `key` is a seed or a repeat (`label`). Stopped part way, the same command carries on from here; a
        run that crashed is run again (its crash row stays in runs.csv).
        """
        done = {(row["setup"], int(row["key"])) for row in self.rows() if row["stage"] != "crashed"}
        todo = [job for job in jobs if job not in done]
        new = not self.rows_file.exists()
        with self.rows_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["setup", "key", *fields], extrasaction="ignore")
            if new:
                writer.writeheader()
            start = time.monotonic()
            for count, (setup, key) in enumerate(todo, start=1):
                try:
                    rows = one(setup, key)
                except Exception:  # a crash is a result too: record it and carry on
                    rows = [
                        {"stage": "crashed", "problems": traceback.format_exc(limit=3).replace("\n", " | ")}
                    ]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                writer.writerows({"setup": setup, "key": key} | row for row in rows)
                file.flush()
                left = (time.monotonic() - start) / count * (len(todo) - count)
                print(
                    f"{setup}, {label} {key} done ({count} of {len(todo)}); about {left / 60:.0f} min left."
                )

    def finish(self, lines: list[str]) -> None:
        text = "\n".join(lines) + "\n"
        (self.folder / "result.txt").write_text(text, encoding="utf-8")
        print(text, end="")


# What a comparison's runs depend on: the package, the core and its build files. Presets, shortcuts, tools,
# benchmarks, tests, documents and results do not count, so committing those never stops a comparison
# part-way. The compared settings files themselves are held by `Experiment`, which refuses one that changed.
_CODE_PATHS = (
    "python",
    "crates",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
)


def _code(root: Path | None = None) -> str:
    """A fingerprint of the committed code in `_CODE_PATHS` (`git ls-tree` of HEAD): the same across commits
    that change nothing a comparison's runs depend on."""
    try:
        done = subprocess.run(
            ["git", "ls-tree", "--full-tree", "-r", "HEAD", "--", *_CODE_PATHS],
            cwd=root or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if done.returncode or not done.stdout:
        return "unknown"
    return hashlib.sha256(done.stdout.strip().encode()).hexdigest()[:16]


def _commit() -> str:
    """The code version, so a comparison never mixes runs from before and after a change."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return done.stdout.strip() or "unknown"


def run_problems(rows: list[dict[str, str]], per: str) -> list[str]:
    crashes = {(r["setup"], r["key"]) for r in rows if r["stage"] == "crashed"}
    finished = {(r["setup"], r["key"]) for r in rows if r["stage"] != "crashed"}
    crashed = sorted(f"{setup} {per} {key}" for setup, key in crashes - finished)
    again = sorted(f"{setup} {per} {key}" for setup, key in crashes & finished)
    unlearned = sorted({f"{r['setup']} {per} {r['key']}" for r in rows if r.get("learned") == "False"})
    found = []
    if crashed:
        found.append(f"these runs crashed and are left out (see runs.csv): {', '.join(crashed)}")
    if again:
        found.append(f"these runs crashed once and were run again (see runs.csv): {', '.join(again)}")
    if unlearned:
        found.append(f"these runs failed the learning check (kept in): {', '.join(unlearned)}")
    return found


def curve_marks(
    steps: int,
    checkpoints: int,
    decide: str = "final",
    auc_smallest: float | None = None,
    likeliest_decides: bool = False,
) -> list[int]:
    """The agent steps of the evaluations before the end for a curve of `checkpoints` = K evenly spaced
    points, the end being the K-th (none for 0), once the curve options are checked to fit together (and
    with `likeliest_decides`, which names another deciding score)."""
    if checkpoints != 0 and checkpoints not in CHECKPOINTS:
        raise ValueError(f"--checkpoints must be 0 (off) or 2 to 10, not {checkpoints}.")
    if decide not in DECIDE:
        raise ValueError(f"--decide must be one of {', '.join(DECIDE)}, not {decide}.")
    if decide == "auc" and (not checkpoints or auc_smallest is None):
        raise ValueError(
            "--decide auc needs --checkpoints and --auc-smallest, both fixed in a plan written before "
            "the runs."
        )
    if decide != "auc" and auc_smallest is not None:
        raise ValueError("--auc-smallest is only for --decide auc.")
    if decide == "auc" and likeliest_decides:
        raise ValueError(
            "--decide auc and --likeliest-decides each name the score that decides; use one. The curve is "
            "scored sampled, as in training; --also-likeliest still shows the likeliest-action score."
        )
    if auc_smallest is not None and not 0 < auc_smallest < math.inf:
        raise ValueError(f"--auc-smallest must be above 0, not {auc_smallest}.")
    marks = [round(steps * i / checkpoints) for i in range(1, checkpoints)] if checkpoints else []
    if marks and (marks[0] <= 0 or len(set(marks)) < len(marks)):
        raise ValueError(f"{steps:,} agent steps are too few for {checkpoints} evaluation points.")
    return marks


def _information_only(lines: list[str]) -> list[str]:
    """A comparison shown beside the deciding one: its verdict line says so."""
    return [
        "Information only, not the verdict: " + line.removeprefix("Verdict: ")
        if line.startswith("Verdict: ")
        else line
        for line in lines
    ]


def _curve_report(
    task: Task,
    rows: list[dict[str, str]],
    steps: int,
    marks: list[int],
    final: Comparison,
    decide: str,
    auc_smallest: float | None,
    problems: list[str],
    also_likeliest: bool = False,
) -> list[str]:
    """The final point, the curve at every point, its area (the mean over the points) and the evaluation
    noise; the final point decides unless `decide` is "auc". The curve, its area and the noise are
    of the sampled score; `final` is the deciding one, the likeliest-action score with --likeliest-decides,
    which `also_likeliest` names in the headings."""
    count, unit = len(marks) + 1, task.unit
    first = str(EVAL_SEEDS[0])
    cautions: list[str] = []
    complete: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for setup in SETUPS:
        at_marks: dict[int, list[tuple[int, float]]] = {}
        ends: dict[int, tuple[int, float]] = {}
        for r in rows:
            if r["setup"] != setup or r["eval_seed"] != first or r["stage"] not in ("checkpoint", "trained"):
                continue
            point = (int(r["agent_steps"]), float(r[task.metric]))
            if r["stage"] == "checkpoint":  # a run's rows are saved in order: the i-th is at marks[i]
                at_marks.setdefault(int(r["key"]), []).append(point)
            else:
                ends[int(r["key"])] = point
        complete[setup] = {
            key: [*at_marks[key], end]
            for key, end in ends.items()
            if len(at_marks.get(key, [])) == len(marks)
        }
        left_out = sorted((set(ends) | set(at_marks)) - set(complete[setup]))
        if left_out:
            cautions.append(
                f"these runs miss a point of the curve and are left out of it and its area: "
                f"{', '.join(f'{setup} seed {key}' for key in left_out)}"
            )
    how = ", sampled as in training" if also_likeliest else ""
    curve = [
        "",
        f"Learning curve at evaluation seed {EVAL_SEEDS[0]:,}{how} (information only, never decides):",
    ]
    if min(len(points) for points in complete.values()) == 0:
        curve.append("  A or B has no run with every point.")
    else:
        for i, mark in enumerate([*marks, steps]):
            means = {s: statistics.mean(p[i][1] for p in complete[s].values()) for s in SETUPS}
            at = {s: round(statistics.mean(p[i][0] for p in complete[s].values())) for s in SETUPS}
            where = f"at {mark:,} steps" if i < len(marks) else f"at the end ({steps:,} steps)"
            curve.append(
                f"  {where}: A {_fmt(means['A'], unit)}, B {_fmt(means['B'], unit)}, B better by "
                f"{_fmt(task.better(means['B'], means['A']), unit)} (evaluated after A {at['A']:,}, "
                f"B {at['B']:,} agent steps)"
            )
            if abs(at["A"] - at["B"]) > _STEPS_APART * steps:
                cautions.append(
                    f"{where} A and B were evaluated {abs(at['A'] - at['B']):,} agent steps apart, more than "
                    f"{_STEPS_APART:.0%} of the run (their iterations differ in size, or one learns while "
                    "it plays and has a batch still waiting), so that point compares unequal training."
                )
    areas = {s: {k: statistics.mean(v for _, v in p) for k, p in complete[s].items()} for s in SETUPS}
    heading = f"Area under the curve, {count} points"
    if min(len(v) for v in areas.values()) < 2:
        area = ["", f"{heading}: too few runs with every point (at least 2 per setup are needed)."]
    else:
        smallest = auc_smallest or task.smallest_difference
        area = report(
            Comparison(areas["A"], areas["B"], smallest, task.higher_is_better),
            f"{task.metric}, mean over the {count} points",
            unit,
            [],
        )
        if decide == "auc":
            area = ["", f"{heading} - decides (planned with --decide auc):", *area]
        else:
            area = ["", f"{heading} - information only, never decides:", *_information_only(area)]
    # Both evaluation seeds score sampled, so the noise is the sampled score's, whichever decides.
    trained = {s: _values(rows, task, s, "trained", EVAL_SEEDS[0]) for s in SETUPS}
    again = {s: _values(rows, task, s, "trained", EVAL_SEEDS[1]) for s in SETUPS}
    pairs = [(trained[s][k], again[s][k]) for s in SETUPS for k in trained[s] if k in again[s]]
    noise: list[str] = []
    if pairs and min(len(v) for v in trained.values()) >= 2:
        sd = math.sqrt(statistics.mean((x - y) ** 2 for x, y in pairs) / 2)
        pooled = (statistics.variance(trained["A"].values()) + statistics.variance(trained["B"].values())) / 2
        sampled = " sampled" if also_likeliest else ""
        share = (
            f", {sd**2 / pooled:.0%} of the seed-to-seed variance of the{sampled} final results"
            if pooled
            else ""
        )
        unmeasured = (
            "; the likeliest-action score is evaluated on one seed only, so its noise is not measured"
            if also_likeliest
            else ""
        )
        noise = [
            "",
            f"Evaluation noise (information only): one{sampled} evaluation of a trained policy varies by "
            f"about {_fmt(sd, unit)} (SD, from its two evaluation seeds){share}{unmeasured}.",
        ]
    final_lines = report(final, task.metric, unit, [])
    if decide == "auc":
        blocks = [
            *area[1:],
            "",
            f"Final point ({steps:,} steps) - information only, never decides:",
            *_information_only(final_lines),
            *curve,
        ]
    else:
        blocks = [*final_lines, *curve, *area]
    return [*blocks, *noise, *(f"Caution: {problem}" for problem in [*problems, *cautions])]


def _values(
    rows: list[dict[str, str]], task: Task, setup: str, stage: str, eval_seed: int
) -> dict[int, float]:
    """The deciding metric per seed for one setup, stage and evaluation seed."""
    return {
        int(r["key"]): float(r[task.metric])
        for r in rows
        if r["setup"] == setup and r["stage"] == stage and r["eval_seed"] == str(eval_seed)
    }


def compare_on_task(
    experiment: Experiment,
    task: Task,
    settings: dict[str, NitroConfig],
    seeds: list[int],
    steps: int,
    with_rewards: bool = False,
    also_likeliest: bool = False,
    likeliest_decides: bool = False,
    checkpoints: int = 0,
    decide: str = "final",
    auc_smallest: float | None = None,
) -> list[str]:
    """`with_rewards`: the two files' `[[rewards]]` replace the task's (a reward-design comparison).
    `also_likeliest`: each trained policy is also scored picking its likeliest action, as the RLBot bot plays;
    the sampled score decides unless `likeliest_decides`, which the plan records before any run.
    `checkpoints` = K: also evaluated at K-1 evenly spaced points before the end, shown as a curve and its
    area (sampled, as in training); the area decides only with `decide="auc"` and its own `auc_smallest`,
    which `likeliest_decides` cannot go with."""
    marks = curve_marks(steps, checkpoints, decide, auc_smallest, likeliest_decides=likeliest_decides)
    also_likeliest = also_likeliest or likeliest_decides
    fields = [
        "stage",
        "agent_steps",
        "eval_seed",
        "episodes",
        *task.metrics,
        "train_minutes",
        "learned",
        "problems",
    ]
    plan: dict[str, Any] = {
        "task": task.name, "steps": steps, "seeds": len(seeds), "eval_seeds": list(EVAL_SEEDS)
    }  # fmt: skip
    if with_rewards:
        plan["rewards"] = "from the settings files"
    if also_likeliest:
        plan["also_likeliest"] = True
    if likeliest_decides:
        plan["decides"] = "trained_likeliest"
    if marks:
        plan["checkpoints"] = marks
    if decide == "auc":
        plan |= {"decide": "auc", "auc_smallest": auc_smallest}
    experiment.check_plan(plan)
    jobs = [(setup, seed) for seed in seeds for setup in SETUPS]  # taking turns spreads any drift evenly
    experiment.run(
        jobs,
        "seed",
        fields,
        lambda setup, seed: practise(
            task,
            practice_config(task, settings[setup], seed, with_rewards),
            steps,
            checkpoints=marks,
            also_likeliest=also_likeliest,
        ),
    )
    rows = experiment.rows()

    def values(setup: str, stage: str, eval_seed: int) -> dict[int, float]:
        return _values(rows, task, setup, stage, eval_seed)

    sampled = {setup: values(setup, "trained", EVAL_SEEDS[0]) for setup in SETUPS}
    likeliest = {setup: values(setup, "trained_likeliest", EVAL_SEEDS[0]) for setup in SETUPS}
    trained = likeliest if likeliest_decides else sampled
    if min(len(v) for v in trained.values()) < 2:
        return ["Too few finished runs to compare (at least 2 per setup are needed); see runs.csv."]
    problems = run_problems(rows, "seed")
    again = {setup: values(setup, "trained", EVAL_SEEDS[1]) for setup in SETUPS}
    # The re-measure is of the sampled score, whichever decides.
    unstable = max(abs(sampled[s][k] - again[s][k]) for s in SETUPS for k in sampled[s] if k in again[s])
    if unstable >= task.smallest_difference:
        problems.append(
            f"evaluating the same policy twice differed by up to {unstable:.3g}{task.unit}, as much as the "
            "smallest difference that matters, so single results are shaky."
        )
    untrained = values("A", "untrained", EVAL_SEEDS[0])
    comparison = Comparison(trained["A"], trained["B"], task.smallest_difference, task.higher_is_better)
    lines = [
        f"Practice task '{task.name}', {steps:,} agent steps per run; the untrained policy scored "
        f"{statistics.mean(untrained.values()):.3g}{task.unit} on average.",
    ]
    if also_likeliest:
        ways = ("sampled as in training", "picking the likeliest action (as the RLBot bot)")
        judged, also = ways[::-1] if likeliest_decides else ways
        other = sampled if likeliest_decides else likeliest
        means = ", ".join(f"{s} {statistics.mean(v.values()):.3g}{task.unit}" for s, v in other.items() if v)
        lines.append(f"Judged {judged}, as the plan says; also scored {also}, not deciding: {means}.")
    if not marks:
        return [*lines, *report(comparison, task.metric, task.unit, problems)]
    curve = _curve_report(
        task, rows, steps, marks, comparison, decide, auc_smallest, problems, also_likeliest
    )
    return [*lines, *curve]


def _speed(config: NitroConfig) -> dict[str, Any]:
    # Self-play needs a pool folder; without one it would silently be off and its cost never measured.
    # A scratch pool starts empty, as a new run's does.
    with tempfile.TemporaryDirectory(prefix="nitro-speed-pool-", ignore_cleanup_errors=True) as pool:
        return _speed_with(config, Path(pool) if config.selfplay.enabled else None)


def _speed_with(config: NitroConfig, pool_folder: Path | None) -> dict[str, Any]:
    setup = build(config, pool_folder=pool_folder)
    for _ in range(SPEED_WARMUP):
        setup.trainer.iteration()
    reports = [setup.trainer.iteration() for _ in range(SPEED_ITERATIONS)]
    problems = learning_problems([r.learn for r in reports])
    seconds = sum(r.times.total for r in reports)
    return {
        "stage": "measured",
        "agent_sps": round(sum(r.agent_steps for r in reports) / seconds),
        "learned": not problems,
        "problems": "; ".join(problems),
    }


def compare_speed(experiment: Experiment, settings: dict[str, NitroConfig], repeats: int) -> list[str]:
    experiment.check_plan({"task": "speed", "repeats": repeats, "iterations": SPEED_ITERATIONS})
    # A B, B A, A B, ...: each setup goes first equally often.
    jobs = [(setup, r) for r in range(repeats) for setup in (SETUPS if r % 2 == 0 else SETUPS[::-1])]
    fields = ["stage", "agent_sps", "learned", "problems"]
    experiment.run(jobs, "repeat", fields, lambda setup, _: [_speed(settings[setup])])
    rows = experiment.rows()
    speeds = {
        setup: {int(r["key"]): float(r["agent_sps"]) for r in rows if r["setup"] == setup and r["agent_sps"]}
        for setup in SETUPS
    }
    if min(len(v) for v in speeds.values()) < 2:
        return ["Too few finished runs to compare (at least 2 per setup are needed); see runs.csv."]
    smallest = SPEED_SMALLEST * statistics.mean(speeds["A"].values())
    comparison = Comparison(speeds["A"], speeds["B"], smallest, higher_is_better=True)
    lines = report(comparison, "agent steps per second", "", run_problems(rows, "repeat"), per="repeat")
    lines.insert(1, "  (The PC should be otherwise idle while this runs: other programs change the speed.)")
    return lines
