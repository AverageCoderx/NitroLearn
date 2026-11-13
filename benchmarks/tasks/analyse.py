"""Applies the narrow-task analysis plan to the runs written by run.py.

For each task, with the plan's thresholds:
- Learner works: every FP32 seed's trained policy beats every seed's untrained policy by at least the
  smallest difference that matters, and the FP32 mean meets the task's bar.
- Noise floor (A vs A): within each precision, seeds 0-4 against seeds 5-9.
- FP16 against FP32: "FP16 better/worse" only if the difference of the means is at least the smallest
  difference that matters, larger than both A-vs-A gaps, and FP16's mean lies outside FP32's seed range;
  "inconclusive" if only the first two hold; otherwise "no measurable difference".
- Measurement stability: each trained policy evaluated with two evaluation seeds; the largest gap.
`--compare A B` compares other setups the same way (B against A), such as `fp16 fp16-sums`.
Every seed is listed; crashed runs and runs that failed the learning check are reported, never dropped.

    python -I benchmarks/tasks/analyse.py ball_touch.csv ...
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from nitro.tasks import EVAL_SEEDS, TASKS

EVAL_A, EVAL_B = (str(seed) for seed in EVAL_SEEDS)


def analyse(path: Path, base: str = "fp32", other: str = "fp16") -> list[str]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rule = TASKS[rows[0]["task"]]
    direction = "higher" if rule.higher_is_better else "lower"
    lines = [f"== {path.name}: {rule.name} ({rule.metric}, {direction} is better)"]
    finished = {(r["precision"], r["seed"]) for r in rows if r["stage"] != "crashed"}
    crashes = [(r["precision"], r["seed"]) for r in rows if r["stage"] == "crashed"]
    crashed = [f"{p} seed {s}" for p, s in crashes if (p, s) not in finished]
    reran = [f"{p} seed {s}" for p, s in crashes if (p, s) in finished]
    unlearned = sorted({f"{r['precision']} seed {r['seed']}" for r in rows if r.get("learned") == "False"})
    note = f" (crashed, then run again: {reran})" if reran else ""
    lines.append(
        f"crashed runs: {crashed or 'none'}{note}; runs failing the learning check: {unlearned or 'none'}"
    )

    def values(precision: str, stage: str, eval_seed: str) -> dict[int, float]:
        return {
            int(r["seed"]): float(r[rule.metric])
            for r in rows
            if r["precision"] == precision and r["stage"] == stage and r["eval_seed"] == eval_seed
        }

    trained = {p: values(p, "trained", EVAL_A) for p in (base, other)}
    untrained = values(base, "untrained", EVAL_A)
    for precision, by_seed in trained.items():
        listed = ", ".join(f"{seed}: {value:.3f}" for seed, value in sorted(by_seed.items()))
        lines.append(f"{precision} trained, every seed: {listed}")
    lines.append("untrained, every seed: " + ", ".join(f"{s}: {v:.3f}" for s, v in sorted(untrained.items())))

    fp32 = list(trained[base].values())
    worst_trained = min(fp32, key=lambda v: rule.better(v, 0)) if fp32 else float("nan")
    best_untrained = max(untrained.values(), key=lambda v: rule.better(v, 0))
    margin = rule.better(worst_trained, best_untrained)
    mean32 = statistics.mean(fp32)
    meets_bar = rule.better(mean32, rule.bar) >= 0
    works = margin >= rule.smallest_difference and meets_bar
    lines.append(
        f"learner works: {'PASS' if works else 'FAIL'} (worst trained {base} seed beats the best "
        f"untrained seed by {margin:.3f}, needed {rule.smallest_difference}; {base} mean {mean32:.3f}, "
        f"bar {rule.bar})"
    )

    gaps = {}
    for precision, by_seed in trained.items():
        first = [v for s, v in by_seed.items() if s < 5]
        second = [v for s, v in by_seed.items() if s >= 5]
        gaps[precision] = abs(statistics.mean(first) - statistics.mean(second))
        lines.append(f"noise floor {precision} (seeds 0-4 vs 5-9): {gaps[precision]:.3f}")

    fp16 = list(trained[other].values())
    mean16 = statistics.mean(fp16)
    difference = rule.better(mean16, mean32)
    beyond_noise = abs(difference) >= rule.smallest_difference and abs(difference) > max(gaps.values())
    outside = not (min(fp32) <= mean16 <= max(fp32))
    if beyond_noise and outside:
        verdict = f"{other} better" if difference > 0 else f"{other} worse"
    elif beyond_noise:
        verdict = "inconclusive"
    else:
        verdict = "no measurable difference"
    lines.append(
        f"{other} vs {base}: {verdict} ({other} mean {mean16:.3f}, range {min(fp16):.3f}-{max(fp16):.3f}; "
        f"{base} mean {mean32:.3f}, range {min(fp32):.3f}-{max(fp32):.3f}; smallest difference that matters "
        f"{rule.smallest_difference})"
    )

    again = {p: values(p, "trained", EVAL_B) for p in (base, other)}
    spread = [abs(trained[p][s] - again[p][s]) for p in trained for s in trained[p] if s in again[p]]
    lines.append(f"stability: largest gap between two evaluations of the same policy {max(spread):.3f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--compare", nargs=2, default=["fp32", "fp16"], metavar=("A", "B"))
    args = parser.parse_args()
    for path in args.results:
        print("\n".join(analyse(path, *args.compare)), end="\n\n")


if __name__ == "__main__":
    main()
