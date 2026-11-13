"""Narrow practice tasks, run for a comparison planned in advance (the tasks themselves: `nitro.tasks`).

For every setup (precision) and seed, trains a fresh policy for `--steps` agent steps and evaluates it
before, at each `--checkpoints` step and after training (`nitro.tasks.practise`). Every run writes one
row per evaluation to `--out`; nothing is filtered, and a run that crashes is recorded as such. The
pass/fail analysis follows the plan fixed before the run (`analyse.py`).

    python -I benchmarks/tasks/run.py --task ball_touch --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import traceback
from pathlib import Path

import torch

from nitro.config.schema import NitroConfig
from nitro.tasks import TASKS, Task, practise

ROOT = Path(__file__).resolve().parents[2]
# A setup's name: (run.precision, run.fp16_sums). "fp16-sums" keeps the playing passes' sums in FP16.
SETUPS = {"fp32": ("fp32", "off"), "fp16": ("fp16", "off"), "fp16-sums": ("fp16", "playing")}


def task_config(
    task: Task, meshes: Path, precision: str, seed: int, layers: tuple[int, ...] | None
) -> NitroConfig:
    config = task.settings()
    model = config.model if layers is None else dataclasses.replace(config.model, policy=layers, value=layers)
    return dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=meshes),
        run=dataclasses.replace(
            config.run,
            precision=SETUPS[precision][0],  # type: ignore[arg-type]
            fp16_sums=SETUPS[precision][1],  # type: ignore[arg-type]
            seed=seed,
        ),
        model=model,
    )


def finished(out: Path) -> tuple[set[tuple[str, int]], bool]:
    """The (precision, seed) runs in `out` that did not crash, and whether it holds any rows to keep (a
    crashed run is run again; its crash row stays)."""
    if not out.exists():
        return set(), False
    with out.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {(r["precision"], int(r["seed"])) for r in rows if r["stage"] != "crashed"}, bool(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--meshes", type=Path, default=ROOT / "collision_meshes")
    parser.add_argument("--precisions", nargs="+", choices=sorted(SETUPS), default=["fp32", "fp16"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--steps", type=int, help="agent steps of training per run (default: the task's)")
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[])
    parser.add_argument(
        "--layers", nargs="+", type=int, help="layer sizes of both networks instead of the defaults"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the runs already in --out and run only the missing ones (and those that crashed)",
    )
    args = parser.parse_args()
    task = TASKS[args.task]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "precision", "seed", "stage", "agent_steps", "eval_seed", "episodes", *task.metrics,
        "train_minutes", "learned", "problems",
    ]  # fmt: skip
    done, kept = finished(args.out) if args.resume else (set(), False)
    with args.out.open("a" if kept else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not kept:
            writer.writeheader()
        for precision in args.precisions:
            for seed in args.seeds:
                if (precision, seed) in done:
                    continue
                base = {"task": task.name, "precision": precision, "seed": seed}
                try:
                    layers = tuple(args.layers) if args.layers else None
                    config = task_config(task, args.meshes, precision, seed, layers)
                    rows = [
                        base | row
                        for row in practise(task, config, args.steps or task.steps, args.checkpoints)
                    ]
                except Exception:  # a crash is a result too: record it and carry on
                    rows = [
                        base
                        | {"stage": "crashed", "problems": traceback.format_exc(limit=3).replace("\n", " | ")}
                    ]
                gc.collect()
                torch.cuda.empty_cache()
                writer.writerows(rows)
                file.flush()
                for row in rows:
                    print(row, flush=True)


if __name__ == "__main__":
    main()
