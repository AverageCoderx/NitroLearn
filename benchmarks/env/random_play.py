"""Environment throughput with random actions.

For each configuration, plays the same games twice: once driven entirely by Rust and once driven by a
Python loop through the PyTorch-tensor bindings (as the trainer will). Both replay one pre-generated
table of random actions, so they simulate identical games; the order of the two alternates between
repeats (Rust first, then Python first, ...). Reports agent steps per second, the Python driver's
overhead relative to Rust, and the share of environment time spent in physics.

Two driving patterns are measured. "lockstep": every group steps, then all are waited for, so any
Python time adds directly to the step time (worst case). "pipelined": each group's next step starts as
soon as that group is free, so Python prepares one group while the others step (the trainer's pattern).

    python -I benchmarks/env/random_play.py --out env_random_play.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from nitro import _core
from nitro.config import default_config
from nitro.config.schema import NitroConfig
from nitro.envs import EnvPool, core_settings

ROOT = Path(__file__).resolve().parents[2]
WARMUP_STEPS = 50


def make_config(meshes: Path, obs: str, arenas: int, threads: int, groups: int) -> NitroConfig:
    base = default_config(ROOT)
    env = dataclasses.replace(base.env, num_arenas=arenas, threads=threads, groups=groups, observation=obs)
    return dataclasses.replace(base, paths=dataclasses.replace(base.paths, collision_meshes=meshes), env=env)


def python_driven(
    config: NitroConfig, table: torch.Tensor, steps: int, pipelined: bool
) -> tuple[float, _core.EnvStats]:
    pool = EnvPool(config)
    bounds = []
    start = 0
    for tensors in pool.groups:
        bounds.append((start, start + tensors.actions.numel()))
        start += tensors.actions.numel()

    def run(pool: EnvPool, first_row: int, count: int) -> None:
        for row in range(first_row, first_row + count):
            for group, tensors in enumerate(pool.groups):
                if pipelined:
                    pool.wait(group)
                low, high = bounds[group]
                tensors.actions.copy_(table[row, low:high])
                pool.step(group)
            if not pipelined:
                for group in range(pool.num_groups):
                    pool.wait(group)
        for group in range(pool.num_groups):
            pool.wait(group)

    run(pool, 0, WARMUP_STEPS)
    pool.take_stats()
    begin = time.perf_counter()
    run(pool, WARMUP_STEPS, steps)
    seconds = time.perf_counter() - begin
    stats = pool.take_stats()
    del pool
    gc.collect()
    return seconds, stats


def agents_of(config: NitroConfig) -> int:
    return config.env.num_arenas * config.env.team_size * 2


def measure(config: NitroConfig, steps: int, pipelined: bool, repeat: int) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(1000 + repeat)
    shape = (WARMUP_STEPS + steps, agents_of(config))
    table = torch.randint(0, 90, shape, generator=generator, dtype=torch.int32)
    settings = core_settings(config)

    def rust() -> tuple[float, _core.EnvStats]:
        result = _core.rust_driven_benchmark(
            settings, table.numpy().tobytes(), WARMUP_STEPS, steps, pipelined
        )
        return result.seconds, result.stats

    def python() -> tuple[float, _core.EnvStats]:
        return python_driven(config, table, steps, pipelined)

    order = [("rust", rust), ("python", python)]
    if repeat % 2 == 1:
        order.reverse()
    results = {name: run() for name, run in order}
    rust_sps = agents_of(config) * steps / results["rust"][0]
    rows = []
    for driver in ("rust", "python"):
        seconds, stats = results[driver]
        sps = agents_of(config) * steps / seconds
        rows.append(
            {
                "driver": driver,
                "repeat": repeat,
                "first": order[0][0],
                "steps": steps,
                "seconds": round(seconds, 4),
                "agent_sps": round(sps),
                "overhead_vs_rust": round(1 - sps / rust_sps, 4),
                "physics_share": round(stats.physics_seconds / stats.step_seconds, 4),
                "env_us_per_step": round(1e6 * stats.step_seconds / stats.env_steps, 2),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("obs", "arenas", "threads", "pattern", "driver")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, members in groups.items():
        sps = [m["agent_sps"] for m in members]
        overhead = [m["overhead_vs_rust"] for m in members]
        summary.append(
            {
                **dict(zip(keys, key, strict=True)),
                "repeats": len(members),
                "agent_sps_mean": round(statistics.mean(sps)),
                "agent_sps_std": round(statistics.stdev(sps)) if len(sps) > 1 else 0,
                "overhead_mean": round(statistics.mean(overhead), 4),
                "overhead_std": round(statistics.stdev(overhead), 4) if len(overhead) > 1 else 0.0,
                "physics_share_mean": round(statistics.mean(m["physics_share"] for m in members), 4),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--meshes", type=Path, default=ROOT / "collision_meshes")
    parser.add_argument("--obs", nargs="+", default=["native", "nitro_v2"])
    parser.add_argument("--arenas", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--threads", nargs="+", type=int, default=[9, 18, 20])
    parser.add_argument("--patterns", nargs="+", default=["lockstep", "pipelined"])
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument(
        "--repeats", type=int, default=4, help="even, so each driver goes first equally often"
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="approximate measured time per run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for obs in args.obs:
        for arenas in args.arenas:
            for threads in args.threads:
                for pattern in args.patterns:
                    config = make_config(args.meshes, obs, arenas, threads, args.groups)
                    pipelined = pattern == "pipelined"
                    # A short probe sizes the runs to roughly `--seconds` each.
                    probe = measure(config, 200, pipelined, repeat=0)[0]
                    steps = max(200, int(args.seconds * 200 / probe["seconds"]))
                    for repeat in range(args.repeats):
                        for row in measure(config, steps, pipelined, repeat):
                            row = {
                                "obs": obs,
                                "arenas": arenas,
                                "threads": threads,
                                "pattern": pattern,
                                **row,
                            }
                            rows.append(row)
                            print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    write_csv(args.out, rows)
    summary = summarise(rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    for row in summary:
        print(", ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
