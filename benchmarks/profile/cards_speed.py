"""One bot on several graphics cards (`run.cards`): how much faster than on one card, on this
machine.

Trains a scratch copy of the bot (its settings, latest save and self-play pool; the bot itself is not touched)
for `--warmup` + `--iterations` iterations on 1 card, then on `--cards` cards, `--rounds` times each in turn
(1, N, 1, N, ...), and writes one row per counted iteration: agent steps per second over all cards, the
seconds playing and learning (card 0's; the cards wait for each other every learning step), and the agent
steps per second on the wall clock, the time between iterations (the run's own bookkeeping) counted too.
The summary gives each arm's mean and range over every counted iteration, the speed-up of the means
and the wall clock's mean. Changes nothing.
With `--bf16`, an arm sends the cards' gradients in bfloat16 (`run.cards_sum_precision`); with `--graphed`,
one replays the learning steps as CUDA graphs on every card (`run.cards_graphed_steps`); with both, also one
graphed arm summing in bfloat16.

    python -I benchmarks/profile/cards_speed.py --bot my-bot --cards 2 \
        --graphed --out my-bot_2cards.csv

`--set section.name=value` (repeatable) changes a setting in every arm, such as `env.threads=0` (all the
machine's threads, split between the cards) or `env.num_arenas=2048`; `--no-one-card` leaves the 1-card arm
out (a second sweep of variants); `--cards 1` measures one card alone (its variants with --set).

Needs a quiet machine with at least `--cards` graphics cards (Linux), the bot stopped (or another copy of it
elsewhere), and a few minutes per round.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import shutil
import statistics
import tempfile
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from nitro.config.schema import NitroConfig
from nitro.multigpu import usable_threads
from nitro.runs import RunFolder
from nitro.training import TrainingRun

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = (
    "cards",
    "sums",
    "graphed",
    "round",
    "iteration",
    "steps_per_second",
    "playing_seconds",
    "learning_seconds",
    "wall_steps_per_second",
)


def overridden(config: NitroConfig, changes: Sequence[str]) -> NitroConfig:
    """`config` with each `section.name=value` (the value as TOML writes it) applied."""
    for change in changes:
        path, _, text = change.partition("=")
        section, _, name = path.partition(".")
        part = getattr(config, section, None)
        if part is None or not hasattr(part, name):
            raise SystemExit(f"there is no setting {path}")
        value = tomllib.loads(f"v = {text}")["v"]
        config = dataclasses.replace(config, **{section: dataclasses.replace(part, **{name: value})})
    return config


def arm(
    bot: RunFolder,
    cards: int,
    sums: str,
    graphed: bool,
    warmup: int,
    iterations: int,
    scratch: Path,
    cpu: bool,
    changes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Counted iterations of the bot's training on `cards` cards (their gradients summed in `sums`, inside
    graphed steps if `graphed`), from its latest save, in a scratch copy."""
    state = bot.load_latest()
    config = overridden(bot.load_settings(state), changes)
    run_settings = dataclasses.replace(
        config.run,
        cards=cards,
        cards_sum_precision=sums,
        cards_graphed_steps=graphed,
        checkpoint_minutes=1e6,
        checkpoint_steps=0,
        device="cpu" if cpu else config.run.device,
    )
    config = dataclasses.replace(config, run=run_settings)
    folder = RunFolder.create(scratch, f"cards{cards}-{time.monotonic_ns()}", bot.settings)
    info = (bot.root / "run.json").read_text(encoding="utf-8")
    (folder.root / "run.json").write_text(info, encoding="utf-8")
    if (bot.root / "pool").is_dir():  # a copy: the scratch run adds to and prunes its own pool
        shutil.copytree(bot.root / "pool", folder.root / "pool")
    run = TrainingRun(config, folder, state)
    rows = []
    threads = torch.get_num_threads()  # joining the cards gives PyTorch a share: the next arms get them back
    try:
        for number in range(warmup + iterations):
            row = run.iteration()
            if number >= warmup:
                rows.append(
                    {
                        "cards": cards,
                        "sums": sums,
                        "graphed": graphed,
                        "iteration": number - warmup + 1,
                        **{name: row[name] for name in COLUMNS[5:]},
                    }
                )
    finally:
        if run.cards is not None:
            run.cards.stop()
        run.metrics.close()
        torch.set_num_threads(threads)
        shutil.rmtree(folder.root, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bot", required=True, help="a bot's name in runs/")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--cards", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True, help="the CSV of every counted iteration")
    parser.add_argument("--bf16", action="store_true", help="also the cards' sums in bfloat16")
    parser.add_argument("--graphed", action="store_true", help="also graphed learning steps on the cards")
    parser.add_argument(
        "--set", action="append", default=[], help="a change in every arm: section.name=value (repeatable)"
    )
    parser.add_argument("--no-one-card", action="store_true", help="leave out the 1-card arm")
    parser.add_argument(
        "--cpu", action="store_true", help="processes on the processor: a check of the script"
    )
    args = parser.parse_args()
    own = {  # each arm sets these itself, over --set, which would be thrown away
        "run.cards": "--cards",
        "run.cards_sum_precision": "--bf16",
        "run.cards_graphed_steps": "--graphed",
    }
    for change in args.set:
        flag = own.get(change.partition("=")[0].strip())
        if flag is not None:
            parser.error(f"--set {change}: every arm sets it itself; use {flag}")
    found = torch.cuda.device_count()
    if found < args.cards and not args.cpu:
        parser.error(f"--cards {args.cards} needs as many graphics cards; PyTorch sees {found}")
    bot = RunFolder(args.runs / args.bot)
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(found))
    print(f"Graphics cards: {names}. Processor threads NitroLearn may use: {usable_threads()}.")
    if args.set:
        print(f"In every arm: {', '.join(args.set)}.")
    arms = [
        *([] if args.no_one_card and args.cards > 1 else [(1, "fp32", False)]),
        (args.cards, "fp32", False),
        *([(args.cards, "bf16", False)] if args.bf16 else []),
        *([(args.cards, "fp32", True)] if args.graphed else []),
        *([(args.cards, "bf16", True)] if args.graphed and args.bf16 else []),
    ]
    if args.cards == 1:  # one card's variants (--set) alone
        arms = [(1, "fp32", False)]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="nitro-cards-speed-", ignore_cleanup_errors=True) as scratch:
        for number in range(1, args.rounds + 1):
            for cards, sums, graphed in arms:
                measured = arm(
                    bot, cards, sums, graphed, args.warmup, args.iterations, Path(scratch), args.cpu, args.set
                )
                rows += [{**row, "round": number} for row in measured]
                speeds = [float(row["steps_per_second"]) for row in measured]
                print(
                    f"round {number}, {label(cards, sums, graphed)}: {statistics.mean(speeds):,.0f} agent "
                    f"steps/s, {wall_mean(measured)} on the wall clock",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    one = None
    for cards, sums, graphed in arms:
        mine = [row for row in rows if (row["cards"], row["sums"], row["graphed"]) == (cards, sums, graphed)]
        speeds = [float(row["steps_per_second"]) for row in mine]
        play = statistics.mean(float(row["playing_seconds"]) for row in mine)
        learn = statistics.mean(float(row["learning_seconds"]) for row in mine)
        mean = statistics.mean(speeds)
        one = one or mean
        print(
            f"{label(cards, sums, graphed)}: mean {mean:,.0f} agent steps/s (range {min(speeds):,.0f}-"
            f"{max(speeds):,.0f} over {len(speeds)} iterations, {mean / one:.2f}x of the first arm); playing "
            f"{play:.3f} s, learning {learn:.3f} s an iteration; wall clock mean {wall_mean(mine)} agent "
            "steps/s"
        )
    print(f"Rows: {args.out}")


def wall_mean(rows: list[dict[str, Any]]) -> str:
    """The mean wall-clock speed of `rows` (a run's first row has none), as text."""
    walls = [float(row["wall_steps_per_second"]) for row in rows if row["wall_steps_per_second"] != ""]
    return f"{statistics.mean(walls):,.0f}" if walls else "-"


def label(cards: int, sums: str, graphed: bool) -> str:
    if cards == 1:
        return "1 card"
    return f"{cards} cards, sums in {sums}" + (", graphed steps" if graphed else "")


if __name__ == "__main__":
    main()
