"""`nitro reward-check`: a bot as it was last saved plays, learning nothing (its policy and
value networks stay exactly as saved, as with a learning rate of 0 for both), and says what each of the
rewards tried would have paid it: each reward's share of all the reward paid (by size: a penalty counts too),
what it pays a step and a minute of play, and its value before the weight. For balancing the rewards of a bot
copied from replays before it trains, instead of guessing them. Nothing of the run changes: the games are
played in a folder of their own, removed at the end, and only the result is kept, in the run's
`reward_checks` folder.

The bot plays itself in every game (no past versions).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nitro.config import NitroConfig
from nitro.config.loader import parse_toml
from nitro.runs import RunFolder

CHECKS = "reward_checks"  # in the run's folder: the results, newest last by name
DEFAULT_STEPS = 1_000_000


class CheckError(ValueError):
    """The check cannot run; the message says why in plain English."""


def trial_settings(folder: RunFolder, rewards_file: Path | None) -> str:
    """The run's settings file with the rewards of `rewards_file` (a reward set or a settings file: only its
    `[[rewards]]` are taken) in place of the run's own; the run's own settings when None."""
    text = folder.settings.read_text(encoding="utf-8-sig")
    if rewards_file is None:
        return text
    from nitro.dashboard import rewards as editor

    views, listed = editor.read(rewards_file.read_text(encoding="utf-8-sig"))
    if not listed:
        raise CheckError(f"{rewards_file.name} lists no [[rewards]] to try")
    return editor.written(text, [{"from": None, "view": view} for view in views])


def check(
    folder: RunFolder,
    settings_text: str,
    agent_steps: int = DEFAULT_STEPS,
    say: Callable[[str], None] = print,
    out: Path | None = None,
) -> dict[str, Any]:
    """Plays the run's newest save for at least `agent_steps` agent steps with the rewards of `settings_text`
    (the run's settings as they would be written), learning nothing; returns the result and writes it as JSON
    to `out` (default: a new file in the run's reward_checks folder), with progress in it while playing."""
    from nitro.training import TrainingRun

    state = folder.load_latest()
    if state is None:
        raise CheckError(
            f"'{folder.root.name}' has no save yet. A bot that copies replays saves every "
            "pretrain.save_every replays and when the copy is ready (press Start, or nitro resume, once); "
            "any other bot saves a few minutes into training"
        )
    if "pretrained" in state:
        raise CheckError(f"'{folder.root.name}' is a copy made by nitro pretrain; start a run from it first")
    settings_dir = Path(json.loads(folder.info.read_text(encoding="utf-8"))["settings_dir"])
    # Mistakes in the rewards tried are named as theirs, not as the run's settings file's.
    parse_toml(settings_text, NitroConfig, file=Path("the rewards tried"), base_dir=settings_dir)
    checks = folder.root / CHECKS
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out or checks / f"{stamp}.json"
    work = RunFolder(checks / f"playing-{stamp}")
    work.root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(work.log, encoding="utf-8")  # setting warnings etc.: not the run's log
    logger = logging.getLogger("nitro")
    logger.addHandler(handler)
    run = None
    try:
        shutil.copyfile(folder.info, work.info)
        work.settings.write_text(settings_text, encoding="utf-8")
        config = work.load_settings(state)
        config = dataclasses.replace(
            config,
            selfplay=dataclasses.replace(config.selfplay, enabled=False),  # the bot plays itself
            wandb=dataclasses.replace(config.wandb, enabled=False),
        )
        state = {key: value for key, value in state.items() if key != "selfplay"}
        saved_steps = int(state["agent_steps"])
        say(f"'{folder.root.name}' as saved at {saved_steps:,} steps plays {config.env.num_arenas} games")
        run = TrainingRun(config, work, state)
        trainer = run.trainer
        per_round = trainer.steps * trainer.rollout.agents
        rounds = max(1, math.ceil(agent_steps / per_round))
        labels = run.labels
        raw = [0.0] * len(labels)
        paid = [0.0] * len(labels)
        size = [0.0] * len(labels)
        fired = [0] * len(labels)
        counted = episodes = 0
        began = time.perf_counter()
        for done in range(1, rounds + 1):
            trainer.collect()  # plays only: no learning step, so neither network changes
            env = run.envs.take_stats()
            for i in range(len(labels)):
                raw[i] += env.raw[i]
                paid[i] += env.weighted[i]
                size[i] += env.weighted_abs[i]
            fired = [a + b for a, b in zip(fired, env.fired, strict=True)]
            counted += env.agent_steps
            episodes += len(env.episodes)
            say(f"  {done * 100 // rounds}% ({done * per_round:,} of {rounds * per_round:,} agent steps)")
            if done < rounds:
                _write(out, {"finished": False, "done": done, "rounds": rounds})
        result = summary(
            labels,
            raw,
            paid,
            size,
            counted,
            tick_skip=config.env.tick_skip,
            fired=fired,
            extra={
                "bot": folder.root.name,
                "saved_steps": saved_steps,
                "episodes": episodes,
                "seconds": time.perf_counter() - began,
                "when": time.time(),
                "rewards": [reward.name for reward in config.active_rewards],
            },
        )
    except BaseException:
        out.unlink(missing_ok=True)  # an unfinished check leaves no result behind
        raise
    finally:
        if run is not None:
            run.metrics.close()  # Windows removes no open file
        logger.removeHandler(handler)
        handler.close()
        shutil.rmtree(work.root, ignore_errors=True)
    _write(out, result)
    return result


def summary(
    labels: list[str],
    raw: list[float],
    paid: list[float],
    size: list[float],
    agent_steps: int,
    tick_skip: int,
    extra: dict[str, Any] | None = None,
    fired: list[int] | None = None,
) -> dict[str, Any]:
    """Each reward's share of all the reward paid (by size), largest first, with what it pays a step and a
    minute of one car's play, its value before the weight, and how often it fired (the share of steps it paid
    on before its weight, and the times a minute; None without counts); and what to look at."""
    steps = max(agent_steps, 1)
    total = sum(size)
    minute = 60 * 120 / tick_skip  # one car's decisions in a minute of play
    ranked: list[dict[str, Any]] = [
        {
            "label": label,
            "share": size[i] / total if total else 0.0,
            "paid_per_step": paid[i] / steps,
            "size_per_step": size[i] / steps,
            "paid_per_minute": paid[i] / steps * minute,
            "raw_per_step": raw[i] / steps,
            "fire_rate": fired[i] / steps if fired is not None else None,
            "fires_per_minute": fired[i] / steps * minute if fired is not None else None,
        }
        for i, label in enumerate(labels)
    ]
    ranked.sort(key=lambda item: float(item["share"]), reverse=True)
    return {
        "finished": True,
        **(extra or {}),
        "agent_steps": agent_steps,
        "minutes_of_play": agent_steps / minute,
        "reward_per_step": sum(paid) / steps,
        "ranked": ranked,
        "alerts": alerts(ranked),
    }


def alerts(ranked: list[dict[str, Any]]) -> list[dict[str, str]]:
    """What to look at in a check: one reward that would drown the others out (the training's farming
    warning), and rewards that paid nothing."""
    from nitro.dashboard.rewards import DOMINANT_SHARE, title

    found = []
    paying = [item for item in ranked if item["share"] > 0]
    if len(paying) > 1 and paying[0]["share"] > DOMINANT_SHARE:
        top = paying[0]
        found.append(
            {
                "label": top["label"],
                "level": "warning",
                "text": f"'{title(top['label'])}' would make up {top['share']:.0%} of all reward paid: the "
                "others would hardly count. Consider lowering its weight.",
            }
        )
    for item in ranked:
        if item["share"] == 0:
            found.append(
                {
                    "label": item["label"],
                    "level": "watch",
                    "text": f"'{title(item['label'])}' paid nothing: the bot never did what it pays for in "
                    "these games (rare, or not learned yet); more steps may show it.",
                }
            )
    return found


def table(result: dict[str, Any]) -> list[str]:
    """The result as lines for the console."""
    lines = [
        f"What each reward would have paid '{result['bot']}' (as saved at {result['saved_steps']:,} steps, "
        f"learning nothing) over {result['agent_steps']:,} agent steps ({result['minutes_of_play']:,.0f} "
        "minutes of one car's play):",
        f"  {'Reward':<32} {'Share':>7} {'A step':>11} {'A minute':>10} {'Before weight':>14} "
        f"{'Fires a minute':>15}",
    ]
    for item in result["ranked"]:
        fires = item.get("fires_per_minute")  # None when no counts were given
        lines.append(
            f"  {item['label'][:32]:<32} {item['share']:>7.1%} {item['paid_per_step']:>11.5f} "
            f"{item['paid_per_minute']:>10.3f} {item['raw_per_step']:>14.5f} "
            + (f"{fires:>15.1f}" if fires is not None else f"{'-':>15}")
        )
    lines.append(f"  All rewards: {result['reward_per_step']:.5f} a step.")
    lines.append(
        "  Share: each reward's part of all the reward paid, counted by size, so a penalty counts as much as "
        "a payment of the same size."
    )
    lines += [f"  Note: {alert['text']}" for alert in result["alerts"]]
    return lines


def newest(folder: RunFolder) -> dict[str, Any] | None:
    """The newest result kept in the run's reward_checks folder (unfinished while a check plays), or None."""
    found = sorted((folder.root / CHECKS).glob("*.json"))
    for path in reversed(found):
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["file"] = path.name
        return data
    return None


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
    partial.replace(path)
