"""Copying converted replays while the bot trains (`[replay_mix]`) for the dashboard's
Replays in training tab: where the copy steps' replays come from and how many there are, and the newest copy
steps' numbers from the run's metrics. Reads no replay and needs no PyTorch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nitro import replays
from nitro.config.schema import NitroConfig
from nitro.dashboard import bots
from nitro.runs import RunFolder

# The metrics columns of the copy steps (nitro.replay_mix.COLUMNS, not imported: that module needs PyTorch).
COLUMNS = ("replay_mix_steps", "replay_mix_loss", "replay_mix_agreement", "replay_mix_wait_seconds")
_LOOK_BACK = 2000  # updates searched back for the newest copy steps (read every few seconds while training)


def sizes(config: NitroConfig) -> list[int]:
    """The game sizes copied by default: each of pretrain.team_sizes, or the bot's own (as
    nitro.replay_mix)."""
    return list(config.pretrain.team_sizes) or [config.env.team_size]


def sources(config: NitroConfig) -> dict[str, Any]:
    """The folders the copy steps take converted replays from (in full, and from the project's folder when
    inside it) and how many each holds (subfolders included), with the total (a replay in two of them counted
    once) and what the store holds of every size. As nitro.replay_mix: the store's folders."""
    named = list(config.replay_mix.replays)
    store = replays.store_folder()
    places = named or [store / f"{n}v{n}" for n in sizes(config)]
    project = replays.OUT.parent  # the store's folders are shown from the project's folder on
    folders, found = [], set()
    for place in places:
        files = _converted(place)
        found |= {os.path.basename(f) for f in files}
        short = place.relative_to(project) if place.is_relative_to(project) else place
        folders.append(
            {"folder": str(place), "short": str(short), "replays": len(files), "missing": not place.is_dir()}
        )
    return {
        "named": bool(named),
        "sizes": [f"{n}v{n}" for n in sizes(config)],
        "folders": folders,
        "total": len(found),
        "store": str(store),
        "store_counts": replays.store_summary(store)["counts"] if store.is_dir() else {},
    }


def _converted(place: Path) -> set[str]:
    """The converted replays in `place` and the folders inside it, by their full path."""
    out: set[str] = set()
    for root, _, names in os.walk(place):
        out.update(os.path.join(root, name) for name in names if name.endswith(replays.STORED))
    return out


def progress(folder: RunFolder) -> dict[str, Any] | None:
    """The copy steps' numbers in the run's metrics: the newest update's, and the newest update's that took
    copy steps among the latest 2,000 (with the agent steps of each); None when the run never copied
    replays."""
    m = bots.metrics(folder)
    if not m.count or "replay_mix_steps" not in m.columns:
        return None

    def row(r: int) -> dict[str, Any]:
        return {"agent_steps": m.value("agent_steps", r)} | {
            column[len("replay_mix_") :]: m.value(column, r) for column in COLUMNS
        }

    newest = row(m.count - 1)
    recent = range(m.count - 1, max(-1, m.count - 1 - _LOOK_BACK), -1)
    last = next((row(r) for r in recent if m.value("replay_mix_loss", r) is not None), None)
    return {"newest": newest, "last": last}
