"""The gauntlet's report card as the run folder keeps it : read without PyTorch, for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT = "gauntlet.json"  # in the run folder: one entry per save played


def verdict(entry: dict[str, Any]) -> str:
    """One line of the report card: 'beats chaser 97% (95-99)'."""
    name = entry["opponent"]
    if entry["share"] is None:
        return f"{name}: nobody scored in {entry['points']} points"
    low, high, share = entry["low"], entry["high"], entry["share"]
    word = "beats" if low > 0.5 else "loses to" if high < 0.5 else "even with"
    return (
        f"{word} {name} {share:.0%} ({low * 100:.0f}-{high * 100:.0f}) of {entry['won'] + entry['lost']} "
        f"goals; {entry['draws']} of {entry['points']} points without one"
    )


def read_report(path: Path) -> list[dict[str, Any]]:
    """The run's cards, oldest save first ([] when none); read by the dashboard without PyTorch."""
    try:
        cards = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [c for c in cards if isinstance(c, dict)] if isinstance(cards, list) else []


def newest(path: Path) -> dict[str, Any] | None:
    """The newest save's card with its lines in words, or None."""
    cards = read_report(path)
    if not cards:
        return None
    card = cards[-1]
    return {
        "agent_steps": card.get("agent_steps"),
        "when": card.get("when"),
        "lines": [{**line, "text": verdict(line)} for line in card.get("lines", [])],
    }
