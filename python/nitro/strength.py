"""A strength chart against pinned anchors.

A few of a run's saves become *anchors*: kept for good (never pruned) and never played in training (the
self-play pool is a folder of its own). After a save, the new save plays a small match against every
anchor, from kickoffs in its own game (`nitro.gauntlet.gauntlet_game`), and gets a rating on a fixed
Elo-like scale where the first anchor is 0 and 400 points mean ten to one in goals: the share of goals `p`
against an anchor rated `a` says `r = a + 400 log10(p / (1 - p))`, and all anchors' goals are combined by
maximum likelihood (Bradley-Terry). An anchor's rating is the one it had when it was pinned, never
changed after, so the scale does not drift. Every count gets half a goal for and against, so a save that
wins or loses every goal still gets a finite rating. The same seed plays the same matches, so the same
saves get the same ratings. For a player's own reading of a run, never evidence for a setting.

Written to runs/<name>/strength.json; the dashboard charts it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILE = "strength.json"
ANCHOR = "anchor"  # a save folder with this file is an anchor (it also has `kept`, so it is never pruned)
SCALE = 400.0 / math.log(10.0)  # Elo points per natural-log unit of odds
_Z95 = 1.959964
_PSEUDO = 0.5  # goals added for and against in every match, so a clean sweep still has a finite rating


@dataclass(frozen=True)
class Match:
    """A save's goals against one anchor."""

    anchor_steps: int
    anchor_rating: float
    won: int
    lost: int
    draws: int = 0


def expected(rating: float, other: float) -> float:
    """The share of goals a player rated `rating` takes against one rated `other`."""
    return 1.0 / (1.0 + math.exp((other - rating) / SCALE))


def fit(matches: Sequence[Match]) -> tuple[float, float, float]:
    """The rating that best explains `matches` against anchors of known ratings, with a 95% interval
    (from the curvature of the likelihood)."""
    if not matches:
        raise ValueError("a rating needs at least one match")
    counts = [(m.anchor_rating, m.won + _PSEUDO, m.won + m.lost + 2 * _PSEUDO) for m in matches]

    def slope(r: float) -> float:  # d log-likelihood / dr, falling in r
        return sum(won - total * expected(r, a) for a, won, total in counts)

    low, high = min(a for a, _, _ in counts) - 4000.0, max(a for a, _, _ in counts) + 4000.0
    for _ in range(200):
        mid = (low + high) / 2
        if slope(mid) > 0:
            low = mid
        else:
            high = mid
    rating = (low + high) / 2
    information = sum(total * expected(rating, a) * (1 - expected(rating, a)) for a, _, total in counts)
    margin = _Z95 * SCALE / math.sqrt(information)
    return rating, rating - margin, rating + margin


def due_anchor(anchors: Sequence[int], agent_steps: int, count: int, every: int) -> bool:
    """Whether the save at `agent_steps` becomes an anchor: the first save always (when `count` > 0), then
    the first save in each later block of `every` agent steps, until there are `count`."""
    if count <= 0 or len(anchors) >= count:
        return False
    if not anchors:
        return True
    return agent_steps // every > max(anchors) // every


def match_seed(run_seed: int, save_steps: int, anchor_steps: int) -> int:
    """The match's seed: the same save and anchor always play the same games."""
    return (run_seed * 1_000_003 + save_steps * 31 + anchor_steps * 7) % (2**63)


def read(path: Path) -> dict[str, Any]:
    """The run's anchors and ratings ({"anchors": [], "ratings": []} when there are none)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"anchors": list(data.get("anchors", [])), "ratings": list(data.get("ratings", []))}


def write(path: Path, data: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
    partial.replace(path)


def rate(
    data: dict[str, Any],
    agent_steps: int,
    play: Callable[[int], tuple[int, int, int]],
    pin: bool,
) -> dict[str, Any] | None:
    """Rates the save at `agent_steps` against `data`'s anchors (`play(anchor_steps)` gives its goals for,
    against and points without one) and pins it as an anchor when `pin`; returns the rating entry (None for
    the first anchor, which has nothing to play and is 0 by definition). Updates `data` in place."""
    anchors = data["anchors"]
    entry = None
    if anchors:
        matches = [Match(a["agent_steps"], a["rating"], *play(a["agent_steps"])) for a in anchors]
        rating, low, high = fit(matches)
        entry = {
            "agent_steps": agent_steps,
            "rating": round(rating, 1) + 0.0,  # never -0
            "low": round(low, 1) + 0.0,
            "high": round(high, 1) + 0.0,
            "matches": [
                {"anchor": m.anchor_steps, "won": m.won, "lost": m.lost, "draws": m.draws} for m in matches
            ],
        }
        data["ratings"] = [r for r in data["ratings"] if r.get("agent_steps") != agent_steps] + [entry]
        data["ratings"].sort(key=lambda r: r["agent_steps"])
    if pin and all(a["agent_steps"] != agent_steps for a in anchors):
        anchors.append({"agent_steps": agent_steps, "rating": 0.0 if entry is None else entry["rating"]})
    return entry


def chart(path: Path) -> dict[str, Any] | None:
    """The dashboard's strength chart: rating against agent steps, with the 95% band and the anchors."""
    data = read(path)
    if not data["ratings"]:
        return None
    return {
        "column": "strength",
        "title": "Strength against the pinned anchors (first anchor = 0)",
        "points": [[r["agent_steps"], r["rating"]] for r in data["ratings"]],
        "band": [[r["agent_steps"], r["low"], r["high"]] for r in data["ratings"]],
        "anchors": [[a["agent_steps"], a["rating"]] for a in data["anchors"]],
    }


def rate_save(folder: Any, state: dict[str, Any], config: Any, save: Path) -> dict[str, Any] | None:
    """After the save at `save` (made from `state` under `config`): plays it against the run's anchors,
    records its rating and pins it when it is due (`config.strength`). Returns the rating entry, None when
    nothing was rated (the first anchor). Loads PyTorch and the matches only here."""
    import torch

    from nitro import eval as matches
    from nitro.gauntlet import gauntlet_game

    settings = config.strength
    path = folder.root / FILE
    data = read(path)
    # An anchor whose save was deleted by hand can no longer play; the others still rate the save.
    data["anchors"] = [
        a for a in data["anchors"] if (folder.checkpoints / f"{a['agent_steps']:012d}" / "state.pt").is_file()
    ]
    agent_steps = int(state["agent_steps"])
    pin = due_anchor(
        [a["agent_steps"] for a in data["anchors"]],
        agent_steps,
        settings.anchors,
        settings.anchor_every_steps,
    )
    me = matches.Side("this save", agent_steps, state, gauntlet_game(config))

    def play(anchor_steps: int) -> tuple[int, int, int]:
        saved = torch.load(
            folder.checkpoints / f"{anchor_steps:012d}" / "state.pt", map_location="cpu", weights_only=True
        )
        them = matches.Side("anchor", anchor_steps, saved, gauntlet_game(folder.load_settings(saved)))
        result = matches.play(
            me, them, settings.points, match_seed(config.run.seed, agent_steps, anchor_steps)
        )
        return result.blue_goals, result.orange_goals, result.no_goal

    entry = rate(data, agent_steps, play, pin)
    if pin:
        (save / "kept").write_text("kept through pruning: a strength anchor\n", encoding="utf-8")
        (save / ANCHOR).write_text("a strength anchor\n", encoding="utf-8")
    write(path, data)
    return entry


def lines(data: dict[str, Any]) -> list[str]:
    """The anchors and ratings in words (`nitro strength`)."""
    if not data["anchors"]:
        return ["No anchors yet: switch strength.anchors on; the first save after that is the first anchor."]
    out = ["Anchors (fixed points of the scale):"]
    out += [f"  {a['agent_steps']:,} agent steps: {a['rating']:.0f}" for a in data["anchors"]]
    if data["ratings"]:
        out.append("Ratings (95% range):")
        out += [
            f"  {r['agent_steps']:,} agent steps: {r['rating']:.0f} ({r['low']:.0f} to {r['high']:.0f})"
            for r in data["ratings"]
        ]
    out.append("400 more means about ten goals for every one against. For your own reading of the run.")
    return out
