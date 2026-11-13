"""Self-play against frozen past versions of the bot.

A share of the arenas has its Orange team driven by one of a few *active* past versions (snapshots of the
policy saved to the run's `pool/` folder); the rest play the current bot against itself. The arenas of
each active slot form a contiguous range, so inside every group they are a contiguous slice of rows: the
inference graph runs the snapshot's network on that slice and lets a boolean tensor choose, row by row,
whose action drives the car. Slots change only at checkpoints, where every episode ends anyway.

Only rows the learner drives are trained on (`learner_rows`). Every goal in a slot's arenas is one rated
game between the learner and that snapshot (`nitro.ratings`); the ratings pick the next active slots.

With several game sizes in one run an arena's rows are its own car count, and a block mixes sizes:
the past versions drive Orange in every size, and one rating covers them all.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

from nitro import _core
from nitro.config.schema import NitroConfig
from nitro.model import ActorCritic, fp16_inference_copy
from nitro.ratings import BETA, Rating, draw, matchmaking_weights, update, win_probability
from nitro.runs import UNREADABLE, unreadable_reason

log = logging.getLogger("nitro")
_SNAPSHOT = "{steps:012d}.pt"


_LOG_COLUMNS = ["agent_steps", "snapshot_steps", "learner_scored", "learner_mu", "snapshot_mu"]


@dataclass(frozen=True)
class Block:
    """One active slot's arenas: the global arena range [start, stop)."""

    slot: int
    start: int
    stop: int


def arena_blocks(num_arenas: int, past_share: float, slots: int) -> list[Block]:
    """Splits the last `past_share` of the arenas into `slots` equal, contiguous blocks."""
    past = round(num_arenas * past_share)
    first = num_arenas - past
    blocks = []
    for slot in range(slots):
        start = first + past * slot // slots
        stop = first + past * (slot + 1) // slots
        if stop > start:
            blocks.append(Block(slot, start, stop))
    return blocks


def group_rows(block: Block, groups: int, group: int, cars_per_env: int | Sequence[int]) -> slice:
    """The rows of `group` that belong to `block` (arena i sits in group i % groups at position
    i // groups); `cars_per_env`: one car count for every arena, or each arena's."""
    first = max(0, -((block.start - group) // -groups))  # ceil((start - group) / groups)
    last = max(0, -((block.stop - group) // -groups))
    if isinstance(cars_per_env, int):
        return slice(first * cars_per_env, last * cars_per_env)
    in_group = cars_per_env[group::groups]
    return slice(sum(in_group[:first]), sum(in_group[:last]))


@dataclass(frozen=True)
class OpponentBlock:
    """What a group's inference needs for one slot: its rows, which of them the opponent drives, and
    the opponent's network."""

    rows: slice
    select: torch.Tensor  # bool [rows]; True where the snapshot's action replaces the learner's
    model: ActorCritic


@dataclass(frozen=True)
class GameStats:
    games: int
    wins: int  # goals the learner scored against past versions
    matchmaking: float  # mean |p - 0.5| of those games' predicted win chance (0 = perfectly even)
    even_games: int = 0  # of those, the games an even setter started
    even_wins: int = 0


class Pool:
    """The snapshots on disk, capped by count and bytes."""

    def __init__(self, folder: Path, size: int, disk_bytes: int) -> None:
        self.folder = folder
        self.size = size
        self.disk_bytes = disk_bytes

    def snapshots(self) -> list[int]:
        if not self.folder.is_dir():
            return []
        return sorted(int(p.stem) for p in self.folder.glob("*.pt") if p.stem.isdigit())

    def path(self, steps: int) -> Path:
        return self.folder / _SNAPSHOT.format(steps=steps)

    def save(self, steps: int, weights: dict[str, torch.Tensor]) -> None:
        """Written to a temporary file, on the disk (fsync) before it is renamed into place, as a run's
        saves are: a power cut leaves the old file or the whole new one, never an empty one."""
        self.folder.mkdir(parents=True, exist_ok=True)
        partial = self.path(steps).with_suffix(".partial")
        try:
            with partial.open("wb") as file:
                torch.save({k: v.detach().cpu() for k, v in weights.items()}, file)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(self.path(steps))

    def load(self, steps: int) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = torch.load(self.path(steps), map_location="cpu", weights_only=True)
        return weights

    def total_bytes(self) -> int:
        return sum(self.path(s).stat().st_size for s in self.snapshots())

    def prune(self, ratings: dict[int, Rating], keep: set[int]) -> list[int]:
        """Removes snapshots while over the count or byte cap: the one rated closest to its neighbours goes
        first, never the newest or an active one (`keep`), so the pool keeps a spread over history."""
        removed: list[int] = []
        while True:
            present = self.snapshots()
            if len(present) <= self.size and self.total_bytes() <= self.disk_bytes:
                return removed
            candidates = [s for s in present if s not in keep and s != present[-1]]
            if len(candidates) < 1 or len(present) < 2:
                return removed
            by_mu = sorted(present, key=lambda s: ratings.get(s, Rating()).mu)
            mus = [ratings.get(s, Rating()).mu for s in by_mu]
            # Distance of each snapshot's rating to its nearest neighbour's.
            gaps = {
                steps: min(
                    [abs(mus[i] - mus[j]) for j in (i - 1, i + 1) if 0 <= j < len(mus)], default=math.inf
                )
                for i, steps in enumerate(by_mu)
            }
            victim = min(candidates, key=lambda s: gaps[s])
            self.path(victim).unlink()
            removed.append(victim)


class SelfPlay:
    """The opponent slots, their networks and selection tensors, the pool and the ratings of one run."""

    def __init__(
        self,
        config: NitroConfig,
        pool_folder: Path | None,
        model: ActorCritic,
        device: torch.device,
        groups: int,
        cars_per_env: int | Sequence[int],
        group_agents: list[int],
    ) -> None:
        """`cars_per_env`: every arena's car count, or one count for all."""
        self.config = config.selfplay
        self.enabled = self.config.enabled and pool_folder is not None
        self.seed = config.run.seed
        self.device = device
        self.model = model
        self.fp16 = config.run.precision == "fp16" and device.type == "cuda"
        self.groups = groups
        arenas = config.env.num_arenas
        self.arena_cars = [cars_per_env] * arenas if isinstance(cars_per_env, int) else list(cars_per_env)
        self.group_agents = group_agents
        self.learner = Rating()
        self.ratings: dict[int, Rating] = {}
        self.games: dict[int, int] = {}
        self.active: list[int | None] = [None] * (self.config.active_opponents if self.enabled else 0)
        self.next_snapshot_steps = self.config.snapshot_steps
        self.blocks = (
            arena_blocks(config.env.num_arenas, self.config.past_share, self.config.active_opponents)
            if self.enabled
            else []
        )
        self.pool = Pool(pool_folder or Path(), self.config.pool_size, self.config.pool_disk_mb * 2**20)
        self.opponents = [self._frozen_copy() for _ in self.active]
        # Per group, per block: the rows and the (in-place updated) selection the graphs read.
        self._selects: dict[tuple[int, int], torch.Tensor] = {}
        for group in range(groups):
            for block in self.blocks:
                rows = group_rows(block, groups, group, self.arena_cars)
                self._selects[group, block.slot] = torch.zeros(
                    rows.stop - rows.start, dtype=torch.bool, device=device
                )
        self._learner_rows = torch.ones(sum(group_agents), dtype=torch.bool, device=device)
        self._log = pool_folder / "games.csv" if self.enabled and pool_folder is not None else None
        self._log_file: TextIO | None = None  # games.csv, open for appending from the first game recorded
        self._pool_size: int | None = None  # the pool's snapshots, counted again after this run changes it
        # What load_state_dict found when the pool does not match the save: warnings and info lines.
        self.resume_notes: list[str] = []
        self.resume_info: list[str] = []
        self.unreadable: set[int] = set()  # snapshots that did not read: said once, never picked again

    def _frozen_copy(self) -> ActorCritic:
        if self.fp16:  # weights held in FP16 as autocast would cast them each step: the same answers
            return fp16_inference_copy(self.model)
        copy = self.model.fresh_copy().to(self.device)
        copy.load_state_dict(self.model.state_dict())
        copy.eval()
        for p in copy.parameters():
            p.requires_grad_(False)
        return copy

    # --- what the trainer and the inference graphs need -------------------------------------------

    def group_opponents(self, group: int) -> list[OpponentBlock]:
        return [
            OpponentBlock(
                group_rows(block, self.groups, group, self.arena_cars),
                self._selects[group, block.slot],
                self.opponents[block.slot],
            )
            for block in self.blocks
        ]

    @property
    def learner_rows(self) -> torch.Tensor | None:
        """Bool [all agents]: rows the learner drives (None when every row is the learner's)."""
        return self._learner_rows if self.enabled else None

    def slot_of_arena(self, arena: int) -> int | None:
        for block in self.blocks:
            if block.start <= arena < block.stop:
                return block.slot
        return None

    # --- rotation at checkpoints --------------------------------------------------------------------

    def rotate(self, rng: random.Random) -> None:
        """Picks the active snapshots (fair matches by rating) and loads their weights in place."""
        if not self.enabled:
            return
        available = [s for s in self.pool.snapshots() if s in self.ratings and s not in self.unreadable]
        weights = matchmaking_weights(
            self.learner, [self.ratings[s] for s in available], self.config.matchmaking
        )
        chosen = [available[i] for i in draw(weights, len(self.active), rng)]
        self.active = [chosen[i] if i < len(chosen) else None for i in range(len(self.active))]
        self._apply_active()

    def _apply_active(self) -> None:
        for slot, steps in enumerate(self.active):
            if steps is None:
                continue
            weights = self._read(steps)
            if weights is None:
                self.active[slot] = None  # the learner plays itself in this slot's arenas until the next draw
            else:
                self.opponents[slot].load_policy_state_dict(weights)
        offset = 0
        self._learner_rows.fill_(True)
        for group in range(self.groups):
            for block in self.blocks:
                rows = group_rows(block, self.groups, group, self.arena_cars)
                # Cars alternate Blue, Orange in every arena and each arena has an even count.
                orange = torch.arange(rows.start, rows.stop, device=self.device) % 2 == 1
                driven = orange & (self.active[block.slot] is not None)
                self._selects[group, block.slot].copy_(driven)
                self._learner_rows[offset + rows.start : offset + rows.stop] &= ~driven
            offset += self.group_agents[group]

    def _read(self, steps: int) -> dict[str, torch.Tensor] | None:
        """A snapshot's weights; None, said once, for a file that does not read (cut short by a crash, a
        power cut or a full disk), which is left where it is and never picked again."""
        try:
            return self.pool.load(steps)
        except UNREADABLE as err:
            self.unreadable.add(steps)
            log.warning(
                "Self-play: the past version at %s agent steps (%s) cannot be read: %s. It was left where it "
                "is and is not played against; the run goes on with the others.",
                f"{steps:,}",
                self.pool.path(steps),
                unreadable_reason(err),
            )
            return None

    # --- games and ratings --------------------------------------------------------------------------

    def counted_cars(self) -> dict[int, list[bool]]:
        """Per arena of an occupied slot, which cars are the learner's (Blue) rather than the frozen
        opponent's (Orange): for `EnvPool.set_counted`, so the statistics are the learner's alone. Arenas
        of empty slots and pure self-play arenas are left at the default (every car counted)."""
        counted: dict[int, list[bool]] = {}
        for block in self.blocks:
            for arena in range(block.start, block.stop):
                cars = self.arena_cars[arena]
                empty = self.active[block.slot] is None
                counted[arena] = [empty or car % 2 == 0 for car in range(cars)]
        return counted

    def record(
        self, episodes: list[_core.EpisodeRecord], agent_steps: int, even: Sequence[bool] | None = None
    ) -> GameStats:
        """Rates every goal in an active slot's arenas as one game (Blue = the learner); `agent_steps` is
        the run's step count, written with each game so a resume can drop games recorded after its save.
        `even`: per setter, whether it starts every car alike; those games are also counted apart (a
        setter past its end counts as uneven). Ratings and games.csv are the same either way."""
        if not self.enabled:
            return GameStats(0, 0, math.nan)
        rows: list[tuple[int, int]] = []
        evenness = []
        even_games = even_wins = 0
        for record in episodes:
            if record.car != 0 or record.goal == 0:
                continue
            slot = self.slot_of_arena(record.env)
            if slot is None or (steps := self.active[slot]) is None:
                continue
            score = 1.0 if record.goal > 0 else 0.0
            evenness.append(abs(win_probability(self.learner, self.ratings[steps]) - 0.5))
            self.learner, self.ratings[steps] = update(
                self.learner, self.ratings[steps], score, min_sigma=BETA
            )
            self.games[steps] = self.games.get(steps, 0) + 1
            rows.append((steps, int(score)))
            if even is not None and record.setter < len(even) and even[record.setter]:
                even_games += 1
                even_wins += int(score)
        if rows and self._log is not None:
            if self._log_file is None:
                new = not self._log.exists()
                self._log_file = self._log.open("a", newline="", encoding="utf-8")
                if new:
                    csv.writer(self._log_file).writerow(_LOG_COLUMNS)
            writer = csv.writer(self._log_file)
            for steps, scored in rows:
                learner_mu, snapshot_mu = round(self.learner.mu, 4), round(self.ratings[steps].mu, 4)
                writer.writerow([agent_steps, steps, scored, learner_mu, snapshot_mu])
            self._log_file.flush()  # whole rows in the file after every call, as when it was closed each time
        return GameStats(
            len(rows),
            sum(s for _, s in rows),
            statistics.mean(evenness) if evenness else math.nan,
            even_games,
            even_wins,
        )

    def close(self) -> None:
        """Closes games.csv; the next game recorded opens it again."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def truncate_log(self, agent_steps: int) -> None:
        """Drops the games recorded after `agent_steps` (a resumed run plays those iterations again)."""
        self.close()
        if self._log is None or not self._log.exists():
            return
        with self._log.open(encoding="utf-8", newline="") as file:
            rows = list(csv.reader(file))
        if not rows or rows[0] != _LOG_COLUMNS:
            return  # a file in another layout is left alone
        kept = [rows[0], *(row for row in rows[1:] if int(row[0]) <= agent_steps)]
        if len(kept) != len(rows):
            with self._log.open("w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(kept)

    def maybe_snapshot(self, agent_steps: int) -> bool:
        """Adds the current policy to the pool when due; the new snapshot starts at the learner's rating."""
        if not self.enabled or agent_steps < self.next_snapshot_steps:
            return False
        self.pool.save(agent_steps, self.model.policy_state_dict())
        self._pool_size = None
        self.ratings[agent_steps] = self.learner
        while self.next_snapshot_steps <= agent_steps:
            self.next_snapshot_steps += self.config.snapshot_steps
        keep = {s for s in self.active if s is not None} | {agent_steps}
        for removed in self.pool.prune(self.ratings, keep):
            self.ratings.pop(removed, None)
            self.games.pop(removed, None)
        return True

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self._pool_size is None:  # only maybe_snapshot changes the pool while a run trains
            self._pool_size = len(self.pool.snapshots())
        return {
            "rating_mu": self.learner.mu,
            "rating_sigma": self.learner.sigma,
            "pool_size": float(self._pool_size),
        }

    # --- checkpoints --------------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "learner": [self.learner.mu, self.learner.sigma],
            "ratings": {str(s): [r.mu, r.sigma] for s, r in self.ratings.items()},
            "games": {str(s): n for s, n in self.games.items()},
            "active": [-1 if s is None else s for s in self.active],
            "next_snapshot_steps": self.next_snapshot_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learner = Rating(*state["learner"])
        self.ratings = {int(s): Rating(*r) for s, r in state["ratings"].items()}
        self.games = {int(s): n for s, n in state["games"].items()}
        self.next_snapshot_steps = state["next_snapshot_steps"]
        present = set(self.pool.snapshots()) if self.enabled else set()
        active = [None if s < 0 or s not in present else s for s in state["active"]]
        self.active = [active[i] if i < len(active) else None for i in range(len(self.active))]
        self._apply_active()
        self.resume_notes, self.resume_info = (
            self._pool_notes(present, state["next_snapshot_steps"]) if self.enabled else ([], [])
        )

    def _pool_notes(self, present: set[int], next_snapshot: int) -> tuple[list[str], list[str]]:
        """Where the pool on disk and the save's ratings disagree, as warnings and info lines. It only
        says so: the ratings and the draws stay as they are, so a version copied back plays at the next draw.
        Snapshots from `next_snapshot` on were taken after the save (a run stopped without saving), and each
        may have pruned one rated version: those are no news."""
        rated = set(self.ratings)
        where = f"{self.pool.folder.parent.name}/{self.pool.folder.name}/"
        missing = sorted(rated - present)
        later = sorted(s for s in present - rated if s >= next_snapshot)
        unrated = sorted(s for s in present - rated if s < next_snapshot)
        notes, info = [], []
        are, they, them = ("is", "It", "it") if len(missing) == 1 else ("are", "They", "them")
        if missing and later and len(missing) <= len(later):
            info.append(
                f"Self-play: this save rates {_versions(missing)} no longer in {where}: the run went on past "
                f"this save ({where} holds {_versions(later, '')} taken after it) and pruned {them} then."
            )
        elif missing and not present:
            notes.append(
                f"Self-play: {where} holds no past versions, but this save rates {_versions(missing)} (moved "
                "without nitro pack?). The bot plays itself in every arena until its next snapshot joins the "
                f"pool at {next_snapshot / 1e9:.2f}B steps, or until {they.lower()} {are} copied back from "
                "the other computer's pool folder; nothing else changes."
            )
        elif missing:
            notes.append(
                f"Self-play: this save rates {_versions(missing)} that {are} not in {where} (moved without "
                f"nitro pack?). {they} cannot play until copied back from the other computer's pool folder; "
                "nothing else changes."
            )
        if unrated:
            notes.append(
                f"Self-play: {where} holds {_versions(unrated, '')} this save never rated (from another copy "
                f"of the run?); {'it is' if len(unrated) == 1 else 'they are'} never picked."
            )
        if later:
            info.append(
                f"Self-play: {where} holds {_versions(later, '')} taken after this save (the run went on "
                f"past it); {'it is' if len(later) == 1 else 'they are'} never picked, and the run takes its "
                "own again as it passes those steps."
            )
        return notes, info


def _versions(steps: Sequence[int], kind: str = "past ") -> str:
    """'2 past versions (0.06B-0.08B steps)': how many snapshots and where they lie in the run's steps."""
    low, high = f"{min(steps) / 1e9:.2f}B", f"{max(steps) / 1e9:.2f}B"
    span = low if low == high else f"{low}-{high}"
    return f"{len(steps)} {kind}version{'s' if len(steps) != 1 else ''} ({span} steps)"
