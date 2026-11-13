"""Copying converted replays while a bot trains (`[replay_mix]`).

After each PPO update the learner takes a few more steps that copy the players in converted replays (the
replay store), with the replay copy's loss over the actions, and an Adam optimizer and
learning rate of their own on the policy and any shared trunk, never the value network. There are
share / (1 - share) of them per policy step of the update, the fraction carried over. Worker threads make
the replays' samples ahead of use; which replays and rows are used depends only on the seed and on counters
kept in the saves, so a run with it repeats and resumes exactly.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nitro import _core, pretrain, replays
from nitro.config.schema import NitroConfig
from nitro.health import TrainingProblem
from nitro.learner import _ADAM_IMPLEMENTATION, _FP16_INITIAL_SCALE, _NEVER, LearnStats
from nitro.model import ActorCritic, observation_room

if TYPE_CHECKING:
    from nitro.multigpu import Cards

log = logging.getLogger("nitro")

WINDOW = 16  # replays drawn from at once, so each copy step mixes many games
WORKERS = 2  # threads making replays' samples ahead of use (the core and zstd work outside Python's lock)
AHEAD = 6  # replays made ready beyond the newest in use
CLIP = 0.5  # gradient norm, as the replay copy's
SMALLEST = 16  # decisions a step: below this, a graphics card out of memory stops the copy steps instead
_SEED = 0x5EED_C0B1  # the row draws' stream, apart from training's


@dataclass(frozen=True)
class CopyStats:
    """An iteration's copy steps: how many, their mean loss and first-choice agreement on the rows learned
    (measured before each step, so mostly on decisions not learned from yet), and seconds spent waiting for
    replays to be made ready."""

    steps: int
    loss: float
    agreement: float
    wait_seconds: float
    # With replay_mix.kickoff_only_seconds: the decisions the kickoff windows kept and left out of the
    # replays taken in during the iteration; None when the window is off.
    rows_kept: int | None = None
    rows_left_out: int | None = None


COLUMNS = ("replay_mix_steps", "replay_mix_loss", "replay_mix_agreement", "replay_mix_wait_seconds")
KICKOFF_COLUMNS = ("replay_mix_rows_kept", "replay_mix_rows_left_out")  # with kickoff_only_seconds only


def columns(stats: CopyStats | None) -> dict[str, Any]:
    """The metrics columns of an iteration's copy steps (blank without any)."""
    kickoffs = (
        {"replay_mix_rows_kept": stats.rows_kept, "replay_mix_rows_left_out": stats.rows_left_out}
        if stats is not None and stats.rows_kept is not None
        else {}
    )
    if stats is None or stats.steps == 0:
        return {**dict.fromkeys(COLUMNS, ""), "replay_mix_steps": "" if stats is None else 0, **kickoffs}
    return {
        "replay_mix_steps": stats.steps,
        "replay_mix_loss": stats.loss,
        "replay_mix_agreement": stats.agreement,
        "replay_mix_wait_seconds": stats.wait_seconds,
        **kickoffs,
    }


def sample_settings(config: NitroConfig) -> str:
    """The core settings the copy steps' samples are made under: the replay copy's for the store's replicar
    rows, with the kickoff window when replay_mix.kickoff_only_seconds is above 0 (absent when off, so
    the samples are those made before it)."""
    settings = pretrain.copy_settings(config, replicar_rows=True)
    seconds = config.replay_mix.kickoff_only_seconds
    if seconds <= 0:
        return settings
    return json.dumps(json.loads(settings) | {"kickoff_only_seconds": seconds})


def team_sizes(config: NitroConfig) -> tuple[int, ...]:
    """The game sizes whose converted replays are copied by default: the copy's (pretrain.team_sizes), or the
    run's own."""
    return tuple(config.pretrain.team_sizes) or (config.env.team_size,)


def folders(config: NitroConfig) -> list[Path]:
    """The folders the replays come from: those named, or the store's folder of each game size."""
    if config.replay_mix.replays:
        return list(config.replay_mix.replays)
    return [replays.store_folder() / f"{n}v{n}" for n in team_sizes(config)]


def files_in(places: list[Path]) -> list[Path]:
    """The converted replays in `places` (subfolders included), in a fixed order; a replay in two of them is
    taken from the first."""
    found: dict[str, Path] = {}
    for place in places:
        if place.is_dir():
            for path in sorted(place.rglob(f"*{replays.STORED}")):
                found.setdefault(path.name, path)
    return sorted(found.values())


def shown(path: Path) -> str:
    """A folder as messages show it: from the NitroLearn folder when inside it, else its last 4 parts (the
    full path holds the user name), so a path that moved shows where it points now."""
    path = Path(os.path.normpath(path))
    if path.is_relative_to(replays.ROOT):
        return str(Path(replays.ROOT.name, path.relative_to(replays.ROOT)))
    return str(Path("...", *path.parts[-4:])) if len(path.parts) > 4 else str(path)


def _where(places: list[Path]) -> str:
    return " and ".join(shown(p) for p in places)


def _size_of(path: Path) -> int:
    """The game size a store folder's name gives (1v1 -> 1); 0 when it gives none."""
    found = re.fullmatch(r"(\d)v\1", path.parent.name)
    return int(found.group(1)) if found else 0


def _prepare(job: replays.Job) -> replays.Prepared:
    """One converted replay's samples, or why not; a replay gone since it was listed is never converted."""
    if not job.kept.is_file():
        return replays.Prepared(job.index, job.replay.stem, problem="the file is gone")
    return replays.prepare(job)


class ReplayMix:
    """The copy steps of a training run and everything they keep: their optimizer and loss scaler, the
    replays in use and the draws' random stream (`state_dict`)."""

    def __init__(
        self,
        config: NitroConfig,
        model: ActorCritic,
        obs_size: int,
        device: torch.device,
        fp16: bool,
        state: dict[str, Any] | None = None,
        moved: bool = False,
    ) -> None:
        """`moved`: the bot was unpacked here (run.json's 'unpacked'); packs carry no folders."""
        chosen = config.replay_mix
        self.model, self.obs_size, self.device, self.fp16 = model, obs_size, device, fp16
        self.share, self.minibatch, self.lr = chosen.share, chosen.minibatch, chosen.lr
        self.chosen_minibatch = chosen.minibatch  # minibatch is halved when the graphics card runs out
        self.seed = config.run.seed
        self.actions = _core.num_actions()
        groups = model.parameter_groups()
        self.params = [p for name in ("shared", "policy") for p in groups.get(name, [])]
        self.optimizer = self._adam()
        self.scaler = self._scaler()
        self.generator = torch.Generator().manual_seed(self.seed ^ _SEED)
        self.next = 0  # the next place in the cycle of replays to take in
        self.window: list[int] = []  # the places of the replays drawn from
        self.budget = 0.0  # decisions still to draw before the next replay comes in
        self.credit = 0.0  # a part of a copy step owed from earlier updates
        self.steps = 0  # copy steps taken in all
        self.notices: list[str] = []  # plain-English events for the run's log and console
        self.settings = sample_settings(config)  # the store holds replicar's rows
        self.kickoff_only_seconds = chosen.kickoff_only_seconds
        self.kickoff_only = chosen.kickoff_only_seconds > 0
        self._rows_kept = self._rows_left_out = 0  # by the kickoff window, in replays taken in since learn
        if state:
            self._load(state)
        self.places = folders(config)
        self.named = bool(chosen.replays)  # folders the settings name, not the store's
        self.moved = moved
        self.sizes = "/".join(f"{n}v{n}" for n in team_sizes(config))
        # Game sizes whose observations fit the bot's: its own, or up to the places a padded one has.
        self.room, self.team_size = observation_room(config), config.env.team_size
        self.files: list[Path] = []
        self.off = ""  # why no copy steps are taken
        self._orders: dict[int, list[int]] = {}
        self._futures: dict[int, concurrent.futures.Future[replays.Prepared]] = {}
        self._ready: dict[int, dict[str, torch.Tensor]] = {}  # the window's samples, compact
        self._failed: dict[str, set[Path]] = {}  # the replays left out, by why
        # Replays that failed to be made ready, and why: not made ready again in this run (each comes once a
        # cycle, so this spares the later cycles; a resumed run tries each once more, with the same outcome).
        self._bad: dict[Path, str] = {}
        self._failed_in_a_row = 0
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._started = False
        # With run.cards: every card takes the same copy steps, each on its own replays' decisions, and
        # applies the cards' average of their gradients, so the networks stay the same on every card.
        self.cards: Cards | None = None

    def start(self) -> None:
        """Lists the replays and starts making them ready, once: at the run's first iteration, so a run made
        only to save or to play (nitro copy-save, nitro reward-check) starts nothing."""
        if self._started:
            return
        self._started = True
        self.files = files_in(self.places)
        if not self.files:
            self._stop(self._none_here())
            return
        self._pool = concurrent.futures.ThreadPoolExecutor(WORKERS, thread_name_prefix="replay-mix")
        log.info(
            "replay_mix: %.0f%% of the policy's learning steps copy %s converted replays from %s (learning "
            "rate %g, %d decisions a step)%s.",
            100 * self.share,
            f"{len(self.files):,}",
            _where(self.places),
            self.lr,
            self.minibatch,
            f", only their kickoffs to {self.kickoff_only_seconds:g} s after the first touch"
            if self.kickoff_only
            else "",
        )
        self._prefetch()

    def _none_here(self) -> str:
        """The warning when the folders hold no converted replays."""
        return (
            f"replay_mix: {self._why_none()}; training goes on without copying replays. {self._remedy()} and "
            "start the bot again."
        )

    def _why_none(self) -> str:
        """'no converted 1v1 replays in A', each folder named once. A named folder (any, for an unpacked bot)
        that is not here is said to be so; the store makes its folders only when replays are converted."""
        gone = [p for p in self.places if not p.is_dir()] if self.named or self.moved else []
        here = [p for p in self.places if p not in gone]
        why = f"no converted {self.sizes} replays" + (f" in {_where(here)}" if here else "")
        if gone:
            why += f"{', and' if here else ':'} {_where(gone)} {'is' if len(gone) == 1 else 'are'} not here"
            why += " (packs do not carry replay folders)" if self.moved else ""
        return why

    def _remedy(self) -> str:
        if self.named:
            return "Copy the replays there (or point replay_mix.replays at where they are)"
        return "Convert some with nitro convert-replays (or the dashboard's Converted replays card)"

    def missing(self) -> str:
        """Why no replay would be copied, found without starting anything (run.missing_data = 'stop');
        empty when the folders hold converted replays."""
        if files_in(self.places):
            return ""
        return f"replay_mix: {self._why_none()}. {self._remedy()}."

    def _adam(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.params, lr=self.lr, fused=self.device.type == "cuda")

    def _scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type,
            init_scale=_FP16_INITIAL_SCALE if self.fp16 else 1.0,
            growth_interval=2000 if self.fp16 else _NEVER,
        )

    def _stop(self, why: str) -> None:
        self.off = why
        log.warning(why)
        self.notices.append(why)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "fp16": self.fp16,
            "generator": self.generator.get_state(),
            "next": self.next,
            "window": list(self.window),
            "budget": self.budget,
            "credit": self.credit,
            "steps": self.steps,
            # Halved after the graphics card ran out of memory, from the setting then (kept while it holds).
            "minibatch": self.minibatch,
            "minibatch_setting": self.chosen_minibatch,
        }

    def _load(self, state: dict[str, Any]) -> None:
        """Carries on from a save; its optimizer runs as this machine's (as the learner's) at the rate
        set now. A save whose optimizer does not fit the networks (grown since) starts a fresh one."""
        saved = state["optimizer"]
        here = {
            key: self.optimizer.defaults[key]
            for key in _ADAM_IMPLEMENTATION
            if key in self.optimizer.defaults
        }
        groups = [{**group, **here, "lr": self.lr} for group in saved["param_groups"]]
        try:
            self.optimizer.load_state_dict({**saved, "param_groups": groups})
            fits = all(
                value.shape == p.shape
                for p in self.params
                for key, value in self.optimizer.state.get(p, {}).items()
                if key != "step"
            )
        except (ValueError, KeyError, RuntimeError):
            fits = False
        if not fits:
            self.optimizer = self._adam()
            log.info(
                "replay_mix: the networks changed since the save; the copy steps' optimizer starts afresh."
            )
        if state.get("fp16") == self.fp16:
            self.scaler.load_state_dict(state["scaler"])
        self.generator.set_state(state["generator"].cpu())  # a helper card loads its start onto the card
        self.next, self.window = int(state["next"]), [int(p) for p in state["window"]]
        self.budget, self.credit, self.steps = (
            float(state["budget"]),
            float(state["credit"]),
            int(state["steps"]),
        )
        if state.get("minibatch_setting") == self.chosen_minibatch:
            self.minibatch = int(state["minibatch"])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    # Which replays: place k of the endless cycle is a file of the list shuffled anew each round of it.
    def _file_at(self, place: int) -> Path:
        cycle, k = divmod(place, len(self.files))
        if cycle not in self._orders:
            order = list(range(len(self.files)))
            random.Random(self.seed * 1_000_003 + cycle).shuffle(order)
            self._orders = {c: o for c, o in self._orders.items() if c >= cycle - 1} | {cycle: order}
        return self.files[self._orders[cycle][k]]

    def _job(self, place: int) -> replays.Job:
        file = self._file_at(place)
        return replays.Job(place, file, (), Path(), Path(), Path(), file, _size_of(file), self.settings, 0.1)

    def _prefetch(self) -> None:
        """The window's replays not in memory (after a resume) and the next AHEAD, made ready in the pool."""
        assert self._pool is not None
        wanted = [p for p in self.window if p not in self._ready] + list(range(self.next, self.next + AHEAD))
        for place in wanted:
            if place not in self._futures and self._skipped(place) is None:
                self._futures[place] = self._pool.submit(_prepare, self._job(place))

    def _fits(self, size: int) -> bool:
        """Whether a replay of `size` a side (0 = not known) can give observations of the bot's size."""
        return size == 0 or (size <= self.room if self.room else size == self.team_size)

    def _skipped(self, place: int) -> str | None:
        """Why replay `place` is left out without being made ready: a game size whose observations cannot
        fit the bot's, or a replay that failed before."""
        file = self._file_at(place)
        size = _size_of(file)
        if not self._fits(size):
            fits = f"up to {self.room}v{self.room}" if self.room else f"{self.team_size}v{self.team_size}"
            return f"{size}v{size} replays, where the bot's observations fit {fits}"
        return self._bad.get(file)

    def _take(self, place: int) -> dict[str, torch.Tensor] | None:
        """Replay `place`'s samples (waiting for them when need be), or None when it cannot be used."""
        assert self._pool is not None
        future = self._futures.pop(place, None)
        reason = self._skipped(place)
        if reason is None:
            ready = (future or self._pool.submit(_prepare, self._job(place))).result()
            if not ready.usable:  # byte offsets and the like: one entry a kind
                reason = re.sub(r"\d{3,}", "#", ready.problem or "no decision steps to learn from")[:120]
            elif ready.obs.shape[1] != self.obs_size:
                reason = f"observations of {ready.obs.shape[1]} numbers, where the bot's have {self.obs_size}"
            if reason is not None:
                self._bad[self._file_at(place)] = reason
            else:
                self._rows_kept += ready.count
                self._rows_left_out += ready.outside_window
                return pretrain._packed([ready])
        elif future is not None:
            future.cancel()
        self._failed.setdefault(reason, set()).add(self._file_at(place))
        return None

    def _fill(self) -> None:
        """The window refilled after a resume, and new replays taken in while it is short or its decisions
        are drawn; each replay in brings its decisions into the budget, so each is drawn about once."""
        for place in [p for p in self.window if p not in self._ready]:
            packed = self._take(place)
            if packed is None:
                self.window.remove(place)
            else:
                self._ready[place] = packed
        size = min(WINDOW, len(self.files))
        while not self.off and (len(self.window) < size or self.budget <= 0):
            place = self.next
            self.next += 1
            packed = self._take(place)
            if packed is None:
                self._failed_in_a_row += 1
                if self._failed_in_a_row >= len(self.files):
                    reasons = "; ".join(f"{len(files)} {why}" for why, files in self._failed.items())
                    self._stop(
                        f"replay_mix: none of the {len(self.files):,} converted replays could be used "
                        f"({reasons}); training goes on without copying replays."
                    )
                continue
            self._failed_in_a_row = 0
            self.window.append(place)
            self._ready[place] = packed
            self.budget += pretrain._size(packed)
            while len(self.window) > size:
                self._ready.pop(self.window.pop(0), None)
        self._prefetch()

    def learn(self, stats: LearnStats) -> CopyStats | None:
        """The copy steps owed for an update with `stats`: share / (1 - share) of its policy steps (none while
        a value warm-up holds the policy). None when replays are not being copied."""
        self.start()
        if self._stopped():
            return None
        self._rows_kept = self._rows_left_out = 0
        policy_steps = stats.gradient_steps - stats.value_only_steps
        owed = self.credit + policy_steps * self.share / (1.0 - self.share)
        count = math.floor(owed + 1e-9)
        self.credit = max(owed - count, 0.0)
        losses: list[torch.Tensor] = []
        agreements: list[torch.Tensor] = []
        waited = 0.0
        while len(losses) < count:
            began = time.perf_counter()
            self._fill()
            waited += time.perf_counter() - began
            if self._stopped():
                break
            try:
                loss, agreement = self._step(*self._rows())
            except torch.cuda.OutOfMemoryError as err:
                if self.cards is not None:  # halving one card's steps would put the cards out of step
                    raise TrainingProblem(
                        "a graphics card ran out of memory copying replays. With several cards (run.cards) "
                        "the copy steps cannot halve their decisions by themselves: lower "
                        "replay_mix.minibatch in the bot's settings, then resume"
                    ) from err
                self._out_of_memory()
                if self.off:
                    break
                continue  # the step is taken again, with half the decisions
            losses.append(loss)
            agreements.append(agreement)
            self.budget -= self.minibatch
            self.steps += 1
        window = (
            {"rows_kept": self._rows_kept, "rows_left_out": self._rows_left_out} if self.kickoff_only else {}
        )
        if not losses:
            return CopyStats(0, math.nan, math.nan, waited, **window)
        return CopyStats(
            len(losses),
            float(torch.stack(losses).mean()),
            float(torch.stack(agreements).mean()),
            waited,
            **window,
        )

    def _stopped(self) -> bool:
        """Whether the copy steps stop here; with several cards, whether any card's did (its replays ran out
        or failed): then they stop on every card, which would otherwise wait for it in the next sum."""
        if self.cards is None:
            return bool(self.off)
        flag = torch.full((1,), 1.0 if self.off else 0.0, device=self.cards.device)
        if float(self.cards.sum_(flag)[0]) > 0 and not self.off:
            self._stop("replay_mix: another graphics card stopped copying replays; this one stops too.")
        return bool(self.off)

    def _out_of_memory(self) -> None:
        """The graphics card ran out of memory in a copy step: its gradients and loss scaler's step are
        dropped, and from now on each step takes half the decisions (kept in the saves); below SMALLEST the
        copy steps stop. The run goes on either way."""
        self.optimizer.zero_grad(set_to_none=True)
        scale = (
            self.scaler.state_dict()
        )  # a step cut short leaves the scaler mid-step: a fresh one, same scale
        self.scaler = self._scaler()
        self.scaler.load_state_dict(scale)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        before, after = self.minibatch, self.minibatch // 2
        if after < SMALLEST:
            self._stop(
                f"replay_mix: the graphics card ran out of memory copying replays even at {before} "
                "decisions a step; training goes on without copying replays."
            )
            return
        self.minibatch = after
        why = (
            f"replay_mix: the graphics card ran out of memory copying replays. Each copy step now takes "
            f"{after} decisions (replay_mix.minibatch {before} -> {after}) for the rest of this bot's "
            f"training; to choose it yourself, set replay_mix.minibatch = {after}."
        )
        log.warning(why)
        self.notices.append(why)

    def _rows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A minibatch drawn from the window's replays together: observations, masks, labels and the
        players' first choices, on the device."""
        packs = [self._ready[p] for p in self.window]
        sizes = torch.tensor([pretrain._size(p) for p in packs])
        ends = sizes.cumsum(0)
        rows = torch.randint(int(ends[-1]), (self.minibatch,), generator=self.generator)
        which = torch.bucketize(rows, ends, right=True)
        parts, firsts = [], []
        for k in torch.unique(which).tolist():
            chosen = rows[which == k] - (ends[k] - sizes[k])
            parts.append(pretrain._unpacked(packs[k], chosen, self.actions, self.device))
            firsts.append(packs[k]["label_index"][chosen, 0])
        obs, mask, label = (torch.cat(t) for t in zip(*parts, strict=True))
        return obs, mask, label, torch.cat(firsts).to(self.device).long()

    def _step(
        self, obs: torch.Tensor, mask: torch.Tensor, label: torch.Tensor, first: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One copy step; its loss and first-choice agreement before it. The gradients are its own: they are
        cleared before and after, so PPO's step never sees them."""
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            logits = self.model.logits(obs)
        logits = logits.float()
        loss = pretrain._loss(logits, mask, label).mean()
        with torch.no_grad():
            agreement = (logits.masked_fill(~mask, -1e9).argmax(-1) == first).float().mean()
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        if self.cards is not None:  # the cards' average gradient, and loss and agreement
            grads = [p.grad for p in self.params if p.grad is not None]
            both = self.cards.average_(grads, torch.stack([loss.detach(), agreement]), obs.shape[0])
            loss, agreement = both[0], both[1]
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.params, CLIP)
        self.scaler.step(self.optimizer)  # skipped by itself when a gradient is not finite
        self.scaler.update(None if self.fp16 else 1.0)
        self.optimizer.zero_grad(set_to_none=True)
        return loss.detach(), agreement
