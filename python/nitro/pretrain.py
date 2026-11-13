"""`nitro pretrain`: behaviour cloning from recorded games.

The core turns each recording (RocketSim's RLPR format: RocketSim's bundled recordings, or human replays
converted to it) into samples: what each car saw at each decision step and which of the bot's actions its
controls amount to, as soft labels (`nitro_core::demos`). This module teaches the policy network to
choose like the recording, holding whole recordings back to measure how well it does on games it has not
seen, and saves the network as a run folder. A training run then
starts from it (`nitro train settings.toml --from <name>`, transfer learning), taking its policy only
(`transfer.untrained_value`: cloning trains no value network); `nitro watch` and `nitro export` also work on
it.
`[pretrain]` in the settings does all of it when a run starts: the replays are converted
(`nitro.replays`), the copy is made and kept in the run folder, and the run starts from it; that first start
saves the copy and stops, the next one trains. With
`pretrain.batch_replays` (the large replay mode) the replays are converted a batch at a time,
several at once, and learned from batch by batch, so any number of them fits in memory.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nitro import __version__, _core, console, tracking, transfer
from nitro import replays as replays_module
from nitro.config.schema import NitroConfig, PretrainConfig
from nitro.envs import core_settings
from nitro.learner import Learner
from nitro.model import ActorCritic, build_networks, entity_parts
from nitro.ppo import discount_from_half_life
from nitro.ratings import MU
from nitro.runs import RunFolder, all_settings, fixed_settings, portable

log = logging.getLogger("nitro")
RECORDING_PATTERNS = ("*.rlpr", "*.rlpr.zst")
COPY = "pretrained"  # the run folder's copy of the replays' players ([pretrain])
COPY_LABEL = "a copy of the players in your replays ([pretrain])"


class PretrainError(ValueError):
    """Nothing to learn from, or recordings that do not fit the settings; the message says why."""


@dataclass
class Samples:
    obs: torch.Tensor  # [n, obs_size] float32
    mask: torch.Tensor  # [n, actions] bool
    label: torch.Tensor  # [n, actions] float32, each row summing to 1
    recording: torch.Tensor  # [n] int64: which file each sample came from
    files: list[Path] = field(default_factory=list)
    skipped: int = 0
    read_as: dict[str, int] = field(default_factory=dict)  # recordings read with each `rows`

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def take(self, rows: torch.Tensor) -> Samples:
        return Samples(self.obs[rows], self.mask[rows], self.label[rows], self.recording[rows], self.files)


def recordings(paths: Sequence[Path]) -> list[Path]:
    """The recording files among `paths`; a folder stands for every recording in it."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for pattern in RECORDING_PATTERNS for p in path.rglob(pattern))
        elif path.is_file():
            found.append(path)
        else:
            raise PretrainError(f"{path} does not exist")
    if not found:
        raise PretrainError("no recordings (.rlpr or .rlpr.zst files) found")
    return found


def made_from_replays(path: Path) -> bool:
    """Whether a recording lies in the project's replays folder, where the converters write unless told
    otherwise: its rows hold the controls applied next, as replicar's do. A folder in it linked to
    another disk counts."""
    out = replays_module.OUT
    listed = Path(os.path.abspath(path)).is_relative_to(os.path.abspath(out))  # its links not followed
    return listed or path.resolve().is_relative_to(out.resolve())


def load(
    files: Sequence[Path],
    config: NitroConfig,
    softness: float = 0.1,
    notes: list[str] | None = None,
    rows: str = "auto",
) -> Samples:
    """Every sample of every file under `config`'s game and observation. A file whose game does not fit
    the observation is left out with a line in `notes`; none fitting is an error. `rows` says what a
    recording's rows hold: "replicar" the controls applied next (made from replays with replicar),
    "recorded" the controls that produced each tick (RocketSim's recordings), "auto" by
    where the file is (`made_from_replays`); the samples' `read_as` counts the recordings read each way."""
    if rows not in ("auto", "replicar", "recorded"):
        raise PretrainError(f'rows must be "auto", "replicar" or "recorded", not {rows!r}')
    settings = {made: copy_settings(config, replicar_rows=made) for made in (False, True)}
    parts: list[dict[str, Any]] = []
    used: list[Path] = []
    problems = []
    read_as = {"replicar": 0, "recorded": 0}
    for path in files:
        try:
            made = rows == "replicar" or (rows == "auto" and made_from_replays(path))
            parts.append(_core.demonstrations(str(path), settings[made], softness))
            read_as["replicar" if made else "recorded"] += 1
            used.append(path)
        except ValueError as err:
            problems.append(str(err))
    if notes is not None:
        notes += [f"left out: {problem}" for problem in problems]
    if not parts:
        raise PretrainError("no recording fits these settings: " + "; ".join(problems))
    actions = _core.num_actions()
    count = sum(int(p["count"]) for p in parts)
    if count == 0:
        raise PretrainError("the recordings hold no usable decision steps (every one crossed a gap)")

    def stacked(key: str, dtype: Any, width: int) -> np.ndarray:
        return np.concatenate([np.frombuffer(p[key], dtype=dtype).reshape(-1, width) for p in parts])

    obs_size = int(parts[0]["obs_size"])
    return Samples(
        obs=torch.from_numpy(stacked("obs", "<f4", obs_size).copy()),
        mask=torch.from_numpy(stacked("mask", np.uint8, actions).astype(bool)),
        label=torch.from_numpy(stacked("label", "<f4", actions).copy()),
        recording=torch.cat(
            [torch.full((int(p["count"]),), i, dtype=torch.int64) for i, p in enumerate(parts)]
        ),
        files=used,
        skipped=sum(int(p["skipped"]) for p in parts),
        read_as=read_as,
    )


def split(samples: Samples, held_out: float, seed: int) -> tuple[Samples, Samples]:
    """Whole recordings held back for measuring (a game is never on both sides); with one recording, its
    last part."""
    files = len(samples.files)
    if held_out <= 0:
        held = torch.zeros(len(samples), dtype=torch.bool)
    elif files >= 2:
        order = list(range(files))
        random.Random(seed).shuffle(order)
        kept = set(order[: min(max(1, round(files * held_out)), files - 1)])  # one file always learns
        held = torch.tensor([int(r) in kept for r in samples.recording])
    else:
        held = torch.arange(len(samples)) >= round(len(samples) * (1 - held_out))
    return samples.take(~held), samples.take(held)


@dataclass
class Epoch:
    epoch: int
    train_loss: float
    held_loss: float
    held_agreement: float  # share of held-back samples where the policy's first choice is the label's


def lr_at(lr: float, end_share: float, done: float) -> float:
    """The copy's learning rate once `done` (0 to 1) of it is learned: `lr` falling along a cosine curve
    to `end_share` of it; `end_share` 1 keeps it."""
    if end_share >= 1.0:
        return lr
    done = min(max(done, 0.0), 1.0)
    return lr * (end_share + (1.0 - end_share) * 0.5 * (1.0 + math.cos(math.pi * done)))


def _loss(logits: torch.Tensor, mask: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the soft labels under the masked policy, per sample."""
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    return -(label * log_probs.masked_fill(~mask, 0.0)).sum(-1)


@torch.no_grad()
def measure(
    model: ActorCritic, samples: Samples, device: torch.device, batch: int = 8192
) -> tuple[float, float]:
    """Mean loss and first-choice agreement on `samples`."""
    loss = agree = 0.0
    for begin in range(0, len(samples), batch):
        obs, mask, label = (
            t[begin : begin + batch].to(device) for t in (samples.obs, samples.mask, samples.label)
        )
        logits = model.logits(obs)
        loss += float(_loss(logits, mask, label).sum())
        choice = logits.masked_fill(~mask, -1e9).argmax(-1)
        agree += float((choice == label.argmax(-1)).sum())
    n = max(len(samples), 1)
    return loss / n, agree / n


def train(
    config: NitroConfig,
    train_set: Samples,
    held: Samples,
    *,
    epochs: int,
    batch: int,
    lr: float,
    device: torch.device,
    seed: int,
    report: Callable[[Epoch], None] | None = None,
    lr_end_share: float = 1.0,
) -> tuple[ActorCritic, list[Epoch]]:
    """A policy network (with the settings' sizes) taught to choose like the samples; a share of each step's
    decisions see no previous action (pretrain.prev_action_dropout), as in the large copy."""
    if len(train_set) == 0:
        raise PretrainError("nothing left to learn from once the held-back samples are set aside")
    torch.manual_seed(seed)
    actions = _core.num_actions()
    model = build_networks(config, train_set.obs.shape[1], actions).to(device)
    params = [p for name, p in model.named_parameters() if not name.startswith("value.")]
    optimizer = torch.optim.Adam(params, lr=lr)
    history: list[Epoch] = []
    generator = torch.Generator().manual_seed(seed)
    blank = blanking(config)
    steps, step = epochs * math.ceil(len(train_set) / batch), 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_set), generator=generator)
        total = 0.0
        for begin in range(0, len(train_set), batch):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(lr, lr_end_share, step / steps)
            step += 1
            rows = order[begin : begin + batch]
            obs, mask, label = (t[rows].to(device) for t in (train_set.obs, train_set.mask, train_set.label))
            if blank is not None:
                obs = _blanked(obs, blank, generator)
            loss = _loss(model.logits(obs), mask, label).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        model.eval()
        held_loss, agreement = measure(model, held, device) if len(held) else (math.nan, math.nan)
        history.append(Epoch(epoch, total / max(len(train_set), 1), held_loss, agreement))
        if report:
            report(history[-1])
    return model, history


def checkpoint(model: ActorCritic, config: NitroConfig, details: dict[str, Any]) -> dict[str, Any]:
    """The cloned network as a save, in the form `start_from` and `nitro watch` read."""
    device = next(model.parameters()).device
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(model, config.ppo, gamma, device, fp16=False)
    return {
        "learner": learner.state_dict(),
        "agent_steps": 0,
        "iterations": 0,
        "checkpoint": 0,
        "settings": fixed_settings(config),
        "all_settings": all_settings(config),
        "pretrained": details,
        "nitro_version": __version__,  # which release wrote it
    }


def save(folder: RunFolder, model: ActorCritic, config: NitroConfig, details: dict[str, Any]) -> Path:
    """The cloned network as the run folder's one save."""
    path = folder.save_checkpoint(checkpoint(model, config, details), 0, keep=1, milestone_steps=0)
    (folder.root / "pretrain.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return path


def copy_settings(config: NitroConfig, replicar_rows: bool = False) -> str:
    """The core settings a copy's samples are made under: the run's game and actions, and whether the
    recording's rows hold the controls applied next, as replicar's do."""
    settings = json.loads(core_settings(config))
    settings["replicar_rows"] = replicar_rows
    return json.dumps(settings)


def _made_with(config: NitroConfig) -> dict[str, Any]:
    """What the copy depends on: the [pretrain] settings that choose the replays and the networks' shape."""
    chosen = config.pretrain
    made: dict[str, Any] = json.loads(
        json.dumps(
            {
                "replays": (
                    [portable(p) for p in chosen.replays]
                    if len(chosen.replays) > 1
                    else next((portable(p) for p in chosen.replays), "")
                ),
                "limit": chosen.limit,
                "passes": chosen.passes,
                **(
                    {"batch_replays": chosen.batch_replays, "keep_batches": chosen.keep_batches}
                    if chosen.batch_replays
                    else {}
                ),
                **({"team_sizes": list(chosen.team_sizes)} if chosen.team_sizes else {}),
                **({"continue_from": chosen.continue_from} if chosen.continue_from else {}),
                **({"new_only": True} if chosen.new_only else {}),
                "seed": config.run.seed,
                "settings": fixed_settings(config),
                # What the labels depend on beyond the fixed settings.
                "env": {
                    name: getattr(config.env, name)
                    for name in ("tick_skip", "action_delay", "team_size", "opponents")
                },
            },
            default=str,
        )
    )
    return made


def _same_copy(saved: Any, made_with: dict[str, Any]) -> bool:
    """Whether a copy made with `saved` serves these settings: the same, or the same but for the run's game
    size, which a copy of several sizes into a padded observation does not depend on (team_sizes), so
    changing team_size before the first save keeps the copy."""
    if saved == made_with:
        return True
    if not isinstance(saved, dict) or not made_with.get("team_sizes"):
        return False

    def loose(made: dict[str, Any]) -> dict[str, Any]:
        made = json.loads(json.dumps(made))
        for name in ("team_size", "opponents"):
            made.get("env", {}).pop(name, None)
            made.get("settings", {}).pop(f"env.{name}", None)
        return made

    return loose(saved) == loose(made_with)


def _sources(chosen: PretrainConfig) -> list[Path]:
    """The replay folders: those named, each of which must be a folder, or the game's own."""
    from nitro import replays

    for folder in chosen.replays:
        if not folder.is_dir():
            raise PretrainError(f"{Path(*folder.parts[-2:])} is not a folder (pretrain.replays)")
    return list(chosen.replays) or replays.game_folders(replays.documents())


def start_source(
    config: NitroConfig,
    folder: RunFolder,
    say: Callable[[str], None],
    status: Callable[[str], None] | None = None,
    save_so_far: Callable[[int], None] | None = None,
) -> transfer.Source | None:
    """[pretrain]: the copy of the replays' players a new run starts from, made the first time
    (converting the replays, then cloning) and kept in the run folder, so a restart before the first save
    starts from the same copy; made again when the settings it depends on changed. None when off. `status`
    takes the large mode's frequent status lines (default: `say`); `save_so_far(replays learned)` saves the
    unfinished copy as the bot's save every pretrain.save_every replays."""
    if not config.pretrain.enabled:
        return None
    path = folder.root / COPY / "state.pt"
    state = torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None
    made_with = _made_with(config)
    if state is not None:
        if _same_copy(state["pretrained"].get("made_with"), made_with):
            return transfer.Source(state, COPY_LABEL, parts="policy")
        say("The [pretrain] settings or the networks changed since the copy was made: making it again.")
    large = bool(config.pretrain.batch_replays)
    charts = (
        tracking.start_copy(config.wandb, folder.root, all_settings(config, folder.root)) if large else None
    )
    try:
        if charts is not None:
            state = _make_copy_in_batches(
                config,
                path.parent,
                made_with,
                say,
                status or say,
                charts,
                save_so_far,
            )
        else:
            state = _make_copy(config, say)
        state["pretrained"].setdefault("made_with", made_with)  # a large copy's: as it began
        path.parent.mkdir(parents=True, exist_ok=True)
        _save(path, state)
        folder.drop_copy_so_far()  # the finished copy is what the bot's first save and training take
        for done in (PROGRESS, HELD):  # the large mode's progress is not needed once the copy is kept
            (path.parent / done).unlink(missing_ok=True)
        for version in (path.parent / VERSIONS).glob("*.pt"):  # its skill checks' versions, likewise
            version.unlink()
        details = json.dumps(state["pretrained"], indent=2)
        (path.parent / "pretrain.json").write_text(details, encoding="utf-8")
    finally:
        if charts is not None:
            charts.finish()  # after the copy is kept: W&B's last upload can take a while
    if not large:
        tracking.chart_passes(
            config.wandb, folder.root, all_settings(config, folder.root), state["pretrained"]["epochs"]
        )
    return transfer.Source(state, COPY_LABEL, parts="policy")


def _make_copy(config: NitroConfig, say: Callable[[str], None]) -> dict[str, Any]:
    from nitro import replays
    from nitro.training import resolve_device

    size = config.env.team_size
    chosen = config.pretrain
    sources = _sources(chosen)
    say(f"[pretrain]: copying the players in your {size}v{size} replays before training.")
    try:
        report = replays.convert_replays(
            sources, size, chosen.limit, config.paths.collision_meshes, replays.OUT, say
        )
    except replays.ReplayError as err:
        raise PretrainError(str(err)) from None
    where = " and ".join(str(Path(*p.parts[-2:])) for p in sources)  # the full path holds the user name
    if report.found == 0:
        raise PretrainError(
            f"no replays (.replay files) in {where}: save some in the game (Match History, Save Replay), or "
            "set pretrain.replays to a folder that has them"
        )
    failed = f"; {len(report.failed)} could not be converted" if report.failed else ""
    stopped = (
        f"; the search stopped after converting {report.converted} (five times pretrain.limit per start), "
        "and the next start goes on where it stopped"
        if report.stopped_early
        else ""
    )
    if not report.recordings:
        if report.stopped_early:
            raise PretrainError(f"no {size}v{size} game yet among the replays in {where}{stopped}{failed}")
        raise PretrainError(
            f"none of the {report.found:,} replays in {where} is a {size}v{size} game{failed}"
        )
    few = (
        f"; only {len(report.recordings)} of the {chosen.limit} wanted"
        if len(report.recordings) < chosen.limit
        else ""
    )
    say(
        f"{len(report.recordings)} {size}v{size} replays ready (of {report.found:,} in {where}{few}{failed}"
        f"{stopped}); converted replays are kept in the project's replays folder."
    )
    notes: list[str] = []
    samples = load(report.recordings, config, notes=notes, rows="replicar")  # every one converted
    for note in notes:
        say(note)
    train_set, held = split(samples, 0.1, config.run.seed)
    say(
        f"Learning to choose as the players did: {chosen.passes} passes over {len(train_set):,} decisions, "
        f"measured on {len(held):,} from replays it does not learn from."
    )

    def report_pass(epoch: Epoch) -> None:
        say(
            f"  pass {epoch.epoch}: first choice agrees with the player's on {epoch.held_agreement:.1%} of "
            "the decisions it did not learn from"
        )

    model, history = train(
        config,
        train_set,
        held,
        epochs=chosen.passes,
        batch=chosen.minibatch,
        lr=chosen.lr,
        device=resolve_device(config.run),
        seed=config.run.seed,
        report=report_pass,
        lr_end_share=chosen.lr_end_share,
    )
    details = {
        "replays": len(samples.files),
        "samples": len(samples),
        "held_out_samples": len(held),
        "softness": 0.1,
        "prev_action_dropout": config.pretrain.prev_action_dropout,
        "epochs": [dataclasses.asdict(e) for e in history],
    }
    return checkpoint(model.cpu(), config, details)


# The large replay mode.
PROGRESS = "progress.pt"  # in the copy's folder: where a stopped copy carries on from
HELD = "held.pt"  # the held-back replays' samples, made once
LIVE = "live.json"  # in the copy's folder: how far it is, for the dashboard
VERSIONS = "versions"  # in the copy's folder: the earlier versions its skill checks play
# A skill check's point starts at a kickoff and ends at a goal, the bot's own no-touch time or this.
SKILL_POINT_SECONDS = 60.0
BASE_SIGMA = 1.0  # the copy where it started: the fixed point its ratings are measured from


def _packed(parts: Sequence[replays_module.Prepared]) -> dict[str, torch.Tensor]:
    """Replays' compact samples as one set of tensors."""
    return {
        name: torch.from_numpy(np.concatenate([getattr(p, name) for p in parts]))
        for name in ("obs", "mask", "label_index", "label_share")
    }


def _resident(packed: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """`packed` on the graphics card when it fits in half its free memory, so each learning step picks its
    rows there instead of on the processor and copying them over: the same samples, the same steps.
    Otherwise (or on the processor) as it is."""
    if device.type != "cuda" or next(iter(packed.values())).device.type == "cuda":
        return packed
    free, _ = torch.cuda.mem_get_info(device)
    if sum(t.nbytes for t in packed.values()) > free // 2:
        return packed
    return {name: t.to(device) for name, t in packed.items()}


def _unpacked(
    packed: dict[str, torch.Tensor], rows: torch.Tensor, actions: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows of compact samples as (observations, mask, label) on `device`, in FP32."""
    rows = rows.to(packed["obs"].device)
    obs = packed["obs"][rows].to(device, non_blocking=True).float()
    bits = packed["mask"][rows].to(device, non_blocking=True)
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    mask = ((bits[:, :, None] >> shifts) & 1).bool().reshape(len(rows), -1)[:, :actions]
    label = torch.zeros(len(rows), actions, device=device).scatter_add_(
        1,
        packed["label_index"][rows].to(device).long(),
        packed["label_share"][rows].to(device).float(),
    )
    return obs, mask, label


def _size(packed: dict[str, torch.Tensor]) -> int:
    return int(packed["obs"].shape[0])


@torch.no_grad()
def _measure_packed(
    model: ActorCritic, packed: dict[str, torch.Tensor], device: torch.device, batch: int = 8192
) -> tuple[float, float]:
    """Mean loss and first-choice agreement on compact samples."""
    loss = agree = 0.0
    actions = _core.num_actions()
    packed = _resident(packed, device)
    for begin in range(0, _size(packed), batch):
        rows = torch.arange(begin, min(begin + batch, _size(packed)))
        obs, mask, label = _unpacked(packed, rows, actions, device)
        logits = model.logits(obs)
        loss += float(_loss(logits, mask, label).sum())
        index = packed["label_index"][rows.to(packed["label_index"].device), 0]
        first = index.to(device).long()  # the player's choice, as the full label's
        agree += float((logits.masked_fill(~mask, -1e9).argmax(-1) == first).sum())
    n = max(_size(packed), 1)
    return loss / n, agree / n


def previous_action(config: NitroConfig) -> slice:
    """Where the previous action (8 numbers) sits in the run's observation (nitro_core::obs)."""
    start = entity_parts(config.env.observation)[0] + 34  # the ball, then the 34 boost pads
    return slice(start, start + 8)


def blanking(config: NitroConfig) -> tuple[slice, float] | None:
    """The previous action's place and the share of decisions it is blanked in; None when off."""
    share = config.pretrain.prev_action_dropout
    return (previous_action(config), share) if share else None


def _blanked(obs: torch.Tensor, blank: tuple[slice, float], generator: torch.Generator) -> torch.Tensor:
    """`obs` (a batch of its own) with the previous action zeroed on a random share of its rows."""
    place, share = blank
    chosen = torch.rand(len(obs), generator=generator) < share
    obs[chosen.to(obs.device), place] = 0.0
    return obs


def _learn_round(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[dict[str, torch.Tensor]],
    passes: int,
    batch: int,
    device: torch.device,
    seed: int,
    scaler: torch.amp.GradScaler | None = None,
    tick: Callable[[int, int], None] | None = None,
    lr_for: Callable[[int, int], float] | None = None,
    blank: tuple[slice, float] | None = None,
) -> float:
    """`passes` shuffled passes over every sample of `batches` together; the mean loss. With `scaler`, in
    FP16 (the output layers stay FP32, as in training). `tick(step, steps)` follows each step;
    `lr_for(step, steps)` sets the learning rate before it. `blank` (the previous action's place in the
    observation, a share): that share of each step's decisions see no previous action."""
    batches = [_resident(b, device) for b in batches]
    sizes = torch.tensor([_size(b) for b in batches])
    ends = sizes.cumsum(0)
    total, actions = int(ends[-1]), _core.num_actions()
    params = [p for name, p in model.named_parameters() if not name.startswith("value.")]
    generator = torch.Generator().manual_seed(seed)
    summed = 0.0
    model.train()
    steps, step = passes * math.ceil(total / batch), 0
    for _ in range(passes):
        order = torch.randperm(total, generator=generator)
        for begin in range(0, total, batch):
            rows = order[begin : begin + batch]
            which = torch.bucketize(rows, ends, right=True)  # each row's batch
            parts = []
            for k in torch.unique(which).tolist():
                chosen = rows[which == k] - (ends[k] - sizes[k])
                parts.append(_unpacked(batches[k], chosen, actions, device))
            obs, mask, label = (torch.cat(t) for t in zip(*parts, strict=True))
            if blank is not None:
                obs = _blanked(obs, blank, generator)
            if lr_for is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_for(step, steps)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
                logits = model.logits(obs)
            loss = _loss(logits.float(), mask, label).mean()
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()
            else:
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                scaler.step(optimizer)
                scaler.update()
            summed += float(loss.detach()) * len(rows)
            step += 1
            if tick is not None:
                tick(step, steps)
    model.eval()
    return summed / max(total * passes, 1)


STATUS_SECONDS = 30.0  # the large copy says how it is going at least this often


class _Pace:
    """The large copy's status lines, at most every STATUS_SECONDS: what it is doing, how far it is and about
    how long is left, from this start's pace (usable replays made ready a minute, seconds a round of
    learning; the two overlap, so the slower one decides)."""

    def __init__(
        self,
        status: Callable[[str], None],
        limit: int,
        batch_replays: int,
        counts: Callable[[], tuple[int, int, int]],
    ) -> None:
        self.status, self.limit, self.batch_replays, self.counts = status, limit, batch_replays, counts
        self.report: Callable[[str], None] = lambda doing: None  # hears what it is doing with each status
        self.began = self.last = time.perf_counter()
        self.usable = 0  # usable replays made ready this start, held-back ones included
        self.learning = 0.0  # seconds spent learning this start
        self.rounds = 0

    def learned(self, seconds: float) -> None:
        self.learning += seconds
        self.rounds += 1

    def left(self) -> float | None:
        """Seconds left, or None before the first replay is ready."""
        used, in_batch, held_left = self.counts()
        elapsed = time.perf_counter() - self.began
        if not self.usable or elapsed <= 0:
            return None
        converting = float(max(self.limit - used - in_batch, 0) + held_left) / (self.usable / elapsed)
        if not self.rounds:
            return converting
        per_round = self.learning / self.rounds
        rounds_left = math.ceil(max(self.limit - used, 0) / self.batch_replays)
        return float(max(converting + per_round, rounds_left * per_round))

    def tick(self, doing: str) -> None:
        now = time.perf_counter()
        if now - self.last < STATUS_SECONDS:
            return
        self.last = now
        used, in_batch, _ = self.counts()
        rate = self.usable / (now - self.began) * 60
        left = self.left()
        if left is None:
            eta = "the time left follows once the first replays are ready"
        else:
            done = time.strftime("%H:%M", time.localtime(time.time() + left))
            eta = f"about {_duration(left)} left (done about {done})"
            if not self.rounds:
                eta += ", the learning's time added once round 1 is timed"
        self.status(
            f"  [{time.strftime('%H:%M')}] {used:,} of {self.limit:,} replays learned, {in_batch} in the "
            f"next batch, {rate:.0f} made ready a minute; now {doing}; {eta}"
        )
        self.report(doing)

    def rate(self) -> float:
        """Usable replays made ready a minute this start."""
        elapsed = time.perf_counter() - self.began
        return self.usable / elapsed * 60 if elapsed > 0 else 0.0


def _finite(value: Any) -> Any:
    """`value` with every number that is not finite as None: JSON has no NaN or infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _write_live(work: Path, data: dict[str, Any]) -> None:
    """The copy's progress for the dashboard, replaced in one step so a reader never sees half of it.
    Not worth stopping the copy for: a moment Windows holds the file just skips one update."""
    partial = work / (LIVE + ".partial")
    try:
        partial.write_text(json.dumps(_finite(data)), encoding="utf-8")
        os.replace(partial, work / LIVE)
    except OSError:
        pass


def read_live(work: Path) -> dict[str, Any] | None:
    """A copy's progress as `_write_live` left it, or None."""
    try:
        data = json.loads((work / LIVE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _keep_version(work: Path, round_number: int, model: ActorCritic) -> None:
    """Keeps the copy at `round_number` for later skill checks."""
    folder = work / VERSIONS
    folder.mkdir(parents=True, exist_ok=True)
    _save(folder / f"{round_number}.pt", {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def _drop_versions(work: Path, skill: dict[str, Any]) -> None:
    """Drops the kept versions no skill check plays again: only where the copy started and the newest are
    played. Called once the progress that no longer needs them is saved, so a stop in between loses none."""
    played = {str(skill["versions"][0]["round"]), str(skill["versions"][-1]["round"])}
    for old in (work / VERSIONS).glob("*.pt"):
        if old.stem not in played:
            old.unlink()


def skill_game(config: NitroConfig) -> NitroConfig:
    """The game of a copy's skill checks: the bot's own, from kickoffs, a point ending at a goal, the bot's
    no-touch time or SKILL_POINT_SECONDS, with the copy's actions."""
    from nitro.config.state_setters import Kickoff

    seconds = config.terminals.max_episode_seconds
    terminals = dataclasses.replace(
        config.terminals,
        max_episode_seconds=min(seconds, SKILL_POINT_SECONDS) if seconds else SKILL_POINT_SECONDS,
        truncate_on_ball_landing=0.0,
    )
    return dataclasses.replace(config, state_setters=(Kickoff(),), terminals=terminals)


def skill_check(
    model: ActorCritic,
    config: NitroConfig,
    work: Path,
    progress: dict[str, Any],
    round_number: int,
    say: Callable[[str], None],
) -> None:
    """The copy after `round_number` plays where the copy started and the version of the last check, Blue
    against Orange from kickoffs, `pretrain.skill_points` points each; its rating starts at the last
    version's and moves with every point (a goal for it 1, against it 0, none 0.5) against their ratings,
    which stay as they were measured (`nitro.ratings`, as self-play's). Kept in progress["skill"]."""
    from nitro import eval as matches
    from nitro.ratings import SIGMA, Rating, update

    skill = progress["skill"]
    versions = skill["versions"]
    game = skill_game(config)
    me = matches.Side("this copy", 0, {"learner": {"model": model.state_dict()}}, game)
    rating = Rating(versions[-1]["mu"], SIGMA)
    opponents = [versions[0]] + ([versions[-1]] if len(versions) > 1 else [])
    heard = []
    for number, version in enumerate(opponents):
        file = work / VERSIONS / f"{version['round']}.pt"
        if not file.is_file():
            heard.append(f"round {version['round']}'s copy is gone (its file was removed), so not played")
            continue
        weights = torch.load(file, map_location="cpu", weights_only=True)
        them = matches.Side(f"round {version['round']}", 0, {"learner": {"model": weights}}, game)
        seed = config.run.seed * 1_000_003 + round_number * 31 + number
        result = matches.play(me, them, config.pretrain.skill_points, seed)
        scores = [1.0] * result.blue_goals + [0.0] * result.orange_goals + [0.5] * result.no_goal
        random.Random(seed).shuffle(scores)
        theirs = Rating(version["mu"], version["sigma"])
        for score in scores:
            rating, _ = update(rating, theirs, score)
        low, high = result.interval()
        skill["matches"].append(
            {
                "round": round_number,
                "against": version["round"],
                "points": result.points,
                "won": result.blue_goals,
                "lost": result.orange_goals,
                "draws": result.no_goal,
                "low": low,
                "high": high,
            }
        )
        share = (
            f"it took {result.blue_share:.0%} of the {result.decided} goals (95% interval {low:.0%} to "
            f"{high:.0%})"
            if result.decided
            else "nobody scored"
        )
        heard.append(f"against round {version['round']}'s copy in {result.points} points {share}")
    versions.append(
        {"round": round_number, "replays": progress["used"], "mu": rating.mu, "sigma": rating.sigma}
    )
    _keep_version(work, round_number, model)
    say(
        f"  skill check after round {round_number}: {'; '.join(heard)}; rating {rating.mu:.1f} (where the "
        f"copy started: {versions[0]['mu']:.1f})"
    )


def held_replays(limit: int) -> int:
    """Replays held back to measure on in the large mode: 1 in 100, between 2 and 50."""
    return min(50, max(2, limit // 100))


def workers(chosen: int) -> int:
    """Replays made ready at once: by default half the threads (on a 20-thread test machine 10 made ready 23%
    more a minute than every thread but two)."""
    return chosen or max(1, min(16, (os.cpu_count() or 2) // 2))


def _make_copy_in_batches(
    config: NitroConfig,
    work: Path,
    made_with: dict[str, Any],
    say: Callable[[str], None],
    status: Callable[[str], None],
    charts: tracking.CopyTracker,
    save_so_far: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """The copy by the large replay mode: the replays are made ready by several threads (replicar is a
    program of its own, and the core works outside Python's lock), a batch at a time, and learned
    from batch by batch (the newest batch with the ones kept before it), each batch dropped once it has had
    its rounds; progress is saved after every batch, so a stopped copy carries on where it was, and every
    pretrain.save_every replays `save_so_far` saves it as the bot's save. Returns the copy as a save."""
    import concurrent.futures

    from nitro import replays
    from nitro.eval import EvalError
    from nitro.training import resolve_device

    chosen = config.pretrain
    wanted = tuple(chosen.team_sizes) or (config.env.team_size,)
    games = "/".join(f"{n}v{n}" for n in wanted)
    sources = _sources(chosen)
    where = " and ".join(str(Path(*p.parts[-2:])) for p in sources)  # the full path holds the user name
    continued = _continued(config, work) if chosen.continue_from else None
    work.mkdir(parents=True, exist_ok=True)
    progress = _progress(work, made_with, say)
    if progress is None:
        charts.restart()  # a copy made from its first round: its rounds are sent afresh
        RunFolder(work.parent).drop_copy_so_far()  # a save of the copy made before, with other settings
        # A stale progress (other settings) must not meet this copy's held-back samples later.
        for stale in (PROGRESS, HELD):
            (work / stale).unlink(missing_ok=True)
        status(f"Looking through the replays in {where} (a minute or two for tens of thousands)...")
        # Several folders named (one per game size): taken in turn, so every batch mixes them.
        found = (
            replays.newest_in_turn(sources) if len(chosen.replays) > 1 else replays.newest_replays(sources)
        )
        if not found:
            raise PretrainError(
                f"no replays (.replay files) in {where}: save some in the game (Match History, Save "
                "Replay), or set pretrain.replays to a folder that has them"
            )
        if not (config.paths.collision_meshes / "soccar").is_dir():
            raise PretrainError(
                f"the arena meshes were not found in {config.paths.collision_meshes} (paths.collision_meshes)"
            )
        sizes = replays.Sizes(replays.OUT) if replays.OUT.is_dir() else None
        planned = [
            str(p) for p in found if sizes is None or not all(sizes.other_size(p.stem, n) for n in wanted)
        ]
        used_before = 0
        if chosen.new_only and sizes is not None:  # made ready by an earlier copy: its size is noted
            made = {str(n) for n in wanted}
            fresh = [p for p in planned if sizes.known.get(Path(p).stem) not in made]
            used_before, planned = len(planned) - len(fresh), fresh
        progress = {
            "made_with": made_with,
            "plan": planned,
            "found": len(found),
            "next": 0,
            "held": [],
            "rounds": 0,
            "used": 0,
            "samples": 0,
            "failed": {},
            "other_sizes": 0,
            "by_size": {},
            "history": [],
            "model": None,
            "optimizer": None,
            "seconds_preparing": 0.0,
            "prepared": 0,
            "used_before": used_before,
            "before": None,
        }
        say(
            f"[pretrain], large replay mode: up to {chosen.limit:,} {games} replays of the "
            f"{len(found):,} in {where}, {chosen.batch_replays} at a time, {workers(chosen.workers)} "
            "converted at once."
            + (f" Left out as made ready before: {used_before:,}." if chosen.new_only else "")
            + (
                f" Carrying on the copy of '{chosen.continue_from}' "
                f"({_replays_in(continued['pretrained']):,} replays learned there)."
                if continued is not None
                else ""
            )
        )
    else:
        say(
            f"[pretrain], large replay mode: carrying on after round {progress['rounds']} "
            f"({progress['used']:,} of {chosen.limit:,} replays learned)."
        )
    plan = [Path(p) for p in progress["plan"]]
    if continued is not None:
        progress["continued_replays"] = _replays_in(continued["pretrained"])
    held_wanted = held_replays(chosen.limit)
    held: dict[str, torch.Tensor] | None = progress.pop("held_samples", None)
    held_parts: list[replays.Prepared] = []
    # The store's conversions, kept recordings and fresh conversions are all replicar's rows.
    settings = copy_settings(config, replicar_rows=True)
    device = resolve_device(config.run)
    # The run's precision: FP16 on a graphics card, expected faster with large networks (training measured
    # about twice as fast; the copy's own speed is not measured on its own).
    scaler = (
        torch.amp.GradScaler("cuda") if config.run.precision == "fp16" and device.type == "cuda" else None
    )
    replays.OUT.mkdir(parents=True, exist_ok=True)
    sizes = replays.Sizes(replays.OUT)
    lean = chosen.converter == "nitro"
    tool = replays.nitro_replicar_tool() if lean else replays.replicar_tool()
    parquet_dir = replays.OUT / "converted"
    parquet_dir.mkdir(exist_ok=True)
    for left in [*parquet_dir.glob("*.partial"), *parquet_dir.glob("*.parquet"), *parquet_dir.glob("*.lean")]:
        left.unlink(missing_ok=True)  # replicar's files of a copy that was stopped

    store = replays.store_folder() if chosen.keep_converted and lean else None
    told_full = False

    def kept(replay: Path) -> tuple[Path, int]:
        """`replay` converted before, in any copied size: a recording or a kept conversion."""
        for n in wanted:
            path = replays.OUT / f"{n}v{n}" / f"{replay.stem}.rlpr.zst"
            if path.is_file():
                return path, n
        found = replays.find_stored(replay.stem, wanted)
        if found is not None:
            return found
        return replays.OUT / f"{wanted[0]}v{wanted[0]}" / f"{replay.stem}.rlpr.zst", wanted[0]

    def job(index: int) -> replays.Job:
        replay = plan[index]
        return replays.Job(
            index,
            replay,
            wanted,
            tool,
            config.paths.collision_meshes,
            parquet_dir,
            *kept(replay),
            settings,
            0.1,
            lean,
            store,
        )

    model: ActorCritic | None = None
    optimizer: torch.optim.Optimizer | None = None

    def networks(obs_size: int) -> tuple[ActorCritic, torch.optim.Optimizer]:
        torch.manual_seed(config.run.seed)
        built = build_networks(config, obs_size, _core.num_actions()).to(device)
        params = [p for name, p in built.named_parameters() if not name.startswith("value.")]
        made = torch.optim.Adam(params, lr=chosen.lr)
        if progress["model"] is not None:
            built.load_state_dict(progress["model"])
            made.load_state_dict(progress["optimizer"])
        elif continued is not None:
            try:
                built.load_state_dict(continued["learner"]["model"])
            except RuntimeError as err:  # the observation's size, checked once the first replays are ready
                why = "; ".join(line.strip() for line in str(err).splitlines()[1:] if line.strip())
                raise PretrainError(
                    f"pretrain.continue_from: the copy of '{chosen.continue_from}' does not fit this run's "
                    f"networks: {why[:400]}"
                ) from None
        return built, made

    count = workers(chosen.workers)
    lookahead = chosen.batch_replays + 2 * count
    recent: list[dict[str, torch.Tensor]] = []
    batch: list[replays.Prepared] = []
    began = time.perf_counter()
    prepared_now = 0
    pace = _Pace(
        status,
        chosen.limit,
        chosen.batch_replays,
        lambda: (
            progress["used"],
            len(batch),
            held_wanted - len(held_parts) if held is None else 0,
        ),
    )

    def live(doing: str, finished: bool = False) -> None:
        """How far the copy is, for the dashboard."""
        _write_live(
            work,
            {
                "updated": time.time(),
                "doing": doing,
                "finished": finished,
                "games": games,
                "limit": chosen.limit,
                "learned": progress["used"],
                "in_batch": len(batch),
                "held_back": len(progress["held"]),
                "held_wanted": held_wanted,
                "found": progress.get("found", 0),
                "made_ready": progress.get("prepared", 0),
                "failed": sum(progress["failed"].values()),
                "other_sizes": progress["other_sizes"],
                "by_size": progress.get("by_size", {}),
                "per_minute": pace.rate(),
                "seconds_left": None if finished else pace.left(),
                "rounds": chart_rounds(progress),
                "skill_every": chosen.skill_every,
                "skill": progress.get("skill"),
                "save_every": chosen.save_every,
            },
        )

    pace.report = live
    live("converting replays")

    def finish_round(last_index: int, final: bool) -> None:
        nonlocal model, optimizer
        replays_in_batch = len(batch)
        replays_before = progress["used"]
        packed = _packed(batch)
        batch.clear()  # held once only, packed
        progress["obs_size"] = int(packed["obs"].shape[1])
        if model is None or optimizer is None:
            model, optimizer = networks(progress["obs_size"])
        if chosen.skill_every and progress.get("skill") is None:  # where the copy starts: its ratings' zero
            progress["skill"] = {
                "versions": [
                    {"round": progress["rounds"], "replays": progress["used"], "mu": MU, "sigma": BASE_SIGMA}
                ],
                "matches": [],
            }
            _keep_version(work, progress["rounds"], model)
        assert held is not None
        if continued is not None and progress["rounds"] == 0 and progress.get("before") is None:
            before_loss, before_agreement = _measure_packed(model, held, device)
            progress["before"] = {"held_loss": before_loss, "held_agreement": before_agreement}
            say(
                f"  before these replays (the copy of '{chosen.continue_from}'): first choice agrees with "
                f"the player's on {before_agreement:.1%} of {_size(held):,} decisions from "
                f"{len(progress['held'])} replays it does not learn from (loss there {before_loss:.3f})"
            )
        recent.append(packed)
        del recent[: -chosen.keep_batches]
        round_number = progress["rounds"] + 1
        learning = time.perf_counter()
        loss = _learn_round(
            model,
            optimizer,
            recent,
            chosen.passes,
            chosen.minibatch,
            device,
            config.run.seed * 1_000_003 + round_number,
            scaler,
            lambda step, steps: pace.tick(f"learning round {round_number} ({step / steps:.0%})"),
            lambda step, steps: lr_at(
                chosen.lr,
                chosen.lr_end_share,
                (replays_before + replays_in_batch * step / steps) / chosen.limit,
            ),
            blanking(config),
        )
        learned = time.perf_counter() - learning
        pace.learned(learned)
        # Measured every measure_every rounds and after the last; a round between says so.
        measured = final or round_number % chosen.measure_every == 0
        held_loss, agreement = _measure_packed(model, held, device) if measured else (None, None)
        progress["rounds"] = round_number
        progress["used"] += replays_in_batch
        progress["samples"] += _size(packed)
        progress["next"] = last_index + 1
        elapsed = time.perf_counter() - began
        rate = prepared_now / elapsed * 60 if elapsed > 0 else 0.0
        progress["history"].append(
            {
                "epoch": round_number,
                "train_loss": loss,
                "held_loss": held_loss,
                "held_agreement": agreement,
                # For the charts and the console report; rounds saved before lack them.
                "replays": progress["used"],
                "decisions": progress["samples"],
                "replays_per_minute": rate,
                "seconds_learning": learned,
            }
        )
        if chosen.skill_every and (final or round_number % chosen.skill_every == 0):
            live(f"skill check: round {round_number}'s copy plays earlier versions")
            try:
                skill_check(model, config, work, progress, round_number, say)
            except (ValueError, RuntimeError, OSError, EvalError) as err:  # never worth stopping the copy
                say(f"  skill check after round {round_number} not played: {str(err)[:300]}")
        progress["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        progress["optimizer"] = optimizer.state_dict()
        _save(work / PROGRESS, progress)
        if progress.get("skill"):
            _drop_versions(work, progress["skill"])
        every = chosen.save_every
        if (
            save_so_far is not None
            and every
            and not final
            and progress["used"] // every > replays_before // every
        ):
            live("saving the copy so far as the bot's save")
            if device.type == "cuda":
                torch.cuda.empty_cache()  # the save is made by a process of its own, on the same card
            save_so_far(progress["used"])
        live(f"learned round {round_number}")
        left = chosen.limit - progress["used"]
        eta = ""
        if rate > 0 and left > 0:
            took = _duration(left / rate * 60)
            eta = f"; {took} left" if took.startswith("under") else f"; about {took} left"
        measure = (
            f"first choice agrees with the player's on {agreement:.1%} of {_size(held):,} decisions from "
            f"{len(progress['held'])} replays it does not learn from (loss there {held_loss:.3f}, lower is "
            "closer)"
            if held_loss is not None and agreement is not None
            else f"held-back replays measured every {chosen.measure_every} rounds"
        )
        say(
            f"  round {round_number}: {progress['used']:,} of {chosen.limit:,} replays learned "
            f"({progress['samples']:,} decisions); {measure}; {rate:.1f} replays made ready a minute{eta}"
        )
        charts.rounds(chart_rounds(progress))
        if config.run.console == "report":
            done = ""
            if rate > 0 and left > 0:
                done = time.strftime("%H:%M", time.localtime(time.time() + left / rate * 60))
            block = console.Round(
                name=work.parent.name,
                number=round_number,
                used=progress["used"],
                limit=chosen.limit,
                decisions=progress["samples"],
                train_loss=loss,
                held_loss=held_loss,
                agreement=agreement,
                held_decisions=_size(held),
                held_replays=len(progress["held"]),
                rate=rate,
                learned_seconds=learned,
                left=eta[2:].removesuffix(" left") + (f" (done about {done})" if done else ""),
                used_before=progress.get("used_before") if chosen.new_only else None,
                before=progress.get("before"),
            )
            status("\n".join(console.round_report(block)))

    charts.rounds(chart_rounds(progress))  # rounds made before charts were on
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=count, thread_name_prefix="replays")
    pending: dict[int, concurrent.futures.Future[replays.Prepared]] = {}
    submit_at = progress["next"]
    try:
        index = progress["next"]
        while index < len(plan) and progress["used"] + len(batch) < chosen.limit:
            while submit_at < len(plan) and len(pending) < lookahead:
                pending[submit_at] = pool.submit(replays.prepare, job(submit_at))
                submit_at += 1
            future = pending.pop(index)
            while True:
                try:
                    ready = future.result(timeout=5)
                    break
                except concurrent.futures.TimeoutError:
                    pace.tick("converting replays")
            prepared_now += 1
            progress["prepared"] = progress.get("prepared", 0) + 1
            if ready.kept_now:
                progress["kept_now"] = progress.get("kept_now", 0) + 1
            elif ready.kept_now is False and not told_full:
                gigabytes = replays.STORE_MIN_FREE // 2**30
                say(
                    f"Converted replays are no longer kept for later bots: under {gigabytes} GB free on the "
                    "disk."
                )
                told_full = True
            progress["seconds_preparing"] += ready.seconds
            if ready.size is not None and ready.size not in wanted:
                sizes.note(ready.stem, str(ready.size))
                progress["other_sizes"] += 1
            elif ready.problem == replays.NO_KICKOFF:
                sizes.note(ready.stem, replays.NO_KICKOFF)
            reason = ready.problem
            if reason is None and ready.size in wanted and not ready.usable:
                reason = "no decision steps to learn from"
            if reason:
                reason = re.sub(r"\d{3,}", "#", reason)[:120]  # byte offsets and the like: one entry a kind
                progress["failed"][reason] = progress["failed"].get(reason, 0) + 1
            if ready.usable and ready.size in wanted:
                pace.usable += 1
                pace.tick("converting replays")
                if not job(index).kept.is_file():
                    sizes.note(ready.stem, str(ready.size))
                by_size = progress.setdefault("by_size", {})
                by_size[str(ready.size)] = by_size.get(str(ready.size), 0) + 1
                if held is None:
                    held_parts.append(ready)
                    progress["held"].append(ready.stem)
                    if len(held_parts) == held_wanted:
                        held = _packed(held_parts)
                        _save(work / HELD, {"made_with": made_with, "samples": held})
                        held_parts.clear()
                else:
                    batch.append(ready)
                    if len(batch) == chosen.batch_replays:
                        finish_round(index, progress["used"] + len(batch) >= chosen.limit)
            index += 1
        if batch:
            finish_round(index - 1, True)
        history = progress["history"]
        if history and history[-1]["held_loss"] is None and model is not None and held is not None:
            # The replays ran out after a round between measurements: the copy as kept is measured.
            loss_now, agreement_now = _measure_packed(model, held, device)
            history[-1].update(held_loss=loss_now, held_agreement=agreement_now)
            _save(work / PROGRESS, progress)
            say(
                f"  after round {history[-1]['epoch']}: first choice agrees with the player's on "
                f"{agreement_now:.1%} of {_size(held):,} decisions from {len(progress['held'])} replays it "
                f"does not learn from (loss there {loss_now:.3f})"
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if not any(parquet_dir.iterdir()):
            parquet_dir.rmdir()
    if progress["used"] == 0:
        failed = sum(progress["failed"].values())
        usable = len(held_parts) + len(progress["held"]) * (held is not None)
        raise PretrainError(
            f"only {usable} of the {len(plan):,} replays in {where} became {games} samples ("
            f"{progress['other_sizes']} of another team size, {failed} could not be converted); the large "
            f"replay mode holds {held_wanted} back to measure on and learns from the rest, so it needs more: "
            "for a few replays leave pretrain.batch_replays at 0"
        )
    if model is None:  # stopped after the last round, before the copy was kept
        model, _ = networks(progress["obs_size"])
    if progress["used"] < chosen.limit:
        say(f"Every replay in {where} has been used: {progress['used']:,} learned from.")
    live("done", finished=True)
    details = {
        "replays": progress["used"],
        "samples": progress["samples"],
        "held_out_samples": _size(held) if held is not None else 0,
        "held_out_replays": len(progress["held"]),
        "softness": 0.1,
        "epochs": progress["history"],
        "made_with": progress["made_with"],  # as begun (`_same_copy`)
        **({"skill": progress["skill"]} if progress.get("skill") else {}),
        **(
            {
                "continued_from": {
                    "run": chosen.continue_from,
                    "replays": _replays_in(continued["pretrained"]),
                },
                "before": progress.get("before"),  # the carried-on copy on this copy's held-back replays
            }
            if continued is not None
            else {}
        ),
        "large": {
            "batch_replays": chosen.batch_replays,
            "keep_batches": chosen.keep_batches,
            "rounds": progress["rounds"],
            "used_before": progress.get("used_before", 0),  # left out by new_only
            "other_team_sizes": progress["other_sizes"],
            "by_size": progress.get("by_size", {}),  # replays made ready of each copied size
            "failed": progress["failed"],
            "seconds_per_replay": progress["seconds_preparing"] / max(progress.get("prepared", 0), 1),
        },
    }
    return checkpoint(model.cpu(), config, details)


def copy_so_far(config: NitroConfig, folder: RunFolder) -> tuple[transfer.Source, dict[str, int]]:
    """The unfinished copy of the large replay mode as its last round left it, in the form `start_source`
    gives a finished one, and how far it is ({"learned", "limit"} replays): for a save of the copy so far
    (pretrain.save_every)."""
    work = folder.root / COPY
    progress = _readable(work / PROGRESS, lambda line: None)
    if not isinstance(progress, dict) or progress.get("model") is None:
        raise PretrainError("the copy has not finished a round yet")
    if not _same_copy(progress.get("made_with"), _made_with(config)):
        raise PretrainError("the copy's settings changed since its last round: it starts again")
    model = build_networks(config, progress["obs_size"], _core.num_actions())
    model.load_state_dict(progress["model"])
    details = {
        "replays": progress["used"],
        "samples": progress["samples"],
        "held_out_replays": len(progress["held"]),
        "epochs": progress["history"],
        "so_far": True,
    }
    state = checkpoint(model, config, details)
    state["pretrained"]["made_with"] = progress["made_with"]
    learned = {"learned": int(progress["used"]), "limit": int(config.pretrain.limit)}
    return transfer.Source(state, COPY_LABEL, parts="policy"), learned


def _continued(config: NitroConfig, work: Path) -> dict[str, Any]:
    """The copy pretrain.continue_from names, in the runs folder that holds this run (`work`'s grandparent),
    checked against this run's settings before any replay is converted."""
    name = config.pretrain.continue_from
    if name.lower() == work.parent.name.lower():  # Windows' names ignore case
        raise PretrainError("pretrain.continue_from names this run itself")
    path = work.parent.parent / name / COPY / "state.pt"
    if not path.is_file():
        raise PretrainError(
            f"pretrain.continue_from: the run '{name}' has no copy of replays to carry on "
            f"({Path(name, COPY, 'state.pt')} is missing)"
        )
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    found = transfer.problems(transfer.Source(state, name, parts="policy"), config)
    theirs, mine = state["pretrained"].get("made_with", {}).get("env", {}), _made_with(config)["env"]
    found += [
        f"env.{key} is {theirs[key]} there and {mine[key]} here (the labels depend on it)"
        for key in ("tick_skip", "action_delay")
        if key in theirs and theirs[key] != mine[key]
    ]
    if found:
        raise PretrainError(
            f"pretrain.continue_from: the copy of '{name}' cannot be carried on here: " + "; ".join(found)
        )
    return state


def chart_rounds(progress: dict[str, Any]) -> list[dict[str, Any]]:
    """A large copy's rounds as W&B charts them: round 0 is the copy carried on, measured before
    learning (continue_from); `replays_in_all` counts the replays of the copies carried on too."""
    offset = int(progress.get("continued_replays", 0))
    out: list[dict[str, Any]] = []
    if before := progress.get("before"):
        out.append(
            {
                "round": 0,
                "replays": 0,
                "replays_in_all": offset,
                "held_loss": before["held_loss"],
                "held_agreement": before["held_agreement"],
            }
        )
    for entry in progress.get("history", []):
        number, replays = int(entry["epoch"]), int(entry["replays"])
        out.append(
            {
                "round": number,
                "replays": replays,
                "replays_in_all": offset + replays,
                "held_loss": entry["held_loss"],
                "held_agreement": entry["held_agreement"],
                "train_loss": entry.get("train_loss"),
                "decisions": entry.get("decisions"),
                "replays_per_minute": entry.get("replays_per_minute"),
                "seconds_learning": entry.get("seconds_learning"),
            }
        )
    return out


def copy_for_charts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The run folder `root`'s copy of replays as charts (`nitro wandb`): a large copy's rounds
    (finished, or stopped part way), or the passes of a copy learned all at once (also `nitro pretrain`'s
    own runs). Empty when it has none."""
    finished, stopped = root / COPY / "state.pt", root / COPY / PROGRESS
    if finished.is_file():
        details = torch.load(finished, map_location="cpu", weights_only=True)["pretrained"]
    elif stopped.is_file():
        progress = torch.load(stopped, map_location="cpu", weights_only=True)
        made_with = progress.get("made_with", {})
        if "continued_replays" not in progress and made_with.get("continue_from"):
            other = root.parent / made_with["continue_from"] / COPY / "state.pt"
            if other.is_file():
                state = torch.load(other, map_location="cpu", weights_only=True)
                progress["continued_replays"] = _replays_in(state["pretrained"])
        return chart_rounds(progress), []
    elif (root / "pretrain.json").is_file():  # nitro pretrain
        details = json.loads((root / "pretrain.json").read_text(encoding="utf-8"))
    else:
        return [], []
    if "large" not in details:
        return [], list(details.get("epochs", []))
    progress = {
        "history": details.get("epochs", []),
        "before": details.get("before"),
        "continued_replays": details.get("continued_from", {}).get("replays", 0),
    }
    return chart_rounds(progress), []


def _replays_in(details: dict[str, Any]) -> int:
    """Replays a copy learned from, with those of the copies it carried on."""
    return int(details.get("replays", 0)) + int(details.get("continued_from", {}).get("replays", 0))


def _readable(path: Path, say: Callable[[str], None]) -> Any:
    """A saved file of the copy, or None when there is none or it cannot be read (cut off by a power cut
    before saves were flushed, say), which is said."""
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        say(f"The copy's {path.name} could not be read (cut off?): that part is made again.")
        return None


def _progress(work: Path, made_with: dict[str, Any], say: Callable[[str], None]) -> dict[str, Any] | None:
    """A stopped copy's progress, when it was made with these settings and its held-back samples are there."""
    progress = _readable(work / PROGRESS, say)
    if not isinstance(progress, dict) or not _same_copy(progress.get("made_with"), made_with):
        return None
    held = _readable(work / HELD, say)
    if not isinstance(held, dict) or not _same_copy(held.get("made_with"), made_with):
        return None
    progress["held_samples"] = held["samples"]
    return progress


def _save(path: Path, value: Any) -> None:
    """Written whole or not at all, even across a power cut (flushed to the disk before it replaces)."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def _duration(seconds: float) -> str:
    hours, minutes = divmod(round(seconds / 60), 60)
    if not hours and not minutes:
        return "under a minute"
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"
