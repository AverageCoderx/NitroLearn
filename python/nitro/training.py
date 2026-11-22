"""Builds everything a training run needs from a settings file, and runs it."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nitro import (
    __version__,
    cards_run,
    console,
    exploiter_folders,
    exploiters,
    fp16_sums,
    multigpu,
    pretrain,
    selector,
    tracking,
    transfer,
)
from nitro.config.schema import NitroConfig, RunConfig
from nitro.config.state_setters import StateSetter, metrics_label
from nitro.envs import EnvPool
from nitro.health import TrainingProblem, TrendWarnings
from nitro.learner import Learner, signal_columns
from nitro.metrics import MetricsLog
from nitro.model import ActorCritic, build_networks
from nitro.ppo import discount_from_half_life
from nitro.replay_mix import COLUMNS as MIX_COLUMNS
from nitro.replay_mix import KICKOFF_COLUMNS as MIX_KICKOFF_COLUMNS
from nitro.replay_mix import ReplayMix
from nitro.replay_mix import columns as mix_columns
from nitro.runs import (
    RunFolder,
    actions_problem,
    all_settings,
    fixed_settings,
    numbered,
    setting_changes,
)
from nitro.selector import Engine, SelectorEnvs, Specialist
from nitro.selector_learning import ModelLearning, model_columns, model_row
from nitro.selfplay import Frozen, SelfPlay
from nitro.trainer import Envs, StageTimes, Trainer, release_memory

log = logging.getLogger("nitro")


def _shown(line: str) -> None:
    """A line for the console as well as train.log: the copy of the replays ([pretrain]) takes a while."""
    log.info(line)
    print(line, flush=True)


_MASK64 = (1 << 64) - 1
# Settings only a new run reads: a resume does not report them as changed.
_STARTING_ONLY = (
    "start_from.enabled",
    "start_from.run",
    "start_from.at",
    "start_from.parts",
    "pretrain.enabled",
    "pretrain.replays",
    "pretrain.limit",
    "pretrain.passes",
    "pretrain.team_sizes",
    "pretrain.batch_replays",
    "pretrain.keep_batches",
    "pretrain.workers",
    "pretrain.continue_from",
    "pretrain.new_only",
)
# Gameplay statistics of the learner's cars per iteration (the core's PlayStats; an idea credited to
# RLMind): what the bots do, whatever they are paid for, so reward hacking and lost skills show.
PLAY_COLUMNS = (
    "in_air_share", "touches_per_minute", "speed_kmh", "speed_to_ball_kmh", "boost_held", "supersonic_share",
    "demolished_share", "ball_on_car_share", *(f"{name}_per_minute" for name in (
        "aerial_touches", "dribbles", "flicks", "double_taps", "flip_resets", "shots", "saves", "demos"
    )),
    "flip_reset_height", "flip_reset_goal_dist", "reset_musty_goals_per_hour", "reset_musty_goal_speed_kmh",
)  # fmt: skip
# Mechanics the core detects (nitro_core::mechanics), counted per car and minute of play.
MECHANICS = ("aerial_touches", "dribbles", "flicks", "double_taps", "flip_resets", "shots", "saves", "demos")


def play_columns(play: dict[str, float], tick_skip: int) -> dict[str, float | str]:
    """The gameplay columns of one iteration (blank when no counted car stepped)."""
    steps = play.get("car_steps", 0.0)
    if steps <= 0:
        return dict.fromkeys(PLAY_COLUMNS, "")
    minutes = steps * tick_skip / 120 / 60  # car-minutes of play
    return {
        "in_air_share": play["in_air"] / steps,
        "touches_per_minute": play["touches"] / minutes,
        "speed_kmh": play["speed"] / steps * 0.036,
        "speed_to_ball_kmh": play["toward_ball"] / steps * 0.036,
        "boost_held": play["boost"] / steps,
        "supersonic_share": play["supersonic"] / steps,
        "demolished_share": play["demolished"] / steps,
        "ball_on_car_share": play.get("ball_on_car", 0.0) / steps,
        **{f"{name}_per_minute": play.get(name, 0.0) / minutes for name in MECHANICS},
        # Where the flip resets happen: the ball's average height and distance from the opponents' net
        **{
            name: play.get(name, 0.0) / resets if (resets := play.get("flip_resets", 0.0)) > 0 else ""
            for name in ("flip_reset_height", "flip_reset_goal_dist")
        },
        # Reset musty goals: per car-hour, and the ball's average speed into the net (km/h)
        "reset_musty_goals_per_hour": play.get("reset_musty_goals", 0.0) / minutes * 60,
        "reset_musty_goal_speed_kmh": (
            play.get("reset_musty_goal_speed", 0.0) / goals * 0.036
            if (goals := play.get("reset_musty_goals", 0.0)) > 0
            else ""
        ),
    }


# Per game size in a run of several (env.team_sizes), each column named "<n>vn_<name>": the size's
# share of the counted car-steps, its finished games' mean length and goals per minute of those games, and
# its touches. Every column above counts all sizes together (steps per second: every player's step).
SIZE_COLUMNS = (
    "car_step_share", "episode_seconds", "goals_per_minute", "touches_per_minute", "aerial_touches_per_minute"
)  # fmt: skip


def size_column_names(sizes: Sequence[int]) -> list[str]:
    return [f"{n}v{n}_{name}" for n in sorted(sizes) for name in SIZE_COLUMNS]


def size_columns(
    sizes: Sequence[int],
    arena_sizes: Sequence[int],
    episodes: Sequence[Any],
    play_by_size: dict[int, dict[str, float]],
    tick_skip: int,
) -> dict[str, float | str]:
    """One iteration's per-size columns (blank where a size had no finished game or counted step)."""
    total = sum(play.get("car_steps", 0.0) for play in play_by_size.values())
    row: dict[str, float | str] = {}
    for n in sorted(sizes):
        own = [e for e in episodes if arena_sizes[e.env] == n]
        games = [e for e in own if e.car == 0]  # one record per game
        game_seconds = sum(e.seconds for e in games)
        play = play_by_size.get(n, {})
        played = play_columns(play, tick_skip)
        row |= {
            f"{n}v{n}_car_step_share": play.get("car_steps", 0.0) / total if total else "",
            f"{n}v{n}_episode_seconds": sum(e.seconds for e in own) / len(own) if own else "",
            f"{n}v{n}_goals_per_minute": (
                sum(e.goal != 0 for e in games) / (game_seconds / 60) if game_seconds > 0 else ""
            ),
            f"{n}v{n}_touches_per_minute": played["touches_per_minute"],
            f"{n}v{n}_aerial_touches_per_minute": played["aerial_touches_per_minute"],
        }
    return row


def sizes_note(arena_sizes: Sequence[int], sides: int) -> str:
    """Each game size's arenas and share of the player-steps, for the log."""
    rows = sum(arena_sizes) * sides
    parts = [
        f"{n}v{n} in {arena_sizes.count(n)} arenas ({arena_sizes.count(n) * n * sides / rows:.0%} of the "
        "player-steps)"
        for n in sorted(set(arena_sizes))
    ]
    return "Game sizes (env.team_sizes): " + ", ".join(parts) + "."


# With ball prediction: the share of decisions whose path was predicted afresh (a touch or a reset) and the
# share of the games' time spent predicting.
BALL_PREDICTION_COLUMNS = ("ball_prediction_fresh_share", "ball_prediction_time_share")
# Where an iteration's time goes: playing the games (of which waiting
# for the CPU workers and for the networks' choices), then learning. The wall clock's speed also counts the
# time between iterations (the run's own bookkeeping, saves), which steps_per_second leaves out.
TIME_COLUMNS = [
    "playing_seconds", "env_wait_seconds", "inference_seconds", "launch_seconds", "submit_seconds",
    "other_seconds", "learning_seconds", "playing_steps_per_second", "wall_steps_per_second",
]  # fmt: skip
SELFPLAY_COLUMNS = ["rating_mu", "rating_sigma", "pool_size", "pool_goal_share", "matchmaking"]
# Self-play read apart from the scripted (uneven) starts: only when some setter is one.
EVEN_START_COLUMNS = ["pool_goal_share_even", "pool_games_even", "start_edge"]


def resolve_device(run: RunConfig) -> torch.device:
    if run.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(run.device)


def configure_determinism(deterministic: bool) -> None:
    """Exact repeatability (`run.deterministic`): deterministic GPU kernels everywhere.

    The environments are always deterministic; this covers the networks. cuBLAS needs a
    fixed workspace layout, which must be set before it first starts in this process.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def arena_seed(run_seed: int, checkpoint: int) -> int:
    """Seed of the arenas built at checkpoint `checkpoint` (0: the run's own seed)."""
    if checkpoint == 0:
        return run_seed
    # SplitMix64 finaliser: well-spread seeds that depend only on the run seed and the checkpoint number.
    x = (run_seed + checkpoint * 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


def build_envs(
    config: NitroConfig, checkpoint: int, device: torch.device, views: Sequence[dict[str, Any]] = ()
) -> EnvPool:
    """The arenas of checkpoint `checkpoint`; `views`: a multi-model bot's models' views of them."""
    seed = arena_seed(config.run.seed, checkpoint)
    seeded = dataclasses.replace(config, run=dataclasses.replace(config.run, seed=seed))
    return EnvPool(seeded, pin_memory=device.type == "cuda", views=views)


def selector_seed(run_seed: int, checkpoint: int) -> int:
    """Seed of a multi-model bot's models' sampling in the games built at checkpoint `checkpoint`."""
    return arena_seed(run_seed, checkpoint) ^ 0x5E1EC7


@dataclass(frozen=True)
class TrainingSetup:
    envs: EnvPool
    model: ActorCritic
    trainer: Trainer
    gamma: float
    selfplay: SelfPlay
    selector: SelectorEnvs | None = None  # the games as a multi-model bot's selector sees them
    models: Engine | None = None  # a multi-model bot's models
    model_learning: ModelLearning | None = None  # their training and goal heads


def build(
    config: NitroConfig,
    state: dict[str, Any] | None = None,
    pool_folder: Path | None = None,
    start: transfer.Source | None = None,
    frozen: Frozen | None = None,
    specialists: Sequence[Specialist] = (),
) -> TrainingSetup:
    """A ready-to-train setup; from `state` (a checkpoint) it continues exactly where that left off.

    Self-play against past versions needs `pool_folder` (the run's `pool/`); without one it stays off. A new
    run (no `state`) starts from `start`'s networks when given (transfer learning). `frozen`: frozen networks
    with games of their own (exploiters). With `selector.specialists` set, `specialists` are the bots the
    selector chooses between (a multi-model bot).
    """
    device = resolve_device(config.run)
    fp16 = config.run.precision == "fp16" and device.type == "cuda"
    configure_determinism(config.run.deterministic)
    torch.manual_seed(config.run.seed)
    release_memory(device)  # what earlier runs in this process left behind (nitro ab, the task scripts)
    checkpoint = state["checkpoint"] if state else 0
    multi = bool(config.selector.specialists)
    if state and not multi and (problem := actions_problem(state["learner"]["model"])):
        raise ValueError(f"{problem}.")  # a selector's choices are its models, not the action table
    envs = build_envs(config, checkpoint, device, selector.views(specialists, config) if multi else ())
    games: Envs = envs
    chooser = None
    models = None
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    if multi:
        if len(specialists) != len(config.selector.specialists):
            raise ValueError("a multi-model run needs its models (nitro.selector.load_specialists)")
        models = selector.engine(
            config,
            envs,
            specialists,
            device,
            fp16,
            selector_seed(config.run.seed, checkpoint),
            record=config.selector.train_models or config.selector.goal_heads,
            pin_memory=device.type == "cuda",
            saved=state.get("selector") if state else None,
        )
        chooser = games = models.games
    obs_size, num_actions = games.groups[0].obs.shape[1], games.groups[0].mask.shape[1]
    game_size = envs.groups[0].obs.shape[1]
    model = (
        selector.network(config, game_size, num_actions)
        if multi
        else build_networks(config, obs_size, num_actions)
    ).to(device)
    model_learning = None
    if models is not None and (config.selector.train_models or config.selector.goal_heads):
        model_learning = ModelLearning(
            models.models, specialists, config, gamma, device, fp16, models.obs_sizes, models.goal_heads
        )
        if state and "selector" in state:
            model_learning.load_state_dict(state["selector"].get("learning", {}))
    if chooser is not None and state and "selector" in state:
        chooser.load_state_dict(state["selector"].get("games", {}))
    learner = Learner(model, config.ppo, gamma, device, fp16)
    learner.goal_gamma = discount_from_half_life(config.ppo.goal_critic_half_life, config.env.tick_skip)
    sums = config.run.fp16_sums if fp16 else "off"
    if sums != "off" and not fp16_sums.available():
        raise ValueError("run.fp16_sums needs PyTorch 2.7 or newer; set it to 'off'")
    learner.fp16_sums = sums
    if start is not None and not state:
        if problem := actions_problem(start.state["learner"]["model"], f"start_from: {start.label}"):
            raise transfer.TransferError(problem)
        transfer.apply(start, learner, config)
    if state:
        # A checkpoint from another device or precision still trains on, but not exactly as before: its
        # random numbers and (when the precision changed) its loss scaler start afresh.
        same_device = state.get("device", device.type) == device.type
        same_precision = state.get("fp16", fp16) == fp16
        learner.load_state_dict(state["learner"], load_scaler=same_precision)
        if same_device:
            # Restored before the inference graphs are captured, as at the checkpoint in the original run.
            torch.set_rng_state(state["rng_cpu"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["rng_cuda"])
        else:
            log.warning(
                "The run was saved on %s and continues on %s: it carries on learning, but not exactly as "
                "it would have (fresh random numbers%s).",
                state.get("device"),
                device.type,
                "" if same_precision else " and a fresh FP16 loss scale",
            )
    sides = 2 if config.env.opponents else 1
    selfplay = SelfPlay(
        config,
        pool_folder,
        model,
        device,
        envs.num_groups,
        [size * sides for size in envs.arena_team_sizes],
        [tensors.reward.numel() for tensors in envs.groups],
        frozen,
    )
    trainer = Trainer(
        games,
        model,
        learner,
        config.ppo.steps_per_iteration,
        device,
        fp16,
        cuda_graphs=device.type == "cuda",
        on_problem=config.run.on_training_problem,
        selfplay=selfplay if selfplay.in_use else None,
        pipelined=config.env.pipelined_inference,
        concurrent=config.env.concurrent_passes,
        card_starts=config.env.card_starts_games,
        fp16_sums=sums in fp16_sums.PLAYING,
    )
    return TrainingSetup(envs, model, trainer, gamma, selfplay, chooser, models, model_learning)


# Per-setter figures, present when the settings list more than one way of starting.
SETTER_COLUMNS = ("episodes", "goal_share", "episode_seconds", "touch_share")


def setter_labels(config: NitroConfig) -> list[str]:
    """One label per state setter for the metrics columns (`metrics_label`; a repeated label gets a
    number), or none when there is only one setter."""
    if len(config.state_setters) < 2:
        return []
    return numbered([metrics_label(setter) for setter in config.state_setters])


def per_setter(episodes: Sequence[Any], labels: Sequence[str]) -> dict[str, Any]:
    """How the episodes each setter started went this iteration, counted once per episode (its first car):
    how many, the share of goals its first car's team scored (blank when nobody scored), the mean length,
    and the share in which that car touched the ball."""
    row: dict[str, Any] = {}
    for index, label in enumerate(labels):
        mine = [e for e in episodes if e.setter == index and e.car == 0]
        scored = sum(1 for e in mine if e.goal > 0)
        decided = scored + sum(1 for e in mine if e.goal < 0)
        row[f"episodes_{label}"] = len(mine)
        row[f"goal_share_{label}"] = scored / decided if decided else ""
        row[f"episode_seconds_{label}"] = sum(e.seconds for e in mine) / len(mine) if mine else ""
        touched = sum(1 for e in mine if not math.isnan(e.first_touch))
        row[f"touch_share_{label}"] = touched / len(mine) if mine else ""
    return row


# Kickoffs by spot (the core's EpisodeRecord.kickoff: 0 the two diagonal spots, 1 the two off-centre ones, 2
# straight back; in 2v2/3v3 the most forward spot taken): present when a kickoff setter is listed.
KICKOFF_CLASSES = ("diagonal", "offcenter", "back")
KICKOFF_COLUMNS = [
    *(f"kickoff_touch_s_{name}" for name in KICKOFF_CLASSES),
    *(f"kickoff_count_{name}" for name in KICKOFF_CLASSES),
]


# The share of the graphics card's memory above which training warns that the card is nearly full.
MEMORY_WARN_SHARE = 0.9


def memory_note(reserved: int, total: int) -> str | None:
    """The plain warning when the networks' card holds more than MEMORY_WARN_SHARE of its memory (`reserved`
    of `total` bytes): on Windows a full card spills into the computer's own memory and crawls instead of
    stopping. None below that."""
    if total <= 0 or reserved <= MEMORY_WARN_SHARE * total:
        return None
    return (
        f"The graphics card's memory is nearly full ({reserved / 2**30:.1f} of {total / 2**30:.1f} GB in "
        "use). When it runs out, Windows moves the rest into the computer's own memory and training slows "
        "to a crawl. Lower env.num_arenas, set ppo.microbatch (for example 5000) so each learning step is "
        "worked out in parts, or make a new bot with smaller layers in [model]."
    )


def card_memory(device: torch.device) -> tuple[int, int] | None:
    """(reserved, total) bytes of the graphics card `device`; None on the processor."""
    if device.type != "cuda":
        return None
    return torch.cuda.memory_reserved(device), torch.cuda.get_device_properties(device).total_memory


def kickoff_columns_on(config: NitroConfig) -> bool:
    """Whether the run writes KICKOFF_COLUMNS: a kickoff setter is listed."""
    return any(s.name == "kickoff" for s in config.state_setters)


def kickoff_touches(episodes: Sequence[Any]) -> dict[str, float | int | str]:
    """Per kickoff class, the mean seconds until either car first touched the ball and how many kickoffs
    that mean is over (those touched before their episode ended), counted once per episode (its first car).
    Blank where none."""
    row: dict[str, float | int | str] = {}
    for index, name in enumerate(KICKOFF_CLASSES):
        times = [
            e.ball_first_touch
            for e in episodes
            if e.car == 0 and e.kickoff == index and not math.isnan(e.ball_first_touch)
        ]
        row[f"kickoff_touch_s_{name}"] = sum(times) / len(times) if times else ""
        row[f"kickoff_count_{name}"] = len(times)
    return row


def _unpacked(folder: RunFolder) -> bool:
    """Whether the bot was unpacked here from a pack (nitro unpack writes 'unpacked' in run.json)."""
    try:
        return "unpacked" in json.loads(folder.info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def even_starts(setters: Sequence[StateSetter]) -> list[bool] | None:
    """Per setter, whether it starts every car alike; None when all do (no columns to add)."""
    even = [setter.even_start for setter in setters]
    return None if all(even) else even


def start_edge(
    episodes: Sequence[Any],
    slot_of_arena: Callable[[int], int | None],
    active: Sequence[int | None],
    even: Sequence[bool],
) -> float | str:
    """Blue's share of the goals in the decided episodes an uneven setter started (a setter past `even`'s
    end counts as uneven) in arenas where the learner plays itself (no slot, or an empty one), counted once
    per episode (its first car): how far the start mix alone leans to Blue. Most scripted starts
    favour Blue; save favours Orange. Blank when none."""
    scored = decided = 0
    for e in episodes:
        if e.car != 0 or e.goal == 0 or (e.setter < len(even) and even[e.setter]):
            continue
        slot = slot_of_arena(e.env)
        if slot is not None and active[slot] is not None:
            continue
        decided += 1
        scored += e.goal > 0
    return scored / decided if decided else ""


def rotation_rng(run_seed: int, checkpoint: int) -> random.Random:
    """The random stream that picks the active past versions at checkpoint `checkpoint`: the same
    in a straight and in a resumed run."""
    return random.Random(arena_seed(run_seed, checkpoint) ^ 0x5E1F)


class TrainingRun:
    """A training run in its folder: iterations, metrics, warnings and checkpoints.

    With `state` (a checkpoint) it continues exactly where that checkpoint left off.
    """

    def __init__(
        self,
        config: NitroConfig,
        folder: RunFolder,
        state: dict[str, Any] | None = None,
        start: transfer.Source | None = None,
        applied: bool = False,
        advise: bool = True,
    ) -> None:
        # `advise`: log the settings' advice (off when the same run starts again after an exploiter phase).
        # `start`: what a new run starts from, instead of what its settings choose (a copy of replays so
        # far). `written` is the settings file as it is, which the saved settings and the "changed since the
        # save" warnings refer to. `applied`: started again after the dashboard's Apply, so the changes since
        # the save are the ones just applied, said as such rather than as warnings.
        self.written = config
        # With run.cards this process is card 0: `config` is its share of the games; `whole_config`
        # the run's, which the other cards take their shares of.
        self.whole_config = config
        multigpu.check(config, resolve_device(config.run))
        self.config = config = multigpu.share(config, 0)
        for note in config.advice() if advise else []:
            log.warning(note)
        self.folder = folder
        if config.selector.specialists and config.start_from.source and not state:
            raise selector.SelectorError(
                "a multi-model bot (selector.specialists) cannot start from another bot yet (start_from or "
                "--from)"
            )
        if config.pretrain.enabled and config.start_from.source and not state:
            raise transfer.TransferError(
                "pretrain.enabled and start_from (or --from) both choose what the run starts from; use one"
            )
        # Transfer learning: a new run may start from another run's networks, or from the copy of the
        # replays' players [pretrain] makes; a resume keeps its own.
        source = None
        if not state:
            source = (
                start
                or pretrain.start_source(config, folder, _shown)
                or transfer.find_source(config.start_from, folder.root.parent)
            )
        self.started_from = source.label if source else (state or {}).get("started_from")
        self.from_replays = (
            source.label == pretrain.COPY_LABEL
            if source
            else (state or {}).get("started_from_replays", False)
        )
        if source:
            log.info("Starting from %s (%s).", source.label, source.parts or config.start_from.parts)
            if not self.from_replays and source.parts == "policy" and config.start_from.parts != "policy":
                log.warning(transfer.policy_only_note(source))
            # What changes meaning with the tick skip; the copy of replays has no settings of its own.
            # The whole run's settings: on several cards `config` holds card 0's share of the steps.
            for note in [] if self.from_replays else transfer.rhythm_notes(source, self.whole_config):
                log.warning(note)
        elif state and config.start_from.source and not self.started_from:
            log.warning("start_from is read only when a run starts; this resumed run keeps its own networks.")
        elif not state and config.start_from.run and not config.start_from.enabled:
            log.info(
                "Transfer learning is off (start_from.enabled = false): starting from scratch, not from %s.",
                config.start_from.run,
            )
        if state and config.pretrain.enabled and not self.from_replays:
            log.warning("[pretrain] is read only when a run starts; this resumed run keeps its own networks.")
        if config.pretrain.value_warmup_steps and not config.pretrain.enabled:
            log.warning(
                "pretrain.value_warmup_steps acts only with pretrain.enabled = true; it does nothing here."
            )
        # A multi-model bot keeps its models in its saves, so it never depends on their bots again.
        if state and state.get("specialists"):
            self.specialists: list[Specialist] = list(selector.from_state(state["specialists"]))
        else:
            self.specialists = list(selector.load_specialists(config, folder.root.parent))
        if self.specialists:
            log.info("A multi-model bot over: %s.", "; ".join(s.label for s in self.specialists))
        steps = state["agent_steps"] if state else 0
        frozen = exploiters.frozen_for(config, folder.root, steps)
        setup = build(config, state, folder.root / "pool", source, frozen, self.specialists)
        self.trainer = setup.trainer
        self.envs = setup.envs
        self.model = setup.model
        self.selfplay = setup.selfplay
        self.selector = setup.selector
        self.selector_labels = selector.labels(self.specialists)
        self.models = setup.models
        self.model_learning = setup.model_learning
        if config.env.team_sizes:
            log.info(sizes_note(self.envs.arena_team_sizes, 2))
            if not state and not source:
                log.warning(
                    "Every game size from scratch (env.team_sizes): the first touches are spread over all "
                    "sizes, and a run like this was seen to stall (Pulsar). Training 1v1 first (with "
                    "env.obs_team_size = %d) and adding the bigger sizes on a resume is the advice.",
                    max(config.env.team_sizes),
                )
        self.checkpoint_index = state["checkpoint"] if state else 0
        self.said_saves_over = False  # the saves disk cap warned once already
        self.said_not_rated = False  # a multi-model bot's skipped ratings were said once already
        self.agent_steps = state["agent_steps"] if state else 0
        self.iterations = state["iterations"] if state else 0
        self.trainer.iterations = self.iterations  # train.log numbers iterations as metrics.csv does
        # The agent step the bot's next exploiter phase begins at (None: off, or an exploiter's run).
        self.exploiter_next = exploiters.next_phase(config, state, self.agent_steps, folder.root)
        if self.exploiter_next is not None:
            exploiter_folders.write_next(folder.root, self.exploiter_next)
        # An exploiter's run: its recent games against the bot, which end its phase.
        self.exploiter_tally = exploiters.tally_for(config, folder.root, state)
        # The bot's recent games against each exploiter it plays, which retire it (retire_share).
        self.exploiter_scores = exploiters.scores_for(config, folder.root, state, self.selfplay.frozen)
        self.exploiters_retired = exploiters.retired_count(config, folder.root, self.agent_steps)
        if state and "selfplay" in state:
            self.selfplay.load_state_dict(state["selfplay"])  # ratings, games and the pool as they were
            self.selfplay.truncate_log(self.agent_steps)  # games recorded after the save are replayed
        # The active past versions are drawn from the checkpoint number, exactly as the run that saved the
        # checkpoint drew them right after saving, so a resumed run meets the same opponents.
        self.selfplay.rotate(rotation_rng(config.run.seed, self.checkpoint_index))
        self._count_learner_cars_only()
        # Copy steps on converted replays after each update ([replay_mix]). A save's copy steps state is
        # kept through later saves while it is off, so switching it off and on again carries on.
        self.replay_mix_state: dict[str, Any] | None = state.get("replay_mix") if state else None
        self.replay_mix: ReplayMix | None = None
        if config.replay_mix.active:
            self.replay_mix = ReplayMix(
                config,
                self.model,
                self.trainer.rollout.obs.shape[-1],
                self.trainer.device,
                self.trainer.fp16,
                self.replay_mix_state,
                moved=_unpacked(folder),
            )
            self.trainer.replay_mix = self.replay_mix
            self.replay_mix_state = None  # in the object now: its state_dict is what is saved
        for line in self.selfplay.resume_info:
            log.info(line)
        self.missing_notes = list(self.selfplay.resume_notes)  # what the save expects that is not here
        self.resumed = bool(state)
        self.labels = numbered(self.envs.reward_labels)  # two all_of groups keep two columns
        self.even_starts = even_starts(config.state_setters) if self.selfplay.enabled else None
        self.goal_reward = next((r for r in config.active_rewards if r.name == "goal"), None)
        columns = [
            "agent_steps", "iteration", "steps_per_second", "reward_per_step",
            *(f"raw_{label}" for label in self.labels),
            # What each reward pays per step after zero-sum and weight, and its size (the reward shares).
            *(f"paid_{label}" for label in self.labels),
            *(f"paid_abs_{label}" for label in self.labels),
            # Its size before zero-sum: what the bots earn themselves.
            *(f"own_abs_{label}" for label in self.labels),
            # The share of agent steps it paid on before its weight: how often it fires.
            *(f"fired_{label}" for label in self.labels),
            "episode_seconds", "entropy", "policy_change", "approx_kl", "clip_fraction", "explained_variance",
            "value_loss", "policy_loss", "skipped_updates", "grad_scale", "reward_scale", "policy_lr",
            # The critic's value of a fresh start, raw and per goal.
            "start_value", *(["start_value_share"] if self.goal_reward is not None else []),
            "update_size", *(f"grad_norm_{name}" for name in self.trainer.learner.groups),
            *(f"update_norm_{name}" for name in self.trainer.learner.groups),
            # Each network's size (update_norm / weight_norm: its relative step); over the layers a
            # LayerNorm follows, the turn per update and the growth from an orthogonal start (else blank).
            *(f"weight_norm_{name}" for name in self.trainer.learner.groups),
            *(f"turn_{name}" for name in self.trainer.learner.scale_free),
            *(f"growth_{name}" for name in self.trainer.learner.scale_free), *TIME_COLUMNS,
            *PLAY_COLUMNS,
            *size_column_names(config.env.team_sizes),
            *(BALL_PREDICTION_COLUMNS if config.env.ball_prediction else []),
            *signal_columns(config.ppo),  # the GCO learning signals'
            *(MIX_COLUMNS if self.replay_mix is not None else []),
            *(MIX_KICKOFF_COLUMNS if self.replay_mix is not None and self.replay_mix.kickoff_only else []),
            *(SELFPLAY_COLUMNS if self.selfplay.enabled else []),
            *exploiters.columns(config, folder.root, self.selfplay.frozen),
            *(EVEN_START_COLUMNS if self.even_starts is not None else []),
            *(f"{name}_{label}" for label in setter_labels(config) for name in SETTER_COLUMNS),
            *(KICKOFF_COLUMNS if kickoff_columns_on(config) else []),
            *(["selector_switches_per_minute"] if self.selector else []),
            *(f"selector_share_{label}" for label in self.selector_labels),
            *(
                f"selector_bonus_{label}"
                for label in (self.selector_labels if any(config.selector.min_share) else [])
            ),
            *model_columns(
                self.selector_labels,
                config.selector.train_models,
                config.selector.goal_heads and self.selector is not None,
            ),
        ]  # fmt: skip
        self.setter_labels = setter_labels(config)
        self.metrics = MetricsLog(folder.metrics, columns, until=self.agent_steps)
        # A selector's held steps have one allowed choice and no entropy, so its mean entropy is lower.
        self.warnings = TrendWarnings(
            config.selfplay.flat_rating_steps if self.selfplay.enabled else 0,
            0.1 / self.selector.hold_steps if self.selector else 0.1,
        )
        self.recent_warnings: list[str] = []
        self.console_warnings: list[str] = []  # new since the last console report
        for note in self.missing_notes:
            log.warning(note)
            self.recent_warnings = [*self.recent_warnings[-4:], note]
        self.console_warnings += self.missing_notes
        self.last_times: StageTimes | None = None
        self.last_row: tuple[float, int] | None = None  # when the last row was written, at how many steps
        self.warming_up = True  # the first iteration after the games are (re)built
        self.memory_checked = False  # the card's memory is checked once, after the first iteration
        self.saved_at_steps = state["agent_steps"] if state else -1
        self.copy_so_far: dict[str, Any] | None = None  # set: its saves hold an unfinished copy
        self.cards = cards_run.Leader(self) if config.run.cards > 1 else None  # starts at the first iteration
        log.info("Networks run on %s.", self.device_note)
        if state and "all_settings" in state:
            saved, now = (
                {k: v for k, v in settings.items() if k not in _STARTING_ONLY}
                for settings in (state["all_settings"], all_settings(self.written, folder.root))
            )
            moved = "unpacked" in json.loads(folder.info.read_text(encoding="utf-8"))
            # A setting newer than the save, left at its default, changes nothing the save knew.
            defaults = all_settings(NitroConfig(), folder.root)
            newer = {k for k in now.keys() - saved.keys() if now[k] == defaults.get(k)}
            for key, before, after in setting_changes(
                saved, {k: v for k, v in now.items() if k not in newer}, moved
            ):
                if applied:
                    _shown(f"Applied: {key} changed from {before} to {after}.")
                else:
                    log.warning("Setting changed since the save: %s: %s -> %s", key, before, after)

    def _refuse_missing(self) -> None:
        """run.missing_data = 'stop': no training when the self-play notes found something or when
        replay_mix's folders hold no converted replays. Checked when training starts, so what only plays or
        saves (reward-check, a copy's save) is never refused."""
        if self.config.run.missing_data != "stop":
            return
        notes = list(self.missing_notes)
        if self.replay_mix is not None and (why := self.replay_mix.missing()):
            notes.append(why)
        if notes:
            raise TrainingProblem(
                "run.missing_data = 'stop', so training does not start: some of what this bot trains with is "
                "not here.\n- "
                + "\n- ".join(notes)
                + "\nOnce that is fixed, start it again; or set run.missing_data = 'warn' to train on "
                "without it."
            )

    @property
    def device_note(self) -> str:
        """Where the networks run and in what precision, as actually used: "cuda (FP16)" or "cpu (FP32)"."""
        note = f"{self.trainer.device.type} ({'FP16' if self.trainer.fp16 else 'FP32'})"
        leader = getattr(self, "cards", None)
        cards = leader.size if leader is not None else 1
        if cards > 1:
            note += f" on {cards} cards"
        return note

    def iteration(self) -> dict[str, Any]:
        """One iteration; returns the row it added to metrics.csv. Warnings go to the log."""
        trained_at = self.agent_steps
        # A bot pretrained from replays learns its critic alone at first (pretrain.value_warmup_steps).
        warmup = self.config.pretrain.value_warmup_steps
        self.trainer.learner.policy_frozen = bool(self.from_replays and trained_at < warmup)
        if self.replay_mix is not None:
            self.replay_mix.start()  # its replays are made ready while the games play
        if self.cards is None:
            self.trainer.learner.restart_twins(trained_at)  # ppo.vdag_reinit_steps
        models_rate = self._selector_turn()
        with self.cards.iteration(trained_at) if self.cards is not None else contextlib.nullcontext():
            # Several cards: once the helpers heard of this iteration, so they restart with it.
            if self.cards is not None:
                self.trainer.learner.restart_twins(trained_at)
            report = self.trainer.iteration()
        learned = self._models_learn(models_rate)
        # Every card played as many steps as this one; the games' columns below are this card's.
        cards = self.config.run.cards
        env = self.envs.take_stats()
        self.agent_steps += report.agent_steps * cards
        self.iterations += 1
        self.selfplay.maybe_snapshot(self.agent_steps)  # a copy joins the pool every snapshot_steps
        per_step = max(env.agent_steps, 1)
        total_abs = sum(env.weighted_abs) or 1.0
        shares = {
            label: value / total_abs for label, value in zip(self.labels, env.weighted_abs, strict=True)
        }
        stats, episodes = report.learn, env.episodes
        row: dict[str, Any] = {
            "agent_steps": self.agent_steps,
            "iteration": self.iterations,
            "steps_per_second": round(report.steps_per_second * cards),
            "reward_per_step": sum(env.weighted) / per_step,
            **{f"raw_{label}": value / per_step for label, value in zip(self.labels, env.raw, strict=True)},
            **{
                f"paid_{label}": value / per_step
                for label, value in zip(self.labels, env.weighted, strict=True)
            },
            **{
                f"paid_abs_{label}": value / per_step
                for label, value in zip(self.labels, env.weighted_abs, strict=True)
            },
            **{
                f"own_abs_{label}": value / per_step
                for label, value in zip(self.labels, env.own_abs, strict=True)
            },
            **{
                f"fired_{label}": count / per_step
                for label, count in zip(self.labels, env.fired, strict=True)
            },
            "episode_seconds": sum(e.seconds for e in episodes) / len(episodes) if episodes else "",
            "entropy": stats.entropy,
            "policy_change": stats.policy_change,
            "approx_kl": stats.approx_kl,
            "clip_fraction": stats.clip_fraction,
            "explained_variance": stats.explained_variance,
            "value_loss": stats.value_loss,
            "policy_loss": stats.policy_loss,
            "skipped_updates": stats.skipped_updates,
            "grad_scale": stats.grad_scale,
            "reward_scale": stats.reward_scale,
            "policy_lr": stats.policy_lr,
            "start_value": stats.start_value if math.isfinite(stats.start_value) else "",
            "update_size": stats.param_change,
            **{f"grad_norm_{name}": norm for name, norm in stats.grad_norms.items()},
            **{f"update_norm_{name}": norm for name, norm in stats.update_norms.items()},
            **{f"weight_norm_{name}": norm for name, norm in stats.weight_norms.items()},
            **{f"turn_{name}": turn for name, turn in stats.turns.items()},
            **{f"growth_{name}": growth for name, growth in stats.growths.items()},
            "playing_seconds": report.times.collect,
            "env_wait_seconds": report.times.env_wait,
            "inference_seconds": report.times.inference,
            # The rest of playing: graphs to the driver, steps to the workers, Python.
            "launch_seconds": report.times.launch,
            "submit_seconds": report.times.submit,
            "other_seconds": report.times.other,
            "learning_seconds": report.times.learn,
            "playing_steps_per_second": (
                round(cards * report.agent_steps / report.times.collect) if report.times.collect > 0 else ""
            ),
            **play_columns(env.play, self.config.env.tick_skip),
        }
        if self.config.env.team_sizes:
            row |= size_columns(
                self.config.env.team_sizes,
                self.envs.arena_team_sizes,
                episodes,
                env.play_by_size,
                self.config.env.tick_skip,
            )
        if self.goal_reward is not None:  # per the goal's weight
            weight = self.goal_reward.weight
            share = stats.start_value / weight if weight != 0 else math.nan
            row["start_value_share"] = share if math.isfinite(share) else ""
        if self.config.env.ball_prediction:  # how often the games predicted afresh, and its time
            row["ball_prediction_fresh_share"] = env.predictions_fresh / max(env.env_steps, 1)
            row["ball_prediction_time_share"] = env.prediction_seconds / max(env.step_seconds, 1e-9)
        row |= stats.signals
        if self.replay_mix is not None:
            row |= mix_columns(report.copy)
        if self.setter_labels:
            row |= per_setter(episodes, self.setter_labels)
        if self.selector:
            row |= self._selector_columns()
            row |= learned
        if kickoff_columns_on(self.config):
            row |= kickoff_touches(episodes)
        matches = episodes
        games = self.selfplay.record(matches, self.agent_steps, self.even_starts)
        rating = None
        if self.selfplay.enabled:
            row |= self.selfplay.metrics()
            row["pool_goal_share"] = games.wins / games.games if games.games else ""
            row["matchmaking"] = games.matchmaking if games.games else ""
            rating = (self.selfplay.learner.mu, self.agent_steps)
        if self.even_starts is not None:
            row["pool_goal_share_even"] = games.even_wins / games.even_games if games.even_games else ""
            row["pool_games_even"] = games.even_games
            row["start_edge"] = start_edge(
                matches, self.selfplay.slot_of_arena, self.selfplay.active, self.even_starts
            )
        if self.exploiter_tally is not None:
            self.exploiter_tally.add(scored for _, scored in self.selfplay.take_frozen_results())
            row |= self.exploiter_tally.columns()
        if self.exploiter_scores is not None:
            frozen = self.selfplay.frozen
            assert frozen is not None
            self.exploiter_scores.add(
                (frozen.numbers[slot], s) for slot, s in self.selfplay.take_frozen_results()
            )
            row |= self.exploiter_scores.columns()
        if self.exploiters_retired is not None:  # counted with this iteration's retirements
            beaten = len(self.exploiter_scores.beaten()) if self.exploiter_scores is not None else 0
            row[exploiter_folders.RETIRED_COLUMN] = self.exploiters_retired + beaten
        row |= self.selfplay.frozen_metrics()  # exploiters' games, or an exploiter's against the bot
        # Agent steps since the last row over the wall clock: all of the time between iterations counted too.
        now = time.perf_counter()
        row["wall_steps_per_second"] = ""
        if self.last_row is not None and now > self.last_row[0]:
            last, steps = self.last_row
            row["wall_steps_per_second"] = round((self.agent_steps - steps) / (now - last))
        self.last_row = (now, self.agent_steps)
        self.metrics.write(row)
        speed = None if self.warming_up else report.steps_per_second * cards
        self.warming_up = False
        notices = self.trainer.notices[:]  # already logged by the trainer when they happened
        self.trainer.notices.clear()
        if self.replay_mix is not None:  # likewise
            notices += self.replay_mix.notices
            self.replay_mix.notices.clear()
        seconds = row["episode_seconds"] if row["episode_seconds"] != "" else None
        warnings = self.warnings.check(stats, speed, shares, rating, seconds)
        if not self.memory_checked:
            self.memory_checked = True
            memory = card_memory(self.trainer.device)
            if memory is not None and (full := memory_note(*memory)):
                warnings.append(full)
        for warning in warnings:
            log.warning(warning)
            notices.append(warning)
        for notice in notices:
            self.recent_warnings = [*self.recent_warnings[-4:], notice]
        self.console_warnings += notices
        self.last_times = report.times
        return row

    def _housekeeping(self) -> None:
        """After a save: the saves' disk cap and metrics.csv's size cap (run.saves_disk_gb,
        metrics_max_mb)."""
        run = self.config.run
        if run.saves_disk_gb > 0:
            deleted, over = self.folder.cap_saves(run.saves_disk_gb, run.milestone_steps)
            for line in deleted:
                print(f"Disk cap: {line}.", flush=True)
                log.info("Disk cap: %s", line)
            if over and not self.said_saves_over:
                self.said_saves_over = True
                log.warning(
                    "The saves still take more than run.saves_disk_gb = %g GB, but the ones left are never "
                    "deleted (the newest, and saves marked to keep): raise the cap or free space by hand.",
                    run.saves_disk_gb,
                )
        if run.metrics_max_mb > 0 and self.metrics.compact(run.metrics_max_mb * 1e6):
            line = (
                f"metrics.csv was over {run.metrics_max_mb:g} MB (run.metrics_max_mb): every other row "
                "dropped (the first and newest kept)"
            )
            print(f"{line}.", flush=True)
            log.info(line)

    def _rate_strength(self, state: dict[str, Any], path: Path) -> None:
        """The new save's rating against the pinned anchors (strength.anchors); a failure is logged and
        training goes on."""
        settings = self.config.strength
        if settings.anchors <= 0 or (self.checkpoint_index - 1) % settings.rate_every:
            return
        if why := selector.not_yet(self.config, "rated against anchors"):
            if not self.said_not_rated:
                log.info("The strength chart (strength.anchors) is skipped for this bot: %s.", why)
                self.said_not_rated = True
            return
        from nitro import strength

        start = time.monotonic()
        try:
            entry = strength.rate_save(self.folder, state, self.config, path)
        except Exception as err:  # a chart must never stop training
            log.warning("The strength rating of this save was skipped: %s", err)
            return
        took = time.monotonic() - start
        pinned = (path / strength.ANCHOR).is_file()
        if entry is None:
            log.info("Strength: this save is the first anchor (rating 0 by definition).")
        else:
            log.info(
                "Strength: %.0f (95%% range %.0f to %.0f) against %d anchor(s), in %.1f s%s.",
                entry["rating"],
                entry["low"],
                entry["high"],
                len(entry["matches"]),
                took,
                "; pinned as a new anchor" if pinned else "",
            )

    def checkpoint(self, continue_after: bool = True, kept: bool = False) -> Path:
        """Saves the run; when continuing, every game starts afresh and the past versions
        playing against the bot are re-drawn."""
        if self.cards is not None:  # networks not checked since an iteration are checked first
            self.cards.check_before_save()
        state = self.state()
        run = self.config.run
        path = self.folder.save_checkpoint(
            state,
            self.agent_steps,
            run.keep_checkpoints,
            run.milestone_steps,
            kept=kept,
            copy_so_far=self.copy_so_far,
        )
        self._housekeeping()
        self.checkpoint_index += 1
        self.saved_at_steps = self.agent_steps
        if self.replay_mix is not None and self.replay_mix.off:  # said again: it stays in the report's list
            log.warning(self.replay_mix.off)
            self.recent_warnings = [*self.recent_warnings[-4:], self.replay_mix.off]
            self.console_warnings.append(self.replay_mix.off)
        self._rate_strength(state, path)
        if run.skills_every > 0 and self.checkpoint_index % run.skills_every == 0:
            self.score_skills(state)
        if continue_after:
            self.selfplay.rotate(rotation_rng(self.config.run.seed, self.checkpoint_index))

            def rebuild() -> Envs:
                self.envs = build_envs(
                    self.config,
                    self.checkpoint_index,
                    self.trainer.device,
                    selector.views(self.specialists, self.config) if self.specialists else (),
                )
                if self.selector is None:
                    return self.envs
                return self.selector.attach(
                    self.envs, selector_seed(self.config.run.seed, self.checkpoint_index)
                )

            del self.envs  # the trainer lets the old games go before `rebuild` makes the new ones
            if self.selector is not None:
                self.selector.release()
            self.trainer.restart(rebuild)
            if self.model_learning is not None:
                self.model_learning.games_rebuilt()  # the goal guessers' waiting states end with their games
            self._count_learner_cars_only()
            if self.cards is not None:  # the other cards build theirs before the next iteration
                self.cards.rebuilt = self.checkpoint_index
            self.warming_up = True
        return path

    def score_skills(self, state: dict[str, Any]) -> None:
        """run.skills_every: the save just made through `nitro skills`' battery on the training device,
        the same situations every time. Never stops training: a problem is reported and training goes on."""
        from nitro.eval import Side
        from nitro.skills import run_battery, save_results
        from nitro.tasks import EVAL_SEEDS

        if why := selector.not_yet(self.config, "scored on skills"):
            if not self.said_not_rated:
                log.info("Skills on a schedule (run.skills_every) are skipped for this bot: %s.", why)
                self.said_not_rated = True
            return
        games, seed = self.config.run.skills_games, EVAL_SEEDS[0]
        start = time.perf_counter()
        skipped: list[str] = []
        try:
            config = self.folder.load_settings(state)
            device = self.trainer.device
            run = dataclasses.replace(config.run, device="cuda" if device.type == "cuda" else "cpu")
            config = dataclasses.replace(config, run=run)
            # The battery's own random draws leave training's exactly where they were.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                side = Side(self.folder.root.name, self.agent_steps, state, config)
                results = run_battery(side, games=games, seed=seed, skipped=skipped)
            save_results(self.folder.root, self.agent_steps, results, seed, games)
        except Exception as err:  # scoring must never end a run
            log.warning(
                "Could not score this save's skills (run.skills_every); training carries on. The problem: %s",
                err,
            )
            return
        log.info(
            "Skills at %s agent steps (%d situations each, took %.0f s; saved in skills.csv):\n  %s",
            f"{self.agent_steps:,}",
            games,
            time.perf_counter() - start,
            "\n  ".join([r.line() for r in results] + skipped),
        )

    def state(self) -> dict[str, Any]:
        """Everything a save holds: the run continues exactly from it."""
        cuda = self.trainer.device.type == "cuda"
        state = {
            "learner": self.trainer.learner.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if cuda else torch.empty(0, dtype=torch.uint8),
            "checkpoint": self.checkpoint_index + 1,
            "agent_steps": self.agent_steps,
            "iterations": self.iterations,
            "settings": fixed_settings(self.written),
            "all_settings": all_settings(self.written, self.folder.root),
            "device": self.trainer.device.type,
            "fp16": self.trainer.fp16,
            "nitro_version": __version__,  # which release wrote it
            "selfplay": self.selfplay.state_dict(),
            "started_from": self.started_from,
            "started_from_replays": self.from_replays,
        }
        if self.exploiter_next is not None:  # when the next exploiter phase begins
            every = self.config.exploiters.every_steps
            state[exploiter_folders.SAVED] = {"next_at": self.exploiter_next, "every": every}
        if self.exploiter_tally is not None:
            state[exploiter_folders.TALLY] = self.exploiter_tally.state_dict()
        if self.exploiter_scores is not None:
            state[exploiter_folders.SCORES] = self.exploiter_scores.state_dict()
        mix = self.replay_mix.state_dict() if self.replay_mix is not None else self.replay_mix_state
        if mix is not None:
            state["replay_mix"] = mix  # the copy steps' optimizer and place in the replays
        if self.specialists:
            state["specialists"] = selector.to_state(self._current_specialists())
            state["selector"] = {
                "games": self.selector.state_dict() if self.selector is not None else {},
                "learning": self.model_learning.state_dict() if self.model_learning is not None else {},
            }
        return state

    def _current_specialists(self) -> list[Specialist]:
        """The models as they are now: trained ones (selector.train_models) with their new weights and the
        weights they started from."""
        learning = self.model_learning
        if learning is None or not learning.learners:
            return list(self.specialists)
        out = []
        for specialist, learner in zip(self.specialists, learning.learners, strict=True):
            anchor = learner.anchor
            assert anchor is not None
            out.append(
                dataclasses.replace(
                    specialist,
                    weights={k: v.detach().cpu().clone() for k, v in learner.model.state_dict().items()},
                    start=specialist.start
                    or {k: v.detach().cpu().clone() for k, v in anchor.state_dict().items()},
                )
            )
        return out

    def _learner_rows(self) -> torch.Tensor | None:
        """The learner's own cars (bool per agent), or None for every car: not past versions."""
        return self.selfplay.learner_rows if self.selfplay.in_use else None

    def _selector_turn(self) -> float:
        """Before a multi-model bot's collection: which cars are the learner's, recording started, and with
        train_schedule = "alternate" whose turn it is. The share of model_lr the models learn with this
        time: 0 when they do not (off, the selector's warm-up, or the selector's turn)."""
        if self.selector is None:
            return 0.0
        self.selector.learner_rows = self._learner_rows()
        self.selector.begin(self.trainer.steps)
        settings = self.config.selector
        rate = settings.models_rate_at(self.agent_steps)
        if rate == 0 or settings.train_schedule == "together":
            return rate  # in the warm-up the selector learns every iteration
        models_turn = (self.iterations // settings.alternate_iterations) % 2 == 0
        if models_turn:  # the selector's policy is held while the models learn; its value keeps learning
            self.trainer.learner.policy_frozen = True
        return rate if models_turn else 0.0

    def _models_learn(self, models_rate: float) -> dict[str, Any]:
        """The models' and goal guessers' update after a collection, as metrics columns."""
        learning, chooser = self.model_learning, self.selector
        if learning is None or chooser is None or chooser.recorded is None:
            return {}
        stats, accuracy = learning.learn(chooser.recorded, self._learner_rows(), models_rate)
        return model_row(self.selector_labels, stats, accuracy, models_rate)

    def _selector_columns(self) -> dict[str, Any]:
        assert self.selector is not None
        stats = self.selector.take_stats()
        self.selector.end_collection(stats.driven)
        steps = sum(stats.driven)
        minutes = steps * self.config.env.tick_skip / 120 / 60  # car-minutes
        row: dict[str, Any] = {
            "selector_switches_per_minute": stats.switches / minutes if minutes else "",
            **{
                f"selector_share_{label}": driven / steps if steps else ""
                for label, driven in zip(self.selector_labels, stats.driven, strict=True)
            },
        }
        if self.selector.bonus is not None:
            row |= {
                f"selector_bonus_{label}": float(bonus)
                for label, bonus in zip(self.selector_labels, self.selector.bonus.bonus.tolist(), strict=True)
            }
        return row

    def _count_learner_cars_only(self) -> None:
        """Keeps the frozen opponents' cars out of the reward statistics (metrics, warnings)."""
        for arena, counted in self.selfplay.counted_cars().items():
            self.envs.set_counted(arena, counted)

    def write_report(self) -> None:
        env = self.config.env
        notes = [
            f"{env.sizes_label}, {env.num_arenas * self.config.run.cards} arenas, networks on "
            f"{self.device_note}",
            f"{self.agent_steps:,} agent steps in {self.iterations:,} iterations",
            *(
                [f"Started from {self.started_from}" + ("" if self.from_replays else " (transfer learning)")]
                if self.started_from
                else []
            ),
            *(
                [
                    f"Copying converted replays in {self.config.replay_mix.share:.0%} of the policy's "
                    f"learning steps (replay_mix): {self.replay_mix.steps:,} copy steps in all"
                ]
                if self.replay_mix is not None and not self.replay_mix.off
                else []
            ),
            *(
                [
                    "replay_mix is on in the settings but copies nothing: "
                    + self.replay_mix.off.removeprefix("replay_mix: ")
                ]
                if self.replay_mix is not None and self.replay_mix.off
                else []
            ),
            *(f"When resumed: {note}" for note in self.missing_notes),  # what the save expected
            *([f"Phase: {phase}"] if (phase := exploiters.phase_text(self)) else []),
            *exploiters.results(self),
            "Every save restarts all games from a kickoff (so a resumed run continues exactly), which shows "
            "as a small blip in some charts at each save.",
        ]
        self.metrics.write_report(self.folder.report, self.folder.root.name, notes, self.recent_warnings)

    @property
    def exploiter_due(self) -> bool:
        """Whether the bot has reached its next exploiter phase or clearly beats an exploiter it plays, or an
        exploiter's phase is over: it saves and stops."""
        if self.exploiter_tally is not None:
            return self.exploiter_tally.done(self.agent_steps)
        if self.exploiter_scores is not None and self.exploiter_scores.beaten():
            return True
        return self.exploiter_next is not None and self.agent_steps >= self.exploiter_next

    def _save_due(self, since_save: float) -> bool:
        """Every run.checkpoint_steps agent steps when set (a save whenever a multiple is passed), else
        every run.checkpoint_minutes."""
        every = self.config.run.checkpoint_steps
        if every:
            return self.agent_steps // every > max(self.saved_at_steps, 0) // every
        return since_save >= 60 * self.config.run.checkpoint_minutes

    def _next_save(self, since_save: float) -> str:
        every = self.config.run.checkpoint_steps
        if every:
            left = (self.agent_steps // every + 1) * every - self.agent_steps
            return f"in {console.steps_text(left)} agent steps (every {console.steps_text(every)})"
        minutes = self.config.run.checkpoint_minutes
        return f"in about {max(60 * minutes - since_save, 0) / 60:.0f} min (every {minutes:g} min)"

    def _console(self, rows: list[dict[str, Any]], speed: float, running: float, since_save: float) -> str:
        """The console's line or report (run.console) for the iterations since the last one."""
        row = rows[-1]
        phase = exploiters.phase_text(self)
        if self.config.run.console == "line":
            return (
                (f"{phase[0].upper()}{phase[1:]} | " if phase else "")
                + f"{self.agent_steps / 1e6:,.1f}M steps | {speed:,.0f} steps/s | "
                f"reward per step {row['reward_per_step']:.4f} | entropy {row['entropy']:.2f} | "
                f"value prediction {row['explained_variance']:.2f}"
            )
        times = self.last_times
        facts = console.Facts(
            name=self.folder.root.name,
            running_seconds=running,
            steps_per_second=speed,
            collect_seconds=times.collect if times else 0.0,
            learn_seconds=times.learn if times else 0.0,
            reward_labels=self.labels,
            started_from=self.started_from or "",
            transfer=bool(self.started_from) and not self.from_replays,
            next_save=self._next_save(since_save),
            warnings=self.console_warnings,
            phase=phase,
        )
        return "\n".join(console.iteration_report(console.averaged(rows), facts))

    def run(self, stop: threading.Event, until_steps: int | None = None) -> None:
        """Trains until `stop` is set (or `until_steps` agent steps), then saves and writes the report."""
        self._refuse_missing()  # before anything plays or is tracked
        began = last_save = last_report = last_print = time.monotonic()
        printed_steps = self.agent_steps
        since_print: list[dict[str, Any]] = []
        tracker = tracking.start(
            self.config.wandb,
            self.folder.root,
            all_settings(self.written, self.folder.root),
            self.setter_labels,
        )
        try:
            while (
                not stop.is_set()
                and (until_steps is None or self.agent_steps < until_steps)
                and not self.exploiter_due  # it saves and stops for a phase, a phase's end or a retirement
            ):
                row = self.iteration()
                tracker.log(row)
                since_print.append(row)
                now = time.monotonic()
                if now - last_print >= self.config.run.console_seconds:
                    speed = (self.agent_steps - printed_steps) / (now - last_print)
                    print(self._console(since_print, speed, now - began, now - last_save), flush=True)
                    last_print, printed_steps = now, self.agent_steps
                    since_print.clear()
                    self.console_warnings.clear()
                if now - last_report >= 60 * self.config.run.report_minutes:
                    self.write_report()
                    last_report = now
                if self._save_due(now - last_save):
                    self.checkpoint()
                    last_save = time.monotonic()
            # Nothing new to save when the last iteration was also a periodic save.
            if self.saved_at_steps != self.agent_steps:
                self.checkpoint(continue_after=False)
        except BaseException:
            self._finish(tracker, quietly=True)
            raise
        self._finish(tracker, quietly=False)

    def _finish(self, tracker: tracking.Tracker, quietly: bool) -> None:
        """Stops the other cards, writes the report and closes the run's files, every step even when one
        cannot write (a full disk). `quietly`: the run already stops with an error, which stays the one
        said; otherwise the first write that failed is raised at the end."""
        steps = [
            *([self.cards.stop] if self.cards is not None else []),
            self.write_report,
            self.metrics.close,
            self.selfplay.close,
            tracker.finish,
            *([self.replay_mix.close] if self.replay_mix is not None else []),
        ]
        failed: OSError | None = None
        for step in steps:
            try:
                step()
            except OSError as err:
                log.info("While stopping: %s", err)
                failed = failed or err
        if failed is not None and not quietly:
            raise failed
