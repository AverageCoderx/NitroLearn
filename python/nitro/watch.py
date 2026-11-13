"""`nitro watch`: plays a game with a run's saved bot and records it for the match viewer (`nitro.viewer`).

One arena with the run's own settings (start states, game endings), every car driven by the saved policy,
sampled as in training. The core captures the game every few physics ticks (`nitro_core::capture`), so
the picture moves as smoothly as the simulation does; at every decision the recording also gets each car's
value estimate, how sure the bot was of the action it took and how random its choice was. A recorded game
becomes one page (`watch.html` in the run folder) that opens from a file and works offline; a live game
(`Live`) is played without end and written line by line for the viewer's feed. Nothing listens on a port.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nitro import viewer
from nitro.config import NitroConfig
from nitro.envs import EnvPool, probe_config
from nitro.model import attention_encoder, build_networks
from nitro.ppo import masked_log_probs
from nitro.runs import actions_problem
from nitro.viewer import feed
from nitro.viewer import recording as layouts

# The frame layout the core writes (nitro_core::capture): a head, then one block per car.
LAYOUT = layouts.FORMAT
HEAD, CAR = layouts.HEAD, layouts.CAR


@dataclass
class Recording:
    frames: np.ndarray  # [frames, HEAD + cars * CAR] float32
    pads: list[list[float]]  # per boost pad: x, y, z, 1 if big
    every: int  # physics ticks per frame
    tick_skip: int
    decisions: list[list[Any]]  # per decision: [frame, [value, sure, randomness, action(, looks)] per car]
    attention: dict[str, Any] | None = None  # with an attention network: {"labels": what each share is}

    @property
    def cars(self) -> int:
        return int(self.frames.shape[1] - HEAD) // CAR

    def data(self, title: str = "", subtitle: str = "") -> dict[str, Any]:
        """The recording as the viewer's page data (what a page shows and a recording file holds)."""
        return layouts.data(
            self.frames,
            fps=120 / self.every,
            pads=self.pads,
            decisions=self.decisions,
            decisions_per_second=120 / self.tick_skip,
            title=title,
            subtitle=subtitle,
            attention=self.attention,
        )


class WatchError(Exception):
    """A game cannot be recorded; the message says why."""


class Game:
    """One arena played on the CPU by the policy in `state` (a checkpoint), one decision at a time; captured
    `fps` times per second of game time (at most 120, the physics rate). `start` holds the first frame."""

    def __init__(self, config: NitroConfig, state: dict[str, Any], seed: int, fps: int = 60) -> None:
        if problem := actions_problem(state["learner"]["model"]):
            raise ValueError(f"{problem}.")
        self.envs = EnvPool(probe_config(config, seed), pin_memory=False)
        self.tensors = self.envs.groups[0]
        self.model = build_networks(config, self.tensors.obs.shape[1], self.tensors.mask.shape[1])
        self.model.load_state_dict(state["learner"]["model"])
        self.model.eval()
        encoder: Any = attention_encoder(self.model)
        if encoder is not None:
            encoder.keep_weights = True  # what the policy paid attention to at each decision
        self.encoder: Any = encoder
        self.generator = torch.Generator().manual_seed(seed)
        # Ticks per frame: the divisor of the tick skip nearest to what `fps` asks, so every decision falls on
        # a frame.
        wanted = 120 / min(max(fps, 1), 120)
        self.tick_skip = config.env.tick_skip
        self.every = min(
            (d for d in range(1, self.tick_skip + 1) if self.tick_skip % d == 0),
            key=lambda d: abs(d - wanted),
        )
        self.cars = int(self.tensors.obs.shape[0])
        self.width = layouts.width(self.cars)
        self.pads = self.envs.start_capture(0, self.every)
        self.start: list[float] = self.envs.take_capture(0)
        if len(self.start) != self.width:
            raise WatchError(
                f"the core wrote {len(self.start)} numbers for the first frame where frame layout {LAYOUT} "
                f"has {self.width}: the core and the Python package are from different versions (install "
                "NitroLearn again to rebuild the core)."
            )
        self.frames = 1  # captured so far
        self.attention: dict[str, Any] | None = None
        if encoder is not None:
            labels = encoder.attention_labels()
            teams = [int(self.start[HEAD + c * CAR]) for c in range(self.cars)]
            self.attention = {
                "labels": labels,
                "talk": hasattr(encoder, "talk_weights"),
                "targets": [attention_targets(labels, teams, c) for c in range(self.cars)],
            }

    @torch.no_grad()
    def decide(self) -> tuple[list[Any], list[float]]:
        """One decision and the step it starts: [frame looked at, figures per car] and the frames captured
        during the step, flat."""
        tensors = self.tensors
        logits, value = self.model(tensors.obs)
        log_probs = masked_log_probs(logits, tensors.mask)
        probs = log_probs.exp()
        actions = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
        allowed = tensors.mask.sum(1).clamp(min=2).float()
        randomness = -(probs * log_probs.clamp(min=-1e9)).nan_to_num().sum(1) / allowed.log()
        sure = probs.gather(1, actions[:, None]).squeeze(1)
        figures: list[list[Any]] = [
            [_round(v), _round(p), _round(r), int(a)]
            for v, p, r, a in zip(value, sure, randomness, actions, strict=True)
        ]
        encoder = self.encoder
        if encoder is not None and encoder.attention_weights is not None:
            for car, shares in zip(figures, encoder.attention_weights.mean(1).tolist(), strict=True):
                car.append([round(share, 2) for share in shares])  # averaged over the heads
            if getattr(encoder, "talk_weights", None) is not None:
                for car, heard in zip(figures, _loudest(encoder.talk_weights), strict=True):
                    car.append(heard)
        decision = [self.frames - 1, *figures]  # the frame the bot is looking at
        tensors.actions.copy_(actions)
        self.envs.step(0)
        self.envs.wait(0)
        new = self.envs.take_capture(0)
        self.frames += len(new) // self.width
        return decision, new


def record(config: NitroConfig, state: dict[str, Any], seconds: float, seed: int, fps: int = 60) -> Recording:
    """One game of `seconds`, played on the CPU with the policy in `state` (a checkpoint), captured `fps`
    times per second of game time (at most 120, the physics rate)."""
    game = Game(config, state, seed, fps)
    flat = list(game.start)
    decisions = []
    for _ in range(round(seconds * 120 / game.tick_skip)):
        decision, frames = game.decide()
        decisions.append(decision)
        flat += frames
    frames_array = np.asarray(flat, dtype=np.float32).reshape(-1, game.width)
    return Recording(frames_array, game.pads, game.every, game.tick_skip, decisions, game.attention)


class Live:
    """A game played without end in real time, as lines for the viewer's feed (`nitro.viewer.feed`):
    `header` is the first line (the page's data), `play` writes a line per decision."""

    def __init__(
        self, config: NitroConfig, state: dict[str, Any], title: str, subtitle: str, seed: int, fps: int = 60
    ) -> None:
        self.seed, self.fps = seed, fps
        self.game = game = Game(config, state, seed, fps)
        nothing = np.zeros((0, game.width), np.float32)
        empty = Recording(nothing, game.pads, game.every, game.tick_skip, [], game.attention)
        self.header = feed.header(empty.data(title, subtitle))

    def play(
        self,
        write: Callable[[str], None],
        stop: threading.Event,
        newer: Callable[[], tuple[NitroConfig, dict[str, Any], str] | None] | None = None,
        check_seconds: float = 20.0,
    ) -> None:
        """Plays until `stop`. With `newer`, asked every `check_seconds`, a newer save (its settings, state
        and title) takes over with a new game, if its games have the same cars and timing."""
        game = self.game

        def newer_game() -> tuple[feed.Playing, str] | None:
            found = newer() if newer is not None else None
            if found is None:
                return None
            config, state, title = found
            other = Game(config, state, self.seed, self.fps)
            if (other.width, other.every, other.tick_skip) != (game.width, game.every, game.tick_skip):
                return None  # the page was built for this game's frames
            return other, title

        # One thread is plenty for one arena's decisions, and more would take time from the picture.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            feed.play(
                game,
                write,
                cars=game.cars,
                seconds_per_decision=game.tick_skip / 120,
                stop=stop,
                newer=newer_game if newer is not None else None,
                check_seconds=check_seconds,
            )
        finally:
            torch.set_num_threads(threads)


def attention_targets(labels: list[str], teams: list[int], car: int) -> list[int | str | None]:
    """What each attention place of `car` is: another car's index, "ball", "itself" or "nothing", or None for
    an empty place. The other cars fill their places as the core writes them (nitro_core::obs
    `write_others`): teammates, then opponents, each in car order."""
    mates = [c for c in range(len(teams)) if c != car and teams[c] == teams[car]]
    opponents = [c for c in range(len(teams)) if teams[c] != teams[car]]
    queues = {"teammate": iter(mates), "opponent": iter(opponents)}
    return [next(queues[label], None) if label in queues else label for label in labels]


def _loudest(talk: torch.Tensor) -> list[list[list[float]]]:
    """Per car, for each token after the null token, the other token it listened to most in the last round
    of talk (not itself, not the null token) and how much: [[token, share], ...]."""
    mean = talk.mean(1)  # over the heads: [cars, tokens, tokens]
    tokens = mean.shape[-1]
    others = mean.masked_fill(torch.eye(tokens, dtype=torch.bool), -1.0)
    others[..., 0] = -1.0
    share, token = others.max(-1)
    return [
        [[int(j), round(float(w), 2)] for j, w in zip(js[1:], ws[1:], strict=True)]
        for js, ws in zip(token.tolist(), share.tolist(), strict=True)
    ]


def page(
    recording: Recording,
    title: str,
    subtitle: str = "",
    follow: bool = False,
    arena: str | None = None,
    arena_src: str | None = None,
) -> str:
    """The viewer's page for `recording`; with `follow`, the page loads itself again at the end of each
    game so it picks up the newest recording (`nitro watch --follow`). `arena` and `arena_src` are
    `nitro.viewer.page`'s: the arena's meshes inside the page, or the address of their script file."""
    data = recording.data(title, subtitle)
    if follow:
        data["follow"] = True
    return viewer.page(data, arena=arena, arena_src=arena_src)


def _round(value: torch.Tensor) -> float:
    number = float(value)
    return round(number, 3) if math.isfinite(number) else 0.0
