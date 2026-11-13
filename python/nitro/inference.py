"""Choosing actions for one group of environments, captured as CUDA graphs when on the GPU.

One run copies the group's observations and masks to the device, runs the networks, samples masked
actions with the Gumbel-max trick (no host synchronisation, unlike `torch.multinomial`), and copies the
actions back into the group's pinned `actions` tensor. When given the group's rollout columns it also
stores the step there and, before that, the results of the group's previous step, all at a step index
kept on the device, so a training step costs one graph launch (Windows charges ~20-30 µs per launch).
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import gc
import sys
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nitro import fp16_sums
from nitro.envs import GroupTensors
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs

if TYPE_CHECKING:
    from nitro.selfplay import OpponentBlock

# Keeps the Gumbel noise finite: log(0) and log(-log(1)) would be infinite.
_UNIFORM_LOW, _UNIFORM_HIGH = 1e-20, 1.0 - 1e-7
_WARMUP_RUNS = 3
# Every capture uses torch.cuda.graph's one capture stream, so two threads must not capture at once.
_CAPTURES = threading.RLock()
# One warm-up stream per card for the whole process. cuBLAS keeps a workspace for every stream it has run on
# until the process ends, and PyTorch hands out its 32 pooled streams in turn, so a new stream per capture
# pinned about 20 MB of card memory per graph until the pool came round (found by a long soak test).
_WARMUP_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
# Side streams the concurrent passes (env.concurrent_passes) run on, made once per card for the same
# reason; every group's graphs share them.
_SIDE_STREAMS: dict[torch.device, list[torch.cuda.Stream]] = {}


@dataclass(frozen=True)
class RolloutColumns:
    """One group's columns of a `Rollout` ([steps, group agents, ...] views)."""

    obs: torch.Tensor
    mask: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminal: torch.Tensor
    truncated: torch.Tensor
    episode: torch.Tensor
    episode_step: torch.Tensor


@contextlib.contextmanager
def capturing(graph: torch.cuda.CUDAGraph, **options: Any) -> Iterator[None]:
    """`torch.cuda.graph(graph, **options)` with cuBLAS's workspaces let go just before and after, as
    PyTorch's own CUDA graph trees do. cuBLAS keeps a workspace per stream, made at the stream's first
    product; every capture on the shared capture stream used the one made inside the first collection
    graph's memory, so a graph that outlived a save (when collection's graphs are captured anew and their
    memory freed) read freed memory: "unspecified launch failure" right after saves.
    Let go before, each capture's products make a workspace inside its own graph's memory; let go after, no
    later capture uses one that a graph owns. One thread captures at a time (`_CAPTURES`). Python's garbage
    collector is off while it captures: a graph dropped in a reference cycle (a learner and its
    graphed steps) and collected then is destroyed inside the capture, which spoils it."""
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    with _CAPTURES:
        collecting = gc.isenabled()
        gc.disable()
        if clear is not None:
            clear()
        try:
            with torch.cuda.graph(graph, **options):
                yield
        finally:
            if clear is not None:
                clear()
            if collecting:
                gc.enable()


def _card(device: torch.device) -> torch.device:
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


def warmup_stream(device: torch.device) -> torch.cuda.Stream:
    """The side stream every graph capture on `device` warms up on (see `_WARMUP_STREAMS`)."""
    key = _card(device)
    if key not in _WARMUP_STREAMS:
        _WARMUP_STREAMS[key] = torch.cuda.Stream(key)  # type: ignore[no-untyped-call]
    return _WARMUP_STREAMS[key]


@functools.cache
def _cuda_driver() -> Any:
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    driver.cuLaunchHostFunc.restype = ctypes.c_int
    return driver


def launch_host_function(stream: torch.cuda.Stream, function: int, argument: int) -> None:
    """Queues the C function at address `function` on `stream` (the driver's cuLaunchHostFunc): it runs on the
    driver's own thread, with `argument`, once the work queued before it is done, and holds back the work
    queued after it until it returns; inside a CUDA graph capture it becomes part of the graph."""
    result = _cuda_driver().cuLaunchHostFunc(stream.cuda_stream, function, argument)
    if result != 0:
        raise RuntimeError(f"cuLaunchHostFunc failed (CUDA driver error {result})")


def side_streams(device: torch.device, count: int) -> list[torch.cuda.Stream]:
    """`count` side streams on `device` for the concurrent passes (see `_SIDE_STREAMS`)."""
    streams = _SIDE_STREAMS.setdefault(_card(device), [])
    while len(streams) < count:
        streams.append(torch.cuda.Stream(_card(device)))  # type: ignore[no-untyped-call]
    return streams[:count]


class GroupInference:
    def __init__(
        self,
        model: ActorCritic,
        tensors: GroupTensors,
        device: torch.device,
        fp16: bool,
        cuda_graph: bool,
        columns: RolloutColumns | None = None,
        opponents: Sequence[OpponentBlock] = (),
        concurrent: bool = False,
        card_start: tuple[int, int, object] | None = None,
        fp16_sums: bool = False,
    ) -> None:
        """`concurrent` (env.concurrent_passes): on a graphics card the value network and each past
        version's pass run on side streams beside the policy's, in the same graph; the same kernels and
        random numbers, so the same answers. `card_start` (env.card_starts_games): the C function and
        argument that start the group's step (`EnvPool.card_start`) and the pool they belong to (kept alive
        here); with CUDA graphs each run ends with the card calling it once the actions are on the host, so
        the group's games start without Python (`starts_games`); `wait` for this group before the pool's.
        `fp16_sums` (run.fp16_sums): the passes' FP16 products keep their sums in FP16."""
        agents, obs_size = tensors.obs.shape
        self.model = model
        self.tensors = tensors
        self.device = device
        self.fp16 = fp16
        self.columns = columns
        # Frozen past versions driving some rows (nitro.selfplay): their networks run on their row slices
        # inside the same graph; `select` says row by row whose action drives the car.
        self.opponents = list(opponents)
        self.fp16_sums = fp16_sums
        concurrent = concurrent and device.type == "cuda"
        self.side = side_streams(device, 1 + len(self.opponents)) if concurrent else []
        self.non_blocking = device.type == "cuda"
        self.obs = torch.zeros(agents, obs_size, device=device)
        self.mask = torch.ones(agents, tensors.mask.shape[1], dtype=torch.bool, device=device)
        self.actions = torch.zeros(agents, dtype=torch.int64, device=device)
        self.actions_int32 = torch.zeros(agents, dtype=torch.int32, device=device)
        self.log_probs = torch.zeros(agents, device=device)
        self.values = torch.zeros(agents, device=device)
        self.rewards = torch.zeros(agents, device=device)
        self.terminal = torch.zeros(agents, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(agents, dtype=torch.bool, device=device)
        self.episode = torch.zeros(agents, dtype=torch.int64, device=device)
        self.episode_step = torch.zeros(agents, dtype=torch.int32, device=device)
        # Rollout row for the next step, and the row of the step before it.
        self.step_index = torch.zeros(1, dtype=torch.int64, device=device)
        self.previous_index = torch.zeros(1, dtype=torch.int64, device=device)
        self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        # Marks the end of this group's queued work, so `wait` waits for this group only: with pipelined
        # launches (Trainer) another group's graph may already be queued behind it on the same stream.
        self.event = torch.cuda.Event() if device.type == "cuda" else None  # type: ignore[no-untyped-call]
        self.graphs: dict[bool, torch.cuda.CUDAGraph] = {}
        self.card_start = card_start if cuda_graph and device.type == "cuda" else None
        self._starting = False  # true only while a graph is captured: its replays start the games
        if cuda_graph:
            # Recording a previous step needs a rollout with at least two rows.
            recorded = columns is not None and columns.rewards.shape[0] > 1
            for record_previous in (False, True) if recorded else (False,):
                self.graphs[record_previous] = self._capture(record_previous)
            self.restart()

    @property
    def starts_games(self) -> bool:
        """Whether each run starts the group's step itself (`card_start`)."""
        return self.card_start is not None and bool(self.graphs)

    def restart(self) -> None:
        """The next run stores its step in rollout row 0."""
        self.step_index.zero_()

    def run(self, record_previous: bool = False) -> None:
        """Queues the work; call `wait` before the environments step this group again.

        `record_previous` first stores the results of the group's previous step (rewards, endings).
        """
        graph = self.graphs.get(record_previous)
        if graph is not None:
            graph.replay()
        else:
            self._work(record_previous)
        if self.event is not None:
            self.event.record(self.stream)

    def record_results(self) -> None:
        """Stores the results of the group's latest step without choosing new actions (rollout end)."""
        self._record()

    def wait(self) -> None:
        """Blocks until this group's queued work is done, the copies to and from its pinned tensors
        included."""
        if self.event is not None:
            self.event.synchronize()

    @torch.no_grad()
    def _work(self, record_previous: bool) -> None:
        if record_previous:
            self._record()
        with fp16_sums.kept(self.fp16_sums):
            self._infer()

    def _record(self) -> None:
        columns = self.columns
        assert columns is not None
        self.rewards.copy_(self.tensors.reward, non_blocking=self.non_blocking)
        self.terminal.copy_(self.tensors.terminal, non_blocking=self.non_blocking)
        self.truncated.copy_(self.tensors.truncated, non_blocking=self.non_blocking)
        torch.sub(self.step_index, 1, out=self.previous_index)
        for column, value in (
            (columns.rewards, self.rewards),
            (columns.terminal, self.terminal),
            (columns.truncated, self.truncated),
        ):
            column.index_copy_(0, self.previous_index, value[None])

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16, cache_enabled=False)

    def _sample(
        self, logits: torch.Tensor, mask: torch.Tensor, out: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Actions drawn from the masked logits' chances with Gumbel noise (into `out` for the learner's
        pass), and every action's log-probability, which `_infer` reads the taken actions' from."""
        log_probs = masked_log_probs(logits, mask)
        scores = log_probs + self._gumbel(log_probs)
        if out is None:
            return torch.argmax(scores, dim=-1), log_probs
        return torch.argmax(scores, dim=-1, out=out), log_probs

    def _opponent_choice(self, block: OpponentBlock) -> torch.Tensor:
        with self._autocast():
            logits = block.model.logits(self.obs[block.rows])
        return self._sample(logits, self.mask[block.rows])[0]

    def _choose(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """The learner's log-probabilities and values, its sampled actions in `self.actions`, and each past
        version's choice for its rows, one pass after another."""
        with self._autocast():
            logits, values = self.model(self.obs)
        log_probs = self._sample(logits, self.mask, out=self.actions)[1]
        return log_probs, values, [self._opponent_choice(block) for block in self.opponents]

    def _choose_concurrently(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """`_choose` with the value network and each past version on a side stream. The host issues every
        kernel and random draw in `_choose`'s order (so the random numbers are the same) and the main stream
        waits for the side streams before anything reads their results."""
        main = torch.cuda.current_stream(self.device)
        value_stream, *opponent_streams = self.side
        for stream in opponent_streams:  # they need the copied observations and masks only
            stream.wait_stream(main)
        with self._autocast():
            features = self.model.shared(self.obs)
        value_stream.wait_stream(main)
        with torch.cuda.stream(value_stream), self._autocast():
            values = self.model.value_from(features)
        features.record_stream(value_stream)
        with self._autocast():
            logits = self.model.logits_from(features)
        log_probs = self._sample(logits, self.mask, out=self.actions)[1]
        chosen = []
        for block, stream in zip(self.opponents, opponent_streams, strict=True):
            with torch.cuda.stream(stream):
                chosen.append(self._opponent_choice(block))
        for stream, result in zip(self.side, [values, *chosen], strict=True):
            main.wait_stream(stream)
            result.record_stream(main)
        return log_probs, values, chosen

    def _infer(self) -> None:
        self.obs.copy_(self.tensors.obs, non_blocking=self.non_blocking)
        self.mask.copy_(self.tensors.mask, non_blocking=self.non_blocking)
        log_probs, values, chosen = self._choose_concurrently() if self.side else self._choose()
        for block, choice in zip(self.opponents, chosen, strict=True):
            self.actions[block.rows].copy_(torch.where(block.select, choice, self.actions[block.rows]))
        # The learner's log-probability of the action taken, whoever chose it (opponent rows are never
        # trained on; see nitro.selfplay).
        self.log_probs.copy_(log_probs.gather(1, self.actions[:, None]).squeeze(1))
        self.values.copy_(values.float())
        self.actions_int32.copy_(self.actions)
        if self.columns is not None:  # read before the games may start writing the group's tensors again
            self._read_episodes()
        self.tensors.actions.copy_(self.actions_int32, non_blocking=self.non_blocking)
        self._start_games()
        if self.columns is not None:
            self._write_rows()

    def _read_episodes(self) -> None:
        self.episode.copy_(self.tensors.episode, non_blocking=self.non_blocking)
        self.episode_step.copy_(self.tensors.step, non_blocking=self.non_blocking)

    def _start_games(self) -> None:
        if self._starting and self.card_start is not None:
            launch_host_function(torch.cuda.current_stream(self.device), *self.card_start[:2])

    def _write_rows(self) -> None:
        columns = self.columns
        assert columns is not None
        for column, value in (
            (columns.episode, self.episode),
            (columns.episode_step, self.episode_step),
            (columns.obs, self.obs),
            (columns.mask, self.mask),
            (columns.actions, self.actions),
            (columns.log_probs, self.log_probs),
            (columns.values, self.values),
        ):
            column.index_copy_(0, self.step_index, value[None])
        self.step_index.add_(1)

    @staticmethod
    def _gumbel(log_probs: torch.Tensor) -> torch.Tensor:
        """Gumbel noise, so argmax(log p + noise) is a sample from p with no host synchronisation."""
        uniform = torch.rand_like(log_probs, dtype=torch.float32).clamp_(_UNIFORM_LOW, _UNIFORM_HIGH)
        return -torch.log(-torch.log(uniform))

    def _capture(self, record_previous: bool) -> torch.cuda.CUDAGraph:
        # Warm-up on a side stream is required before capture (cuBLAS workspaces, autocast, allocator).
        # It writes into rollout rows 0-1 and the actions tensor, which collection overwrites anyway.
        stream = warmup_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_RUNS):
                self.step_index.fill_(1 if record_previous else 0)
                self._work(record_previous)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        self._starting = self.card_start is not None
        try:
            # Only this thread's calls can spoil the capture: with several cards NCCL's watchdog thread may
            # use the card meanwhile.
            with capturing(graph, capture_error_mode="thread_local"):
                self._work(record_previous)
        finally:
            self._starting = False
        return graph
