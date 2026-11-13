"""One bot on several graphics cards (`run.cards`): every card plays its share of the games
and learns on it; the cards sum their gradients every learning step, so each applies the same update and the
networks stay identical. Card 0 is the process `nitro train` started; the others are helper processes
(`nitro.cards_run`).

`Cards` holds the sums the learner needs. Everything it computes is over all cards' samples, so a learning
update is the one a single card would make on the whole batch (`tests/test_multigpu.py`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from nitro.config.schema import NitroConfig

# A card waiting this long for the others (a helper that died, a stuck card) stops with an error.
_TIMEOUT = timedelta(minutes=10)
_MASK64 = (1 << 64) - 1


class CardsError(ValueError):
    """Settings that cannot train on several cards, with what to change."""


@dataclass(frozen=True)
class Cards:
    """This process's place among the cards that train one bot."""

    rank: int
    size: int
    device: torch.device
    # Sum gradients while the backward pass runs (`GradientSums`); and in what precision they travel.
    overlap: bool = True
    sum_dtype: torch.dtype = torch.float32
    # Learning steps replayed as CUDA graphs with the sums inside (run.cards_graphed_steps).
    graphed: bool = False

    @property
    def first(self) -> bool:
        return self.rank == 0

    def sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sums `tensor` over the cards, in place."""
        dist.all_reduce(tensor)
        return tensor

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        """The plain mean of a scalar over the cards."""
        return self.sum_(value.detach().double().reshape(1).clone())[0] / self.size

    def moments(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count, mean and sum of squared deviations of every card's `values` together (FP64)."""
        values = values.detach().double().flatten()
        # This card's count filled on the card: new_tensor would copy it from the host and wait for the card.
        mine = torch.full((), float(values.numel()), dtype=torch.float64, device=values.device)
        totals = self.sum_(torch.stack([mine, values.sum()]))
        count = totals[0]
        mean = totals[1] / count.clamp_min(1)
        m2 = self.sum_(((values - mean) ** 2).sum().reshape(1))[0]
        return count, mean, m2

    def sizes(self, count: int) -> list[int]:
        """Every card's `count`, in card order."""
        return self.sizes_later(count)()

    def sizes_later(self, count: int) -> Callable[[], list[int]]:
        """`sizes`, its sum queued now and read when called: on a card copied to the host without waiting, so
        the processor goes on queueing work meanwhile instead of leaving the card idle after a wait here."""
        mine = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        mine.narrow(0, self.rank, 1).fill_(float(count))  # an index put would copy the count from the host
        self.sum_(mine)
        if self.device.type != "cuda":
            return lambda: [int(n) for n in mine.tolist()]
        host = torch.empty(self.size, dtype=torch.float64, pin_memory=True)
        host.copy_(mine, non_blocking=True)
        copied = torch.cuda.Event()  # type: ignore[no-untyped-call]
        copied.record()

        def read() -> list[int]:
            copied.synchronize()
            return [int(n) for n in host.tolist()]

        return read

    def gathered(self, values: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
        """Every card's `values` (1-D; card r's `sizes[r]` long) end to end in card order, on every card."""
        longest = max(sizes)
        mine = torch.nn.functional.pad(values.reshape(-1), (0, longest - values.numel()))
        pieces = [torch.empty_like(mine) for _ in range(self.size)]
        dist.all_gather(pieces, mine)
        return torch.cat([piece[:n] for piece, n in zip(pieces, sizes, strict=True)])

    def standardised(self, advantages: torch.Tensor, samples: torch.Tensor | None = None) -> torch.Tensor:
        """`nitro.learner._standardised` over every card's step: zero mean and unit spread (unbiased);
        `samples` (1, or 0 for a graphed step's padding) weighs each sample."""
        group = torch.zeros_like(advantages, dtype=torch.long)
        values = advantages.detach().double()
        weight = samples.double() if samples is not None else torch.ones_like(values)
        width = 1
        totals = torch.zeros(2 * width, dtype=torch.float64, device=values.device)
        totals[:width].index_add_(0, group, weight)
        totals[width:].index_add_(0, group, values * weight)
        self.sum_(totals)
        counts, sums = totals[:width], totals[width:]
        means = sums / counts.clamp_min(1)
        squares = torch.zeros(width, dtype=torch.float64, device=values.device)
        squares.index_add_(0, group, (values - means[group]) ** 2 * weight)
        self.sum_(squares)
        spread = (squares / (counts - 1).clamp_min(1)).sqrt()
        return (advantages - means[group].to(advantages.dtype)) / (spread[group].to(advantages.dtype) + 1e-8)

    def average_gradients(
        self, parameters: Sequence[torch.Tensor], measured: torch.Tensor, samples: int | torch.Tensor
    ) -> torch.Tensor:
        """Replaces each parameter's gradient by the cards' average, each card weighted by its `samples` in
        this step (so the result is the gradient of the mean loss over all cards' samples), and returns the
        same average of `measured`. A parameter without a gradient here has none on any card (the same step
        on every card) and is left out."""
        return self.average_([p.grad for p in parameters if p.grad is not None], measured, samples)

    def average_(
        self, grads: Sequence[torch.Tensor], measured: torch.Tensor, samples: int | torch.Tensor
    ) -> torch.Tensor:
        """`average_gradients` on the gradient tensors themselves, in place: one sum for everything (two with
        bfloat16: the gradients in it, then `measured` and the count in FP32, `_started`). A gradient that is
        not finite on one card is not finite on all, so every card skips the same steps."""
        flat = torch.cat(
            [
                *(g.reshape(-1).float() for g in grads),
                measured.detach().float().reshape(-1),
                measured.new_ones(1, dtype=torch.float32),
            ]
        )
        flat.mul_(samples if isinstance(samples, torch.Tensor) else float(samples))
        if self.sum_dtype == torch.float32:
            self.sum_(flat)
        else:
            self._started(flat, flat.numel() - measured.numel() - 1)()
        flat.div_(flat[-1].clone())
        offset = 0
        for g in grads:
            g.copy_(flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()
        return flat[offset : offset + measured.numel()].view_as(measured).to(measured.dtype)

    def _started(self, flat: torch.Tensor, gradients: int) -> Callable[[], None]:
        """Starts the cards' sum of `flat` (FP32), in place, and returns what waits for it. Its first
        `gradients` numbers travel in `sum_dtype`; with bfloat16 the rest (measured values and a count of
        samples, thousands that bfloat16 cannot hold exactly) are a small FP32 sum of their own."""
        if self.sum_dtype == torch.float32:
            works = [dist.all_reduce(flat, async_op=True)]
            part = None
        else:
            part = flat[:gradients].to(self.sum_dtype)
            works = [dist.all_reduce(part, async_op=True)]
            if gradients < flat.numel():
                works.append(dist.all_reduce(flat[gradients:], async_op=True))

        def wait() -> None:
            for work in works:
                work.wait()
            if part is not None:
                flat[:gradients].copy_(part)

        return wait

    @torch.no_grad()
    def broadcast_(self, tensors: Iterable[torch.Tensor]) -> None:
        """Copies card 0's values of `tensors` into every card's, in place."""
        for tensor in tensors:
            dist.broadcast(tensor, src=0)

    @torch.no_grad()
    def same_weights(self, parameters: Iterable[torch.Tensor]) -> bool:
        """Whether every card holds the same weights: the sums of every run of 4,096 weights in a row and the
        sum of all their squares (FP64), compared exactly: the same weights give the same bits on the same
        kind of card. A handful of launches for the whole network, not a few per weight."""
        flat = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
        runs = torch.nn.functional.pad(flat, (0, -flat.numel() % 4096)).view(-1, 4096)
        mine = torch.cat([runs.sum(1), runs.square().sum().reshape(1)])
        low, high = mine.clone(), mine.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        return bool(torch.equal(low, high))

    def send(self, value: object) -> object:
        """Card 0's `value` on every card (any picklable object; the others pass None)."""
        box = [value]
        dist.broadcast_object_list(box, src=0, device=self.device if self.device.type == "cuda" else None)
        return box[0]


def usable_threads() -> int:
    """Processor threads this process may use: its affinity, capped by a container's CPU quota (as the Rust
    core's automatic count sees it: a rented machine may give a share of a big processor)."""
    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4
    for files in (
        ["/sys/fs/cgroup/cpu.max"],
        ["/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"],
    ):
        try:
            quota, period = " ".join(Path(f).read_text() for f in files).split()[:2]
        except (OSError, ValueError):
            continue
        if quota not in ("max", "-1"):
            threads = min(threads, max(1, -(-int(quota) // int(period))))
        break
    return threads


def cache_groups(cpus: Sequence[int], root: Path = Path("/sys/devices/system/cpu")) -> list[list[int]]:
    """`cpus` in groups that share a last-level cache (a CCX on an EPYC), each group's CPUs in order and the
    groups in the order of their first CPU; one group per CPU where Linux does not say."""
    groups: dict[int, list[int]] = {}
    for cpu in sorted(cpus):
        shared = _cpu_list(root / f"cpu{cpu}" / "cache" / "index3" / "shared_cpu_list")
        key = min(shared) if shared else cpu
        groups.setdefault(key, []).append(cpu)
    return [groups[key] for key in sorted(groups)]


def _cpu_list(path: Path) -> list[int]:
    """A Linux CPU list file ("0-3,8-11") as numbers; empty when it cannot be read."""
    try:
        text = path.read_text().strip()
    except OSError:
        return []
    found: list[int] = []
    for part in filter(None, text.split(",")):
        first, _, last = part.partition("-")
        found.extend(range(int(first), int(last or first) + 1))
    return found


def card_cpus(rank: int, size: int, cpus: Sequence[int], groups: list[list[int]] | None = None) -> list[int]:
    """Card `rank`'s share of `cpus` among `size` cards (run.cards_pin_threads): whole cache groups,
    split into `size` runs as even as the groups allow, so each card's games and Python share caches only with
    each other; plain even runs of the CPUs when there are fewer groups than cards."""
    groups = groups if groups is not None else cache_groups(cpus)
    if len(groups) < size:
        ordered = sorted(cpus)
        return ordered[rank * len(ordered) // size : (rank + 1) * len(ordered) // size]
    total = sum(len(g) for g in groups)
    shares: list[list[int]] = [[] for _ in range(size)]
    taken = 0
    for group in groups:  # a group goes to the card whose run of CPUs its middle falls in
        middle = taken + len(group) / 2
        shares[min(size - 1, int(middle * size / total))].extend(group)
        taken += len(group)
    return shares[rank]


def pin_threads(cpus: Sequence[int]) -> None:
    """Every thread of this process, and the ones it starts later, on `cpus` only (Linux)."""
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    os.sched_setaffinity(0, cpus)
    for task in Path("/proc/self/task").iterdir():  # the threads already running (the games' among them)
        with contextlib.suppress(OSError, ValueError):
            os.sched_setaffinity(int(task.name), cpus)


def card_seed(seed: int, rank: int) -> int:
    """Card `rank`'s seed: its own games and sampling noise (card 0 keeps the run's seed)."""
    if rank == 0:
        return seed
    x = (seed ^ (rank * 0xD1B54A32D192ED03)) & _MASK64  # SplitMix64's finaliser
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


def share(config: NitroConfig, rank: int) -> NitroConfig:
    """Card `rank`'s share of a run of `run.cards` cards: its games, threads, steps per iteration and copy
    steps' decisions ([replay_mix]), and its own seed. One card: `config` itself."""
    cards = config.run.cards
    if cards <= 1:
        return config
    env, ppo = config.env, config.ppo
    threads = env.threads // cards if env.threads else max(1, (usable_threads() - 2) // cards)
    return dataclasses.replace(
        config,
        run=dataclasses.replace(config.run, seed=card_seed(config.run.seed, rank)),
        env=dataclasses.replace(env, num_arenas=env.num_arenas // cards, threads=threads),
        ppo=dataclasses.replace(ppo, steps_per_iteration=ppo.steps_per_iteration // cards),
        # Each copy step's decisions shared out too, so a step moves the bot as one card's would.
        replay_mix=dataclasses.replace(
            config.replay_mix, minibatch=max(1, config.replay_mix.minibatch // cards)
        ),
    )


# Gradients sent in one piece: big enough that each sum moves at full speed, small enough that the first
# pieces start early in the backward pass (DDP's default is 25 MB).
_BUCKET_BYTES = 25 * 2**20


class GradientSums:
    """Sums each learning step's gradients over the cards while the backward pass is still running.

    The parameters are cut into pieces of about 25 MB in reverse order (roughly the order the backward pass
    finishes them). A hook on every parameter counts the gradients done; once a piece is complete, its sum
    starts in the background and the backward pass carries on. Pieces start strictly in order, so every card
    sums them in the same order. The caller scales each card's loss by its share of all cards' samples, so the
    sum is the gradient of the mean loss over all of them. Only the backward pass after `start` counts (a step
    in parts sums once, after its last part); `finish` starts any pieces left (a value-only step leaves the
    policy's without gradients), waits, and writes the sums back. With `sum_dtype` bfloat16 the pieces travel
    at half the size, rounded to about 3 significant digits."""

    def __init__(
        self, cards: Cards, parameters: Sequence[torch.nn.Parameter], piece_bytes: int = _BUCKET_BYTES
    ) -> None:
        self.cards = cards
        pieces: list[list[torch.nn.Parameter]] = [[]]
        filled = 0
        for p in reversed([p for p in parameters if p.requires_grad]):
            if filled >= piece_bytes:
                pieces.append([])
                filled = 0
            pieces[-1].append(p)
            filled += p.numel() * p.element_size()
        self.pieces = pieces
        self.piece_of = {id(p): k for k, piece in enumerate(pieces) for p in piece}
        self.armed = False
        self.missing: list[int] = []
        self.next = 0
        self.sent: list[tuple[dist.Work, torch.Tensor, list[torch.Tensor]] | None] = []
        for piece in pieces:
            for p in piece:
                p.register_post_accumulate_grad_hook(self._done)  # type: ignore[no-untyped-call]

    def start(self) -> None:
        """Counts the gradients of the next backward pass."""
        self.armed = True
        self.missing = [len(piece) for piece in self.pieces]
        self.next = 0
        self.sent = []

    def _done(self, parameter: torch.Tensor) -> None:
        if not self.armed:
            return
        self.missing[self.piece_of[id(parameter)]] -= 1
        self._send(every=False)

    def _send(self, every: bool) -> None:
        while self.next < len(self.pieces) and (every or self.missing[self.next] == 0):
            grads = [p.grad for p in self.pieces[self.next] if p.grad is not None]
            self.next += 1
            if not grads:
                self.sent.append(None)
                continue
            flat = torch.cat([g.reshape(-1) for g in grads]).to(self.cards.sum_dtype)
            self.sent.append((dist.all_reduce(flat, async_op=True), flat, grads))

    def finish(self) -> None:
        self._send(every=True)
        self.armed = False
        for item in self.sent:
            if item is None:
                continue
            work, flat, grads = item
            work.wait()
            offset = 0
            for g in grads:
                g.copy_(flat[offset : offset + g.numel()].view_as(g))
                offset += g.numel()
        self.sent = []


def sum_dtype(config: NitroConfig) -> torch.dtype:
    return torch.bfloat16 if config.run.cards_sum_precision == "bf16" else torch.float32


def backend(device: torch.device) -> str:
    return "nccl" if device.type == "cuda" else "gloo"


def card_device(device: torch.device, rank: int) -> torch.device:
    """Card `rank`'s device: `cuda:rank` on graphics cards; the processor (tests) otherwise."""
    return torch.device("cuda", rank) if device.type == "cuda" else device


def join(
    store: Path,
    rank: int,
    size: int,
    device: torch.device,
    overlap: bool = True,
    sum_dtype: torch.dtype = torch.float32,
    graphed: bool = False,
) -> Cards:
    """Joins the cards of one run: the processes meet through `store`, a file (no rendezvous server); their
    sums talk over local sockets on the loopback interface only, alive while the run trains."""
    for name in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"):
        os.environ.setdefault(name, "lo")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # A sum keeps its tensors until it is waited for, instead of marking them for its stream: the marking is
    # not allowed while a CUDA graph captures (graphed steps on several cards, run.cards_graphed_steps).
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
    mine = card_device(device, rank)
    if mine.type == "cuda":
        torch.cuda.set_device(mine)
    # Each process's PyTorch takes every thread by default: N of them fight over the processor (on 4 cores, 2
    # processes on the processor ran 30 times slower). Each takes its share.
    torch.set_num_threads(max(1, usable_threads() // size))
    dist.init_process_group(
        backend(device),
        init_method=store.resolve().as_uri(),
        rank=rank,
        world_size=size,
        timeout=_TIMEOUT,
        device_id=mine if mine.type == "cuda" else None,
    )
    return Cards(rank, size, mine, overlap, sum_dtype, graphed)


def leave(clean: bool = False) -> None:
    """Leaves the cards' group; `clean` (every card is leaving too) waits for the others first: tearing down
    while another card still finishes its last sum aborts that card's process (gloo, about 1 run in 6)."""
    if dist.is_initialized():
        if clean:
            dist.barrier()
        dist.destroy_process_group()


def check(config: NitroConfig, device: torch.device) -> None:
    """Refuses settings several cards cannot train with, saying what to change."""
    cards = config.run.cards
    if cards <= 1:
        return
    problems: list[str] = []
    if sys.platform != "linux":
        # NCCL is Linux only, and so is the processes' meeting over the loopback interface ("lo").
        problems.append("several graphics cards train one bot on Linux only; here use run.cards = 1")
    elif device.type == "cuda" and torch.cuda.device_count() < cards:
        problems.append(f"run.cards = {cards} but PyTorch sees {torch.cuda.device_count()} graphics card(s)")
    if config.env.num_arenas % cards:
        problems.append(f"env.num_arenas ({config.env.num_arenas}) must divide by run.cards ({cards})")
    if config.ppo.steps_per_iteration % cards:
        problems.append(
            f"ppo.steps_per_iteration ({config.ppo.steps_per_iteration}) must divide by run.cards ({cards})"
        )
    if config.env.threads and config.env.threads < cards:
        problems.append(f"env.threads ({config.env.threads}) must be at least run.cards ({cards})")
    if problems:
        raise CardsError("; ".join(problems))
