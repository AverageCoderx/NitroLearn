"""Collects fixed-length rollouts from the environments and hands them to the learner.

Environment groups are pipelined: while the CPU workers step one group, the GPU chooses actions for the
next one. Each group's step results are recorded in the rollout (inside the next
inference graph) before the group steps again, so nothing is lost or spliced.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch

from nitro import fp16_sums
from nitro.envs import GroupTensors
from nitro.health import HealthMonitor, TrainingProblem
from nitro.inference import GroupInference, RolloutColumns
from nitro.learner import HeadValues, Learner, LearnStats, head_values_on
from nitro.model import ActorCritic, copy_inference_weights, fp16_inference_copy
from nitro.rollout import Rollout

if TYPE_CHECKING:
    from nitro.replay_mix import CopyStats, ReplayMix
    from nitro.selfplay import SelfPlay

log = logging.getLogger("nitro")


def on_cpu(value: Any) -> Any:
    """A copy of a (nested) state with every tensor on the CPU: the out-of-memory fallback's snapshot must
    not take graphics memory itself. `load_state_dict` copies the values back to the device."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: on_cpu(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(on_cpu(item) for item in value)
    return copy.deepcopy(value)


class Snapshot:
    """The out-of-memory fallback's copy of the learner's state, as `on_cpu` makes it, but into CPU buffers
    made once (pinned when on a graphics card): each iteration's copy is queued behind the card's work and
    overlaps it, instead of the host waiting for some 50 MB every iteration. The card's own order
    finishes the copy before learning changes anything; read it only after `torch.cuda.synchronize()`,
    and restore from `on_cpu` of it, so nothing loaded keeps one of these buffers."""

    def __init__(self, buffered: bool) -> None:
        self.buffered = buffered
        self._buffers: dict[tuple[Any, ...], torch.Tensor] = {}

    def take(self, value: Any) -> Any:
        if not self.buffered:
            return on_cpu(value)
        copies: list[tuple[torch.Tensor, torch.Tensor]] = []
        taken = self._planned(value, (), copies)
        if copies:  # one call for every tensor: a few hundred copy calls from Python cost ~1.5 ms
            buffers, sources = zip(*copies, strict=True)
            torch._foreach_copy_(list(buffers), list(sources), non_blocking=True)
        return taken

    def _planned(
        self, value: Any, path: tuple[Any, ...], copies: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        if isinstance(value, torch.Tensor):
            buffer = self._buffers.get(path)
            if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                pin = value.device.type == "cuda"
                buffer = torch.empty(value.shape, dtype=value.dtype, pin_memory=pin)
                self._buffers[path] = buffer
            copies.append((buffer, value.detach()))
            return buffer
        if isinstance(value, dict):
            return {key: self._planned(item, (*path, key), copies) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._planned(item, (*path, k), copies) for k, item in enumerate(value))
        return copy.deepcopy(value)


def release_memory(device: torch.device) -> None:
    """Gives back what dropped objects still hold: any reference cycle to Python's collector, and freed
    blocks (CUDA graphs' private pools among them, which stay reserved until asked for) to the card."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Envs(Protocol):
    """What the trainer needs from a set of environments (`nitro.envs.EnvPool`, `nitro.cartpole`)."""

    @property
    def groups(self) -> tuple[GroupTensors, ...]: ...

    @property
    def num_groups(self) -> int: ...

    def step(self, group: int) -> None: ...

    def wait(self, group: int) -> None: ...


@dataclass
class StageTimes:
    """Wall-clock seconds spent in each part of an iteration."""

    env_wait: float = 0.0  # waiting for CPU workers to finish a group's step
    inference: float = 0.0  # waiting for the GPU to choose a group's actions
    launch: float = 0.0  # handing the inference graph to the GPU driver
    submit: float = 0.0  # handing a group's step to the Rust workers
    other: float = 0.0  # the rest of collection: Python bookkeeping
    learn: float = 0.0

    @property
    def collect(self) -> float:
        return self.env_wait + self.inference + self.launch + self.submit + self.other

    @property
    def total(self) -> float:
        """The iteration's wall-clock time."""
        return self.collect + self.learn


@dataclass(frozen=True)
class IterationReport:
    agent_steps: int
    times: StageTimes
    learn: LearnStats
    copy: CopyStats | None = None  # the steps copying converted replays ([replay_mix])

    @property
    def steps_per_second(self) -> float:
        return self.agent_steps / self.times.total


# How many times one iteration may halve the learner's minibatch after the graphics card ran out of memory.
_OOM_RETRIES = 4


@dataclass
class _Truncations:
    """Final observations of truncated steps, valued in one batch at the end of the rollout."""

    steps: list[int] = field(default_factory=list)
    rows: list[torch.Tensor] = field(default_factory=list)
    obs: list[torch.Tensor] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        envs: Envs,
        model: ActorCritic,
        learner: Learner,
        steps_per_iteration: int,
        device: torch.device,
        fp16: bool,
        cuda_graphs: bool,
        on_problem: Literal["stop", "warn"] = "stop",
        selfplay: SelfPlay | None = None,
        pipelined: bool = False,
        concurrent: bool = False,
        card_starts: bool = False,
        fp16_sums: bool = False,
    ) -> None:
        """`pipelined`: launch the next group's network pass before waiting for the previous group's (the
        launch work then overlaps the GPU's; `env.pipelined_inference`). The collected data is
        the same either way; only the order of the waits changes. `concurrent`: the value network's and the
        past versions' passes beside the policy's.
        `card_starts`: with CUDA graphs, the card starts each group's games itself (env.card_starts_games),
        for environments that can be started so (`nitro.envs.EnvPool`). `fp16_sums`: the playing
        passes' FP16 products keep their sums in FP16 (run.fp16_sums)."""
        self.envs = envs
        self.model = model
        self.learner = learner
        self.device = device
        self.fp16 = fp16
        self.selfplay = selfplay
        self.pipelined = pipelined
        self.concurrent = concurrent
        self.card_starts = card_starts
        self.fp16_sums = fp16_sums
        # Steps copying converted replays after each update ([replay_mix]); set by the training run.
        self.replay_mix: ReplayMix | None = None
        self.truncations = _Truncations()
        self.snapshot = Snapshot(buffered=device.type == "cuda")
        self.notices: list[str] = []  # plain-English events for the run log (out-of-memory recoveries)
        self.health = HealthMonitor(on_problem, fp16)
        self.iterations = 0
        sizes = [tensors.reward.numel() for tensors in envs.groups]
        self.sizes = sizes
        self.offsets = [sum(sizes[:g]) for g in range(len(sizes))]
        agents = sum(sizes)
        self.steps = math.ceil(steps_per_iteration / agents)
        if self.steps * agents < 2 * learner.config.minibatches:
            raise ValueError(
                f"{learner.config.minibatches} minibatches need at least twice that many agent steps "
                "per iteration"
            )
        obs_size = envs.groups[0].obs.shape[1]
        num_actions = envs.groups[0].mask.shape[1]
        self.rollout = Rollout.allocate(self.steps, agents, obs_size, num_actions, device)
        self.cuda_graphs = cuda_graphs
        # Choosing actions in FP16 on a graphics card uses a copy holding in FP16 what autocast would cast
        # on every step, brought up to date once per collection: the same answers, fewer launches.
        self.inference_model = fp16_inference_copy(model) if fp16 and device.type == "cuda" else model
        self.engines = self._engines()

    def _engines(self) -> list[GroupInference]:
        starter = getattr(self.envs, "card_start", None)
        starts = self.card_starts and starter is not None and self.cuda_graphs and self.device.type == "cuda"
        return self._make_engines(starter if starts else None)

    def _make_engines(self, starter: Callable[[int], tuple[int, int]] | None) -> list[GroupInference]:
        return [
            GroupInference(
                self.inference_model,
                tensors,
                self.device,
                self.fp16,
                self.cuda_graphs,
                self._columns(group),
                self.selfplay.group_opponents(group) if self.selfplay else (),
                self.concurrent,
                (*starter(group), self.envs) if starter is not None else None,
                self.fp16_sums,
            )
            for group, tensors in enumerate(self.envs.groups)
        ]

    def restart(self, build: Callable[[], Envs]) -> None:
        """Continues with freshly built environments of the same shape (a checkpoint).

        The old environments and inference engines are let go before `build` makes the new ones, so only
        one set is in memory at a time, and the memory the old CUDA graphs held goes back to the card (a
        long soak test found each save adding to both). The episodes in progress end here as
        truncations: the last rollout already valued the states they reached, so nothing more is needed
        than forgetting their running returns.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # no old graph still running: a card start calls into the old pool
        self.engines = []
        del self.envs
        release_memory(self.device)
        envs = build()
        if [tensors.reward.numel() for tensors in envs.groups] != self.sizes:
            raise ValueError("the new environments must have the same groups as the old ones")
        self.envs = envs
        self.learner.reward_scaler.running = None
        self.engines = self._engines()

    def _rows(self, group: int) -> slice:
        return slice(self.offsets[group], self.offsets[group] + self.envs.groups[group].reward.numel())

    def _columns(self, group: int) -> RolloutColumns:
        rows, r = self._rows(group), self.rollout
        return RolloutColumns(
            obs=r.obs[:, rows],
            mask=r.mask[:, rows],
            actions=r.actions[:, rows],
            log_probs=r.log_probs[:, rows],
            values=r.values[:, rows],
            rewards=r.rewards[:, rows],
            terminal=r.terminal[:, rows],
            truncated=r.truncated[:, rows],
            episode=r.episode[:, rows],
            episode_step=r.episode_step[:, rows],
        )

    def iteration(self) -> IterationReport:
        times = self.collect()
        rows = self.selfplay.learner_rows if self.selfplay else None
        stats, copy, times.learn = self._learn(rows)
        return self._report(self.steps * self.rollout.agents, times, stats, copy)

    def _report(
        self, agent_steps: int, times: StageTimes, stats: LearnStats, copy: CopyStats | None
    ) -> IterationReport:
        report = IterationReport(agent_steps, times, stats, copy)
        self.iterations += 1
        log.info(
            "iteration %d: %.0f steps/s, entropy %.3f, policy change (KL) %.2e, explained variance %.2f, "
            "skipped updates %d of %d, loss scale %g",
            self.iterations,
            report.steps_per_second,
            stats.entropy,
            stats.policy_change,
            stats.explained_variance,
            stats.skipped_updates,
            stats.gradient_steps,
            stats.grad_scale,
        )
        self.health.check(stats)
        return report

    def _learn(self, rows: torch.Tensor | None) -> tuple[LearnStats, CopyStats | None, float]:
        """One learning update on the collected rollout, `rows` the agents the learner drives; returns its
        statistics, the replay copying's and the seconds it took."""
        start = time.perf_counter()
        # A retry must redo the whole update: weights, optimizer, reward scale and value normaliser as they
        # were before any minibatch step or statistic update of the failed attempt, on the CPU; also the
        # rewards' running returns (no save keeps them) and a loss scaler not left mid-step.
        # Several cards never retry (below), so they copy nothing.
        many = self.learner.cards is not None
        before = None if many else self.snapshot.take(self.learner.state_dict())
        carried = None if many else self.learner.reward_scaler.running  # never written in place
        heads = (
            self._head_values(self.rollout, self.truncations, self.model, rows)
            if self.learner.extra_heads
            else None
        )
        for attempt in range(_OOM_RETRIES + 1):
            try:
                stats = self.learner.learn(
                    self.rollout,
                    rows,
                    **({"heads": heads} if heads is not None else {}),
                )
                break
            except torch.cuda.OutOfMemoryError as err:
                if many:  # halving one card's steps puts the cards out of step
                    raise TrainingProblem(
                        "a graphics card ran out of memory while learning. With several cards "
                        "(run.cards) the run cannot halve its learning steps by itself: set "
                        "ppo.microbatch (e.g. 16384, or lower if it is set already) in the bot's "
                        "settings, then resume"
                    ) from err
                if attempt == _OOM_RETRIES or before is None:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self.learner.load_state_dict(on_cpu(before))
                self.learner.reward_scaler.running = carried
                self.learner.fresh_loss_scaler()
                self._halve_minibatch()
        copy = self.replay_mix.learn(stats) if self.replay_mix is not None else None
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return stats, copy, time.perf_counter() - start

    def _head_values(
        self, r: Rollout, cuts: _Truncations, model: ActorCritic, rows: torch.Tensor | None = None
    ) -> HeadValues:
        """The extra value heads the learner needs on a rollout's states by `model` (the networks
        before the update): the states after its last step, from the games; the truncated steps' final
        states. With ppo.lean_heads_pass only the agents `rows` marks, with run.fp16_sums' FP16 sums
        from 'forward' on."""
        # Without waiting for the card: the games write these tensors again only in the next collection,
        # after this update's end waits for the card.
        last = torch.cat([tensors.obs.to(self.device, non_blocking=True) for tensors in self.envs.groups])
        found = None
        if cuts.steps:
            steps = torch.cat(
                [torch.full_like(rows, t) for t, rows in zip(cuts.steps, cuts.rows, strict=True)]
            )
            obs = torch.cat(cuts.obs).to(self.device)
            found = (steps.to(self.device), torch.cat(cuts.rows).to(self.device), obs)
        lean = self.learner.config.lean_heads_pass
        return head_values_on(
            model,
            self.learner.extra_heads,
            self.fp16,
            r,
            last,
            found,
            columns=self.learner._driven(rows) if lean and rows is not None else None,
            sums=lean and self.learner.fp16_sums in fp16_sums.FORWARD,
        )

    def _halve_minibatch(self) -> None:
        """The graphics card ran out of memory while learning: from now on each
        learning step takes half the data, i.e. twice as many minibatches, so the run goes on. The notice
        says how to make the change permanent."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        config = self.learner.config
        self.learner.config = dataclasses.replace(config, minibatches=config.minibatches * 2)
        before, after = config.minibatches, config.minibatches * 2
        notice = (
            "The graphics card ran out of memory during learning. Each learning step now uses half the data "
            f"(ppo.minibatches {before} -> {after}) for the rest of this run, which changes how the bot "
            f"learns a little. To make it permanent, set ppo.minibatches = {after} in the run's settings "
            "file; or lower env.num_arenas or ppo.steps_per_iteration."
        )
        log.warning(notice)
        self.notices.append(notice)

    def collect(self) -> StageTimes:
        """Fills `self.rollout` with the next `self.steps` steps of every agent."""
        times = StageTimes()
        begin = time.perf_counter()
        truncations = _Truncations()
        if self.inference_model is not self.model:
            copy_inference_weights(self.inference_model, self.model)
        for engine in self.engines:
            engine.restart()
        # The card starts each group's games itself once its actions are back (env.card_starts_games): the
        # loop only waits for the card, then for the games, and prepares one group while the other plays.
        starts = all(engine.starts_games for engine in self.engines)
        try:
            self._collect_in_python(truncations, times, starts)
        except BaseException:
            # With the card starting the games, another group's graph may still be queued to start its
            # games: let it finish before anything frees the games it would start.
            if starts and self.device.type == "cuda":
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
            raise
        return self._collected(truncations, times, begin)

    def _collect_in_python(self, truncations: _Truncations, times: StageTimes, starts: bool) -> None:
        # Pipelined: the group whose network pass is launched but whose environments have not been sent off.
        pending: int | None = None
        for t in range(self.steps):
            for group, engine in enumerate(self.engines):
                if pending == group:  # one group only: nothing to overlap with
                    self._finish_inference(pending, times)
                    pending = None
                if t > 0:
                    if starts:
                        self._card_started(group, times)
                    self._wait_for_step(group, t - 1, truncations, times)
                start = time.perf_counter()
                # Also stores the results of the group's previous step in the rollout.
                engine.run(record_previous=t > 0)
                times.launch += time.perf_counter() - start
                if starts:
                    continue
                if self.pipelined:
                    if pending is not None:
                        self._finish_inference(pending, times)
                    pending = group
                else:
                    self._finish_inference(group, times)
        if pending is not None:
            self._finish_inference(pending, times)
        for group in range(len(self.engines)):
            if starts:
                self._card_started(group, times)
            self._wait_for_step(group, self.steps - 1, truncations, times)

    def _collected(self, truncations: _Truncations, times: StageTimes, begin: float) -> StageTimes:
        """The rollout's end, after the last step of every group: its results recorded, the final states
        valued."""
        for engine in self.engines:
            engine.record_results()
        self.truncations = truncations
        self._value_final_states(truncations, self.model, self.rollout)
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # so queued collection work isn't timed as learning
        times.other = (
            time.perf_counter() - begin - times.env_wait - times.inference - times.launch - times.submit
        )
        return times

    def _finish_inference(self, group: int, times: StageTimes) -> None:
        """Waits for `group`'s network pass, then sends its environments off. The environments overwrite the
        pinned tensors when they step, so the copies to and from them must have finished first."""
        start = time.perf_counter()
        self.engines[group].wait()
        inferred = time.perf_counter()
        self.envs.step(group)
        times.inference += inferred - start
        times.submit += time.perf_counter() - inferred

    def _card_started(self, group: int, times: StageTimes) -> None:
        """Waits for `group`'s network pass, at whose end the card started its games."""
        start = time.perf_counter()
        self.engines[group].wait()
        times.inference += time.perf_counter() - start

    def _wait_for_step(self, group: int, t: int, truncations: _Truncations, times: StageTimes) -> None:
        """Waits for `group`'s step `t` and keeps the final observations of any truncated episodes."""
        start = time.perf_counter()
        self.envs.wait(group)
        times.env_wait += time.perf_counter() - start
        tensors = self.envs.groups[group]
        if tensors.truncated.any():
            local = tensors.truncated.nonzero().squeeze(1)
            truncations.steps.append(t)
            truncations.rows.append(local + self.offsets[group])
            truncations.obs.append(tensors.final_obs.index_select(0, local))

    def _normalised_value(self, model: ActorCritic, obs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            return model.normalised_value(obs).float()

    @torch.no_grad()
    def _value_final_states(self, truncations: _Truncations, model: ActorCritic, rollout: Rollout) -> None:
        """The values of the states after the last step (from the games) and of the truncated steps' ends,
        by `model`."""
        for group, tensors in enumerate(self.envs.groups):
            obs = tensors.obs.to(self.device)  # pinned on a card
            rollout.bootstrap_values[self._rows(group)] = self._normalised_value(model, obs)
        rollout.truncated_values.zero_()
        if truncations.steps:
            obs = torch.cat(truncations.obs).to(self.device)
            cuts = zip(truncations.steps, truncations.rows, strict=True)
            steps = torch.cat([torch.full_like(rows, t) for t, rows in cuts]).to(self.device)
            agents = torch.cat(truncations.rows).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                rollout.truncated_values[steps, agents] = model.normalised_value(obs).float()
