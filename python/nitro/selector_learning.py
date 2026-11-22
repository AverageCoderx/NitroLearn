"""Training a multi-model bot's models and their goal heads (`selector.train_models`, `selector.goal_heads`),
from the steps `nitro.selector.SelectorEnvs` recorded.

Each model learns by PPO from the steps it drove, with its own optimizer, value network and learning rate, and
a KL pull back toward the bot it started as (`Learner.set_anchor`). Its stretches are cut where the car is
handed to another model: there its own value of the next state carries the rest (a truncation), so the
receiving model learns the states it is handed and the one handing over learns what its stretch was worth. The
goal heads learn without rewards: from the states each model drove, who scored within `goal_head_seconds`
(nobody, the car's team, the other team); a horizon a timeout cut short is left out, and one longer than a
collection waits for the next (`GoalSamples`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from nitro.config.schema import NitroConfig
from nitro.learner import Learner, LearnStats
from nitro.model import ActorCritic
from nitro.rollout import Rollout
from nitro.selector import GoalHeads, Recorded, Specialist

_GOAL_LR = 3e-4
_GOAL_EPOCHS = 2
_GOAL_MINIBATCH = 4096
# Fewer samples than this many per minibatch and a model sits this update out.
_MIN_PER_MINIBATCH = 8


UNKNOWN, DROPPED = -1, -2  # a goal label not yet known, or never to be (a timeout cut the horizon short)


@dataclasses.dataclass
class Labels:
    """`goal_labels`' answer: per step and car, the label; per car, the first episode end of the collection
    (its step, `steps` for none) and what it was."""

    labels: torch.Tensor  # [T, N] int64: 0 no goal, 1 the car's team, 2 the other team, UNKNOWN or DROPPED
    first_end: torch.Tensor  # [N] int64
    first_kind: torch.Tensor  # [N] int64: as a label (0, 1, 2 or DROPPED)


def goal_labels(goals: torch.Tensor, terminal: torch.Tensor, truncated: torch.Tensor, horizon: int) -> Labels:
    """Who scored within `horizon` steps of each step (`goals[t]`: how step t ended, 1, -1 or 0): a goal by
    the car's team first is 1, by the other team 2, an end without one or no end within the horizon 0. UNKNOWN
    when the collection ends before the horizon does (a later one tells), DROPPED when a timeout cut it."""
    steps = goals.shape[0]
    labels = torch.full(goals.shape, UNKNOWN, dtype=torch.int64, device=goals.device)
    end_at = torch.full_like(goals[0], steps, dtype=torch.int64)
    end_kind = torch.full_like(labels[0], UNKNOWN)
    for t in range(steps - 1, -1, -1):
        ended = terminal[t] | truncated[t]
        kind = torch.where(
            goals[t] > 0, 1, torch.where(goals[t] < 0, 2, torch.where(truncated[t], DROPPED, 0))
        )
        end_at = torch.where(ended, t, end_at)
        end_kind = torch.where(ended, kind, end_kind)
        within = end_at - t < horizon
        labels[t] = torch.where(within, end_kind, 0 if t + horizon <= steps else UNKNOWN)
    return Labels(labels, end_at, end_kind)


@dataclasses.dataclass
class _Samples:
    obs: torch.Tensor  # [n, widest]
    model: torch.Tensor  # [n] int64
    row: torch.Tensor  # [n] int64: the car
    episode: torch.Tensor  # [n] int64
    seen: torch.Tensor  # [n] int64: steps of its horizon watched so far

    def __getitem__(self, keep: torch.Tensor) -> _Samples:
        return _Samples(*(t[keep] for t in dataclasses.astuple(self)))

    @staticmethod
    def joined(parts: Sequence[_Samples]) -> _Samples:
        return _Samples(*(torch.cat(ts) for ts in zip(*(dataclasses.astuple(p) for p in parts), strict=True)))


class GoalSamples:
    """The goal heads' training states: a uniform sample of the learner's car-steps, each learned from
    once its label is known; a collection is shorter than the horizon, so most wait for the next ones (their
    observations kept in FP16). A collection's states are taken all or none (none while too many still wait):
    which of them settle at once depends on what happened, so keeping only some would favour goals."""

    def __init__(self, horizon: int, per_collection: int = 8192, waiting: int = 32) -> None:
        self.horizon, self.per_collection = horizon, per_collection
        self.cap = waiting * per_collection  # states that may wait at once
        self.pending: _Samples | None = None

    def take(self, recorded: Recorded, rows: torch.Tensor | None) -> tuple[_Samples, torch.Tensor] | None:
        """The states whose label this collection settles, and their labels."""
        found = goal_labels(recorded.goals, recorded.terminal, recorded.truncated, self.horizon)
        steps = recorded.goals.shape[0]
        done: list[tuple[_Samples, torch.Tensor]] = []
        if self.pending is not None and self.pending.row.numel():
            p = self.pending
            need = self.horizon - p.seen
            end, kind = found.first_end[p.row], found.first_kind[p.row]
            label = torch.where(end < need, kind, torch.where(need <= steps, 0, UNKNOWN))
            settled = label >= 0
            done.append((p[settled], label[settled]))
            self.pending = p[label == UNKNOWN]
            self.pending.seen += steps
        eligible = (
            torch.ones_like(found.labels, dtype=torch.bool)
            if rows is None
            else rows[None, :].expand_as(found.labels)
        )
        waits = 0 if self.pending is None else self.pending.row.numel()
        room = self.cap - waits >= self.per_collection
        rate = min(1.0, self.per_collection / max(1, int(eligible.sum()))) if room else 0.0
        chosen = eligible & (torch.rand(found.labels.shape, device=found.labels.device) < rate)
        t, row = chosen.nonzero(as_tuple=True)
        fresh = _Samples(
            recorded.obs[t, row].half(), recorded.driver[t, row], row, recorded.episode[t, row], steps - t
        )
        label = found.labels[t, row]
        done.append((fresh[label >= 0], label[label >= 0]))
        waiting = fresh[label == UNKNOWN]
        self.pending = waiting if self.pending is None else _Samples.joined([self.pending, waiting])
        samples = _Samples.joined([d[0] for d in done])
        labels = torch.cat([d[1] for d in done])
        return (samples, labels) if labels.numel() else None

    def restart(self) -> None:
        """The games were rebuilt (a save): every waiting state's game ended unseen, so all are dropped,
        whatever they would have become (keeping some would favour an outcome)."""
        self.pending = None


def model_rollout(recorded: Recorded, model: int, obs_size: int) -> tuple[Rollout, torch.Tensor]:
    """Model `model`'s view of the collection as a rollout, and the samples it drove ([T, N] bool). Where it
    hands the car over, its stretch ends as a truncation valued by its own value of the next state."""
    r = recorded
    mine = r.driver == model
    following = torch.cat([r.driver[1:], torch.full_like(r.driver[:1], model)])
    ended = r.terminal | r.truncated
    handed = mine & ~ended & (following != model)
    values = r.values[..., model]
    next_values = torch.cat([values[1:], r.bootstrap_values[None, :, model]])
    rollout = Rollout(
        obs=r.obs[..., :obs_size],
        mask=r.mask,
        actions=r.actions,
        log_probs=r.log_probs,  # a collection's first step was chosen before the last update: harmless
        values=values,
        rewards=r.rewards,
        terminal=r.terminal,
        truncated=r.truncated | handed,
        truncated_values=torch.where(r.truncated, r.truncated_values[..., model], next_values),
        bootstrap_values=r.bootstrap_values[:, model],
        episode=r.episode,
        episode_step=r.episode_step,
    )
    return rollout, mine


@dataclasses.dataclass
class ModelStats:
    """One model's update: its learning stats (None: it sat out) and the steps it learned from."""

    learn: LearnStats | None
    samples: int


class ModelLearning:
    """The models' learners and the goal heads' optimizer of a multi-model run."""

    def __init__(
        self,
        models: Sequence[ActorCritic],
        specialists: Sequence[Specialist],
        config: NitroConfig,
        gamma: float,
        device: torch.device,
        fp16: bool,
        obs_sizes: Sequence[int],
        goal_heads: GoalHeads | None,
    ) -> None:
        self.config = config
        self.models = list(models)
        self.obs_sizes = list(obs_sizes)
        self.goal_heads = goal_heads
        self.horizon = max(1, round(config.selector.goal_head_seconds * 120 / config.env.tick_skip))
        self.learners: list[Learner] = []
        if config.selector.train_models:
            for index, (model, specialist) in enumerate(zip(models, specialists, strict=True)):
                rate = config.selector.model_rate(index)
                ppo = dataclasses.replace(
                    config.ppo,
                    policy_lr=rate,
                    value_lr=rate,
                    graphed_steps=False,
                    goal_critic_weight=0.0,
                    vdag_beta=0.0,
                    entropy_target=0.0,
                    advantage_keep=1.0,
                    advantage_keep_rest=1.0,
                    value_share=1.0,
                )
                learner = Learner(model, ppo, gamma, device, fp16)
                if specialist.value_norm is not None and ppo.popart:
                    learner.value_norm.load_state_dict(
                        {key: value.to(device) for key, value in specialist.value_norm.items()}
                    )
                # Its weights when this run first trained it.
                start = specialist.start if specialist.start is not None else model.state_dict()
                learner.set_anchor({k: v for k, v in start.items() if k.startswith(("shared.", "policy."))})
                learner.anchor_weight = config.selector.model_pull
                self.learners.append(learner)
        self.goal_optimizer = (
            torch.optim.Adam(goal_heads.parameters(), lr=_GOAL_LR) if goal_heads is not None else None
        )
        self.goal_samples = GoalSamples(self.horizon)

    def learn(
        self, recorded: Recorded, learner_rows: torch.Tensor | None, models_rate: float
    ) -> tuple[list[ModelStats], float | None]:
        """One update of every model that drove enough (at `models_rate` times its model_lr; none at 0) and
        of the goal heads, on the learner's own cars; returns each model's stats and the goal heads'
        accuracy on the held labels."""
        rows = learner_rows.to(recorded.driver.device) if learner_rows is not None else None
        stats: list[ModelStats] = [ModelStats(None, 0) for _ in self.learners]
        if models_rate > 0 and self.learners:
            stats = []
            for index, learner in enumerate(self.learners):
                rate = self.config.selector.model_rate(index) * float(
                    models_rate
                )  # eased in (models_ramp_steps)
                learner.learning_rates.update(policy=rate, shared=rate, value=rate)
                learner._apply_learning_rates()
                rollout, mine = model_rollout(recorded, index, self.obs_sizes[index])
                kept = mine if rows is None else mine & rows[None, :]
                samples = int(kept.sum())
                if samples < _MIN_PER_MINIBATCH * learner.config.minibatches:
                    stats.append(ModelStats(None, samples))
                    continue
                self.models[index].train()
                try:
                    stats.append(ModelStats(learner.learn(rollout, samples_kept=kept), samples))
                finally:
                    self.models[index].eval()
        accuracy = self._learn_goals(recorded, rows) if self.goal_heads is not None else None
        return stats, accuracy

    def _learn_goals(self, recorded: Recorded, rows: torch.Tensor | None) -> float | None:
        """Supervised passes of each model's goal head on the states it drove whose label is now known; the
        accuracy of its guesses on them before the update (states it had not learned from)."""
        assert self.goal_heads is not None and self.goal_optimizer is not None
        taken = self.goal_samples.take(recorded, rows)
        if taken is None:
            return None
        samples, labels = taken
        right = 0
        batches: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for model, size in enumerate(self.obs_sizes):
            mine = samples.model == model
            obs, target = samples.obs[mine][:, :size].float(), labels[mine]
            if target.numel() == 0:
                continue
            with torch.no_grad():
                right += int((self.goal_heads(model, obs).argmax(-1) == target).sum())
            batches.append((model, obs, target))
        self.goal_heads.train()
        for _ in range(_GOAL_EPOCHS):
            for model, obs, target in batches:
                order = torch.randperm(target.numel(), device=target.device)
                for begin in range(0, target.numel(), _GOAL_MINIBATCH):
                    pick = order[begin : begin + _GOAL_MINIBATCH]
                    loss = F.cross_entropy(self.goal_heads(model, obs[pick]), target[pick])
                    self.goal_optimizer.zero_grad(set_to_none=True)
                    loss.backward()  # type: ignore[no-untyped-call]
                    self.goal_optimizer.step()
        self.goal_heads.eval()
        return right / labels.numel()

    def games_rebuilt(self) -> None:
        """The games were rebuilt (a save), which ends every episode: as when a run resumes, the waiting goal
        states are dropped and each model's running returns start again from zero (the trainer does the
        same for the selector's)."""
        self.goal_samples.restart()
        for learner in self.learners:
            learner.reward_scaler.running = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "learners": [learner.state_dict() for learner in self.learners],
            **(
                {
                    "goal_heads": self.goal_heads.state_dict(),
                    "goal_optimizer": self.goal_optimizer.state_dict(),
                }
                if self.goal_heads is not None and self.goal_optimizer is not None
                else {}
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """A save's learners (when train_models was on then, and is now) and goal heads."""
        if self.learners and len(state.get("learners", [])) == len(self.learners):
            for learner, saved in zip(self.learners, state["learners"], strict=True):
                learner.load_state_dict(saved)
        if self.goal_heads is not None and "goal_heads" in state:
            self.goal_heads.load_state_dict(state["goal_heads"])
            if self.goal_optimizer is not None and "goal_optimizer" in state:
                self.goal_optimizer.load_state_dict(state["goal_optimizer"])


def model_columns(labels: Sequence[str], train: bool, goals: bool) -> list[str]:
    """metrics.csv's columns for the models' learning."""
    columns: list[str] = ["models_lr_share"] if train else []
    if train:
        for label in labels:
            columns += [
                f"model_kl_to_start_{label}",
                f"model_policy_change_{label}",
                f"model_explained_variance_{label}",
            ]
    if goals:
        columns.append("selector_goal_accuracy")
    return columns


def model_row(
    labels: Sequence[str], stats: Sequence[ModelStats], accuracy: float | None, models_rate: float
) -> dict[str, Any]:
    row: dict[str, Any] = {"models_lr_share": models_rate} if stats else {}
    for label, model in zip(labels, stats, strict=False):
        learn = model.learn
        row[f"model_kl_to_start_{label}"] = learn.start_kl if learn is not None else ""
        row[f"model_policy_change_{label}"] = learn.policy_change if learn is not None else ""
        ev = learn.explained_variance if learn is not None else math.nan
        row[f"model_explained_variance_{label}"] = ev if math.isfinite(ev) else ""
    if accuracy is not None:
        row["selector_goal_accuracy"] = accuracy
    return row
