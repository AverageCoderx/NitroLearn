"""The PPO update: advantages, value targets and several minibatch gradient steps per iteration."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from nitro import fp16_sums
from nitro.config.schema import PPOConfig
from nitro.graphed_steps import HEAD_TARGET, GraphedSteps
from nitro.model import ENCODERS, ActorCritic
from nitro.muon import MuonAdam
from nitro.ppo import (
    PopArt,
    RewardScaler,
    expectile_loss,
    gae,
    kept_advantages,
    kept_count,
    kept_counts,
    kept_rows,
    masked_kl,
    masked_log_probs,
    mean_of,
    normalized_entropy,
    policy_loss,
    potential_shaping,
    scaled_shaping,
    spread_of,
    twin_target,
    variance_of,
    weighted_mean,
)
from nitro.rollout import Rollout

if TYPE_CHECKING:
    from nitro.multigpu import Cards, GradientSums

log = logging.getLogger(__name__)

# FP16 loss scaling starts here and adapts; FP32 uses a fixed scale of 1 and the scaler only serves to
# skip (and count) any update whose gradients are not finite, so NaN never reaches the weights.
_FP16_INITIAL_SCALE = 2.0**12
_NEVER = 2**31 - 1
# States per iteration on which policy_change and inference_mismatch are measured, spread evenly. Their two
# extra forward passes cost time in proportion to this number.
_PROBE_STATES = 1024
# The entropy controller keeps the bonus at most this multiple of ppo.entropy (its floor is
# ppo.entropy_target_min's).
_ENTROPY_MAX = 10.0


@dataclass(frozen=True)
class LearnStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    grad_norms: dict[str, float]  # mean over applied updates, before clipping
    reward_scale: float
    gradient_steps: int
    skipped_updates: int  # updates dropped because their gradients were not finite
    grad_scale: float  # current loss scale (FP16) or 1 (FP32)
    param_change: float  # |θ_after - θ_before| / |θ_before| over the whole iteration
    policy_change: float  # exact KL(policy before || after) the iteration's update, on sampled states
    inference_mismatch: float  # KL estimate, collected vs learner log-probs before any update (numerics only)
    params_finite: bool
    policy_lr: float = math.nan  # the policy's learning rate for the next update
    value_only_steps: int = 0  # of gradient_steps, those that trained the value side alone (a value warm-up)
    # Per network (policy, value, shared): |θ_after - θ_before| over the whole iteration.
    update_norms: dict[str, float] = field(default_factory=dict)
    # Per network: |θ_after|; update_norms / weight_norms is the network's relative step.
    weight_norms: dict[str, float] = field(default_factory=dict)
    # Per network with layers a LayerNorm follows (`ActorCritic.scale_free_pairs`): their weight
    # matrices' change over their size before the update, and that size over an orthogonal start's (growth:
    # empty unless model.init = "orthogonal").
    turns: dict[str, float] = field(default_factory=dict)
    growths: dict[str, float] = field(default_factory=dict)
    # The GCO learning signals' measurements, by metrics column (`signal_columns`); empty when off.
    signals: dict[str, float] = field(default_factory=dict)
    # The critic's mean value of the fresh starts learned from, raw reward units; NaN without any.
    start_value: float = math.nan

    @property
    def losses_finite(self) -> bool:
        values = (
            self.policy_loss,
            self.value_loss,
            self.entropy,
            self.approx_kl,
            self.explained_variance,
            self.policy_change,
            self.inference_mismatch,
        )
        return all(math.isfinite(v) for v in values)


def signal_columns(config: PPOConfig) -> list[str]:
    """The metrics columns of the GCO learning signals switched on."""
    on = config.signals_on()
    columns = {
        "goal_critic_weight": ["goal_explained_variance", "goal_loss"],
        "vdag_beta": ["vdag_loss", "vdag_dose", "vdag_headroom", "vdag_clamped"],
        "advantage_keep": ["advantage_kept", "advantage_kept_size"]
        + (["advantage_rows"] if config.advantage_keep_rest < 1 else []),
        "entropy_target": ["entropy_coef"],
    }
    found = [column for name in on for column in columns[name]]
    return found + (["head_loss_weight"] if config.head_loss_scale == "popart" else [])


@dataclass(frozen=True)
class HeadValues:
    """The extra value heads' outputs before an update, [T, N] by head: at each step's state, and at
    the state each step reached (`Rollout.next_values`' layout: the next step's state, a cut episode's final
    state, after the last step the state the games stand in)."""

    now: dict[str, torch.Tensor]
    reached: dict[str, torch.Tensor]


@torch.no_grad()
def head_values_on(
    model: ActorCritic,
    names: tuple[str, ...],
    fp16: bool,
    rollout: Rollout,
    last_obs: torch.Tensor,
    cuts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    chunk: int = 65_536,
    columns: torch.Tensor | None = None,
    sums: bool = False,
) -> HeadValues:
    """The heads `names` on a rollout's states: `last_obs` [N, obs] the states after its last step, `cuts`
    (steps, agents, final observations) its truncated steps' final states. Only the heads in `names` run.
    ppo.lean_heads_pass: `columns` the agents valued (the learner's; 0 on the others, a cut
    of another agent dropped), `sums` FP16 products keeping FP16 sums (run.fp16_sums)."""

    def run(parts: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
        pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        with fp16_sums.kept(sums):
            for part in parts:
                with torch.autocast(part.device.type, dtype=torch.float16, enabled=fp16):
                    values = model.head_values(part, names)
                for name in names:
                    pieces[name].append(values[name].float())
        return {name: torch.cat(v) for name, v in pieces.items()}

    steps, agents = rollout.steps, rollout.agents
    if columns is None:
        flat = rollout.obs.reshape(steps * agents, -1)
        now = {k: v.view(steps, agents) for k, v in run(torch.split(flat, chunk)).items()}
        last = run(torch.split(last_obs, chunk))
    else:
        width = columns.numel()
        every = max(1, chunk // max(width, 1))  # steps per part: the columns copied a part at a time
        parts = (
            rollout.obs[t : t + every].index_select(1, columns).reshape(-1, rollout.obs.shape[-1])
            for t in range(0, steps, every)
        )
        now = {k: v.view(steps, width) for k, v in run(parts).items()}
        last = run(torch.split(last_obs.index_select(0, columns), chunk))
    reached = {name: torch.cat([now[name][1:], last[name][None]]) for name in names}
    if columns is not None:
        now = {name: _spread(v, columns, agents) for name, v in now.items()}
        reached = {name: _spread(v, columns, agents) for name, v in reached.items()}
    if cuts is not None and cuts[0].numel():
        cut_steps, cut_agents, cut_obs = cuts
        keep = None
        if columns is not None:  # without waiting for the card: other agents' cuts written as 0
            valued = torch.zeros(agents, dtype=torch.bool, device=columns.device)
            valued[columns] = True
            keep = valued[cut_agents]
        for name, values in run(torch.split(cut_obs, chunk)).items():
            reached[name][cut_steps, cut_agents] = values if keep is None else torch.where(keep, values, 0.0)
    return HeadValues(now, reached)


def _spread(values: torch.Tensor, columns: torch.Tensor, agents: int) -> torch.Tensor:
    """[T, k] values of the agents `columns` as [T, agents], 0 elsewhere."""
    out = values.new_zeros(values.shape[0], agents)
    out[:, columns] = values
    return out


@dataclass(frozen=True)
class _Signals:
    """What the goal critic and the twin critics add to an update (`Learner._signals`)."""

    extra: torch.Tensor  # added to the advantages, [T, N] of the learner's agents
    targets: dict[str, torch.Tensor]  # the heads' targets, [T, N]: "goal", "vdag" (both twins')
    stats: dict[str, torch.Tensor]  # 0-dim, read with the update's other statistics


@dataclass(frozen=True)
class _Rows:
    """The samples an update's steps run on with ppo.advantage_keep_rest (`Learner._advantage_rows`)."""

    picked: torch.Tensor  # their places among the update's n samples: the kept ones, then the drawn ones
    weights: torch.Tensor  # [n]: 1 kept, m / r drawn, 0 left out
    chosen: torch.Tensor  # [n] bool: the kept ones
    advantages: torch.Tensor  # [n]: the standardised advantage kept, else 0
    kept: torch.Tensor  # k / n and the kept share of |A|, added to `_kept` once per joint step
    share: float  # |R| / n, the metrics' advantage_rows

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A step's advantages and weights: the kept ones x (the step's weights) / (its kept count), so its
        weighted policy loss is the mean over its kept samples, as advantage_keep's step, whatever the drawn
        ones weigh. Computed on the device: no wait."""
        weights = self.weights[batch]
        scale = weights.sum() / self.chosen[batch].sum().clamp_min(1)
        return self.advantages[batch] * scale, weights


# How Adam runs, chosen for this machine; never taken from a save.
_ADAM_IMPLEMENTATION = ("fused", "foreach", "capturable", "differentiable")


class _Identity:
    """Stands in for PopArt when value targets are learned unnormalised."""

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def update(self, targets: torch.Tensor, cards: Cards | None = None) -> None:
        pass

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass


def _standardised(advantages: torch.Tensor, samples: torch.Tensor | None = None) -> torch.Tensor:
    """Zero mean and unit spread. `samples`: per sample 1, or 0 for padding, which the statistics leave
    out."""
    if samples is None:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    inside = samples > 0
    count = inside.sum()
    mean = torch.where(inside, advantages, 0.0).sum() / count.clamp_min(1)
    spread = torch.where(inside, (advantages - mean).pow(2), 0.0).sum() / (count - 1).clamp_min(1)
    return torch.where(inside, (advantages - mean) / (spread.sqrt() + 1e-8), torch.zeros_like(advantages))


def _rate_groups(
    model: ActorCritic, groups: dict[str, list[torch.nn.Parameter]], encoder_scale: float
) -> list[tuple[str, float, list[torch.nn.Parameter]]]:
    """Each network's parameters as one optimizer group, or, when `encoder_scale` is not 1, its attention
    encoder's apart with that factor on the rate. The parameters keep their order either way, so a save
    loads whichever way it was grouped (`_regrouped`)."""
    out: list[tuple[str, float, list[torch.nn.Parameter]]] = []
    for name, params in groups.items():
        network = getattr(model, name)
        first = network[0] if len(network) else None
        if not isinstance(first, ENCODERS) or encoder_scale == 1.0:
            out.append((name, 1.0, params))
            continue
        mine = list(first.parameters())
        rest = params[len(mine) :]
        assert all(a is b for a, b in zip(mine + rest, params, strict=True)), "the encoder comes first"
        out += [(name, encoder_scale, mine), (name, 1.0, rest)]
    return out


def _regrouped(saved: list[dict[str, Any]], optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """The saved optimizer groups split as `optimizer`'s are, when the same parameters were grouped
    another way (ppo.encoder_lr_scale switched on or off between stop and resume)."""
    sizes = [len(group["params"]) for group in optimizer.param_groups]
    ids = [index for group in saved for index in group["params"]]
    if [len(group["params"]) for group in saved] == sizes or len(ids) != sum(sizes):
        return saved
    starts = itertools.accumulate([0, *sizes[:-1]])
    return [
        {**saved[0], "params": ids[start : start + size]} for start, size in zip(starts, sizes, strict=True)
    ]


def _compacted(values: torch.Tensor, flags: torch.Tensor, count: int) -> torch.Tensor:
    """`values[flags]` when `count`, the number of flags set, is known already: without the wait for the
    graphics card that a boolean index takes. The values not flagged go to a spare place at the end."""
    places = torch.where(flags, flags.cumsum(0) - 1, count)
    out = values.new_empty(count + 1)
    out[places] = values
    return out[:count]


def goals_continuing(rollout: Rollout, goals: torch.Tensor) -> Rollout:
    """`rollout` with each `goals` row ([steps, agents] bool) as a cut that reached the next row's state, the
    next game's first (after the last row, the bootstrap value's), for ppo.goal_ends = 'continue':
    GAE bootstraps from that value and stops there. The plain loops and the fused routine take it as it is."""
    following = torch.cat([rollout.values[1:], rollout.bootstrap_values[None]])
    return replace(
        rollout,
        terminal=rollout.terminal & ~goals,
        truncated=rollout.truncated | goals,
        truncated_values=torch.where(goals, following, rollout.truncated_values),
    )


def _host_values(tensors: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Each tensor's values as Python floats, read from the graphics card at once instead of one wait
    each. Widened to FP64 first, which holds every FP32, FP64, bool and integer below 2**53 exactly, so each
    float is the one `.item()` would give. One concatenation per type, then one read."""
    by_type: dict[torch.dtype, list[torch.Tensor]] = {}
    for tensor in tensors.values():
        by_type.setdefault(tensor.dtype, []).append(tensor.reshape(-1))
    read = torch.cat([torch.cat(group).double() for group in by_type.values()]).tolist()
    starts, start = {}, 0
    for dtype, group in by_type.items():
        starts[dtype] = start
        start += sum(t.numel() for t in group)
    out = {}
    for name, tensor in tensors.items():
        begin = starts[tensor.dtype]
        out[name] = read[begin : begin + tensor.numel()]
        starts[tensor.dtype] = begin + tensor.numel()
    return out


# What a learning step's first half hands over (`Learner._step_parts`): measured, this card's samples in it,
# and the cards' sums running beside the backward pass (or None).
_Half = tuple[torch.Tensor, "int | torch.Tensor", "GradientSums | None"]


def _finished(
    parts: Generator[_Half, torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]],
    measured: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Resumes a learning step's second half with `measured` as summed; returns what the step returns."""
    try:
        parts.send(measured)
    except StopIteration as done:
        result: tuple[torch.Tensor, dict[str, torch.Tensor]] = done.value
        return result
    raise RuntimeError("a learning step handed over twice")


class Learner:
    def __init__(
        self,
        model: ActorCritic,
        config: PPOConfig,
        gamma: float,
        device: torch.device,
        fp16: bool,
    ) -> None:
        self.model = model
        self.config = config
        self.gamma = gamma
        self.device = device
        self.fp16 = fp16
        # run.fp16_sums: which of learning's passes keep their FP16 products' sums in FP16.
        self.fp16_sums = "off"
        groups = model.parameter_groups()
        self.groups = groups
        # The layers a LayerNorm follows, by network in `groups`' order: ppo.weight_decay shrinks them
        # (weight and bias), the turn and growth metrics measure their weights; with model.init =
        # "orthogonal" their weights' starting size, sqrt(2 x sum of min(rows, cols)), per network.
        pairs = model.scale_free_pairs()
        self.scale_free = {
            name: [(w, b) for owner, w, b in pairs if owner == name]
            for name in groups
            if any(owner == name for owner, _, _ in pairs)
        }
        self._start_sizes: torch.Tensor | None = None
        if model.config.init == "orthogonal" and self.scale_free:
            starts = [2.0 * sum(min(w.shape) for w, _ in found) for found in self.scale_free.values()]
            self._start_sizes = torch.tensor(starts, device=device).sqrt()
        self.learning_rates = {
            "policy": config.policy_lr,
            "shared": config.policy_lr,
            "value": config.value_lr,
        }
        # The optimizer's groups: (network, factor on its rate, parameters). An attention encoder gets a
        # group of its own when ppo.encoder_lr_scale is not 1; clipping stays per network.
        self.rate_groups = _rate_groups(model, groups, config.encoder_lr_scale)
        rate_groups = [
            {"params": params, "lr": self.learning_rates[name] * factor}
            for name, factor, params in self.rate_groups
        ]
        fused = device.type == "cuda"
        # ppo.optimizer: Muon for the hidden layers' matrices, Adam for the rest, in the same groups.
        self.optimizer = (
            MuonAdam(
                rate_groups,
                model.muon_parameters(),
                eps=config.adam_eps,
                fused=fused,
                batched=config.muon_batched,
                scale=config.muon_lr_scale,
            )
            if config.optimizer == "muon"
            else torch.optim.Adam(rate_groups, eps=config.adam_eps, fused=fused)
        )
        self.scaler = self._loss_scaler()
        self.value_norm: PopArt | _Identity = PopArt(model.value_output) if config.popart else _Identity()
        if config.head_loss_scale == "popart" and not config.popart:
            raise ValueError("ppo.head_loss_scale = 'popart' needs ppo.popart = true")
        self.reward_scaler = RewardScaler(gamma, config.reward_clip, device)
        # The GCO learning signals: the entropy bonus (moved by ppo.entropy_target, else ppo.entropy),
        # the goal critic's discount (the run sets it from goal_critic_half_life), the agent step the twin
        # critics last started afresh at, and advantage_keep's sums over the steps (kept samples, their |A|).
        self.entropy_coef = config.entropy
        self.goal_gamma: float | None = None
        self.twins_started: float | None = None
        self._kept = torch.zeros(2, device=device)
        # With several cards: every card's samples in each learning step, for advantage_keep's choice,
        # read from the card at its first use (`_cards_step_sizes`).
        self._step_sizes: list[int] | None = None
        self._step_sizes_later: Callable[[], list[int]] | None = None
        # While set, every pass trains the value network alone and the policy stays as it is: the value
        # warm-up after a copy of replays (pretrain.value_warmup_steps).
        self.policy_frozen = False
        # Whole learning steps as CUDA graphs on a graphics card (ppo.graphed_steps).
        self.graphed = GraphedSteps(self) if config.graphed_steps and device.type == "cuda" else None
        # The cards this bot learns on with run.cards: every statistic that steers learning, and every
        # step's gradient, is taken over all cards' samples. None = one card.
        self.cards: Cards | None = None
        # Whether this update's steps sum while the backward pass runs (`learn` decides, the same way on
        # every card).
        self.overlapped = False
        self._gradient_sums: GradientSums | None = None
        # The last `rows` passed in, its version and its agents' positions (`_driven`).
        self._driven_rows: tuple[torch.Tensor, int, torch.Tensor] | None = None
        # Whether ppo.value_share's rows per step were logged (once, at the first joint update).
        self._value_share_said = False

    def _loss_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type,
            init_scale=_FP16_INITIAL_SCALE if self.fp16 else 1.0,
            growth_interval=2000 if self.fp16 else _NEVER,
        )

    def fresh_loss_scaler(self) -> None:
        """A new loss scaler with this one's scale and growth count: a step cut short after `unscale_` leaves
        a scaler mid-step until its `update`, which its `load_state_dict` does not undo. Its tensors
        are made now, not at its first `scale`: graphed steps' capture puts back only those that exist."""
        state = self.scaler.state_dict()
        self.scaler = self._loss_scaler()
        self.scaler.load_state_dict(state)
        self.scaler.scale(torch.zeros((), device=self.device))

    @property
    def _sums(self) -> GradientSums | None:
        if self._gradient_sums is None and self.cards is not None:
            from nitro.multigpu import GradientSums

            self._gradient_sums = GradientSums(self.cards, list(self.model.parameters()))
        return self._gradient_sums

    def state_dict(self) -> dict[str, Any]:
        """Everything the learner needs to continue exactly."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "value_norm": self.value_norm.state_dict(),
            "reward_scaler": self.reward_scaler.state_dict(),
            "learning_rates": dict(self.learning_rates),
            **({"optimizer_kind": self.config.optimizer} if self.config.optimizer != "adam" else {}),
            **({"entropy_coef": self.entropy_coef} if self.config.entropy_target > 0 else {}),
            **({"twins_started": self.twins_started} if self.twins_started is not None else {}),
        }

    def load_state_dict(self, state: dict[str, Any], load_scaler: bool = True) -> None:
        """Restores a checkpoint; `load_scaler=False` keeps a fresh loss scaler (the precision changed)."""
        self.model.load_state_dict(state["model"])
        if state.get("optimizer_kind", "adam") != self.config.optimizer:
            # ppo.optimizer switched between stop and resume: the other kind's state does not fit.
            log.warning(
                "ppo.optimizer is now '%s', the save's was '%s': the optimizer starts afresh.",
                self.config.optimizer,
                state.get("optimizer_kind", "adam"),
            )
            state = {**state, "optimizer": self.optimizer.state_dict()}
        # PyTorch takes every setting of the saved parameter groups, including how Adam runs (fused on a
        # graphics card, foreach on the CPU). A save from another machine or from `nitro surgery` (built on
        # the CPU) must not switch off the fused kernel here; set before loading, so the step counters are
        # also placed as that kernel expects.
        saved = state["optimizer"]
        here = {
            key: self.optimizer.defaults[key]
            for key in _ADAM_IMPLEMENTATION
            if key in self.optimizer.defaults
        }
        groups = [{**group, **here} for group in _regrouped(saved["param_groups"], self.optimizer)]
        self.optimizer.load_state_dict({**saved, "param_groups": groups})
        # The saved parameter groups carry the learning rates and epsilon of the run that saved them; the
        # current settings win, so a learning rate changed between stop and resume takes effect.
        self._apply_learning_rates()
        for group in self.optimizer.param_groups:
            group["eps"] = self.config.adam_eps
        if load_scaler:
            self.scaler.load_state_dict(state["scaler"])
        self.value_norm.load_state_dict(state["value_norm"])
        self.reward_scaler.load_state_dict(state["reward_scaler"])
        controlled = self.config.entropy_target > 0 and "entropy_coef" in state
        # Within the current range: a floor raised since the save holds from the first update.
        self.entropy_coef = (
            self._entropy_kept(float(state["entropy_coef"])) if controlled else self.config.entropy
        )
        self.twins_started = state.get("twins_started")

    def learn(
        self,
        rollout: Rollout,
        rows: torch.Tensor | None = None,
        heads: HeadValues | None = None,
    ) -> LearnStats:
        """One PPO update on the rollout.

        `rows` (bool per agent) limits it to the agents the learner drives (self-play against frozen past
        versions, nitro.selfplay); every other agent's data is left out. `heads`: the extra value heads on
        the rollout before the update (`head_values_on`), which the goal critic and the twin critics need.
        """
        if self.config.advantage_keep_rest < 1:
            refused = {
                "several cards": self.cards is not None,
                "graphed steps": self.graphed is not None,
                "ppo.value_share below 1": self.config.value_share < 1,
            }
            if any(refused.values()):
                used = ", ".join(what for what, on in refused.items() if on)
                raise ValueError(f"ppo.advantage_keep_rest below 1 does not work with {used} yet")
        episode_end = rollout.terminal | rollout.truncated
        index = self._driven(rows) if rows is not None else None
        rewards = self.reward_scaler(rollout.rewards, episode_end, index, self.cards)
        goals = self._goal_rows(rollout)

        def select(t: torch.Tensor) -> torch.Tensor:
            return t.index_select(1, index) if index is not None else t

        advantages, targets, values = self._advantages(rollout, rewards, select, goals)
        obs, mask, actions, old_log_probs = (
            select(t) for t in (rollout.obs, rollout.mask, rollout.actions, rollout.log_probs)
        )
        kept_targets = targets.reshape(-1)
        kept_values = values.reshape(-1)
        explained_variance = 1 - (kept_targets - kept_values).var() / kept_targets.var().clamp_min(1e-8)
        self.value_norm.update(kept_targets, self.cards)
        signals = self._signals(rollout, rewards, select, advantages, values, heads)
        if signals is not None:
            advantages = advantages + signals.extra
        self._kept.zero_()

        samples = rollout.steps * obs.shape[1]
        minibatch = samples // self.config.minibatches
        if minibatch < 2:
            raise ValueError(
                f"{self.config.minibatches} minibatches need at least twice that many samples, got {samples}"
            )
        obs = obs.reshape(samples, -1)
        mask = mask.reshape(samples, -1)
        actions = actions.reshape(samples)
        old_log_probs = old_log_probs.reshape(samples)
        starts = self._start_values(select(rollout.episode_step), kept_values)
        self.overlapped = self.cards is not None and self.cards.overlap
        # The probes look at collected samples only; they come first, so their indices stay valid below.
        probe = torch.arange(0, samples, max(1, samples // _PROBE_STATES), device=self.device)[:_PROBE_STATES]
        if self.config.value_share < 1 and not self.policy_frozen and not self._value_share_said:
            self._value_share_said = True
            self._say_value_share(minibatch)
        self._step_sizes = None
        self._step_sizes_later = (
            self.cards.sizes_later(minibatch)
            if self.cards is not None and 0 < self.config.advantage_keep < 1
            else None
        )

        def flat_rows(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1)

        flat_advantages = flat_rows(advantages)
        flat_targets = flat_rows(self.value_norm.normalize(targets))
        head_targets = (
            {name: flat_rows(t) for name, t in signals.targets.items()} if signals is not None else None
        )
        picked: _Rows | None = None
        if self.config.advantage_keep_rest < 1 and not self.policy_frozen:
            # A value warm-up keeps every sample, as ppo.value_share's value-only passes.
            picked = self._advantage_rows(flat_advantages)
            minibatch = picked.picked.numel() // self.config.minibatches
            if minibatch < 1:
                raise ValueError(
                    f"ppo.advantage_keep_rest leaves {picked.picked.numel()} samples for "
                    f"{self.config.minibatches} minibatches"
                )
        before = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        probe_obs, probe_mask = obs[probe], mask[probe]
        probe_before = self._log_probs(probe_obs, probe_mask)

        totals = torch.zeros(5 + (len(signals.targets) if signals is not None else 0), device=self.device)
        grad_norms = {name: torch.zeros((), device=self.device) for name in self.groups}
        skipped = torch.zeros((), dtype=torch.int64, device=self.device)
        skipped_value_only = torch.zeros((), dtype=torch.int64, device=self.device)
        steps = value_only_steps = 0
        # A frozen policy (a value warm-up) trains the value side alone in every pass.
        passes = [self.policy_frozen] * self.config.epochs
        if self.graphed is not None:  # its buffers fit the largest minibatch this rollout can give
            driven = index.numel() if index is not None else rollout.agents
            capacity = rollout.steps * driven // self.config.minibatches
            example = {"obs": obs, "mask": mask, "actions": actions, "old_log_probs": old_log_probs}
            if head_targets is not None:  # the extra heads' targets, buffers as the advantages'
                example |= {HEAD_TARGET + name: t for name, t in head_targets.items()}
            self.graphed.reserve(
                capacity, {**example, "advantages": flat_advantages, "targets": flat_targets}
            )
        for value_only in passes:
            if picked is not None:
                every = picked.picked.numel()
                order = picked.picked[torch.randperm(every, device=self.device)]
            else:
                order = torch.randperm(samples, device=self.device)
            for begin in range(0, minibatch * self.config.minibatches, minibatch):
                batch = order[begin : begin + minibatch]
                if self.graphed is not None and self.graphed.takes():
                    flats = {"obs": obs, "mask": mask, "actions": actions, "old_log_probs": old_log_probs}
                    flats |= {"advantages": flat_advantages, "targets": flat_targets}
                    if head_targets is not None:
                        flats |= {HEAD_TARGET + name: t for name, t in head_targets.items()}
                    measured, norms = self.graphed.step(flats, batch, value_only)
                else:
                    extra: dict[str, Any] = {}
                    if head_targets is not None:
                        extra["head_targets"] = {name: t[batch] for name, t in head_targets.items()}
                    step_advantages = flat_advantages[batch]
                    if picked is not None:  # standardised and filtered for the whole update already
                        step_advantages, weights = picked.step(batch)
                        extra |= {"samples": weights, "prepared": True}
                        if not value_only:
                            self._kept += picked.kept
                    measured, norms = self._step(
                        obs[batch],
                        mask[batch],
                        actions[batch],
                        old_log_probs[batch],
                        step_advantages,
                        flat_targets[batch],
                        value_only,
                        **extra,
                    )
                applied = torch.stack(list(norms.values())).isfinite().all()
                skipped += (~applied).long()
                if value_only:
                    skipped_value_only += (~applied).long()
                for name, norm in norms.items():
                    grad_norms[name] += torch.where(applied, norm, torch.zeros_like(norm))
                if value_only:
                    value_only_steps += 1
                    totals[1] += measured[1]
                    totals[5:] += measured[5:]
                else:
                    totals += measured
                    steps += 1

        after = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        moved = after - before
        change = moved.norm() / before.norm().clamp_min(1e-12)
        update_sizes = self._group_norms(moved)
        weight_sizes = self._group_norms(after)
        probe_after = self._log_probs(probe_obs, probe_mask)
        policy_change = masked_kl(probe_before, probe_after, probe_mask).mean()
        if self.cards is not None:  # every card reports the same
            policy_change = self.cards.mean(policy_change)
        # Low-variance estimate of KL(collecting policy || the learner's copy of it) on the actions taken;
        # the weights are the same, so only numerical differences between inference and learning remain.
        # FP64 because the differences can be far below FP32's resolution.
        log_ratio = (probe_before.gather(1, actions[probe, None]).squeeze(1) - old_log_probs[probe]).double()
        mismatch = (torch.expm1(log_ratio) - log_ratio).mean()
        if steps > 0:
            averages = totals[:5] / steps
        else:  # a frozen policy: no policy loss; the entropy is the probes' (what the policy would do)
            averages = normalized_entropy(probe_before, probe_mask, self.config.entropy_normalization).mean()
        # Every statistic in one read from the graphics card, each the float its own read would give.
        reads = {
            "averages": averages,
            "value_loss": totals[1],
            "skipped": skipped,
            "skipped_value_only": skipped_value_only,
            "explained_variance": explained_variance,
            "reward_scale": self.reward_scaler.moments.std,
            "change": change,
            "policy_change": policy_change,
            "mismatch": mismatch,
            "finite": after.isfinite().all(),
            "starts": starts,
            **{f"norm {name}": norm for name, norm in grad_norms.items()},
        }
        scale = self.scaler._get_scale_async() if self.scaler.is_enabled() else None
        if scale is not None:
            reads["grad_scale"] = scale
        reads["update_norms"] = update_sizes
        reads["weight_norms"] = weight_sizes
        if self.scale_free:
            reads["turns"], sizes = self._turns(before, moved)
            if self._start_sizes is not None:
                reads["growths"] = sizes / self._start_sizes
        if signals is not None:  # the GCO signals' statistics and their heads' losses
            reads |= {f"signal {name}": value for name, value in signals.stats.items()}
            reads["head losses"] = totals[5:] / max(steps + value_only_steps, 1)
            if self.config.head_loss_scale == "popart":
                reads["head_loss_weight"] = self._head_loss_weight()
        if 0 < self.config.advantage_keep < 1 and steps > 0:
            reads["kept"] = self._kept / steps
        got = _host_values(reads)
        update_norms = dict(zip(self.groups, got["update_norms"], strict=True))

        def one(name: str) -> float:
            return got[name][0] if name in got else math.nan

        if self.config.weight_decay > 0:  # by the steps applied, at the rates they ran at
            skipped_joint = int(one("skipped")) - int(one("skipped_value_only"))
            self._decay(steps - skipped_joint, steps + value_only_steps - int(one("skipped")))
        means = got["averages"] if steps > 0 else [0.0, 0.0, one("averages"), 0.0, 0.0]
        skipped_updates = int(one("skipped"))
        measured_signals = self._signal_stats(signals, got, steps, means[2])
        if self.config.advantage_keep_rest < 1:  # a value warm-up's steps run on every sample
            measured_signals["advantage_rows"] = picked.share if picked is not None else 1.0
        applied_steps = max(steps + value_only_steps - skipped_updates, 1)
        # The policy and the shared trunk only move in joint steps; value-only steps add nothing to them.
        applied_joint = max(steps - (skipped_updates - int(one("skipped_value_only"))), 1)
        start_sum, start_count = got["starts"]
        return LearnStats(
            policy_loss=means[0],
            value_loss=one("value_loss") / (steps + value_only_steps),
            entropy=means[2],
            approx_kl=means[3],
            clip_fraction=means[4],
            explained_variance=one("explained_variance"),
            grad_norms={
                name: one(f"norm {name}") / (applied_steps if name == "value" else applied_joint)
                for name in grad_norms
            },
            reward_scale=one("reward_scale"),
            gradient_steps=steps + value_only_steps,
            skipped_updates=skipped_updates,
            grad_scale=one("grad_scale") if scale is not None else float(self.scaler.get_scale()),
            param_change=one("change"),
            policy_change=one("policy_change"),
            inference_mismatch=one("mismatch"),
            params_finite=bool(one("finite")),
            policy_lr=self.learning_rates["policy"],
            value_only_steps=value_only_steps,
            update_norms=update_norms,
            weight_norms=dict(zip(self.groups, got["weight_norms"], strict=True)),
            turns=dict(zip(self.scale_free, got.get("turns", []), strict=True)),
            growths=dict(zip(self.scale_free, got["growths"], strict=True)) if "growths" in got else {},
            signals=measured_signals,
            start_value=start_sum / start_count * one("reward_scale") if start_count > 0 else math.nan,
        )

    @property
    def extra_heads(self) -> tuple[str, ...]:
        """The value heads whose values `learn` needs before the update (`heads`)."""
        goal = ("goal",) if self.config.goal_critic_weight > 0 else ()
        return goal + (("vdag1", "vdag2") if self.config.vdag_beta > 0 else ())

    @torch.no_grad()
    def _signals(
        self,
        rollout: Rollout,
        rewards: torch.Tensor,
        select: Callable[[torch.Tensor], torch.Tensor],
        advantages: torch.Tensor,
        values: torch.Tensor,
        heads: HeadValues | None,
    ) -> _Signals | None:
        """The goal critic's and the twin critics' share of the advantages and their heads' targets,
        from the heads before the update. Their statistics are over the samples learned from; the order is the
        design's: the twins' shaping, then the goal critic's blend (then advantage_keep, in each step)."""
        goal, twins = self.config.goal_critic_weight > 0, self.config.vdag_beta > 0
        if not (goal or twins):
            return None
        if heads is None:
            raise ValueError("ppo.goal_critic_weight and ppo.vdag_beta need the heads' values (`heads`)")

        def kept(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1)

        def placed(t: torch.Tensor) -> torch.Tensor:  # the samples back into [T, N]
            return t.view_as(advantages)

        terminal, truncated, scaled = select(rollout.terminal), select(rollout.truncated), select(rewards)
        cards = self.cards  # several cards: every statistic over all cards' samples
        spread = spread_of(kept(advantages), cards)  # the usual (extrinsic) advantages'
        added = torch.zeros_like(kept(advantages))
        targets: dict[str, torch.Tensor] = {}
        stats: dict[str, torch.Tensor] = {}  # left on the card: read with the update's statistics
        if twins:
            now_1, now_2 = select(heads.now["vdag1"]), select(heads.now["vdag2"])
            reached_1, reached_2 = select(heads.reached["vdag1"]), select(heads.reached["vdag2"])
            targets["vdag"] = twin_target(scaled, reached_1, reached_2, terminal, truncated, self.gamma)
            potential = torch.minimum(now_1, now_2)  # Phi = V-dagger, the lower twin (the design's code)
            reached = torch.minimum(reached_1, reached_2)
            shaping = potential_shaping(potential, reached, terminal, truncated, self.gamma)
            shaped = scaled_shaping(kept(shaping), spread, self.config.vdag_beta, cards=cards)
            added = added + shaped.added
            stats["vdag_dose"] = spread_of(shaped.added, cards) / spread.clamp_min(1e-12)
            stats["vdag_headroom"] = mean_of(kept(potential - values).clamp_min(0.0), cards)
            stats["vdag_clamped"] = shaped.clamped
        if goal:
            if self.goal_gamma is None:
                raise ValueError("the goal critic needs its discount (Learner.goal_gamma)")
            now_goal, reached_goal = select(heads.now["goal"]), select(heads.reached["goal"])
            goal_advantages, goal_targets = gae(
                scaled, now_goal, reached_goal, terminal, truncated, self.goal_gamma, self.config.gae_lambda
            )
            targets["goal"] = goal_targets
            a, t = kept(goal_advantages), kept(goal_targets)
            centre, unit = mean_of(a, cards), spread_of(a, cards)
            added = added + self.config.goal_critic_weight * spread * (a - centre) / (unit + 1e-8)
            missed = variance_of(t - kept(now_goal), cards)
            stats["goal_explained_variance"] = 1 - missed / variance_of(t, cards).clamp_min(1e-8)
        return _Signals(placed(added), targets, stats)

    def _signal_stats(
        self, signals: _Signals | None, got: dict[str, list[float]], steps: int, entropy: float
    ) -> dict[str, float]:
        """The signals' metrics (`signal_columns`) from the update's read (`got`); the entropy controller
        moves the bonus here."""
        out: dict[str, float] = {}
        if signals is not None:
            out |= {name: got[f"signal {name}"][0] for name in signals.stats}
            out |= dict(zip(signals.targets, got["head losses"], strict=True))
            if "vdag" in out:
                out["vdag_loss"] = out.pop("vdag")
            if "goal" in out:
                out["goal_loss"] = out.pop("goal")
            if self.config.head_loss_scale == "popart":
                out["head_loss_weight"] = got["head_loss_weight"][0]
        if 0 < self.config.advantage_keep < 1 and steps > 0:
            kept, size = got["kept"]
            out |= {"advantage_kept": kept, "advantage_kept_size": size}
        if self.config.entropy_target > 0:
            out["entropy_coef"] = self.entropy_coef  # the bonus this update used
            if steps > 0:
                self._adapt_entropy(entropy)
        return out

    def _adapt_entropy(self, entropy: float) -> None:
        """The entropy controller: the bonus times exp(rate x (target - entropy)), kept within its
        range (`_entropy_kept`); a non-finite entropy leaves it alone. With several cards the entropy is every
        card's mean already (the steps' measurements are averaged), so the bonus moves alike on all."""
        if not math.isfinite(entropy):
            return
        config = self.config
        coef = self.entropy_coef * math.exp(config.entropy_target_rate * (config.entropy_target - entropy))
        self.entropy_coef = self._entropy_kept(coef)

    def _entropy_kept(self, coef: float) -> float:
        """`coef` within ppo.entropy_target_min to `_ENTROPY_MAX` times ppo.entropy."""
        config = self.config
        low, high = config.entropy * config.entropy_target_min, config.entropy * _ENTROPY_MAX
        return min(max(coef, low), high)

    def restart_twins(self, agent_steps: float) -> bool:
        """With ppo.vdag_reinit_steps: the twin critics start again from fresh weights (torch's
        generator, so an exact resume replays it) and fresh optimizer state once that many agent steps have
        passed since they last did (or since the first call). Returns whether they did."""
        period = self.config.vdag_reinit_steps
        if self.config.vdag_beta <= 0 or period <= 0:
            return False
        if self.twins_started is None or agent_steps < self.twins_started:
            self.twins_started = agent_steps
            return False
        if agent_steps - self.twins_started < period:
            return False
        for name in ("vdag1", "vdag2"):
            fresh = self.model.reset_value_head(name)
            if self.cards is not None:  # each card drew its own: card 0's on every card
                self.cards.broadcast_(fresh)
            for parameter in fresh:
                self.optimizer.state.pop(parameter, None)
        self.twins_started = agent_steps
        log.info("The twin critics (vdag1, vdag2) start again from fresh weights at %d steps.", agent_steps)
        return True

    def _pieces(self, flat: torch.Tensor) -> dict[int, torch.Tensor]:
        """`flat` (a number per parameter value, in `model.parameters()` order) cut into each parameter's
        piece, by the parameter's id."""
        params = list(self.model.parameters())
        return dict(zip((id(p) for p in params), torch.split(flat, [p.numel() for p in params]), strict=True))

    def _group_norms(self, flat: torch.Tensor) -> torch.Tensor:
        """Per network (`self.groups`' order), the size of its part of `flat` (`_pieces`): the iteration's
        change of its weights, or the weights themselves; read with the update's statistics."""
        pieces = self._pieces(flat)
        zero = torch.zeros((), device=flat.device)
        norms = [
            torch.sqrt(sum((pieces[id(p)].pow(2).sum() for p in group), zero))
            for group in self.groups.values()
        ]
        return torch.stack(norms)

    def _turns(self, before: torch.Tensor, moved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per network in `self.scale_free`: its LayerNorm-fed weight matrices' change over their size before
        the update (the turn), and that size."""
        was, step = self._pieces(before), self._pieces(moved)
        zero = torch.zeros((), device=before.device)

        def sizes(pieces: dict[int, torch.Tensor]) -> torch.Tensor:
            return torch.stack(
                [
                    torch.sqrt(sum((pieces[id(w)].pow(2).sum() for w, _ in pairs), zero))
                    for pairs in self.scale_free.values()
                ]
            )

        size = sizes(was)
        return sizes(step) / size.clamp_min(1e-12), size

    @torch.no_grad()
    def _decay(self, joint: int, value: int) -> None:
        """ppo.weight_decay: every pair of `self.scale_free` shrinks by (1 - rate x weight_decay) per
        learning step applied to its network (`joint` for the policy and the shared trunk, `value` for the
        value side) at the network's current rate. In place (graphs keep their tensors); every card does the
        same. A LayerNorm follows each pair, so the networks compute the same (to its eps)."""
        for name, pairs in self.scale_free.items():
            steps = value if name == "value" else joint
            if steps > 0:
                factor = (1.0 - self.learning_rates[name] * self.config.weight_decay) ** steps
                torch._foreach_mul_([p for pair in pairs for p in pair], factor)

    def _advantages(
        self,
        rollout: Rollout,
        rewards: torch.Tensor,
        select: Callable[[torch.Tensor], torch.Tensor],
        goals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GAE advantages, value targets and the (denormalised) values they were built from, for the
        selected agents, from the rollout's current value estimates. `goals` (`_goal_rows`): those rows
        continue into the next row's state (ppo.goal_ends = 'continue')."""
        if goals is not None:
            rollout = goals_continuing(rollout, goals)
        advantages, targets, values = self._plain_advantages(rollout, rewards)
        return select(advantages), select(targets), select(values)

    def _goal_rows(self, rollout: Rollout) -> torch.Tensor | None:
        """With ppo.goal_ends = 'continue', the rows a goal ended: every terminal row (the refusals
        leave goals the only terminal rows). None: 'terminal'."""
        return rollout.terminal if self.config.goal_ends == "continue" else None

    def _start_values(self, episode_step: torch.Tensor, kept_values: torch.Tensor) -> torch.Tensor:
        """The sum and count (FP64, over every card) of the denormalised values of the fresh starts among
        the samples learned from: start_value's. Read with the update's statistics."""
        fresh = episode_step.reshape(-1) == 0
        pair = torch.stack([torch.where(fresh, kept_values.double(), 0.0).sum(), fresh.sum().double()])
        return self.cards.sum_(pair) if self.cards is not None else pair

    def _plain_advantages(
        self, rollout: Rollout, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GAE advantages, value targets and denormalised values, every agent."""
        values = self.value_norm.denormalize(rollout.values)
        next_values = self.value_norm.denormalize(rollout.next_values())
        advantages, targets = gae(
            rewards,
            values,
            next_values,
            rollout.terminal,
            rollout.truncated,
            self.gamma,
            self.config.gae_lambda,
        )
        return advantages, targets, values

    def _step(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        targets: torch.Tensor,
        value_only: bool,
        samples: torch.Tensor | None = None,
        apart: bool = False,
        head_targets: dict[str, torch.Tensor] | None = None,
        entropy_coef: torch.Tensor | None = None,
        prepared: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One gradient step on a minibatch. Returns [policy loss, value loss, entropy, approx KL, clip
        fraction] (the policy entries 0 when `value_only`) and each network's gradient norm before
        clipping. With `ppo.microbatch` the step is worked out in parts whose gradients add up to the whole
        step's. `samples`: per sample 1, or 0 for padding (the graphed steps': a fixed-size step,
        the same maths). `apart` (graphed steps on several cards): the advantages come standardised
        (and with ppo.advantage_keep filtered) over every card already and the cards' gradients are
        summed after the backward pass, as `_step_parts`' two halves replay around that sum. `head_targets`:
        the extra heads' targets ("goal", "vdag"), whose losses join the step's. `entropy_coef`: the
        entropy bonus as a tensor (the graphed steps': the controller moves it between updates); None =
        `self.entropy_coef`.
        `prepared` (ppo.advantage_keep_rest): the advantages come standardised and filtered over the
        whole update (`_advantage_rows`), `samples` the rows' weights."""
        parts = self._step_parts(
            obs,
            mask,
            actions,
            old_log_probs,
            advantages,
            targets,
            value_only,
            samples,
            apart,
            head_targets,
            entropy_coef,
            prepared=prepared,
        )
        measured, count, sums = next(parts)
        if sums is not None:
            sums.finish()
            assert self.cards is not None
            measured = self.cards.sum_(measured)
        elif self.cards is not None:  # every card applies the average of the cards' gradients
            measured = self.cards.average_gradients(list(self.model.parameters()), measured, count)
        return _finished(parts, measured)

    def _step_parts(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        targets: torch.Tensor,
        value_only: bool,
        samples: torch.Tensor | None = None,
        apart: bool = False,
        head_targets: dict[str, torch.Tensor] | None = None,
        entropy_coef: torch.Tensor | None = None,
        prepared: bool = False,
    ) -> Generator[_Half, torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """`_step` in two halves: yields (measured, this card's samples, the sums running beside the backward
        pass or None) once the gradients are in, takes `measured` back as the cards summed it, then clips and
        steps."""

        if not value_only and self.config.normalize_advantages and not apart and not prepared:
            # Over the whole step, before any parts; with several cards, over every card's share of it.
            if self.cards is not None:
                advantages = self.cards.standardised(advantages, samples)
            else:
                advantages = _standardised(advantages, samples)
        if not value_only and 0 < self.config.advantage_keep < 1 and not apart and not prepared:
            advantages = self._filtered(advantages, samples)  # over the whole step, before any parts
        self.optimizer.zero_grad(set_to_none=True)
        size = obs.shape[0]
        part = self.config.microbatch or size
        measured: torch.Tensor | None = None
        total = samples.sum().clamp_min(1.0) if samples is not None else None
        # Several cards summing while the backward pass runs: each card's loss counts by its share of
        # all the cards' samples in this step, so the sum is the gradient of the mean over all of them.
        sums = self._sums if self.cards is not None and self.overlapped and not apart else None
        # This card's samples in the step: a graphed step's padding does not count.
        count: int | torch.Tensor = size if samples is None else samples.sum()
        if sums is not None:
            assert self.cards is not None
            # Made on the card: a copy from the processor would wait every step, and a graph cannot hold one.
            mine = (
                count.reshape(1).clone()
                if isinstance(count, torch.Tensor)
                else torch.full((1,), float(size), device=self.device)
            )
            weight = count / self.cards.sum_(mine)
        for begin in range(0, size, part):
            rows = slice(begin, begin + part)
            with fp16_sums.kept(self.fp16_sums in fp16_sums.FORWARD):
                loss, measured_part = self._losses(
                    obs[rows],
                    mask[rows],
                    actions[rows],
                    old_log_probs[rows],
                    advantages[rows],
                    targets[rows],
                    value_only,
                    samples[rows] if samples is not None else None,
                    {name: t[rows] for name, t in head_targets.items()} if head_targets is not None else None,
                    entropy_coef,
                )
            if part < size:  # each part's means count by its share of the step
                if samples is not None and total is not None:
                    share: float | torch.Tensor = samples[rows].sum() / total
                else:
                    share = (min(begin + part, size) - begin) / size
                loss, measured_part = loss * share, measured_part * share
            if sums is not None:
                loss, measured_part = loss * weight, measured_part * weight
                if begin + part >= size:  # the last part: its backward pass completes the gradients
                    sums.start()
            with fp16_sums.kept(self.fp16_sums in fp16_sums.BACKWARD):
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            measured = measured_part if measured is None else measured + measured_part
        assert measured is not None
        measured = yield measured, count, sums
        self.scaler.unscale_(self.optimizer)
        # A network without gradients (the policy in a value-only step) has a norm of 0, made on the device:
        # PyTorch's own 0 would be on the CPU and copying it over would make the host wait.
        norms = {
            name: torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm).detach()
            if any(p.grad is not None for p in params)
            else torch.zeros((), device=self.device)
            for name, params in self.groups.items()
        }
        # Skips the step by itself when any gradient is not finite.
        self.scaler.step(self.optimizer)
        # FP32 keeps its scale at exactly 1; only FP16 adapts (halving on a skip, growing slowly).
        self.scaler.update(None if self.fp16 else 1.0)
        return measured, norms

    def _say_value_share(self, minibatch: int) -> None:
        part = self.config.microbatch or minibatch
        share = self.config.value_share
        taken = sum(
            max(1, math.ceil(share * min(part, minibatch - begin))) for begin in range(0, minibatch, part)
        )
        log.info(
            "ppo.value_share %g: the value side learns from %d of %d rows per learning step.",
            share,
            taken,
            minibatch,
        )

    def _cards_step_sizes(self) -> list[int] | None:
        """Every card's samples in each learning step of this update (several cards), read once."""
        if self._step_sizes is None and self._step_sizes_later is not None:
            self._step_sizes, self._step_sizes_later = self._step_sizes_later(), None
        return self._step_sizes

    def _filtered(self, advantages: torch.Tensor, samples: torch.Tensor | None) -> torch.Tensor:
        """ppo.advantage_keep's filter on a step's standardised advantages (`kept_advantages`), its
        kept share and size added to the update's sums; with several cards over every card's step;
        `samples`: a graphed step's padding (one card), left out, n and k worked out on the card."""
        keep = self.config.advantage_keep
        if samples is not None:
            assert self.cards is None, "several cards filter a graphed step outside its graphs"
            n, k = kept_counts(samples, keep)
            advantages, kept_size = kept_advantages(advantages, keep, samples=samples)
            kept_share = (k / n).to(kept_size.dtype)
        else:
            sizes = self._cards_step_sizes() if self.cards is not None else None
            count = sum(sizes) if sizes is not None else advantages.numel()
            advantages, kept_size = kept_advantages(advantages, keep, self.cards, sizes)
            # Filled on the card: new_tensor would copy it from the host, which no graph can hold.
            kept_share = torch.full(
                (), kept_count(count, keep) / count, dtype=kept_size.dtype, device=kept_size.device
            )
        self._kept += torch.stack([kept_share, kept_size])
        return advantages

    def _advantage_rows(self, advantages: torch.Tensor) -> _Rows:
        """ppo.advantage_keep_rest: the update's n advantages standardised together; advantage_keep's
        k largest in size (ties by place) kept, weight 1; of the other m, r = rest x m (at least 1) drawn at
        random, advantage 0, weight m / r. The weights add up to n, so a step's value, heads' and entropy
        means estimate those over every sample without bias; `_Rows.step` scales the kept advantages so its
        policy loss is the mean over its kept samples. Unlike advantage_keep's own steps, the standardising
        and the choice are over the whole update, not each step (the same at minibatches 1)."""
        n = advantages.numel()
        k = kept_count(n, self.config.advantage_keep)
        m = n - k
        r = min(m, max(1, round(self.config.advantage_keep_rest * m)))
        standardised = _standardised(advantages, None)
        size = standardised.abs()
        chosen = kept_rows(size, k)
        places = torch.arange(n, device=advantages.device)
        others = _compacted(places, ~chosen, m)  # no wait for the card: k and m are known
        drawn = others[torch.randperm(m, device=advantages.device)[:r]]
        weights = chosen.float().index_fill_(0, drawn, m / r if r else 0.0)
        share = torch.where(chosen, size, 0.0).sum() / size.sum().clamp_min(1e-12)
        return _Rows(
            picked=torch.cat([_compacted(places, chosen, k), drawn]),
            weights=weights,
            chosen=chosen,
            advantages=torch.where(chosen, standardised, 0.0),
            kept=torch.stack([torch.full_like(share, k / n), share]),
            share=(k + r) / n,
        )

    def _losses(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        targets: torch.Tensor,
        value_only: bool,
        samples: torch.Tensor | None = None,
        head_targets: dict[str, torch.Tensor] | None = None,
        entropy_coef: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A step's loss on these samples (advantages already standardised) and [policy loss, value loss,
        entropy, approx KL, clip fraction], each a mean over them (`samples`: over the weighted ones; see
        `_step`), then with `head_targets` each extra head's raw loss. With ppo.value_share
        below 1 a joint step's value side learns from its first rows only, and its losses are means
        over them."""
        outputs: dict[str, torch.Tensor] = {}
        value_weights = samples
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            if not value_only and self.config.value_share < 1:
                k = max(1, math.ceil(self.config.value_share * obs.shape[0]))
                features = self.model.shared(obs)
                logits = self.model.logits_from(features)
                if head_targets is not None:
                    outputs = self.model.head_values_from(features[:k])
                    predicted = outputs["main"]
                    head_targets = {name: t[:k] for name, t in head_targets.items()}
                else:
                    predicted = self.model.value_from(features[:k])
                targets = targets[:k]
                value_weights = self._value_rows(samples, k)
            elif head_targets is not None:  # every head, the trunks run once
                if value_only and self.policy_frozen and len(self.model.shared):
                    with torch.no_grad():  # held as `normalised_value(hold_trunk=True)` holds it
                        features = self.model.shared(obs)
                else:
                    features = self.model.shared(obs)
                outputs = self.model.head_values_from(features)
                predicted = outputs["main"]
                if not value_only:
                    logits = self.model.logits_from(features)
            elif value_only:
                # A frozen policy's warm-up holds a shared trunk too: it is the policy's as well.
                predicted = self.model.normalised_value(obs, hold_trunk=self.policy_frozen)
            else:
                logits, predicted = self.model(obs)
        value_loss = weighted_mean((predicted - targets).pow(2), value_weights)
        head_losses = self._head_losses(outputs, head_targets, value_weights)
        pulls = head_losses  # what the extra heads add to the loss; their measurements stay raw
        if head_losses and self.config.head_loss_scale == "popart":  # in the main value's units
            weight = self._head_loss_weight()
            pulls = [weight * h for h in head_losses]
        if value_only:
            loss = value_loss + sum(pulls, torch.zeros((), device=self.device))
            zero = torch.zeros((), device=self.device)
            measured = torch.stack([zero, value_loss.detach(), zero, zero, zero])
        else:
            log_probs = masked_log_probs(logits, mask)
            new_log_prob = log_probs.gather(1, actions[:, None]).squeeze(1)
            entropy = weighted_mean(
                normalized_entropy(log_probs, mask, self.config.entropy_normalization),
                samples,
            )
            policy = policy_loss(new_log_prob, old_log_probs, advantages, self.config.clip_range, samples)
            # A tensor bonus (graphed steps) is the float's FP32 value: the same product to the bit.
            bonus = self.entropy_coef if entropy_coef is None else entropy_coef
            loss = policy.loss - bonus * entropy + value_loss
            for head_loss in pulls:
                loss = loss + head_loss
            measured = torch.stack(
                [
                    policy.loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    policy.approx_kl,
                    policy.clip_fraction,
                ]
            )
        if head_targets is not None:  # the twins' loss as one twin's mean
            names = list(head_targets)
            reported = [
                h.detach() / (2 if n == "vdag" else 1) for n, h in zip(names, head_losses, strict=True)
            ]
            measured = torch.cat([measured, torch.stack(reported)])
        return loss, measured

    def _head_loss_weight(self) -> torch.Tensor:
        """ppo.head_loss_scale = 'popart': 1 / PopArt's variance, read from its statistics when called
        (a step's graph reads each update's: `PopArt.update` changes them in place)."""
        assert isinstance(self.value_norm, PopArt)
        return self.value_norm.std.detach().square().reciprocal()

    def _head_losses(
        self,
        outputs: dict[str, torch.Tensor],
        head_targets: dict[str, torch.Tensor] | None,
        samples: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Per entry of `head_targets`: the goal critic's squared error; the twins' expectile losses
        (both added, each twin to the same target)."""
        if head_targets is None:
            return []
        out = []
        for name, target in head_targets.items():
            if name == "goal":
                out.append(weighted_mean((target - outputs["goal"]).pow(2), samples))
            else:
                tau = self.config.vdag_expectile
                twins = [
                    weighted_mean(expectile_loss(target - outputs[n], tau), samples)
                    for n in ("vdag1", "vdag2")
                ]
                out.append(twins[0] + twins[1])
        return out

    def _value_rows(self, samples: torch.Tensor | None, k: int) -> torch.Tensor | None:
        """ppo.value_share's weights on the first `k` rows of a padded (graphed) step: its real samples come
        first, and the value side takes ceil(share x their count) of them, worked out on the card in FP64
        as the unpadded step's count on the host. None without padding."""
        if samples is None:
            return None
        count = torch.ceil(samples.sum(dtype=torch.float64) * self.config.value_share).clamp_min(1.0)
        inside = torch.arange(k, dtype=torch.float64, device=samples.device) < count
        return samples[:k] * inside.to(samples.dtype)

    def _driven(self, rows: torch.Tensor) -> torch.Tensor:
        """The positions of the agents `rows` marks: read from the graphics card once while `rows` stays the
        same tensor, unchanged (its version counts every change in place), as self-play's rows do between its
        draws of past versions."""
        known = self._driven_rows
        if known is None or known[0] is not rows or known[1] != rows._version:
            known = self._driven_rows = (rows, rows._version, rows.nonzero().squeeze(1))
        return known[2]

    def _apply_learning_rates(self) -> None:
        for (name, factor, _), group in zip(self.rate_groups, self.optimizer.param_groups, strict=True):
            group["lr"] = self.learning_rates[name] * factor

    def head_values(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every value head's normalised output for `obs` (`ActorCritic.head_values`), with gradients,
        under the precision the learning steps use (FP16 autocast in an FP16 run; the output layers FP32).
        For the losses of the heads beyond "main"; `ActorCritic.forward_heads` gives the logits with them."""
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            return self.model.head_values(obs)

    @torch.no_grad()
    def _log_probs(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            logits = self.model.logits(obs)  # the same logits as the full model's, without its value
        return masked_log_probs(logits, mask)
