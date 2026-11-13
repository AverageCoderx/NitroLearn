"""PPO maths as small pure functions (unit-tested against hand-computed values in tests/test_ppo.py).

Everything here runs in FP32 regardless of the network's precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from nitro.multigpu import Cards


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """FP32 log-probabilities over the allowed actions; forbidden actions get -inf (probability 0).

    Callers read the result only at allowed actions (the sampled or taken one) or through
    `normalized_entropy`; any arithmetic on the -inf entries would turn gradients into NaN.
    One `where` rather than inverting the mask and filling a copy: the same numbers and gradients, one GPU
    job instead of three.
    """
    return torch.log_softmax(torch.where(mask, logits.float(), -math.inf), dim=-1)


def normalized_entropy(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    over: Literal["allowed", "all"] = "allowed",
) -> torch.Tensor:
    """Entropy of the allowed actions, divided by its maximum: 1 = uniform, 0 = certain.

    `over="allowed"` divides by log(number of allowed actions), so states with many forbidden actions
    don't look more certain; `over="all"` divides by log(number of actions, every column of the mask).
    Forbidden actions are replaced by an exact 0 before any arithmetic, so neither the value nor its
    gradient ever meets 0 · (-inf).
    """
    allowed_log_probs = log_probs.masked_fill(~mask, 0.0)
    probs = allowed_log_probs.exp().masked_fill(~mask, 0.0)
    entropy = -(probs * allowed_log_probs).sum(dim=-1)
    if over == "all":
        return entropy / math.log(mask.shape[-1])
    allowed = mask.sum(dim=-1)
    return torch.where(allowed > 1, entropy / allowed.clamp_min(2).float().log(), torch.zeros_like(entropy))


def masked_kl(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exact KL(p || q) per row over the allowed actions; forbidden actions contribute an exact 0."""
    log_p = log_p.masked_fill(~mask, 0.0)
    log_q = log_q.masked_fill(~mask, 0.0)
    return (log_p.exp().masked_fill(~mask, 0.0) * (log_p - log_q)).sum(dim=-1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """The mean, or with `weights` (1 for a sample, 0 for padding; the graphed steps) the mean over the
    weighted samples: the same maths, other rounding."""
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass(frozen=True)
class PolicyLossResult:
    loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


def policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    samples: torch.Tensor | None = None,
) -> PolicyLossResult:
    """PPO's clipped loss, the ratio taken against the policy that collected the data (`old_log_prob`).
    `samples`: per sample 1, or 0 for padding (`weighted_mean`).
    """
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -weighted_mean(torch.minimum(unclipped, clipped), samples)
    with torch.no_grad():
        clip_fraction = weighted_mean(((ratio - 1.0).abs() > clip_range).float(), samples)
        # Low-variance KL(old || new) estimator: E[(r - 1) - log r].
        approx_kl = weighted_mean((ratio - 1.0) - log_ratio, samples)
    return PolicyLossResult(loss, clip_fraction, approx_kl)


def discount_from_half_life(half_life_seconds: float, tick_skip: int, tick_rate: float = 120.0) -> float:
    """Discount factor whose rewards lose half their weight after `half_life_seconds` (the formula of Kuenec's
    guide)."""
    return float(0.5 ** (tick_skip / (tick_rate * half_life_seconds)))


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminal: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over [steps, agents] tensors.

    `next_values[t]` is the value of the state step t reached: the next step's value normally, the value
    of the saved final observation when step t was truncated, and the bootstrap value after the last step.
    A terminal step (a goal) is worth nothing afterwards; any episode end stops the advantage recursion.
    Returns (advantages, value targets).
    """
    not_terminal = (~terminal).float()
    # gamma * lam * continues[t] * running per step, its first product made once and each step's
    # sum written straight into its row: two small launches a step instead of four, the same bits.
    carry = gamma * lam * (~(terminal | truncated)).float()
    deltas = rewards + gamma * next_values * not_terminal - values
    advantages = torch.empty_like(deltas)
    running = torch.zeros_like(deltas[0])
    for t in range(deltas.shape[0] - 1, -1, -1):
        torch.add(deltas[t], carry[t] * running, out=advantages[t])
        running = advantages[t]
    return advantages, advantages + values


def discounted_returns(
    rewards: torch.Tensor, episode_end: torch.Tensor, running: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each step's discounted return so far, carried on from `running` and restarted after an episode's end
    ([steps, agents]); returns them and the running return after the last step. Two small launches a step."""
    returns = torch.empty_like(rewards)
    for t in range(rewards.shape[0]):
        torch.add(running * gamma, rewards[t], out=returns[t])  # written in place
        running = returns[t].masked_fill(episode_end[t], 0.0)
    return returns, running


def expectile_loss(error: torch.Tensor, tau: float) -> torch.Tensor:
    """|tau - 1[u < 0]| u² per sample, u = target - prediction: above 0.5 a prediction too low costs more
    than one too high, so the head learns an upper expectile of its targets."""
    return torch.where(error < 0, 1.0 - tau, tau) * error.pow(2)


def twin_target(
    rewards: torch.Tensor,
    reached_1: torch.Tensor,
    reached_2: torch.Tensor,
    terminal: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """The twin critics' target r + gamma (1 - d) min(V1(s'), V2(s')), with `reached_*` the twins'
    values of the state each step reached; d is any episode end, a cut too, as in the design: the rollout's
    own reward is used, so a cut pays no gamma V(final state) into it."""
    ended = terminal | truncated
    return rewards + gamma * (~ended).float() * torch.minimum(reached_1, reached_2)


def potential_shaping(
    potential: torch.Tensor,
    reached: torch.Tensor,
    terminal: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """gamma (1 - d) Phi(s') - Phi(s) per step (Ng et al. 1999), d any episode end."""
    ended = terminal | truncated
    return gamma * (~ended).float() * reached - potential


@dataclass(frozen=True)
class Shaping:
    added: torch.Tensor  # what joins the advantages
    clamped: torch.Tensor  # share of samples clamped


def mean_of(values: torch.Tensor, cards: Cards | None = None) -> torch.Tensor:
    """The mean; with `cards`, over every card's values (FP64 sums)."""
    if cards is None:
        return values.mean()
    flat = values.detach().double().flatten()
    # The count filled on the card: new_tensor would copy it from the host and wait for the card.
    count = torch.full((), float(flat.numel()), dtype=torch.float64, device=flat.device)
    totals = cards.sum_(torch.stack([count, flat.sum()]))
    return (totals[1] / totals[0].clamp_min(1)).to(values.dtype)


def variance_of(values: torch.Tensor, cards: Cards | None = None) -> torch.Tensor:
    """The (unbiased) variance; with `cards`, over every card's values (FP64 sums, as
    `Cards.standardised`)."""
    if cards is None:
        return values.var()
    count, _, m2 = cards.moments(values)
    return (m2 / (count - 1).clamp_min(1)).to(values.dtype)


def spread_of(values: torch.Tensor, cards: Cards | None = None) -> torch.Tensor:
    """The (unbiased) standard deviation; with `cards`, over every card's values."""
    if cards is None:
        return values.std()
    count, _, m2 = cards.moments(values)
    return (m2 / (count - 1).clamp_min(1)).sqrt().to(values.dtype)


def scaled_shaping(
    shaping: torch.Tensor,
    spread: torch.Tensor,
    beta: float,
    limit: float = 3.0,
    cards: Cards | None = None,
) -> Shaping:
    """`shaping` centred, divided by its spread (at least 0.05 x `spread`, so a nearly flat one is not blown
    up), clamped to +-`limit` and multiplied by beta x `spread` (the advantages' spread); with
    `cards`, every statistic over all cards' samples."""
    centred = shaping - mean_of(shaping, cards)
    unit = centred / torch.maximum(spread_of(centred, cards), 0.05 * spread).clamp_min(1e-8)
    clamped = mean_of((unit.abs() > limit).float(), cards)
    return Shaping(beta * spread * unit.clamp(-limit, limit), clamped)


def kept_count(n: int, keep: float) -> int:
    """How many of n samples `kept_advantages` keeps: ceil(keep x n), at least 1."""
    return min(n, max(1, math.ceil(keep * n)))


def kept_counts(samples: torch.Tensor, keep: float) -> tuple[torch.Tensor, torch.Tensor]:
    """n and `kept_count`'s k for a graphed step whose `samples` are 1, or 0 for padding, as FP64
    numbers on the card: the same double arithmetic as `kept_count`'s, without waiting for the card."""
    n = (samples > 0).sum(dtype=torch.float64)
    return n, torch.minimum(torch.ceil(n * keep).clamp_min(1.0), n)


def kept_rows(size: torch.Tensor, k: int) -> torch.Tensor:
    """Flags on the `k` largest of `size`: above the k-th largest, then ties at it by place, so the choice
    does not depend on what the top-k kernel does with ties."""
    threshold = size.topk(k).values[-1]
    above = size > threshold
    tied = size == threshold
    return above | (tied & (tied.cumsum(0) <= k - above.sum()))


def kept_advantages(
    advantages: torch.Tensor,
    keep: float,
    cards: Cards | None = None,
    sizes: list[int] | None = None,
    samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `kept_count` advantages largest in size, times n / k, and 0 for the rest: a 0 advantage
    gives the clipped loss no gradient, so its mean over all n samples is the mean over the k kept. Also
    the share of the advantages' total size the kept ones hold. With `cards` the n samples are every
    card's step (card r holds `sizes[r]` of them): each card picks its own out of the same global choice.
    `samples` (one card; 1, or 0 for a graphed step's padding): the padding is never kept and n is the
    other samples' count, k and n / k worked out on the card (`kept_counts`); the same choice as the step
    without the padding unless two |A| tie at the k-th largest (here ties go by place)."""
    if cards is None and samples is None:
        n = advantages.numel()
        k = kept_count(n, keep)
        size = advantages.abs()
        # index_fill_: an index put of a Python True would copy it from the host (no graph can hold that).
        chosen = torch.zeros_like(advantages, dtype=torch.bool).index_fill_(0, size.topk(k).indices, True)
        share = torch.where(chosen, size, 0.0).sum() / size.sum().clamp_min(1e-12)
        return torch.where(chosen, advantages * (n / k), 0.0), share
    if cards is None:
        assert samples is not None
        real = samples > 0
        n_real, k_real = kept_counts(samples, keep)
        size = torch.where(real, advantages.abs(), -1.0)  # the padding below every real |A|
        # The k-th largest by a sort (k lives on the card, topk wants it on the host), then as below.
        threshold = size.sort(descending=True).values.gather(0, k_real.long().reshape(1) - 1)
        above = size > threshold
        tied = size == threshold
        chosen = above | (tied & (tied.cumsum(0) <= k_real - above.sum()))
        share = torch.where(chosen, size, 0.0).sum() / torch.where(real, size, 0.0).sum().clamp_min(1e-12)
        return torch.where(chosen, advantages * (n_real / k_real).to(advantages.dtype), 0.0), share
    assert sizes is not None and sizes[cards.rank] == advantages.numel()
    every = cards.gathered(advantages.detach().abs(), sizes)
    n = every.numel()
    k = kept_count(n, keep)
    chosen_all = kept_rows(every, k)  # ties by place: every card makes the same choice
    share = torch.where(chosen_all, every, 0.0).sum() / every.sum().clamp_min(1e-12)
    start = sum(sizes[: cards.rank])
    chosen = chosen_all[start : start + advantages.numel()]
    return torch.where(chosen, advantages * (n / k), 0.0), share


class RunningMoments:
    """Running mean and variance over every value seen (parallel Welford update, FP64)."""

    def __init__(self, device: torch.device | str) -> None:
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.mean = torch.zeros((), dtype=torch.float64, device=device)
        self.m2 = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, values: torch.Tensor, cards: Cards | None = None) -> None:
        """Adds `values`; with `cards`, every card's values together."""
        n: torch.Tensor | int
        if cards is not None:
            n, batch_mean, batch_m2 = cards.moments(values)
        else:
            values = values.detach().double().flatten()
            n = values.numel()
            batch_mean = values.mean()
            batch_m2 = ((values - batch_mean) ** 2).sum()
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.m2 += batch_m2 + delta**2 * self.count * n / total
        self.count = total

    @property
    def std(self) -> torch.Tensor:
        return (self.m2 / self.count.clamp_min(1)).sqrt()


class RewardScaler:
    """Divides rewards by the running standard deviation of the agents' discounted returns, then clips.

    The scale comes from discounted returns. Each agent's return is accumulated forwards
    and carried over from one iteration to the next (like Gym's NormalizeReward), so the statistics are
    the same however many steps a rollout holds, and therefore whatever the number of arenas. An
    iteration is scaled with the statistics of the returns seen before it. Each scaled reward is then clipped
    to plus/minus `clip` (0: not clipped).
    """

    def __init__(self, gamma: float, clip: float, device: torch.device | str) -> None:
        self.gamma = gamma
        self.clip = clip
        self.moments = RunningMoments(device)
        self.running: torch.Tensor | None = None

    def __call__(
        self,
        rewards: torch.Tensor,
        episode_end: torch.Tensor,
        rows: torch.Tensor | None = None,
        cards: Cards | None = None,
    ) -> torch.Tensor:
        """Scaled rewards for every agent; with `rows` (bool per agent, or the agents' positions, which the
        graphics card picks without the host waiting) the statistics learn from those agents only; with
        `cards`, from every card's agents."""
        running = self.running if self.running is not None else torch.zeros_like(rewards[0])
        scaled, returns, self.running = self.plain(rewards, episode_end, running)
        self.moments.update(returns if rows is None else returns[:, rows], cards)
        return scaled

    def plain(
        self, rewards: torch.Tensor, episode_end: torch.Tensor, running: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards scaled by the statistics so far and clipped, each step's return carried on from
        `running`, and the running return after the last step."""
        seen = self.moments.count > 0
        std = torch.where(seen, self.moments.std, torch.ones_like(self.moments.std)).float()
        scaled = rewards / std.clamp_min(1e-8)
        if self.clip > 0:
            scaled = scaled.clamp(-self.clip, self.clip)
        each, after = discounted_returns(rewards, episode_end, running, self.gamma)
        return scaled, each, after

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The return statistics; the running returns are not kept, since a checkpoint ends every
        episode."""
        return {"count": self.moments.count, "mean": self.moments.mean, "m2": self.moments.m2}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("count", "mean", "m2"):
            getattr(self.moments, name).copy_(state[name])
        self.running = None


class PopArt:
    """Adaptive value-target normalisation that preserves the value network's outputs (van Hasselt 2016).

    The value head predicts normalised values; `update` moves the running statistics towards the new
    targets and rescales the head's last linear layer so its denormalised outputs are unchanged. Adam's
    moment estimates for that layer are not rescaled with it (the paper used SGD).
    """

    # Weight of each iteration's targets in the running statistics (roughly a 50-iteration memory).
    BETA = 0.02

    def __init__(self, layer: torch.nn.Linear) -> None:
        self.layer = layer
        device = layer.weight.device
        self.mean = torch.zeros((), device=device)
        self.mean_sq = torch.ones((), device=device)
        self.initialised = False

    @property
    def std(self) -> torch.Tensor:
        return (self.mean_sq - self.mean**2).clamp_min(1e-8).sqrt()

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "mean_sq": self.mean_sq, "initialised": torch.tensor(self.initialised)}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"])
        self.mean_sq.copy_(state["mean_sq"])
        self.initialised = bool(state["initialised"])

    @torch.no_grad()
    def update(self, targets: torch.Tensor, cards: Cards | None = None) -> None:
        """Moves toward `targets`; with `cards`, every card's targets together."""
        targets = targets.detach().float()
        if cards is not None:
            wide = targets.double()
            # The count filled on the card: new_tensor would copy it from the host and wait for the card.
            count = torch.full((), float(wide.numel()), dtype=torch.float64, device=wide.device)
            sums = cards.sum_(torch.stack([count, wide.sum(), (wide**2).sum()]))
            batch_mean, batch_mean_sq = (sums[1] / sums[0]).float(), (sums[2] / sums[0]).float()
        else:
            batch_mean, batch_mean_sq = targets.mean(), (targets**2).mean()
        old_mean, old_std = self.mean.clone(), self.std
        beta = 1.0 if not self.initialised else self.BETA
        # Non-finite targets (that iteration's update is skipped) leave the statistics and the layer as
        # they are; decided on the GPU, without waiting for it.
        finite = batch_mean.isfinite() & batch_mean_sq.isfinite()
        self.mean.copy_(torch.where(finite, self.mean.lerp(batch_mean, beta), self.mean))
        self.mean_sq.copy_(torch.where(finite, self.mean_sq.lerp(batch_mean_sq, beta), self.mean_sq))
        self.initialised = True
        new_std = self.std
        self.layer.weight.mul_(old_std / new_std)
        self.layer.bias.mul_(old_std).add_(old_mean - self.mean).div_(new_std)
