from __future__ import annotations

import math

import pytest
import torch

from nitro.ppo import (
    PopArt,
    RewardScaler,
    RunningMoments,
    discount_from_half_life,
    gae,
    masked_kl,
    masked_log_probs,
    normalized_entropy,
    policy_loss,
)


def column(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)[:, None]


def flags(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)[:, None]


def test_gae_handles_terminals_truncations_and_bootstrap_by_hand() -> None:
    # Step 1 ends in a goal (terminal), step 3 is cut short and bootstraps from its final state (value 10).
    rewards = column(1, 1, 1, 1)
    values = column(1, 2, 3, 4)
    next_values = column(2, 3, 4, 10)
    advantages, targets = gae(
        rewards, values, next_values, flags(0, 1, 0, 0), flags(0, 0, 0, 1), gamma=0.9, lam=0.8
    )
    # deltas: 1+0.9*2-1=1.8, 1-2=-1, 1+0.9*4-3=1.6, 1+0.9*10-4=6; recursion weight 0.72, cut at ends.
    expected = column(1.8 + 0.72 * -1, -1, 1.6 + 0.72 * 6, 6)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(targets, expected + values)


def test_gae_bootstraps_at_the_end_of_the_rollout() -> None:
    advantages, _ = gae(column(0), column(0), column(5), flags(0), flags(0), gamma=0.5, lam=1.0)
    torch.testing.assert_close(advantages, column(2.5))


def test_clipped_policy_loss_kl_and_clip_fraction_by_hand() -> None:
    new = torch.full((2,), math.log(1.5))
    old = torch.zeros(2)
    result = policy_loss(new, old, torch.tensor([1.0, -1.0]), clip_range=0.2)
    # Positive advantage is clipped at 1.2; negative keeps the worse unclipped -1.5.
    torch.testing.assert_close(result.loss, torch.tensor(-(1.2 - 1.5) / 2))
    torch.testing.assert_close(result.clip_fraction, torch.tensor(1.0))
    torch.testing.assert_close(result.approx_kl, torch.tensor(0.5 - math.log(1.5)))


def test_entropy_is_normalised_by_the_allowed_actions() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [math.log(3.0), 0.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, False, False], [False, True, False, False]])
    log_probs = masked_log_probs(logits, mask)
    assert torch.isinf(log_probs[~mask]).all() and (log_probs[~mask] < 0).all()
    entropy = normalized_entropy(log_probs, mask)
    p = torch.tensor([0.75, 0.25])
    torch.testing.assert_close(
        entropy, torch.stack([torch.tensor(1.0), -(p * p.log()).sum() / math.log(2), torch.tensor(0.0)])
    )


@pytest.mark.parametrize("over", ["allowed", "all"])
def test_gradients_stay_finite_with_forbidden_actions(over: str) -> None:
    # Regression test: entropy used to multiply 0 by -inf at forbidden actions, which made every
    # gradient NaN while the loss itself looked fine.
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(32, 90, generator=generator).requires_grad_()
    mask = torch.rand(32, 90, generator=generator) < 0.3
    mask[:, 0] = True
    taken = mask.float().argmax(dim=1)
    log_probs = masked_log_probs(logits, mask)
    entropy = normalized_entropy(log_probs, mask, over)  # type: ignore[arg-type]
    loss = -entropy.mean() - log_probs.gather(1, taken[:, None]).mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[~mask] == 0).all(), "forbidden actions must get no gradient"


def test_masked_entropy_and_its_gradient_equal_those_of_the_allowed_actions_alone() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    ours = normalized_entropy(masked_log_probs(logits, mask), mask)
    (ours_grad,) = torch.autograd.grad(ours.sum(), logits)
    probs = logits[:, [0, 2]].softmax(dim=-1)
    reference = -(probs * probs.log()).sum(dim=-1) / math.log(2)
    (reference_grad,) = torch.autograd.grad(reference.sum(), logits)
    torch.testing.assert_close(ours, reference)
    torch.testing.assert_close(ours_grad, reference_grad)
    # "all" divides by log(number of actions) instead.
    all_actions = normalized_entropy(masked_log_probs(logits, mask), mask, "all")
    torch.testing.assert_close(all_actions, reference * math.log(2) / math.log(4))


def test_masked_kl_equals_the_kl_of_the_allowed_actions_alone() -> None:
    before = torch.tensor([[2.0, -1.0, 0.5, 3.0], [0.0, 1.0, 2.0, 3.0]])
    after = torch.tensor([[1.0, 4.0, 1.5, -2.0], [0.0, 1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True, False], [True, True, False, True]])
    kl = masked_kl(masked_log_probs(before, mask), masked_log_probs(after, mask), mask)
    p, q = before[0, [0, 2]].softmax(-1), after[0, [0, 2]].softmax(-1)
    torch.testing.assert_close(kl, torch.stack([(p * (p / q).log()).sum(), torch.tensor(0.0)]))
    assert kl[1] == 0, "an unchanged policy must give exactly 0"


def test_running_moments_match_the_whole_data() -> None:
    data = torch.randn(1000, generator=torch.Generator().manual_seed(0)) * 3 + 2
    moments = RunningMoments("cpu")
    for chunk in data.split(137):
        moments.update(chunk)
    torch.testing.assert_close(moments.mean.float(), data.mean())
    torch.testing.assert_close(moments.std.float(), data.std(unbiased=False))


def test_reward_scaler_divides_by_the_previous_return_std_and_clips() -> None:
    scaler = RewardScaler(gamma=0.5, clip=3.0, device="cpu")
    rewards, ends = column(1, 1), flags(0, 0)
    torch.testing.assert_close(scaler(rewards, ends), rewards)  # no statistics yet
    # Running returns of the first call were 1 and 1 + 0.5 = 1.5: std 0.25, so rewards scale by 4,
    # then clip at 3.
    torch.testing.assert_close(scaler(rewards, ends), column(3, 3))


def test_reward_scale_does_not_depend_on_rollout_length() -> None:
    generator = torch.Generator().manual_seed(0)
    rewards = torch.randn(24, 5, generator=generator)
    ends = torch.rand(24, 5, generator=generator) < 0.1
    whole, chunked = RewardScaler(0.9, 0.0, "cpu"), RewardScaler(0.9, 0.0, "cpu")
    whole(rewards, ends)
    for start in range(0, 24, 4):
        chunked(rewards[start : start + 4], ends[start : start + 4])
    torch.testing.assert_close(chunked.moments.mean, whole.moments.mean)
    torch.testing.assert_close(chunked.moments.std, whole.moments.std)


def test_popart_update_preserves_denormalised_outputs() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(5, 1)
    popart = PopArt(layer)
    x = torch.randn(64, 5)
    for targets in (torch.randn(500) * 40 + 7, torch.randn(500) * 3 - 20):
        before = popart.denormalize(layer(x)).detach()
        popart.update(targets)
        torch.testing.assert_close(popart.denormalize(layer(x)).detach(), before, rtol=1e-4, atol=1e-3)
    assert float(popart.std) > 0


def test_discount_from_half_life_matches_gamma_0_99_at_tick_skip_8() -> None:
    assert discount_from_half_life(4.6, tick_skip=8) == pytest.approx(0.99, abs=1e-4)
    assert discount_from_half_life(1.0, tick_skip=120) == pytest.approx(0.5)


def test_gae_and_the_reward_scaler_give_the_bits_of_their_step_by_step_form() -> None:
    """The loops write each step's sum in place with fewer launches; the results are bit-identical to
    the plain recursions they replace."""
    generator = torch.Generator().manual_seed(7)
    steps, agents, gamma, lam = 37, 64, 0.993, 0.95
    rewards = torch.randn(steps, agents, generator=generator) * 3
    values = torch.randn(steps, agents, generator=generator)
    next_values = torch.randn(steps, agents, generator=generator)
    terminal = torch.rand(steps, agents, generator=generator) < 0.05
    truncated = torch.rand(steps, agents, generator=generator) < 0.05
    advantages, targets = gae(rewards, values, next_values, terminal, truncated, gamma, lam)
    deltas = rewards + gamma * next_values * (~terminal).float() - values
    continues = (~(terminal | truncated)).float()
    expected = torch.empty_like(deltas)
    running = torch.zeros_like(deltas[0])
    for t in range(steps - 1, -1, -1):
        running = deltas[t] + gamma * lam * continues[t] * running
        expected[t] = running
    assert torch.equal(advantages, expected) and torch.equal(targets, expected + values)

    scaler = RewardScaler(gamma, 0.0, "cpu")
    end = terminal | truncated
    for _ in range(2):  # the running returns carry over to the next call
        before = scaler.running if scaler.running is not None else torch.zeros(agents)
        scaler(rewards, end)
        running, returns = before.clone(), torch.empty_like(rewards)
        for t in range(steps):
            running = running * gamma + rewards[t]
            returns[t] = running
            running = running.masked_fill(end[t], 0.0)
        assert scaler.running is not None and torch.equal(scaler.running, running)
