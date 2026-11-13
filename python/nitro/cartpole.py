"""CartPole-v1 behind the same interface as the Rocket League environments (correctness proof).

The physics, start states, termination and the 500-step time limit follow Gymnasium's CartPole-v1;
`tests/test_cartpole.py` checks that against Gymnasium itself. Pole falls are terminal, the time limit is
a truncation, as in Gymnasium.

The masked variant exercises action masking, which plain CartPole doesn't: 8 actions, three copies each
of "push left" (0-2) and "push right" (3-5) of which one random copy of each is allowed at every step,
and two actions (6, 7) that are never allowed, like Rocket League's air actions on the ground.
"""

from __future__ import annotations

import math

import torch

from nitro.envs import GroupTensors

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_HALF_LENGTH = 0.5
POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH
FORCE = 10.0
TAU = 0.02
ANGLE_LIMIT = 12 * 2 * math.pi / 360
POSITION_LIMIT = 2.4
TIME_LIMIT = 500
MASKED_ACTIONS = 8
_COPIES = 3


def cartpole_step(state: torch.Tensor, push_right: torch.Tensor) -> torch.Tensor:
    """One Euler step of Gymnasium's CartPole equations; `state` is [n, 4] (x, x_dot, theta, theta_dot)."""
    x, x_dot, theta, theta_dot = state.unbind(1)
    force = torch.where(push_right, FORCE, -FORCE)
    cos, sin = torch.cos(theta), torch.sin(theta)
    temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (
        POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos**2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos / TOTAL_MASS
    return torch.stack(
        [x + TAU * x_dot, x_dot + TAU * x_acc, theta + TAU * theta_dot, theta_dot + TAU * theta_acc], dim=1
    )


def fallen(state: torch.Tensor) -> torch.Tensor:
    return (state[:, 0].abs() > POSITION_LIMIT) | (state[:, 2].abs() > ANGLE_LIMIT)


class CartPoleEnvs:
    """`num_envs` CartPoles split into `num_groups` groups; stepping happens synchronously in `step`."""

    def __init__(
        self, num_envs: int, num_groups: int, seed: int, pin_memory: bool, masked: bool = False
    ) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.masked = masked
        sizes = [num_envs // num_groups + (g < num_envs % num_groups) for g in range(num_groups)]
        num_actions = MASKED_ACTIONS if masked else 2
        self.groups = tuple(GroupTensors.allocate(n, 4, num_actions, pin_memory=pin_memory) for n in sizes)
        self.state = [self._fresh(n) for n in sizes]
        self.elapsed = [torch.zeros(n, dtype=torch.int32) for n in sizes]
        self.finished_lengths: list[int] = []
        for group, tensors in enumerate(self.groups):
            self._write_mask(tensors)
            tensors.obs.copy_(self.state[group])

    def _write_mask(self, tensors: GroupTensors) -> None:
        if not self.masked:
            tensors.mask.fill_(True)
            return
        n = tensors.mask.shape[0]
        left = torch.randint(0, _COPIES, (n,), generator=self.generator)
        right = torch.randint(_COPIES, 2 * _COPIES, (n,), generator=self.generator)
        tensors.mask.fill_(False)
        rows = torch.arange(n)
        tensors.mask[rows, left] = True
        tensors.mask[rows, right] = True

    def _pushes_right(self, tensors: GroupTensors) -> torch.Tensor:
        actions = tensors.actions.long()
        if not tensors.mask.gather(1, actions[:, None]).all():
            raise ValueError("an action that was not allowed was taken")
        return actions >= _COPIES if self.masked else actions == 1

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def _fresh(self, n: int) -> torch.Tensor:
        return torch.rand(n, 4, generator=self.generator, dtype=torch.float64) * 0.1 - 0.05

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        state = cartpole_step(self.state[group], self._pushes_right(tensors))
        elapsed = self.elapsed[group] + 1
        terminal = fallen(state)
        truncated = ~terminal & (elapsed >= TIME_LIMIT)
        ended = terminal | truncated
        tensors.reward.fill_(1.0)
        tensors.terminal.copy_(terminal)
        tensors.truncated.copy_(truncated)
        tensors.final_obs[truncated] = state[truncated].float()
        tensors.episode.add_(ended.long())
        tensors.step.copy_(torch.where(ended, 0, tensors.step + 1))
        if ended.any():
            self.finished_lengths.extend(elapsed[ended].tolist())
            state[ended] = self._fresh(int(ended.sum()))
            elapsed[ended] = 0
        self.state[group] = state
        self.elapsed[group] = elapsed
        tensors.obs.copy_(state)
        self._write_mask(tensors)

    def wait(self, group: int) -> None:
        """Steps finish inside `step`; nothing to wait for."""

    def take_episode_lengths(self) -> list[int]:
        """Lengths (= returns, one reward per step) of the episodes finished since the last call."""
        lengths, self.finished_lengths = self.finished_lengths, []
        return lengths
