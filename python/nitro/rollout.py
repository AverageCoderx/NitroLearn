"""One iteration of experience, stored on the learning device as [steps, agents] tensors.

Row `n` always belongs to the same car (the environments guarantee it), so each column is that car's
trajectory in time order; episode ends are marked by `terminal`/`truncated`, never spliced over.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rollout:
    obs: torch.Tensor  # [T, N, obs] float32
    mask: torch.Tensor  # [T, N, actions] bool
    actions: torch.Tensor  # [T, N] int64
    log_probs: torch.Tensor  # [T, N] float32, of the behaviour policy
    values: torch.Tensor  # [T, N] float32, normalised (see nitro.ppo.PopArt)
    rewards: torch.Tensor  # [T, N] float32, as the environment paid them
    terminal: torch.Tensor  # [T, N] bool: step t ended the episode for real (a goal)
    truncated: torch.Tensor  # [T, N] bool: step t cut the episode short
    truncated_values: torch.Tensor  # [T, N] float32, normalised value of the state a truncated step reached
    bootstrap_values: torch.Tensor  # [N] float32, normalised value of the state after the last step
    episode: torch.Tensor  # [T, N] int64: episode the observation at step t belongs to
    episode_step: torch.Tensor  # [T, N] int32: its step number within that episode

    @classmethod
    def allocate(
        cls,
        steps: int,
        agents: int,
        obs_size: int,
        num_actions: int,
        device: torch.device,
    ) -> Rollout:
        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            obs=zeros(steps, agents, obs_size),
            mask=zeros(steps, agents, num_actions, dtype=torch.bool),
            actions=zeros(steps, agents, dtype=torch.int64),
            log_probs=zeros(steps, agents),
            values=zeros(steps, agents),
            rewards=zeros(steps, agents),
            terminal=zeros(steps, agents, dtype=torch.bool),
            truncated=zeros(steps, agents, dtype=torch.bool),
            truncated_values=zeros(steps, agents),
            bootstrap_values=zeros(agents),
            episode=zeros(steps, agents, dtype=torch.int64),
            episode_step=zeros(steps, agents, dtype=torch.int32),
        )

    @property
    def steps(self) -> int:
        return self.rewards.shape[0]

    @property
    def agents(self) -> int:
        return self.rewards.shape[1]

    def next_values(self) -> torch.Tensor:
        """Normalised value of the state each step reached (see `nitro.ppo.gae`)."""
        following = torch.cat([self.values[1:], self.bootstrap_values[None]], dim=0)
        return torch.where(self.truncated, self.truncated_values, following)
