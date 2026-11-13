"""Known-answer value check: in a game whose returns are known exactly, the value network must learn them."""

from __future__ import annotations

import torch

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.envs import GroupTensors
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.trainer import Trainer

LENGTH = 10  # every episode lasts exactly this many steps
GAMMA = 0.9
CPU = torch.device("cpu")


class CountdownEnvs:
    """Every step pays 1 and every episode ends (a goal) after LENGTH steps, whatever the actions.

    Agents start at different points of the episode, so every position is seen in every step. The
    observation is the step number within the episode, one-hot, so a state's value is exactly
    1 + gamma + ... + gamma^(remaining - 1).
    """

    def __init__(self, agents: int) -> None:
        self.groups = (GroupTensors.allocate(agents, LENGTH, 3, pin_memory=False),)
        self.t = torch.arange(agents) % LENGTH
        self._observe()

    def _observe(self) -> None:
        tensors = self.groups[0]
        tensors.obs.zero_()
        tensors.obs[torch.arange(len(self.t)), self.t] = 1.0
        tensors.mask.fill_(True)
        tensors.step.copy_(self.t.int())

    @property
    def num_groups(self) -> int:
        return 1

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        ended = self.t == LENGTH - 1
        tensors.reward.fill_(1.0)
        tensors.terminal.copy_(ended)
        tensors.truncated.zero_()
        tensors.episode.add_(ended.long())
        self.t = torch.where(ended, 0, self.t + 1)
        self._observe()

    def wait(self, group: int) -> None:
        pass


def test_the_value_network_learns_the_exact_discounted_return() -> None:
    torch.manual_seed(0)
    envs = CountdownEnvs(agents=40)
    model = ActorCritic(LENGTH, 3, ModelConfig(policy=(32,), value=(32,)))
    learner = Learner(
        model, PPOConfig(epochs=4, minibatches=4, value_lr=3e-3), gamma=GAMMA, device=CPU, fp16=False
    )
    trainer = Trainer(envs, model, learner, LENGTH * 40, CPU, fp16=False, cuda_graphs=False)
    for _ in range(150):
        trainer.iteration()

    remaining = LENGTH - torch.arange(LENGTH)
    expected = (1 - GAMMA**remaining) / (1 - GAMMA)
    with torch.no_grad():
        predicted = learner.value_norm.denormalize(model.normalised_value(torch.eye(LENGTH)))
    # Rewards are divided by the running return scale before learning, so values come out in those units.
    predicted = predicted * float(learner.reward_scaler.moments.std)
    torch.testing.assert_close(predicted, expected.float(), rtol=0.05, atol=0.05)
