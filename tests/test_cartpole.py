from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from nitro import cartpole
from nitro.cartpole import CartPoleEnvs, cartpole_step, fallen


def test_physics_match_gymnasium() -> None:
    env = gymnasium.make("CartPole-v1")
    rng = np.random.default_rng(0)
    for episode in range(20):
        env.reset(seed=episode)
        state = torch.tensor(np.asarray(env.unwrapped.state, dtype=np.float64))[None]
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(rng.integers(2))
            _, reward, terminated, truncated, _ = env.step(action)
            state = cartpole_step(state, torch.tensor([action == 1]))
            np.testing.assert_allclose(state[0].numpy(), np.asarray(env.unwrapped.state), rtol=0, atol=1e-12)
            assert reward == 1.0
            assert bool(fallen(state)[0]) == terminated


def test_envs_reset_fallen_poles_and_truncate_at_the_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cartpole, "TIME_LIMIT", 5)
    envs = CartPoleEnvs(num_envs=3, num_groups=2, seed=0, pin_memory=False)
    assert [t.obs.shape[0] for t in envs.groups] == [2, 1]
    tensors = envs.groups[0]
    for step in range(5):
        tensors.actions.copy_(torch.tensor([step % 2, (step + 1) % 2], dtype=torch.int32))
        before = envs.state[0].clone()
        envs.step(0)
    # Alternating pushes keep the poles up, so both hit the 5-step limit on the last step.
    assert tensors.truncated.all() and not tensors.terminal.any()
    expected_final = cartpole_step(before, tensors.actions == 1).float()
    torch.testing.assert_close(tensors.final_obs, expected_final)
    assert (tensors.obs.abs() <= 0.05).all(), "a new episode starts after the limit"
    assert envs.take_episode_lengths() == [5, 5]
    assert envs.take_episode_lengths() == []
