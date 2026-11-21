# A small RLGym-PPO project for NitroLearn's import tests: written for the tests, in the style of
# RLGym-PPO's example (rlgym-sim environment, rlgym-tools LookupAction).
import numpy as np
from my_rewards import SpeedFlipReward
from rlgym_ppo import Learner
from rlgym_sim.utils import common_values
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    EventReward,
    FaceBallReward,
    LiuDistancePlayerToBallReward,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
)
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction


def build_env():
    import rlgym_sim

    tick_skip = 8
    timeout_steps = round(10 * 120 / tick_skip)
    rewards = CombinedReward(
        reward_functions=(
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(own_goal=False),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            EventReward(team_goal=1, concede=-1, touch=0.5, demo=0.3, boost_pickup=0.1, shot=0.2),
            SpeedFlipReward(),
        ),
        reward_weights=(0.01, 0.1, 0.02, 0.05, 10.0, 1.0),
    )
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )
    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,
        spawn_opponents=True,
        terminal_conditions=[NoTouchTimeoutCondition(timeout_steps), GoalScoredCondition()],
        reward_fn=rewards,
        obs_builder=obs_builder,
        action_parser=LookupAction(),
        state_setter=DefaultState(),
    )


if __name__ == "__main__":
    learner = Learner(
        build_env,
        n_proc=32,
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,
        ppo_minibatch_size=25_000,
        ppo_ent_coef=0.01,
        ppo_epochs=2,
        policy_layer_sizes=(32, 24),
        critic_layer_sizes=(24,),
        standardize_returns=True,
        standardize_obs=True,
        save_every_ts=1_000_000,
        policy_lr=2e-4,
        critic_lr=2e-4,
        gae_gamma=0.995,
        timestep_limit=10_000_000_000,
        log_to_wandb=False,
    )
    learner.learn()
