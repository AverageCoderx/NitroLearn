# A reward of the project's own: the import lists it for porting by hand.
from rlgym_sim.utils.reward_functions import RewardFunction


class SpeedFlipReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return float(player.car_data.linear_velocity[1] > 2000)
