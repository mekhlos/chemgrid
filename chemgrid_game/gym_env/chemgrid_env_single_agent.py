from typing import Tuple
from typing import Union

import gym
from gym.core import ObsType

from chemgrid_game.gym_env import ChemGridEnv


class ChemGridSingleAgentEnv(gym.Wrapper):
    def __init__(self, env: ChemGridEnv):
        super().__init__(env)
        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

    def step(self, action) -> Tuple[ObsType, float, bool, dict]:
        states, rewards, dones, infos = super().step((action,))
        return states[0], rewards[0], dones[0], infos[0]

    def reset(self, **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        return super().reset(**kwargs)[0]

    def render(self, mode="human", **kwargs):
        if mode == "rgb_array":
            return self.get_observation()[0]
        elif mode == "human":
            self.game.render()
        else:
            super().render(mode=mode)  # raise an exception
