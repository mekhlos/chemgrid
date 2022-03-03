from typing import Tuple

import gym
import numpy as np
from chemgrid_game.game import Game
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.game_frontend import GameFrontend
from chemgrid_game.utils import setup_logger


class ChemGridEnv(gym.Env):
    def __init__(self, continuous_actions: bool = False, **kwargs):
        config = Config(**kwargs)
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.n_agents = config.n_agents
        self.continuous_actions = continuous_actions
        frontend = GameFrontend(config)
        backend = GameBackend(
            config.initial_inventories,
            config.survival_mols,
            config.initial_contracts,
            logging_level=config.logging_level
        )
        self.game = Game(frontend, backend, config)
        self.logger = setup_logger(self.__class__.__name__, config.logging_level)

        self.height, self.width = self.game.shape
        shape = self.height, self.width, 3
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        if continuous_actions:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
        else:
            self.action_space = gym.spaces.Discrete(self.height * self.width)

        self.observation_space = gym.spaces.Tuple([self.observation_space] * self.n_agents)
        self.action_space = gym.spaces.Tuple([self.action_space] * self.n_agents)
        self.dones = [False] * self.n_agents

    def step(self, action: Tuple[np.ndarray]):
        action = np.array(action)
        if all(self.dones):
            raise RuntimeError("Reset env before step")

        infos = [{}] * self.n_agents
        if self.continuous_actions:
            click_coords = (action * [[self.height, self.width]])
        else:
            click_coords = np.stack(np.unravel_index(action, (self.height, self.width))).T

        self.logger.debug("Sending actions: %s", click_coords)
        self.dones = self.game.step(tuple(click_coords))
        rewards = [0] * self.n_agents

        return self.get_observation(), tuple(rewards), tuple(self.dones), tuple(infos)

    def reset(self, **kwargs):
        self.game.reset()
        self.logger.debug("reset")
        self.done = False
        return self.get_observation()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.get_observation()
        elif mode == "human":
            self.game.render()
        else:
            super().render(mode=mode)  # raise an exception

    def get_observation(self) -> np.ndarray:
        return np.stack(self.game.get_observation())

    def close(self):
        self.game.close()
