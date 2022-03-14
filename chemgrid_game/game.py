from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pygame

from chemgrid_game import utils
from chemgrid_game.game_backend import Action
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.game_frontend import GameFrontend
from chemgrid_game.game_helpers import GameState


class Game:
    def __init__(
            self,
            frontend: GameFrontend,
            backend: GameBackend,
            config: Config,
    ):
        self.config = config
        self.frontend = frontend
        self.backend = backend
        self.logger = utils.setup_logger(self.__class__.__name__, config.logging_level)
        self.game_states: Optional[List[GameState]] = None

    @property
    def shape(self):
        return self.frontend.screen.get_height(), self.frontend.screen.get_width()

    def _step_agent(self, agent_id: int, p1: int, p2: int) -> Action:
        self.frontend.update_game(self.game_states[agent_id])
        action = self.frontend.step((p2, p1))
        self.frontend.update_game(self.game_states[agent_id])
        return action

    def step(self, coords: Tuple[np.ndarray]):
        actions = [self._step_agent(i, p1, p2) for i, (p1, p2) in enumerate(coords)]
        new_states, rewards, dones, infos = self.backend.step(tuple(actions))
        for i, (new_state, done) in enumerate(zip(new_states, dones)):
            self.game_states[i].agent_state = new_state
            self.game_states[i].done = done

        return rewards, dones

    def _get_agent_click(self, agent_id: int) -> np.ndarray:
        self.frontend.update_game(self.game_states[agent_id])
        self.frontend.render()

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    self.frontend.close()
                    exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()[::-1]
                    return np.array(pos)

    def render(self):
        self.frontend.render()

    def get_clicks(self) -> Tuple[np.ndarray]:
        coords = []
        for agent_id in range(self.backend.n_agents):
            coords.append(self._get_agent_click(agent_id))

        return tuple(coords)

    def reset(self):
        states = self.backend.reset()
        if self.game_states is None:
            self.game_states = [
                GameState(i, self.backend.archive, state, self.config) for i, state in enumerate(states)
            ]

        for game_state, agent_state in zip(self.game_states, states):
            game_state.reset()
            game_state.agent_state = agent_state

    def _to_img_array(self):
        return self.frontend.to_img_array()

    def get_observation(self) -> Tuple[np.ndarray]:
        obs = []
        for game_state in self.game_states:
            self.frontend.update_game(game_state)
            img = self.frontend.to_img_array()
            obs.append(img)

        return tuple(obs)

    def close(self):
        pygame.quit()
