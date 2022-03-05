from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from chemgrid_game import utils
from chemgrid_game.game import Game
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.game_frontend import GameFrontend


def save_img(obs, path):
    im = Image.fromarray(obs)
    im.save(path)


class GameWithFrameSave(Game):
    def __init__(self, frontend: GameFrontend, backend: GameBackend, config: Config, frames_dir):
        super().__init__(frontend, backend, config)
        self.step_counter = 0
        run = utils.get_datetime_str()
        self.frames_dir = Path(frames_dir).joinpath(run)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def step(self, coords: Tuple[np.ndarray]):
        res = super().step(coords)
        self.step_counter += 1
        self.save_frames()
        return res

    def reset(self):
        res = super().reset()
        self.step_counter = 0
        self.save_frames()
        return res

    def save_frames(self):
        for agent_id, obs in enumerate(self.get_observation()):
            path = self.frames_dir.joinpath(f"agent{agent_id}_step{self.step_counter}.png")
            save_img(obs, path)
