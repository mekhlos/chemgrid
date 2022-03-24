from typing import Optional
from typing import Tuple

import gym
import numpy as np

from chemgrid_game.chemistry.mol_chemistry import Action
from chemgrid_game.chemistry.molecule import Bond
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.plotting import plot_atoms_list
from chemgrid_game.utils import setup_logger


class ChemGridBackendEnv(gym.Env):
    def __init__(self, max_inv_size: int = 100, **kwargs):
        config = Config(**kwargs)
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self._config = config
        self._backend = GameBackend(
            inventories=config.initial_inventories,
            targets=config.initial_targets,
            contracts=config.initial_contracts,
            inventory_generators=config.inventory_generators,
            target_generators=config.target_generators,
            logging_level=config.logging_level
        )
        self.logger = setup_logger(self.__class__.__name__, config.logging_level)
        self.max_inv_size = max_inv_size
        n_join_options = (config.grid_size * 2 + 1) ** 2
        n_break_options = config.grid_size * config.grid_size * 2
        shape = [max_inv_size + 3, config.grid_size, config.grid_size]
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=shape, dtype=np.uint8)
        nvec = [max_inv_size, n_break_options, n_join_options]
        self.action_space = gym.spaces.MultiDiscrete(nvec=nvec, dtype=np.uint8)
        self._done = False
        self._selected_mol_pos1 = 0
        self._selected_mol_pos2 = 0
        self._state = None

    def _get_state(self, states) -> np.ndarray:
        inventory_hashes, target_hash, _ = states[0]
        inventory = [self._backend.archive[i].atoms for i in inventory_hashes]
        target = self._backend.archive[target_hash].atoms
        if self._selected_mol_pos1 != 0:
            selected_mol1 = inventory[self._selected_mol_pos1 - 1]
        else:
            selected_mol1 = np.zeros_like(target)

        if self._selected_mol_pos2 != 0:
            selected_mol2 = inventory[self._selected_mol_pos2 - 1]
        else:
            selected_mol2 = np.zeros_like(target)

        state = np.concatenate([np.stack([target, selected_mol1, selected_mol2]), inventory])
        # diff = max(0, self.max_inv_size + 3 - len(state))
        # pad = np.zeros(diff, *state.shape[1:])
        # state = np.concatenate([state, pad])
        return state

    def _break_offset_to_edge(self, offset: int) -> Optional[Bond]:
        n = self._config.grid_size

        id1 = offset // 2
        if offset % 2 == 0:
            id2 = id1 + 1
        else:
            id2 = id1 + n

        if id2 < n:
            x1, y1 = np.unravel_index(id1, (n, n))
            x2, y2 = np.unravel_index(id2, (n, n))

            return frozenset(((x1, y1), (x2, y2)))

    def _join_offset_to_edge(self, offset: int) -> Tuple[int, int]:
        d = self._config.grid_size
        n = d * 2 + 1
        x, y = np.unravel_index(offset, (n, n))
        return x - d, y - d

    def step(self, action: np.ndarray):
        if self._done:
            raise RuntimeError("Reset env before step")
        inventory = self._backend.inventories[0]

        if self._selected_mol_pos1 == 0:
            if action[0] <= len(inventory):
                self._selected_mol_pos1 = action[0]
            action = Action()
        else:
            mol_hash = self._backend.inventories[0][self._selected_mol_pos1 - 1]
            if self._selected_mol_pos2 == 0:
                if action[0] == 0:
                    offset = action[1]
                    edge = self._break_offset_to_edge(offset)
                    if edge is not None:
                        action = Action("break", (mol_hash,), edge)
                    else:
                        action = Action()
                    self._selected_mol_pos1 = 0

                else:
                    if action[0] <= len(inventory):
                        self._selected_mol_pos2 = action[0]
                    action = Action()
            else:
                mol_hash2 = self._backend.inventories[0][self._selected_mol_pos2 - 1]
                offset = action[2]
                offset = self._join_offset_to_edge(offset)
                action = Action("join", (mol_hash, mol_hash2), offset)
                self._selected_mol_pos1 = 0
                self._selected_mol_pos2 = 0

        self.logger.debug("Sending action: %s", str(action))
        states, rewards, dones, infos = self._backend.step((action,))
        self._done = dones[0]

        self._state = self._get_state(states)

        return self._state, rewards[0], self._done, infos[0]

    def reset(self, **kwargs) -> np.ndarray:
        states = self._backend.reset()
        self.logger.debug("reset")
        self._done = False
        self._selected_mol_pos1 = 0
        self._selected_mol_pos2 = 0
        self._state = self._get_state(states)
        return self._state

    def render(self, mode="human"):
        if mode == "rgb_array":
            n, h, w = self._state.shape
            img = self._state.reshape((n * h, w))
            rgb_img = np.stack([img == i for i in [1, 2, 3]], -1).astype(int) * 255
            return rgb_img
        elif mode == "human":
            titles = ["target", "selection 1", "selection 2"] + [f"inv {i}" for i in range(len(self._state) - 3)]
            plot_atoms_list(self._state, titles=titles, background=True, constrained_layout=False, scale=0.5)
        else:
            super().render(mode=mode)  # raise an exception

    def close(self):
        pass
