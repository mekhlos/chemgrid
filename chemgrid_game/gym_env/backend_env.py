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
        n_break_options = config.grid_size ** 2 * 2
        shape = [max_inv_size + 3, config.grid_size, config.grid_size, 3]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)
        nvec = [max_inv_size, n_break_options, n_join_options]
        self.action_space = gym.spaces.MultiDiscrete(nvec=nvec, dtype=np.uint8)
        self._done = False
        self._selected_mol_id1 = None
        self._selected_mol_id2 = None
        self._obs, self._rgb_obs = None, None

    def _update_obs(self, states):
        inventory_hashes, target_hash, _ = states[0]
        inventory = [self._backend.archive[i].atoms for i in inventory_hashes]
        target = self._backend.archive[target_hash].atoms
        if self._selected_mol_id1 is not None:
            selected_mol1 = inventory[self._selected_mol_id1]
        else:
            selected_mol1 = np.zeros_like(target)

        if self._selected_mol_id2 is not None:
            selected_mol2 = inventory[self._selected_mol_id2]
        else:
            selected_mol2 = np.zeros_like(target)

        obs = np.concatenate([np.stack([target, selected_mol1, selected_mol2]), inventory])
        # pad obs so that its shape is always the same
        diff = max(0, self.max_inv_size + 3 - len(obs))
        pad = np.zeros((diff, *obs.shape[1:]))
        obs = np.concatenate([obs, pad])
        self._obs, self._rgb_obs = obs, self._to_rgb(obs)

    def _break_offset_to_edge(self, offset: int) -> Optional[Bond]:
        n = self._config.grid_size

        id1 = offset // 2
        if offset % 2 == 0:
            id2 = id1 + 1
        else:
            id2 = id1 + n

        if id2 < n * n:
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
        action_type_id, break_offset, join_offset = action
        mol_id = None if action_type_id < 3 else action_type_id - 3
        action_type_id = min(action_type_id, 3)
        action_type = ["reset", "break", "join", "mol_selection"][action_type_id]

        action = Action()
        if action_type == "reset":
            self._selected_mol_id1 = None
            self._selected_mol_id2 = None
        elif action_type == "break" and self._selected_mol_id1 is not None:
            mol_hash = inventory[self._selected_mol_id1]
            edge = self._break_offset_to_edge(break_offset)
            if edge is not None:
                action = Action("break", (mol_hash,), edge)
            self._selected_mol_id1 = None
            self._selected_mol_id2 = None
        elif action_type == "join" and self._selected_mol_id2 is not None:
            mol_hash1 = inventory[self._selected_mol_id1]
            mol_hash2 = inventory[self._selected_mol_id2]
            offset = self._join_offset_to_edge(join_offset)
            action = Action("join", (mol_hash1, mol_hash2), offset)
            self._selected_mol_id1 = None
            self._selected_mol_id2 = None
        elif action_type == "mol_selection":
            if self._selected_mol_id1 is None and mol_id < len(inventory):
                self._selected_mol_id1 = mol_id
            elif self._selected_mol_id2 is None and mol_id < len(inventory):
                self._selected_mol_id2 = mol_id

        self.logger.debug("Sending action: %s", str(action))
        states, rewards, dones, infos = self._backend.step((action,))
        self._done = dones[0]

        self._update_obs(states)

        return self._rgb_obs, rewards[0], self._done, infos[0]

    def reset(self, **kwargs) -> np.ndarray:
        states = self._backend.reset()
        self.logger.debug("reset")
        self._done = False
        self._selected_mol_id1 = None
        self._selected_mol_id2 = None
        self._update_obs(states)
        return self._rgb_obs

    def _to_rgb(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([obs == i for i in [1, 2, 3]], -1).astype(np.uint8) * 255

    def render(self, mode="human"):
        if mode == "rgb_array":
            n, h, w, d = self._rgb_obs.shape
            return self._rgb_obs.reshape((n * h, w, d))
        elif mode == "human":
            titles = ["target", "selection 1", "selection 2"] + [f"inv {i}" for i in range(len(self._obs) - 3)]
            plot_atoms_list(self._obs, titles=titles, background=True, constrained_layout=False, scale=0.5)
        else:
            super().render(mode=mode)  # raise an exception

    def close(self):
        pass
