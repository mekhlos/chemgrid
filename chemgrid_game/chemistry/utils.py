from typing import Optional

import numpy as np

from chemgrid_game.chemistry.molecule import Molecule


def create_unit_mol(color_id: int, grid_size=8) -> Molecule:
    assert 1 <= color_id <= 3
    atoms = np.zeros((grid_size, grid_size), dtype=np.uint8)
    atoms[0, 0] = color_id
    return Molecule(atoms, grid_size=grid_size)


def generate_random_mol(seed: Optional[int], n_atoms: int, n_colors: int = 1, grid_size: int = 8, rng=None) -> Molecule:
    if rng is None:
        rng = np.random.default_rng(seed)
    atoms = np.zeros((grid_size, grid_size), dtype=np.uint8)
    i = rng.integers(grid_size * grid_size)
    x, y = np.unravel_index(i, (grid_size, grid_size))
    atoms[x, y] = 1
    while (atoms > 0).sum() < n_atoms:
        i = rng.integers(grid_size * grid_size)
        x, y = np.unravel_index(i, (grid_size, grid_size))
        if atoms[x - 1:x + 2, y].sum() > 0 or atoms[x, y - 1:y + 2].sum() > 0:
            atoms[x, y] = rng.integers(1, n_colors + 1)

    return Molecule(atoms, grid_size=grid_size)


class RandomMolGenerator:
    def __init__(self, grid_size: int, n_colors: int = 1, seed=0, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed=seed)

        self.rng = rng
        self.grid_size = grid_size
        self.n_colors = n_colors

    def __call__(self, min_atoms: int, max_atoms: Optional[int] = None) -> Molecule:
        if max_atoms is None:
            max_atoms = min_atoms
        n_atoms = self.rng.integers(min_atoms, max_atoms + 1)

        return generate_random_mol(
            seed=None,
            n_atoms=n_atoms,
            n_colors=self.n_colors,
            grid_size=self.grid_size,
            rng=self.rng
        )
