import json
from pathlib import Path
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from chemgrid_game import CHEMGRID_GAME_PATH
from chemgrid_game import graph_utils
from chemgrid_game.plotting import plot_atoms

Bond = FrozenSet[Tuple[int, int]]


class Molecule:
    def __init__(
            self,
            atoms: Optional[Union[np.ndarray, List[List[int]]]] = None,
            bonds: Set[Bond] = None,
            adjust_top_left: bool = True,
            grid_size: int = 8
    ):
        if atoms is None:
            self.atoms = np.zeros((grid_size, grid_size), dtype=np.uint8)
        else:
            self.atoms = np.array(atoms, dtype=np.uint8)

        self.grid_size = grid_size
        self._hash = None
        h, w = self.atoms.shape
        assert 0 <= h <= self.grid_size and 0 <= w <= self.grid_size
        if h != self.grid_size or w != self.grid_size:
            self.atoms = np.pad(self.atoms, [(0, self.grid_size - h), (0, self.grid_size - w)])
        (self.start_row, self.start_col), (self.height, self.width) = self.get_rect()
        self.actual_shape = (self.height, self.width)
        self._bonds = bonds

        if adjust_top_left:
            self.adjust_top_left()

    @property
    def bonds(self):
        if self._bonds is None:
            self._bonds = graph_utils.find_edges(self.atoms)

        return self._bonds

    @property
    def cut_edges(self):
        if self._cut_edges is None:
            self._cut_edges = graph_utils.find_cut_edges(self.bonds)

        return self._cut_edges

    def get_rect(self):
        is_non_empty_row = self.atoms.any(1)
        is_non_empty_col = self.atoms.any(0)
        start_row, start_col = 0, 0
        while start_row < self.atoms.shape[0] and not is_non_empty_row[start_row]:
            start_row += 1
        while start_col < self.atoms.shape[1] and not is_non_empty_col[start_col]:
            start_col += 1
        height, width = is_non_empty_row.sum(), is_non_empty_col.sum()
        return (start_row, start_col), (height, width)

    def adjust_top_left(self) -> "Molecule":
        self.atoms = self.get_core(pad=True)
        self.start_row, self.start_col = 0, 0
        self._hash = None
        self._bonds = None
        self._cut_edges = None
        return self

    def get_core(self, pad=False) -> np.ndarray:
        r, c = self.start_row, self.start_col
        h, w = self.actual_shape
        core = self.atoms[r:r + h, c:c + w]
        if pad:
            atoms = np.zeros_like(self.atoms)
            atoms[:h, :w] = core
        else:
            atoms = core

        return atoms

    def get_img_path(self) -> Path:
        p = CHEMGRID_GAME_PATH.joinpath(f"mol_imgs/{hash(self)}.png")
        if not p.parent.is_dir():
            p.parent.mkdir(exist_ok=True)
        return p

    def get_mol_path(self) -> str:
        p = CHEMGRID_GAME_PATH.joinpath(f"mols/{hash(self)}.json")

        if not p.parent.is_dir():
            p.parent.mkdir(exist_ok=True)
        return str(p)

    def save(self) -> str:
        p = self.get_mol_path()
        with open(p, "w") as f:
            data = {
                "atoms": self.atoms.tolist(),
                "bonds": self.bonds
            }
            json.dump(data, f)

        return p

    @staticmethod
    def load(path) -> "Molecule":
        with open(path) as f:
            data = json.load(f)

        return Molecule(data["atoms"])

    def render(self, square_size=1, ax=None, core_only=False, save_fig=False, add_title=False):
        if core_only:
            data = self.get_core()
        else:
            data = self.atoms
        fig = None
        if ax is None:
            n_rows, n_cols = data.shape
            h, w = n_rows * square_size, n_cols * square_size
            fig, ax = plt.subplots(figsize=(w, h))

        plot_atoms(data, scale=square_size, ax=ax)

        if add_title:
            ax.set_title(f"{self}")
        if save_fig and fig is not None and not self.get_img_path().is_file():
            fig.savefig(self.get_img_path())

        if fig is not None:
            plt.close()

    def __repr__(self):
        h, w = self.actual_shape
        return f"{h}x{w} molecule with {len(self.bonds)} bonds"

    def __eq__(self, other):
        if isinstance(other, Molecule):
            return np.array_equal(self.atoms, other.atoms) and self.bonds == other.bonds

        return False

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.atoms.flatten()))

        return self._hash
