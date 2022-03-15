from pathlib import Path
from typing import FrozenSet
from typing import List
from typing import Optional
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
            bonds: List[Bond] = None,
            adjust_top_left: bool = True,
            max_size: int = 8
    ):
        if atoms is None:
            self.atoms = np.zeros((max_size, max_size), dtype=np.uint8)
        else:
            self.atoms = np.array(atoms, dtype=np.uint8)

        self.max_size = max_size
        self._hash = None
        h, w = self.atoms.shape
        assert 0 <= h <= self.max_size and 0 <= w <= self.max_size
        if h != self.max_size or w != self.max_size:
            self.atoms = np.pad(self.atoms, [(0, self.max_size - h), (0, self.max_size - w)])
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
            return np.array_equal(self.atoms, other.atoms)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.atoms.flatten()))

        return self._hash


def break_atoms(atoms: np.ndarray, edges, edge, dim):
    other_edges = edges.difference({edge})

    node1, node2 = edge
    v1 = graph_utils.visit_nodes_via_edges(*node1, other_edges, dim=dim)
    v2 = graph_utils.visit_nodes_via_edges(*node2, other_edges, dim=dim)

    return atoms * v1, atoms * v2


def break_mol(mol: Molecule, edge: Bond) -> Tuple[Molecule, Molecule]:
    m1, m2 = break_atoms(mol.atoms, mol.bonds, edge, mol.max_size)
    m1, m2 = Molecule(m1, max_size=mol.max_size), Molecule(m2, max_size=mol.max_size)
    return m1, m2


def _join_atoms(
        atoms1: np.ndarray, atoms2: np.ndarray, offset1: Tuple, offset2: Tuple, mol_grid_length: int
) -> Optional[np.ndarray]:
    if offset1[0] or offset1[1]:
        shifted1 = graph_utils.shift_atoms(atoms1, *offset1, mol_grid_length)
    else:
        shifted1 = atoms1

    if offset2[0] or offset2[1]:
        shifted2 = graph_utils.shift_atoms(atoms2, *offset2, mol_grid_length)
    else:
        shifted2 = atoms2

    combined = shifted1 + shifted2

    if graph_utils.node_sum_match_parent(combined, [atoms1, atoms2]) and graph_utils.is_connected(combined):
        return combined


def _join_mols(
        mol1: Molecule, mol2: Molecule, offset1: Tuple, offset2: Tuple
) -> List[Molecule]:
    atoms = _join_atoms(mol1.atoms, mol2.atoms, offset1, offset2, mol1.max_size)
    res = []
    if atoms is not None:
        mol = Molecule(atoms, max_size=mol1.max_size)
        res.append(mol)

    return res


def join_mols(mol1: Molecule, mol2: Molecule, row_offset: int, col_offset: int) -> List[Molecule]:
    r1, r2 = 0, row_offset
    c1, c2 = 0, col_offset

    if row_offset < 0:
        r1, r2 = -row_offset, 0

    if col_offset < 0:
        c1, c2 = -col_offset, 0

    return _join_mols(mol1, mol2, (r1, c1), (r2, c2))


def find_join_offsets(mol1: Molecule, mol2: Molecule) -> List[Tuple[int, int]]:
    max_size = mol1.max_size
    offsets = []
    for r in range(-mol2.height, mol1.height + 1):
        for c in range(-mol2.width, mol1.width + 1):
            height = max(mol1.height, r + mol2.height) - min(0, r)
            width = max(mol1.width, c + mol2.width) - min(0, c)
            if height <= max_size and width <= max_size:
                offsets.append((r, c))

    return offsets


def get_all_breaks(mol: Molecule) -> List[Molecule]:
    res = []
    for edge in mol.cut_edges:
        breaks = break_mol(mol, edge)
        res.extend(breaks)

    return res


def get_all_joins(mol1: Molecule, mol2: Molecule) -> List[Molecule]:
    offsets = find_join_offsets(mol1, mol2)
    joins = []
    for offset in offsets:
        joins.extend(join_mols(mol1, mol2, *offset))

    return joins


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

    return Molecule(atoms, max_size=grid_size)


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


def create_unit_mol(col_id: int, grid_size=8) -> Molecule:
    assert 1 <= col_id <= 3
    atoms = np.zeros((grid_size, grid_size), dtype=np.uint8)
    atoms[0, 0] = col_id
    return Molecule(atoms, max_size=grid_size)
