import abc
import copy
from typing import List
from typing import Optional

import numpy as np

from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.chemistry.utils import RandomMolGenerator
from chemgrid_game.chemistry.utils import create_unit_mol


class TargetGeneratorBase(abc.ABC):
    def __call__(self):
        return copy.deepcopy(self.reset())

    @property
    @abc.abstractmethod
    def target(self) -> Molecule:
        pass

    @abc.abstractmethod
    def reset(self) -> Molecule:
        pass


class InventoryGeneratorBase(abc.ABC):
    def __call__(self):
        return copy.deepcopy(self.reset())

    @abc.abstractmethod
    def reset(self) -> List[Molecule]:
        pass


class CustomInventoryGenerator(InventoryGeneratorBase):
    def __init__(self, inventory: List[Molecule]):
        self._inventory = inventory

    def reset(self) -> List[Molecule]:
        return self._inventory


class BasicInventoryGenerator(InventoryGeneratorBase):
    def __init__(self, grid_size: int, n_colors: int = 1):
        self._inventory = [create_unit_mol(i + 1, grid_size=grid_size) for i in range(n_colors)]

    def reset(self) -> List[Molecule]:
        return self._inventory


class RandomInventoryGenerator(InventoryGeneratorBase):
    def __init__(
            self,
            grid_size: int,
            min_atoms: int,
            max_atoms: Optional[int] = None,
            min_length: int = 1,
            max_length: Optional[int] = None,
            n_colors=1,
            regenerate: bool = False,
            seed: int = 0,
            rng=None
    ):
        if rng is None:
            rng = np.random.default_rng(seed=seed)

        if max_length is None:
            max_length = min_length

        self.rng = rng
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_length = min_length
        self.max_length = max_length
        self.regenerate = regenerate
        self._inventory = None
        self._mol_generator = RandomMolGenerator(grid_size, n_colors, rng=self.rng)
        self._retries = 100

    def reset(self) -> List[Molecule]:
        if self.regenerate or self._inventory is None:
            n = self.rng.integers(self.min_length, self.max_length + 1)
            self._inventory = []
            for _ in range(self._retries):
                while len(self._inventory) < n:
                    mol = self._mol_generator(self.min_atoms, self.max_atoms)
                    if mol not in self._inventory:
                        self._inventory.append(mol)

        return self._inventory


class CustomTargetGenerator(TargetGeneratorBase):
    def __init__(self, target: Molecule):
        self._target = target

    @property
    def target(self) -> Molecule:
        return self._target

    def reset(self) -> Molecule:
        return self._target


class RandomTargetGenerator(TargetGeneratorBase):
    def __init__(
            self,
            grid_size: int,
            min_atoms: int,
            max_atoms: Optional[int] = None,
            n_colors=1,
            regenerate: bool = False,
            seed: int = 0,
            rng=None
    ):
        if rng is None:
            rng = np.random.default_rng(seed=seed)

        self.rng = rng
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.regenerate = regenerate
        self._target: Optional[Molecule] = None
        self._mol_generator = RandomMolGenerator(grid_size, n_colors, rng=rng)

    @property
    def target(self) -> Molecule:
        return self._target

    def reset(self) -> Molecule:
        if self.regenerate or self._target is None:
            self._target = self._mol_generator(self.min_atoms, self.max_atoms)

        return self._target
