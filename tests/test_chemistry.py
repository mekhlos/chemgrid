import copy

from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.chemistry.utils import generate_random_mol


def test_mol_eq1():
    mol1 = generate_random_mol(0, 10, 3)
    mol2 = copy.deepcopy(mol1)

    assert mol1 == mol2


def test_mol_eq2():
    bonds1 = {
        frozenset(((0, 0), (0, 1))),
        frozenset(((0, 0), (1, 0))),
        frozenset(((1, 0), (1, 1))),
    }

    bonds2 = {
        frozenset(((0, 0), (0, 1))),
        frozenset(((0, 0), (1, 0))),
        frozenset(((0, 1), (1, 1))),
    }

    bonds3 = {
        frozenset(((0, 0), (0, 1))),
        frozenset(((0, 0), (1, 0))),
        frozenset(((1, 1), (1, 0))),
    }

    mol1 = Molecule([[1, 1], [2, 2]], bonds1, adjust_top_left=False)
    mol2 = Molecule([[1, 1], [2, 2]], bonds2, adjust_top_left=False)
    mol3 = Molecule([[1, 1], [2, 2]], bonds3, adjust_top_left=False)

    assert mol1 != mol2
    assert mol1 == mol3
