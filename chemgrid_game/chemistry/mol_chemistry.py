from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from chemgrid_game import graph_utils
from chemgrid_game.chemistry.molecule import Bond
from chemgrid_game.chemistry.molecule import Molecule


@dataclass(frozen=True)
class Action:
    op: str = "noop"
    operands: Tuple = ()
    params: Optional = None
    res: Optional[Any] = None

    def add_res(self, res) -> "Action":
        return Action(self.op, self.operands, self.params, res)


def _break_atoms(atoms: np.ndarray, edges, edge, dim):
    other_edges = edges.difference({edge})

    node1, node2 = edge
    v1 = graph_utils.visit_nodes_via_edges(*node1, other_edges, dim=dim)
    v2 = graph_utils.visit_nodes_via_edges(*node2, other_edges, dim=dim)

    return atoms * v1, atoms * v2


def break_mol(mol: Molecule, edge: Bond) -> Tuple[Molecule, Molecule]:
    m1, m2 = _break_atoms(mol.atoms, mol.bonds, edge, mol.grid_size)
    m1, m2 = Molecule(m1, grid_size=mol.grid_size), Molecule(m2, grid_size=mol.grid_size)
    return m1, m2


def _join_atoms(
        atoms1: np.ndarray,
        atoms2: np.ndarray,
        offset1: Tuple,
        offset2: Tuple,
        grid_size: int,
        check_valid: bool = True
) -> Optional[np.ndarray]:
    if offset1[0] or offset1[1]:
        shifted1 = graph_utils.shift_atoms(atoms1, *offset1, grid_size)
    else:
        shifted1 = atoms1

    if offset2[0] or offset2[1]:
        shifted2 = graph_utils.shift_atoms(atoms2, *offset2, grid_size)
    else:
        shifted2 = atoms2

    combined = shifted1 + shifted2

    if not check_valid or \
            graph_utils.node_sum_match_parent(combined, [atoms1, atoms2]) and graph_utils.is_connected(combined):
        return combined


def join_mols(mol1: Molecule, mol2: Molecule, row_offset: int, col_offset: int, check_valid=True) -> List[Molecule]:
    r1, r2 = 0, row_offset
    c1, c2 = 0, col_offset

    if row_offset < 0:
        r1, r2 = -row_offset, 0

    if col_offset < 0:
        c1, c2 = -col_offset, 0

    atoms = _join_atoms(mol1.atoms, mol2.atoms, (r1, c1), (r2, c2), mol1.grid_size, check_valid)
    res = []
    if atoms is not None:
        mol = Molecule(atoms, grid_size=mol1.grid_size)
        res.append(mol)

    return res


def find_join_offsets(mol1: Molecule, mol2: Molecule, check_valid=False) -> List[Tuple[int, int]]:
    max_size = mol1.grid_size
    offsets = []
    for r in range(-mol2.height, mol1.height + 1):
        for c in range(-mol2.width, mol1.width + 1):
            height = max(mol1.height, r + mol2.height) - min(0, r)
            width = max(mol1.width, c + mol2.width) - min(0, c)
            if height <= max_size and width <= max_size:
                if not check_valid or is_valid_join(mol1, mol2, (r, c)):
                    offsets.append((r, c))

    return offsets


def is_valid_join(mol1: Molecule, mol2: Molecule, offset: Tuple[int, int]) -> bool:
    res = join_mols(mol1, mol2, *offset)
    for _ in res:
        return True

    return False


class ChemistryWrapper:
    def __init__(self, use_caching=True):
        self.use_caching = use_caching
        if use_caching:
            self._offset_dict: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            self._cut_edge_dict: Dict[int, List[Bond]] = {}

    def is_valid_join(self, mol1: Molecule, mol2: Molecule, offset: Tuple[int, int]) -> bool:
        return is_valid_join(mol1, mol2, offset)

    def find_join_offsets(self, mol1: Molecule, mol2: Molecule, check_valid: bool = False) -> List[Tuple[int, int]]:
        if self.use_caching:
            k = (hash(mol1), hash(mol2))
            if k not in self._offset_dict:
                self._offset_dict[k] = find_join_offsets(mol1, mol2, check_valid)

            return self._offset_dict[k]

        return find_join_offsets(mol1, mol2)

    def find_cut_edges(self, mol: Molecule):
        if self.use_caching:
            k = hash(mol)
            if k not in self._cut_edge_dict:
                self._cut_edge_dict[k] = graph_utils.find_cut_edges(mol.bonds)

            return self._cut_edge_dict[k]

        return graph_utils.find_cut_edges(mol.bonds)

    def _get_all_joins(self, mol1: Molecule, mol2: Molecule) -> List[Molecule]:
        offsets = self.find_join_offsets(mol1, mol2)
        joins = []
        for offset in offsets:
            joins.extend(join_mols(mol1, mol2, *offset))

        return joins

    def _get_all_breaks(self, mol: Molecule) -> List[Molecule]:
        res = []
        for edge in mol.cut_edges:
            breaks = break_mol(mol, edge)
            res.extend(breaks)

        return res

    def join_mols(self, mol1: Molecule, mol2: Molecule, offset=None, check_valid=True) -> List[Molecule]:
        if offset is None:
            return self._get_all_joins(mol1, mol2)
        return join_mols(mol1, mol2, *offset, check_valid=check_valid)

    def break_mol(self, mol: Molecule, edge: Bond = None) -> List[Molecule]:
        if edge is None:
            return self._get_all_breaks(mol)

        return break_mol(mol, edge)

    def get_valid_actions(
            self,
            mol1: Molecule,
            mol2: Optional[Molecule] = None,
            op: Optional[str] = None,
            check_join_valid: bool = False
    ) -> List[Action]:
        actions = []
        if op == "break" or op is None:
            cut_edges = self.find_cut_edges(mol1)
            for edge in cut_edges:
                actions.append(Action(op="break", operands=(hash(mol1),), params=edge))

        if mol2 is not None and (op == "join" or op is None):
            offsets = self.find_join_offsets(mol1, mol2, check_valid=check_join_valid)
            for offset in offsets:
                hash1, hash2 = hash(mol1), hash(mol2)
                actions.append(Action(op="join", operands=(hash1, hash2), params=offset))

        return actions

    def process_action(self, action: Action, check_valid=True) -> List[Molecule]:
        if action.op == "join":
            return self.join_mols(*action.operands, offset=action.params, check_valid=check_valid)
        elif action.op == "break":
            return self.break_mol(*action.operands, edge=action.params)
        else:
            raise ValueError(f"Unknown op {action.op}")
