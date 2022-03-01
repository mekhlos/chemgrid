from typing import FrozenSet
from typing import List
from typing import Set
from typing import Tuple

import numpy as np

Bond = FrozenSet[Tuple[int, int]]


def find_edges(x: np.ndarray) -> Set[Bond]:
    m, n = x.shape

    edges = set()
    for r in range(m):
        for c in range(n):
            if x[r, c] > 0:
                if c + 1 < n and x[r, c + 1] > 0:
                    edges.add(frozenset({(r, c), (r, c + 1)}))
                if r + 1 < m and x[r + 1, c] > 0:
                    edges.add(frozenset({(r, c), (r + 1, c)}))
    return edges


def visit_nodes_via_edges(i: int, j: int, edges: Set[Bond], visited=None, dim=8) -> np.ndarray:
    if visited is None:
        visited = np.zeros((dim, dim), dtype=bool)

    visited[i, j] = 1

    for next_i, next_j in ((i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)):
        edge = frozenset({(i, j), (next_i, next_j)})
        if edge in edges and not visited[next_i, next_j]:
            visit_nodes_via_edges(next_i, next_j, edges, visited)

    return visited


def find_cut_edges(edges: Set[Bond]) -> List[Bond]:
    cut_edges = []
    for node1, node2 in edges:
        other_edges = edges - {frozenset({node1, node2})}
        v1 = visit_nodes_via_edges(*node1, other_edges)
        v2 = visit_nodes_via_edges(*node2, other_edges)
        if not np.array_equal(v1, v2):
            cut_edges.append(frozenset({node1, node2}))

    return cut_edges


def find_connected_nodes(a: np.ndarray, i: int = 0, j: int = 0, visited=None) -> np.ndarray:
    m, n = a.shape
    if visited is None:
        visited = np.zeros((m, n), dtype=bool)

        for i in range(m):
            for j in range(n):
                if a[i, j] != 0:
                    find_connected_nodes(a, i, j, visited)
                    return visited

    visited[i, j] = True
    for next_i, next_j in ([i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]):
        if 0 <= next_i < m and 0 <= next_j < n and a[next_i, next_j] > 0 and not visited[next_i, next_j]:
            find_connected_nodes(a, next_i, next_j, visited)

    return visited


def generate_all_shifts(x: np.ndarray, flatten: bool = False) -> np.ndarray:
    m, n = x.shape
    res = []
    for i in range(2 * m):
        for j in range(2 * n):
            base = np.zeros((3 * m, 3 * n))
            base[i:i + m, j:j + n] = x
            res.append(base)

    x = np.array(res, dtype=np.uint8).reshape((2 * m, 2 * n, 3 * m, 3 * n))
    if flatten:
        x = x.reshape(-1, 3 * m, 3 * n)
    return x


def is_connected(x: np.ndarray) -> bool:
    n_atoms = np.sum(x > 0)
    visited = find_connected_nodes(x).sum()
    return visited == n_atoms


def node_sum_match_parent(combined: np.ndarray, parents: List[np.ndarray]) -> bool:
    n_atoms = np.sum(combined > 0)
    n_parents_atoms = sum(np.sum(p > 0) for p in parents)
    return n_atoms == n_parents_atoms


def shift_atoms(atoms: np.ndarray, row_offset: int, col_offset: int, mol_grid_length: int):
    new_atoms = np.zeros((mol_grid_length, mol_grid_length), dtype=np.uint8)
    new_atoms[row_offset:, col_offset:] = atoms[:mol_grid_length - row_offset, :mol_grid_length - col_offset]
    return new_atoms


def count_nodes(atoms: np.ndarray) -> int:
    return np.sum(atoms > 0).item()


def combine_atoms(atoms1: np.ndarray, atoms2: np.ndarray) -> np.ndarray:
    return atoms1 + atoms2


def goes_offscreen(atoms: np.ndarray, clicked_row, clicked_col, mol_grid_length: int):
    shifted = shift_atoms(atoms, clicked_row, clicked_col, mol_grid_length)
    return (shifted > 0).sum() != (atoms > 0).sum()
