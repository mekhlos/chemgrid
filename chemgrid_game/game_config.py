from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from chemgrid_game.chemistry import Molecule
from chemgrid_game.game_backend import Contract

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (122, 55, 139)

ZERO_MOLECULE = Molecule(np.zeros((8, 8), dtype=np.uint8))

DEMO_MOLECULE1 = Molecule(np.array([
    [1, 1, 1, 1, 1, 1, 0, 0],
    [2, 2, 2, 0, 0, 2, 0, 0],
    [3, 3, 3, 0, 0, 3, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [2, 2, 2, 2, 2, 2, 0, 0],
    [3, 3, 3, 3, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]))

DEMO_MOLECULE2 = Molecule(np.array([
    [0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0],
    [0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]))

DEMO_MOLECULE3 = Molecule(np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]))

DEMO_MOLECULE4 = Molecule(np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 0, 0, 0, 0],
    [0, 2, 1, 2, 1, 0, 0, 0],
    [0, 0, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]))

SURVIVAL_MOL = Molecule(np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 0, 0, 0],
    [0, 0, 1, 2, 3, 0, 0, 0],
    [0, 0, 0, 1, 2, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]))


@dataclass
class Config:
    size_mult: int = 1
    input_mode: str = "ai"
    pixel_size: int = 4
    pixel_pad: int = 1
    button_width = 20
    button_height = 9
    width: int = 10
    height: int = 10
    margin: int = 5
    mol_grid_length = 8
    chem_grid_length = 8
    n_agents: int = 1
    visible_inventory_len: int = 4
    visible_contract_viewer_len: int = 3
    logging_level: str = "INFO"
    use_dummy_logger: bool = False
    move_cursor: bool = False
    fps: Optional[int] = 60
    atom_width: float = width
    atom_height: float = width
    atom_colors = [WHITE, RED, GREEN, BLUE]
    enable_create_contract: bool = True
    enable_view_contracts: bool = True
    initial_inventories: Tuple[List[Molecule]] = ([DEMO_MOLECULE3],)
    initial_contracts: Set[Contract] = field(default_factory=set)
    survival_mols: Tuple[Molecule] = (SURVIVAL_MOL,)

    def get_tiny_mol_size(self) -> int:
        return self.mol_grid_length * self.pixel_size + (self.mol_grid_length - 1) * self.pixel_pad

    def get_inventory_size(self) -> int:
        return self.visible_inventory_len * self.get_tiny_mol_size() + (self.visible_inventory_len - 1) * self.margin

    def get_big_mol_size(self) -> float:
        return self.mol_grid_length * self.atom_width + (self.mol_grid_length - 1) * self.margin


if __name__ == '__main__':
    config = Config(n_agents=2)
