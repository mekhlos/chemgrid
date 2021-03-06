from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from chemgrid_game.chemistry.molecule import Molecule
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


@dataclass
class Config:
    screen_height: int = 256
    screen_width: int = 256
    scale: float = 1
    input_mode: str = "ai"
    pixel_size: int = 4
    pixel_pad: int = 1
    button_width = 20
    button_height = 9
    width: int = 10
    height: int = 10
    margin: int = 5
    grid_size: int = 8
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
    enable_view_agent_states: bool = False
    initial_inventories: Tuple[List[Molecule]] = None
    initial_contracts: Set[Contract] = None
    initial_targets: Tuple[Molecule] = None
    inventory_generators: Optional[List] = None
    target_generators: Optional[List] = None

    def __post_init__(self):
        self.screen_width *= self.scale
        self.screen_height *= self.scale
        self.pixel_size *= self.scale
        self.pixel_pad *= self.scale
        self.button_width *= self.scale
        self.button_height *= self.scale
        self.width *= self.scale
        self.height *= self.scale
        self.atom_width *= self.scale
        self.atom_height *= self.scale
        self.margin *= self.scale
        self.locations = self.get_locations()

    def get_locations(self) -> Dict[str, Dict[str, Any]]:
        pix_dir = Path(__file__).parent.joinpath("pix")

        scale, tiny_mol_size, big_mol_size = self.scale, self.get_tiny_mol_size(), self.get_big_mol_size()
        w_color_picker, h_color_picker = 4 * self.atom_width + 3 * self.margin, self.atom_width
        w_inventory, h_inventory = tiny_mol_size, self.get_inventory_size()
        locations = {
            "join_button": {"x": 50, "y": 220, "w": 20, "h": 10, "img": "join_small.png"},
            "break_button": {"x": 80, "y": 220, "w": 20, "h": 10, "img": "break_small.png"},
            "create_contract_button": {"x": 50, "y": 235, "w": 20, "h": 10, "img": "contract_small.png"},
            "view_contracts_button": {"x": 80, "y": 235, "w": 20, "h": 10, "img": "contract_small_2.png"},
            "view_states_button": {"x": 110, "y": 235, "w": 20, "h": 10, "img": "view_states.png"},
            "accept_button": {"x": 135, "y": 220, "w": 20, "h": 20, "img": "accept.png"},
            "cancel_button": {"x": 160, "y": 220, "w": 20, "h": 20, "img": "cancel.png"},
            "up_arrow": {"x": 208, "y": 1, "w": 20, "h": 13, "img": "up_triangle.png"},
            "down_arrow": {"x": 208, "y": 230, "w": 20, "h": 13, "img": "down_triangle.png"},
            "left_arrow": {"x": 10, "y": 100, "w": 20, "h": 20, "img": "left_triangle.png"},
            "right_arrow": {"x": 190, "y": 100, "w": 20, "h": 20, "img": "right_triangle.png"},
            "white_arrow": {"x": 100, "y": 35, "w": 20, "h": 20, "img": "right_triangle_white.png"},
            "contract_viewer": {"x": 30, "y": 20, "w": 160, "h": 160},
            "agent_state_viewer": {"x": 10, "y": 20, "w": 236, "h": 160},
        }

        scaled_locations = {
            "inventory": {"x": 205 * scale, "y": 20 * scale, "w": w_inventory, "h": h_inventory},
            "color_picker": {"x": 45 * scale, "y": 190 * scale, "w": w_color_picker, "h": h_color_picker},
            "grid": {"x": 5 * scale, "y": 5 * scale, "w": big_mol_size, "h": big_mol_size},
            "game_molecule": {"x": 5 * scale, "y": 5 * scale, "w": big_mol_size, "h": big_mol_size},
            "survival_molecule": {"x": 5 * scale, "y": 220 * scale, "w": tiny_mol_size, "h": tiny_mol_size}
        }

        for name, data in locations.items():
            scaled_locations[name] = {}
            for k, v in data.items():
                if k in ["x", "y", "w", "h"]:
                    scaled_locations[name][k] = v * scale
                elif k == "img":
                    scaled_locations[name]["img_path"] = pix_dir.joinpath(v)
                else:
                    scaled_locations[name][k] = v

        return scaled_locations

    def get_tiny_mol_size(self) -> int:
        return self.grid_size * self.pixel_size + (self.grid_size - 1) * self.pixel_pad

    def get_inventory_size(self) -> int:
        return self.visible_inventory_len * self.get_tiny_mol_size() + (self.visible_inventory_len - 1) * self.margin

    def get_big_mol_size(self) -> float:
        return self.grid_size * self.atom_width + (self.grid_size - 1) * self.margin


if __name__ == '__main__':
    config = Config(n_agents=2)
