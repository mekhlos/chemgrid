import copy
import enum
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from chemgrid_game import game_config
from chemgrid_game.chemistry import Molecule
from chemgrid_game.game_backend import Archive
from chemgrid_game.game_backend import State
from chemgrid_game.game_config import Config
from chemgrid_game.utils import setup_logger


class Menu(enum.Enum):
    MAIN = 0
    JOIN = 1
    BREAK = 2
    CREATE_CONTRACT = 3
    VIEW_CONTRACTS = 4
    VIEW_AGENT_STATES = 5


@dataclass
class GameState:
    agent_id: int
    mol_archive: Archive
    agent_states: List[State]
    config: Config
    selected_molecules: List[Optional[int]] = field(default_factory=list)
    join_positions: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    demo_molecule: Optional[Molecule] = None
    done: Optional[bool] = None
    mode: Optional["Menu"] = None
    accept: Optional[bool] = None
    constructing: Optional[bool] = None
    inventory_start: Optional[int] = None
    inventory_starts: Optional[List[int]] = None
    states_start: Optional[int] = None
    combo_candidate: Optional[Molecule] = None
    selected_edge: Optional[np.ndarray] = None
    draw_color: Optional[Tuple[int, int, int]] = None
    contracts_start: Optional[int] = None

    def __post_init__(self):
        self.logger = setup_logger(self.__class__.__name__, self.config.logging_level)
        self.agent_state = self.agent_states[self.agent_id]
        self.n_agents = len(self.agent_states)

    @property
    def inventory(self) -> List[Molecule]:
        mol_hashes = self.agent_state[0]
        return [self.mol_archive[h] for h in mol_hashes]

    @property
    def survival_molecule(self) -> Molecule:
        mol_hash = self.agent_state[1]
        return self.mol_archive[mol_hash]

    @property
    def contracts(self) -> List[Tuple[Molecule, Molecule]]:
        contracts = self.agent_state[2]
        contract_mols = []
        for other_agent_id, offer, ask in contracts:
            contract_mols.append((self.mol_archive[offer], self.mol_archive[ask]))

        return contract_mols

    def get_other_agent_states(self) -> List[State]:
        return [self.agent_states[i] for i in range(len(self.agent_states)) if i != self.agent_id]

    def get_selected_mols(self) -> List[Optional[Molecule]]:
        return [self.get_mol(m_id) if m_id is not None else None for m_id in self.selected_molecules]

    def get_mol(self, mol_id: int) -> Molecule:
        inventory, target, contracts = self.agent_state
        mol_hash = inventory[mol_id]
        return self.mol_archive[mol_hash]

    def survived(self) -> bool:
        inventory, target, contracts = self.agent_state
        return target in inventory

    def reset(self):
        self.done = False
        self.selected_molecules = [None, None]
        self.join_positions = [None, None]
        self.demo_molecule = copy.deepcopy(game_config.ZERO_MOLECULE)
        self.mode = Menu.MAIN
        self.accept = False
        self.constructing = False
        self.inventory_start = 0
        self.inventory_starts = [0] * (self.n_agents - 1)
        self.states_start = 0
        self.combo_candidate = None
        self.selected_edge = None
        self.draw_color = None
        self.contracts_start = 0

    def reset_menu(self):
        # self.logger.debug("going (back) to menu mode")
        self.selected_molecules = [None, None]
        self.join_positions = [None, None]
        self.demo_molecule = copy.deepcopy(game_config.ZERO_MOLECULE)
        self.mode = Menu.MAIN
        self.accept = False
        self.constructing = False
        self.inventory_start = 0
        self.inventory_starts = [0] * (self.n_agents - 1)
        self.states_start = 0
        self.combo_candidate = None
        self.selected_edge = None
        self.draw_color = None
        self.contracts_start = 0
