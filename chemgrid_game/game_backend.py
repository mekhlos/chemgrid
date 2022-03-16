import copy
from collections import deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from chemgrid_game.chemistry import mol_chemistry
from chemgrid_game.chemistry.mol_chemistry import Action
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.inventory_and_target_generation import CustomInventoryGenerator
from chemgrid_game.inventory_and_target_generation import CustomTargetGenerator
from chemgrid_game.utils import setup_logger

Inventory = List[int]
Contract = Tuple[int, int, int]
Archive = Dict[int, Molecule]
State = Tuple[Inventory, int, Set[Contract]]


class GameBackend:
    def __init__(
            self,
            inventories: Tuple[List[Molecule]] = None,
            targets: Tuple[Molecule] = None,
            contracts: Set[Contract] = None,
            logging_level: str = "INFO",
            inventory_generators=None,
            target_generators=None
    ):
        if contracts is None:
            contracts = set()

        if target_generators is None:
            target_generators = [CustomTargetGenerator(t) for t in targets]

        if inventory_generators is None:
            inventory_generators = [CustomInventoryGenerator(inv) for inv in inventories]

        self.target_generators = target_generators
        self.inventory_generators = inventory_generators

        self.archive: Archive = {}
        self.inventories: Optional[Tuple[Inventory]] = None
        self.targets: Optional[Tuple[int]] = None
        self.initial_contracts = tuple(contracts)
        self.contracts = contracts
        self.n_agents = len(inventory_generators)
        self.logger = setup_logger(logging_level)
        self.contract_queue = deque()
        self.reached_target = [False] * self.n_agents

    def check_contracts(self):
        for agent_id in range(self.n_agents):
            self.contract_queue.extend((agent_id, mol_id) for mol_id in self.inventories[agent_id])

        while len(self.contract_queue) > 0:
            agent_id, mol_id = self.contract_queue.popleft()
            for other_agent_id, offer_id, ask_id in self.contracts:
                other_inventory = self.inventories[other_agent_id]
                if ask_id == mol_id and offer_id in other_inventory:
                    self.add_mol(agent_id, mol_id=offer_id, parent_op="contract", parent_ids=(ask_id,))
                    self.add_mol(other_agent_id, mol_id=ask_id, parent_op="contract", parent_ids=(offer_id,))

                if offer_id == mol_id and ask_id in other_inventory:
                    self.add_mol(agent_id, mol_id=ask_id, parent_op="contract", parent_ids=(offer_id,))
                    self.add_mol(other_agent_id, mol_id=offer_id, parent_op="contract", parent_ids=(ask_id,))

    def _get_state(self, agent_id: int) -> State:
        inventories = copy.deepcopy(self.inventories[agent_id])
        targets = copy.deepcopy(self.targets[agent_id])
        contracts = copy.deepcopy(self.contracts)

        return inventories, targets, contracts

    def get_states(self) -> Tuple[State]:
        return tuple([self._get_state(i) for i in range(self.n_agents)])

    def _get_reached_target(self, agent_id: int) -> bool:
        self.reached_target[agent_id] = self.targets[agent_id] in self.inventories[agent_id]
        return self.reached_target[agent_id]

    def get_reached_target(self) -> List[bool]:
        return [self._get_reached_target(i) for i in range(self.n_agents)]

    def _is_done(self, agent_id: int) -> bool:
        return False

    def is_done(self) -> List[bool]:
        return [self._is_done(i) for i in range(self.n_agents)]

    def add_mol(
            self,
            agent_id: int,
            mol: Optional[Molecule] = None,
            mol_id: Optional[int] = None,
            parent_op: Optional[str] = None,
            parent_ids: Optional[List[int]] = None
    ):
        if mol_id is None:
            mol_id = hash(mol)
            self.archive[mol_id] = mol
        if mol_id not in self.inventories[agent_id]:
            self.inventories[agent_id].append(mol_id)
            self.contract_queue.append((agent_id, mol_id))

    def process_join_action(self, agent_id: int, action: Action) -> List[Molecule]:
        mol1_id, mol2_id = action.operands
        mol1, mol2 = self.archive[mol1_id], self.archive[mol2_id]
        row_offset, col_offset = action.params
        new_mols = mol_chemistry.join_mols(mol1, mol2, row_offset, col_offset)
        return new_mols

    def process_break_action(self, agent_id: int, action: Action) -> List[Molecule]:
        mol_id, = action.operands
        mol = self.archive[mol_id]
        edge = action.params
        new_mols = mol_chemistry.break_mol(mol, edge)
        return new_mols

    def _step_one(self, agent_id: int, action: Action):
        self.logger.debug("Action: %s" % action.op)
        if action.op == "join":
            new_mols = self.process_join_action(agent_id, action)
            for new_mol in new_mols:
                self.add_mol(agent_id, new_mol, parent_op=action.op, parent_ids=action.operands)

        elif action.op == "break":
            new_mols = self.process_break_action(agent_id, action)
            for new_mol in new_mols:
                self.add_mol(agent_id, new_mol, parent_op=action.op, parent_ids=action.operands)

        elif action.op == "contract":
            offer, ask = action.operands
            new_contract = (agent_id, offer, ask)
            self.contracts.add(new_contract)

        elif action.op == "noop":
            pass

        else:
            raise ValueError(f"Unknown op {action.op}")

    def step(self, actions: Tuple[Action]) -> Tuple[Tuple[State], List[float], List[bool], Dict]:
        [self._step_one(a_id, action) for a_id, action in enumerate(actions)]
        self.check_contracts()
        old_reached_target = np.array(self.reached_target, dtype=float)
        new_reached_target = np.array(self.get_reached_target(), dtype=float)

        rewards = (new_reached_target - old_reached_target).tolist()

        return self.get_states(), rewards, self.is_done(), {}

    def reset(self) -> Tuple[State]:
        self.archive.clear()
        self.contracts.clear()
        self.contract_queue.clear()

        inventories = [g() for g in self.inventory_generators]
        targets = [g() for g in self.target_generators]

        self.inventories = [[hash(m) for m in mols] for mols in inventories]
        self.targets = [hash(m) for m in targets]

        for inventory in inventories:
            for mol in inventory:
                self.archive[hash(mol)] = mol

        for mol in targets:
            self.archive[hash(mol)] = mol

        self.contracts.union(self.initial_contracts)
        self.reached_target = [False] * self.n_agents

        return self.get_states()
