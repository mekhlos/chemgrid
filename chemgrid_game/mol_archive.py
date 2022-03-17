from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from chemgrid_game.chemistry.molecule import Molecule


class MolArchive:
    def __init__(
            self,
            initial_mols: List[Molecule] = (),
            max_len: Optional[int] = None
    ):
        self.max_len = np.inf if max_len is None else max_len
        self.hashes = []
        self.hash2mol = {}

        for mol in initial_mols:
            self.add(mol)

    def add(self, mol: Molecule, **kwargs):
        self[hash(mol)] = mol

    def get(self, mol_hash: Optional[int] = None, pos: Optional[int] = None) -> Molecule:
        if mol_hash is None:
            mol_hash = self.hashes[pos]

        return self[mol_hash]

    def get_molecules(self) -> List[Molecule]:
        return [self[h] for h in self.hashes]

    def get_dict(self, hashes: List[int]) -> Dict[int, Molecule]:
        return {h: self[h] for h in hashes}

    def __setitem__(self, mol_hash: int, molecule: Molecule):
        if len(self) < self.max_len:
            if mol_hash not in self.hash2mol:
                self.hashes.append(mol_hash)
            self.hash2mol[mol_hash] = molecule

    def __getitem__(self, mol_hash) -> Molecule:
        return self.hash2mol[mol_hash]

    def __len__(self) -> int:
        return len(self.hashes)

    def __contains__(self, x) -> bool:
        return x in self.hash2mol

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self)} molecules"

    def reset(self):
        self.hashes.clear()
        self.hash2mol.clear()
