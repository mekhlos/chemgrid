from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
import pygame
from pygame.sprite import AbstractGroup

from chemgrid_game import graph_utils
from chemgrid_game.chemistry.mol_chemistry import Action
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_config import BLACK
from chemgrid_game.game_config import BLUE
from chemgrid_game.game_config import Config
from chemgrid_game.game_config import GREEN
from chemgrid_game.game_config import PURPLE
from chemgrid_game.game_config import RED
from chemgrid_game.game_config import WHITE
from chemgrid_game.game_config import YELLOW
from chemgrid_game.game_helpers import GameState
from chemgrid_game.game_helpers import Menu
from chemgrid_game.utils import setup_logger


class ClickySprite(pygame.sprite.DirtySprite):
    def __init__(self, *groups: AbstractGroup):
        super().__init__(*groups)

    def is_clicked(self, pos: Tuple[int, int]):
        clicked = self.rect.collidepoint(*pos)
        return clicked

    def on_click(self, game_state: GameState):
        pass

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        if self.is_clicked(pos):
            return self.on_click(game_state)


class ClickySpriteWithImg(ClickySprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, screen):
        screen.blit(self.image, self.rect)


class Button(ClickySprite):
    def __init__(self, x, y, w, h, img_path: str):
        super().__init__()
        image = pygame.image.load(img_path)
        self.image = pygame.transform.scale(image, (w, h))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, screen):
        screen.blit(self.image, self.rect)


class BreakButton(Button):
    def on_click(self, game_state: GameState):
        if game_state.selected_molecules[0] is not None:
            game_state.mode = Menu.BREAK


class JoinButton(Button):
    def on_click(self, game_state: GameState):
        if game_state.selected_molecules[0] is not None:
            game_state.mode = Menu.JOIN


class ContractCreateButton(Button):
    def on_click(self, game_state: GameState):
        if game_state.selected_molecules[0] is not None and game_state.config.enable_create_contract:
            game_state.mode = Menu.CREATE_CONTRACT


class ContractViewButton(Button):
    def on_click(self, game_state: GameState):
        if game_state.selected_molecules[0] is None and game_state.config.enable_view_contracts:
            game_state.mode = Menu.VIEW_CONTRACTS


class AgentStateViewButton(Button):
    def on_click(self, game_state: GameState):
        if game_state.selected_molecules[0] is None and game_state.config.enable_view_agent_states:
            game_state.mode = Menu.VIEW_AGENT_STATES


class AcceptButton(Button):

    def on_click(self, game_state: GameState):
        game_state.logger.debug("accept button clicked")

        mol1_id = game_state.selected_molecules[0]
        action = Action("noop")

        if game_state.mode == Menu.CREATE_CONTRACT:
            ask_mol = Molecule(game_state.demo_molecule.atoms.copy(), max_size=game_state.config.mol_grid_length)
            offer, ask = hash(game_state.inventory[mol1_id]), hash(ask_mol)
            game_state.mol_archive[ask] = ask_mol
            action = Action("contract", (offer, ask), ())
        elif game_state.mode == Menu.VIEW_CONTRACTS:
            pass
        elif game_state.mode == Menu.VIEW_AGENT_STATES:
            pass
        elif game_state.mode == Menu.JOIN:
            mol2_id = game_state.selected_molecules[1]
            mol1 = game_state.inventory[mol1_id]
            mol2 = game_state.inventory[mol2_id]
            (x1, y1), (x2, y2) = game_state.join_positions
            action = Action("join", (hash(mol1), hash(mol2)), ((x2 - x1), (y2 - y1)))

        elif game_state.mode == Menu.BREAK and game_state.selected_edge is not None:
            mol = game_state.inventory[mol1_id]
            action = Action("break", (hash(mol),), game_state.selected_edge)

        game_state.reset_menu()
        return action


class CancelButton(Button):
    def on_click(self, game_state: GameState):
        mol1 = game_state.selected_molecules[0]
        game_state.reset_menu()


class UpArrow(Button):
    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_AGENT_STATES:
            if game_state.states_start > 0:
                game_state.states_start -= 1
        else:
            if game_state.inventory_start > 0:
                game_state.inventory_start -= 1


class DownArrow(Button):
    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_AGENT_STATES:
            n_items = len(game_state.get_other_agent_states())
            if game_state.states_start < n_items - game_state.config.visible_contract_viewer_len:
                game_state.states_start += 1
                game_state.inventory_starts = [0] * (game_state.n_agents - 1)

        else:
            n_items = len(game_state.inventory)
            if game_state.inventory_start < n_items - game_state.config.visible_inventory_len:
                game_state.inventory_start += 1


class LeftArrow(Button):
    def __init__(self, x, y, w, h, img_path: str, list_item_id: Optional[int] = None):
        super().__init__(x, y, w, h, img_path)
        self.list_item_id = list_item_id

    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_CONTRACTS:
            if game_state.contracts_start > 0:
                game_state.contracts_start -= 1
                game_state.inventory_starts = [0] * (game_state.n_agents - 1)

        elif game_state.mode == Menu.VIEW_AGENT_STATES:
            inventory_starts = game_state.inventory_starts
            if inventory_starts[self.list_item_id] > 0:
                inventory_starts[self.list_item_id] -= 1


class RightArrow(Button):
    def __init__(self, x, y, w, h, img_path: str, list_item_id: Optional[int] = None):
        super().__init__(x, y, w, h, img_path)
        self.list_item_id = list_item_id

    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_CONTRACTS:
            n_items = len(game_state.contracts)
            if game_state.contracts_start < n_items - game_state.config.visible_contract_viewer_len:
                game_state.contracts_start += 1

        elif game_state.mode == Menu.VIEW_AGENT_STATES:
            inventory_starts = game_state.inventory_starts
            inventory, target, _ = game_state.get_other_agent_states()[self.list_item_id]
            n_items = len(inventory)

            if inventory_starts[self.list_item_id] < n_items - game_state.config.visible_contract_viewer_len:
                inventory_starts[self.list_item_id] += 1


class WhiteArrow(Button):
    def on_click(self, game_state: GameState):
        pass


class AtomSprite(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, grid_pos, color):
        super().__init__(x, y, w, h)
        self.grid_pos = grid_pos
        self.c = w / 2
        self.color = color

    def update(self, game_state: GameState):
        pygame.draw.circle(self.image, self.color, (self.c, self.c), self.c)

    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.JOIN:
            if game_state.selected_molecules[1] is not None:
                game_state.join_positions[1] = self.grid_pos
            elif game_state.selected_molecules[0] is not None:
                game_state.join_positions[0] = self.grid_pos
        elif game_state.mode == Menu.CREATE_CONTRACT:
            draw_color = game_state.draw_color
            if draw_color is None:
                draw_color = WHITE
            color_id = game_state.config.atom_colors.index(draw_color)
            atoms = game_state.demo_molecule.atoms
            atoms[self.grid_pos[0], self.grid_pos[1]] = color_id
            game_state.demo_molecule = Molecule(atoms, adjust_top_left=False)
            game_state.accept = graph_utils.is_connected(atoms)


class ColorPickerSprite(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, color):
        super().__init__(x, y, w, h)
        self.c = w / 2
        self.color = color
        self.image.fill(color)

    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.CREATE_CONTRACT:
            game_state.draw_color = self.color


class ColorPickerBar(ClickySpriteWithImg):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.atoms = []

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        x, y = pos
        x, y = x - self.rect.x, y - self.rect.y
        for atom in self.atoms:
            atom.check_click((x, y), game_state)

    def update(self, game_state: GameState):
        conf = game_state.config
        self.atoms.clear()
        for i, color in enumerate(conf.atom_colors):
            x = i * (conf.width + conf.margin)
            y = 0
            self.atoms.append(ColorPickerSprite(x, y, conf.width, conf.height, color))

        atoms = pygame.sprite.Group(self.atoms)
        atoms.update(game_state)
        atoms.draw(self.image)


class GridSprite(ClickySpriteWithImg):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.atoms = []

    def create_atoms(self, game_state):
        self.atoms.clear()
        conf = game_state.config
        mol_w, mol_h = conf.width, conf.height

        for row in range(conf.mol_grid_length):
            for col in range(conf.mol_grid_length):
                x = (conf.margin + mol_w) * col
                y = (conf.margin + mol_h) * row

                game_atom = AtomSprite(x, y, mol_w, mol_h, grid_pos=(row, col), color=WHITE)
                self.atoms.append(game_atom)

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        x, y = pos
        x, y = x - self.rect.x, y - self.rect.y
        if self.is_clicked(pos):
            for atom in self.atoms:
                atom.check_click((x, y), game_state)

    def update(self, game_state: GameState):
        self.create_atoms(game_state)
        atoms = pygame.sprite.Group(*self.atoms)
        atoms.update(game_state)
        atoms.draw(self.image)


class GameBond(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, color, bond):
        super().__init__(x, y, w, h)
        self.color = color
        self.c = w / 2
        self.bond = bond

    def update(self, game_state: GameState):
        pygame.draw.circle(self.image, self.color, [self.c, self.c], self.c)

    def on_click(self, game_state: GameState):
        mol = game_state.get_selected_mols()[0]
        if mol is not None and self.bond in mol.cut_edges and game_state.mode == Menu.BREAK:
            game_state.selected_edge = self.bond


class GameMolecule(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, molecule: Molecule):
        super().__init__(x, y, w, h)
        # Set transparent color
        self.image.set_colorkey(BLACK)
        self.molecule = molecule
        self.atom_colors = [WHITE, RED, GREEN, BLUE]
        self.atoms = []
        self.bonds = []

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        x, y = pos
        x, y = x - self.rect.x, y - self.rect.y
        if self.is_clicked(pos):
            if game_state.mode == Menu.BREAK:
                for bond in self.bonds:
                    bond.check_click((x, y), game_state)

    def create_atoms(self, game_state):
        self.atoms.clear()
        conf = game_state.config
        mol_w, mol_h = conf.width, conf.height
        atoms = self.molecule.atoms

        for row in range(conf.mol_grid_length):
            for col in range(conf.mol_grid_length):
                atom = atoms[row, col]
                if atom > 0:
                    x = (conf.margin + mol_w) * col
                    y = (conf.margin + mol_h) * row

                    game_atom = AtomSprite(x, y, mol_w, mol_h, (row, col), self.atom_colors[atom])
                    self.atoms.append(game_atom)

    def create_bonds(self, game_state):
        self.bonds.clear()
        conf = game_state.config

        mol_w, mol_h = conf.width, conf.height
        bond_w, bond_h = mol_w / 2, mol_h / 2
        selected_edge = game_state.selected_edge
        for edge in self.molecule.bonds:
            (row, col), (next_row, next_col) = sorted(edge)
            if edge == selected_edge:
                color = PURPLE
            elif edge in self.molecule.cut_edges:
                color = YELLOW
            else:
                color = WHITE
            # vertical
            if row != next_row:
                x = (conf.margin + mol_w) * col + 0.5 * mol_w - 0.5 * bond_w
                y = (conf.margin + mol_h) * row + mol_h + 0.5 * conf.margin - 0.5 * bond_h
                game_bond = GameBond(x, y, bond_w, bond_h, color, edge)
                self.bonds.append(game_bond)
                # self.image.blit(game_bond.image, game_bond.rect)

            # horizontal
            if col != next_col:
                x = (conf.margin + mol_h) * col + mol_w + 0.5 * conf.margin - 0.5 * bond_w
                y = (conf.margin + mol_w) * row + 0.5 * mol_h - 0.5 * bond_h
                game_bond = GameBond(x, y, bond_w, bond_h, color, edge)
                self.bonds.append(game_bond)
                # self.image.blit(game_bond.image, game_bond.rect)

    def update(self, game_state: GameState):
        self.create_atoms(game_state)
        atoms = pygame.sprite.Group(*self.atoms)
        atoms.update(game_state)
        atoms.draw(self.image)

        if game_state.mode == Menu.BREAK:
            self.create_bonds(game_state)
            bonds = pygame.sprite.Group(*self.bonds)
            bonds.update(game_state)
            bonds.draw(self.image)


class TinyMolecule(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, molecule: Molecule, is_survival_mol=False, is_selected=False):
        super().__init__(x, y, w, h)
        self.molecule = molecule
        self.is_survival_mol = is_survival_mol
        self.is_selected = is_selected

    def draw_atoms(self, game_state: GameState):
        conf = game_state.config
        if self.is_selected:
            d = conf.get_tiny_mol_size()
            pygame.draw.rect(self.image, WHITE, (0, 0, d, d), max(int(conf.scale), 1))

        for row in range(conf.mol_grid_length):
            for column in range(conf.mol_grid_length):
                atom = self.molecule.atoms[row, column]
                if atom > 0:
                    color = game_state.config.atom_colors[atom]

                    cx = column * (conf.pixel_size + conf.pixel_pad)
                    cy = row * (conf.pixel_size + conf.pixel_pad)
                    pygame.draw.rect(self.image, color, [cx, cy, conf.pixel_size, conf.pixel_size])

    def draw_bonds(self, game_state: GameState):
        conf = game_state.config

        bond_size = 0.5 * conf.pixel_size
        for bond in self.molecule.bonds:
            (x1, y1), (x2, y2) = bond
            # horizontal bond
            if x1 == x2:
                row = x1
                column = min(y1, y2)  # + 0.5
                h_offset = conf.pixel_size + conf.pixel_pad * 0.5 - bond_size * 0.5
                w_offset = 0.5 * conf.pixel_size - bond_size * 0.5

            # vertical bond
            else:
                row = min(x1, x2)  # + 0.5
                column = y1
                h_offset = 0.5 * conf.pixel_size - bond_size * 0.5
                w_offset = conf.pixel_size + conf.pixel_pad * 0.5 - bond_size * 0.5

            rx = h_offset + column * (conf.pixel_size + conf.pixel_pad)
            ry = w_offset + row * (conf.pixel_size + conf.pixel_pad)
            pygame.draw.rect(self.image, WHITE, [rx, ry, bond_size, bond_size])

    def update(self, game_state: GameState) -> None:
        self.image.fill(BLACK)
        h = w = game_state.config.get_tiny_mol_size() + game_state.config.scale
        if self.is_survival_mol and game_state.survived():
            pygame.draw.rect(self.image, WHITE, [0, 0, w, h])

        self.draw_atoms(game_state)
        self.draw_bonds(game_state)

    def on_click(self, game_state: GameState):
        inventory = game_state.inventory
        if self.molecule in inventory and not self.is_survival_mol:
            if game_state.mode == Menu.MAIN:
                if game_state.selected_molecules[0] is None:
                    game_state.selected_molecules[0] = inventory.index(self.molecule)
            elif game_state.mode == Menu.JOIN:
                if game_state.join_positions[0] is not None and game_state.selected_molecules[1] is None:
                    game_state.selected_molecules[1] = inventory.index(self.molecule)

            game_state.inventory_start = 0
            game_state.logger.debug("Selected mols: %s" % game_state.selected_molecules)


class ContractViewer(ClickySpriteWithImg):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        buttons_dir = Path(__file__).parent.joinpath("pix")
        self.arrow_path = f'{buttons_dir}/right_triangle_white.png'

    def update(self, game_state: GameState) -> None:
        contracts = game_state.contracts[game_state.contracts_start:]
        w_tiny_mol = h_tiny_mol = game_state.config.get_tiny_mol_size()
        w_arrow = game_state.config.locations["white_arrow"]["w"]
        h_arrow = game_state.config.locations["white_arrow"]["w"]
        scale = game_state.config.scale
        for i, contract in enumerate(contracts):
            y = i * 50 * scale
            pygame.draw.rect(self.image, WHITE, rect=(0 * scale, y, 150 * scale, 50 * scale), width=int(3 * scale))
            mol = TinyMolecule(20 * scale, y + 5 * scale, w_tiny_mol, h_tiny_mol, molecule=contract[1])
            mol.update(game_state)
            self.image.blit(mol.image, mol.rect)
            arrow = WhiteArrow(70 * scale, y + 15 * scale, w_arrow, h_arrow, self.arrow_path)
            self.image.blit(arrow.image, arrow.rect)
            mol = TinyMolecule(95 * scale, y + 5 * scale, w_tiny_mol, h_tiny_mol, contract[0])
            mol.update(game_state)
            self.image.blit(mol.image, mol.rect)


class AgentStateWithArrows(ClickySpriteWithImg):
    def __init__(self, x, y, w, h, list_item_id: Optional[int] = None):
        super().__init__(x, y, w, h)
        self.items = []
        self.list_item_id = list_item_id

    def create_molecules(self, game_state: GameState, x_start, y, w, h):
        conf = game_state.config
        states = game_state.get_other_agent_states()
        offset = game_state.states_start
        states = states[offset: offset + conf.visible_contract_viewer_len]
        inventory, target, _ = states[self.list_item_id]
        offset = game_state.inventory_starts[self.list_item_id]
        mol_ids = inventory[offset:offset + conf.visible_contract_viewer_len]

        game_state.logger.debug(f"Drawing items {len(mol_ids)}")
        for i, mol_id in enumerate(mol_ids):
            mol = game_state.mol_archive[mol_id]
            x = x_start + i * (w + conf.margin)
            game_mol = TinyMolecule(x, y, w, h, molecule=mol, is_selected=False)
            self.items.append(game_mol)

    def create_left_arrow(self, x, y, w, h, img_path):
        left_arrow = LeftArrow(x, y, w, h, img_path, self.list_item_id)
        self.items.append(left_arrow)

    def create_right_arrow(self, x, y, w, h, img_path):
        right_arrow = RightArrow(x, y, w, h, img_path, self.list_item_id)
        self.items.append(right_arrow)

    def create_survival_mol(self, game_state: GameState, x, y, w, h):
        inventory, target, _ = game_state.get_other_agent_states()[self.list_item_id]
        mol = game_state.mol_archive[target]
        game_mol = TinyMolecule(x, y, w, h, molecule=mol, is_selected=False)
        self.items.append(game_mol)

    def update(self, game_state: GameState):
        self.items.clear()

        locs = game_state.config.get_locations()
        left_arrow_info = locs["left_arrow"]
        right_arrow_info = locs["right_arrow"]
        h_arrow, w_arrow = left_arrow_info["h"], left_arrow_info["w"]
        w_mol = h_mol = game_state.config.get_tiny_mol_size()

        conf = game_state.config
        margin = conf.margin
        left_arrow_start_x = margin
        left_arrow_start_y = margin
        right_arrow_start_x = self.image.get_width() - margin - w_arrow
        right_arrow_start_y = margin

        frame_thickness = int(2 * conf.scale)

        frame_start_x = left_arrow_start_x + w_arrow + margin
        frame_start_y = 0
        target_start_x = frame_start_x + frame_thickness + 1
        inv_start_x = target_start_x + w_mol + margin
        w_frame1 = 2 * (frame_thickness + 1) + w_mol
        w_frame2 = w_frame1 + (w_mol + margin) * conf.visible_contract_viewer_len
        h_frame = 2 * (frame_thickness + 1) + h_mol

        self.create_left_arrow(left_arrow_start_x, left_arrow_start_y, w_arrow, h_arrow, left_arrow_info["img_path"])
        self.create_right_arrow(
            right_arrow_start_x, right_arrow_start_y, w_arrow, h_arrow, right_arrow_info["img_path"])
        self.create_survival_mol(game_state, target_start_x, frame_thickness + 1, w_mol, h_mol)
        self.create_molecules(game_state, inv_start_x, frame_thickness + 1, w_mol, h_mol)
        self.image.fill(BLACK)
        mols = pygame.sprite.Group(*self.items)
        mols.update(game_state)
        mols.draw(self.image)

        pygame.draw.rect(
            self.image, WHITE, rect=(frame_start_x, frame_start_y, w_frame1, h_frame), width=frame_thickness)
        pygame.draw.rect(
            self.image, WHITE, rect=(frame_start_x, frame_start_y, w_frame2, h_frame), width=frame_thickness)

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        if self.is_clicked(pos):
            x, y = pos
            x -= self.rect.x
            y -= self.rect.y
            for item in self.items:
                item.check_click((x, y), game_state)


class AgentStateViewer(ClickySpriteWithImg):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.items = []

    def update(self, game_state: GameState) -> None:
        self.items.clear()
        other_agent_states = game_state.get_other_agent_states()
        offset = game_state.states_start
        other_agent_states = other_agent_states[offset:offset + game_state.config.visible_contract_viewer_len]
        h_tiny_mol = game_state.config.get_tiny_mol_size()
        frame_thickness = int(2 * game_state.config.scale)
        h_frame = h_tiny_mol + 2 * (frame_thickness + 1)

        for i, (inventory_ids, target_id, _) in enumerate(other_agent_states):
            y = i * h_frame + game_state.config.margin
            item = AgentStateWithArrows(0, y, self.image.get_width(), h_frame, i)
            self.image.blit(item.image, item.rect)
            item.update(game_state)
            item.draw(self.image)
            self.items.append(item)

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        if self.is_clicked(pos):
            x, y = pos
            x -= self.rect.x
            y -= self.rect.y
            for item in self.items:
                item.check_click((x, y), game_state)


class Inventory(ClickySpriteWithImg):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.visible_mols = []

    def create_molecules(self, game_state: GameState):
        self.visible_mols.clear()
        selected_mol_id1 = game_state.selected_molecules[0]
        selected_mol_id2 = game_state.selected_molecules[1]
        conf = game_state.config
        inventory = game_state.inventory

        selected_mols = [inventory[i] for i in [selected_mol_id1, selected_mol_id2] if i is not None]
        if game_state.mode == Menu.JOIN and selected_mol_id1 is not None and selected_mol_id2 is not None:
            mols = [inventory[selected_mol_id1], inventory[selected_mol_id2]]
        elif selected_mol_id1 is not None and game_state.mode != Menu.JOIN:
            mols = [inventory[selected_mol_id1]]
        else:
            offset = game_state.inventory_start
            mols = inventory[offset:offset + conf.visible_inventory_len]

        game_state.logger.debug(f"Drawing items {len(mols)}")
        w = h = game_state.config.get_tiny_mol_size()
        for i, mol in enumerate(mols):
            y = i * (conf.get_tiny_mol_size() + conf.margin)
            game_mol = TinyMolecule(0, y, w, h, molecule=mol, is_selected=mol in selected_mols)
            self.visible_mols.append(game_mol)

    def update(self, game_state: GameState):
        self.create_molecules(game_state)
        self.image.fill(BLACK)
        mols = pygame.sprite.Group(*self.visible_mols)
        mols.update(game_state)
        mols.draw(self.image)

    def check_click(self, pos: Tuple[int, int], game_state: GameState):
        if self.is_clicked(pos):
            x, y = pos
            x -= self.rect.x
            y -= self.rect.y
            for mol in self.visible_mols:
                mol.check_click((x, y), game_state)

            self.create_molecules(game_state)


class GameFrontend:
    def __init__(self, config: Config):
        pygame.init()

        self.logger = setup_logger(self.__class__.__name__, config.logging_level)

        self.config = config

        size = (config.screen_width, config.screen_height)
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()

        locs = config.get_locations()

        self.join_button = JoinButton(**locs["join_button"])
        self.break_button = BreakButton(**locs["break_button"])
        self.create_contract_button = ContractCreateButton(**locs["create_contract_button"])
        self.view_contracts_button = ContractViewButton(**locs["view_contracts_button"])
        self.view_states_button = AgentStateViewButton(**locs["view_states_button"])
        self.accept_button = AcceptButton(**locs["accept_button"])
        self.cancel_button = CancelButton(**locs["cancel_button"])
        self.up_arrow = UpArrow(**locs["up_arrow"])
        self.down_arrow = DownArrow(**locs["down_arrow"])
        self.left_arrow = LeftArrow(**locs["left_arrow"])
        self.right_arrow = RightArrow(**locs["right_arrow"])
        self.white_arrow = WhiteArrow(**locs["white_arrow"])
        self.inventory = Inventory(**locs["inventory"])
        self.color_picker = ColorPickerBar(**locs["color_picker"])
        self.contract_viewer = ContractViewer(**locs["contract_viewer"])
        self.agent_state_viewer = AgentStateViewer(**locs["agent_state_viewer"])
        self.survival_mol = None
        self.active_group = pygame.sprite.Group()

        self.done = False
        # self.reset()

    def menu_mode(self):
        self.active_group.empty()
        self.active_group.add(
            self.inventory,
            self.join_button,
            self.break_button,
            self.create_contract_button,
            self.view_contracts_button,
            self.cancel_button,
            self.up_arrow,
            self.down_arrow,
            self.survival_mol
        )

        if self.config.enable_view_agent_states:
            self.active_group.add(self.view_states_button)

        self.inventory.create_molecules(self.game_state)

    def join_mode(self):
        grid = GridSprite(**self.config.locations["grid"])
        mol1 = self.game_state.get_selected_mols()[0]
        mol2 = self.game_state.get_selected_mols()[1]
        join_pos1 = self.game_state.join_positions[0]
        join_pos2 = self.game_state.join_positions[1]

        self.logger.debug("Join pos 1: %s" % str(join_pos1))
        self.logger.debug("Join pos 2: %s" % str(join_pos2))

        self.active_group.empty()
        self.active_group.add(
            self.inventory,
            self.join_button,
            self.cancel_button,
            self.survival_mol,
            grid
        )
        if mol2 is None:
            self.active_group.add(
                self.up_arrow,
                self.down_arrow
            )

        self.game_state.accept = False

        if join_pos1 is not None:
            shifted1 = graph_utils.shift_atoms(mol1.atoms, *join_pos1, self.config.mol_grid_length)
            # Check for overlap
            if graph_utils.goes_offscreen(mol1.atoms, *join_pos1, self.config.mol_grid_length):
                self.logger.debug("Join failed (mol 1 goes offscreen)")
            else:
                mol = Molecule(shifted1, adjust_top_left=False)
                mol_sprite = GameMolecule(**self.config.locations["game_molecule"], molecule=mol)
                self.active_group.add(mol_sprite)
                if join_pos2 is not None:
                    shifted2 = graph_utils.shift_atoms(mol2.atoms, *join_pos2, self.config.mol_grid_length)

                    self.logger.debug("shifted 1: \n%s" % shifted1)
                    self.logger.debug("shifted 2: \n%s" % shifted2)

                    if graph_utils.goes_offscreen(mol2.atoms, *join_pos2, self.config.mol_grid_length):
                        self.logger.debug("Join failed (mol 2 goes offscreen)")
                    else:
                        combo_atoms = graph_utils.combine_atoms(shifted1, shifted2)
                        sum_matches_parent = graph_utils.node_sum_match_parent(combo_atoms, [shifted1, shifted2])
                        is_connected = graph_utils.is_connected(combo_atoms)
                        if sum_matches_parent and is_connected:
                            self.logger.debug("Join success")
                            mol = Molecule(shifted2, adjust_top_left=False)
                            mol_sprite = GameMolecule(**self.config.locations["game_molecule"], molecule=mol)
                            self.active_group.add(mol_sprite)

                            combo_atoms = graph_utils.combine_atoms(shifted1, shifted2)
                            self.game_state.combo_candidate = Molecule(combo_atoms)
                            self.game_state.accept = True
                            self.active_group.add(self.accept_button)

        self.inventory.create_molecules(self.game_state)

        # self.logger.debug(self.active_group)

    def break_mode(self):
        mol = self.game_state.get_selected_mols()[0]
        break_mol = GameMolecule(**self.config.locations["game_molecule"], molecule=mol)
        self.active_group.empty()
        self.active_group.add(
            self.inventory,
            self.break_button,
            self.cancel_button,
            self.survival_mol,
            break_mol
        )
        if self.game_state.selected_edge is not None:
            self.game_state.accept = True
            self.active_group.add(self.accept_button)

        # self.logger.debug(self.active_group)

    def create_contract_mode(self):
        self.logger.debug("create contract mode")
        selected_mol = self.game_state.get_selected_mols()[0]
        grid_sprite = GridSprite(**self.config.locations["grid"])
        mol_sprite = GameMolecule(**self.config.locations["game_molecule"], molecule=self.game_state.demo_molecule)
        self.active_group.empty()
        self.active_group.add(
            self.inventory,
            self.create_contract_button,
            self.cancel_button,
            self.color_picker,
            self.survival_mol,
            grid_sprite,
            mol_sprite
        )

    def view_contract_mode(self):
        self.logger.debug("view contract mode")
        self.active_group.empty()
        self.active_group.add(
            self.left_arrow,
            self.right_arrow,
            self.view_contracts_button,
            self.contract_viewer,
            self.cancel_button
        )

    def view_agent_states_mode(self):
        self.logger.debug("view agent states mode")
        self.active_group.empty()
        self.active_group.add(
            self.view_states_button,
            self.agent_state_viewer,
            self.cancel_button,
            self.up_arrow,
            self.down_arrow
        )

    def draw(self, screen):
        screen.fill(BLACK)
        self.active_group.update(self.game_state)
        self.active_group.draw(screen)

    def check_click(self, pos) -> Action:
        action = Action()
        for item in self.active_group:
            action_candidate = item.check_click(pos, self.game_state)
            if action_candidate is not None and action_candidate.op != "noop":
                action = action_candidate

        self.logger.debug("Action: %s" % action.op)
        return action

    def step(self, pos) -> Action:
        action = self.check_click(pos)
        return action

    def update_game(self, game_state: GameState):
        self.game_state = game_state
        self.survival_mol = TinyMolecule(
            **self.config.locations["survival_molecule"],
            molecule=self.game_state.survival_molecule,
            is_survival_mol=True
        )

        self.logger.debug(self.game_state.mode)
        if self.game_state.mode == Menu.MAIN:
            self.menu_mode()
        elif self.game_state.mode == Menu.BREAK:
            self.break_mode()
        elif self.game_state.mode == Menu.JOIN:
            self.join_mode()
        elif self.game_state.mode == Menu.CREATE_CONTRACT:
            self.create_contract_mode()
        elif self.game_state.mode == Menu.VIEW_CONTRACTS:
            self.view_contract_mode()
        elif self.game_state.mode == Menu.VIEW_AGENT_STATES:
            self.view_agent_states_mode()

        if self.game_state.accept:
            self.active_group.add(self.accept_button)

        self.draw(self.screen)

    def render(self):
        pygame.display.flip()
        if self.config.fps is not None:
            self.clock.tick(self.config.fps)

    def to_img_array(self) -> np.ndarray:
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def close(self):
        pygame.quit()
