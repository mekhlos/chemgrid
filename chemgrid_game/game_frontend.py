from pathlib import Path
from typing import Tuple

import numpy as np
import pygame
from pygame.sprite import AbstractGroup

from chemgrid_game import graph_utils
from chemgrid_game.chemistry import Molecule
from chemgrid_game.game_backend import Action
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
        # if clicked:
        #     print(f"{self.__class__.__name__} was clicked")
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

    def __init__(self, x, y, img_path: str):
        super().__init__()
        self.image = pygame.image.load(img_path)
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
        if game_state.inventory_start > 0:
            game_state.inventory_start -= 1


class DownArrow(Button):
    def on_click(self, game_state: GameState):
        n_items = len(game_state.inventory)
        if game_state.inventory_start < n_items - game_state.config.visible_inventory_len:
            game_state.inventory_start += 1


class LeftArrow(Button):
    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_CONTRACTS:
            if game_state.contracts_start > 0:
                game_state.contracts_start -= 1


class RightArrow(Button):
    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.VIEW_CONTRACTS:
            n_items = len(game_state.contracts)
            if game_state.contracts_start < n_items - game_state.config.visible_contract_viewer_len:
                game_state.contracts_start += 1


class WhiteArrow(Button):
    def on_click(self, game_state: GameState):
        pass


class AtomSprite(ClickySpriteWithImg):
    def __init__(self, x, y, h, w, grid_pos, color):
        super().__init__(x, y, h, w)
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
    def __init__(self, x, y, h, w, color):
        super().__init__(x, y, h, w)
        self.c = w / 2
        self.color = color
        self.image.fill(color)

    def on_click(self, game_state: GameState):
        if game_state.mode == Menu.CREATE_CONTRACT:
            game_state.draw_color = self.color


class ColorPickerBar(ClickySpriteWithImg):
    def __init__(self, x, y, conf: Config):
        w = 4 * conf.atom_width + 3 * conf.margin
        h = conf.atom_width
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
    def __init__(self, x, y, game_state: GameState):
        conf = game_state.config
        super().__init__(x, y, conf.get_big_mol_size(), conf.get_big_mol_size())
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
    def __init__(self, x, y, molecule: Molecule, game_state: GameState):
        conf = game_state.config
        super().__init__(x, y, conf.get_big_mol_size(), conf.get_big_mol_size())
        # Set our transparent color
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
    def __init__(self, x, y, molecule: Molecule, game_state: GameState, is_survival_mol=False, is_selected=False):
        conf = game_state.config
        self.w = conf.get_tiny_mol_size()
        self.h = conf.get_tiny_mol_size()
        super().__init__(x, y, self.w, self.h)
        self.molecule = molecule
        self.is_survival_mol = is_survival_mol
        self.is_selected = is_selected

    def draw_atoms(self, game_state: GameState):
        conf = game_state.config
        if self.is_selected:
            d = conf.get_tiny_mol_size()
            pygame.draw.rect(self.image, WHITE, (0, 0, d, d), 1)

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
        if self.is_survival_mol and game_state.survived():
            pygame.draw.rect(self.image, WHITE, [0, 0, 40, 40])

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
        for i, contract in enumerate(contracts):
            y = i * 50
            pygame.draw.rect(self.image, WHITE, rect=(0, y, 150, 50), width=3)
            mol = TinyMolecule(20, y + 5, contract[1], game_state)
            mol.update(game_state)
            self.image.blit(mol.image, mol.rect)
            arrow = WhiteArrow(70, y + 15, self.arrow_path)
            self.image.blit(arrow.image, arrow.rect)
            mol = TinyMolecule(95, y + 5, contract[0], game_state)
            mol.update(game_state)
            self.image.blit(mol.image, mol.rect)


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
        for i, mol in enumerate(mols):
            y = i * (conf.get_tiny_mol_size() + conf.margin)
            game_mol = TinyMolecule(0, y, mol, game_state, is_selected=mol in selected_mols)
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

        size = (256 * config.size_mult, 256 * config.size_mult)
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()

        buttons_dir = Path(__file__).parent.joinpath("pix")

        self.join_button = JoinButton(50, 220, f'{buttons_dir}/join_small.png')
        self.break_button = BreakButton(80, 220, f'{buttons_dir}/break_small.png')
        self.create_contract_button = ContractCreateButton(50, 235, f'{buttons_dir}/contract_small.png')
        self.view_contracts_button = ContractViewButton(80, 235, f'{buttons_dir}/contract_small_2.png')
        self.accept_button = AcceptButton(135, 220, f'{buttons_dir}/accept.png')
        self.cancel_button = CancelButton(160, 220, f'{buttons_dir}/cancel.png')
        self.up_arrow = UpArrow(208, 1, f'{buttons_dir}/up_triangle.png')
        self.down_arrow = DownArrow(208, 230, f'{buttons_dir}/down_triangle.png')
        self.left_arrow = LeftArrow(10, 100, f'{buttons_dir}/left_triangle.png')
        self.right_arrow = RightArrow(190, 100, f'{buttons_dir}/right_triangle.png')
        self.white_arrow = WhiteArrow(100, 35, f'{buttons_dir}/right_triangle_white.png')

        self.inventory = Inventory(205, 20, config.get_tiny_mol_size(), config.get_inventory_size())
        # self.survival_mol = TinyMolecule(5, 220, self.game_state.survival_molecule, self.game_state, True)
        self.survival_mol = None
        self.color_picker = ColorPickerBar(25 + 2 * config.width, 190, self.config)
        self.contract_viewer = ContractViewer(30, 20, 160, 150)
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

        self.inventory.create_molecules(self.game_state)

    def join_mode(self):
        grid = GridSprite(5, 5, self.game_state)
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
                mol_sprite = GameMolecule(5, 5, Molecule(shifted1, adjust_top_left=False), self.game_state)
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
                            mol_sprite = GameMolecule(5, 5, Molecule(shifted2, adjust_top_left=False), self.game_state)
                            self.active_group.add(mol_sprite)

                            combo_atoms = graph_utils.combine_atoms(shifted1, shifted2)
                            self.game_state.combo_candidate = Molecule(combo_atoms)
                            self.game_state.accept = True
                            self.active_group.add(self.accept_button)

        self.inventory.create_molecules(self.game_state)

        # self.logger.debug(self.active_group)

    def break_mode(self):
        mol = self.game_state.get_selected_mols()[0]
        break_mol = GameMolecule(5, 5, mol, self.game_state)
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
        grid_sprite = GridSprite(5, 5, self.game_state)
        mol_sprite = GameMolecule(5, 5, self.game_state.demo_molecule, self.game_state)
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
        self.survival_mol = TinyMolecule(5, 220, self.game_state.survival_molecule, self.game_state, True)

        self.logger.debug(self.game_state.mode)
        # self.logger.debug(self.active_group)
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

        if self.game_state.accept:
            self.active_group.add(self.accept_button)

        self.update_image()

    def update_image(self):
        self.draw(self.screen)

    def render(self):
        pygame.display.flip()
        if self.config.fps is not None:
            self.clock.tick(self.config.fps)

    def to_img_array(self) -> np.ndarray:
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def close(self):
        pygame.quit()
