from chemgrid_game import chemistry
from chemgrid_game.game import Game
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.game_frontend import GameFrontend

if __name__ == '__main__':
    n_agents = 1
    n_atoms = 10
    max_size = 8
    seeds = range(n_agents)
    n_colors = 3
    targets = [chemistry.generate_random_mol(s, n_atoms, n_colors, max_size=max_size) for s in seeds]
    inventories = [
        [chemistry.create_unit_mol(i + 1, max_size=max_size) for i in range(n_colors)] for _ in range(n_agents)
    ]
    # inventories = [[chemistry.generate_random_mol(i, 2, n_colors, max_size=max_size)] for i in range(n_agents)]
    # inventories = [[Molecule([[1]]), Molecule([[1, 1]])]]
    # config = Config(input_mode="human", fps=30, survival_mols=[DEMO_MOLECULE1], initial_inventories=[[DEMO_MOLECULE2]]
    config = Config(
        initial_inventories=tuple(inventories),
        survival_mols=tuple(targets),
        logging_level="DEBUG",
        scale=2
    )
    frontend = GameFrontend(config)
    backend = GameBackend(
        config.initial_inventories, config.survival_mols, config.initial_contracts, logging_level="DEBUG"
    )
    game = Game(frontend, backend, config)
    game.reset()

    for i in range(1000):
        actions = game.get_clicks()
        print(f"actions: {actions}")
        game.step(actions)

    game.close()
