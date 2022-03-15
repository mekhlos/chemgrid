import numpy as np

from chemgrid_game.game import Game
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_config import Config
from chemgrid_game.game_frontend import GameFrontend
from chemgrid_game.inventory_and_target_generation import RandomInventoryGenerator
from chemgrid_game.inventory_and_target_generation import RandomTargetGenerator

if __name__ == '__main__':
    n_agents = 2
    n_atoms = 10
    grid_size = 8
    seeds = range(n_agents)
    n_colors = 3
    rng = np.random.default_rng(seed=0)

    target_kwargs = dict(grid_size=grid_size, min_atoms=n_atoms, n_colors=n_colors, rng=rng, regenerate=True)
    inventory_kwargs = dict(
        grid_size=grid_size,
        min_atoms=1,
        max_atoms=3,
        min_length=1,
        max_length=3,
        n_colors=n_colors,
        regenerate=True,
        rng=rng
    )

    target_generators = [RandomTargetGenerator(**target_kwargs) for _ in range(n_agents)]
    inventory_generators = [RandomInventoryGenerator(**inventory_kwargs) for _ in range(n_agents)]
    config = Config(
        target_generators=target_generators,
        inventory_generators=inventory_generators,
        logging_level="INFO",
        scale=2,
    )
    frontend = GameFrontend(config)
    backend = GameBackend(
        inventory_generators=config.inventory_generators,
        target_generators=config.target_generators,
        contracts=config.initial_contracts,
        logging_level="DEBUG"
    )
    game = Game(frontend, backend, config)
    game.reset()

    for i in range(1000):
        actions = game.get_clicks()
        print(f"actions: {actions}")
        rewards, dones = game.step(actions)
        print(rewards, dones)
        if (i + 1) % 100 == 0:
            game.reset()

    game.close()
