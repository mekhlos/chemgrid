import gym
import numpy as np
from tqdm import tqdm

from chemgrid_game.inventory_and_target_generation import BasicInventoryGenerator
from chemgrid_game.inventory_and_target_generation import RandomTargetGenerator

if __name__ == '__main__':
    n_agents = 2
    grid_size = 5
    n_colors = 3
    rng = np.random.default_rng(seed=0)
    env_id = 'ChemGridBackend-v1'

    target_kwargs = dict(grid_size=grid_size, min_atoms=10, n_colors=n_colors, rng=rng, regenerate=True)
    inventory_kwargs = dict(grid_size=grid_size, n_colors=n_colors)
    target_generators = [RandomTargetGenerator(**target_kwargs) for _ in range(n_agents)]
    inventory_generators = [BasicInventoryGenerator(**inventory_kwargs) for _ in range(n_agents)]

    env_kwargs = dict(
        inventory_generators=inventory_generators,
        target_generators=target_generators,
        max_inv_size=10,
        grid_size=grid_size
    )
    env = gym.make(env_id, **env_kwargs)

    obs = env.reset()
    for i in tqdm(range(1000)):
        action = env.action_space.sample()
        new_obs, rewards, dones, infos = env.step(action)

    env.render()
    env.close()
