import gym
import numpy as np
from tqdm import tqdm

from chemgrid_game.gym_env.chemgrid_env_single_agent import ChemGridSingleAgentEnv
from chemgrid_game.inventory_and_target_generation import BasicInventoryGenerator
from chemgrid_game.inventory_and_target_generation import RandomTargetGenerator

if __name__ == '__main__':
    n_agents = 2
    grid_size = 8
    n_colors = 3
    rng = np.random.default_rng(seed=0)
    env_id = 'ChemGrid-v1'

    target_kwargs = dict(grid_size=grid_size, min_atoms=10, n_colors=n_colors, rng=rng, regenerate=True)
    inventory_kwargs = dict(grid_size=grid_size, n_colors=n_colors)
    target_generators = [RandomTargetGenerator(**target_kwargs) for _ in range(n_agents)]
    inventory_generators = [BasicInventoryGenerator(**inventory_kwargs) for _ in range(n_agents)]

    env_kwargs = dict(
        n_agents=n_agents,
        continuous_actions=False,
        inventory_generators=inventory_generators,
        target_generators=target_generators
    )
    env = gym.make(env_id, **env_kwargs)
    env = ChemGridSingleAgentEnv(env)

    obs = env.reset()
    for i in tqdm(range(1000)):
        action = env.action_space.sample()
        new_obs, rewards, dones, infos = env.step(action)

    env.close()
