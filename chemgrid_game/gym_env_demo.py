from tqdm import tqdm

from chemgrid_game.chemistry import generate_random_mol
from chemgrid_game.gym_env import ChemGridEnv

if __name__ == '__main__':
    n_agents = 2
    n_atoms = 10
    max_size = 8
    n_colors = 3

    seeds = range(n_agents)
    targets = [generate_random_mol(s, n_atoms, n_colors, max_size=max_size) for s in seeds]
    inventories = [[generate_random_mol(i, n_atoms=2, n_colors=n_colors, max_size=max_size)] for i in range(n_agents)]
    env = ChemGridEnv(
        n_agents=n_agents,
        continuous_actions=False,
        initial_inventories=inventories,
        survival_mols=targets
    )
    obs = env.reset()
    for i in tqdm(range(1000)):
        action = env.action_space.sample()
        new_obs, rewards, dones, infos = env.step(action)
        if all(dones):
            break
