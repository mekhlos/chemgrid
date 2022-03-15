# ChemGrid 2022
![ChemGrid GIF](./chemgrid_example1.gif) 

## Install

- Clone repo and install requirements.txt

## Run

- Run `game_demo.py` for human play
- Run `gym_env_demo.py` for gym example

## Game description

In this game agents can build molecules by joining, breaking and exchanging other molecules. Agents need to complete
their survival molecule in order to stay in the game. The faster they build it the more time they can spend to build
other stuff.

Each agent starts off with an initial inventory and a survival molecule. Agents can use the molecules in their
inventories to construct new molecules until (and beyond) their target is reached.

There are four high-level actions

### ![Join](chemgrid_game/pix/join_small.png) Join

Join two existing molecules by connecting them such that there is no overlap between the two and at least one pair of
molecules are adjacent.

### ![Break](chemgrid_game/pix/break_small.png) Break

Break an existing molecule on a bond (edge) that is the only bond between the two subcomponents.

### ![Create contract](chemgrid_game/pix/contract_small.png) Create a contract

Create a "contract" by offering one of the items of the agent in exchange for any existing or non-existent molecule.

### ![Contract view](chemgrid_game/pix/contract_small_2.png) View contracts

View contracts that have been created so far.

## Gym env

```python
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
```