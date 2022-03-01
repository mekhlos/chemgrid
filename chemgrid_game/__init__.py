from pathlib import Path

from gym.envs.registration import register

register(id='ChemGrid-v1', entry_point='chemgrid_game.gym_env:ChemGridEnv')

CHEMGRID_GAME_PATH = Path(__file__).parent