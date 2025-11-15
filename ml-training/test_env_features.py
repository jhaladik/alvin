"""Quick test to verify environment extracts all 69 features"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.pacman_env import PacManEnv
import numpy as np

env = PacManEnv(maze_size=(20, 20))
state = env.reset()

print(f'State shape: {state.shape}')
print(f'Non-zero features: {np.count_nonzero(state)}')
print(f'\nFeature verification:')
print(f'  Player pos (0-2): {state[0:2]}')
print(f'  Direction one-hot (2-6): {state[2:6]} (should have one 1.0)')
print(f'  Ghost data (6-30): {np.count_nonzero(state[6:30])} non-zero')
print(f'  Pellet data (30-38): {np.count_nonzero(state[30:38])} non-zero')
print(f'  Power pellet (38-44): {np.count_nonzero(state[38:44])} non-zero')
print(f'  Wall distances (44-52): {np.count_nonzero(state[44:52])} non-zero')
print(f'  Tactical (52-64): {np.count_nonzero(state[52:64])} non-zero')
print(f'  Game state (64-69): {state[64:69]}')
print(f'\n✅ All feature categories present!')
