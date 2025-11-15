"""
Headless Pac-Man Environment for Reinforcement Learning
Simulates the game without rendering for fast training
"""
import numpy as np
from typing import Tuple, Dict, List
import random


class PacManEnv:
    """
    Pac-Man environment following Gym-like interface

    State: Game state representation
    Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Reward: Score changes + shaped rewards
    """

    def __init__(self, maze_size=(20, 20), max_steps=1000):
        self.maze_width = maze_size[0]
        self.maze_height = maze_size[1]
        self.max_steps = max_steps

        # Game state
        self.player_pos = None
        self.ghost_positions = []
        self.pellets = set()
        self.power_pellets = set()
        self.walls = set()
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0

        # For tracking
        self.last_distance_to_pellet = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Simple maze layout
        self._create_maze()

        # Place player
        self.player_pos = (self.maze_width // 2, self.maze_height // 2)

        # Place ghosts
        self.ghost_positions = [
            (3, 3),
            (self.maze_width - 4, 3),
            (3, self.maze_height - 4),
            (self.maze_width - 4, self.maze_height - 4)
        ]

        # Reset game state
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0

        return self._get_state()

    def _create_maze(self):
        """Create simple maze with walls, pellets, power pellets"""
        self.walls = set()
        self.pellets = set()
        self.power_pellets = set()

        # Create border walls
        for x in range(self.maze_width):
            self.walls.add((x, 0))
            self.walls.add((x, self.maze_height - 1))
        for y in range(self.maze_height):
            self.walls.add((0, y))
            self.walls.add((self.maze_width - 1, y))

        # Add some internal walls (simple pattern)
        for x in range(5, self.maze_width - 5, 5):
            for y in range(5, self.maze_height - 5, 3):
                if random.random() < 0.5:
                    self.walls.add((x, y))

        # Place pellets everywhere except walls
        for x in range(1, self.maze_width - 1):
            for y in range(1, self.maze_height - 1):
                if (x, y) not in self.walls:
                    self.pellets.add((x, y))

        # Place 4 power pellets in corners
        self.power_pellets = {
            (2, 2),
            (self.maze_width - 3, 2),
            (2, self.maze_height - 3),
            (self.maze_width - 3, self.maze_height - 3)
        }

        # Remove power pellets from regular pellets
        self.pellets -= self.power_pellets

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            state: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.steps += 1
        reward = 0

        # Move player
        new_pos = self._move(self.player_pos, action)

        # Check if move is valid (not into wall)
        if new_pos not in self.walls:
            self.player_pos = new_pos
        else:
            reward -= 1  # Small penalty for hitting walls

        # Check pellet collection
        if self.player_pos in self.pellets:
            self.pellets.remove(self.player_pos)
            self.score += 10
            reward += 10

        # Check power pellet collection
        if self.player_pos in self.power_pellets:
            self.power_pellets.remove(self.player_pos)
            self.score += 50
            reward += 50
            self.powered_up = 40  # 40 steps of power

        # Decay power-up
        if self.powered_up > 0:
            self.powered_up -= 1

        # Move ghosts (simple AI - move toward player)
        new_ghost_positions = []
        for ghost_pos in self.ghost_positions:
            new_ghost_pos = self._move_ghost(ghost_pos)
            new_ghost_positions.append(new_ghost_pos)
        self.ghost_positions = new_ghost_positions

        # Check ghost collision
        for ghost_pos in self.ghost_positions:
            if self._manhattan_distance(self.player_pos, ghost_pos) < 1.5:
                if self.powered_up > 0:
                    # Eat ghost
                    self.score += 200
                    reward += 200
                    # Respawn ghost at corner
                    self.ghost_positions.remove(ghost_pos)
                    self.ghost_positions.append((3, 3))
                else:
                    # Die
                    self.lives -= 1
                    reward -= 500
                    if self.lives > 0:
                        # Respawn player
                        self.player_pos = (self.maze_width // 2, self.maze_height // 2)
                    break

        # Shaped rewards to encourage good behavior
        # 1. Get closer to nearest pellet
        if len(self.pellets) > 0:
            min_pellet_dist = min(self._manhattan_distance(self.player_pos, p) for p in self.pellets)
            if min_pellet_dist < self.last_distance_to_pellet:
                reward += 1  # Moving toward pellet
            self.last_distance_to_pellet = min_pellet_dist

        # 2. Stay away from ghosts when not powered
        if self.powered_up == 0:
            min_ghost_dist = min(self._manhattan_distance(self.player_pos, g) for g in self.ghost_positions)
            if min_ghost_dist < 3:
                reward -= 5  # Danger zone

        # 3. Small time penalty to encourage finishing quickly
        reward -= 0.1

        # Check if episode is done
        done = (
            self.lives <= 0 or                          # Game over
            len(self.pellets) + len(self.power_pellets) == 0 or  # Won
            self.steps >= self.max_steps                # Timeout
        )

        # Info
        info = {
            'score': self.score,
            'lives': self.lives,
            'pellets_remaining': len(self.pellets) + len(self.power_pellets),
            'powered_up': self.powered_up > 0
        }

        return self._get_state(), reward, done, info

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = pos
        if action == 0:  # UP
            return (x, y - 1)
        elif action == 1:  # DOWN
            return (x, y + 1)
        elif action == 2:  # LEFT
            return (x - 1, y)
        elif action == 3:  # RIGHT
            return (x + 1, y)
        return pos

    def _move_ghost(self, ghost_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Simple ghost AI - move toward player"""
        gx, gy = ghost_pos
        px, py = self.player_pos

        # Try to move toward player
        possible_moves = []

        # Calculate which direction gets closer to player
        if px < gx:
            possible_moves.append((gx - 1, gy))  # LEFT
        elif px > gx:
            possible_moves.append((gx + 1, gy))  # RIGHT

        if py < gy:
            possible_moves.append((gx, gy - 1))  # UP
        elif py > gy:
            possible_moves.append((gx, gy + 1))  # DOWN

        # Add random movement sometimes
        if random.random() < 0.2:
            possible_moves.extend([
                (gx, gy - 1),
                (gx, gy + 1),
                (gx - 1, gy),
                (gx + 1, gy)
            ])

        # Filter out walls
        valid_moves = [m for m in possible_moves if m not in self.walls]

        if valid_moves:
            return random.choice(valid_moves)
        return ghost_pos

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_state(self) -> np.ndarray:
        """
        Get current state as feature vector (128 dimensions)
        Matches frontend feature engineering
        """
        features = []

        # Player position (normalized)
        px, py = self.player_pos
        features.extend([
            px / self.maze_width,
            py / self.maze_height
        ])

        # Ghost positions and distances (4 ghosts * 3 features = 12)
        for ghost_pos in self.ghost_positions[:4]:  # Ensure max 4 ghosts
            gx, gy = ghost_pos
            dist = self._manhattan_distance(self.player_pos, ghost_pos)
            features.extend([
                gx / self.maze_width,
                gy / self.maze_height,
                min(dist / 20.0, 1.0)  # Normalized distance
            ])

        # Pad if fewer than 4 ghosts
        while len(features) < 2 + 12:
            features.extend([0, 0, 1.0])

        # Nearest pellet direction and distance (4)
        if self.pellets:
            nearest_pellet = min(self.pellets,
                               key=lambda p: self._manhattan_distance(self.player_pos, p))
            pellet_dist = self._manhattan_distance(self.player_pos, nearest_pellet)
            dx = (nearest_pellet[0] - px) / self.maze_width
            dy = (nearest_pellet[1] - py) / self.maze_height
            features.extend([dx, dy, pellet_dist / 20.0, 1.0])
        else:
            features.extend([0, 0, 0, 0])

        # Power pellet info (4)
        if self.power_pellets:
            nearest_power = min(self.power_pellets,
                              key=lambda p: self._manhattan_distance(self.player_pos, p))
            power_dist = self._manhattan_distance(self.player_pos, nearest_power)
            dx = (nearest_power[0] - px) / self.maze_width
            dy = (nearest_power[1] - py) / self.maze_height
            features.extend([dx, dy, power_dist / 20.0, 1.0])
        else:
            features.extend([0, 0, 0, 0])

        # Game state (6)
        features.extend([
            self.lives / 3.0,
            min(self.score / 1000.0, 1.0),
            len(self.pellets) / 200.0,
            len(self.power_pellets) / 4.0,
            1.0 if self.powered_up > 0 else 0.0,
            self.powered_up / 40.0
        ])

        # Valid moves (4)
        for action in range(4):
            new_pos = self._move(self.player_pos, action)
            features.append(0.0 if new_pos in self.walls else 1.0)

        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)

        return np.array(features[:128], dtype=np.float32)

    def render(self):
        """Optional: Print simple ASCII visualization"""
        grid = [[' ' for _ in range(self.maze_width)] for _ in range(self.maze_height)]

        # Walls
        for x, y in self.walls:
            grid[y][x] = '█'

        # Pellets
        for x, y in self.pellets:
            grid[y][x] = '·'

        # Power pellets
        for x, y in self.power_pellets:
            grid[y][x] = 'O'

        # Ghosts
        for x, y in self.ghost_positions:
            grid[y][x] = 'G' if self.powered_up == 0 else 'g'

        # Player
        px, py = self.player_pos
        grid[py][px] = 'P' if self.powered_up == 0 else 'Ⓟ'

        # Print
        print('\n'.join([''.join(row) for row in grid]))
        print(f"Score: {self.score}, Lives: {self.lives}, Powered: {self.powered_up > 0}")
        print("-" * self.maze_width)


if __name__ == "__main__":
    # Test the environment
    env = PacManEnv(maze_size=(20, 20))
    state = env.reset()

    print(f"State shape: {state.shape}")
    print(f"State sample: {state[:10]}")

    # Play a few random steps
    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done, info = env.step(action)

        print(f"\nStep {i+1}: Action={action}, Reward={reward:.1f}, Done={done}")
        print(f"Info: {info}")

        if done:
            print("Episode finished!")
            break

    print("\n[+] Environment test passed!")
