"""
Headless Pac-Man Environment for Reinforcement Learning
Simulates the game without rendering for fast training
MATCHES FRONTEND FEATURE ENGINEERING EXACTLY (69 features)
"""
import numpy as np
from typing import Tuple, Dict, List
import random
import math


class PacManEnv:
    """
    Pac-Man environment following Gym-like interface

    State: Game state representation (69 features matching frontend)
    Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Reward: Score changes + shaped rewards
    """

    def __init__(self, maze_size=(20, 20), max_steps=1000):
        self.maze_width = maze_size[0]
        self.maze_height = maze_size[1]
        self.max_steps = max_steps
        self.max_distance = math.sqrt(2) * self.maze_width  # Diagonal

        # Game state
        self.player_pos = None
        self.player_direction = 'RIGHT'  # Track player direction
        self.ghost_positions = []
        self.ghost_scared = []  # Track if each ghost is scared
        self.pellets = set()
        self.power_pellets = set()
        self.walls = set()
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0
        self.total_pellets = 0

        # For tracking
        self.last_distance_to_pellet = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Simple maze layout
        self._create_maze()

        # Place player
        self.player_pos = (self.maze_width // 2, self.maze_height // 2)
        self.player_direction = 'RIGHT'

        # Place ghosts
        self.ghost_positions = [
            (3, 3),
            (self.maze_width - 4, 3),
            (3, self.maze_height - 4),
            (self.maze_width - 4, self.maze_height - 4)
        ]
        self.ghost_scared = [False, False, False, False]

        # Reset game state
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0
        self.total_pellets = len(self.pellets) + len(self.power_pellets)

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

        # Update player direction based on action
        direction_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.player_direction = direction_map[action]

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
            # Make all ghosts scared
            self.ghost_scared = [True] * len(self.ghost_positions)

        # Decay power-up
        if self.powered_up > 0:
            self.powered_up -= 1
            if self.powered_up == 0:
                # Power expired, ghosts no longer scared
                self.ghost_scared = [False] * len(self.ghost_positions)

        # Move ghosts (simple AI - move toward player)
        new_ghost_positions = []
        for i, ghost_pos in enumerate(self.ghost_positions):
            new_ghost_pos = self._move_ghost(ghost_pos)
            new_ghost_positions.append(new_ghost_pos)
        self.ghost_positions = new_ghost_positions

        # Check ghost collision
        for i, ghost_pos in enumerate(self.ghost_positions):
            if self._manhattan_distance(self.player_pos, ghost_pos) < 1.5:
                if self.ghost_scared[i]:
                    # Eat ghost
                    self.score += 200
                    reward += 200
                    # Respawn ghost at corner (no longer scared)
                    self.ghost_positions[i] = (3, 3)
                    self.ghost_scared[i] = False
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

    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_direction_encoding(self, dx: float, dy: float) -> float:
        """Encode direction as continuous value (0-1) based on angle"""
        angle = math.atan2(dy, dx)
        return (angle + math.pi) / (2 * math.pi)

    def _count_pellets_in_directions(self, range_limit: int = 5) -> Dict:
        """Count pellets in each direction within range"""
        counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

        px, py = self.player_pos
        for pellet in self.pellets:
            dx = pellet[0] - px
            dy = pellet[1] - py
            dist = abs(dx) + abs(dy)

            if dist > range_limit:
                continue

            if abs(dy) > abs(dx):
                if dy < 0:
                    counts['up'] += 1
                else:
                    counts['down'] += 1
            else:
                if dx < 0:
                    counts['left'] += 1
                else:
                    counts['right'] += 1

        # Normalize
        total = sum(counts.values())
        if total > 0:
            for key in counts:
                counts[key] /= total

        return counts

    def _get_wall_distance(self, dx: int, dy: int, max_check: int = 5) -> int:
        """Get distance to wall in given direction"""
        px, py = self.player_pos

        for dist in range(1, max_check + 1):
            x = px + dx * dist
            y = py + dy * dist

            # Out of bounds = wall
            if x < 0 or x >= self.maze_width or y < 0 or y >= self.maze_height:
                return dist

            # Check if wall
            if (x, y) in self.walls:
                return dist

        return max_check

    def _is_direction_safe(self, dx: int, dy: int) -> bool:
        """Check if direction is safe (no wall, no nearby non-scared ghost)"""
        px, py = self.player_pos
        new_x, new_y = px + dx, py + dy

        # Check bounds
        if new_x < 0 or new_x >= self.maze_width or new_y < 0 or new_y >= self.maze_height:
            return False

        # Check walls
        if (new_x, new_y) in self.walls:
            return False

        # Check ghosts (not safe if non-scared ghost within 2 tiles)
        for i, ghost_pos in enumerate(self.ghost_positions):
            if self.ghost_scared[i]:
                continue  # Scared ghosts are safe

            dist = abs(ghost_pos[0] - new_x) + abs(ghost_pos[1] - new_y)
            if dist <= 2:
                return False

        return True

    def _get_state(self) -> np.ndarray:
        """
        Get current state as feature vector (69 features + padding to 128)
        EXACTLY MATCHES frontend/feature-engineering.js
        """
        features = []
        px, py = self.player_pos

        # ===== BASIC PLAYER INFO (6 features) =====
        features.extend([
            px / self.maze_width,  # player_x
            py / self.maze_height  # player_y
        ])

        # Player direction (one-hot encoded)
        features.extend([
            1.0 if self.player_direction == 'UP' else 0.0,
            1.0 if self.player_direction == 'DOWN' else 0.0,
            1.0 if self.player_direction == 'LEFT' else 0.0,
            1.0 if self.player_direction == 'RIGHT' else 0.0
        ])

        # ===== GHOST INFORMATION (24 features: 4 ghosts × 6) =====
        ghost_data = []
        for i, ghost_pos in enumerate(self.ghost_positions):
            gx, gy = ghost_pos
            dx = gx - px
            dy = gy - py
            euclidean = self._euclidean_distance(self.player_pos, ghost_pos)
            manhattan = self._manhattan_distance(self.player_pos, ghost_pos)
            scared = self.ghost_scared[i] if i < len(self.ghost_scared) else False

            ghost_data.append({
                'distance': euclidean,
                'manhattan': manhattan,
                'dx': dx,
                'dy': dy,
                'scared': scared
            })

        # Sort by distance (closest first)
        ghost_data.sort(key=lambda g: g['distance'])

        # Extract features for up to 4 ghosts
        for i in range(4):
            if i < len(ghost_data):
                g = ghost_data[i]
                features.extend([
                    g['distance'] / self.max_distance,  # Normalized Euclidean
                    g['manhattan'] / (self.maze_width * 2),  # Normalized Manhattan
                    g['dx'] / self.maze_width,  # Normalized delta X
                    g['dy'] / self.maze_height,  # Normalized delta Y
                    1.0 if g['scared'] else 0.0,  # Is scared?
                    self._get_direction_encoding(g['dx'], g['dy'])  # Direction
                ])
            else:
                # Padding for missing ghosts
                features.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # ===== PELLET INFORMATION (8 features) =====
        if self.pellets:
            # Find nearest pellet
            nearest = min(self.pellets, key=lambda p: self._euclidean_distance(self.player_pos, p))
            dx = nearest[0] - px
            dy = nearest[1] - py
            dist = self._euclidean_distance(self.player_pos, nearest)

            # Count pellets in each direction
            pellet_density = self._count_pellets_in_directions(5)

            features.extend([
                dist / self.max_distance,  # nearest_pellet_dist
                dx / self.maze_width,  # nearest_pellet_dx
                dy / self.maze_height,  # nearest_pellet_dy
                self._get_direction_encoding(dx, dy),  # nearest_pellet_dir
                pellet_density['up'],  # pellets_up
                pellet_density['down'],  # pellets_down
                pellet_density['left'],  # pellets_left
                pellet_density['right']  # pellets_right
            ])
        else:
            features.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ===== POWER PELLET INFORMATION (6 features) =====
        if self.power_pellets:
            nearest = min(self.power_pellets, key=lambda p: self._euclidean_distance(self.player_pos, p))
            dx = nearest[0] - px
            dy = nearest[1] - py
            dist = self._euclidean_distance(self.player_pos, nearest)

            features.extend([
                dist / self.max_distance,  # nearest_power_dist
                dx / self.maze_width,  # nearest_power_dx
                dy / self.maze_height,  # nearest_power_dy
                self._get_direction_encoding(dx, dy),  # nearest_power_dir
                len(self.power_pellets) / 4.0,  # power_pellets_remaining
                0.0 if self.powered_up > 0 else 1.0  # need_power
            ])
        else:
            features.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # ===== WALL/NAVIGATION INFORMATION (8 features) =====
        # 4 cardinal directions
        features.append(self._get_wall_distance(0, -1) / 5.0)  # wall_up
        features.append(self._get_wall_distance(0, 1) / 5.0)   # wall_down
        features.append(self._get_wall_distance(-1, 0) / 5.0)  # wall_left
        features.append(self._get_wall_distance(1, 0) / 5.0)   # wall_right

        # 4 diagonals
        features.append(self._get_wall_distance(-1, -1) / 5.0)  # wall_up_left
        features.append(self._get_wall_distance(1, -1) / 5.0)   # wall_up_right
        features.append(self._get_wall_distance(-1, 1) / 5.0)   # wall_down_left
        features.append(self._get_wall_distance(1, 1) / 5.0)    # wall_down_right

        # ===== TACTICAL SITUATION (12 features) =====
        # Danger level (how close is nearest non-scared ghost?)
        min_danger_dist = float('inf')
        for i, ghost_pos in enumerate(self.ghost_positions):
            if not self.ghost_scared[i]:
                dist = self._euclidean_distance(self.player_pos, ghost_pos)
                min_danger_dist = min(min_danger_dist, dist)
        danger_level = 1.0 - min(min_danger_dist / 10.0, 1.0) if min_danger_dist != float('inf') else 0.0
        features.append(danger_level)

        # Safe directions count
        safe_count = sum([
            self._is_direction_safe(0, -1),  # UP
            self._is_direction_safe(0, 1),   # DOWN
            self._is_direction_safe(-1, 0),  # LEFT
            self._is_direction_safe(1, 0)    # RIGHT
        ])
        features.append(safe_count / 4.0)  # safe_directions

        # Trapped indicator
        features.append(1.0 if safe_count <= 1 else 0.0)  # trapped

        # Ghost convergence (are ghosts in multiple quadrants?)
        quadrants = {'nw': 0, 'ne': 0, 'sw': 0, 'se': 0}
        for i, ghost_pos in enumerate(self.ghost_positions):
            if self.ghost_scared[i]:
                continue
            dx = ghost_pos[0] - px
            dy = ghost_pos[1] - py
            if dx < 0 and dy < 0:
                quadrants['nw'] = 1
            elif dx >= 0 and dy < 0:
                quadrants['ne'] = 1
            elif dx < 0 and dy >= 0:
                quadrants['sw'] = 1
            else:
                quadrants['se'] = 1
        convergence = sum(quadrants.values()) / 4.0
        features.append(convergence)  # ghost_convergence

        # Centrality (distance from center, normalized)
        center_x, center_y = self.maze_width / 2, self.maze_height / 2
        dist_from_center = math.sqrt((px - center_x)**2 + (py - center_y)**2)
        centrality = 1.0 - (dist_from_center / (self.max_distance / 2))
        features.append(max(0.0, min(1.0, centrality)))  # centrality

        # Progress (pellet collection ratio)
        pellets_left = len(self.pellets) + len(self.power_pellets)
        progress = 1.0 - (pellets_left / max(self.total_pellets, 1))
        features.append(progress)  # progress

        # Risk/reward (pellets near ghosts)
        risky_pellets = 0
        for pellet in self.pellets:
            dist_to_player = self._manhattan_distance(self.player_pos, pellet)
            if dist_to_player <= 5:
                for i, ghost_pos in enumerate(self.ghost_positions):
                    if self.ghost_scared[i]:
                        continue
                    dist_to_ghost = self._manhattan_distance(pellet, ghost_pos)
                    if dist_to_ghost <= 3:
                        risky_pellets += 1
                        break
        risk_reward = min(risky_pellets / 5.0, 1.0)
        features.append(risk_reward)  # risk_reward

        # Scared ghost nearby
        has_scared_nearby = False
        for i, ghost_pos in enumerate(self.ghost_scared):
            if self.ghost_scared[i]:
                dist = self._manhattan_distance(self.player_pos, self.ghost_positions[i])
                if dist <= 8:
                    has_scared_nearby = True
                    break
        features.append(1.0 if has_scared_nearby else 0.0)  # scared_ghost_nearby

        # Optimal direction (simplified heuristic)
        optimal = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}

        # Avoid non-scared ghosts
        for i, ghost_pos in enumerate(self.ghost_positions):
            if self.ghost_scared[i]:
                continue
            dx = ghost_pos[0] - px
            dy = ghost_pos[1] - py
            if abs(dy) > abs(dx):
                if dy < 0:
                    optimal['down'] += 0.25  # Ghost above, go down
                else:
                    optimal['up'] += 0.25
            else:
                if dx < 0:
                    optimal['right'] += 0.25
                else:
                    optimal['left'] += 0.25

        # Seek nearest pellet
        if self.pellets:
            nearest = min(self.pellets, key=lambda p: self._euclidean_distance(self.player_pos, p))
            dx = nearest[0] - px
            dy = nearest[1] - py
            if abs(dy) > abs(dx):
                if dy < 0:
                    optimal['up'] += 0.5
                else:
                    optimal['down'] += 0.5
            else:
                if dx < 0:
                    optimal['left'] += 0.5
                else:
                    optimal['right'] += 0.5

        # Normalize
        total = sum(optimal.values())
        if total > 0:
            for key in optimal:
                optimal[key] /= total

        features.extend([
            optimal['up'],  # optimal_up
            optimal['down'],  # optimal_down
            optimal['left'],  # optimal_left
            optimal['right']  # optimal_right
        ])

        # ===== GAME STATE (5 features) =====
        features.extend([
            1.0 if self.powered_up > 0 else 0.0,  # power_mode
            self.powered_up / 50.0,  # power_timer (normalized)
            self.lives / 3.0,  # lives
            min(self.score / 10000.0, 1.0),  # score (capped)
            pellets_left / max(self.total_pellets, 1)  # pellets_ratio
        ])

        # Total: 2+4+24+8+6+8+12+5 = 69 features
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)

        return np.array(features[:128], dtype=np.float32)


if __name__ == "__main__":
    # Test the environment
    env = PacManEnv(maze_size=(20, 20))
    state = env.reset()

    print(f"State shape: {state.shape}")
    print(f"State first 69 features: {state[:69]}")
    print(f"Non-zero features: {np.count_nonzero(state)}")

    # Play a few random steps
    for i in range(10):
        action = random.randint(0, 3)
        state, reward, done, info = env.step(action)

        print(f"\nStep {i+1}: Action={action}, Reward={reward:.1f}, Done={done}")
        print(f"Info: {info}")
        print(f"Direction: {env.player_direction}")

        if done:
            print("Episode finished!")
            break

    print("\n[+] Environment test passed!")
    print(f"[+] Feature vector has {len(state)} dimensions")
