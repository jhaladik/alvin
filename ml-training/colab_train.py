"""
GPU Training Script for Google Colab (FIXED MAZE VERSION)
Upload this file to Colab and run!

UPDATED: Now uses FIXED maze layout (same maze every episode)
This provides consistent difficulty and clearer learning curves.

Previous version used random mazes which caused high variance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
from tqdm import tqdm
import os


# ===== DQN Network =====
class DQNNetwork(nn.Module):
    """Deep Q-Network for Pac-Man"""

    def __init__(self, input_dim=128, hidden_dims=None, output_dim=4, dropout=0.0):
        super(DQNNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


# ===== Pac-Man Environment =====
class PacManEnv:
    """Simplified Pac-Man environment for RL training"""

    def __init__(self, maze_size=(20, 20), max_steps=1000):
        self.maze_width = maze_size[0]
        self.maze_height = maze_size[1]
        self.max_steps = max_steps
        self.max_distance = math.sqrt(2) * self.maze_width

        self.player_pos = None
        self.player_direction = 'RIGHT'
        self.ghost_positions = []
        self.ghost_scared = []
        self.pellets = set()
        self.power_pellets = set()
        self.walls = set()
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0
        self.total_pellets = 0
        self.last_distance_to_pellet = 0

    def reset(self):
        self._create_maze()
        self.player_pos = (self.maze_width // 2, self.maze_height // 2)
        self.player_direction = 'RIGHT'
        self.ghost_positions = [(3, 3), (self.maze_width - 4, 3),
                                 (3, self.maze_height - 4), (self.maze_width - 4, self.maze_height - 4)]
        self.ghost_scared = [False] * 4
        self.score = 0
        self.lives = 3
        self.steps = 0
        self.powered_up = 0
        self.total_pellets = len(self.pellets) + len(self.power_pellets)
        return self._get_state()

    def _create_maze(self):
        """Create FIXED maze layout (not random) for consistent learning"""
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

        # FIXED internal walls (classic Pac-Man inspired pattern)
        # Horizontal walls
        for x in range(3, 8):
            self.walls.add((x, 3))
            self.walls.add((x, self.maze_height - 4))
        for x in range(self.maze_width - 8, self.maze_width - 3):
            self.walls.add((x, 3))
            self.walls.add((x, self.maze_height - 4))

        # Vertical walls
        for y in range(5, 10):
            self.walls.add((5, y))
            self.walls.add((self.maze_width - 6, y))
        for y in range(self.maze_height - 10, self.maze_height - 5):
            self.walls.add((5, y))
            self.walls.add((self.maze_width - 6, y))

        # Center ghost house (with opening at top for Pac-Man to exit)
        center_x = self.maze_width // 2
        center_y = self.maze_height // 2

        # Bottom wall
        for x in range(center_x - 2, center_x + 3):
            self.walls.add((x, center_y + 2))

        # Left wall
        for y in range(center_y - 2, center_y + 3):
            self.walls.add((center_x - 2, y))

        # Right wall
        for y in range(center_y - 2, center_y + 3):
            self.walls.add((center_x + 2, y))

        # Top wall with opening in the middle (Pac-Man can exit here)
        self.walls.add((center_x - 2, center_y - 2))
        self.walls.add((center_x - 1, center_y - 2))
        # center_x is open (exit)
        self.walls.add((center_x + 1, center_y - 2))
        self.walls.add((center_x + 2, center_y - 2))

        # Corner blocks
        for dx in [0, 1]:
            for dy in [0, 1]:
                self.walls.add((3 + dx, 5 + dy))
                self.walls.add((self.maze_width - 4 - dx, 5 + dy))
                self.walls.add((3 + dx, self.maze_height - 6 - dy))
                self.walls.add((self.maze_width - 4 - dx, self.maze_height - 6 - dy))

        # Place pellets everywhere except walls
        for x in range(1, self.maze_width - 1):
            for y in range(1, self.maze_height - 1):
                if (x, y) not in self.walls:
                    self.pellets.add((x, y))

        # Place 4 power pellets in corners (FIXED positions)
        self.power_pellets = {(2, 2), (self.maze_width - 3, 2),
                              (2, self.maze_height - 3), (self.maze_width - 3, self.maze_height - 3)}
        self.pellets -= self.power_pellets

    def step(self, action):
        self.steps += 1
        reward = 0  # Start with neutral reward

        direction_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.player_direction = direction_map[action]

        new_pos = self._move(self.player_pos, action)
        if new_pos not in self.walls:
            self.player_pos = new_pos
            reward += 0.5  # Small reward for valid movement
        else:
            reward -= 2  # Penalty for hitting walls

        if self.player_pos in self.pellets:
            self.pellets.remove(self.player_pos)
            self.score += 10
            reward += 25  # Increased from +10 - pellets are important!
            # Progress bonus
            pellets_collected = self.total_pellets - (len(self.pellets) + len(self.power_pellets))
            progress = pellets_collected / max(1, self.total_pellets)
            reward += progress * 20  # Bonus for collecting more of the maze

        if self.player_pos in self.power_pellets:
            self.power_pellets.remove(self.player_pos)
            self.score += 50
            reward += 100  # Increased from +50 - power pellets are strategic!
            self.powered_up = 40
            self.ghost_scared = [True] * 4

        if self.powered_up > 0:
            self.powered_up -= 1
            if self.powered_up == 0:
                self.ghost_scared = [False] * 4

        new_ghost_positions = []
        for ghost_pos in self.ghost_positions:
            new_ghost_positions.append(self._move_ghost(ghost_pos))
        self.ghost_positions = new_ghost_positions

        for i, ghost_pos in enumerate(self.ghost_positions):
            if self._manhattan_distance(self.player_pos, ghost_pos) < 1.5:
                if self.ghost_scared[i]:
                    self.score += 200
                    reward += 200
                    self.ghost_positions[i] = (3, 3)
                    self.ghost_scared[i] = False
                else:
                    self.lives -= 1
                    reward -= 100  # REDUCED from -500 to -100
                    if self.lives > 0:
                        self.player_pos = (self.maze_width // 2, self.maze_height // 2)
                    break

        # Shaped rewards to encourage good behavior

        # 1. Survival bonus - staying alive is good!
        reward += 1.0

        # 2. Get closer to nearest pellet
        if len(self.pellets) > 0:
            min_pellet_dist = min(self._manhattan_distance(self.player_pos, p) for p in self.pellets)
            if min_pellet_dist < self.last_distance_to_pellet:
                reward += 3  # Increased from +1
            elif min_pellet_dist > self.last_distance_to_pellet:
                reward -= 1  # Moving away is bad
            self.last_distance_to_pellet = min_pellet_dist

        # 3. Ghost interaction rewards
        min_ghost_dist = min(self._manhattan_distance(self.player_pos, g) for g in self.ghost_positions)

        if self.powered_up > 0:
            # When powered, encourage hunting ghosts
            if min_ghost_dist < 3:
                reward += 5  # Get close to ghosts when powered!
            if min_ghost_dist < 5:
                reward += 2  # Chase them!
        else:
            # When not powered, penalize being too close to ghosts
            if min_ghost_dist < 2:
                reward -= 10  # Very dangerous!
            elif min_ghost_dist < 4:
                reward -= 3  # Danger zone
            else:
                reward += 1  # Safe distance is good

        done = (self.lives <= 0 or
                len(self.pellets) + len(self.power_pellets) == 0 or
                self.steps >= self.max_steps)

        info = {
            'score': self.score,
            'lives': self.lives,
            'pellets_remaining': len(self.pellets) + len(self.power_pellets),
            'powered_up': self.powered_up > 0
        }

        return self._get_state(), reward, done, info

    def _move(self, pos, action):
        x, y = pos
        if action == 0: return (x, y - 1)
        elif action == 1: return (x, y + 1)
        elif action == 2: return (x - 1, y)
        elif action == 3: return (x + 1, y)
        return pos

    def _move_ghost(self, ghost_pos):
        gx, gy = ghost_pos
        px, py = self.player_pos

        possible_moves = []
        if px < gx: possible_moves.append((gx - 1, gy))
        elif px > gx: possible_moves.append((gx + 1, gy))
        if py < gy: possible_moves.append((gx, gy - 1))
        elif py > gy: possible_moves.append((gx, gy + 1))

        if random.random() < 0.2:
            possible_moves.extend([(gx, gy - 1), (gx, gy + 1), (gx - 1, gy), (gx + 1, gy)])

        valid_moves = [m for m in possible_moves if m not in self.walls]
        return random.choice(valid_moves) if valid_moves else ghost_pos

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_state(self):
        """Generate 128-dim feature vector (simplified)"""
        features = []
        px, py = self.player_pos

        features.extend([px / self.maze_width, py / self.maze_height])
        features.extend([
            1.0 if self.player_direction == 'UP' else 0.0,
            1.0 if self.player_direction == 'DOWN' else 0.0,
            1.0 if self.player_direction == 'LEFT' else 0.0,
            1.0 if self.player_direction == 'RIGHT' else 0.0
        ])

        for ghost_pos in self.ghost_positions:
            dx = (ghost_pos[0] - px) / self.maze_width
            dy = (ghost_pos[1] - py) / self.maze_height
            dist = math.sqrt(dx**2 + dy**2)
            manhattan = self._manhattan_distance(self.player_pos, ghost_pos) / (self.maze_width * 2)
            features.extend([dist, manhattan, dx, dy, 0.0, 0.0])

        if self.pellets:
            nearest = min(self.pellets, key=lambda p: math.sqrt((p[0]-px)**2 + (p[1]-py)**2))
            dx = (nearest[0] - px) / self.maze_width
            dy = (nearest[1] - py) / self.maze_height
            dist = math.sqrt(dx**2 + dy**2)
            features.extend([dist, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([1.0] * 8)

        if self.power_pellets:
            nearest = min(self.power_pellets, key=lambda p: math.sqrt((p[0]-px)**2 + (p[1]-py)**2))
            dx = (nearest[0] - px) / self.maze_width
            dy = (nearest[1] - py) / self.maze_height
            dist = math.sqrt(dx**2 + dy**2)
            features.extend([dist, dx, dy, 0.0, len(self.power_pellets)/4.0, 0.0])
        else:
            features.extend([1.0] * 6)

        features.extend([0.5] * 8)  # Walls
        features.extend([0.5] * 12)  # Tactical

        pellets_left = len(self.pellets) + len(self.power_pellets)
        features.extend([
            1.0 if self.powered_up > 0 else 0.0,
            self.powered_up / 50.0,
            self.lives / 3.0,
            min(self.score / 10000.0, 1.0),
            pellets_left / max(self.total_pellets, 1)
        ])

        while len(features) < 128:
            features.append(0.0)

        return np.array(features[:128], dtype=np.float32)


# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ===== Training Functions =====
def train_step(model, target_model, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def play_episode(env, model, epsilon, max_steps, device):
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    experiences = []

    for step in range(max_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, info = env.step(action)
        experiences.append((state, action, reward, next_state, done))

        episode_reward += reward
        episode_steps += 1
        state = next_state

        if done:
            break

    return experiences, episode_reward, episode_steps, info


# ===== Main Training =====
def main():
    print("="*70)
    print("Alvin Pac-Man - GPU Training (Google Colab)")
    print("="*70)
    print()

    # Configuration
    NUM_EPISODES = 20000  # Reduced from 100k for faster training
    MAX_STEPS_PER_EPISODE = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.999  # Faster decay for shorter training (20k episodes)
    TARGET_UPDATE_FREQ = 100
    REPLAY_BUFFER_SIZE = 100000
    MIN_REPLAY_SIZE = 1000

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Check for checkpoint
    checkpoint_files = ['best_rl_model.pth', 'best_dqn_model.pth', 'checkpoint.pth']
    checkpoint_path = None
    for filename in checkpoint_files:
        if os.path.exists(filename):
            checkpoint_path = filename
            break

    # Load checkpoint to detect architecture
    start_episode = 0
    epsilon = EPSILON_START
    best_reward = -float('inf')
    has_dropout = False

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Detect dropout by checking layer count
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        max_layer_idx = max([int(k.split('.')[1]) for k in state_dict_keys if k.startswith('network.') and len(k.split('.')) > 1])
        has_dropout = max_layer_idx > 6

        print(f"  Architecture: {'WITH dropout' if has_dropout else 'NO dropout'}")
        print(f"  Max layer index: {max_layer_idx}")

    # Create models with correct architecture
    dropout_value = 0.3 if has_dropout else 0.0
    print(f"  Creating model with dropout={dropout_value}")

    model = DQNNetwork(input_dim=128, hidden_dims=[256, 256, 128], output_dim=4, dropout=dropout_value)
    target_model = DQNNetwork(input_dim=128, hidden_dims=[256, 256, 128], output_dim=4, dropout=dropout_value)

    # Load checkpoint weights
    if checkpoint_path:
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'episode' in checkpoint:
            start_episode = checkpoint['episode']
            epsilon = checkpoint.get('epsilon', EPSILON_START)
            print(f"  Resuming from episode: {start_episode:,}")
            print(f"  Avg reward: {checkpoint.get('avg_reward', 'N/A')}")
            print(f"  Epsilon: {epsilon:.4f}")
        elif 'epoch' in checkpoint:
            print(f"  Loaded DQN checkpoint (epoch {checkpoint['epoch']})")

        print("✅ Checkpoint loaded successfully")
    else:
        print("ℹ️  No checkpoint found. Starting from scratch.")

    print()

    # Move to device
    model = model.to(device)
    target_model.load_state_dict(model.state_dict())
    target_model = target_model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Environment
    env = PacManEnv(maze_size=(20, 20), max_steps=MAX_STEPS_PER_EPISODE)

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("="*70)
    print("Starting Training...")
    print("="*70)
    print()

    rewards_history = []

    for episode in tqdm(range(start_episode, NUM_EPISODES), desc="Training"):
        # Play episode
        experiences, episode_reward, episode_steps, info = play_episode(
            env, model, epsilon, MAX_STEPS_PER_EPISODE, device
        )

        # Add to replay buffer
        for exp in experiences:
            replay_buffer.push(*exp)

        # Train
        if len(replay_buffer) >= MIN_REPLAY_SIZE:
            loss = train_step(model, target_model, optimizer, replay_buffer, BATCH_SIZE, GAMMA, device)
        else:
            loss = None

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Track rewards
        rewards_history.append(episode_reward)
        avg_reward_100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else episode_reward

        # Log
        if (episode + 1) % 100 == 0:
            loss_str = f"{loss:.2f}" if loss else "N/A"
            print(f"Ep {episode+1:,} | R: {episode_reward:.1f} | Avg100: {avg_reward_100:.1f} | "
                  f"Steps: {episode_steps} | ε: {epsilon:.3f} | Loss: {loss_str} | Score: {info['score']}")

        # Save best model
        if avg_reward_100 > best_reward:
            best_reward = avg_reward_100

            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward_100,
                'epsilon': epsilon,
                'config': {
                    'input_dim': 128,
                    'hidden_dims': [256, 256, 128],
                    'output_dim': 4,
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA
                }
            }, 'best_rl_model_colab.pth')

            if (episode + 1) % 1000 == 0:
                print(f"  ✅ Saved (avg_reward: {avg_reward_100:.1f})")

    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best avg reward: {best_reward:.1f}")
    print(f"Final epsilon: {epsilon:.3f}")
    print(f"Model saved: best_rl_model_colab.pth")


if __name__ == '__main__':
    main()
