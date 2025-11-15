"""
Headless RL Training (Phase 2: Self-Play)
AI plays Pac-Man thousands of times and learns from experience
Uses trained DQN model as starting point and improves via RL
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.dqn_network import DQNNetwork, count_parameters
from environment.pacman_env import PacManEnv


class ReplayBuffer:
    """Experience replay buffer for stable training"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train_rl(
    env,
    model,
    target_model,
    optimizer,
    replay_buffer,
    device,
    batch_size=64,
    gamma=0.99
):
    """Train DQN using experience replay"""

    if len(replay_buffer) < batch_size:
        return None

    # Sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Current Q-values
    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values (using target network for stability)
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # Gradient clipping
    optimizer.step()

    return loss.item()


def play_episode(env, model, device, epsilon=0.1, max_steps=1000):
    """Play one episode and collect experience"""

    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    experiences = []

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()  # Exploit

        # Take action
        next_state, reward, done, info = env.step(action)

        # Store experience
        experiences.append((state, action, reward, next_state, done))

        episode_reward += reward
        episode_steps += 1
        state = next_state

        if done:
            break

    return experiences, episode_reward, episode_steps, info


def main():
    print("=" * 70)
    print("Headless RL Training (Phase 2: Self-Play)")
    print("=" * 70)
    print()

    # Hyperparameters
    NUM_EPISODES = 100000
    MAX_STEPS_PER_EPISODE = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9999  # Slower decay for longer training
    TARGET_UPDATE_FREQ = 100  # Update target network every N episodes
    REPLAY_BUFFER_SIZE = 100000
    MIN_REPLAY_SIZE = 1000

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create environment
    print("Creating Pac-Man environment...")
    env = PacManEnv(maze_size=(20, 20), max_steps=MAX_STEPS_PER_EPISODE)
    state_dim = 128  # Environment returns 128-dim states
    print(f"State dimension: {state_dim}\n")

    # Load pre-trained model (Phase 1) or create new one
    checkpoint_path = 'checkpoints/best_dqn_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        input_dim = checkpoint['config'].get('input_dim', 128)

        model = DQNNetwork(
            input_dim=input_dim,
            hidden_dims=checkpoint['config']['hidden_dims'],
            output_dim=4,
            dropout=0.0  # No dropout during RL training
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_or_episode = checkpoint.get('episode', checkpoint.get('epoch', 'N/A'))
        print(f"[+] Loaded pre-trained model (episode/epoch {epoch_or_episode})")
    else:
        print("No pre-trained model found. Starting from scratch...")
        model = DQNNetwork(input_dim=state_dim, hidden_dims=[256, 256, 128], output_dim=4, dropout=0.0)

    model = model.to(device)

    # Create target network (for stable Q-learning)
    target_model = DQNNetwork(
        input_dim=state_dim,
        hidden_dims=[256, 256, 128],
        output_dim=4,
        dropout=0.0
    )
    target_model.load_state_dict(model.state_dict())
    target_model = target_model.to(device)

    print(f"Model parameters: {count_parameters(model):,}\n")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    # Training loop
    epsilon = EPSILON_START
    best_reward = -float('inf')
    rewards_history = []

    print("=" * 70)
    print("Starting RL Training (AI Self-Play)...")
    print("=" * 70)
    print()

    for episode in range(NUM_EPISODES):
        # Play episode
        experiences, episode_reward, episode_steps, info = play_episode(
            env, model, device, epsilon, MAX_STEPS_PER_EPISODE
        )

        # Add experiences to replay buffer
        for exp in experiences:
            replay_buffer.push(*exp)

        # Train if enough experiences
        if len(replay_buffer) >= MIN_REPLAY_SIZE:
            loss = train_rl(
                env, model, target_model, optimizer, replay_buffer,
                device, BATCH_SIZE, GAMMA
            )
        else:
            loss = None

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Track rewards
        rewards_history.append(episode_reward)
        avg_reward_100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else episode_reward

        # Log progress
        if (episode + 1) % 100 == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "0.0000"
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg (100): {avg_reward_100:.1f} | "
                  f"Steps: {episode_steps} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Loss: {loss_str} | "
                  f"Score: {info.get('score', 0)}")

        # Save best model
        if avg_reward_100 > best_reward:
            best_reward = avg_reward_100

            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward_100,
                'epsilon': epsilon,
                'config': {
                    'input_dim': state_dim,
                    'hidden_dims': [256, 256, 128],
                    'output_dim': 4,
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA
                }
            }, 'checkpoints/best_rl_model.pth')

            print(f"  [SAVED] Best RL model (avg_reward: {avg_reward_100:.1f})")

    print()
    print("=" * 70)
    print("RL Training Complete!")
    print("=" * 70)
    print(f"Best avg reward (100 episodes): {best_reward:.1f}")
    print(f"Final epsilon: {epsilon:.3f}")
    print(f"Checkpoint: checkpoints/best_rl_model.pth")
    print()
    print("Next steps:")
    print("1. Replace best_dqn_model.pth with best_rl_model.pth")
    print("2. Deploy to HF Spaces")
    print()


if __name__ == '__main__':
    main()
