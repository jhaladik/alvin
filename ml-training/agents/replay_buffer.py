"""
Experience Replay Buffer for DQN
Stores transitions and samples random batches for training
"""
import numpy as np
import random
from collections import deque
from typing import List, Tuple


class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) transitions
    Samples random batches for training to break correlation
    """

    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions

        Returns:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, state_dim)
            dones: (batch_size,)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)

    # Add some dummy transitions
    for i in range(100):
        state = np.random.randn(128)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(128)
        done = np.random.rand() < 0.1

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(32)

    print(f"Sampled batch:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")

    print("\n[+] Replay buffer test passed!")
