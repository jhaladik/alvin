"""
DQN Agent with Epsilon-Greedy Exploration
Hybrid approach: Can bootstrap from pre-trained model, then improve via RL
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn_network import DQNNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent for Pac-Man with:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network for stability
    - Can load pre-trained weights for bootstrapping
    """

    def __init__(
        self,
        state_dim=128,
        action_dim=4,
        hidden_dims=[256, 256, 128],
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        device='cpu',
        pretrained_path=None
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_capacity: Replay buffer size
            device: 'cpu' or 'cuda'
            pretrained_path: Path to pre-trained model checkpoint (optional)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Create Q-network and target network
        self.q_network = DQNNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            dropout=0.0  # No dropout during RL training
        ).to(self.device)

        self.target_network = DQNNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            dropout=0.0
        ).to(self.device)

        # Load pre-trained weights if provided (HYBRID APPROACH)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[+] Loading pre-trained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Pre-trained model loaded (bootstrapping from human demonstrations)")
            # Start with lower epsilon since we have good initial policy
            self.epsilon = 0.3
        else:
            print(f"[+] Starting from scratch (no pre-trained model)")

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training statistics
        self.training_steps = 0

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state (numpy array)
            training: If True, use epsilon-greedy; if False, always exploit

        Returns:
            action: Selected action (int)
        """
        # Exploration vs Exploitation
        if training and np.random.rand() < self.epsilon:
            # EXPLORE: Random action
            return np.random.randint(0, self.action_dim)
        else:
            # EXPLOIT: Best action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self, batch_size=64):
        """
        Train on a batch of experiences from replay buffer

        Args:
            batch_size: Number of transitions to sample

        Returns:
            loss: TD loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample random batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute TD loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save agent checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)

    def load(self, path):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_steps = checkpoint.get('training_steps', 0)


if __name__ == "__main__":
    # Test the agent
    print("Testing DQN Agent...")

    # Create agent
    agent = DQNAgent(
        state_dim=128,
        action_dim=4,
        epsilon_start=1.0,
        device='cpu'
    )

    print(f"Agent created")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Q-network params: {sum(p.numel() for p in agent.q_network.parameters()):,}")

    # Test action selection
    state = np.random.randn(128)
    action = agent.select_action(state, training=True)
    print(f"\nSelected action (exploration): {action}")

    action = agent.select_action(state, training=False)
    print(f"Selected action (exploitation): {action}")

    # Test storing transitions and training
    for i in range(100):
        state = np.random.randn(128)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(128)
        done = np.random.rand() < 0.1

        agent.store_transition(state, action, reward, next_state, done)

    print(f"\nReplay buffer size: {len(agent.replay_buffer)}")

    # Train on a batch
    loss = agent.train_step(batch_size=32)
    print(f"Training loss: {loss:.4f}")

    # Test epsilon decay
    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon}")

    print("\n[+] DQN Agent test passed!")
