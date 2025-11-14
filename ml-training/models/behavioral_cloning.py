"""
Behavioral Cloning Model for Pac-Man
Neural network that learns to imitate human gameplay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PacManPolicyNetwork(nn.Module):
    """
    Policy network for imitating human Pac-Man gameplay

    Architecture:
    - Input: 128 features (game state)
    - Hidden Layer 1: 256 neurons + ReLU + Dropout
    - Hidden Layer 2: 256 neurons + ReLU + Dropout
    - Hidden Layer 3: 128 neurons + ReLU + Dropout
    - Output: 4 actions (UP, DOWN, LEFT, RIGHT)
    """

    def __init__(self, input_dim=128, hidden_dims=[256, 256, 128], output_dim=4, dropout=0.3):
        super(PacManPolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, 128) - game state features

        Returns:
            logits: (batch_size, 4) - action logits
        """
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return logits

    def predict_action(self, x, temperature=1.0):
        """
        Predict action with optional temperature sampling

        Args:
            x: (batch_size, 128) or (128,) - game state
            temperature: float - sampling temperature (1.0 = greedy, >1.0 = exploratory)

        Returns:
            action: int - predicted action index
            probs: (4,) - action probabilities
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits / temperature, dim=-1)

            if temperature == 1.0:
                # Greedy
                action = torch.argmax(probs, dim=-1)
            else:
                # Sample
                action = torch.multinomial(probs, 1).squeeze(-1)

            return action.item(), probs.squeeze().cpu().numpy()


class PacManDQN(nn.Module):
    """
    Deep Q-Network for Pac-Man (if you want to do RL instead)

    Architecture:
    - Input: 128 features
    - Hidden: [256, 256, 128]
    - Output: 4 Q-values (one per action)
    """

    def __init__(self, input_dim=128, hidden_dims=[256, 256, 128], output_dim=4):
        super(PacManDQN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, 128) - game state

        Returns:
            q_values: (batch_size, 4) - Q-values for each action
        """
        x = self.layers(x)
        q_values = self.value_head(x)
        return q_values

    def select_action(self, x, epsilon=0.0):
        """
        Select action using epsilon-greedy

        Args:
            x: (128,) - game state
            epsilon: float - exploration rate

        Returns:
            action: int - selected action
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 4, (1,)).item()

        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            q_values = self.forward(x)
            return torch.argmax(q_values).item()


def create_model(model_type='policy', **kwargs):
    """
    Factory function to create models

    Args:
        model_type: 'policy' or 'dqn'
        **kwargs: model parameters

    Returns:
        model: PyTorch model
    """
    if model_type == 'policy':
        return PacManPolicyNetwork(**kwargs)
    elif model_type == 'dqn':
        return PacManDQN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    print()

    # Policy Network
    print("1. Policy Network (Behavioral Cloning)")
    policy = PacManPolicyNetwork()
    print(f"   Parameters: {count_parameters(policy):,}")
    print(f"   Input shape: (batch, 128)")
    print(f"   Output shape: (batch, 4)")

    # Test forward pass
    x = torch.randn(32, 128)
    logits = policy(x)
    print(f"   Test input: {x.shape}")
    print(f"   Test output: {logits.shape}")
    print()

    # DQN
    print("2. DQN Network (Reinforcement Learning)")
    dqn = PacManDQN()
    print(f"   Parameters: {count_parameters(dqn):,}")
    q_values = dqn(x)
    print(f"   Test output: {q_values.shape}")
    print()

    print("✓ Models created successfully!")
