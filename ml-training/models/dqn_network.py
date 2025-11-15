"""
DQN (Deep Q-Network) Model
Learns Q-values: Q(state, action) = expected future reward
"""
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Pac-Man
    Input: 128-dim feature vector (state)
    Output: 4 Q-values (one for each action)
    """

    def __init__(self, input_dim=128, hidden_dims=None, output_dim=4, dropout=0.3):
        super(DQNNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer: Q-values for each action
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, input_dim) feature vectors
        Returns:
            (batch_size, output_dim) Q-values for each action
        """
        return self.network(x)

    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy
        Args:
            state: (input_dim,) single state vector
            epsilon: exploration rate (0 = greedy, 1 = random)
        Returns:
            action_idx: int (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        if torch.rand(1).item() < epsilon:
            # Explore: random action
            return torch.randint(0, 4, (1,)).item()
        else:
            # Exploit: best action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing DQN Network...")

    model = DQNNetwork(input_dim=128, hidden_dims=[256, 256, 128], output_dim=4)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 128)
    q_values = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0]}")

    # Test action selection
    state = torch.randn(128)
    action_greedy = model.get_action(state, epsilon=0.0)
    action_explore = model.get_action(state, epsilon=1.0)

    print(f"Greedy action: {action_greedy}")
    print(f"Random action: {action_explore}")

    print("\n[+] DQN Network test passed!")
