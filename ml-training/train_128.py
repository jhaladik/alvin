"""
Unified DQN Training Script (128-dim Feature Vectors)
Trains on data from Cloudflare KV export
Supports offline DQN training with proper dimension handling
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.dqn_network import DQNNetwork, count_parameters


class DQNDataset(Dataset):
    """DQN Dataset with state-action-reward-next_state tuples"""

    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.transitions[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action])[0],
            torch.FloatTensor([reward])[0],
            torch.FloatTensor(next_state) if next_state is not None else torch.zeros(len(state)),
            torch.FloatTensor([done])[0]
        )


def load_training_data(data_path):
    """Load training data from Cloudflare KV export"""
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r') as f:
        json_data = json.load(f)

    data = json_data['data']
    print(f"[+] Loaded {len(data)} samples")

    # Extract transitions (state, action, reward, next_state)
    transitions = []
    action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    # Detect vector dimension from first sample
    if data:
        first_vector = data[0]['vector']
        vector_dim = len(first_vector)
        print(f"[+] Detected vector dimension: {vector_dim}")
    else:
        raise ValueError("No data found in export file")

    for i in range(len(data) - 1):
        curr = data[i]
        next = data[i + 1]

        state = curr['vector']
        action = action_map[curr['action']]
        reward = curr['reward']
        next_state = next['vector']
        done = False  # We don't have episode boundaries

        # Validate dimensions
        if len(state) != vector_dim or len(next_state) != vector_dim:
            print(f"[WARNING] Skipping sample {i} - dimension mismatch")
            continue

        transitions.append((state, action, reward, next_state, done))

    # Last transition (no next state)
    last = data[-1]
    transitions.append((
        last['vector'],
        action_map[last['action']],
        last['reward'],
        last['vector'],  # Use same state as next_state
        True   # Terminal state
    ))

    print(f"[+] Created {len(transitions)} transitions")

    # Statistics
    rewards = [t[2] for t in transitions]
    print(f"  Reward range: [{min(rewards):.1f}, {max(rewards):.1f}]")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Positive rewards: {sum(r > 0 for r in rewards)} ({100*sum(r > 0 for r in rewards)/len(rewards):.1f}%)")

    return transitions, vector_dim


def compute_q_targets(rewards, next_q_values, dones, gamma=0.99):
    """Compute Q-learning targets using Bellman equation"""
    # For terminal states, Q_target = reward only
    # For non-terminal: Q_target = reward + gamma * max(Q(next_state))
    max_next_q = next_q_values.max(dim=1)[0]
    q_targets = rewards + gamma * max_next_q * (1 - dones)
    return q_targets


def train_epoch(model, dataloader, optimizer, criterion, device, gamma=0.99):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_td_error = 0

    for states, actions, rewards, next_states, dones in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Get current Q-values for taken actions
        q_values = model(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q-values (for target)
        with torch.no_grad():
            next_q_values = model(next_states)
            q_targets = compute_q_targets(rewards, next_q_values, dones, gamma)

        # Compute loss (TD error)
        loss = criterion(q_values_for_actions, q_targets)
        td_error = torch.abs(q_values_for_actions - q_targets).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_td_error += td_error.item()

    return total_loss / len(dataloader), total_td_error / len(dataloader)


def validate(model, dataloader, criterion, device, gamma=0.99):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_td_error = 0

    with torch.no_grad():
        for states, actions, rewards, next_states, dones in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # Get Q-values
            q_values = model(states)
            q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get targets
            next_q_values = model(next_states)
            q_targets = compute_q_targets(rewards, next_q_values, dones, gamma)

            # Compute loss
            loss = criterion(q_values_for_actions, q_targets)
            td_error = torch.abs(q_values_for_actions - q_targets).mean()

            total_loss += loss.item()
            total_td_error += td_error.item()

    return total_loss / len(dataloader), total_td_error / len(dataloader)


def main():
    print("=" * 70)
    print("DQN Training (128-dim Feature Vectors)")
    print("=" * 70)
    print()

    # Hyperparameters
    DATA_PATH = 'data/training_data.json'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    HIDDEN_DIMS = [256, 256, 128]
    GAMMA = 0.99
    VAL_SPLIT = 0.2

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    transitions, input_dim = load_training_data(DATA_PATH)
    dataset = DQNDataset(transitions)

    # Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}\n")

    # Create model
    model = DQNNetwork(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        output_dim=4,
        dropout=0.3
    )
    model = model.to(device)

    print(f"Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {HIDDEN_DIMS}")
    print(f"  Output dim: 4 (actions)")
    print(f"  Parameters: {count_parameters(model):,}\n")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # TensorBoard
    log_dir = f"runs/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}\n")

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    print("=" * 70)
    print("Training...")
    print("=" * 70)

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_td_error = train_epoch(
            model, train_loader, optimizer, criterion, device, GAMMA
        )

        # Validate
        val_loss, val_td_error = validate(
            model, val_loader, criterion, device, GAMMA
        )

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('TD_Error/train', train_td_error, epoch)
        writer.add_scalar('TD_Error/val', val_td_error, epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.2f}, TD: {train_td_error:.2f} | "
              f"Val Loss: {val_loss:.2f}, TD: {val_td_error:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_td_error': val_td_error,
                'train_loss': train_loss,
                'train_td_error': train_td_error,
                'config': {
                    'input_dim': input_dim,  # IMPORTANT: Save input dimension
                    'hidden_dims': HIDDEN_DIMS,
                    'output_dim': 4,
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA
                }
            }, 'checkpoints/best_dqn_model.pth')

            print(f"  → Saved best model (val_loss: {val_loss:.2f})")

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best val loss: {best_val_loss:.2f}")
    print(f"Checkpoint: checkpoints/best_dqn_model.pth")
    print()

    writer.close()


if __name__ == '__main__':
    main()
