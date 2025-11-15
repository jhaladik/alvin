"""
DQN Training Script
Trains using rewards, not just actions!
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn_network import DQNNetwork, count_parameters


class DQNDataset(Dataset):
    """
    DQN Dataset with state-action-reward-next_state tuples
    """

    def __init__(self, transitions):
        """
        Args:
            transitions: list of (state, action, reward, next_state, done) tuples
        """
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.transitions[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action])[0],
            torch.FloatTensor([reward])[0],
            torch.FloatTensor(next_state) if next_state is not None else torch.zeros(768),
            torch.FloatTensor([done])[0]
        )


def load_training_data(data_path):
    """Load training data from JSON"""
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r') as f:
        json_data = json.load(f)

    data = json_data['data']
    print(f"[+] Loaded {len(data)} samples")

    # Extract transitions (state, action, reward, next_state)
    transitions = []
    action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    for i in range(len(data) - 1):
        curr = data[i]
        next = data[i + 1]

        state = curr['vector']
        action = action_map[curr['action']]
        reward = curr['reward']
        next_state = next['vector']
        done = False  # We don't have episode boundaries, so assume not terminal

        transitions.append((state, action, reward, next_state, done))

    # Last transition (no next state)
    last = data[-1]
    transitions.append((
        last['vector'],
        action_map[last['action']],
        last['reward'],
        None,  # No next state
        True   # Terminal state
    ))

    print(f"[+] Created {len(transitions)} transitions")

    # Statistics
    rewards = [t[2] for t in transitions]
    print(f"  Reward range: [{min(rewards):.1f}, {max(rewards):.1f}]")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Positive rewards: {sum(r > 0 for r in rewards)} ({100*sum(r > 0 for r in rewards)/len(rewards):.1f}%)")

    return transitions


def compute_q_targets(rewards, next_q_values, dones, gamma=0.99):
    """
    Compute Q-learning targets using Bellman equation
    Q_target = reward + gamma * max(Q(next_state, action'))
    """
    # For terminal states, Q_target = reward only
    # For non-terminal states, Q_target = reward + gamma * max(next_Q)
    targets = rewards + gamma * next_q_values.max(dim=1)[0] * (1 - dones)
    return targets


def train_epoch(model, target_model, train_loader, criterion, optimizer, device, gamma=0.99):
    """Train for one epoch using DQN"""
    model.train()
    target_model.eval()

    total_loss = 0
    total_td_error = 0

    pbar = tqdm(train_loader, desc="Training")
    for states, actions, rewards, next_states, dones in pbar:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Current Q-values: Q(s, a)
        q_values = model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = target_model(next_states)
            target_q = compute_q_targets(rewards, next_q_values, dones, gamma)

        # TD error (for logging)
        td_error = torch.abs(target_q - current_q).mean().item()
        total_td_error += td_error

        # Loss: MSE between current Q and target Q
        loss = criterion(current_q, target_q)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'td_error': f'{td_error:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_td_error = total_td_error / len(train_loader)

    return avg_loss, avg_td_error


def validate(model, val_loader, criterion, device, gamma=0.99):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_td_error = 0

    with torch.no_grad():
        for states, actions, rewards, next_states, dones in val_loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # Current Q-values
            q_values = model(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values
            next_q_values = model(next_states)
            target_q = compute_q_targets(rewards, next_q_values, dones, gamma)

            # TD error
            td_error = torch.abs(target_q - current_q).mean().item()
            total_td_error += td_error

            # Loss
            loss = criterion(current_q, target_q)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_td_error = total_td_error / len(val_loader)

    return avg_loss, avg_td_error


def train(config):
    """Main DQN training function"""

    print("=" * 70)
    print("Alvin Pac-Man AI - DQN Training (Reward-Based)")
    print("=" * 70)
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    transitions = load_training_data(config['data_path'])

    # Create dataloaders
    dataset = DQNDataset(transitions)
    total_size = len(dataset)

    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples ({config['train_split']*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({config['val_split']*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({(1-config['train_split']-config['val_split'])*100:.0f}%)")

    # Create model
    print("\nCreating DQN model...")
    model = DQNNetwork(
        input_dim=768,
        hidden_dims=config['hidden_dims'],
        output_dim=4,
        dropout=config['dropout']
    ).to(device)

    # Create target network (for stable training)
    target_model = DQNNetwork(
        input_dim=768,
        hidden_dims=config['hidden_dims'],
        output_dim=4,
        dropout=0.0  # No dropout in target network
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    print(f"[+] DQN created")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Architecture: {config['hidden_dims']}")

    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for Q-value regression
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/dqn_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard: {log_dir}")

    # Training loop
    print("\nStarting DQN training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)

        # Train
        train_loss, train_td_error = train_epoch(
            model, target_model, train_loader, criterion, optimizer, device, config['gamma']
        )

        # Validate
        val_loss, val_td_error = validate(model, val_loader, criterion, device, config['gamma'])

        # Update target network every N epochs
        if (epoch + 1) % config['target_update_freq'] == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"  [+] Updated target network")

        # Scheduler step
        scheduler.step(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('TD_Error/train', train_td_error, epoch)
        writer.add_scalar('TD_Error/val', val_td_error, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n  Train Loss: {train_loss:.4f}  TD Error: {train_td_error:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  TD Error:   {val_td_error:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_dqn_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'target_model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_td_error': val_td_error,
                'config': config
            }, checkpoint_path)
            print(f"  [+] Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered (patience={config['early_stopping_patience']})")
            break

    # Test final model
    print("\n" + "=" * 70)
    print("Testing best model...")
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_dqn_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_td_error = validate(model, test_loader, criterion, device, config['gamma'])
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test TD Error: {test_td_error:.4f}")

    writer.close()

    print("\n[+] DQN Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Model saved: {checkpoint_path}")

    return model, test_loss


def main():
    # DQN Training configuration
    config = {
        'data_path': 'data/raw/training_data_20251114_000000.json',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'early_stopping_patience': 15,
        'train_split': 0.8,
        'val_split': 0.1,
        'hidden_dims': [256, 256, 128],
        'dropout': 0.3,
        'gamma': 0.99,  # Discount factor for future rewards
        'target_update_freq': 5  # Update target network every N epochs
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Train
    model, test_loss = train(config)

    print("\nNext steps:")
    print("  1. python export/to_onnx_dqn.py  # Export DQN to ONNX")
    print("  2. Update inference server to use DQN")
    print("  3. Test locally and compare to behavioral cloning")

    return 0


if __name__ == "__main__":
    exit(main())
