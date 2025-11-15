"""
Training Script for Pac-Man Behavioral Cloning
Trains neural network to imitate human gameplay on NGC
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.behavioral_cloning import PacManPolicyNetwork, count_parameters


class PacManDataset(Dataset):
    """PyTorch Dataset for Pac-Man training data"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(data_path):
    """Load processed dataset"""
    print(f"Loading dataset from {data_path}...")

    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']

    print(f"[+] Loaded {len(X)} samples")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

    return X, y


def create_dataloaders(X, y, batch_size=64, train_split=0.8, val_split=0.1):
    """Create train/val/test dataloaders"""

    dataset = PacManDataset(X, y)
    total_size = len(dataset)

    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({(1-train_split-val_split)*100:.0f}%)")

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train(config):
    """Main training function"""

    print("=" * 70)
    print("Alvin Pac-Man AI - Behavioral Cloning Training")
    print("=" * 70)
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Load data
    X, y = load_dataset(config['data_path'])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y,
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        val_split=config['val_split']
    )

    # Create model
    print("\nCreating model...")
    model = PacManPolicyNetwork(
        input_dim=128,
        hidden_dims=config['hidden_dims'],
        output_dim=4,
        dropout=config['dropout']
    )
    model = model.to(device)
    print(f"[+] Model created")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Architecture: {config['hidden_dims']}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/pacman_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard: {log_dir}")

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  [+] Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered (patience={config['early_stopping_patience']})")
            break

    # Test final model
    print("\n" + "=" * 70)
    print("Testing best model...")
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")

    writer.close()

    print("\n[+] Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Model saved: {checkpoint_path}")

    return model, test_acc


def main():
    # Training configuration
    config = {
        'data_path': 'data/processed/features.npz',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'early_stopping_patience': 15,
        'train_split': 0.8,
        'val_split': 0.1,
        'hidden_dims': [256, 256, 128],
        'dropout': 0.3
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Train
    model, test_acc = train(config)

    print("\nNext steps:")
    print("  1. python export/to_onnx.py  # Export to ONNX")
    print("  2. Deploy to Cloudflare Workers")

    return 0


if __name__ == "__main__":
    exit(main())
