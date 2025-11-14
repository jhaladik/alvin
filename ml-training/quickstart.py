#!/usr/bin/env python3
"""
NGC Quickstart Script
Runs complete training pipeline automatically
"""
import os
import sys
import subprocess
import time

def print_banner(text):
    """Print fancy banner"""
    width = 70
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width + "\n")

def run_command(description, command, check_output=False):
    """Run command with nice formatting"""
    print(f"⏳ {description}...")
    print(f"   Command: {command}")
    print()

    try:
        if check_output:
            result = subprocess.run(
                command, shell=True, check=True,
                capture_output=True, text=True
            )
            print(result.stdout)
            return result.stdout
        else:
            subprocess.run(command, shell=True, check=True)
        print(f"✓ {description} - Done!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed!")
        print(f"   Error: {e}")
        if e.stderr:
            print(f"   Details: {e.stderr}")
        return False

def check_gpu():
    """Verify GPU is available"""
    print_banner("Step 1: GPU Verification")

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            shell=True, capture_output=True, text=True, check=True
        )
        gpu_info = result.stdout.strip()
        print(f"✓ GPU detected: {gpu_info}\n")
        return True
    except:
        print("✗ No GPU detected!")
        print("  Make sure you selected GPU when creating workspace")
        return False

def download_data():
    """Download training data from Cloudflare"""
    print_banner("Step 2: Download Training Data")

    if os.path.exists("data/raw/training_data.json"):
        print("⚠ Data already exists, skipping download")
        return True

    return run_command(
        "Downloading training data",
        "python data/fetch_training_data.py"
    )

def extract_features():
    """Extract features from raw data"""
    print_banner("Step 3: Feature Extraction")

    if os.path.exists("data/processed/features.npz"):
        print("⚠ Features already exist, skipping extraction")
        return True

    return run_command(
        "Extracting features",
        "python preprocessing/feature_extraction.py"
    )

def train_model():
    """Train the model"""
    print_banner("Step 4: Model Training (This Takes ~10-15 min)")

    print("🔥 Training on GPU...")
    print("   You'll see real-time progress below")
    print()

    return run_command(
        "Training model",
        "python training/train.py"
    )

def export_model():
    """Export to ONNX"""
    print_banner("Step 5: Export to ONNX")

    return run_command(
        "Exporting to ONNX",
        "python export/to_onnx.py"
    )

def show_results():
    """Show final results"""
    print_banner("✓ TRAINING COMPLETE!")

    print("📊 Results:")
    print()

    # Check if model exists
    if os.path.exists("checkpoints/best_model.pth"):
        import torch
        checkpoint = torch.load("checkpoints/best_model.pth", map_location='cpu')
        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', 0)
        print(f"  ✓ Best model: epoch {epoch}")
        print(f"  ✓ Validation accuracy: {val_acc:.2f}%")
    else:
        print("  ⚠ Model checkpoint not found")

    if os.path.exists("export/pacman_model.onnx"):
        size = os.path.getsize("export/pacman_model.onnx") / (1024 * 1024)
        print(f"  ✓ ONNX model: {size:.2f} MB")
    else:
        print("  ⚠ ONNX model not found")

    print()
    print("📦 Files created:")
    print("  - checkpoints/best_model.pth  (PyTorch model)")
    print("  - export/pacman_model.onnx    (Deployment model)")
    print("  - runs/                        (TensorBoard logs)")
    print()
    print("⏭️  Next steps:")
    print("  1. Download export/pacman_model.onnx")
    print("  2. Integrate into your app")
    print("  3. Replace heuristics with real ML!")
    print()
    print("💡 Optional:")
    print("  - View training curves: tensorboard --logdir=runs")
    print("  - Test model: python -c \"from export.to_onnx import *; ...\"")
    print()

def main():
    """Run complete pipeline"""
    print_banner("Alvin Pac-Man AI - NGC Training Pipeline")

    print("This script will:")
    print("  1. Verify GPU")
    print("  2. Download training data (946 samples)")
    print("  3. Extract features (128-dim vectors)")
    print("  4. Train neural network (~10-15 min)")
    print("  5. Export to ONNX format")
    print()
    print("Total time: ~15-20 minutes")
    print()

    input("Press Enter to continue (Ctrl+C to cancel)...")

    start_time = time.time()

    # Step 1: Check GPU
    if not check_gpu():
        print("\n✗ GPU check failed. Exiting.")
        return 1

    # Step 2: Download data
    if not download_data():
        print("\n✗ Data download failed. Exiting.")
        return 1

    # Step 3: Extract features
    if not extract_features():
        print("\n✗ Feature extraction failed. Exiting.")
        return 1

    # Step 4: Train
    if not train_model():
        print("\n✗ Training failed. Check errors above.")
        return 1

    # Step 5: Export
    if not export_model():
        print("\n✗ Export failed. Check errors above.")
        return 1

    # Done!
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    show_results()

    print(f"⏱️  Total time: {minutes}m {seconds}s")
    print()
    print("=" * 70)
    print("                    🎉 SUCCESS! 🎉")
    print("=" * 70)
    print()

    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
