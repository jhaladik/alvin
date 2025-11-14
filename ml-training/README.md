# Alvin Pac-Man AI - Machine Learning Training

Real ML training pipeline for behavioral cloning on NVIDIA NGC.

## Overview

This replaces the fake "AI" (text embeddings + heuristics) with **actual machine learning**:

- **Proper feature extraction**: 128-dim structured vectors (not text)
- **Neural network**: 3-layer policy network (~200K parameters)
- **Real training**: Behavioral cloning on GPU
- **Deployment**: ONNX export for production

## Project Structure

```
ml-training/
├── data/
│   ├── fetch_training_data.py    # Download from Cloudflare
│   ├── raw/                        # Downloaded JSON data
│   └── processed/                  # Preprocessed features.npz
├── preprocessing/
│   └── feature_extraction.py      # Convert states to 128-dim vectors
├── models/
│   └── behavioral_cloning.py      # PyTorch model architecture
├── training/
│   └── train.py                    # Training script for NGC
├── export/
│   └── to_onnx.py                  # Export to ONNX format
├── notebooks/
│   └── (optional Jupyter notebooks)
└── requirements.txt
```

## Quick Start

### Step 1: Deploy Data Export Endpoint

The worker now has `/api/export-training-data` endpoint. Deploy it:

```bash
cd ..  # Back to project root
npm run deploy
```

### Step 2: Play Games to Generate Data

Go to https://alvin-pacman-ai.jhaladik.workers.dev and play Pac-Man for a while.
**Target: 1000+ moves** (more is better)

### Step 3: NGC Setup

1. Log in to https://ngc.nvidia.com/
2. Go to "Workspaces" → "Create Workspace"
3. Select:
   - **Container**: `nvcr.io/nvidia/pytorch:23.10-py3`
   - **GPU**: A100 or V100 (free tier)
   - **Storage**: 50GB
4. Launch workspace
5. Open JupyterLab terminal

### Step 4: Upload This Directory to NGC

In NGC JupyterLab:

```bash
# Clone your repo or upload ml-training/ folder
git clone https://github.com/jhaladik/alvin.git
cd alvin/ml-training

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Download Training Data

```bash
python data/fetch_training_data.py
```

This downloads all human gameplay data from Cloudflare KV.

### Step 6: Preprocess Data

```bash
python preprocessing/feature_extraction.py
```

Converts game states to 128-dim feature vectors.

### Step 7: Train on GPU

```bash
python training/train.py
```

This will:
- Train for up to 100 epochs
- Early stopping if no improvement
- Save best model to `checkpoints/best_model.pth`
- Log to TensorBoard

Monitor training:
```bash
tensorboard --logdir=runs
```

### Step 8: Export Model

```bash
python export/to_onnx.py
```

Exports to `export/pacman_model.onnx` for deployment.

### Step 9: Download and Deploy

Download `pacman_model.onnx` from NGC and integrate into your app.

## Feature Engineering

Our 128-dimensional feature vector includes:

1. **Player** (6): position (x,y), direction (one-hot)
2. **Ghosts** (24): 4 ghosts × (x, y, scared, distance, relative dx/dy)
3. **Pellets** (6): nearest pellet (distance, dx, dy) + nearest power pellet
4. **Game State** (6): score, pellets left, lives, power mode, power timer, total pellets
5. **Spatial Context** (25): 5×5 grid of pellet density
6. **Danger Zones** (8): ghost threat in 8 directions
7. **Wall Proximity** (4): distance to wall in 4 cardinal directions
8. **Tactical** (10): escape routes, trapped indicator, etc.
9. **Padding** (39): reserved for future features

**Total: 128 features**

This is MUCH better than text embeddings because:
- ✓ Spatial relationships preserved
- ✓ Numerical values directly comparable
- ✓ No semantic confusion
- ✓ Faster inference

## Model Architecture

```
Input: (batch, 128)
  ↓
Linear(128 → 256) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Linear(256 → 256) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Linear(256 → 128) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Linear(128 → 4)
  ↓
Output: (batch, 4)  # Logits for UP/DOWN/LEFT/RIGHT
```

Parameters: ~200,000

## Training Configuration

```python
{
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 15,
    'train_split': 0.8,    # 80% training
    'val_split': 0.1,      # 10% validation
    # test_split': 0.1     # 10% testing
}
```

## Expected Performance

With 1000+ quality samples:
- Training accuracy: 70-80%
- Validation accuracy: 65-75%
- Test accuracy: 60-70%

**Note**: Humans aren't perfect! The AI learns human strategies, including mistakes.

## Deployment Options

### Option 1: ONNX Runtime in Browser

```javascript
import * as ort from 'onnxruntime-web';

// Load model
const session = await ort.InferenceSession.create('/pacman_model.onnx');

// Extract features (implement feature extraction in JS)
const features = extractFeatures(gameState);  // Returns Float32Array(128)

// Inference
const input = new ort.Tensor('float32', features, [1, 128]);
const outputs = await session.run({input: input});

// Get action
const logits = outputs.output.data;
const action = argmax(logits);  // 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
```

### Option 2: Cloudflare Workers AI

Check if Cloudflare Workers AI supports ONNX. If yes, upload model and use:

```javascript
const result = await env.AI.run('custom-model', {
  input: features
});
```

### Option 3: Backend API

Deploy ONNX Runtime on a server and call via API (adds latency).

## Troubleshooting

### Not Enough Data
```
✗ Warning: Only 237 samples available
```
**Solution**: Play more games! Aim for 1000+

### Low Accuracy
```
✓ Test accuracy: 45.23%
```
**Solution**:
- Collect more data
- Check feature extraction quality
- Try different model architectures
- Filter bad samples (reward < -50)

### NGC GPU Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch_size in `training/train.py`

## Next Steps After Training

1. **Integrate model** into frontend or worker
2. **Replace heuristics** with neural network predictions
3. **A/B test** human vs AI performance
4. **Iterate**: collect more data, retrain, deploy

## Advanced: Reinforcement Learning

Want to go beyond imitation? Implement proper DQN:

1. Use `models/behavioral_cloning.py` → `PacManDQN` class
2. Implement experience replay
3. Train with Q-learning updates
4. Add target network

This requires online training (not just offline from saved games).

## Credits

- Framework: PyTorch
- Training: NVIDIA NGC
- Deployment: ONNX Runtime
- Project: Alvin Pac-Man AI

---

**Now you have REAL machine learning! 🎉**
