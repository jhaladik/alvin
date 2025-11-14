# NGC Training - Quick Checklist

## Before You Start

- [ ] NVIDIA NGC account created
- [ ] Training data ready (946 samples ✓)
- [ ] 30 minutes available

---

## NGC Workspace Setup (5 minutes)

1. [ ] Go to https://ngc.nvidia.com/
2. [ ] Click "Workspaces" → "Launch"
3. [ ] Configure:
   - [ ] Name: `alvin-pacman-training`
   - [ ] Container: `nvcr.io/nvidia/pytorch:23.10-py3`
   - [ ] GPU: A100 or V100
   - [ ] Storage: 50GB
4. [ ] Click "Launch Workspace"
5. [ ] Wait for "Running" status (2-5 min)
6. [ ] Click "Open JupyterLab"
7. [ ] Open Terminal (click + icon, then Terminal)

---

## Setup in NGC Terminal (3 minutes)

```bash
# Check GPU
nvidia-smi

# Clone repo
cd ~
git clone https://github.com/jhaladik/alvin.git
cd alvin/ml-training

# Install packages
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

---

## Training (10-15 minutes)

### Option A: Automatic (Recommended)

```bash
python quickstart.py
```

Then press Enter. It will:
- Download data
- Extract features
- Train model
- Export ONNX
- Show results

### Option B: Manual

```bash
# 1. Download data
python data/fetch_training_data.py

# 2. Extract features
python preprocessing/feature_extraction.py

# 3. Train
python training/train.py

# 4. Export
python export/to_onnx.py
```

---

## After Training

1. [ ] Check results:
   - [ ] Validation accuracy: 65-75% ✓
   - [ ] Test accuracy: 60-70% ✓
   - [ ] File exists: `export/pacman_model.onnx` ✓

2. [ ] Download model:
   - [ ] In JupyterLab, navigate to `alvin/ml-training/export/`
   - [ ] Right-click `pacman_model.onnx`
   - [ ] Click "Download"

3. [ ] Stop workspace:
   - [ ] Go back to NGC Workspaces page
   - [ ] Click "Stop" button (saves credits!)

---

## Success Criteria

✅ Training completed without errors
✅ Test accuracy 60-70%
✅ ONNX model downloaded (1-2 MB)
✅ Workspace stopped

---

## Troubleshooting

**No GPU detected?**
```bash
nvidia-smi  # Check if GPU is there
```
If not, recreate workspace with GPU selected.

**Out of memory?**
Edit `training/train.py`:
```python
'batch_size': 32  # Change from 64 to 32
```

**Low accuracy (<50%)?**
- Need more training data
- Play more games (target: 2000+ samples)

**Can't download data?**
```bash
# Test connection
curl https://alvin-pacman-ai.jhaladik.workers.dev/api/export-training-data
```

---

## Files You'll Have

```
checkpoints/
  └── best_model.pth          ← PyTorch checkpoint

export/
  └── pacman_model.onnx       ← Deploy this! (1-2 MB)

runs/
  └── pacman_*/               ← TensorBoard logs

data/
  ├── raw/training_data_*.json
  └── processed/features.npz
```

---

## Next Steps After NGC

1. Integrate `pacman_model.onnx` into your app
2. Replace heuristics with neural network predictions
3. Test AI performance
4. Collect more data and retrain if needed

---

**Total time: ~20 minutes**
**Cost: Free (or <$1 if using paid GPU)**
**Result: Real machine learning! 🎉**
