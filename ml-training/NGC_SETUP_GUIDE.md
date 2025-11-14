# NVIDIA NGC Setup Guide - Step by Step

Complete walkthrough for training Alvin Pac-Man AI on NGC.

## 📋 Prerequisites Checklist

- [ ] NVIDIA NGC account (https://ngc.nvidia.com/)
- [ ] GitHub repo cloned locally
- [ ] Training data verified (946 samples ✓)
- [ ] 30 minutes of time

---

## 🚀 Step-by-Step Instructions

### **Step 1: Login to NGC**

1. Go to: https://ngc.nvidia.com/
2. Click **"Sign In"** (top right)
3. Use your NVIDIA account credentials
4. If you don't have an account:
   - Click "Sign Up"
   - Fill in details (free!)
   - Verify email

✅ **Checkpoint**: You should see NGC dashboard with "Catalog", "Workspaces", "Resources"

---

### **Step 2: Navigate to Workspaces**

1. Click **"Workspaces"** in top navigation bar
2. You'll see list of workspaces (probably empty if first time)
3. Click **"Launch"** button (blue button, top right)

✅ **Checkpoint**: You should see "Create Workspace" page

---

### **Step 3: Configure Workspace**

**IMPORTANT**: Use these EXACT settings:

#### **Workspace Name**
```
alvin-pacman-training
```

#### **Container Image**
Click "Select Container" and search for:
```
pytorch
```
Select: **`nvcr.io/nvidia/pytorch:23.10-py3`**

This includes:
- PyTorch 2.1
- CUDA 12.2
- Python 3.10
- All necessary libraries

#### **GPU Selection**
Choose one of:
- **A100 (40GB)** ← Best, fastest
- **V100 (32GB)** ← Good alternative
- **T4 (16GB)** ← Slower but works

**Free tier**: You get some free GPU hours/month

#### **Storage**
```
50 GB
```

#### **Advanced Settings** (expand if you see it)
- Region: Choose closest to you
- Auto-shutdown: 2 hours (saves credits)
- Everything else: leave default

#### **Review**
- Check all settings
- Estimated cost: Should show free credits or $0.XX/hour

Click **"Launch Workspace"** button

✅ **Checkpoint**: Workspace will show "Starting..." status

---

### **Step 4: Wait for Workspace to Start**

⏱️ This takes **2-5 minutes**

You'll see status:
```
Starting... → Running
```

When ready, you'll see:
- Green "Running" badge
- "Open JupyterLab" button

Click **"Open JupyterLab"**

✅ **Checkpoint**: JupyterLab interface opens in new tab

---

### **Step 5: Open Terminal in JupyterLab**

In JupyterLab interface:
1. Look at left sidebar
2. Click **"+"** icon (New Launcher)
3. Under "Other", click **"Terminal"**

You should see:
```bash
root@container-id:~#
```

✅ **Checkpoint**: Terminal is open and ready

---

### **Step 6: Verify GPU**

In terminal, run:
```bash
nvidia-smi
```

You should see:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   35C    P0    52W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ **Checkpoint**: GPU is detected, 0% usage, plenty of memory

---

### **Step 7: Clone Your Repository**

In terminal:
```bash
# Go to home directory
cd ~

# Clone repo
git clone https://github.com/jhaladik/alvin.git

# Navigate to ML training folder
cd alvin/ml-training

# Verify files are there
ls -la
```

You should see:
```
data/
models/
preprocessing/
training/
export/
requirements.txt
README.md
```

✅ **Checkpoint**: All files present

---

### **Step 8: Install Dependencies**

```bash
pip install -r requirements.txt
```

Expected output:
```
Collecting torch>=2.0.0
  Using cached torch-2.1.0...
Collecting numpy>=1.24.0
  Using cached numpy-1.24.3...
...
Successfully installed torch-2.1.0 numpy-1.24.3 ...
```

⏱️ Takes **2-3 minutes**

✅ **Checkpoint**: No errors, all packages installed

---

### **Step 9: Run Quickstart Script**

I'll create this script for you. It runs all steps automatically:

```bash
python quickstart.py
```

This will:
1. Download training data (946 samples)
2. Extract features (128-dim vectors)
3. Train model on GPU
4. Export to ONNX
5. Show results

Expected output: See below ⬇️

---

### **Step 10: Monitor Training**

You'll see real-time progress:

```
==================================================================
Alvin Pac-Man AI - Behavioral Cloning Training
==================================================================

Device: cuda
GPU: NVIDIA A100-SXM4-40GB
CUDA Version: 12.2

Loading dataset from data/processed/features.npz...
✓ Loaded 946 samples
  Features shape: (946, 128)
  Labels shape: (946,)

Dataset splits:
  Train: 757 samples (80%)
  Val:   94 samples (10%)
  Test:  95 samples (10%)

Creating model...
✓ Model created
  Parameters: 214,020
  Architecture: [256, 256, 128]

Starting training...

Epoch 1/100
----------------------------------------------------------------------
Training: 100%|██████████| 12/12 [00:01<00:00,  8.5it/s, loss=1.3862, acc=25.23%]

  Train Loss: 1.3862  Train Acc: 25.23%
  Val Loss:   1.3801  Val Acc:   26.60%
  LR: 0.001000
  ✓ Saved best model (val_acc: 26.60%)

Epoch 2/100
----------------------------------------------------------------------
Training: 100%|██████████| 12/12 [00:01<00:00, 10.2it/s, loss=1.2456, acc=42.14%]

  Train Loss: 1.2456  Train Acc: 42.14%
  Val Loss:   1.2123  Val Acc:   45.74%
  LR: 0.001000
  ✓ Saved best model (val_acc: 45.74%)

...

Epoch 45/100
----------------------------------------------------------------------
Training: 100%|██████████| 12/12 [00:01<00:00, 10.8it/s, loss=0.5234, acc=78.45%]

  Train Loss: 0.5234  Train Acc: 78.45%
  Val Loss:   0.6789  Val Acc:   72.34%
  LR: 0.000125
  ✓ Saved best model (val_acc: 72.34%)

==================================================================
Testing best model...
  Test Loss: 0.7012
  Test Acc:  69.47%

✓ Training complete!
  Best val accuracy: 72.34%
  Test accuracy: 69.47%
  Model saved: checkpoints/best_model.pth

Exporting to ONNX...
✓ Exported to: export/pacman_model.onnx
✓ Model verified
  File size: 1.23 MB

==================================================================
✓ ALL DONE!
==================================================================

Next steps:
  1. Download: export/pacman_model.onnx
  2. Deploy to your app
  3. Replace heuristics with real ML!
```

⏱️ **Total time**: ~10-15 minutes on A100

✅ **Checkpoint**: Training completed successfully, test accuracy ~70%

---

### **Step 11: Download Trained Model**

In JupyterLab file browser (left sidebar):
1. Navigate to: `alvin/ml-training/export/`
2. Right-click `pacman_model.onnx`
3. Click **"Download"**
4. Save to your local machine

✅ **Checkpoint**: Model file downloaded (should be ~1-2 MB)

---

### **Step 12: Optional - View TensorBoard**

To see training visualizations:

```bash
# In NGC terminal
cd ~/alvin/ml-training
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```

Then in JupyterLab:
- File → New → Terminal (new terminal)
- Note the container URL
- Open: `http://[container-url]:6006`

You'll see:
- Loss curves
- Accuracy curves
- Learning rate schedule

---

### **Step 13: Stop Workspace (IMPORTANT!)**

**To save your free credits:**

1. Go back to NGC Workspaces page
2. Find your workspace
3. Click **"Stop"** button

**Note**: You can restart it anytime, files are saved!

✅ **Checkpoint**: Workspace stopped, credits saved

---

## 🎯 Success Criteria

You've succeeded if you see:

- ✅ Training completed without errors
- ✅ Validation accuracy: 65-75%
- ✅ Test accuracy: 60-70%
- ✅ ONNX model exported
- ✅ Model downloaded to local machine

---

## 🚨 Troubleshooting

### **Problem: "No GPU detected"**

```bash
nvidia-smi
# Shows: command not found
```

**Solution**:
- Your workspace didn't get GPU allocation
- Stop workspace
- Create new one with GPU explicitly selected

### **Problem: "Out of memory"**

```
RuntimeError: CUDA out of memory
```

**Solution**:
- Edit `training/train.py`
- Change `batch_size: 64` → `batch_size: 32`
- Re-run training

### **Problem: "Low accuracy (<50%)"**

**Possible causes**:
- Not enough training data
- Bad quality data (too many deaths)
- Model too small

**Solutions**:
- Collect more data (play more games)
- Filter bad samples
- Try larger model

### **Problem: "Can't download data from Cloudflare"**

```python
# In fetch_training_data.py
WORKER_URL = "https://alvin-pacman-ai.jhaladik.workers.dev"
```

**Solution**:
- Verify URL is correct
- Check worker is deployed
- Try curl manually:
```bash
curl https://alvin-pacman-ai.jhaladik.workers.dev/api/export-training-data
```

### **Problem: "Workspace costs money"**

**Solution**:
- Check free tier limits (usually ~50 hours/month)
- Stop workspace when not using
- Use auto-shutdown (2 hours)

---

## 📞 Support

If stuck:
1. Check NGC documentation: https://docs.nvidia.com/ngc/
2. Check error message carefully
3. Google the exact error
4. Ask in NVIDIA forums

---

## ⏭️ After Training

Once you have `pacman_model.onnx`:

1. **Test locally** (optional):
   - Install ONNX Runtime
   - Load model
   - Test with sample game states

2. **Deploy**:
   - Option A: ONNX Runtime in browser
   - Option B: Backend API
   - Option C: Cloudflare Workers (if supported)

3. **Replace heuristics**:
   - Current: `calculateActionScore()` with hardcoded rules
   - New: Neural network predictions!

4. **Measure improvement**:
   - AI win rate
   - Average score
   - Survival time

---

**Ready? Let's do this! 🚀**
