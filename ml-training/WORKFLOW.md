# Streamlined HF Spaces Workflow (Option A)

## 🎯 Overview

**Fixed Issues:**
- ✅ Vector dimension mismatch (now using 128-dim throughout)
- ✅ Removed zero-padding hack
- ✅ Model auto-detects input dimension
- ✅ Unified training pipeline
- ✅ Headless RL training ready for Phase 2

**Architecture:**
```
Frontend (Cloudflare Workers)
  ↓ 128-dim features
Cloudflare Worker (storage)
  ↓ /api/export-training-data
Local Training (train_128.py)
  ↓ best_dqn_model.pth
Hugging Face Spaces (inference)
  ↓ 128-dim → Q-values
Frontend (displays predictions)
```

---

## 📂 File Structure (Cleaned Up)

### ✅ ACTIVE FILES (Keep & Use)

```
ml-training/
├── models/
│   └── dqn_network.py            # DQN model (128-dim input)
├── environment/
│   └── pacman_env.py             # Headless Pac-Man for RL
├──  train_128.py                  # Phase 1: Train on human data (128-dim) ⭐ NEW
├── train_rl_headless.py          # Phase 2: Self-play RL training ⭐ NEW
├── export_cloudflare_data.py     # Export data from Cloudflare ⭐ NEW
├── inference_server.py           # HF Spaces inference server (UPDATED)
├── checkpoints/
│   └── best_dqn_model.pth        # Trained model
├── Dockerfile                    # HF Spaces deployment
├── README.md                     # HF Spaces description
├── requirements.txt              # Dependencies
└── .gitattributes                # Git LFS for *.pth files
```

### ❌ DEPRECATED FILES (Archive or Remove)

```
ml-training/
├── training/
│   ├── train.py                  # ❌ OLD - Used 768-dim (superseded by train_128.py)
│   ├── train_dqn.py              # ❌ OLD - Used 768-dim (superseded by train_128.py)
│   └── train_rl.py               # ❌ OLD - Not using agents/ subdirectory
├── agents/
│   ├── dqn_agent.py              # ❌ OLD - Duplicates model functionality
│   └── replay_buffer.py          # ❌ OLD - Now in train_rl_headless.py
├── NGC_*.md                      # ⚠️  OPTIONAL - NGC GPU cloud docs (if not using)
├── RL_TRAINING.md                # ⚠️  PARTIALLY OUTDATED - Update or remove
└── worker/ml-predictor.js        # ❌ NOT USED - Worker just stores data now
```

---

## 🚀 Complete Workflow

### Phase 1: Bootstrap from Human Data (CURRENT)

#### Step 1: Collect Human Gameplay Data

```bash
# Frontend deployed at https://alvin-pacman-ai.jhaladik.workers.dev
# Play the game in human mode
# Data automatically stored in Cloudflare KV with 128-dim vectors
```

#### Step 2: Export Training Data

```bash
cd ml-training
python export_cloudflare_data.py
```

This exports to `data/training_data.json` with:
- 128-dim feature vectors
- Actions (UP/DOWN/LEFT/RIGHT)
- Rewards (multi-dimensional)
- Metadata (strategy, success, etc.)

#### Step 3: Train DQN Model (128-dim)

```bash
python train_128.py
```

**What it does:**
- Auto-detects vector dimension (128)
- Trains using offline DQN with Q-learning
- Saves `checkpoints/best_dqn_model.pth` with input_dim in config
- Uses TensorBoard for monitoring

**Training Time:** ~5-10 minutes on CPU, ~1-2 minutes on GPU

#### Step 4: Test Locally

```bash
# Start inference server
python inference_server.py

# In another terminal, start frontend
cd ..
node local-server.js

# Open http://localhost:3000
# Enable AI mode and test predictions
```

#### Step 5: Deploy to Hugging Face Spaces

See `HF_SPACES_DEPLOYMENT.md` for detailed instructions.

**Quick Deploy:**
1. Create HF Space with Docker SDK
2. Upload files: `Dockerfile`, `README.md`, `inference_server.py`, `requirements.txt`, `.gitattributes`, `models/`, `checkpoints/`
3. Wait for build (5-10 min)
4. Test: `https://YOUR-USERNAME-alvin-pacman-dqn.hf.space/health`

---

### Phase 2: Self-Play RL Training (FUTURE)

#### When to Use:

After Phase 1 model is working well, use headless RL to improve beyond human skill.

#### Step 1: Run Headless Training

```bash
python train_rl_headless.py
```

**What it does:**
- Loads pre-trained model from Phase 1
- Creates headless Pac-Man environment
- AI plays 5000+ games
- Learns from rewards (not human moves)
- Explores new strategies
- Saves `checkpoints/best_rl_model.pth`

**Training Time:** ~2-4 hours on CPU, ~30-60 minutes on GPU

#### Step 2: Replace Model

```bash
# Backup old model
cp checkpoints/best_dqn_model.pth checkpoints/best_dqn_model_phase1.pth

# Use new RL model
cp checkpoints/best_rl_model.pth checkpoints/best_dqn_model.pth
```

#### Step 3: Redeploy to HF Spaces

Upload updated `checkpoints/best_dqn_model.pth` to HF Space.

---

## 🔧 Configuration

### Model Hyperparameters (train_128.py)

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 50
HIDDEN_DIMS = [256, 256, 128]
GAMMA = 0.99
VAL_SPLIT = 0.2
```

### RL Hyperparameters (train_rl_headless.py)

```python
NUM_EPISODES = 5000
EPSILON_START = 1.0  # 100% exploration initially
EPSILON_END = 0.01   # 1% exploration finally
EPSILON_DECAY = 0.995
GAMMA = 0.99
TARGET_UPDATE_FREQ = 100
```

---

## 📊 Monitoring

### TensorBoard (During Training)

```bash
# While training is running
tensorboard --logdir=runs

# Open http://localhost:6006
```

**Metrics:**
- Train/Val Loss
- TD Error (temporal difference)
- Q-value distribution
- Reward history (RL only)

### Inference Server Logs

```bash
# Start server with logging
python inference_server.py

# Watch logs for:
# - [DQN] Predictions with Q-values
# - [INFO] Input dimension validation
# - [ERROR] Any issues
```

---

## 🐛 Troubleshooting

### Issue: Dimension mismatch error

```
[ERROR] Wrong vector dimension: got 768, expected 128
```

**Fix:** Old checkpoint uses 768-dim. Retrain with `train_128.py`.

### Issue: No training data

```
[WARNING] No data found. Play some games first!
```

**Fix:** Play the game in human mode to collect data, then export.

### Issue: Poor performance

**Phase 1 (Human Data):**
- Collect more diverse gameplay data
- Play in different strategies (aggressive, defensive, pellet-focused)
- Ensure successful games are included

**Phase 2 (RL):**
- Let it train longer (10k+ episodes)
- Adjust epsilon decay for more exploration
- Tune reward shaping in `environment/pacman_env.py`

---

## 📈 Expected Results

### Phase 1 (Offline DQN)

- **Training:** 50 epochs, ~5-10 min
- **Val Loss:** ~20-30 (depends on data quality)
- **Performance:** Matches human skill level
- **Win Rate:** 30-50% (similar to human)

### Phase 2 (RL Self-Play)

- **Training:** 5000 episodes, ~2-4 hours
- **Avg Reward:** Improves over time
- **Performance:** Can exceed human skill
- **Win Rate:** 50-70% (better than human)

---

## 🎯 Next Steps

1. **Immediate:**
   - ✅ Export data: `python export_cloudflare_data.py`
   - ✅ Train model: `python train_128.py`
   - ✅ Test locally: `python inference_server.py`
   - ✅ Deploy to HF Spaces

2. **Short-term:**
   - Enable headless RL training
   - Monitor performance improvements
   - Compare Phase 1 vs Phase 2 models

3. **Long-term:**
   - Automate retraining pipeline
   - Continuous learning from new human data
   - Hybrid models (combine human + RL data)

---

## 📚 Documentation

- `WORKFLOW.md` (this file) - Complete workflow guide
- `HF_SPACES_DEPLOYMENT.md` - HF Spaces deployment
- `models/dqn_network.py` - Model architecture docs
- `environment/pacman_env.py` - Environment API docs

---

## 🔑 Key Differences from Old System

| Aspect | OLD (768-dim) | NEW (128-dim) |
|--------|---------------|---------------|
| **Vector Dim** | 768 (Cloudflare AI) | 128 (feature engineering) |
| **Padding** | Zero-padding hack | No padding needed |
| **Training** | train_dqn.py (768-dim) | train_128.py (auto-detect) |
| **Inference** | Pads 128→768 | Direct 128-dim |
| **RL Support** | Not integrated | train_rl_headless.py ready |
| **Config** | Hard-coded 768 | Saved in checkpoint |

---

## ✅ Success Criteria

You know the system is working when:

1. ✅ Export shows 128-dim vectors
2. ✅ Training completes without dimension errors
3. ✅ Inference server loads model with correct input_dim
4. ✅ HF Spaces health check returns `{"status": "ok"}`
5. ✅ Predictions show realistic Q-values
6. ✅ AI makes reasonable moves in the game

---

## 🚀 Ready to Go!

The system is now streamlined and ready for:
- ✅ Phase 1: Train on human data (works now)
- ✅ Phase 2: RL self-play (code ready, run when needed)
- ✅ HF Spaces deployment (no padding issues)
- ✅ Continuous improvement workflow

**Start here:** `python export_cloudflare_data.py`
