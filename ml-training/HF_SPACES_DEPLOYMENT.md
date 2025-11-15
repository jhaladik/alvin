# Deploying ML Inference Server to Hugging Face Spaces

## Overview

This guide explains how to deploy the Alvin Pac-Man DQN inference server to Hugging Face Spaces.

**Why Hugging Face Spaces:**
- ✅ **16 GB RAM free tier** (vs Render's 512 MB)
- ✅ Persistent storage with Git LFS
- ✅ Automatic HTTPS
- ✅ Built-in Docker support
- ✅ Free GPU option for inference
- ✅ Integrated with Hugging Face Hub
- ✅ No cold starts on free tier

## Architecture

```
Frontend (Cloudflare Workers)
    ↓ HTTPS
Hugging Face Spaces (Docker + Flask + PyTorch)
    ↓
DQN Model (best_dqn_model.pth)
    ↓
Predictions (UP/DOWN/LEFT/RIGHT)
```

---

## Prerequisites

1. **Hugging Face Account**
   - Sign up at https://huggingface.co (free)
   - Get your access token: https://huggingface.co/settings/tokens

2. **Git LFS** (for model files)
   ```bash
   # Install Git LFS
   # Windows (via Git for Windows)
   git lfs install

   # Mac
   brew install git-lfs
   git lfs install

   # Linux
   sudo apt-get install git-lfs
   git lfs install
   ```

3. **Hugging Face CLI** (optional but recommended)
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

---

## Deployment Methods

### Method 1: Using Hugging Face Web UI (Easiest)

#### Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Space name:** `alvin-pacman-dqn` (or your choice)
   - **License:** MIT
   - **SDK:** Docker
   - **Visibility:** Public (or Private)
3. Click "Create Space"

#### Step 2: Upload Files

You can upload files via the web UI:

1. Click "Files" tab in your Space
2. Click "Add file" → "Upload files"
3. Upload these files from `ml-training/`:
   - `Dockerfile`
   - `README.md`
   - `inference_server.py`
   - `requirements.txt`
   - `.gitattributes`
   - `models/dqn_network.py`
   - `checkpoints/best_dqn_model.pth` (will be tracked with Git LFS)

**Important:** The `README.md` must have the YAML frontmatter:
```yaml
---
title: Alvin Pac-Man DQN Inference
emoji: 👻
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
license: mit
---
```

#### Step 3: Wait for Build

- Hugging Face will automatically build your Docker container
- Monitor build logs in the "Build" tab
- First build takes ~5-10 minutes
- Look for "Running on http://0.0.0.0:7860" in logs

#### Step 4: Test Your Space

Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn
```

Test the API:
```bash
# Health check
curl https://YOUR-USERNAME-alvin-pacman-dqn.hf.space/health

# Prediction
curl -X POST https://YOUR-USERNAME-alvin-pacman-dqn.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.5, 0.3, 0.8, ...]}'  # 128-dim vector
```

---

### Method 2: Using Git (Advanced)

#### Step 1: Clone Your Space Repository

```bash
# Create Space on HF website first, then clone it
git clone https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn
cd alvin-pacman-dqn

# Initialize Git LFS
git lfs install
```

#### Step 2: Copy Files from ml-training/

```bash
# From your alvin project root
cp ml-training/Dockerfile alvin-pacman-dqn/
cp ml-training/README.md alvin-pacman-dqn/
cp ml-training/inference_server.py alvin-pacman-dqn/
cp ml-training/requirements.txt alvin-pacman-dqn/
cp ml-training/.gitattributes alvin-pacman-dqn/
cp -r ml-training/models alvin-pacman-dqn/
cp -r ml-training/checkpoints alvin-pacman-dqn/
```

#### Step 3: Track Model with Git LFS

```bash
cd alvin-pacman-dqn

# Git LFS will track *.pth files (configured in .gitattributes)
git add .gitattributes
git add checkpoints/best_dqn_model.pth

# Verify LFS tracking
git lfs ls-files
# Should show: checkpoints/best_dqn_model.pth
```

#### Step 4: Commit and Push

```bash
git add .
git commit -m "Initial deployment of DQN inference server"
git push
```

Hugging Face will automatically build and deploy your Space.

---

### Method 3: Using Hugging Face CLI

```bash
# Login first
huggingface-cli login

# Create Space
huggingface-cli repo create alvin-pacman-dqn --type space --space_sdk docker

# Clone and add files
git clone https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn
cd alvin-pacman-dqn

# Copy files (same as Method 2)
cp ../alvin/ml-training/Dockerfile .
cp ../alvin/ml-training/README.md .
cp ../alvin/ml-training/inference_server.py .
cp ../alvin/ml-training/requirements.txt .
cp ../alvin/ml-training/.gitattributes .
cp -r ../alvin/ml-training/models .
cp -r ../alvin/ml-training/checkpoints .

# Commit and push
git add .
git commit -m "Deploy DQN inference server"
git push
```

---

## File Structure for Hugging Face Space

Your Space repository should have this structure:

```
alvin-pacman-dqn/
├── .gitattributes           # Git LFS configuration
├── Dockerfile               # Docker container definition
├── README.md                # Space description (with YAML frontmatter)
├── inference_server.py      # Flask API server
├── requirements.txt         # Python dependencies
├── models/
│   └── dqn_network.py      # DQN model architecture
└── checkpoints/
    └── best_dqn_model.pth  # Trained model (tracked with Git LFS)
```

---

## Post-Deployment

### 1. Get Your Space URL

After deployment, your Space will be accessible at:
```
https://YOUR-USERNAME-alvin-pacman-dqn.hf.space
```

**Important:** Hugging Face Spaces use this format:
- Web UI: `https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn`
- API Endpoint: `https://YOUR-USERNAME-alvin-pacman-dqn.hf.space`

### 2. Test the Deployment

**Health Check:**
```bash
curl https://YOUR-USERNAME-alvin-pacman-dqn.hf.space/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

**Test Prediction:**
```bash
curl -X POST https://YOUR-USERNAME-alvin-pacman-dqn.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, ...]  # 128-dim or 768-dim
  }'
```

Expected response:
```json
{
  "success": true,
  "prediction": {
    "action": "UP",
    "confidence": 0.85,
    "q_values": {
      "UP": 125.3,
      "DOWN": 42.1,
      "LEFT": 78.5,
      "RIGHT": 91.2
    },
    "probabilities": {
      "UP": 0.85,
      "DOWN": 0.05,
      "LEFT": 0.07,
      "RIGHT": 0.03
    },
    "source": "pytorch_dqn_model"
  },
  "usedML": true,
  "method": "ml_dqn"
}
```

### 3. Update Frontend

Update your frontend to use the Hugging Face Spaces URL.

**In `frontend/dqn-agent.js`:**
```javascript
constructor() {
    // Use local ML inference server when running on localhost
    // Otherwise use Hugging Face Spaces ML server
    const isLocal = window.location.hostname === 'localhost' ||
                    window.location.hostname === '127.0.0.1';
    this.workerURL = isLocal
        ? 'http://localhost:5000'
        : 'https://YOUR-USERNAME-alvin-pacman-dqn.hf.space';

    // ... rest of constructor
}
```

### 4. Deploy Updated Frontend to Cloudflare

```bash
cd ..  # Back to project root
npm run deploy  # or npx wrangler deploy
```

---

## Monitoring & Debugging

### View Logs

**Web UI:**
1. Go to https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn
2. Click "Logs" tab
3. See real-time logs

**Look for:**
```
[+] Loading DQN model...
  Device: cpu
  Checkpoint epoch: 16
  Validation loss: 24211.89
[+] DQN model loaded successfully!
  Parameters: 296,068
```

### Common Issues

**Issue: Model file too large for Git**
```
remote: error: File checkpoints/best_dqn_model.pth is 1.13 MB; this exceeds GitHub's file size limit of 100 MB
```
**Fix:** Ensure Git LFS is tracking the file:
```bash
git lfs track "*.pth"
git add .gitattributes
git add checkpoints/best_dqn_model.pth
git commit --amend
git push --force
```

**Issue: Docker build fails**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```
**Fix:** Check `requirements.txt` syntax and ensure all packages are available

**Issue: Space shows "Building" forever**
```
Building... (stuck)
```
**Fix:** Check build logs for errors. Common issues:
- Missing files (models/dqn_network.py)
- Wrong paths in Dockerfile
- Python syntax errors

**Issue: API returns 500 Internal Server Error**
```
{"success": false, "error": "..."}
```
**Fix:** Check Space logs for Python exceptions. Common issues:
- Model file not loaded (missing checkpoint)
- Import errors (missing dependencies)
- CORS issues (already configured)

---

## Performance & Optimization

### Free Tier Specs

- **RAM:** 16 GB (plenty for PyTorch!)
- **CPU:** 2 vCPU
- **Storage:** 50 GB
- **GPU:** Optional (upgrade to PRO for free GPU)

### Expected Performance

- **Model loading:** ~3-5 seconds (one-time)
- **Inference time:** ~50-100ms per prediction
- **Memory usage:** ~1.5 GB
- **Concurrent requests:** Supported (gunicorn with 1 worker)

### Enable GPU Inference (PRO Tier)

Hugging Face PRO ($9/month) includes free GPU for Spaces.

**Update Dockerfile:**
```dockerfile
# Enable GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ... rest of Dockerfile

# Update device selection in inference_server.py
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Updating the Deployment

### Method 1: Git Push (Recommended)

```bash
cd alvin-pacman-dqn

# Make changes to files
# ...

# Commit and push
git add .
git commit -m "Update model or code"
git push

# HF automatically rebuilds and redeploys
```

### Method 2: Web UI

1. Go to https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn
2. Click "Files" tab
3. Edit or upload files
4. Space automatically rebuilds

---

## Custom Domain (Optional)

Hugging Face Spaces support custom domains:

1. Go to Space settings
2. Add custom domain: `ml.alvin-pacman.com`
3. Update DNS CNAME record
4. SSL certificate auto-provisioned

---

## Security

### CORS

Already configured in `inference_server.py`:
```python
from flask_cors import CORS
CORS(app)  # Allow requests from frontend
```

### HTTPS

Hugging Face provides automatic HTTPS with TLS 1.3.

### Rate Limiting

**Optional:** Add rate limiting for production:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("60 per minute")
def predict():
    ...
```

Add to `requirements.txt`:
```
Flask-Limiter>=3.5.0
```

---

## Cost Comparison

| Feature | Render Free | Render Starter | HF Spaces Free | HF Spaces PRO |
|---------|-------------|----------------|----------------|---------------|
| **Price** | $0 | $7/month | $0 | $9/month |
| **RAM** | 512 MB ❌ | 512 MB ❌ | 16 GB ✅ | 16 GB ✅ |
| **CPU** | Shared | 0.5 vCPU | 2 vCPU | 8 vCPU |
| **GPU** | ❌ | ❌ | ❌ | ✅ Free GPU |
| **Sleep** | 15 min | Never | Never ✅ | Never ✅ |
| **Build Time** | 15 min | 10 min | 5-10 min | 3-5 min |
| **Storage** | 1 GB | 10 GB | 50 GB ✅ | 1 TB |
| **Custom Domain** | ✅ | ✅ | ✅ | ✅ |

**Winner:** Hugging Face Spaces (16 GB RAM free tier!)

---

## Next Steps

1. ✅ Create Hugging Face account
2. ✅ Create new Space with Docker SDK
3. ✅ Upload deployment files
4. ✅ Wait for build to complete
5. ✅ Test health check endpoint
6. ✅ Test prediction endpoint
7. ✅ Update frontend to use HF Spaces URL
8. ✅ Deploy updated frontend to Cloudflare
9. ✅ Test end-to-end integration
10. 🔄 Train RL model and redeploy

---

## Troubleshooting Checklist

- [ ] Git LFS installed and configured
- [ ] `.gitattributes` includes `*.pth` tracking
- [ ] `checkpoints/best_dqn_model.pth` exists
- [ ] `models/dqn_network.py` exists
- [ ] `README.md` has YAML frontmatter with `sdk: docker`
- [ ] Dockerfile uses port 7860
- [ ] `requirements.txt` includes all dependencies
- [ ] Build logs show "DQN model loaded successfully!"
- [ ] Health endpoint returns `{"status": "ok"}`
- [ ] Frontend updated with correct HF Spaces URL

---

## Useful Commands

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space
huggingface-cli repo create alvin-pacman-dqn --type space --space_sdk docker

# Clone Space
git clone https://huggingface.co/spaces/YOUR-USERNAME/alvin-pacman-dqn

# Check Git LFS tracking
git lfs ls-files

# Test locally before deploying
cd ml-training
docker build -t alvin-dqn .
docker run -p 7860:7860 alvin-dqn

# Test health
curl http://localhost:7860/health

# Test prediction
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"vector": [...]}'
```

---

## Support

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces-overview
- **Git LFS Docs:** https://git-lfs.com/
- **Docker Docs:** https://docs.docker.com/
- **GitHub Issues:** https://github.com/YOUR-USERNAME/alvin/issues

---

## Summary

**Deployment Checklist:**
- [x] Create `.gitattributes` for Git LFS
- [x] Create `Dockerfile` with port 7860
- [x] Update `README.md` with YAML frontmatter
- [x] Create Hugging Face account
- [ ] Create new Space with Docker SDK
- [ ] Upload files to Space
- [ ] Wait for build
- [ ] Test health check
- [ ] Test predictions
- [ ] Update frontend URL
- [ ] Deploy frontend
- [ ] Test end-to-end

**Key Files:**
- `ml-training/Dockerfile` - Docker container configuration
- `ml-training/README.md` - Space description with metadata
- `ml-training/.gitattributes` - Git LFS tracking
- `ml-training/inference_server.py` - Flask API server
- `ml-training/requirements.txt` - Python dependencies
- `ml-training/checkpoints/best_dqn_model.pth` - Trained model

You're ready to deploy on Hugging Face Spaces! 🚀
