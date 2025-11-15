# Deploying ML Inference Server to Render.com

## Overview

This guide explains how to deploy the Alvin Pac-Man DQN inference server to Render.com.

**Why Render.com:**
- ✅ Supports Python/PyTorch (unlike Cloudflare Workers)
- ✅ Free tier available
- ✅ Built-in CLI for deployment
- ✅ Automatic HTTPS
- ✅ Easy integration with GitHub
- ✅ Supports long-running inference requests

## Architecture

```
Frontend (Cloudflare Workers)
    ↓ HTTPS
Render.com ML Server (Flask + PyTorch)
    ↓
DQN Model (best_dqn_model.pth)
    ↓
Predictions (UP/DOWN/LEFT/RIGHT)
```

---

## Prerequisites

1. **Render.com Account**
   - Sign up at https://render.com (free tier available)
   - Connect your GitHub account

2. **Render CLI** (optional, but recommended)
   ```bash
   # Install Render CLI
   npm install -g render-cli

   # Or using pip
   pip install render-cli
   ```

3. **Git Repository**
   - Your code must be in a GitHub repository
   - Ensure `checkpoints/best_dqn_model.pth` is committed

---

## Deployment Methods

### Method 1: Using Render Dashboard (Recommended for First Deployment)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Render.com deployment configuration"
   git push
   ```

2. **Create New Web Service on Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repository: `alvin`
   - Select branch: `claude/initial-setup-01Tmc3RekGmegeiG9vhiV8xQ`

3. **Configure the Service**

   **Basic Settings:**
   - Name: `alvin-ml-inference`
   - Region: `Oregon (US West)` (or closest to your users)
   - Branch: `claude/initial-setup-01Tmc3RekGmegeiG9vhiV8xQ`
   - Root Directory: `ml-training`

   **Build & Deploy:**
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 inference_server:app`

   **Instance Type:**
   - Free (512 MB RAM) - sufficient for inference
   - Or Starter ($7/month) for better performance

4. **Environment Variables** (if needed)
   - `PYTHON_VERSION`: `3.11`
   - Render automatically provides `PORT`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes for first deploy)
   - Monitor logs for model loading

### Method 2: Using render.yaml (Infrastructure as Code)

The `render.yaml` file in `ml-training/` is already configured.

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Add Render.com deployment configuration"
   git push
   ```

2. **Create Blueprint on Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Blueprint"
   - Select your repository
   - Render will auto-detect `render.yaml`
   - Click "Apply"

3. **Monitor Deployment**
   - Watch logs in Render dashboard
   - First deployment takes 5-10 minutes
   - Look for "DQN model loaded successfully!"

### Method 3: Using Render CLI

```bash
# Authenticate
render login

# Create service from render.yaml
render blueprint deploy

# Or manually create service
render create web \
  --name alvin-ml-inference \
  --region oregon \
  --plan free \
  --repo https://github.com/YOUR_USERNAME/alvin \
  --branch claude/initial-setup-01Tmc3RekGmegeiG9vhiV8xQ \
  --root ml-training \
  --build-command "pip install -r requirements.txt" \
  --start-command "gunicorn --bind 0.0.0.0:\$PORT --workers 1 --timeout 120 inference_server:app"
```

---

## Deployment Configuration

### render.yaml
```yaml
services:
  - type: web
    name: alvin-ml-inference
    env: python
    region: oregon
    plan: free
    branch: claude/initial-setup-01Tmc3RekGmegeiG9vhiV8xQ
    rootDir: ml-training
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 inference_server:app
    healthCheckPath: /health
```

### Gunicorn Configuration

**Why Gunicorn:**
- Production-grade WSGI server
- Better than Flask's built-in server
- Handles concurrent requests
- Timeout settings for ML inference

**Settings:**
- `--workers 1`: Single worker (PyTorch models aren't thread-safe)
- `--timeout 120`: 2-minute timeout (ML inference can be slow on free tier)
- `--bind 0.0.0.0:$PORT`: Bind to Render's assigned port

---

## Post-Deployment

### 1. Get Your Service URL

After deployment, Render provides a URL:
```
https://alvin-ml-inference.onrender.com
```

### 2. Test the Deployment

**Health Check:**
```bash
curl https://alvin-ml-inference.onrender.com/health
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
curl -X POST https://alvin-ml-inference.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.5, 0.3, 0.8, ...]  # 128-dim vector
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
    }
  },
  "usedML": true
}
```

### 3. Update Frontend

Update your frontend to use the Render.com URL:

**In `frontend/dqn-agent.js`:**
```javascript
// Change from localhost to Render.com
const ML_API_URL = 'https://alvin-ml-inference.onrender.com/api/predict';
```

**In Cloudflare Workers `worker/index.js`:**
```javascript
// Add Render.com endpoint for ML predictions
const ML_SERVER_URL = 'https://alvin-ml-inference.onrender.com/api/predict';

// Update fetch call
const mlResponse = await fetch(ML_SERVER_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ vector: featureVector })
});
```

### 4. Deploy Frontend to Cloudflare

```bash
cd ..  # Back to project root
npm run deploy  # or npx wrangler deploy
```

---

## Monitoring & Debugging

### View Logs

**Dashboard:**
- Go to https://dashboard.render.com
- Click on your service
- Click "Logs" tab

**CLI:**
```bash
render logs alvin-ml-inference
```

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

**Issue: Model not loading**
```
FileNotFoundError: checkpoints/best_dqn_model.pth
```
**Fix:** Ensure checkpoint is committed to Git:
```bash
git add ml-training/checkpoints/best_dqn_model.pth
git commit -m "Add trained DQN checkpoint"
git push
```

**Issue: Out of memory**
```
MemoryError: Cannot allocate memory
```
**Fix:** Upgrade to Starter plan ($7/month, 2 GB RAM)

**Issue: Timeout errors**
```
Worker timeout
```
**Fix:** Already configured with `--timeout 120`. If still happening, upgrade plan.

**Issue: Cold starts (first request slow)**
- Free tier services sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Upgrade to Starter plan for always-on service

---

## Performance Optimization

### Free Tier Limitations

- **RAM:** 512 MB (tight for PyTorch)
- **CPU:** Shared
- **Sleep:** After 15 min inactivity
- **Build time:** Up to 15 minutes

### Recommended: Starter Tier ($7/month)

- **RAM:** 2 GB (plenty for inference)
- **CPU:** Dedicated
- **Always on:** No sleep
- **Faster builds**

### Model Optimization

**Current model size:** 296,068 parameters (768-dim input)

**To reduce size:**
1. Train smaller model (128-dim input instead of 768-dim)
2. Quantize model (INT8 instead of FP32)
3. Prune unnecessary layers

**Update `training/train_dqn.py`:**
```python
config = {
    'input_dim': 128,  # Use 128-dim instead of 768-dim
    'hidden_dims': [128, 128],  # Smaller hidden layers
}
```

Retrain, then redeploy.

---

## Updating the Deployment

### Method 1: Auto-deploy (Recommended)

Render automatically deploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update model or code"
git push

# Render detects changes and auto-deploys
```

### Method 2: Manual Deploy

In Render Dashboard:
- Click "Manual Deploy" → "Deploy latest commit"

### Method 3: CLI

```bash
render deploy alvin-ml-inference
```

---

## Rollback

If deployment fails:

**Dashboard:**
- Click "Deploys" tab
- Find last successful deploy
- Click "Rollback"

**CLI:**
```bash
render rollback alvin-ml-inference
```

---

## Cost Estimates

### Free Tier
- ✅ $0/month
- ❌ 750 hours/month (sleeps after 15 min)
- ❌ 512 MB RAM (tight)
- ✅ Good for testing

### Starter Tier
- ✅ $7/month
- ✅ Always on
- ✅ 2 GB RAM (plenty)
- ✅ Better for production

### Custom Domains

Free custom domain support:
- `ml.alvin-pacman.com` → Render service

---

## Security

### CORS

Already configured in `inference_server.py`:
```python
from flask_cors import CORS
CORS(app)  # Allow requests from frontend
```

### HTTPS

Render provides automatic HTTPS with Let's Encrypt certificates.

### Rate Limiting

**TODO:** Add rate limiting for production:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("60 per minute")  # 60 requests/min per IP
def predict():
    ...
```

---

## Next Steps

1. ✅ Deploy to Render.com
2. ✅ Test health check
3. ✅ Test predictions
4. ✅ Update frontend URLs
5. ✅ Deploy frontend to Cloudflare
6. ✅ Test end-to-end
7. 🔄 Train RL model (`python training/train_rl.py`)
8. 🔄 Deploy updated RL model
9. 🔄 Compare performance: Behavioral Cloning vs RL

---

## Useful Commands

```bash
# Install Render CLI
npm install -g render-cli

# Login
render login

# List services
render services list

# View logs
render logs alvin-ml-inference --tail

# SSH into service (Starter tier+)
render ssh alvin-ml-inference

# Check service status
render status alvin-ml-inference

# Force redeploy
render deploy alvin-ml-inference --clear-cache
```

---

## Troubleshooting

### Service won't start
- Check logs for Python errors
- Verify `requirements.txt` has all dependencies
- Ensure checkpoint file exists

### Slow responses
- Free tier has shared CPU
- Upgrade to Starter for dedicated resources
- Consider model quantization

### Out of memory
- Free tier: 512 MB (very tight)
- Starter tier: 2 GB (recommended)
- Or optimize model size

### Model not found
- Ensure `checkpoints/best_dqn_model.pth` is in Git
- Check `rootDir: ml-training` in render.yaml
- Verify path in `inference_server.py`

---

## Support

- Render Docs: https://render.com/docs
- Render Status: https://status.render.com
- Community: https://community.render.com
- GitHub Issues: https://github.com/YOUR_USERNAME/alvin/issues

---

## Summary

**Deployment Checklist:**
- [x] Update `requirements.txt` (add Flask, gunicorn)
- [x] Update `inference_server.py` (use PORT env var)
- [x] Create `render.yaml`
- [x] Create `.renderignore`
- [x] Commit checkpoint to Git
- [ ] Push to GitHub
- [ ] Deploy to Render.com
- [ ] Test health check
- [ ] Update frontend URLs
- [ ] Deploy frontend
- [ ] Test end-to-end

**Key Files:**
- `ml-training/render.yaml` - Deployment config
- `ml-training/.renderignore` - Exclude large files
- `ml-training/inference_server.py` - Flask app with lazy loading
- `ml-training/requirements.txt` - Python dependencies
- `ml-training/checkpoints/best_dqn_model.pth` - Trained model

You're ready to deploy! 🚀
