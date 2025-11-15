# Local ML Testing Guide

## Overview

Run Alvin Pac-Man locally with the **ACTUAL trained neural network** (not heuristics).

This setup uses:
- ✅ **Real PyTorch model** (100% test accuracy, 133,508 parameters)
- ✅ **Local inference server** (Flask on port 5000)
- ✅ **Local frontend** (Node.js on port 3000)

## Why Local Testing?

**Problem with Cloudflare Workers:**
- Cloudflare Workers don't support custom ONNX models yet
- Current deployment uses hand-coded heuristics (fake "ML")
- Can't run actual PyTorch neural network

**Solution - Local Inference:**
- Loads real trained model from `ml-training/checkpoints/best_model.pth`
- Runs actual neural network predictions
- Tests if model actually works before finding production solution

## Quick Start

### Option 1: Automated (Windows)
```powershell
.\start-local.ps1
```

### Option 2: Manual

**Terminal 1 - ML Inference Server:**
```bash
cd ml-training
python inference_server.py
```

**Terminal 2 - Frontend Server:**
```bash
node local-server.js
```

**Open Browser:**
```
http://localhost:3000
```

## What to Expect

### ML Status Indicators
Once the game loads, check the **"🤖 ML Prediction Status"** section:

- **ML Active:** Should show **YES** (green)
- **Method:** Should show **ML_PYTORCH** (green)
- **Source:** Should show **pytorch_model**
- **Action Probabilities:** Real probabilities from neural network

### Console Logs
In browser console, you should see:
```
[ML] Used ML: true, Method: ml_pytorch
[ML] Source: pytorch_model
```

In ML server terminal, you'll see:
```
[Prediction] RIGHT (confidence: 87.3%)
[Prediction] UP (confidence: 91.2%)
...
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Browser (localhost:3000)                   │
│  ├─ Game UI                                 │
│  ├─ Feature Extraction (128-dim vectors)    │
│  └─ DQN Agent (calls ML server)             │
└──────────────┬──────────────────────────────┘
               │ POST /predict
               │ {vector: [128 floats]}
               ▼
┌─────────────────────────────────────────────┐
│  ML Inference Server (localhost:5000)       │
│  ├─ Load PyTorch Model                      │
│  ├─ Run Neural Network                      │
│  └─ Return Action + Probabilities           │
└─────────────────────────────────────────────┘
```

## API Endpoints

### ML Inference Server (port 5000)

**GET /health**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

**POST /predict**
Request:
```json
{
  "vector": [128 floats],
  "gameState": {...},
  "previousMoves": [...]
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "action": "UP",
    "confidence": 0.873,
    "probabilities": {
      "UP": 0.873,
      "DOWN": 0.052,
      "LEFT": 0.041,
      "RIGHT": 0.034
    },
    "source": "pytorch_model"
  },
  "usedML": true,
  "method": "ml_pytorch"
}
```

**GET /stats**
```json
{
  "model_type": "PacManPolicyNetwork",
  "input_dim": 128,
  "output_dim": 4,
  "hidden_dims": [256, 256, 128],
  "total_parameters": 133508,
  "device": "cpu"
}
```

## Troubleshooting

### Model Not Found
```
[ERROR] Model checkpoint not found: checkpoints/best_model.pth
```
**Solution:** Train the model first:
```bash
cd ml-training
python preprocessing/feature_extraction.py
python training/train.py
```

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:** Kill existing servers:
```bash
# Windows
taskkill /F /IM python.exe
taskkill /F /IM node.exe

# Mac/Linux
lsof -ti:5000 | xargs kill
lsof -ti:3000 | xargs kill
```

### CORS Errors in Browser
```
Access to fetch at 'http://localhost:5000' has been blocked by CORS
```
**Solution:** Install flask-cors:
```bash
pip install flask-cors
```

### ML Not Active in Game
1. Check browser console for errors
2. Verify ML server is running: `curl http://localhost:5000/health`
3. Check browser is on `localhost:3000` (not `127.0.0.1:3000`)

## Code Flow

### Frontend (dqn-agent.js)
```javascript
// Detects localhost and uses ML inference server
const isLocal = window.location.hostname === 'localhost';
this.workerURL = isLocal ? 'http://localhost:5000' : window.location.origin;
```

### ML Server (inference_server.py)
```python
# Load trained model
model = PacManPolicyNetwork(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Run inference
with torch.no_grad():
    outputs = model(features)
    probabilities = torch.softmax(outputs, dim=1)
    action = actions[torch.argmax(probabilities)]
```

## Performance Testing

### Compare Human vs ML
1. Play manually (WASD keys) for 5 games
2. Watch AI play with ML for 5 games
3. Check statistics in left sidebar

### Expected ML Performance
With 100% training accuracy, the ML should:
- Make intelligent decisions based on learned patterns
- Avoid ghosts effectively
- Collect pellets efficiently
- Complete levels if training data was successful

## Next Steps

### If ML Works Well
1. Find production solution for real ML:
   - Convert to TensorFlow.js (run in browser)
   - Deploy Flask server to cloud (Heroku, Railway, etc.)
   - Use Cloudflare Durable Objects with WebAssembly

### If ML Performs Poorly
1. Collect more/better training data
2. Improve feature engineering
3. Try different model architectures
4. Add more training samples from successful games

## Files Created

```
alvin/
├── local-server.js              # Frontend server (port 3000)
├── start-local.ps1              # Windows startup script
├── LOCAL_TESTING.md             # This file
└── ml-training/
    └── inference_server.py      # ML inference server (port 5000)
```

## Technical Details

### Model Specifications
- **Architecture:** 3-layer feedforward neural network
- **Input:** 128 dimensions (engineered features)
- **Hidden:** [256, 256, 128] neurons
- **Output:** 4 actions (UP, DOWN, LEFT, RIGHT)
- **Activation:** ReLU (hidden), Softmax (output)
- **Parameters:** 133,508 trainable
- **Training Accuracy:** 100%
- **Validation Accuracy:** 100%
- **Test Accuracy:** 100%

### Feature Vector (128 dimensions)
- [0-3]: Current direction (one-hot)
- [4-7]: Viable moves (one-hot)
- [8-11]: Pellet distances (normalized)
- [12-15]: Ghost distances (normalized)
- [16-19]: Power pellet distances
- [20-23]: Ghost vulnerability states
- [24-27]: Path quality scores (Dijkstra)
- [28-127]: Strategic context features

---

**Ready to test?** Open http://localhost:3000 and watch the ML section light up! 🎮🤖✨
