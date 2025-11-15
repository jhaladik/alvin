---
title: Alvin Pac-Man DQN Inference
emoji: 👻
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Alvin Pac-Man - DQN Inference Server

PyTorch Deep Q-Network (DQN) inference server for Alvin Pac-Man AI.

## Overview

This Space provides ML inference for the Alvin Pac-Man game, using a trained DQN model to predict optimal moves in real-time.

**Model Details:**
- **Type:** Deep Q-Network (DQN)
- **Learning:** Reward-based reinforcement learning
- **Input:** 768-dimensional feature vector
- **Output:** 4 actions (UP, DOWN, LEFT, RIGHT)
- **Parameters:** 296,068
- **Training:** Behavioral cloning + offline DQN

## API Endpoints

### Health Check
```bash
GET /health
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

### Predict Action
```bash
POST /predict
POST /api/predict
```

**Request:**
```json
{
  "vector": [0.5, 0.3, 0.8, ...],
  "gameState": { ... },
  "previousMoves": ["UP", "RIGHT"]
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "action": "UP",
    "confidence": 0.85,
    "probabilities": {
      "UP": 0.85,
      "DOWN": 0.05,
      "LEFT": 0.07,
      "RIGHT": 0.03
    },
    "q_values": {
      "UP": 125.3,
      "DOWN": 42.1,
      "LEFT": 78.5,
      "RIGHT": 91.2
    },
    "source": "pytorch_dqn_model"
  },
  "usedML": true,
  "method": "ml_dqn"
}
```

### Model Statistics
```bash
GET /stats
```

**Response:**
```json
{
  "model_type": "DQNNetwork",
  "learning_type": "Deep Q-Learning (Reward-Based)",
  "input_dim": 768,
  "output_dim": 4,
  "hidden_dims": [256, 256, 128],
  "total_parameters": 296068,
  "device": "cpu"
}
```

## Usage from Frontend

```javascript
// Call inference API
const response = await fetch('https://YOUR-SPACE.hf.space/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    vector: featureVector,
    gameState: currentGameState,
    previousMoves: recentMoves
  })
});

const data = await response.json();
console.log('Predicted action:', data.prediction.action);
console.log('Confidence:', data.prediction.confidence);
```

## Architecture

```
Frontend (Cloudflare Workers)
    ↓ HTTPS
Hugging Face Spaces (Flask + PyTorch)
    ↓
DQN Model (best_dqn_model.pth)
    ↓
Predictions (UP/DOWN/LEFT/RIGHT)
```

## Technical Details

**Stack:**
- Python 3.11
- PyTorch (CPU inference)
- Flask + Gunicorn
- CORS enabled

**Performance:**
- Inference time: ~50-100ms per prediction
- Memory usage: ~1.5 GB
- Concurrent requests: Supported (single worker)

## Links

- **Game:** [Alvin Pac-Man](https://alvin-pacman-ai.jhaladik.workers.dev)
- **GitHub:** [github.com/YOUR-USERNAME/alvin](https://github.com/YOUR-USERNAME/alvin)
- **Documentation:** See `ml-training/` directory

## License

MIT License - See LICENSE file for details

---

Built with PyTorch and deployed on Hugging Face Spaces 🤗
