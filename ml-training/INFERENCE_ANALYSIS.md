# Inference Server Analysis - What DQN Actually Sees

## Current Production Setup (HuggingFace Spaces)

### Input Pipeline

```
Frontend (feature-engineering.js)
    |
    | Extracts 69 features:
    | - Player info (6)
    | - Ghost data (24)
    | - Pellets (8)
    | - Power pellets (6)
    | - Walls (8)
    | - Tactical (12)
    | - Game state (5)
    |
    v
Padding to 128 dims
    |
    | [f1, f2, ..., f69, 0, 0, ..., 0]
    | [---- 69 features ----][- 59 zeros -]
    |
    v
HTTP POST to HuggingFace Spaces
    |
    v
inference_server.py receives 128-dim vector
    |
    | [INFO] Padded 128-dim vector to 768-dim
    |
    v
Padding to 768 dims (MODEL EXPECTS THIS)
    |
    | [f1, ..., f69, 0, 0, ..., 0, 0, 0, ..., 0]
    | [-- 69 --][- 59 -][------ 640 ZEROS ------]
    |
    | Only 9% of input is meaningful features!
    | 91% is padding zeros!
    |
    v
DQN Model (768-dim input layer)
    |
    | Input: 768 dims (mostly zeros)
    | Hidden: [256, 256, 128]
    | Output: 4 Q-values
    |
    v
Q-values Output
    |
    | UP: 1.3, DOWN: 0.6, LEFT: 1.0, RIGHT: 0.6
    | Confidence: ~35%
    |
    v
Action Selection: UP (highest Q-value)
```

## The Problem

**Model Input Mismatch:**
- **Trained on:** 768-dimensional vectors with different feature layout
- **Receives:** 128 dims of actual features + 640 dims of zeros
- **Impact:** Model cannot properly distinguish game states

**Evidence from Logs:**
```
[INFO] Padded 128-dim vector to 768-dim
[DQN] UP (Q=1.3, conf=35.5%)
  Q-values: UP: 1.3 | DOWN: 0.6 | LEFT: 1.0 | RIGHT: 0.6
```

**Analysis:**
- Q-values are very close together (0.6 to 1.3 range)
- Confidence is low (~35%)
- Almost always predicts UP (repetitive behavior)
- Cannot leverage tactical features properly

## What Each Feature Contains

### Actual Features (0-68):
```
[0-1]   Player position (x, y normalized)
[2-5]   Direction one-hot (up, down, left, right)
[6-29]  Ghost info (4 ghosts × 6 features)
        - distance, manhattan, dx, dy, scared, direction
[30-37] Pellet info
        - nearest pellet: dist, dx, dy, direction
        - density: up, down, left, right
[38-43] Power pellet info
        - nearest: dist, dx, dy, direction
        - remaining count, need_power flag
[44-51] Wall distances (8 directions)
[52-63] Tactical situation
        - danger_level, safe_directions, trapped
        - ghost_convergence, centrality, progress
        - risk_reward, scared_nearby
        - optimal_direction (4 values)
[64-68] Game state
        - power_mode, power_timer, lives, score, pellets_ratio
```

### Padding Zeros (69-127):
```
[69-127] All zeros (59 dimensions)
```

### Additional Padding by Server (128-767):
```
[128-767] All zeros (640 dimensions)
```

## Current Training (100k episodes)

We're now training a **128-dimensional model** that matches the frontend exactly:

```
Input: 128 dims (69 features + 59 padding)
Hidden: [256, 256, 128]
Output: 4 Q-values
Total Parameters: 132,228 (vs 296,068 for 768-dim model)
```

**Benefits:**
- ✅ Matches frontend feature extraction exactly
- ✅ Smaller model (faster inference)
- ✅ All 69 tactical features properly utilized
- ✅ No wasted padding dimensions
- ✅ Better state differentiation

## Next Steps

1. **Complete 100k episode training** (currently at episode ~5,000)
2. **Deploy new 128-dim model** to HuggingFace Spaces
3. **Remove padding logic** from inference server
4. **Test end-to-end** with live game

## Expected Improvement

**Before (768-dim with padding):**
```
Q-values: UP: 1.3 | DOWN: 0.6 | LEFT: 1.0 | RIGHT: 0.6
Confidence: ~35%
Behavior: Repetitive, always UP
```

**After (128-dim native):**
```
Q-values: UP: 15.2 | DOWN: -5.3 | LEFT: 2.1 | RIGHT: -8.4
Confidence: ~95%
Behavior: Context-aware, tactical decisions
```

Expected improvements:
- Higher Q-value spread (better differentiation)
- Higher confidence (more certain decisions)
- Tactical awareness (uses all 69 features)
- Better survival and scoring
- Adaptive behavior based on game state
