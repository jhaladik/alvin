# Google Colab Fixed Maze Training Guide

## What Changed

**Previous Training Versions**:
1. **Random Maze** (82k episodes): -609 avg reward, high variance
2. **Fixed Closed Maze** (98k episodes): +352 avg reward, but Pac-Man was trapped!

**NEW: Fixed Maze with Opening** (Current):
- Uses **same maze layout** every episode
- **Ghost house has opening** at top - Pac-Man can escape!
- Previous maze had a bug: center was completely closed box
- More consistent learning (like classic Pac-Man)
- Expected: Even BETTER than +352 reward (no more trapped Pac-Man!)

## Why Fixed Maze is Better

1. **Consistent Difficulty**: Every episode has same maze difficulty
2. **Clearer Learning**: Can see AI actually learning optimal paths
3. **Reproducible**: Same strategies work every time
4. **Like Classic Pac-Man**: Real Pac-Man has fixed maze layout
5. **Lower Variance**: Reduces luck factor, focuses on skill

## CPU Results (Fixed Maze, 21k episodes so far)

- **Best Avg Reward**: -1,193 (vs -609 from Colab random maze)
- **Best Games**: 1,540 points (episode 18,300), 1,520 points (episode 20,400)
- **Learning Trend**: Clear improvement from -1,430 → -1,193
- **Consistency**: Most games 100-400 points, occasional 1,500+ breakthroughs

## How to Run on Google Colab

### Option 1: Upload Single File (Simplest)

1. Go to Google Colab: https://colab.research.google.com/
2. Create new notebook
3. Upload `ml-training/colab_train.py` to Colab
4. Run this cell:

```python
!python colab_train.py
```

5. Training will start from scratch (no pre-trained model needed)
6. Monitor progress in output

### Option 2: Clone from GitHub

1. In Colab, run:

```python
!git clone https://github.com/jhaladik/alvin.git
%cd alvin/ml-training
!python colab_train.py
```

2. This gets latest version with fixed maze

### Training Configuration

**Default Settings** (in `colab_train.py`):
- Episodes: **20,000** (reduced from 100k for faster training)
- Batch size: 64
- Learning rate: 0.0001
- Epsilon: 1.0 → 0.01 (decay 0.999 - faster for 20k episodes)
- Hidden layers: [256, 256, 128]
- Input: 128 dimensions
- Output: 4 actions (UP, DOWN, LEFT, RIGHT)

**Maze**: Fixed 20×20 grid with:
- Horizontal bars
- Vertical columns
- **Center ghost house with opening** (Pac-Man can escape through top!)
- Corner blocks
- 4 power pellets in corners

### Expected Training Time

**Google Colab GPU (T4)**:
- ~1-2 minutes per 1,000 episodes
- **20k episodes: ~30-40 minutes total**

**Local CPU**:
- ~5-7 minutes per 1,000 episodes
- **20k episodes: ~2-2.5 hours total**

### Monitoring Progress

Watch for these metrics:
- **Avg (100)**: Average reward over last 100 episodes
- **Episode scores**: Raw points (higher = better)
- **Epsilon**: Exploration rate (decreases over time)
- **Loss**: Training loss (should generally decrease)

**Good Signs**:
- Avg reward improving: -1,400 → -1,200 → -1,000
- Occasional high-score games (1,000+ points)
- Loss stabilizing or decreasing
- Epsilon decreasing toward 0.01

**Bad Signs**:
- Avg reward not improving after 20k episodes
- All games scoring <100 points
- Loss exploding (>100,000)

### Checkpoints

Colab saves checkpoints automatically:
- `best_rl_model_colab.pth`: Best average reward so far
- Includes: model weights, episode number, avg reward, epsilon

To download checkpoint from Colab:

```python
from google.colab import files
files.download('best_rl_model_colab.pth')
```

### Comparing Results

After training completes, compare:

**Random Maze (Previous Colab)**:
- Best: Episode 51k, +250 avg reward
- Final: Episode 82k, -609 avg reward
- High variance

**Fixed Maze (New Colab)**:
- Expected: Consistent improvement
- Lower variance
- More reproducible strategies

**Fixed Maze (CPU, in progress)**:
- Best: Episode 19k, -1,193 avg reward
- Breakthrough games: 1,540 points
- Clear learning trend

### Deployment

After training:

1. Download the best checkpoint
2. Copy to `ml-training/checkpoints/best_rl_model.pth`
3. Deploy to Hugging Face Spaces (see main README)
4. Test on live Pac-Man game

### Troubleshooting

**Issue**: Training stuck at -1,400 reward
**Solution**: Let it run longer (needs 15k-20k episodes to break through)

**Issue**: Loss exploding
**Solution**: Reduce learning rate to 0.00005

**Issue**: Colab disconnects
**Solution**:
- Colab free tier: 12-hour limit
- Checkpoints saved every 100 episodes
- Can resume from checkpoint if disconnected

**Issue**: Out of memory
**Solution**: Reduce batch size from 64 to 32

### Next Steps After Training

1. **Analyze Results**:
   - Check final avg reward
   - Compare to CPU fixed maze training
   - Review learning curve

2. **Deploy Best Model**:
   - Upload to Hugging Face Spaces
   - Test on live game
   - Compare to previous random maze model

3. **Future Improvements**:
   - Try different maze layouts
   - Tune hyperparameters
   - Experiment with reward structure

## Key Files

- `colab_train.py`: Standalone Colab training script (UPDATED with fixed maze)
- `environment/pacman_env.py`: Main environment (also has fixed maze)
- `train_rl_headless.py`: Local CPU training script
- `FIXED_MAZE_ANALYSIS.md`: Analysis of why fixed maze is better

## Success Criteria

Training is successful if:
✅ Avg reward improves from -1,400 to < -1,000
✅ Achieves consistent 800-1,200 point games
✅ Occasional 1,500+ point breakthroughs
✅ Variance decreases (scores cluster together)
✅ Strategy looks consistent (not random)

Training needs more time if:
⏳ Still at -1,400 after 20k episodes (try 50k)
⏳ High variance (0-2000 swings) - wait for convergence
⏳ Loss still high (>10,000) - needs more training
