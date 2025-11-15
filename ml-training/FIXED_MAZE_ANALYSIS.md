# Fixed Maze Analysis - Solving the Variance Problem

## Problem Identified

During GPU training at episode ~70,000, we observed massive score variance:

```
Episode 69,500: 1,770 points (reward: +428)
Episode 69,600: 120 points (reward: -1,399)
Episode 70,800: 2,030 points (reward: +429)
Episode 72,000: 130 points (reward: -1,380)
```

**Root Cause**: Random maze generation in `environment/pacman_env.py:88`

```python
# OLD CODE - Generated different maze every episode
if random.random() < 0.5:
    self.walls.add((x, y))
```

## Why This Prevented Learning

1. **Environment Randomness**: Some randomly generated mazes were inherently easier
   - More open space → easier pellet collection
   - Fewer walls → better ghost escape routes
   - Lucky layouts → 2,000+ point games

2. **No Consistent Strategy**: Classic Pac-Man has a FIXED maze
   - Players learn optimal paths through THAT specific layout
   - Expert players memorize patterns for the same maze
   - Our AI couldn't learn maze-specific strategies

3. **High Variance ≠ Learning**:
   - 2,030 → 130 point swings suggest luck, not skill
   - Robust learning should produce consistent scores
   - The AI was encountering different difficulty levels each episode

## Solution: Fixed Maze Layout

Replaced random generation with deterministic classic Pac-Man inspired pattern:

```python
# NEW CODE - Same maze every episode
# Horizontal walls
for x in range(3, 8):
    self.walls.add((x, 3))
    self.walls.add((x, self.maze_height - 4))

# Vertical walls, center structure, corner blocks
# ... (fixed positions)
```

## Expected Training Improvements

### With Fixed Maze, AI Should:

✅ **Consistent Scores**: All episodes ~1,200-1,500 points once learned
✅ **Clear Learning Curve**: Gradual improvement from -1,400 → +1,000 → +1,500
✅ **Low Variance**: ±100 points, not ±1,900 points
✅ **Reproducible Strategy**: Same decisions in same situations
✅ **Maze-Specific Optimization**: Learn optimal paths, safe zones, pellet routes

### If Previous High Scores Were Luck:

❌ 2,000+ point games will disappear (they were easy random mazes)
❌ Scores will stabilize lower (~800-1,200 range)
❌ But variance will be MUCH lower and consistent

## Training Comparison

### Random Maze Training (OLD)
- Episode 72,000: -885 avg reward
- Max individual: +430 (2,030 points)
- Min individual: -1,399 (120 points)
- **Variance: 1,910 points** ⚠️

### Fixed Maze Training (NEW)
- TBD - Restart training from scratch
- Expected: Lower max scores, but MUCH more consistent
- Goal: Variance < 200 points with gradual improvement

## Next Steps

1. ✅ Updated `pacman_env.py` with fixed maze
2. ⏳ Restart CPU training with fixed maze
3. ⏳ Restart Colab GPU training with fixed maze
4. ⏳ Compare learning curves (should be smoother)
5. ⏳ Verify if AI can consistently achieve 1,000+ points on SAME maze

## Technical Details

**File Modified**: `ml-training/environment/pacman_env.py:71-135`

**Changes**:
- Removed `random.random()` from wall generation
- Added deterministic horizontal bars (lines 87-92)
- Added deterministic vertical columns (lines 95-100)
- Added center T-shaped structure (lines 103-110)
- Added corner blocks (lines 113-118)
- Fixed power pellet positions (lines 127-132)

**Impact**: All downstream training scripts automatically use fixed maze (no code changes needed)

## Success Criteria

Training is TRULY learning if with fixed maze:
1. Average reward improves episode-to-episode
2. Variance decreases (scores cluster together)
3. Strategy becomes consistent (same paths in same situations)
4. Final scores: 1,200-1,500 points consistently

Training was LUCKY before if with fixed maze:
1. Scores stabilize at ~800 (lower than 2,030 max)
2. Variance remains high initially but decreases
3. Clear that 2,000+ games were outlier easy mazes
