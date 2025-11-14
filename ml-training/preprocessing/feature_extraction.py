"""
Feature Extraction for Pac-Man Game States
Converts raw game states into 128-dimensional feature vectors
"""
import numpy as np
import json
import os
from glob import glob

FEATURE_DIM = 128
GRID_SIZE = 20

def extract_features(game_state):
    """
    Extract 128-dimensional feature vector from game state

    Feature breakdown:
    - Player position: 2 (x, y normalized)
    - Player direction: 4 (one-hot: UP, DOWN, LEFT, RIGHT)
    - Ghosts (4 ghosts): 12 (x, y, scared per ghost)
    - Ghost relative distances: 4
    - Ghost directions: 8 (dx, dy per ghost)
    - Nearest pellet: 3 (distance, dx, dy)
    - Nearest power pellet: 3 (distance, dx, dy)
    - Game state: 6 (score, pellets left, lives, power mode, power timer, total pellets)
    - Spatial context: 25 (5x5 grid of pellet density)
    - Danger zones: 8 (8 directions, ghost threat level)
    - Wall proximity: 4 (distance to wall in 4 directions)
    - Tactical features: 10 (trapped, escape routes, convergence, etc.)
    - Padding: remaining to reach 128
    """
    features = []

    # === PLAYER FEATURES (6) ===
    player_x = game_state['playerX'] / GRID_SIZE
    player_y = game_state['playerY'] / GRID_SIZE
    features.append(player_x)
    features.append(player_y)

    # Direction one-hot encoding
    direction_map = {'UP': [1,0,0,0], 'DOWN': [0,1,0,0],
                     'LEFT': [0,0,1,0], 'RIGHT': [0,0,0,1]}
    direction = game_state.get('direction', 'RIGHT')
    features.extend(direction_map.get(direction, [0,0,0,1]))

    # === GHOST FEATURES (24) ===
    ghosts = game_state.get('ghosts', [])[:4]  # Max 4 ghosts

    # Pad to exactly 4 ghosts
    while len(ghosts) < 4:
        ghosts.append({'x': -1, 'y': -1, 'scared': False})

    # Ghost positions, states, and relative info
    for ghost in ghosts[:4]:
        gx = ghost['x'] / GRID_SIZE if ghost['x'] >= 0 else -1
        gy = ghost['y'] / GRID_SIZE if ghost['y'] >= 0 else -1
        scared = 1.0 if ghost.get('scared', False) else 0.0

        features.append(gx)
        features.append(gy)
        features.append(scared)

        # Relative distance (Manhattan)
        if ghost['x'] >= 0:
            dist = (abs(ghost['x'] - game_state['playerX']) +
                   abs(ghost['y'] - game_state['playerY'])) / GRID_SIZE
            features.append(min(dist, 1.0))
        else:
            features.append(1.0)  # Max distance if ghost doesn't exist

        # Relative direction
        if ghost['x'] >= 0:
            dx = (ghost['x'] - game_state['playerX']) / GRID_SIZE
            dy = (ghost['y'] - game_state['playerY']) / GRID_SIZE
            features.append(dx)
            features.append(dy)
        else:
            features.append(0.0)
            features.append(0.0)

    # === PELLET FEATURES (6) ===
    pellets = game_state.get('pellets', [])
    power_pellets = game_state.get('powerPellets', [])

    # Nearest pellet
    if pellets:
        min_dist = float('inf')
        nearest = None
        for pellet in pellets:
            dist = abs(pellet['x'] - game_state['playerX']) + abs(pellet['y'] - game_state['playerY'])
            if dist < min_dist:
                min_dist = dist
                nearest = pellet

        if nearest:
            features.append(min(min_dist / GRID_SIZE, 1.0))
            features.append((nearest['x'] - game_state['playerX']) / GRID_SIZE)
            features.append((nearest['y'] - game_state['playerY']) / GRID_SIZE)
        else:
            features.extend([1.0, 0.0, 0.0])
    else:
        features.extend([1.0, 0.0, 0.0])

    # Nearest power pellet
    if power_pellets:
        min_dist = float('inf')
        nearest = None
        for pellet in power_pellets:
            dist = abs(pellet['x'] - game_state['playerX']) + abs(pellet['y'] - game_state['playerY'])
            if dist < min_dist:
                min_dist = dist
                nearest = pellet

        if nearest:
            features.append(min(min_dist / GRID_SIZE, 1.0))
            features.append((nearest['x'] - game_state['playerX']) / GRID_SIZE)
            features.append((nearest['y'] - game_state['playerY']) / GRID_SIZE)
        else:
            features.extend([1.0, 0.0, 0.0])
    else:
        features.extend([1.0, 0.0, 0.0])

    # === GAME STATE FEATURES (6) ===
    score = min(game_state.get('score', 0) / 1000.0, 1.0)
    pellets_left = game_state.get('pelletsLeft', 0) / 250.0
    lives = game_state.get('lives', 3) / 3.0
    power_mode = 1.0 if game_state.get('powerMode', False) else 0.0
    power_timer = min(game_state.get('powerModeTimer', 0) / 100.0, 1.0)
    total_pellets = game_state.get('totalPellets', 250) / 250.0

    features.extend([score, pellets_left, lives, power_mode, power_timer, total_pellets])

    # === SPATIAL CONTEXT (25) ===
    # 5x5 grid of pellet density around player
    grid_size = 5
    cell_size = GRID_SIZE // grid_size
    pellet_grid = np.zeros((grid_size, grid_size))

    for pellet in pellets:
        rel_x = pellet['x'] - game_state['playerX'] + GRID_SIZE // 2
        rel_y = pellet['y'] - game_state['playerY'] + GRID_SIZE // 2

        if 0 <= rel_x < GRID_SIZE and 0 <= rel_y < GRID_SIZE:
            grid_x = min(rel_x // cell_size, grid_size - 1)
            grid_y = min(rel_y // cell_size, grid_size - 1)
            pellet_grid[grid_y, grid_x] += 1

    # Normalize
    max_density = pellet_grid.max()
    if max_density > 0:
        pellet_grid = pellet_grid / max_density

    features.extend(pellet_grid.flatten().tolist())

    # === DANGER ZONES (8) ===
    # Check 8 directions for ghost threats
    directions = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    for dx, dy in directions:
        danger = 0.0
        for dist in range(1, 6):  # Check up to 5 tiles
            check_x = game_state['playerX'] + dx * dist
            check_y = game_state['playerY'] + dy * dist

            for ghost in ghosts:
                if ghost['x'] >= 0 and ghost['x'] == check_x and ghost['y'] == check_y:
                    if not ghost.get('scared', False):
                        danger = 1.0 / dist  # Closer = more dangerous
                        break
            if danger > 0:
                break
        features.append(danger)

    # === WALL PROXIMITY (4) ===
    walls = game_state.get('walls', [])
    wall_set = set((w['x'], w['y']) for w in walls)

    # Check 4 cardinal directions
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        wall_dist = GRID_SIZE
        for dist in range(1, GRID_SIZE):
            check_x = game_state['playerX'] + dx * dist
            check_y = game_state['playerY'] + dy * dist

            if check_x < 0 or check_x >= GRID_SIZE or check_y < 0 or check_y >= GRID_SIZE:
                wall_dist = dist
                break
            if (check_x, check_y) in wall_set:
                wall_dist = dist
                break

        features.append(min(wall_dist / 5.0, 1.0))

    # === TACTICAL FEATURES (10) ===
    # 1. Trapped indicator (few escape routes)
    escape_routes = 0
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx = game_state['playerX'] + dx
        ny = game_state['playerY'] + dy
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
            (nx, ny) not in wall_set):
            escape_routes += 1
    features.append(escape_routes / 4.0)

    # 2-10. Additional tactical features (simplified)
    features.extend([0.0] * 9)  # Placeholder for additional tactical features

    # === PADDING ===
    # Pad or trim to exactly 128 features
    current_len = len(features)
    if current_len < FEATURE_DIM:
        features.extend([0.0] * (FEATURE_DIM - current_len))
    else:
        features = features[:FEATURE_DIM]

    return np.array(features, dtype=np.float32)

def process_training_data(input_file, output_file):
    """Process raw JSON data into features and labels"""
    print(f"Processing {input_file}...")

    with open(input_file, 'r') as f:
        data = json.load(f)

    samples = data.get('data', [])
    print(f"Found {len(samples)} samples")

    # Extract features and labels
    X = []  # Features
    y = []  # Actions
    metadata = []  # Additional info

    action_to_idx = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    for i, sample in enumerate(samples):
        try:
            game_state = sample['gameState']
            action = sample['action']
            reward = sample.get('reward', 0)

            # Extract features
            features = extract_features(game_state)
            X.append(features)

            # Convert action to index
            action_idx = action_to_idx.get(action, -1)
            if action_idx == -1:
                print(f"Warning: Unknown action '{action}' in sample {i}")
                continue
            y.append(action_idx)

            # Store metadata
            metadata.append({
                'reward': reward,
                'score': game_state.get('score', 0),
                'lives': game_state.get('lives', 0)
            })

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Processed {len(X)} samples successfully")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file,
                       X=X,
                       y=y,
                       metadata=metadata,
                       action_to_idx=action_to_idx)

    print(f"✓ Saved to {output_file}")

    return X, y, metadata

def main():
    print("=" * 60)
    print("Feature Extraction")
    print("=" * 60)
    print()

    # Find latest raw data file
    raw_files = glob("data/raw/training_data_*.json")
    if not raw_files:
        print("✗ No training data found!")
        print("Run: python data/fetch_training_data.py first")
        return 1

    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Using: {latest_file}")
    print()

    # Process
    output_file = "data/processed/features.npz"
    X, y, metadata = process_training_data(latest_file, output_file)

    # Statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Action distribution:")
    for action, idx in sorted([('UP', 0), ('DOWN', 1), ('LEFT', 2), ('RIGHT', 3)]):
        count = np.sum(y == idx)
        print(f"    {action}: {count} ({count/len(y)*100:.1f}%)")

    print("\n✓ Feature extraction complete!")
    print("\nNext step:")
    print("  python training/train.py")

    return 0

if __name__ == "__main__":
    exit(main())
