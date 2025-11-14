# Alvin - Architecture Documentation

## What is Alvin?

**Alvin** is an AI-powered Pac-Man game where a human player and an AI agent play simultaneously. The AI learns from the human's gameplay in real-time using Deep Q-Network (DQN) techniques and Cloudflare's AI infrastructure.

## Core Concept

1. **Human plays Pac-Man** on the left side using arrow keys
2. **AI avatar plays in parallel** on the right side, making its own decisions
3. **Human moves are vectorized** (converted to numerical representations) using Cloudflare AI
4. **AI predicts next moves** based on similar past game states
5. **Learning happens in real-time** as you play

## Project Structure

```
alvin/
├── frontend/                    # Client-side code (browser)
│   ├── index.html              # Game UI and layout
│   ├── game.js                 # Pac-Man game engine (450 lines)
│   ├── dqn-agent.js            # AI prediction client (197 lines)
│   └── vectorization.js        # Vectorization client (252 lines)
│
├── worker/                      # Server-side code (Cloudflare Worker)
│   └── index.js                # API endpoints + AI logic (297 lines)
│
├── dist/                        # Build output (auto-generated, git-ignored)
│   └── index.js                # Bundled worker with frontend embedded
│
├── bundle-worker.js            # Build script
├── wrangler.toml               # Cloudflare configuration
├── package.json                # Project dependencies
└── README.md                   # User documentation
```

## How It Works - Detailed Flow

### 1. Game Initialization

```
Browser loads → index.html → loads 3 JS files:
├── game.js          (creates two Pac-Man instances)
├── dqn-agent.js     (initializes AI prediction system)
└── vectorization.js (sets up move recording system)
```

### 2. Human Gameplay Loop

```
User presses arrow key
    ↓
game.js updates human player position
    ↓
game.js calls vectorizeMove()
    ↓
vectorization.js batches the move
    ↓
POST /api/vectorize (sends game state to worker)
    ↓
Worker calls Cloudflare AI (@cf/baai/bge-base-en-v1.5)
    ↓
AI generates 768-dimensional vector from game state
    ↓
Vector stored in Vectorize (similarity search database)
    ↓
Move also stored in KV (key-value store) for retrieval
```

### 3. AI Gameplay Loop (Parallel)

```
Every 200ms, game.js calls updateAI()
    ↓
dqn-agent.js calls predictNextMove()
    ↓
POST /api/predict (sends current AI game state)
    ↓
Worker vectorizes current state using Cloudflare AI
    ↓
Worker queries Vectorize for similar past states
    ↓
Worker runs DQN-inspired heuristics:
    - Avoid ghosts (negative reward)
    - Seek pellets (positive reward)
    - Avoid repeating moves (exploration)
    ↓
Returns best action (UP/DOWN/LEFT/RIGHT)
    ↓
AI player moves in that direction
    ↓
UI shows AI's confidence and next move
```

## Key Technologies

### Frontend (Browser)

- **HTML5 Canvas**: Renders two 400x400 pixel game boards
- **Vanilla JavaScript**: No frameworks, pure ES6+ classes
- **requestAnimationFrame**: Smooth 60 FPS rendering
- **Async/Await**: For API calls to backend

### Backend (Cloudflare Worker)

- **Cloudflare Workers**: Serverless edge computing
- **Workers AI**: Run ML models at the edge
  - Model: `@cf/baai/bge-base-en-v1.5` (text embeddings)
  - Output: 768-dimensional vectors
- **Vectorize**: Vector similarity search database
  - Stores game state embeddings
  - Finds similar past situations (cosine similarity)
- **KV Store**: Key-value storage
  - Stores full game state history with metadata
  - Fast global access

## Components Deep Dive

### 1. Game Engine (game.js)

**Responsibilities:**
- Maintain two separate game states (human + AI)
- Handle collision detection
- Render graphics on canvas
- Process keyboard input
- Update game at 10 FPS for movement

**Key Methods:**
- `createInitialState()`: Creates maze, pellets, ghosts
- `updatePlayer()`: Move player based on direction
- `updateGhosts()`: Simple random ghost AI
- `checkCollisions()`: Pellet collection, ghost contact
- `renderGame()`: Draw everything on canvas

### 2. DQN Agent (dqn-agent.js)

**Responsibilities:**
- Communicate with worker for predictions
- Cache predictions to reduce API calls
- Provide fallback logic when API fails
- Track move history

**Key Methods:**
- `predictNextMove()`: Get next action from AI
  - Throttled to max 5 predictions/second
  - Uses cache for repeated states
- `fallbackPrediction()`: Ghost avoidance heuristic
- `learnFromMove()`: Store human moves for training

### 3. Vectorization System (vectorization.js)

**Responsibilities:**
- Batch moves to reduce API calls
- Convert game states to text descriptions
- Calculate rewards for reinforcement learning
- Track statistics

**Key Methods:**
- `vectorizeGameState()`: Add move to batch
- `processBatch()`: Send batch to API
- `calculateReward()`: Compute reward signal
  - +points for score
  - -points for ghost proximity
  - +points for pellet collection

### 4. Worker API (worker/index.js)

**Endpoints:**

#### POST /api/vectorize
```javascript
Input:  { gameState: { playerX, playerY, ghosts, ... } }
Output: { success: true, vector: [0.123, -0.456, ...] }
```
Converts game state to 768D vector using AI model.

#### POST /api/predict
```javascript
Input:  { gameState: {...}, previousMoves: ["UP", "RIGHT"] }
Output: { success: true, prediction: { action: "UP", confidence: 0.85 } }
```
Predicts best next move based on game state and history.

#### POST /api/store-move
```javascript
Input:  { gameState: {...}, action: "UP", reward: 10 }
Output: { success: true, id: "move-12345-0.678" }
```
Stores human move with its vector in both Vectorize and KV.

#### POST /api/get-similar-moves
```javascript
Input:  { gameState: {...} }
Output: { success: true, similarMoves: [...] }
```
Finds top 5 most similar past game states.

## Build Process

### Development Workflow

```bash
# Source files remain clean and separate
frontend/index.html      (readable)
frontend/game.js         (readable)
worker/index.js          (readable)

# Build process
npm run build
    ↓
bundle-worker.js runs
    ↓
Reads all frontend files
    ↓
Reads worker source
    ↓
Concatenates: worker + static file constants
    ↓
Outputs: dist/index.js (43 KB)

# Deployment
npm run deploy
    ↓
Builds first (generates dist/index.js)
    ↓
wrangler deploys dist/index.js
    ↓
Live on Cloudflare edge network
```

### Why This Architecture?

1. **Source files stay clean**: worker/index.js is only 297 lines
2. **Build output is separate**: dist/ is git-ignored
3. **No build tools needed**: Just Node.js built-in fs module
4. **Single worker file**: Cloudflare Workers need everything in one file
5. **Easy to maintain**: Edit frontend files normally, build handles bundling

## Data Flow Example

Let's trace a single move:

```
1. User presses UP arrow
   ↓
2. game.js changes direction to "UP"
   ↓
3. Next frame (100ms later), player moves up
   ↓
4. vectorizeMove("UP") called with state:
   {
     playerX: 10,
     playerY: 9,
     ghosts: [{x:5,y:5}, ...],
     score: 100,
     pelletsLeft: 87
   }
   ↓
5. vectorization.js adds to batch
   ↓
6. After 1 second or 5 moves, batch sent to worker
   ↓
7. Worker converts to text:
   "Player at (10,9) facing UP. Ghosts: (5,5), (15,5), (5,15), (15,15). Pellets: 87. Score: 100"
   ↓
8. Cloudflare AI model processes text
   ↓
9. Returns vector: [0.234, -0.567, 0.891, ..., 0.123] (768 numbers)
   ↓
10. Vector stored in Vectorize with metadata
    ↓
11. Full state stored in KV
    ↓
12. AI can now query for similar states when in position (10,9)
```

## Cloudflare Resources Created

### KV Namespace
```
Name: GAME_STATE
ID: 441ecf79d508425da5e9de41f1ebb4e7
Purpose: Store game state snapshots
Size: ~1 KB per move
```

### Vectorize Index
```
Name: pacman-moves
Dimensions: 768
Metric: cosine similarity
Purpose: Find similar game situations
```

### Workers AI Binding
```
Model: @cf/baai/bge-base-en-v1.5
Purpose: Generate text embeddings
Input: Text description of game state
Output: 768-dimensional vector
```

## Performance Characteristics

- **Game renders at**: 60 FPS (browser)
- **Player moves at**: 10 FPS (game logic)
- **AI predicts at**: 5 Hz (200ms throttle)
- **Vectorization batches**: Every 5 moves or 1 second
- **Worker response time**: ~100-300ms
- **AI model inference**: ~50-150ms
- **Vectorize query**: ~10-50ms

## Learning Mechanism

The AI doesn't use a real neural network (yet), but uses a DQN-inspired approach:

1. **State Representation**: Game state → 768D vector
2. **Experience Replay**: Past moves stored in Vectorize
3. **Similarity Search**: Find similar past situations
4. **Q-Value Estimation**: Calculate scores for each action:
   - Ghost avoidance: Higher distance = higher score
   - Pellet seeking: Random exploration for now
   - Anti-repetition: Penalize recent moves
5. **Action Selection**: Choose action with highest score

**Future Enhancement**: Replace heuristics with actual neural network trained on human gameplay.

## Security & CORS

- **CORS enabled**: Allows browser to call worker from any origin
- **No authentication**: Free tier, public API
- **Rate limiting**: Cloudflare Workers have built-in limits
- **No sensitive data**: Just game states

## Cost Estimation

With Cloudflare's free tier:
- Workers: 100,000 requests/day FREE
- KV: 100,000 reads/day FREE
- Vectorize: 30M query dimensions/month FREE
- Workers AI: 10,000 neurons/day FREE

Typical usage per game session (5 minutes):
- ~150 player moves
- ~1,500 AI predictions
- ~150 vectorizations
- ~1,500 Vectorize queries

**Conclusion**: Easily within free tier for personal use.

## Common Issues & Solutions

### Issue: AI not moving
**Cause**: API calls failing
**Solution**: Check browser console for errors

### Issue: Slow AI responses
**Cause**: Cold start or network latency
**Solution**: First few moves are slower, then caches warm up

### Issue: Build fails
**Cause**: Missing frontend files
**Solution**: Ensure all files exist in frontend/

## Development Tips

1. **Test locally**: `npm run dev` (opens on localhost:8787)
2. **Check logs**: `wrangler tail` (view live worker logs)
3. **Debug vectors**: Open console, type `getVectorizationStats()`
4. **Modify AI**: Edit worker/index.js `calculateActionScore()`
5. **Change maze**: Edit game.js `generateWalls()`

## Next Steps for Enhancement

1. **Real DQN training**: Add neural network library
2. **Better AI model**: Train on collected human gameplay
3. **Multiplayer**: Use Durable Objects for shared game state
4. **Leaderboard**: Store high scores in D1 (SQL database)
5. **Analytics**: Track which strategies work best
6. **Mobile support**: Add touch controls
7. **Replay system**: Record and replay best games

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Author**: Built with Claude Code
