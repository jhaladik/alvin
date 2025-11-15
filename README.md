# Alvin Pac-Man AI

**AI-powered Pac-Man with Deep Q-Network (DQN) learning from human gameplay**

🎮 **Play:** https://alvin-pacman-ai.jhaladik.workers.dev
🤖 **ML Backend:** https://jozefh01-alvin-pacman-dqn.hf.space
📊 **Architecture:** Cloudflare Workers + Hugging Face Spaces + PyTorch DQN

---

## 🎯 What is This?

An intelligent Pac-Man game where AI learns to play by:
1. **Watching humans play** (Phase 1: Behavioral cloning)
2. **Playing thousands of games itself** (Phase 2: Reinforcement learning)
3. **Exceeding human skill level** (Future: Self-play optimization)

**Key Features:**
- ✅ Real-time AI predictions using PyTorch DQN
- ✅ 128-dimensional feature engineering
- ✅ Multi-dimensional reward system
- ✅ Headless environment for RL training
- ✅ Deployed on Hugging Face Spaces (16GB RAM free tier)
- ✅ Live Q-value visualization
- **Vector Similarity Search**: Find similar past game states using Cloudflare Vectorize
- **Reactive Frontend**: Simple, fast HTML5 Canvas rendering
- **Learning System**: AI learns from human gameplay patterns

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Pac-Man Game │  │  DQN Agent   │  │ Vectorization   │   │
│  │   (Human)    │  │  (AI Avatar) │  │    System       │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ API Calls
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cloudflare Worker                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ AI Vectorize │  │ DQN Predict  │  │  State Storage  │   │
│  │  (Embeddings)│  │   Logic      │  │   (KV Store)    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Cloudflare account (free tier works)
- Wrangler CLI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Install Wrangler (Cloudflare CLI)

```bash
npm install -g wrangler
```

### 3. Login to Cloudflare

```bash
wrangler login
```

### 4. Create Required Cloudflare Resources

#### Create KV Namespace

```bash
wrangler kv:namespace create "GAME_STATE"
```

Copy the generated namespace ID and update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "GAME_STATE"
id = "YOUR_KV_NAMESPACE_ID"  # Replace with your ID
```

#### Create Vectorize Index

```bash
wrangler vectorize create pacman-moves --dimensions=768 --metric=cosine
```

### 5. Test Locally

```bash
npm run dev
```

The game will be available at `http://localhost:8787`

### 6. Deploy to Cloudflare

```bash
npm run deploy
```

After deployment, you'll get a URL like: `https://alvin-pacman-ai.YOUR_SUBDOMAIN.workers.dev`

## 🎮 How to Play

1. Open the game in your browser
2. Use **Arrow Keys** to control Pac-Man
3. Watch the AI avatar play in parallel on the right side
4. The AI learns from your moves in real-time!

### Game Controls

- **↑ ↓ ← →** : Move Pac-Man
- **Reset Game** : Restart both human and AI games
- **Toggle AI** : Show/hide AI avatar

### Scoring

- Small pellet: 10 points
- Power pellet: 50 points
- Eating ghost (power mode): 200 points

## 🧠 How It Works

### 1. Human Gameplay Vectorization

When you move Pac-Man:
1. Game state is captured (position, ghosts, score)
2. State is sent to Cloudflare Worker
3. Worker uses `@cf/baai/bge-base-en-v1.5` model to create embeddings
4. Vector is stored in Cloudflare Vectorize for similarity search

### 2. DQN Agent Prediction

The AI avatar:
1. Receives current game state
2. Queries similar past states using vector similarity
3. Runs prediction algorithm (DQN-inspired heuristics)
4. Selects optimal action based on:
   - Ghost avoidance
   - Pellet collection
   - Exploration vs exploitation balance

### 3. Prefrontal Cortex Simulation

The "prefrontal cortex" refers to:
- Looking ahead at potential moves
- Evaluating consequences of each action
- Learning from past similar situations
- Making decisions based on predicted outcomes

## 📁 Project Structure

```
alvin/
├── frontend/
│   ├── index.html           # Main game UI
│   ├── game.js              # Pac-Man game engine
│   ├── dqn-agent.js         # DQN agent logic
│   └── vectorization.js     # Vectorization system
├── worker/
│   └── index.js             # Cloudflare Worker API
├── wrangler.toml            # Cloudflare configuration
├── package.json             # Dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables

You can add custom configuration in `.dev.vars` (for local development):

```env
# Add any custom environment variables here
AI_MODEL=@cf/baai/bge-base-en-v1.5
```

### Adjusting AI Behavior

Edit `worker/index.js` to modify:
- `calculateActionScore()`: Change how AI evaluates actions
- `predictNextMove()`: Modify prediction logic
- Ghost avoidance sensitivity
- Exploration rate

### Customizing Game

Edit `frontend/game.js` to adjust:
- `gridSize`: Maze dimensions (default: 20x20)
- `generateWalls()`: Maze pattern
- Ghost AI behavior
- Scoring system

## 📊 API Endpoints

### POST `/api/vectorize`
Vectorize a game state

**Request:**
```json
{
  "gameState": {
    "playerX": 10,
    "playerY": 10,
    "direction": "RIGHT",
    "ghosts": [...],
    "score": 100
  }
}
```

**Response:**
```json
{
  "success": true,
  "vector": [0.123, -0.456, ...]
}
```

### POST `/api/predict`
Get AI prediction for next move

**Request:**
```json
{
  "gameState": {...},
  "previousMoves": ["UP", "RIGHT", "RIGHT"]
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "action": "UP",
    "confidence": 0.85,
    "allScores": {...}
  }
}
```

### POST `/api/store-move`
Store a human move for learning

**Request:**
```json
{
  "gameState": {...},
  "action": "UP",
  "reward": 10
}
```

### POST `/api/get-similar-moves`
Find similar past game states

**Request:**
```json
{
  "gameState": {...}
}
```

**Response:**
```json
{
  "success": true,
  "similarMoves": [...]
}
```

## 🧪 Development

### Running Tests

```bash
# Test worker locally
wrangler dev

# Test with remote resources
wrangler dev --remote
```

### Debugging

Open browser console to see:
- Vectorization statistics: `getVectorizationStats()`
- DQN predictions
- API call logs

### Performance Monitoring

Monitor in Cloudflare Dashboard:
- Worker requests/second
- AI model inference time
- Vectorize query performance
- KV read/write operations

## 🚀 Deployment Options

### Option 1: Cloudflare Workers (Recommended)

```bash
npm run deploy
```

### Option 2: Cloudflare Pages

```bash
wrangler pages publish frontend
```

### Option 3: Custom Domain

Add to `wrangler.toml`:

```toml
[env.production]
name = "alvin-pacman-ai"
routes = [
  { pattern = "pacman.yourdomain.com", zone_name = "yourdomain.com" }
]
```

## 🎯 Roadmap

- [ ] Implement full neural network DQN training
- [ ] Add multiplayer mode
- [ ] Save/load trained models
- [ ] Leaderboard with Cloudflare D1
- [ ] More complex maze patterns
- [ ] Different AI personalities
- [ ] Mobile touch controls
- [ ] Game replay system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this project for learning and experimentation!

## 🙏 Acknowledgments

- Cloudflare for Workers AI platform
- Classic Pac-Man game design
- DQN algorithm inspiration from DeepMind

## 📞 Support

For issues or questions:
1. Check Cloudflare Workers documentation
2. Review API logs in Cloudflare dashboard
3. Open an issue in the repository

---

**Built with ❤️ using Cloudflare Workers, AI, and Vectorize**
