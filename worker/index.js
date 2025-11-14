/**
 * Alvin Pac-Man AI - Cloudflare Worker
 * Handles DQN agent, vectorization, and game state management
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // API Route handling
    if (url.pathname === '/api/vectorize') {
      return handleVectorize(request, env, corsHeaders);
    } else if (url.pathname === '/api/predict') {
      return handlePredict(request, env, corsHeaders);
    } else if (url.pathname === '/api/store-move') {
      return handleStoreMove(request, env, corsHeaders);
    } else if (url.pathname === '/api/get-similar-moves') {
      return handleSimilarMoves(request, env, corsHeaders);
    }

    // Static file serving (injected during build)
    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(HTML, {
        headers: { 'Content-Type': 'text/html' }
      });
    } else if (url.pathname === '/statistics.js') {
      return new Response(STATISTICS_JS, {
        headers: { 'Content-Type': 'application/javascript' }
      });
    } else if (url.pathname === '/game.js') {
      return new Response(GAME_JS, {
        headers: { 'Content-Type': 'application/javascript' }
      });
    } else if (url.pathname === '/dqn-agent.js') {
      return new Response(DQN_AGENT_JS, {
        headers: { 'Content-Type': 'application/javascript' }
      });
    } else if (url.pathname === '/vectorization.js') {
      return new Response(VECTORIZATION_JS, {
        headers: { 'Content-Type': 'application/javascript' }
      });
    } else if (url.pathname === '/favicon.ico') {
      return new Response('', { status: 204 });
    }

    return new Response('Not Found', { status: 404 });
  }
};

/**
 * Vectorize game state using Cloudflare AI
 */
async function handleVectorize(request, env, corsHeaders) {
  try {
    const { gameState } = await request.json();

    // Create a text representation of the game state for vectorization
    const stateText = serializeGameState(gameState);

    // Use Cloudflare AI for text embeddings
    const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
      text: stateText
    });

    return new Response(JSON.stringify({
      success: true,
      vector: embeddings.data[0]
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * DQN Agent prediction endpoint
 * Implements the "prefrontal cortex" for next step prediction
 */
async function handlePredict(request, env, corsHeaders) {
  try {
    const { gameState, previousMoves } = await request.json();

    // Get vectorized representation
    const stateText = serializeGameState(gameState);
    const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
      text: stateText
    });

    // Query similar past states using Vectorize
    const vector = embeddings.data[0];

    // Simple DQN-like decision making
    // In production, this would use a trained model
    const prediction = await predictNextMove(gameState, previousMoves, vector, env);

    return new Response(JSON.stringify({
      success: true,
      prediction,
      confidence: prediction.confidence
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Store human move with its vectorized state
 */
async function handleStoreMove(request, env, corsHeaders) {
  try {
    const { gameState, action, reward } = await request.json();

    const stateText = serializeGameState(gameState);
    const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
      text: stateText
    });

    const vector = embeddings.data[0];
    const id = `move-${Date.now()}-${Math.random()}`;

    // Store in Vectorize for similarity search
    if (env.VECTORIZE) {
      await env.VECTORIZE.upsert([{
        id,
        values: vector,
        metadata: { action, reward, timestamp: Date.now() }
      }]);
    }

    // Also store in KV for retrieval
    if (env.GAME_STATE) {
      await env.GAME_STATE.put(id, JSON.stringify({
        gameState,
        action,
        reward,
        vector
      }));
    }

    return new Response(JSON.stringify({
      success: true,
      id
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Find similar past moves using vectorization
 */
async function handleSimilarMoves(request, env, corsHeaders) {
  try {
    const { gameState } = await request.json();

    const stateText = serializeGameState(gameState);
    const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
      text: stateText
    });

    const vector = embeddings.data[0];

    // Query Vectorize for similar states
    let similarMoves = [];
    if (env.VECTORIZE) {
      const results = await env.VECTORIZE.query(vector, { topK: 5 });
      similarMoves = results.matches || [];
    }

    return new Response(JSON.stringify({
      success: true,
      similarMoves
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Serialize game state into text for vectorization
 */
function serializeGameState(state) {
  return `Player at (${state.playerX},${state.playerY}) facing ${state.direction}. ` +
    `Ghosts: ${state.ghosts.map(g => `(${g.x},${g.y})`).join(', ')}. ` +
    `Pellets remaining: ${state.pelletsLeft}. Score: ${state.score}`;
}

/**
 * DQN-inspired prediction logic
 * This is a simplified version - in production, use a trained model
 */
async function predictNextMove(gameState, previousMoves, vector, env) {
  const actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

  // Simple heuristic-based prediction
  // In a full DQN implementation, this would use neural network weights

  const scores = {};
  for (const action of actions) {
    scores[action] = calculateActionScore(gameState, action, previousMoves);
  }

  // Find best action
  let bestAction = actions[0];
  let maxScore = scores[bestAction];

  for (const action of actions) {
    if (scores[action] > maxScore) {
      maxScore = scores[action];
      bestAction = action;
    }
  }

  return {
    action: bestAction,
    confidence: maxScore,
    allScores: scores
  };
}

/**
 * Calculate heuristic score for an action
 */
function calculateActionScore(state, action, previousMoves) {
  let score = 0;

  // Avoid ghosts (negative reward)
  const nextPos = getNextPosition(state.playerX, state.playerY, action);
  for (const ghost of state.ghosts) {
    const distance = Math.abs(nextPos.x - ghost.x) + Math.abs(nextPos.y - ghost.y);
    if (distance < 3) {
      score -= 10 / (distance + 1);
    }
  }

  // Seek pellets (positive reward)
  score += Math.random() * 2; // Simplified - would check actual pellet positions

  // Avoid repeating recent moves (exploration)
  if (previousMoves && previousMoves.slice(-3).includes(action)) {
    score -= 1;
  }

  return score;
}

/**
 * Get next position based on action
 */
function getNextPosition(x, y, action) {
  switch (action) {
    case 'UP': return { x, y: y - 1 };
    case 'DOWN': return { x, y: y + 1 };
    case 'LEFT': return { x: x - 1, y };
    case 'RIGHT': return { x: x + 1, y };
    default: return { x, y };
  }
}
