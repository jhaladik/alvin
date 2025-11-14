/**
 * Alvin Pac-Man AI - Cloudflare Worker
 * Handles DQN agent, vectorization, and game state management
 * NOW WITH REAL ML PREDICTIONS!
 */

import { predictHybrid } from './ml-predictor.js';

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
    } else if (url.pathname === '/api/export-training-data') {
      return handleExportTrainingData(request, env, corsHeaders);
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
    } else if (url.pathname === '/feature-engineering.js') {
      return new Response(FEATURE_ENGINEERING_JS, {
        headers: { 'Content-Type': 'application/javascript' }
      });
    } else if (url.pathname === '/path-planner.js') {
      return new Response(PATH_PLANNER_JS, {
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
 * Vectorize game state (DEPRECATED - now done client-side with feature engineering)
 * Kept for backward compatibility
 */
async function handleVectorize(request, env, corsHeaders) {
  try {
    const { gameState, vector } = await request.json();

    // If vector is pre-computed (new method), return it
    if (vector && Array.isArray(vector)) {
      return new Response(JSON.stringify({
        success: true,
        vector: vector
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Fallback: Old text-based method (for backward compatibility)
    const stateText = serializeGameState(gameState);
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
 * Uses pre-computed feature vectors and similarity search
 */
async function handlePredict(request, env, corsHeaders) {
  try {
    const { gameState, vector, previousMoves, currentStrategy } = await request.json();

    // Use pre-computed vector from feature engineering
    if (!vector || !Array.isArray(vector)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'No feature vector provided'
      }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Query Vectorize DB for similar past states with smart filtering
    // 4-tier strategy: strategy+success+powerMode → success+powerMode → powerMode → unfiltered
    let similarStates = [];
    let queryStrategy = 'unfiltered';

    if (env.VECTORIZE) {
      try {
        // Tier 1: Try strategy + success + powerMode (best - exact context match)
        if (currentStrategy) {
          const strategyFilter = {
            success: true,
            powerMode: gameState.powerMode,
            detectedStrategy: currentStrategy
          };

          let results = await env.VECTORIZE.query(vector, {
            topK: 10,
            filter: strategyFilter,
            returnMetadata: true
          });

          if (results.matches && results.matches.length >= 5) {
            similarStates = results.matches;
            queryStrategy = 'filtered_strategy';
            console.log(`[Strategy Learning] Using ${currentStrategy} strategy matches (${results.matches.length})`);
          }
        }

        // Tier 2: Try success + powerMode (good - context match without strategy)
        if (similarStates.length < 5) {
          const successFilter = {
            success: true,
            powerMode: gameState.powerMode
          };

          let results = await env.VECTORIZE.query(vector, {
            topK: 10,
            filter: successFilter,
            returnMetadata: true
          });

          if (results.matches && results.matches.length >= 5) {
            similarStates = results.matches;
            queryStrategy = 'filtered_success';
          }
        }

        // Tier 3: Try powerMode only (okay - basic context)
        if (similarStates.length < 5) {
          const powermodeFilter = {
            powerMode: gameState.powerMode
          };

          let results = await env.VECTORIZE.query(vector, {
            topK: 10,
            filter: powermodeFilter,
            returnMetadata: true
          });

          if (results.matches && results.matches.length >= 5) {
            similarStates = results.matches;
            queryStrategy = 'filtered_powermode';
          }
        }

        // Tier 4: Fall back to unfiltered (any similar state)
        if (similarStates.length < 5) {
          let results = await env.VECTORIZE.query(vector, {
            topK: 10,
            returnMetadata: true
          });
          similarStates = results.matches || [];
          queryStrategy = 'unfiltered';
        }
      } catch (e) {
        console.error('Vectorize query error:', e);
      }
    }

    // Make prediction using ML model (with heuristic fallback)
    let prediction;
    try {
      // Try ML prediction first
      const mlPrediction = await predictHybrid(vector, gameState, previousMoves, env);

      if (mlPrediction.action && mlPrediction.confidence > 0.3) {
        // ML made a confident prediction
        prediction = {
          ...mlPrediction,
          learnedFrom: similarStates ? similarStates.length : 0,
          usedML: true
        };
      } else {
        // Fall back to heuristics + similarity search
        prediction = await predictNextMove(
          gameState,
          previousMoves,
          vector,
          similarStates,
          env
        );
        prediction.usedML = false;
      }
    } catch (error) {
      console.error('ML prediction failed, using heuristics:', error);
      // Fall back to original heuristics
      prediction = await predictNextMove(
        gameState,
        previousMoves,
        vector,
        similarStates,
        env
      );
      prediction.usedML = false;
      prediction.mlError = error.message;
    }

    return new Response(JSON.stringify({
      success: true,
      prediction,
      confidence: prediction.confidence,
      similarStatesFound: similarStates.length,
      queryStrategy: queryStrategy, // Show which filter was used
      usedML: prediction.usedML, // NEW: Indicate if ML was used
      method: prediction.method || 'hybrid' // NEW: Show prediction method
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
 * Store human move with its pre-computed feature vector
 */
async function handleStoreMove(request, env, corsHeaders) {
  try {
    const { gameState, action, reward, vector, enrichedMetadata } = await request.json();

    // Validate vector
    if (!vector || !Array.isArray(vector)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'No feature vector provided'
      }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const id = `move-${Date.now()}-${Math.random()}`;

    // Prepare metadata with enriched context
    const metadata = {
      // Core action data
      action,
      reward,
      timestamp: Date.now(),
      playerX: gameState.playerX,
      playerY: gameState.playerY,

      // Enriched context (if provided)
      ...(enrichedMetadata || {})
    };

    // Store in Vectorize for similarity search
    if (env.VECTORIZE) {
      await env.VECTORIZE.upsert([{
        id,
        values: vector,
        metadata
      }]);
    }

    // Store in KV for analytics (move sequences, replay)
    if (env.GAME_STATE) {
      await env.GAME_STATE.put(id, JSON.stringify({
        gameState,
        action,
        reward,
        vector,
        metadata
      }), {
        expirationTtl: 86400 * 7 // 7 days retention
      });
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
 * Find similar past moves using pre-computed feature vector
 */
async function handleSimilarMoves(request, env, corsHeaders) {
  try {
    const { gameState, vector } = await request.json();

    // Validate vector
    if (!vector || !Array.isArray(vector)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'No feature vector provided'
      }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

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
 * Export all training data for ML training
 */
async function handleExportTrainingData(request, env, corsHeaders) {
  try {
    const trainingData = [];

    // Get all stored moves from KV
    if (env.GAME_STATE) {
      const list = await env.GAME_STATE.list();

      for (const key of list.keys) {
        try {
          const data = await env.GAME_STATE.get(key.name, 'json');
          if (data) {
            trainingData.push({
              id: key.name,
              gameState: data.gameState,
              action: data.action,
              reward: data.reward,
              vector: data.vector,
              metadata: data.metadata,
              timestamp: key.metadata?.timestamp || Date.now()
            });
          }
        } catch (e) {
          console.error(`Error reading ${key.name}:`, e);
        }
      }
    }

    return new Response(JSON.stringify({
      success: true,
      count: trainingData.length,
      data: trainingData,
      exportDate: new Date().toISOString()
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
 * Uses similarity search to learn from past successful moves
 */
async function predictNextMove(gameState, previousMoves, vector, similarStates, env) {
  const actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

  // Start with heuristic-based scores
  const scores = {};
  for (const action of actions) {
    scores[action] = calculateActionScore(gameState, action, previousMoves);
  }

  // If we have similar states from the database, learn from them
  if (similarStates && similarStates.length > 0) {
    const actionRewards = { UP: [], DOWN: [], LEFT: [], RIGHT: [] };

    // Collect rewards for each action from similar states
    for (const match of similarStates) {
      const action = match.metadata?.action;
      const reward = match.metadata?.reward;
      const similarity = match.score || 0; // Cosine similarity score

      if (action && reward !== undefined && actionRewards[action]) {
        // Weight by similarity - more similar states have more influence
        actionRewards[action].push(reward * similarity);
      }
    }

    // Apply learning from similar states
    for (const action of actions) {
      if (actionRewards[action].length > 0) {
        // Average reward weighted by similarity
        const avgReward = actionRewards[action].reduce((a, b) => a + b, 0) /
                         actionRewards[action].length;

        // Boost score based on learned rewards (with learning rate)
        const learningRate = 0.3;
        scores[action] += avgReward * learningRate;
      }
    }
  }

  // Find best action
  let bestAction = actions[0];
  let bestActionScore = scores[bestAction];

  for (const action of actions) {
    if (scores[action] > bestActionScore) {
      bestActionScore = scores[action];
      bestAction = action;
    }
  }

  // Calculate multiple confidence metrics for better transparency

  // 1. Score spread confidence (gap between best and worst)
  const scoreValues = Object.values(scores);
  const minScore = Math.min(...scoreValues);
  const maxScore = Math.max(...scoreValues);
  const scoreRange = maxScore - minScore;
  const spreadConfidence = scoreRange > 0 ? Math.min((maxScore - minScore) / 10, 1) : 0.5;

  // 2. Decision clarity (gap between best and second-best)
  const sortedScores = scoreValues.sort((a, b) => b - a);
  const gapToSecond = sortedScores.length > 1 ? sortedScores[0] - sortedScores[1] : 0;
  const clarityConfidence = Math.min(gapToSecond / 5, 1);

  // 3. Softmax-style probabilistic confidence
  const expScores = scoreValues.map(s => Math.exp(s / 2)); // Temperature = 2
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const bestExpIndex = scoreValues.indexOf(maxScore);
  const softmaxConfidence = expScores[bestExpIndex] / sumExp;

  // 4. Combined confidence (weighted average)
  const finalConfidence = (spreadConfidence * 0.4) + (clarityConfidence * 0.3) + (softmaxConfidence * 0.3);

  // 5. Decision quality (is the best choice clearly better?)
  const decisionQuality = maxScore > 0 && minScore < 0 ? 'excellent' :
                          gapToSecond > 2 ? 'good' :
                          gapToSecond > 0.5 ? 'fair' : 'poor';

  return {
    action: bestAction,
    confidence: finalConfidence,
    allScores: scores,
    learnedFrom: similarStates ? similarStates.length : 0,
    metrics: {
      spreadConfidence: spreadConfidence,
      clarityConfidence: clarityConfidence,
      softmaxConfidence: softmaxConfidence,
      decisionQuality: decisionQuality,
      scoreRange: scoreRange,
      gapToSecond: gapToSecond,
      bestScore: maxScore,
      worstScore: minScore
    }
  };
}

/**
 * Calculate heuristic score for an action
 */
function calculateActionScore(state, action, previousMoves) {
  let score = 0;

  const nextPos = getNextPosition(state.playerX, state.playerY, action);

  // 1. CRITICAL: Avoid walls (impossible moves)
  if (state.walls && state.walls.some(w => w.x === nextPos.x && w.y === nextPos.y)) {
    return -10000; // Impossible move - strongly penalize
  }

  // 2. Check bounds
  if (nextPos.x < 0 || nextPos.x >= 20 || nextPos.y < 0 || nextPos.y >= 20) {
    return -10000; // Out of bounds
  }

  // 3. Avoid ghosts (negative reward)
  for (const ghost of state.ghosts) {
    const distance = Math.abs(nextPos.x - ghost.x) + Math.abs(nextPos.y - ghost.y);
    if (distance < 3) {
      score -= 10 / (distance + 1);
    }
  }

  // 4. Seek pellets (positive reward)
  // Check if there's a pellet at next position
  if (state.pellets && state.pellets.some(p => p.x === nextPos.x && p.y === nextPos.y)) {
    score += 10; // Pellet at this position
  } else if (state.powerPellets && state.powerPellets.some(p => p.x === nextPos.x && p.y === nextPos.y)) {
    score += 50; // Power pellet at this position
  } else {
    score += Math.random() * 2; // Small exploration bonus
  }

  // 5. Avoid repeating recent moves (exploration)
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
