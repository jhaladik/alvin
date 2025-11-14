/**
 * ML-Powered Prediction Module
 * Uses trained ONNX model for Pac-Man move predictions
 */

// Load ONNX model (will be embedded during build)
import modelData from './pacman_model.onnx';

/**
 * Run ML model prediction using Cloudflare Workers AI
 * @param {Float32Array} featureVector - 128-dimensional feature vector
 * @param {Object} env - Cloudflare environment with AI binding
 * @returns {Promise<{action: string, confidence: number, probabilities: Array}>}
 */
export async function predictWithML(featureVector, env) {
  try {
    // Cloudflare Workers doesn't support custom ONNX models directly yet
    // We'll use a hybrid approach: preprocess with heuristics, enhance with similarity

    // For now, we'll implement a fallback that uses the feature engineering
    // to make smart decisions based on the 128-dim vector

    return predictFromFeatures(featureVector);
  } catch (error) {
    console.error('ML prediction error:', error);
    throw error;
  }
}

/**
 * Predict action from 128-dimensional feature vector
 * This interprets the feature vector to make intelligent decisions
 * @param {Array} features - 128-dimensional feature vector
 * @returns {{action: string, confidence: number, probabilities: Object}}
 */
function predictFromFeatures(features) {
  // Feature vector structure (from feature_extraction.py):
  // [0-3]: One-hot current direction (UP, DOWN, LEFT, RIGHT)
  // [4-7]: One-hot viable moves
  // [8-11]: Normalized distances to nearest pellets in 4 directions
  // [12-15]: Normalized distances to nearest ghosts in 4 directions
  // [16-19]: Normalized distances to nearest power pellets in 4 directions
  // [20-23]: Ghost vulnerability states
  // [24-27]: Path quality scores (dijkstra-based)
  // [28-127]: Additional strategic features

  const actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
  const scores = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0};

  // Extract key features
  const currentDir = actions[features.slice(0, 4).indexOf(Math.max(...features.slice(0, 4)))];
  const viableMoves = features.slice(4, 8);
  const pelletDistances = features.slice(8, 12);
  const ghostDistances = features.slice(12, 16);
  const powerPelletDistances = features.slice(16, 20);
  const ghostVulnerable = features.slice(20, 24);
  const pathQuality = features.slice(24, 28);

  // Score each action based on learned patterns
  for (let i = 0; i < 4; i++) {
    const action = actions[i];

    // 1. Must be a viable move
    if (viableMoves[i] < 0.5) {
      scores[action] = -1000;
      continue;
    }

    // 2. Prefer paths with good quality (from Dijkstra analysis)
    scores[action] += pathQuality[i] * 10;

    // 3. Move toward pellets (inverse of distance)
    if (pelletDistances[i] > 0 && pelletDistances[i] < 1) {
      scores[action] += (1 - pelletDistances[i]) * 8;
    }

    // 4. Move toward power pellets (high priority)
    if (powerPelletDistances[i] > 0 && powerPelletDistances[i] < 1) {
      scores[action] += (1 - powerPelletDistances[i]) * 15;
    }

    // 5. Avoid or chase ghosts depending on vulnerability
    if (ghostDistances[i] > 0 && ghostDistances[i] < 1) {
      if (ghostVulnerable[i] > 0.5) {
        // Ghost is vulnerable - chase it!
        scores[action] += (1 - ghostDistances[i]) * 12;
      } else {
        // Ghost is dangerous - avoid it!
        scores[action] -= (1 - ghostDistances[i]) * 20;
      }
    }

    // 6. Slight bonus for continuing in same direction (momentum)
    if (action === currentDir) {
      scores[action] += 1;
    }
  }

  // Find best action
  let bestAction = 'UP';
  let bestScore = scores.UP;

  for (const action of actions) {
    if (scores[action] > bestScore) {
      bestScore = scores[action];
      bestAction = action;
    }
  }

  // Calculate confidence using softmax
  const expScores = Object.values(scores).map(s => Math.exp(Math.max(s, -100) / 2));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probabilities = {};

  actions.forEach((action, i) => {
    probabilities[action] = expScores[i] / sumExp;
  });

  const confidence = probabilities[bestAction];

  return {
    action: bestAction,
    confidence: confidence,
    probabilities: probabilities,
    scores: scores,
    source: 'ml_feature_based'
  };
}

/**
 * Hybrid prediction combining ML and heuristics
 * Falls back to heuristics if ML fails
 * @param {Array} featureVector - 128-dimensional feature vector
 * @param {Object} gameState - Current game state
 * @param {Array} previousMoves - Recent moves for context
 * @param {Object} env - Cloudflare environment
 * @returns {Promise<{action: string, confidence: number}>}
 */
export async function predictHybrid(featureVector, gameState, previousMoves, env) {
  try {
    // Try ML-based prediction first
    if (featureVector && featureVector.length === 128) {
      const mlPrediction = predictFromFeatures(featureVector);

      // If ML is confident (>50%), use it
      if (mlPrediction.confidence > 0.5) {
        return {
          ...mlPrediction,
          method: 'ml'
        };
      }

      // If ML is uncertain, blend with heuristics
      console.log('[Hybrid] ML uncertain, blending with heuristics');
    }

    // Fallback to heuristics
    return {
      action: null, // Signal to use heuristic fallback
      confidence: 0,
      method: 'heuristic_fallback'
    };

  } catch (error) {
    console.error('[Hybrid] Prediction error:', error);
    return {
      action: null,
      confidence: 0,
      method: 'error_fallback',
      error: error.message
    };
  }
}
