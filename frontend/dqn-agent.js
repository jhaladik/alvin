/**
 * DQN Agent with Prefrontal Cortex Prediction
 * Communicates with Cloudflare Worker for AI predictions
 */

class DQNAgent {
    constructor() {
        // Change this to your worker URL when deployed
        this.workerURL = window.location.origin;

        this.previousMoves = [];
        this.maxHistoryLength = 10;
        this.predictionCache = new Map();
        this.lastPredictionTime = 0;
        this.predictionInterval = 200; // milliseconds
    }

    reset() {
        this.previousMoves = [];
        this.predictionCache.clear();

        // Reset path planner too
        if (window.pathPlanner) {
            window.pathPlanner.reset();
        }
    }

    async predictNextMove(gameState) {
        const now = Date.now();

        // Throttle predictions to avoid overwhelming the API
        if (now - this.lastPredictionTime < this.predictionInterval) {
            return this.getLastPrediction();
        }

        this.lastPredictionTime = now;

        // Exploration vs Exploitation
        const explorationRate = window.gameStats ?
            window.gameStats.learningMetrics.explorationRate : 0.1;

        // Occasionally explore (random move) to discover new strategies
        if (Math.random() < explorationRate) {
            const randomPrediction = this.fallbackPrediction(gameState);
            randomPrediction.explored = true;

            // Track exploration
            if (window.gameStats) {
                window.gameStats.learningMetrics.totalPredictions++;
            }

            return randomPrediction;
        }

        // === PATH PLANNING LAYER (Strategic) ===
        // Peek at planned move (non-destructive)
        let plannedMove = null;
        if (window.pathPlanner) {
            plannedMove = window.pathPlanner.peekPlannedMove(gameState);
        }

        try {
            // Check cache first
            const stateKey = this.getStateKey(gameState);
            if (this.predictionCache.has(stateKey)) {
                const cached = this.predictionCache.get(stateKey);
                cached.cached = true;
                return cached;
            }

            // Extract features using feature engineering
            if (!window.featureEngineer) {
                console.error('Feature engineer not initialized');
                return this.fallbackPrediction(gameState);
            }

            const vector = window.featureEngineer.extractFeatures(gameState);

            // Calculate current strategy for filtering
            let currentStrategy = null;
            if (window.vectorization) {
                // Use vectorization system to detect current strategy
                const enrichedMeta = window.vectorization.calculateEnrichedMetadata(gameState, 0);
                currentStrategy = enrichedMeta.detectedStrategy;
            }

            // Call Cloudflare Worker for prediction
            const response = await fetch(`${this.workerURL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gameState,
                    vector,
                    previousMoves: this.previousMoves,
                    currentStrategy: currentStrategy
                })
            });

            if (!response.ok) {
                console.error('Prediction API error:', response.statusText);
                return this.fallbackPrediction(gameState);
            }

            const data = await response.json();

            if (data.success && data.prediction) {
                // Debug: Log similar states found
                if (data.similarStatesFound !== undefined) {
                    console.log(`[AI Learning] Found ${data.similarStatesFound} similar past states`);
                    if (data.prediction.learnedFrom) {
                        console.log(`[AI Learning] Learned from ${data.prediction.learnedFrom} states`);
                    }
                }

                // === HYBRID DECISION: Path Planning + Vectorization ===
                let finalAction = data.prediction.action;
                let decisionSource = 'vectorization';

                // If we have a planned move, validate it with vectorization
                if (plannedMove && data.prediction.allScores) {
                    const plannedScore = data.prediction.allScores[plannedMove];
                    const bestScore = Math.max(...Object.values(data.prediction.allScores));

                    console.log(`[Hybrid AI] Planned: ${plannedMove} (score: ${plannedScore?.toFixed(1)}), Best: ${data.prediction.action} (score: ${bestScore.toFixed(1)})`);

                    // Use planned move if it's reasonably safe (score > -5)
                    if (plannedScore !== undefined && plannedScore > -5) {
                        finalAction = plannedMove;
                        decisionSource = 'path_planning';
                        console.log(`[Hybrid AI] Following plan: ${plannedMove}`);

                        // Consume the planned move (advance plan)
                        if (window.pathPlanner) {
                            window.pathPlanner.consumeMove();
                        }
                    } else {
                        console.log(`[Hybrid AI] Plan too dangerous, using vectorization: ${data.prediction.action}`);
                        // Path is dangerous, invalidate plan
                        if (window.pathPlanner) {
                            window.pathPlanner.currentPlan = null;
                        }
                    }
                }

                // Create final prediction
                const finalPrediction = {
                    ...data.prediction,
                    action: finalAction,
                    decisionSource: decisionSource,
                    plannedMove: plannedMove,
                    overridden: plannedMove && finalAction !== plannedMove,
                    queryStrategy: data.queryStrategy // Show which filter was used
                };

                // Cache the prediction
                this.predictionCache.set(stateKey, finalPrediction);

                // Limit cache size
                if (this.predictionCache.size > 100) {
                    const firstKey = this.predictionCache.keys().next().value;
                    this.predictionCache.delete(firstKey);
                }

                // Update move history
                this.previousMoves.push(finalPrediction.action);
                if (this.previousMoves.length > this.maxHistoryLength) {
                    this.previousMoves.shift();
                }

                // Track prediction
                if (window.gameStats) {
                    window.gameStats.learningMetrics.totalPredictions++;
                    window.gameStats.learningMetrics.vectorizedStates =
                        this.predictionCache.size;
                    // Track unique states from training data (vectorized human moves)
                    if (window.vectorization) {
                        window.gameStats.learningMetrics.uniqueStatesLearned =
                            window.vectorization.countUniqueStates();
                    }
                }

                return finalPrediction;
            } else {
                return this.fallbackPrediction(gameState);
            }
        } catch (error) {
            console.error('DQN prediction error:', error);
            return this.fallbackPrediction(gameState);
        }
    }

    /**
     * Fallback prediction using improved heuristics
     * Used when API is unavailable or for exploration
     */
    fallbackPrediction(gameState) {
        const actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
        const scores = {};

        for (const action of actions) {
            let score = 0;
            const nextPos = this.getNextPosition(
                gameState.playerX,
                gameState.playerY,
                action
            );

            // 1. Ghost avoidance (highest priority)
            let minGhostDistance = Infinity;
            for (const ghost of gameState.ghosts) {
                const distance = Math.abs(nextPos.x - ghost.x) +
                               Math.abs(nextPos.y - ghost.y);
                minGhostDistance = Math.min(minGhostDistance, distance);

                if (distance === 0) {
                    score -= 1000; // Death!
                } else if (distance === 1) {
                    score -= 100; // Very dangerous
                } else if (distance === 2) {
                    score -= 20; // Close call
                } else {
                    score += distance * 5; // Reward distance
                }
            }

            // 2. Avoid recent moves (exploration)
            if (this.previousMoves.slice(-3).includes(action)) {
                score -= 10;
            }

            // 3. Prefer center positions (more options)
            const distanceFromCenter = Math.abs(nextPos.x - 10) + Math.abs(nextPos.y - 10);
            score -= distanceFromCenter * 0.5;

            // 4. Add randomness for exploration
            score += Math.random() * 5;

            scores[action] = score;
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

        // Normalize confidence (0-1)
        const totalScore = Object.values(scores).reduce((a, b) => Math.abs(a) + Math.abs(b), 0);
        const confidence = totalScore > 0 ? Math.abs(maxScore) / totalScore : 0.5;

        return {
            action: bestAction,
            confidence: Math.min(confidence, 0.9),
            allScores: scores,
            fallback: true
        };
    }

    getNextPosition(x, y, action) {
        switch (action) {
            case 'UP': return { x, y: y - 1 };
            case 'DOWN': return { x, y: y + 1 };
            case 'LEFT': return { x: x - 1, y };
            case 'RIGHT': return { x: x + 1, y };
            default: return { x, y };
        }
    }

    getStateKey(gameState) {
        return `${gameState.playerX},${gameState.playerY},${gameState.direction}`;
    }

    getLastPrediction() {
        if (this.previousMoves.length > 0) {
            return {
                action: this.previousMoves[this.previousMoves.length - 1],
                confidence: 0.5,
                cached: true
            };
        }
        return { action: 'RIGHT', confidence: 0.5 };
    }

    /**
     * Learn from human moves (for future training)
     */
    async learnFromMove(gameState, humanAction, reward, vector, enrichedMetadata) {
        try {
            await fetch(`${this.workerURL}/api/store-move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gameState,
                    action: humanAction,
                    reward,
                    vector,  // Pre-computed feature vector
                    enrichedMetadata  // Additional context for filtering
                })
            });
        } catch (error) {
            console.error('Error storing move:', error);
        }
    }

    /**
     * Get similar past moves for analysis
     */
    async getSimilarMoves(gameState) {
        try {
            const response = await fetch(`${this.workerURL}/api/get-similar-moves`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ gameState })
            });

            if (response.ok) {
                const data = await response.json();
                return data.similarMoves || [];
            }
        } catch (error) {
            console.error('Error fetching similar moves:', error);
        }
        return [];
    }
}

// Initialize DQN agent globally
window.dqnAgent = new DQNAgent();
