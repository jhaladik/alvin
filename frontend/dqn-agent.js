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

        try {
            // Check cache first
            const stateKey = this.getStateKey(gameState);
            if (this.predictionCache.has(stateKey)) {
                const cached = this.predictionCache.get(stateKey);
                cached.cached = true;
                return cached;
            }

            // Call Cloudflare Worker for prediction
            const response = await fetch(`${this.workerURL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gameState,
                    previousMoves: this.previousMoves
                })
            });

            if (!response.ok) {
                console.error('Prediction API error:', response.statusText);
                return this.fallbackPrediction(gameState);
            }

            const data = await response.json();

            if (data.success && data.prediction) {
                // Cache the prediction
                this.predictionCache.set(stateKey, data.prediction);

                // Limit cache size
                if (this.predictionCache.size > 100) {
                    const firstKey = this.predictionCache.keys().next().value;
                    this.predictionCache.delete(firstKey);
                }

                // Update move history
                this.previousMoves.push(data.prediction.action);
                if (this.previousMoves.length > this.maxHistoryLength) {
                    this.previousMoves.shift();
                }

                // Track prediction
                if (window.gameStats) {
                    window.gameStats.learningMetrics.totalPredictions++;
                    window.gameStats.learningMetrics.vectorizedStates =
                        this.predictionCache.size;
                }

                return data.prediction;
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
    async learnFromMove(gameState, humanAction, reward) {
        try {
            await fetch(`${this.workerURL}/api/store-move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gameState,
                    action: humanAction,
                    reward
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
