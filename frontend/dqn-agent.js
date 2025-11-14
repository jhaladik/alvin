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

        try {
            // Check cache first
            const stateKey = this.getStateKey(gameState);
            if (this.predictionCache.has(stateKey)) {
                return this.predictionCache.get(stateKey);
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
     * Fallback prediction using simple heuristics
     * Used when API is unavailable or returns errors
     */
    fallbackPrediction(gameState) {
        const actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

        // Simple heuristic: avoid closest ghost
        let bestAction = actions[Math.floor(Math.random() * actions.length)];
        let maxDistance = -1;

        for (const action of actions) {
            const nextPos = this.getNextPosition(
                gameState.playerX,
                gameState.playerY,
                action
            );

            // Calculate minimum distance to any ghost
            let minGhostDistance = Infinity;
            for (const ghost of gameState.ghosts) {
                const distance = Math.abs(nextPos.x - ghost.x) + Math.abs(nextPos.y - ghost.y);
                minGhostDistance = Math.min(minGhostDistance, distance);
            }

            // Choose action that maximizes distance from nearest ghost
            if (minGhostDistance > maxDistance) {
                maxDistance = minGhostDistance;
                bestAction = action;
            }
        }

        return {
            action: bestAction,
            confidence: 0.5,
            allScores: {},
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
