/**
 * Vectorization System
 * Handles vectorization of human gameplay using Cloudflare AI
 */

class VectorizationSystem {
    constructor() {
        // Change this to your worker URL when deployed
        this.workerURL = window.location.origin;

        this.vectorHistory = [];
        this.maxVectorHistory = 50;
        this.pendingVectorizations = [];
        this.batchSize = 5;
        this.batchTimeout = null;
    }

    /**
     * Vectorize current game state
     */
    async vectorizeGameState(gameState, action, recentMoves = []) {
        try {
            // Add to pending batch with recent moves
            this.pendingVectorizations.push({ gameState, action, recentMoves });

            // Process batch if it's full
            if (this.pendingVectorizations.length >= this.batchSize) {
                await this.processBatch();
            } else {
                // Schedule batch processing
                this.scheduleBatchProcessing();
            }
        } catch (error) {
            console.error('Vectorization error:', error);
        }
    }

    scheduleBatchProcessing() {
        if (this.batchTimeout) {
            clearTimeout(this.batchTimeout);
        }

        this.batchTimeout = setTimeout(() => {
            this.processBatch();
        }, 1000); // Process batch after 1 second of inactivity
    }

    async processBatch() {
        if (this.pendingVectorizations.length === 0) return;

        const batch = [...this.pendingVectorizations];
        this.pendingVectorizations = [];

        if (this.batchTimeout) {
            clearTimeout(this.batchTimeout);
            this.batchTimeout = null;
        }

        // Process each item in the batch
        for (const item of batch) {
            await this.vectorizeSingle(item.gameState, item.action, item.recentMoves || []);
        }
    }

    async vectorizeSingle(gameState, action, recentMoves = []) {
        try {
            // Use feature engineering instead of text embeddings
            if (!window.featureEngineer) {
                console.error('Feature engineer not initialized');
                return null;
            }

            // Extract numerical features from game state
            const vector = window.featureEngineer.extractFeatures(gameState);

            // Debug: Verify feature extraction
            console.log(`[Vectorization] Extracted ${vector.length}D vector for action ${action}`);
            console.log(`[Vectorization] Sample features: player=(${gameState.playerX},${gameState.playerY}), ghosts=${gameState.ghosts.length}, pellets=${gameState.pelletsLeft}`);

            // Store vector with metadata
            const vectorEntry = {
                vector: vector,
                gameState,
                action,
                timestamp: Date.now()
            };

            this.vectorHistory.push(vectorEntry);

            // Limit history size
            if (this.vectorHistory.length > this.maxVectorHistory) {
                this.vectorHistory.shift();
            }

            // Calculate reward based on game state
            const reward = this.calculateReward(gameState);

            // Calculate enriched metadata for better learning
            const enrichedMetadata = this.calculateEnrichedMetadata(gameState, reward);

            // Add recent moves to metadata for sequence learning
            enrichedMetadata.recentMoves = recentMoves;

            // Send to DQN agent for learning (store in backend)
            if (window.dqnAgent) {
                console.log(`[Vectorization] Storing move with reward ${reward.toFixed(2)}, strategy: ${enrichedMetadata.detectedStrategy}, outcome: ${enrichedMetadata.outcomeType}`);
                await window.dqnAgent.learnFromMove(gameState, action, reward, vector, enrichedMetadata);
            }

            return vectorEntry;
        } catch (error) {
            console.error('Vectorization error:', error);
            return null;
        }
    }

    /**
     * Calculate reward for reinforcement learning
     * Enhanced with better signal quality for AI learning
     */
    calculateReward(gameState, previousState) {
        let reward = 0;
        let rewardDetails = {
            score: 0,
            ghostAvoidance: 0,
            pelletProgress: 0,
            positioning: 0,
            survival: 0
        };

        // 1. Score-based reward (immediate feedback)
        if (previousState) {
            const scoreDelta = gameState.score - previousState.score;
            rewardDetails.score = scoreDelta;
            reward += scoreDelta;
        } else {
            rewardDetails.score = gameState.score * 0.01;
            reward += gameState.score * 0.01;
        }

        // 2. Ghost proximity penalty (danger awareness)
        let minGhostDistance = Infinity;
        for (const ghost of gameState.ghosts) {
            const distance = Math.abs(gameState.playerX - ghost.x) +
                             Math.abs(gameState.playerY - ghost.y);
            minGhostDistance = Math.min(minGhostDistance, distance);

            if (distance === 0) {
                // Death or ghost eaten
                rewardDetails.ghostAvoidance -= 100;
                reward -= 100;
            } else if (distance === 1) {
                // Very close call!
                rewardDetails.ghostAvoidance -= 30;
                reward -= 30;
            } else if (distance === 2) {
                // Close call
                rewardDetails.ghostAvoidance -= 10;
                reward -= 10;
            } else if (distance <= 4) {
                // Nearby threat
                rewardDetails.ghostAvoidance -= (5 - distance) * 2;
                reward -= (5 - distance) * 2;
            }
        }

        // 3. Pellet collection progress (completion incentive)
        const totalPellets = gameState.totalPellets || 250;
        const pelletsCollected = totalPellets - gameState.pelletsLeft;
        const progressRatio = pelletsCollected / totalPellets;
        rewardDetails.pelletProgress = progressRatio * 10;
        reward += progressRatio * 10;

        // 4. Positioning reward (strategic placement)
        // Reward being in center of board (more options)
        const centerX = 10, centerY = 10;
        const distanceFromCenter = Math.abs(gameState.playerX - centerX) +
                                   Math.abs(gameState.playerY - centerY);
        if (distanceFromCenter < 5 && minGhostDistance > 3) {
            rewardDetails.positioning = 2;
            reward += 2;
        }

        // 5. Survival reward (staying alive is good)
        if (gameState.lives > 0) {
            rewardDetails.survival = 1;
            reward += 1;
        }

        // 6. Exploration bonus (encourage new areas)
        // Small random reward to encourage exploration
        if (Math.random() < 0.1) {
            rewardDetails.positioning += 0.5;
            reward += 0.5;
        }

        // Store reward components for analysis
        if (window.gameStats) {
            window.gameStats.learningMetrics.rewardHistory.push({
                timestamp: Date.now(),
                total: reward,
                details: rewardDetails
            });

            // Keep only last 1000 rewards
            if (window.gameStats.learningMetrics.rewardHistory.length > 1000) {
                window.gameStats.learningMetrics.rewardHistory.shift();
            }
        }

        return reward;
    }

    /**
     * Calculate enriched metadata for improved AI learning
     * Provides context beyond just the reward value
     */
    calculateEnrichedMetadata(gameState, reward) {
        // Calculate ghost proximity metrics
        let minGhostDistance = Infinity;
        let ghostsNearby = 0; // Within 3 tiles
        let avgGhostDistance = 0;

        for (const ghost of gameState.ghosts) {
            const distance = Math.abs(gameState.playerX - ghost.x) +
                           Math.abs(gameState.playerY - ghost.y);
            minGhostDistance = Math.min(minGhostDistance, distance);
            avgGhostDistance += distance;

            if (distance <= 3) {
                ghostsNearby++;
            }
        }
        avgGhostDistance = gameState.ghosts.length > 0 ? avgGhostDistance / gameState.ghosts.length : 10;

        // Classify outcome type
        let outcomeType = 'safe';
        if (reward <= -100) {
            outcomeType = 'death';
        } else if (reward >= 200) {
            outcomeType = 'ghost_eaten';
        } else if (reward >= 10) {
            outcomeType = 'pellet';
        } else if (reward < -10) {
            outcomeType = 'near_miss';
        } else if (minGhostDistance <= 2) {
            outcomeType = 'close_call';
        }

        // Determine success (learn from this move?)
        const success = reward > -10; // Exclude deaths and close calls

        // Game phase (early/mid/late)
        const totalPellets = gameState.totalPellets || 250;
        const pelletsRemaining = gameState.pelletsLeft || 0;
        const progressRatio = (totalPellets - pelletsRemaining) / totalPellets;
        let gamePhase = 'early';
        if (progressRatio > 0.66) {
            gamePhase = 'late';
        } else if (progressRatio > 0.33) {
            gamePhase = 'mid';
        }

        // Calculate strategic metrics
        const strategicMetrics = this.calculateStrategicMetrics(
            gameState,
            minGhostDistance,
            avgGhostDistance,
            pelletsRemaining,
            totalPellets
        );

        // Detect current strategy
        const detectedStrategy = this.detectStrategy(
            gameState,
            minGhostDistance,
            avgGhostDistance,
            strategicMetrics
        );

        return {
            // Game context
            powerMode: gameState.powerMode || false,
            powerModeTimer: gameState.powerModeTimer || 0,
            lives: gameState.lives || 3,
            pelletsRemaining: pelletsRemaining,
            gamePhase: gamePhase,

            // Ghost context
            ghostsNearby: ghostsNearby,
            minGhostDistance: minGhostDistance,
            avgGhostDistance: avgGhostDistance,

            // Outcome classification
            outcomeType: outcomeType,
            scoreGain: reward,
            success: success,

            // Strategic context
            detectedStrategy: detectedStrategy,
            riskLevel: strategicMetrics.riskLevel,
            efficiency: strategicMetrics.efficiency,
            aggressionScore: strategicMetrics.aggressionScore,

            // Move sequence context (will be populated by game.js)
            recentMoves: [],

            // Strategy context (will be added by DQN agent)
            wasExploration: false,
            wasPathPlanned: false,
            confidence: 0
        };
    }

    /**
     * Calculate strategic metrics from current game state
     */
    calculateStrategicMetrics(gameState, minGhostDistance, avgGhostDistance, pelletsRemaining, totalPellets) {
        // 1. Risk Level (0 = safe, 1 = dangerous)
        let riskLevel = 0;

        if (gameState.powerMode) {
            // In power mode, risk is lower
            riskLevel = Math.max(0, 1 - (minGhostDistance / 10));
            riskLevel *= 0.3; // Scale down in power mode
        } else {
            // Normal mode - closer ghosts = higher risk
            if (minGhostDistance <= 2) {
                riskLevel = 1.0; // Maximum danger
            } else if (minGhostDistance <= 4) {
                riskLevel = 0.7;
            } else if (minGhostDistance <= 6) {
                riskLevel = 0.4;
            } else {
                riskLevel = 0.1;
            }
        }

        // 2. Efficiency (pellets collected / total pellets)
        const efficiency = totalPellets > 0 ?
            (totalPellets - pelletsRemaining) / totalPellets : 0;

        // 3. Aggression Score (moving toward danger?)
        let aggressionScore = 0;

        if (gameState.powerMode) {
            // In power mode, closer to ghosts = more aggressive (good!)
            aggressionScore = Math.max(0, 1 - (minGhostDistance / 8));
        } else {
            // In normal mode, closer to ghosts = reckless (bad)
            aggressionScore = minGhostDistance < 5 ? 0.8 : 0.2;
        }

        return {
            riskLevel: Math.min(1, Math.max(0, riskLevel)),
            efficiency: Math.min(1, Math.max(0, efficiency)),
            aggressionScore: Math.min(1, Math.max(0, aggressionScore))
        };
    }

    /**
     * Detect current strategy from behavior patterns
     */
    detectStrategy(gameState, minGhostDistance, avgGhostDistance, metrics) {
        const { riskLevel, aggressionScore } = metrics;

        // HUNTER: Power mode + moving toward ghosts
        if (gameState.powerMode && aggressionScore > 0.6) {
            return 'hunter';
        }

        // DEFENSIVE: Not in power mode + keeping distance + low risk
        if (!gameState.powerMode && avgGhostDistance > 5 && riskLevel < 0.3) {
            return 'defensive';
        }

        // AGGRESSIVE: Taking risks (close to ghosts) without power mode
        if (!gameState.powerMode && riskLevel > 0.6) {
            return 'aggressive';
        }

        // EFFICIENT: Good progress, moderate safety
        if (metrics.efficiency > 0.4 && riskLevel < 0.5) {
            return 'efficient';
        }

        // SURVIVOR: Low lives, playing very safe
        if (gameState.lives <= 1 && avgGhostDistance > 6) {
            return 'survivor';
        }

        // BALANCED: Default
        return 'balanced';
    }

    /**
     * Find similar game states using vector similarity
     */
    findSimilarStates(targetVector, topK = 5) {
        if (!targetVector || this.vectorHistory.length === 0) {
            return [];
        }

        // Calculate cosine similarity for each vector
        const similarities = this.vectorHistory.map(entry => ({
            ...entry,
            similarity: this.cosineSimilarity(targetVector, entry.vector)
        }));

        // Sort by similarity and return top K
        similarities.sort((a, b) => b.similarity - a.similarity);
        return similarities.slice(0, topK);
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) {
            return 0;
        }

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA === 0 || normB === 0) {
            return 0;
        }

        return dotProduct / (normA * normB);
    }

    /**
     * Get vectorization statistics
     */
    getStats() {
        return {
            totalVectorized: this.vectorHistory.length,
            pendingVectorizations: this.pendingVectorizations.length,
            avgVectorMagnitude: this.calculateAvgMagnitude(),
            uniqueStates: this.countUniqueStates()
        };
    }

    calculateAvgMagnitude() {
        if (this.vectorHistory.length === 0) return 0;

        const sum = this.vectorHistory.reduce((acc, entry) => {
            const magnitude = Math.sqrt(
                entry.vector.reduce((sum, val) => sum + val * val, 0)
            );
            return acc + magnitude;
        }, 0);

        return sum / this.vectorHistory.length;
    }

    countUniqueStates() {
        const states = new Set();
        this.vectorHistory.forEach(entry => {
            const key = `${entry.gameState.playerX},${entry.gameState.playerY}`;
            states.add(key);
        });
        return states.size;
    }

    /**
     * Export vectorization data for analysis
     */
    exportData() {
        return {
            vectors: this.vectorHistory,
            stats: this.getStats(),
            exportDate: new Date().toISOString()
        };
    }

    /**
     * Clear all vectorization data
     */
    clear() {
        this.vectorHistory = [];
        this.pendingVectorizations = [];
        if (this.batchTimeout) {
            clearTimeout(this.batchTimeout);
            this.batchTimeout = null;
        }
    }
}

// Initialize vectorization system globally
window.vectorization = new VectorizationSystem();

// Add console helper for debugging
window.getVectorizationStats = () => {
    const stats = window.vectorization.getStats();
    console.log('Vectorization Statistics:', stats);
    return stats;
};
