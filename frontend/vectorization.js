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
    async vectorizeGameState(gameState, action) {
        try {
            // Add to pending batch
            this.pendingVectorizations.push({ gameState, action });

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
            await this.vectorizeSingle(item.gameState, item.action);
        }
    }

    async vectorizeSingle(gameState, action) {
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

            // Send to DQN agent for learning (store in backend)
            if (window.dqnAgent) {
                console.log(`[Vectorization] Storing move with reward ${reward.toFixed(2)}`);
                await window.dqnAgent.learnFromMove(gameState, action, reward, vector);
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
