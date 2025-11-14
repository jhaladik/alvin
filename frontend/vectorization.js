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
            const response = await fetch(`${this.workerURL}/api/vectorize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ gameState })
            });

            if (!response.ok) {
                console.error('Vectorization API error:', response.statusText);
                return null;
            }

            const data = await response.json();

            if (data.success && data.vector) {
                // Store vector with metadata
                const vectorEntry = {
                    vector: data.vector,
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

                // Send to DQN agent for learning
                if (window.dqnAgent) {
                    await window.dqnAgent.learnFromMove(gameState, action, reward);
                }

                return vectorEntry;
            }
        } catch (error) {
            console.error('Vectorization error:', error);
            return null;
        }
    }

    /**
     * Calculate reward for reinforcement learning
     */
    calculateReward(gameState) {
        let reward = 0;

        // Positive reward for score increase
        reward += gameState.score * 0.1;

        // Negative reward for being close to ghosts
        for (const ghost of gameState.ghosts) {
            const distance = Math.abs(gameState.playerX - ghost.x) +
                             Math.abs(gameState.playerY - ghost.y);
            if (distance < 3) {
                reward -= (3 - distance) * 10;
            }
        }

        // Positive reward for collecting pellets
        reward += (250 - gameState.pelletsLeft) * 0.5;

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
