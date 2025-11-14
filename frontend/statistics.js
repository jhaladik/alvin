/**
 * Statistics and Analytics System
 * Tracks AI learning progress and performance metrics
 */

class GameStatistics {
    constructor() {
        this.sessionStartTime = Date.now();
        this.resetStatistics();
        this.loadFromStorage();
    }

    resetStatistics() {
        // Human player stats
        this.humanStats = {
            gamesPlayed: 0,
            gamesWon: 0,
            totalScore: 0,
            highScore: 0,
            totalDeaths: 0,
            ghostsEaten: 0,
            pelletsCollected: 0,
            powerPelletsUsed: 0,
            averageSurvivalTime: 0,
            movesPerGame: []
        };

        // AI player stats
        this.aiStats = {
            gamesPlayed: 0,
            gamesWon: 0,
            totalScore: 0,
            highScore: 0,
            totalDeaths: 0,
            ghostsEaten: 0,
            pelletsCollected: 0,
            powerPelletsUsed: 0,
            averageSurvivalTime: 0,
            movesPerGame: []
        };

        // AI Learning metrics
        this.learningMetrics = {
            totalPredictions: 0,
            successfulPredictions: 0,
            failedPredictions: 0,
            averageConfidence: 0,
            predictionAccuracy: 0,
            explorationRate: 0.2, // 20% random exploration
            learningRate: 0.001,
            vectorizedStates: 0,
            uniqueStatesLearned: 0,
            rewardHistory: [],
            accuracyHistory: []
        };

        // Comparison metrics
        this.comparisonMetrics = {
            humanWinRate: 0,
            aiWinRate: 0,
            averageScoreDifference: 0,
            aiImprovement: 0, // % improvement over time
            gamesWhereAIWon: 0,
            gamesWhereHumanWon: 0
        };

        // Real-time game tracking
        this.currentGame = {
            humanStartTime: Date.now(),
            aiStartTime: Date.now(),
            humanMoves: 0,
            aiMoves: 0,
            humanPredictions: [],
            aiPredictions: []
        };
    }

    // Track move outcome
    trackMove(player, action, outcome) {
        const prediction = {
            timestamp: Date.now(),
            action: action,
            outcome: outcome, // 'death', 'pellet', 'power', 'ghost_eaten', 'neutral'
            reward: this.calculateOutcomeReward(outcome)
        };

        if (player === 'human') {
            this.currentGame.humanMoves++;
            this.currentGame.humanPredictions.push(prediction);
        } else {
            this.currentGame.aiMoves++;
            this.currentGame.aiPredictions.push(prediction);
            this.learningMetrics.totalPredictions++;
        }
    }

    calculateOutcomeReward(outcome) {
        const rewards = {
            'death': -100,
            'pellet': 10,
            'power': 50,
            'ghost_eaten': 200,
            'neutral': -0.1, // Small penalty for time
            'close_call': -20, // Almost hit by ghost
            'safe_move': 5 // Good positioning
        };
        return rewards[outcome] || 0;
    }

    // Track AI prediction accuracy
    trackPredictionAccuracy(wasSuccessful, confidence) {
        if (wasSuccessful) {
            this.learningMetrics.successfulPredictions++;
        } else {
            this.learningMetrics.failedPredictions++;
        }

        // Update accuracy - use actual evaluations count, not totalPredictions
        const totalEvaluations = this.learningMetrics.successfulPredictions +
                                this.learningMetrics.failedPredictions;
        const successful = this.learningMetrics.successfulPredictions;
        this.learningMetrics.predictionAccuracy = totalEvaluations > 0 ? (successful / totalEvaluations) * 100 : 0;

        // Update average confidence - use evaluations count
        const avgConf = this.learningMetrics.averageConfidence;
        this.learningMetrics.averageConfidence = totalEvaluations > 0 ?
            (avgConf * (totalEvaluations - 1) + confidence) / totalEvaluations : confidence;

        // Track accuracy over time
        if (totalEvaluations % 10 === 0) {
            this.learningMetrics.accuracyHistory.push({
                timestamp: Date.now(),
                accuracy: this.learningMetrics.predictionAccuracy,
                predictions: totalEvaluations
            });
        }
    }

    // End game and record statistics
    endGame(player, gameState) {
        const stats = player === 'human' ? this.humanStats : this.aiStats;
        const gameTime = Date.now() - (player === 'human' ?
            this.currentGame.humanStartTime : this.currentGame.aiStartTime);

        stats.gamesPlayed++;
        stats.totalScore += gameState.score;

        if (gameState.score > stats.highScore) {
            stats.highScore = gameState.score;
        }

        if (gameState.gameState === 'won') {
            stats.gamesWon++;
            if (player === 'human') {
                this.comparisonMetrics.gamesWhereHumanWon++;
            } else {
                this.comparisonMetrics.gamesWhereAIWon++;
            }
        }

        stats.ghostsEaten += gameState.ghostsEaten || 0;
        stats.pelletsCollected += (gameState.totalPellets - gameState.pellets.length - gameState.powerPellets.length);

        const moves = player === 'human' ? this.currentGame.humanMoves : this.currentGame.aiMoves;
        stats.movesPerGame.push(moves);

        // Keep only last 100 games
        if (stats.movesPerGame.length > 100) {
            stats.movesPerGame.shift();
        }

        // Update survival time
        stats.averageSurvivalTime =
            (stats.averageSurvivalTime * (stats.gamesPlayed - 1) + gameTime) / stats.gamesPlayed;

        this.updateComparisons();
        this.saveToStorage();
    }

    updateComparisons() {
        const humanGames = this.humanStats.gamesPlayed;
        const aiGames = this.aiStats.gamesPlayed;

        if (humanGames > 0) {
            this.comparisonMetrics.humanWinRate =
                (this.humanStats.gamesWon / humanGames) * 100;
        }

        if (aiGames > 0) {
            this.comparisonMetrics.aiWinRate =
                (this.aiStats.gamesWon / aiGames) * 100;
        }

        // Calculate average score difference
        if (humanGames > 0 && aiGames > 0) {
            const humanAvg = this.humanStats.totalScore / humanGames;
            const aiAvg = this.aiStats.totalScore / aiGames;
            this.comparisonMetrics.averageScoreDifference = aiAvg - humanAvg;
        }

        // Calculate AI improvement (comparing recent 10 games to all games)
        if (this.aiStats.movesPerGame.length >= 10) {
            const recentGames = this.aiStats.movesPerGame.slice(-10);
            const recentAvg = recentGames.reduce((a, b) => a + b, 0) / recentGames.length;
            const allAvg = this.aiStats.movesPerGame.reduce((a, b) => a + b, 0) / this.aiStats.movesPerGame.length;

            if (allAvg > 0) {
                this.comparisonMetrics.aiImprovement = ((recentAvg - allAvg) / allAvg) * 100;
            }
        }
    }

    // Get summary statistics
    getSummary() {
        return {
            human: {
                ...this.humanStats,
                winRate: this.comparisonMetrics.humanWinRate,
                avgScore: this.humanStats.gamesPlayed > 0 ?
                    this.humanStats.totalScore / this.humanStats.gamesPlayed : 0
            },
            ai: {
                ...this.aiStats,
                winRate: this.comparisonMetrics.aiWinRate,
                avgScore: this.aiStats.gamesPlayed > 0 ?
                    this.aiStats.totalScore / this.aiStats.gamesPlayed : 0
            },
            learning: this.learningMetrics,
            comparison: this.comparisonMetrics,
            sessionTime: Date.now() - this.sessionStartTime
        };
    }

    // Save to localStorage
    saveToStorage() {
        try {
            const data = {
                humanStats: this.humanStats,
                aiStats: this.aiStats,
                learningMetrics: this.learningMetrics,
                comparisonMetrics: this.comparisonMetrics,
                lastSaved: Date.now()
            };
            localStorage.setItem('alvin_statistics', JSON.stringify(data));
        } catch (e) {
            console.error('Failed to save statistics:', e);
        }
    }

    // Load from localStorage
    loadFromStorage() {
        try {
            const data = localStorage.getItem('alvin_statistics');
            if (data) {
                const parsed = JSON.parse(data);
                this.humanStats = parsed.humanStats || this.humanStats;
                this.aiStats = parsed.aiStats || this.aiStats;
                this.learningMetrics = parsed.learningMetrics || this.learningMetrics;
                this.comparisonMetrics = parsed.comparisonMetrics || this.comparisonMetrics;
                console.log('Loaded statistics from storage');
            }
        } catch (e) {
            console.error('Failed to load statistics:', e);
        }
    }

    // Clear all statistics
    clearAll() {
        this.resetStatistics();
        localStorage.removeItem('alvin_statistics');
        console.log('Statistics cleared');
    }

    // Export statistics as JSON
    exportData() {
        return {
            ...this.getSummary(),
            exportDate: new Date().toISOString(),
            version: '1.0'
        };
    }
}

// Initialize global statistics
window.gameStats = new GameStatistics();

// Console helpers
window.getStats = () => {
    const stats = window.gameStats.getSummary();
    console.log('=== Game Statistics ===');
    console.log('Human:', stats.human);
    console.log('AI:', stats.ai);
    console.log('Learning:', stats.learning);
    console.log('Comparison:', stats.comparison);
    return stats;
};

window.clearStats = () => {
    if (confirm('Clear all statistics?')) {
        window.gameStats.clearAll();
        console.log('Statistics cleared');
    }
};

window.exportStats = () => {
    const data = window.gameStats.exportData();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alvin-stats-${Date.now()}.json`;
    a.click();
    console.log('Statistics exported');
};

// Training control functions
window.updateExplorationRate = (value) => {
    const rate = parseInt(value) / 100;
    if (window.gameStats) {
        window.gameStats.learningMetrics.explorationRate = rate;
    }
    const display = document.getElementById('explorationValue');
    if (display) {
        display.textContent = value + '%';
    }
    console.log(`Exploration rate set to ${value}%`);
};

window.updatePredictionSpeed = (value) => {
    if (window.dqnAgent) {
        window.dqnAgent.predictionInterval = parseInt(value);
    }
    const display = document.getElementById('speedValue');
    if (display) {
        display.textContent = value + 'ms';
    }
    console.log(`Prediction speed set to ${value}ms`);
};
