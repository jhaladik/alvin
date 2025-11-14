/**
 * Feature Engineering System
 * Converts game states into meaningful numerical vectors for AI learning
 * This replaces text-based embeddings with proper spatial/geometric features
 */

class FeatureEngineer {
    constructor() {
        this.gridSize = 20;
        this.maxDistance = Math.sqrt(2) * this.gridSize; // Max diagonal distance
    }

    /**
     * Extract features from game state and create a numerical vector
     * Returns a vector that preserves spatial relationships and game geometry
     */
    extractFeatures(gameState) {
        const features = [];

        // ===== BASIC PLAYER INFO (6 features) =====
        // Player position (normalized to 0-1)
        features.push(gameState.playerX / this.gridSize);  // 0
        features.push(gameState.playerY / this.gridSize);  // 1

        // Player direction (one-hot encoded, 4 features)
        features.push(gameState.direction === 'UP' ? 1 : 0);     // 2
        features.push(gameState.direction === 'DOWN' ? 1 : 0);   // 3
        features.push(gameState.direction === 'LEFT' ? 1 : 0);   // 4
        features.push(gameState.direction === 'RIGHT' ? 1 : 0);  // 5

        // ===== GHOST INFORMATION (24 features) =====
        const ghostFeatures = this.extractGhostFeatures(gameState);
        features.push(...ghostFeatures);  // 6-29

        // ===== PELLET INFORMATION (8 features) =====
        const pelletFeatures = this.extractPelletFeatures(gameState);
        features.push(...pelletFeatures);  // 30-37

        // ===== POWER PELLET INFORMATION (6 features) =====
        const powerPelletFeatures = this.extractPowerPelletFeatures(gameState);
        features.push(...powerPelletFeatures);  // 38-43

        // ===== WALL/NAVIGATION INFORMATION (8 features) =====
        const wallFeatures = this.extractWallFeatures(gameState);
        features.push(...wallFeatures);  // 44-51

        // ===== TACTICAL SITUATION (12 features) =====
        const tacticalFeatures = this.extractTacticalFeatures(gameState);
        features.push(...tacticalFeatures);  // 52-63

        // ===== GAME STATE (5 features) =====
        features.push(gameState.powerMode ? 1 : 0);           // 64
        features.push(gameState.powerModeTimer / 50);         // 65 (normalized)
        features.push(gameState.lives / 3);                   // 66
        features.push(Math.min(gameState.score / 10000, 1));  // 67 (capped)
        features.push(gameState.pelletsLeft / gameState.totalPellets);  // 68

        // Total: 69 features
        // Pad to 128 dimensions for Vectorize DB compatibility (requires power-of-2)
        while (features.length < 128) {
            features.push(0);
        }

        return features;
    }

    /**
     * Extract features related to ghosts (4 ghosts × 6 features = 24)
     */
    extractGhostFeatures(gameState) {
        const features = [];
        const maxGhosts = 4;

        // Calculate features for each ghost
        const ghostData = [];
        for (const ghost of gameState.ghosts) {
            const dx = ghost.x - gameState.playerX;
            const dy = ghost.y - gameState.playerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const manhattanDist = Math.abs(dx) + Math.abs(dy);

            ghostData.push({
                distance: distance,
                manhattanDist: manhattanDist,
                dx: dx,
                dy: dy,
                scared: ghost.scared || false
            });
        }

        // Sort by distance (closest first)
        ghostData.sort((a, b) => a.distance - b.distance);

        // Extract features for up to 4 ghosts
        for (let i = 0; i < maxGhosts; i++) {
            if (i < ghostData.length) {
                const g = ghostData[i];
                features.push(g.distance / this.maxDistance);      // Normalized distance
                features.push(g.manhattanDist / (this.gridSize * 2)); // Normalized Manhattan
                features.push(g.dx / this.gridSize);               // Normalized delta X
                features.push(g.dy / this.gridSize);               // Normalized delta Y
                features.push(g.scared ? 1 : 0);                   // Is scared?
                features.push(this.getDirectionToPoint(g.dx, g.dy)); // Direction encoding
            } else {
                // Padding for missing ghosts
                features.push(1, 1, 0, 0, 0, 0);  // Max distance = no threat
            }
        }

        return features;
    }

    /**
     * Extract features related to regular pellets (8 features)
     */
    extractPelletFeatures(gameState) {
        if (!gameState.pellets || gameState.pellets.length === 0) {
            return [1, 1, 0, 0, 0, 0, 0, 0]; // No pellets
        }

        // Find nearest pellet
        let nearest = null;
        let minDist = Infinity;
        for (const pellet of gameState.pellets) {
            const dx = pellet.x - gameState.playerX;
            const dy = pellet.y - gameState.playerY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                nearest = { x: pellet.x, y: pellet.y, dx, dy, dist };
            }
        }

        // Count pellets in each direction (within 5 tiles)
        const pelletDensity = this.countPelletsInDirections(gameState, 5);

        return [
            nearest.dist / this.maxDistance,     // Distance to nearest
            nearest.dx / this.gridSize,          // Delta X
            nearest.dy / this.gridSize,          // Delta Y
            this.getDirectionToPoint(nearest.dx, nearest.dy), // Direction
            pelletDensity.up,                    // Pellets above
            pelletDensity.down,                  // Pellets below
            pelletDensity.left,                  // Pellets left
            pelletDensity.right                  // Pellets right
        ];
    }

    /**
     * Extract features related to power pellets (6 features)
     */
    extractPowerPelletFeatures(gameState) {
        if (!gameState.powerPellets || gameState.powerPellets.length === 0) {
            return [1, 1, 0, 0, 0, 0]; // No power pellets
        }

        // Find nearest power pellet
        let nearest = null;
        let minDist = Infinity;
        for (const pellet of gameState.powerPellets) {
            const dx = pellet.x - gameState.playerX;
            const dy = pellet.y - gameState.playerY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                nearest = { dx, dy, dist };
            }
        }

        return [
            nearest.dist / this.maxDistance,     // Distance to nearest
            nearest.dx / this.gridSize,          // Delta X
            nearest.dy / this.gridSize,          // Delta Y
            this.getDirectionToPoint(nearest.dx, nearest.dy), // Direction
            gameState.powerPellets.length / 4,   // How many left (0-1)
            gameState.powerMode ? 0 : 1          // Need power? (1=yes, 0=already powered)
        ];
    }

    /**
     * Extract features related to walls and navigation (8 features)
     */
    extractWallFeatures(gameState) {
        const features = [];
        const directions = [
            { dx: 0, dy: -1 },  // UP
            { dx: 0, dy: 1 },   // DOWN
            { dx: -1, dy: 0 },  // LEFT
            { dx: 1, dy: 0 }    // RIGHT
        ];

        // For each direction, check if there's a wall and how far
        for (const dir of directions) {
            const wallDist = this.getWallDistance(gameState, dir.dx, dir.dy);
            features.push(wallDist / 5);  // Normalized (check up to 5 tiles)
        }

        // Check diagonals for escape route planning
        const diagonals = [
            { dx: -1, dy: -1 },  // UP-LEFT
            { dx: 1, dy: -1 },   // UP-RIGHT
            { dx: -1, dy: 1 },   // DOWN-LEFT
            { dx: 1, dy: 1 }     // DOWN-RIGHT
        ];

        for (const dir of diagonals) {
            const wallDist = this.getWallDistance(gameState, dir.dx, dir.dy);
            features.push(wallDist / 5);
        }

        return features;
    }

    /**
     * Extract tactical situation features (12 features)
     */
    extractTacticalFeatures(gameState) {
        const features = [];

        // Danger level: how close is nearest ghost?
        const nearestGhostDist = this.getNearestGhostDistance(gameState);
        features.push(1 - Math.min(nearestGhostDist / 10, 1));  // 0=safe, 1=danger

        // Escape routes: how many safe directions?
        const safeDirections = this.countSafeDirections(gameState);
        features.push(safeDirections / 4);  // 0-4 safe directions

        // Trapped indicator: in a corner or dead end?
        features.push(safeDirections <= 1 ? 1 : 0);

        // Ghost convergence: are ghosts closing in from multiple sides?
        const convergence = this.calculateGhostConvergence(gameState);
        features.push(convergence);

        // Center vs edge position (center = more options)
        const centerX = this.gridSize / 2;
        const centerY = this.gridSize / 2;
        const distFromCenter = Math.sqrt(
            Math.pow(gameState.playerX - centerX, 2) +
            Math.pow(gameState.playerY - centerY, 2)
        );
        features.push(1 - (distFromCenter / (this.maxDistance / 2)));  // 1=center, 0=edge

        // Pellet collection progress rate (how efficiently collecting?)
        const progressRatio = 1 - (gameState.pelletsLeft / gameState.totalPellets);
        features.push(progressRatio);

        // Risk/reward: is there a pellet near a ghost?
        features.push(this.calculateRiskReward(gameState));

        // Power mode opportunity: is there a scared ghost nearby?
        const scaredGhostNearby = this.hasScaredGhostNearby(gameState, 8);
        features.push(scaredGhostNearby ? 1 : 0);

        // Optimal direction indicator: best theoretical direction
        const optimalDir = this.calculateOptimalDirection(gameState);
        features.push(optimalDir.up);
        features.push(optimalDir.down);
        features.push(optimalDir.left);
        features.push(optimalDir.right);

        return features;
    }

    // ===== HELPER FUNCTIONS =====

    getDirectionToPoint(dx, dy) {
        // Encode as continuous value based on angle
        const angle = Math.atan2(dy, dx);
        return (angle + Math.PI) / (2 * Math.PI);  // Normalized 0-1
    }

    countPelletsInDirections(gameState, range) {
        const counts = { up: 0, down: 0, left: 0, right: 0 };

        if (!gameState.pellets) return counts;

        for (const pellet of gameState.pellets) {
            const dx = pellet.x - gameState.playerX;
            const dy = pellet.y - gameState.playerY;
            const dist = Math.abs(dx) + Math.abs(dy);

            if (dist > range) continue;

            if (Math.abs(dy) > Math.abs(dx)) {
                // Primarily vertical
                if (dy < 0) counts.up++;
                else counts.down++;
            } else {
                // Primarily horizontal
                if (dx < 0) counts.left++;
                else counts.right++;
            }
        }

        // Normalize
        const total = counts.up + counts.down + counts.left + counts.right;
        if (total > 0) {
            counts.up /= total;
            counts.down /= total;
            counts.left /= total;
            counts.right /= total;
        }

        return counts;
    }

    getWallDistance(gameState, dx, dy) {
        let dist = 0;
        let x = gameState.playerX;
        let y = gameState.playerY;

        for (let i = 1; i <= 5; i++) {
            x += dx;
            y += dy;

            // Out of bounds = wall
            if (x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) {
                return i;
            }

            // Check if wall
            if (gameState.walls && gameState.walls.some(w => w.x === x && w.y === y)) {
                return i;
            }

            dist = i;
        }

        return 5; // No wall within 5 tiles
    }

    getNearestGhostDistance(gameState) {
        let minDist = Infinity;
        for (const ghost of gameState.ghosts) {
            const dx = ghost.x - gameState.playerX;
            const dy = ghost.y - gameState.playerY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            minDist = Math.min(minDist, dist);
        }
        return minDist;
    }

    countSafeDirections(gameState) {
        const directions = [
            { dx: 0, dy: -1 },  // UP
            { dx: 0, dy: 1 },   // DOWN
            { dx: -1, dy: 0 },  // LEFT
            { dx: 1, dy: 0 }    // RIGHT
        ];

        let safeCount = 0;
        for (const dir of directions) {
            if (this.isDirectionSafe(gameState, dir.dx, dir.dy)) {
                safeCount++;
            }
        }
        return safeCount;
    }

    isDirectionSafe(gameState, dx, dy) {
        const newX = gameState.playerX + dx;
        const newY = gameState.playerY + dy;

        // Check bounds
        if (newX < 0 || newX >= this.gridSize || newY < 0 || newY >= this.gridSize) {
            return false;
        }

        // Check walls
        if (gameState.walls && gameState.walls.some(w => w.x === newX && w.y === newY)) {
            return false;
        }

        // Check ghosts (not safe if ghost within 2 tiles in that direction)
        for (const ghost of gameState.ghosts) {
            if (ghost.scared) continue; // Scared ghosts are safe

            const distToGhost = Math.abs(ghost.x - newX) + Math.abs(ghost.y - newY);
            if (distToGhost <= 2) {
                return false;
            }
        }

        return true;
    }

    calculateGhostConvergence(gameState) {
        // Check if ghosts are approaching from multiple quadrants
        const quadrants = { nw: 0, ne: 0, sw: 0, se: 0 };

        for (const ghost of gameState.ghosts) {
            if (ghost.scared) continue;

            const dx = ghost.x - gameState.playerX;
            const dy = ghost.y - gameState.playerY;

            if (dx < 0 && dy < 0) quadrants.nw = 1;
            else if (dx >= 0 && dy < 0) quadrants.ne = 1;
            else if (dx < 0 && dy >= 0) quadrants.sw = 1;
            else quadrants.se = 1;
        }

        const activeQuadrants = quadrants.nw + quadrants.ne + quadrants.sw + quadrants.se;
        return activeQuadrants / 4;  // 0-1 (1 = surrounded)
    }

    calculateRiskReward(gameState) {
        // Are there pellets near ghosts? (risky but rewarding)
        if (!gameState.pellets) return 0;

        let riskyPellets = 0;
        for (const pellet of gameState.pellets) {
            const distToPlayer = Math.abs(pellet.x - gameState.playerX) +
                                Math.abs(pellet.y - gameState.playerY);

            if (distToPlayer > 5) continue; // Too far

            for (const ghost of gameState.ghosts) {
                if (ghost.scared) continue;

                const distToGhost = Math.abs(pellet.x - ghost.x) +
                                   Math.abs(pellet.y - ghost.y);

                if (distToGhost <= 3) {
                    riskyPellets++;
                    break;
                }
            }
        }

        return Math.min(riskyPellets / 5, 1);  // Normalized
    }

    hasScaredGhostNearby(gameState, range) {
        for (const ghost of gameState.ghosts) {
            if (!ghost.scared) continue;

            const dist = Math.abs(ghost.x - gameState.playerX) +
                        Math.abs(ghost.y - gameState.playerY);

            if (dist <= range) return true;
        }
        return false;
    }

    calculateOptimalDirection(gameState) {
        // Calculate scores for each direction (simplified heuristic)
        const directions = {
            up: 0,
            down: 0,
            left: 0,
            right: 0
        };

        // Avoid ghosts
        for (const ghost of gameState.ghosts) {
            if (ghost.scared) continue;

            const dx = ghost.x - gameState.playerX;
            const dy = ghost.y - gameState.playerY;

            if (Math.abs(dy) > Math.abs(dx)) {
                if (dy < 0) directions.down += 0.25;  // Ghost above, go down
                else directions.up += 0.25;
            } else {
                if (dx < 0) directions.right += 0.25;
                else directions.left += 0.25;
            }
        }

        // Seek nearest pellet
        if (gameState.pellets && gameState.pellets.length > 0) {
            const nearest = this.findNearestPellet(gameState);
            if (nearest) {
                if (Math.abs(nearest.dy) > Math.abs(nearest.dx)) {
                    if (nearest.dy < 0) directions.up += 0.5;
                    else directions.down += 0.5;
                } else {
                    if (nearest.dx < 0) directions.left += 0.5;
                    else directions.right += 0.5;
                }
            }
        }

        // Normalize
        const total = directions.up + directions.down + directions.left + directions.right;
        if (total > 0) {
            directions.up /= total;
            directions.down /= total;
            directions.left /= total;
            directions.right /= total;
        }

        return directions;
    }

    findNearestPellet(gameState) {
        let nearest = null;
        let minDist = Infinity;

        for (const pellet of gameState.pellets) {
            const dx = pellet.x - gameState.playerX;
            const dy = pellet.y - gameState.playerY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < minDist) {
                minDist = dist;
                nearest = { dx, dy, dist };
            }
        }

        return nearest;
    }

    /**
     * Get feature names for debugging/analysis
     */
    getFeatureNames() {
        const names = [
            // Player (6)
            'player_x', 'player_y', 'dir_up', 'dir_down', 'dir_left', 'dir_right',
            // Ghosts (24)
            ...this.generateGhostFeatureNames(),
            // Pellets (8)
            'nearest_pellet_dist', 'nearest_pellet_dx', 'nearest_pellet_dy', 'nearest_pellet_dir',
            'pellets_up', 'pellets_down', 'pellets_left', 'pellets_right',
            // Power Pellets (6)
            'nearest_power_dist', 'nearest_power_dx', 'nearest_power_dy', 'nearest_power_dir',
            'power_pellets_remaining', 'need_power',
            // Walls (8)
            'wall_up', 'wall_down', 'wall_left', 'wall_right',
            'wall_up_left', 'wall_up_right', 'wall_down_left', 'wall_down_right',
            // Tactical (12)
            'danger_level', 'safe_directions', 'trapped', 'ghost_convergence',
            'centrality', 'progress', 'risk_reward', 'scared_ghost_nearby',
            'optimal_up', 'optimal_down', 'optimal_left', 'optimal_right',
            // Game State (5)
            'power_mode', 'power_timer', 'lives', 'score', 'pellets_ratio'
        ];
        return names;
    }

    generateGhostFeatureNames() {
        const names = [];
        for (let i = 1; i <= 4; i++) {
            names.push(`ghost${i}_dist`, `ghost${i}_manhattan`, `ghost${i}_dx`,
                      `ghost${i}_dy`, `ghost${i}_scared`, `ghost${i}_dir`);
        }
        return names;
    }
}

// Initialize global feature engineer
window.featureEngineer = new FeatureEngineer();

// Debug helper
window.debugFeatures = (gameState) => {
    const features = window.featureEngineer.extractFeatures(gameState);
    const names = window.featureEngineer.getFeatureNames();

    console.log('=== Feature Vector (69 dimensions) ===');
    names.forEach((name, i) => {
        console.log(`${i.toString().padStart(2)}: ${name.padEnd(25)} = ${features[i].toFixed(4)}`);
    });

    return { features, names };
};
