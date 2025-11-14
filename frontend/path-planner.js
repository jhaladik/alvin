/**
 * Path Planning System for AI
 * Provides strategic, multi-step planning on top of tactical vectorization
 */

class PathPlanner {
    constructor() {
        this.currentPlan = null;  // { goal, path, step }
        this.gridSize = 20;
        this.planningInterval = 5;  // Re-plan every N moves
        this.movesSincePlan = 0;
    }

    /**
     * Get next move from path planner
     * Returns null if no plan or plan complete
     */
    getPlannedMove(gameState) {
        // Check if we need a new plan
        if (!this.currentPlan || this.shouldReplan(gameState)) {
            this.currentPlan = this.createPlan(gameState);
            this.movesSincePlan = 0;
        }

        // No valid plan
        if (!this.currentPlan || !this.currentPlan.path) {
            return null;
        }

        // Follow current plan
        const nextMove = this.currentPlan.path[this.currentPlan.step];
        this.currentPlan.step++;
        this.movesSincePlan++;

        // Plan complete?
        if (this.currentPlan.step >= this.currentPlan.path.length) {
            this.currentPlan = null;
        }

        return nextMove;
    }

    /**
     * Decide if we should abandon current plan
     */
    shouldReplan(gameState) {
        if (!this.currentPlan) return true;

        // Reached goal
        if (this.currentPlan.step >= this.currentPlan.path.length) {
            return true;
        }

        // Ghost blocking path
        if (this.isPathBlocked(gameState)) {
            console.log('[Path Planning] Ghost blocking path, replanning...');
            return true;
        }

        // Goal no longer exists (pellet collected)
        if (!this.goalStillValid(gameState)) {
            console.log('[Path Planning] Goal collected, replanning...');
            return true;
        }

        // Been too long since last plan
        if (this.movesSincePlan > this.planningInterval) {
            return true;
        }

        return false;
    }

    /**
     * Create a new plan
     */
    createPlan(gameState) {
        // 1. Select goal
        const goal = this.selectGoal(gameState);
        if (!goal) {
            console.log('[Path Planning] No goal found');
            return null;
        }

        // 2. Plan path using A*
        const path = this.aStar(
            { x: gameState.playerX, y: gameState.playerY },
            goal,
            gameState
        );

        if (!path || path.length === 0) {
            console.log('[Path Planning] No path to goal');
            return null;
        }

        console.log(`[Path Planning] New plan: ${goal.type} at (${goal.x},${goal.y}), ${path.length} moves`);

        return {
            goal: goal,
            path: path,
            step: 0,
            createdAt: Date.now()
        };
    }

    /**
     * Select best goal based on priority and distance
     */
    selectGoal(gameState) {
        const playerPos = { x: gameState.playerX, y: gameState.playerY };

        // Priority 1: Power pellets (if ghosts nearby)
        if (!gameState.powerMode && gameState.powerPellets && gameState.powerPellets.length > 0) {
            const nearestGhost = this.getNearestGhost(gameState);
            const ghostDistance = nearestGhost ?
                Math.abs(nearestGhost.x - playerPos.x) + Math.abs(nearestGhost.y - playerPos.y) : 999;

            // If ghost is close, prioritize power pellet
            if (ghostDistance < 6) {
                const nearestPower = this.findNearest(playerPos, gameState.powerPellets);
                if (nearestPower) {
                    return { ...nearestPower, type: 'power_pellet', priority: 1 };
                }
            }
        }

        // Priority 2: Nearest regular pellet
        if (gameState.pellets && gameState.pellets.length > 0) {
            const nearestPellet = this.findNearest(playerPos, gameState.pellets);
            if (nearestPellet) {
                return { ...nearestPellet, type: 'pellet', priority: 2 };
            }
        }

        // Priority 3: Any remaining power pellet
        if (gameState.powerPellets && gameState.powerPellets.length > 0) {
            const nearestPower = this.findNearest(playerPos, gameState.powerPellets);
            if (nearestPower) {
                return { ...nearestPower, type: 'power_pellet', priority: 3 };
            }
        }

        return null;
    }

    /**
     * Find nearest item from a list
     */
    findNearest(from, items) {
        if (!items || items.length === 0) return null;

        let nearest = null;
        let minDist = Infinity;

        for (const item of items) {
            const dist = Math.abs(item.x - from.x) + Math.abs(item.y - from.y);
            if (dist < minDist) {
                minDist = dist;
                nearest = item;
            }
        }

        return nearest;
    }

    /**
     * A* Pathfinding Algorithm
     */
    aStar(start, goal, gameState) {
        const openSet = [start];
        const cameFrom = new Map();
        const gScore = new Map();
        const fScore = new Map();

        const key = (pos) => `${pos.x},${pos.y}`;
        gScore.set(key(start), 0);
        fScore.set(key(start), this.heuristic(start, goal));

        let iterations = 0;
        const maxIterations = 100;

        while (openSet.length > 0 && iterations < maxIterations) {
            iterations++;

            // Find node with lowest fScore
            let current = openSet[0];
            let currentIdx = 0;
            for (let i = 1; i < openSet.length; i++) {
                if (fScore.get(key(openSet[i])) < fScore.get(key(current))) {
                    current = openSet[i];
                    currentIdx = i;
                }
            }

            // Reached goal?
            if (current.x === goal.x && current.y === goal.y) {
                return this.reconstructPath(cameFrom, current, start);
            }

            openSet.splice(currentIdx, 1);

            // Check neighbors
            const neighbors = this.getNeighbors(current, gameState);
            for (const neighbor of neighbors) {
                const tentativeGScore = gScore.get(key(current)) + 1;

                if (!gScore.has(key(neighbor)) || tentativeGScore < gScore.get(key(neighbor))) {
                    cameFrom.set(key(neighbor), current);
                    gScore.set(key(neighbor), tentativeGScore);
                    fScore.set(key(neighbor), tentativeGScore + this.heuristic(neighbor, goal));

                    if (!openSet.some(p => p.x === neighbor.x && p.y === neighbor.y)) {
                        openSet.push(neighbor);
                    }
                }
            }
        }

        // No path found
        return null;
    }

    /**
     * Manhattan distance heuristic
     */
    heuristic(a, b) {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }

    /**
     * Get valid neighbors (not walls, avoid ghosts)
     */
    getNeighbors(pos, gameState) {
        const directions = [
            { dx: 0, dy: -1, action: 'UP' },
            { dx: 0, dy: 1, action: 'DOWN' },
            { dx: -1, dy: 0, action: 'LEFT' },
            { dx: 1, dy: 0, action: 'RIGHT' }
        ];

        const neighbors = [];

        for (const dir of directions) {
            const newPos = {
                x: pos.x + dir.dx,
                y: pos.y + dir.dy,
                action: dir.action
            };

            // Check bounds
            if (newPos.x < 0 || newPos.x >= this.gridSize ||
                newPos.y < 0 || newPos.y >= this.gridSize) {
                continue;
            }

            // Check walls
            if (gameState.walls && gameState.walls.some(w => w.x === newPos.x && w.y === newPos.y)) {
                continue;
            }

            // Avoid ghosts (unless in power mode)
            if (!gameState.powerMode && gameState.ghosts) {
                const tooCloseToGhost = gameState.ghosts.some(g => {
                    const dist = Math.abs(g.x - newPos.x) + Math.abs(g.y - newPos.y);
                    return dist <= 1;  // Don't path through ghost positions
                });
                if (tooCloseToGhost) {
                    continue;
                }
            }

            neighbors.push(newPos);
        }

        return neighbors;
    }

    /**
     * Reconstruct path from A* results
     */
    reconstructPath(cameFrom, current, start) {
        const path = [];
        const key = (pos) => `${pos.x},${pos.y}`;

        while (current && !(current.x === start.x && current.y === start.y)) {
            if (current.action) {
                path.unshift(current.action);
            }
            current = cameFrom.get(key(current));
        }

        return path;
    }

    /**
     * Check if current path is blocked by ghost
     */
    isPathBlocked(gameState) {
        if (!this.currentPlan || !this.currentPlan.path) return false;

        // Check next 2 moves in path
        const lookAhead = 2;
        const startStep = this.currentPlan.step;
        let currentPos = { x: gameState.playerX, y: gameState.playerY };

        for (let i = 0; i < Math.min(lookAhead, this.currentPlan.path.length - startStep); i++) {
            const action = this.currentPlan.path[startStep + i];
            currentPos = this.getNextPosition(currentPos, action);

            // Check if ghost nearby
            for (const ghost of gameState.ghosts) {
                if (ghost.scared) continue;
                const dist = Math.abs(ghost.x - currentPos.x) + Math.abs(ghost.y - currentPos.y);
                if (dist <= 2) {
                    return true;
                }
            }
        }

        return false;
    }

    getNextPosition(pos, action) {
        switch (action) {
            case 'UP': return { x: pos.x, y: pos.y - 1 };
            case 'DOWN': return { x: pos.x, y: pos.y + 1 };
            case 'LEFT': return { x: pos.x - 1, y: pos.y };
            case 'RIGHT': return { x: pos.x + 1, y: pos.y };
            default: return pos;
        }
    }

    /**
     * Check if goal still exists
     */
    goalStillValid(gameState) {
        if (!this.currentPlan || !this.currentPlan.goal) return false;

        const goal = this.currentPlan.goal;

        if (goal.type === 'pellet') {
            return gameState.pellets && gameState.pellets.some(p =>
                p.x === goal.x && p.y === goal.y
            );
        } else if (goal.type === 'power_pellet') {
            return gameState.powerPellets && gameState.powerPellets.some(p =>
                p.x === goal.x && p.y === goal.y
            );
        }

        return false;
    }

    getNearestGhost(gameState) {
        if (!gameState.ghosts || gameState.ghosts.length === 0) return null;

        const playerPos = { x: gameState.playerX, y: gameState.playerY };
        let nearest = null;
        let minDist = Infinity;

        for (const ghost of gameState.ghosts) {
            if (ghost.scared) continue;
            const dist = Math.abs(ghost.x - playerPos.x) + Math.abs(ghost.y - playerPos.y);
            if (dist < minDist) {
                minDist = dist;
                nearest = ghost;
            }
        }

        return nearest;
    }

    /**
     * Reset planner
     */
    reset() {
        this.currentPlan = null;
        this.movesSincePlan = 0;
    }

    /**
     * Get current plan info for debugging
     */
    getPlanInfo() {
        if (!this.currentPlan) return null;

        return {
            goal: this.currentPlan.goal,
            pathLength: this.currentPlan.path.length,
            currentStep: this.currentPlan.step,
            remainingMoves: this.currentPlan.path.length - this.currentPlan.step,
            nextMoves: this.currentPlan.path.slice(this.currentPlan.step, this.currentPlan.step + 3)
        };
    }
}

// Initialize global path planner
window.pathPlanner = new PathPlanner();

// Debug helper
window.debugPathPlanner = () => {
    const info = window.pathPlanner.getPlanInfo();
    if (info) {
        console.log('=== Path Planner Status ===');
        console.log(`Goal: ${info.goal.type} at (${info.goal.x},${info.goal.y})`);
        console.log(`Progress: ${info.currentStep}/${info.pathLength} moves`);
        console.log(`Next moves: ${info.nextMoves.join(' → ')}`);
    } else {
        console.log('No active plan');
    }
    return info;
};
