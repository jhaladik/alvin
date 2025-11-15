/**
 * Pac-Man Game Engine
 * Handles game logic, rendering, and parallel AI avatar
 */

class PacManGame {
    constructor() {
        this.humanCanvas = document.getElementById('humanCanvas');
        this.aiCanvas = document.getElementById('aiCanvas');
        this.humanCtx = this.humanCanvas.getContext('2d');
        this.aiCtx = this.aiCanvas.getContext('2d');

        this.gridSize = 20;
        this.cellSize = this.humanCanvas.width / this.gridSize;

        this.reset();
        this.setupControls();
        this.startGameLoop();
    }

    reset() {
        // Human game state
        this.humanState = this.createInitialState();

        // AI game state (parallel avatar)
        this.aiState = this.createInitialState();

        this.aiEnabled = true;
        this.moveCount = 0;
        this.lastMoveTime = Date.now();

        // Game state management
        this.humanState.gameState = 'playing'; // playing, paused, gameover, won
        this.aiState.gameState = 'playing';
        this.humanState.invincible = false;
        this.aiState.invincible = false;
        this.humanState.invincibleTimer = 0;
        this.aiState.invincibleTimer = 0;

        // Speed settings
        this.moveSpeed = 150; // ms per move (lower = faster)
        this.ghostSpeed = 200;
        this.lastGhostMoveTime = Date.now();

        // AI prediction tracking
        this.lastAIPrediction = null; // Stores last prediction to evaluate
        this.previousAIState = null;  // State before AI moved

        // Move sequence tracking for strategy learning
        this.humanRecentMoves = [];
        this.aiRecentMoves = [];
        this.maxRecentMoves = 5;

        // Initialize DQN agent and vectorization
        if (window.dqnAgent) {
            window.dqnAgent.reset();
        }

        // Reset statistics for new game
        if (window.gameStats) {
            window.gameStats.currentGame = {
                humanStartTime: Date.now(),
                aiStartTime: Date.now(),
                humanMoves: 0,
                aiMoves: 0,
                humanPredictions: [],
                aiPredictions: []
            };
        }

        this.updateUI();
        this.updateStatsUI();
    }

    createInitialState() {
        const walls = this.generateWalls();
        const pellets = this.generatePellets(walls);

        return {
            player: {
                x: 10,
                y: 10,
                direction: 'RIGHT',
                mouthOpen: true,
                startX: 10,
                startY: 10
            },
            ghosts: [
                { x: 5, y: 5, color: '#ff0000', direction: 'RIGHT', startX: 5, startY: 5, dead: false, respawnTimer: 0 },
                { x: 15, y: 5, color: '#ffb8ff', direction: 'LEFT', startX: 15, startY: 5, dead: false, respawnTimer: 0 },
                { x: 5, y: 15, color: '#00ffff', direction: 'UP', startX: 5, startY: 15, dead: false, respawnTimer: 0 },
                { x: 15, y: 15, color: '#ffb852', direction: 'DOWN', startX: 15, startY: 15, dead: false, respawnTimer: 0 }
            ],
            pellets: pellets,
            powerPellets: [
                { x: 2, y: 2 },
                { x: 17, y: 2 },
                { x: 2, y: 17 },
                { x: 17, y: 17 }
            ],
            walls: walls,
            score: 0,
            lives: 3,
            powerMode: false,
            powerModeTimer: 0,
            totalPellets: pellets.length + 4, // including power pellets
            level: 1,
            ghostsEaten: 0
        };
    }

    generatePellets(walls) {
        const pellets = [];
        const powerPelletPositions = [
            { x: 2, y: 2 }, { x: 17, y: 2 },
            { x: 2, y: 17 }, { x: 17, y: 17 }
        ];

        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                // Don't place on walls
                const isWall = walls.some(w => w.x === x && w.y === y);
                if (isWall) continue;

                // Don't place on power pellet positions
                const isPowerPellet = powerPelletPositions.some(p => p.x === x && p.y === y);
                if (isPowerPellet) continue;

                // Don't place on starting positions
                if ((x === 10 && y === 10) ||
                    (x === 5 && y === 5) || (x === 15 && y === 5) ||
                    (x === 5 && y === 15) || (x === 15 && y === 15)) {
                    continue;
                }

                // 70% chance to place a pellet
                if (Math.random() > 0.3) {
                    pellets.push({ x, y });
                }
            }
        }
        return pellets;
    }

    generateWalls() {
        const walls = [];

        // FIXED MAZE (matches training environment exactly!)
        // This is the same maze layout the AI was trained on

        // Border walls
        for (let x = 0; x < this.gridSize; x++) {
            walls.push({ x: x, y: 0 });
            walls.push({ x: x, y: this.gridSize - 1 });
        }
        for (let y = 0; y < this.gridSize; y++) {
            walls.push({ x: 0, y: y });
            walls.push({ x: this.gridSize - 1, y: y });
        }

        // Horizontal walls (top and bottom)
        for (let x = 3; x < 8; x++) {
            walls.push({ x: x, y: 3 });
            walls.push({ x: x, y: this.gridSize - 4 });
        }
        for (let x = this.gridSize - 8; x < this.gridSize - 3; x++) {
            walls.push({ x: x, y: 3 });
            walls.push({ x: x, y: this.gridSize - 4 });
        }

        // Vertical walls (left and right)
        for (let y = 5; y < 10; y++) {
            walls.push({ x: 5, y: y });
            walls.push({ x: this.gridSize - 6, y: y });
        }
        for (let y = this.gridSize - 10; y < this.gridSize - 5; y++) {
            walls.push({ x: 5, y: y });
            walls.push({ x: this.gridSize - 6, y: y });
        }

        // Center ghost house (with opening at top for Pac-Man to exit)
        const centerX = Math.floor(this.gridSize / 2);
        const centerY = Math.floor(this.gridSize / 2);

        // Bottom wall
        for (let x = centerX - 2; x <= centerX + 2; x++) {
            walls.push({ x: x, y: centerY + 2 });
        }
        // Left wall
        for (let y = centerY - 2; y <= centerY + 2; y++) {
            walls.push({ x: centerX - 2, y: y });
        }
        // Right wall
        for (let y = centerY - 2; y <= centerY + 2; y++) {
            walls.push({ x: centerX + 2, y: y });
        }
        // Top wall with opening in the middle (Pac-Man can exit here)
        walls.push({ x: centerX - 2, y: centerY - 2 });
        walls.push({ x: centerX - 1, y: centerY - 2 });
        // centerX is open (exit)
        walls.push({ x: centerX + 1, y: centerY - 2 });
        walls.push({ x: centerX + 2, y: centerY - 2 });

        // Corner blocks
        for (let dx = 0; dx <= 1; dx++) {
            for (let dy = 0; dy <= 1; dy++) {
                walls.push({ x: 3 + dx, y: 5 + dy });
                walls.push({ x: this.gridSize - 4 - dx, y: 5 + dy });
                walls.push({ x: 3 + dx, y: this.gridSize - 6 - dy });
                walls.push({ x: this.gridSize - 4 - dx, y: this.gridSize - 6 - dy });
            }
        }

        return walls;
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            const key = e.key;

            // Pause/Resume
            if (key === ' ' || key === 'Escape') {
                e.preventDefault();
                this.togglePause();
                return;
            }

            // Only accept movement if game is playing
            if (this.humanState.gameState !== 'playing') {
                return;
            }

            let newDirection = null;
            switch (key) {
                case 'ArrowUp':
                case 'w':
                case 'W':
                    newDirection = 'UP';
                    break;
                case 'ArrowDown':
                case 's':
                case 'S':
                    newDirection = 'DOWN';
                    break;
                case 'ArrowLeft':
                case 'a':
                case 'A':
                    newDirection = 'LEFT';
                    break;
                case 'ArrowRight':
                case 'd':
                case 'D':
                    newDirection = 'RIGHT';
                    break;
            }

            if (newDirection) {
                e.preventDefault();
                this.humanState.player.direction = newDirection;

                // Vectorize human move
                this.vectorizeMove(newDirection);
            }
        });
    }

    togglePause() {
        if (this.humanState.gameState === 'playing') {
            this.humanState.gameState = 'paused';
            this.aiState.gameState = 'paused';
        } else if (this.humanState.gameState === 'paused') {
            this.humanState.gameState = 'playing';
            this.aiState.gameState = 'playing';
        }
        this.updateUI();
    }

    async vectorizeMove(direction) {
        this.moveCount++;
        document.getElementById('vectorizedMoves').textContent = this.moveCount;

        // Track recent moves for sequence learning
        this.humanRecentMoves.push(direction);
        if (this.humanRecentMoves.length > this.maxRecentMoves) {
            this.humanRecentMoves.shift();
        }

        // Send to vectorization system with recent moves
        if (window.vectorization) {
            await window.vectorization.vectorizeGameState(
                this.getSerializableState(this.humanState),
                direction,
                [...this.humanRecentMoves] // Pass copy of recent moves
            );
        }
    }

    startGameLoop() {
        const loop = () => {
            this.update();
            this.render();
            requestAnimationFrame(loop);
        };
        loop();
    }

    update() {
        const now = Date.now();

        // Don't update if paused
        if (this.humanState.gameState === 'paused') {
            return;
        }

        // Player movement update
        const deltaTime = now - this.lastMoveTime;
        if (deltaTime > this.moveSpeed) {
            this.lastMoveTime = now;

            // Update human player
            if (this.humanState.gameState === 'playing') {
                this.updatePlayer(this.humanState);
                this.checkCollisions(this.humanState);
                this.checkWinCondition(this.humanState);
            }

            // Update AI avatar
            if (this.aiEnabled && this.aiState.gameState === 'playing') {
                // Capture state before AI moves
                this.previousAIState = {
                    score: this.aiState.score,
                    lives: this.aiState.lives,
                    playerX: this.aiState.player.x,
                    playerY: this.aiState.player.y,
                    pelletsLeft: this.aiState.pellets.length
                };

                this.updateAI();
                this.updatePlayer(this.aiState);
                this.checkCollisions(this.aiState);
                this.checkWinCondition(this.aiState);

                // Evaluate the AI's prediction after move completes
                this.evaluateAIPrediction();
            }

            // Toggle mouth animation
            this.humanState.player.mouthOpen = !this.humanState.player.mouthOpen;
            this.aiState.player.mouthOpen = !this.aiState.player.mouthOpen;
        }

        // Ghost movement update (slower than player)
        const ghostDeltaTime = now - this.lastGhostMoveTime;
        if (ghostDeltaTime > this.ghostSpeed) {
            this.lastGhostMoveTime = now;

            if (this.humanState.gameState === 'playing') {
                this.updateGhosts(this.humanState);
            }
            if (this.aiEnabled && this.aiState.gameState === 'playing') {
                this.updateGhosts(this.aiState);
            }
        }

        // Update power mode timer
        this.updatePowerMode(this.humanState);
        this.updatePowerMode(this.aiState);

        // Update invincibility timer
        this.updateInvincibility(this.humanState);
        this.updateInvincibility(this.aiState);
    }

    updatePowerMode(state) {
        if (state.powerMode) {
            state.powerModeTimer--;
            if (state.powerModeTimer <= 0) {
                state.powerMode = false;
            }
        }
    }

    updateInvincibility(state) {
        if (state.invincible) {
            state.invincibleTimer--;
            if (state.invincibleTimer <= 0) {
                state.invincible = false;
            }
        }
    }

    checkWinCondition(state) {
        const remainingPellets = state.pellets.length + state.powerPellets.length;
        if (remainingPellets === 0 && state.gameState === 'playing') {
            state.gameState = 'won';

            // Record win
            if (window.gameStats) {
                const player = state === this.humanState ? 'human' : 'ai';
                window.gameStats.endGame(player, state);
            }

            this.updateUI();
        }
    }

    async updateAI() {
        // Get AI prediction from DQN agent
        if (window.dqnAgent) {
            const prediction = await window.dqnAgent.predictNextMove(
                this.getSerializableState(this.aiState)
            );

            if (prediction && prediction.action) {
                // Store prediction for evaluation after move
                this.lastAIPrediction = {
                    action: prediction.action,
                    confidence: prediction.confidence,
                    timestamp: Date.now(),
                    explored: prediction.explored,
                    cached: prediction.cached,
                    fallback: prediction.fallback
                };

                this.aiState.player.direction = prediction.action;

                // Update UI with AI's next move
                document.getElementById('aiNextMove').textContent = prediction.action;
                const confidence = Math.round(prediction.confidence * 100);
                document.getElementById('aiConfidence').textContent = confidence + '%';

                // Update confidence bar
                const confidenceBar = document.getElementById('confidenceBar');
                if (confidenceBar) {
                    confidenceBar.style.width = confidence + '%';
                }

                // Update prediction mode (show decision source)
                const modeEl = document.getElementById('predictionMode');
                if (modeEl) {
                    let modeText = '';
                    if (prediction.explored) {
                        modeText = 'Explore';
                    } else if (prediction.cached) {
                        modeText = 'Cached';
                    } else if (prediction.fallback) {
                        modeText = 'Fallback';
                    } else if (prediction.decisionSource === 'path_planning') {
                        modeText = 'Path Plan';
                    } else if (prediction.overridden) {
                        modeText = 'Override';
                    } else {
                        modeText = 'Exploit';
                    }
                    modeEl.textContent = modeText;
                }

                // Update decision quality (NEW!)
                const qualityEl = document.getElementById('decisionQuality');
                if (qualityEl && prediction.metrics) {
                    const quality = prediction.metrics.decisionQuality;
                    qualityEl.textContent = quality.toUpperCase();
                    qualityEl.className = 'ai-prediction quality-' + quality;
                }

                // Update score spread (NEW!)
                this.updateScoreSpread(prediction.allScores, prediction.action);

                // Update learned from counter (show similarStatesFound from backend)
                const learnedFromEl = document.getElementById('vectorizedMoves');
                if (learnedFromEl && prediction.learnedFrom !== undefined) {
                    learnedFromEl.textContent = prediction.learnedFrom;
                }

                // Update ML Prediction Status
                this.updateMLStatusUI(prediction);

                // Obsolete sections removed (path planning, strategy, decision details)
                // Now using pure DQN Q-value based decisions from HF Spaces
            }
        }
    }

    updatePlayer(state) {
        const player = state.player;
        const newPos = this.getNextPosition(player.x, player.y, player.direction);

        // Check if move is valid (not a wall)
        if (!this.isWall(newPos.x, newPos.y, state.walls)) {
            player.x = newPos.x;
            player.y = newPos.y;

            // Wrap around edges
            player.x = (player.x + this.gridSize) % this.gridSize;
            player.y = (player.y + this.gridSize) % this.gridSize;
        }
    }

    updateGhosts(state) {
        state.ghosts.forEach(ghost => {
            // Handle ghost respawn timer
            if (ghost.dead) {
                ghost.respawnTimer--;
                if (ghost.respawnTimer <= 0) {
                    ghost.dead = false;
                }
                return; // Skip movement for dead ghosts
            }

            // Simple AI for ghosts - random movement
            if (Math.random() < 0.1) {
                const directions = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
                ghost.direction = directions[Math.floor(Math.random() * directions.length)];
            }

            const newPos = this.getNextPosition(ghost.x, ghost.y, ghost.direction);
            if (!this.isWall(newPos.x, newPos.y, state.walls)) {
                ghost.x = newPos.x;
                ghost.y = newPos.y;

                // Wrap around
                ghost.x = (ghost.x + this.gridSize) % this.gridSize;
                ghost.y = (ghost.y + this.gridSize) % this.gridSize;
            }
        });
    }

    checkCollisions(state) {
        const player = state.player;

        // Check pellet collection
        state.pellets = state.pellets.filter(pellet => {
            if (pellet.x === player.x && pellet.y === player.y) {
                state.score += 10;
                return false;
            }
            return true;
        });

        // Check power pellet collection
        state.powerPellets = state.powerPellets.filter(pellet => {
            if (pellet.x === player.x && pellet.y === player.y) {
                state.score += 50;
                state.powerMode = true;
                state.powerModeTimer = 360; // 6 seconds at 60 FPS
                return false;
            }
            return true;
        });

        // Check ghost collisions (only if not invincible)
        if (!state.invincible) {
            for (const ghost of state.ghosts) {
                // Skip dead ghosts
                if (ghost.dead) continue;

                if (ghost.x === player.x && ghost.y === player.y) {
                    if (state.powerMode) {
                        // Eat ghost
                        state.score += 200;
                        state.ghostsEaten++;
                        // Mark ghost as dead and set respawn timer (3 seconds = 180 frames)
                        ghost.dead = true;
                        ghost.respawnTimer = 180;
                        ghost.x = ghost.startX;
                        ghost.y = ghost.startY;
                    } else {
                        // Die
                        this.handleDeath(state);
                        break; // Only die once per frame
                    }
                }
            }
        }

        // Update UI
        this.updateUI();
    }

    handleDeath(state) {
        state.lives--;

        // Track death in statistics
        if (window.gameStats) {
            const player = state === this.humanState ? 'human' : 'ai';
            window.gameStats.trackMove(player, 'death', 'death');

            if (player === 'human') {
                window.gameStats.humanStats.totalDeaths++;
            } else {
                window.gameStats.aiStats.totalDeaths++;
            }
        }

        if (state.lives <= 0) {
            // Game Over
            state.gameState = 'gameover';

            // Record game end
            if (window.gameStats) {
                const player = state === this.humanState ? 'human' : 'ai';
                window.gameStats.endGame(player, state);
            }

            this.updateUI();
        } else {
            // Respawn player
            state.player.x = state.player.startX;
            state.player.y = state.player.startY;
            state.player.direction = 'RIGHT';

            // Respawn ghosts to starting positions and reset their state
            state.ghosts.forEach(ghost => {
                ghost.x = ghost.startX;
                ghost.y = ghost.startY;
                ghost.dead = false;
                ghost.respawnTimer = 0;
            });

            // Grant invincibility for 3 seconds (180 frames at 60 FPS)
            state.invincible = true;
            state.invincibleTimer = 180;

            // Clear power mode on death
            state.powerMode = false;
            state.powerModeTimer = 0;
        }
    }

    getNextPosition(x, y, direction) {
        switch (direction) {
            case 'UP': return { x, y: y - 1 };
            case 'DOWN': return { x, y: y + 1 };
            case 'LEFT': return { x: x - 1, y };
            case 'RIGHT': return { x: x + 1, y };
            default: return { x, y };
        }
    }

    isWall(x, y, walls) {
        return walls.some(wall => wall.x === x && wall.y === y);
    }

    render() {
        // Render human game
        this.renderGame(this.humanCtx, this.humanState, '#ffff00');

        // Render AI game
        if (this.aiEnabled) {
            this.renderGame(this.aiCtx, this.aiState, '#ff00ff');
            // Render planned path overlay for AI
            this.renderPlannedPath(this.aiCtx, this.aiState);
        }
    }

    updateUI() {
        // Update human stats
        document.getElementById('humanScore').textContent = this.humanState.score;
        document.getElementById('humanLives').textContent = this.humanState.lives;
        document.getElementById('humanLevel').textContent = this.humanState.level;
        document.getElementById('humanPellets').textContent =
            this.humanState.pellets.length + this.humanState.powerPellets.length;

        // Update AI stats
        document.getElementById('aiScore').textContent = this.aiState.score;
        document.getElementById('aiLives').textContent = this.aiState.lives;
        document.getElementById('aiLevel').textContent = this.aiState.level;
        document.getElementById('aiPellets').textContent =
            this.aiState.pellets.length + this.aiState.powerPellets.length;

        // Update vectorization count
        document.getElementById('vectorizedMoves').textContent = this.moveCount;

        // Update game state indicators
        this.updateGameStateIndicator('humanStatus', this.humanState);
        this.updateGameStateIndicator('aiStatus', this.aiState);

        // Update statistics UI
        this.updateStatsUI();
    }

    updateStatsUI() {
        if (!window.gameStats) return;

        const stats = window.gameStats.getSummary();

        // Learning metrics
        const learningEl = document.getElementById('learningMetrics');
        if (learningEl) {
            learningEl.innerHTML = `
                <div class="stat-mini">
                    <span>Predictions:</span> <strong>${stats.learning.totalPredictions}</strong>
                </div>
                <div class="stat-mini">
                    <span>Avg Confidence:</span> <strong>${(stats.learning.averageConfidence * 100).toFixed(1)}%</strong>
                </div>
            `;
        }

        // Comparison metrics
        const comparisonEl = document.getElementById('comparisonMetrics');
        if (comparisonEl) {
            const humanWins = stats.comparison.gamesWhereHumanWon;
            const aiWins = stats.comparison.gamesWhereAIWon;
            const winner = humanWins > aiWins ? '👤 Human Leading!' :
                          aiWins > humanWins ? '🤖 AI Leading!' :
                          '⚖️ Tied!';

            comparisonEl.innerHTML = `
                <div class="stat-mini">
                    <span>Human Wins:</span> <strong style="color: #ffff00">${humanWins}</strong>
                </div>
                <div class="stat-mini">
                    <span>AI Wins:</span> <strong style="color: #ff00ff">${aiWins}</strong>
                </div>
                <div class="stat-mini">
                    <span>Human Avg:</span> <strong>${stats.human.avgScore.toFixed(0)}</strong>
                </div>
                <div class="stat-mini">
                    <span>AI Avg:</span> <strong>${stats.ai.avgScore.toFixed(0)}</strong>
                </div>
                <div class="stat-mini winner">
                    ${winner}
                </div>
            `;
        }
    }


    updateGameStateIndicator(elementId, state) {
        const statusEl = document.getElementById(elementId);
        if (!statusEl) return;

        statusEl.className = 'game-status';

        switch (state.gameState) {
            case 'playing':
                if (state.invincible) {
                    statusEl.textContent = 'INVINCIBLE';
                    statusEl.classList.add('status-invincible');
                } else if (state.powerMode) {
                    statusEl.textContent = 'POWER MODE';
                    statusEl.classList.add('status-power');
                } else {
                    statusEl.textContent = 'PLAYING';
                    statusEl.classList.add('status-playing');
                }
                break;
            case 'paused':
                statusEl.textContent = 'PAUSED';
                statusEl.classList.add('status-paused');
                break;
            case 'gameover':
                statusEl.textContent = 'GAME OVER';
                statusEl.classList.add('status-gameover');
                break;
            case 'won':
                statusEl.textContent = 'YOU WON!';
                statusEl.classList.add('status-won');
                break;
        }
    }

    renderGame(ctx, state, playerColor) {
        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Draw walls
        ctx.fillStyle = '#0000ff';
        state.walls.forEach(wall => {
            ctx.fillRect(
                wall.x * this.cellSize,
                wall.y * this.cellSize,
                this.cellSize,
                this.cellSize
            );
        });

        // Draw pellets
        ctx.fillStyle = '#fff';
        state.pellets.forEach(pellet => {
            ctx.beginPath();
            ctx.arc(
                pellet.x * this.cellSize + this.cellSize / 2,
                pellet.y * this.cellSize + this.cellSize / 2,
                2,
                0,
                Math.PI * 2
            );
            ctx.fill();
        });

        // Draw power pellets (blinking effect)
        const blinkOn = Math.floor(Date.now() / 300) % 2 === 0;
        if (blinkOn) {
            ctx.fillStyle = '#00ffff';
            state.powerPellets.forEach(pellet => {
                ctx.beginPath();
                ctx.arc(
                    pellet.x * this.cellSize + this.cellSize / 2,
                    pellet.y * this.cellSize + this.cellSize / 2,
                    4,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            });
        }

        // Draw ghosts
        state.ghosts.forEach(ghost => {
            // Don't draw dead ghosts (they're being eaten / respawning)
            if (ghost.dead) return;

            ctx.fillStyle = state.powerMode ? '#0000ff' : ghost.color;
            this.drawGhost(
                ctx,
                ghost.x * this.cellSize,
                ghost.y * this.cellSize,
                this.cellSize
            );
        });

        // Draw Pac-Man (blink during invincibility)
        if (!state.invincible || Math.floor(Date.now() / 150) % 2 === 0) {
            ctx.fillStyle = state.invincible ? '#ffffff' : playerColor;
            this.drawPacMan(
                ctx,
                state.player.x * this.cellSize,
                state.player.y * this.cellSize,
                this.cellSize,
                state.player.direction,
                state.player.mouthOpen
            );
        }

        // Draw game state overlay
        if (state.gameState !== 'playing') {
            this.drawOverlay(ctx, state);
        }
    }

    drawOverlay(ctx, state) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 24px "Courier New"';
        ctx.textAlign = 'center';

        let message = '';
        switch (state.gameState) {
            case 'paused':
                message = 'PAUSED';
                ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2 - 20);
                ctx.font = '14px "Courier New"';
                ctx.fillText('Press SPACE to resume', ctx.canvas.width / 2, ctx.canvas.height / 2 + 20);
                break;
            case 'gameover':
                ctx.fillStyle = '#ff0000';
                message = 'GAME OVER';
                ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2 - 30);
                ctx.fillStyle = '#fff';
                ctx.font = '18px "Courier New"';
                ctx.fillText(`Score: ${state.score}`, ctx.canvas.width / 2, ctx.canvas.height / 2 + 10);
                ctx.font = '14px "Courier New"';
                ctx.fillText('Click "Reset Game" to play again', ctx.canvas.width / 2, ctx.canvas.height / 2 + 40);
                break;
            case 'won':
                ctx.fillStyle = '#00ff00';
                message = 'YOU WON!';
                ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2 - 30);
                ctx.fillStyle = '#fff';
                ctx.font = '18px "Courier New"';
                ctx.fillText(`Final Score: ${state.score}`, ctx.canvas.width / 2, ctx.canvas.height / 2 + 10);
                ctx.font = '14px "Courier New"';
                ctx.fillText('Click "Reset Game" for next level', ctx.canvas.width / 2, ctx.canvas.height / 2 + 40);
                break;
        }

        ctx.textAlign = 'left';
    }

    renderPlannedPath(ctx, state) {
        // Only render path for AI when there's an active plan
        if (!window.pathPlanner || !window.pathPlanner.currentPlan || state.gameState !== 'playing') {
            return;
        }

        const plan = window.pathPlanner.currentPlan;
        if (!plan.path || plan.path.length === 0) {
            return;
        }

        // Start from current AI position
        let currentX = state.player.x;
        let currentY = state.player.y;

        // Draw goal marker (pulsing circle)
        const pulseSize = 3 + Math.sin(Date.now() / 200) * 2;
        ctx.strokeStyle = '#9966ff';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.arc(
            plan.goal.x * this.cellSize + this.cellSize / 2,
            plan.goal.y * this.cellSize + this.cellSize / 2,
            pulseSize + 5,
            0,
            Math.PI * 2
        );
        ctx.stroke();

        // Draw path as connected lines with gradient
        ctx.strokeStyle = '#9966ff';
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.6;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        ctx.moveTo(
            currentX * this.cellSize + this.cellSize / 2,
            currentY * this.cellSize + this.cellSize / 2
        );

        // Draw remaining path
        for (let i = plan.step; i < plan.path.length; i++) {
            const move = plan.path[i];
            const nextPos = this.getNextPosition(currentX, currentY, move);
            currentX = nextPos.x;
            currentY = nextPos.y;

            ctx.lineTo(
                currentX * this.cellSize + this.cellSize / 2,
                currentY * this.cellSize + this.cellSize / 2
            );
        }

        ctx.stroke();

        // Draw waypoint markers (small dots along the path)
        currentX = state.player.x;
        currentY = state.player.y;
        ctx.fillStyle = '#cc99ff';
        ctx.globalAlpha = 0.5;

        for (let i = plan.step; i < plan.path.length; i++) {
            const move = plan.path[i];
            const nextPos = this.getNextPosition(currentX, currentY, move);
            currentX = nextPos.x;
            currentY = nextPos.y;

            ctx.beginPath();
            ctx.arc(
                currentX * this.cellSize + this.cellSize / 2,
                currentY * this.cellSize + this.cellSize / 2,
                3,
                0,
                Math.PI * 2
            );
            ctx.fill();
        }

        // Reset alpha
        ctx.globalAlpha = 1.0;
    }

    drawPacMan(ctx, x, y, size, direction, mouthOpen) {
        const centerX = x + size / 2;
        const centerY = y + size / 2;
        const radius = size / 2 - 2;

        ctx.beginPath();

        if (mouthOpen) {
            let startAngle, endAngle;
            switch (direction) {
                case 'RIGHT':
                    startAngle = 0.25;
                    endAngle = 1.75;
                    break;
                case 'LEFT':
                    startAngle = 1.25;
                    endAngle = 0.75;
                    break;
                case 'UP':
                    startAngle = 1.75;
                    endAngle = 1.25;
                    break;
                case 'DOWN':
                    startAngle = 0.75;
                    endAngle = 0.25;
                    break;
            }
            ctx.arc(centerX, centerY, radius, startAngle * Math.PI, endAngle * Math.PI);
            ctx.lineTo(centerX, centerY);
        } else {
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        }

        ctx.fill();
    }

    drawGhost(ctx, x, y, size) {
        const centerX = x + size / 2;
        const centerY = y + size / 2;

        ctx.beginPath();
        ctx.arc(centerX, centerY - size / 4, size / 3, Math.PI, 0, false);
        ctx.lineTo(x + size - 2, y + size - 2);
        ctx.lineTo(x + 2, y + size - 2);
        ctx.closePath();
        ctx.fill();

        // Eyes
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(centerX - 4, centerY - size / 4, 2, 0, Math.PI * 2);
        ctx.arc(centerX + 4, centerY - size / 4, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    updateScoreSpread(allScores, chosenAction) {
        const spreadEl = document.getElementById('scoreSpread');
        if (!spreadEl || !allScores) return;

        const scores = Object.entries(allScores).map(([action, score]) => ({
            action,
            score,
            chosen: action === chosenAction
        }));

        // Sort by score (best first)
        scores.sort((a, b) => b.score - a.score);

        // Find min/max for scaling
        const minScore = Math.min(...scores.map(s => s.score));
        const maxScore = Math.max(...scores.map(s => s.score));
        const range = maxScore - minScore;

        spreadEl.innerHTML = scores.map(s => {
            // Scale to 0-100 for visualization
            const normalized = range > 0 ? ((s.score - minScore) / range) * 100 : 50;
            const barColor = s.score > 0 ? '#00ff00' :
                           s.score < 0 ? '#ff0000' : '#666';
            const highlight = s.chosen ? 'border: 2px solid #ffff00; font-weight: bold;' : '';

            return `<div style="margin: 5px 0; padding: 3px; ${highlight}">
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 2px;">
                    <span style="color: ${s.chosen ? '#ffff00' : '#aaa'};">${s.action}</span>
                    <span style="color: ${barColor};">${s.score > 0 ? '+' : ''}${s.score.toFixed(1)}</span>
                </div>
                <div style="background: #222; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: ${barColor}; width: ${normalized}%; height: 100%; transition: width 0.3s;"></div>
                </div>
            </div>`;
        }).join('');
    }

    updatePathPlanningUI() {
        const statusEl = document.getElementById('planStatus');
        const goalEl = document.getElementById('planGoal');
        const movesLeftEl = document.getElementById('planMovesLeft');
        const progressBar = document.getElementById('planProgressBar');

        if (!window.pathPlanner || !window.pathPlanner.currentPlan) {
            // No active plan
            if (statusEl) statusEl.textContent = 'No Plan';
            if (statusEl) statusEl.style.color = '#666';
            if (goalEl) goalEl.textContent = '-';
            if (movesLeftEl) movesLeftEl.textContent = '-';
            if (progressBar) progressBar.style.width = '0%';
            return;
        }

        const plan = window.pathPlanner.currentPlan;
        const movesTotal = plan.path.length;
        const movesLeft = movesTotal - plan.step;
        const progress = movesTotal > 0 ? ((plan.step / movesTotal) * 100) : 0;

        // Update UI
        if (statusEl) {
            statusEl.textContent = 'Active';
            statusEl.style.color = '#9966ff';
        }

        if (goalEl) {
            const goalType = plan.goal.type || 'pellet';
            goalEl.textContent = `${goalType} at (${plan.goal.x}, ${plan.goal.y})`;
        }

        if (movesLeftEl) {
            movesLeftEl.textContent = `${movesLeft} / ${movesTotal}`;
        }

        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
    }

    updateMLStatusUI(prediction) {
        // Update ML Active status
        const mlActiveEl = document.getElementById('mlActive');
        if (mlActiveEl) {
            const isMLActive = prediction.usedML === true;
            mlActiveEl.textContent = isMLActive ? 'YES' : 'NO';
            mlActiveEl.style.color = isMLActive ? '#00ff00' : '#ff6600';
        }

        // Update ML Method
        const mlMethodEl = document.getElementById('mlMethod');
        if (mlMethodEl) {
            if (prediction.mlMethod) {
                mlMethodEl.textContent = prediction.mlMethod.toUpperCase();
                // Color code by method
                switch(prediction.mlMethod) {
                    case 'ml':
                        mlMethodEl.style.color = '#00ff00';
                        break;
                    case 'hybrid':
                        mlMethodEl.style.color = '#ffaa00';
                        break;
                    case 'heuristic_fallback':
                        mlMethodEl.style.color = '#ff6600';
                        break;
                    default:
                        mlMethodEl.style.color = '#888';
                }
            } else {
                mlMethodEl.textContent = 'N/A';
                mlMethodEl.style.color = '#666';
            }
        }

        // Update ML Source
        const mlSourceEl = document.getElementById('mlSource');
        if (mlSourceEl) {
            if (prediction.mlSource) {
                const sourceText = prediction.mlSource === 'ml_feature_based'
                    ? 'Feature-Based'
                    : prediction.mlSource === 'similarity_search'
                    ? 'Similarity Search'
                    : prediction.mlSource;
                mlSourceEl.textContent = sourceText;
                mlSourceEl.style.color = '#00ffff';
            } else {
                mlSourceEl.textContent = 'N/A';
                mlSourceEl.style.color = '#666';
            }
        }

        // Update ML Probabilities
        if (prediction.mlProbabilities) {
            const probs = prediction.mlProbabilities;

            // UP
            const upEl = document.getElementById('mlProbUp');
            if (upEl) {
                const upProb = (probs.UP || 0) * 100;
                upEl.textContent = upProb.toFixed(1) + '%';
                upEl.style.color = upProb > 50 ? '#00ff00' : '#888';
            }

            // DOWN
            const downEl = document.getElementById('mlProbDown');
            if (downEl) {
                const downProb = (probs.DOWN || 0) * 100;
                downEl.textContent = downProb.toFixed(1) + '%';
                downEl.style.color = downProb > 50 ? '#00ff00' : '#888';
            }

            // LEFT
            const leftEl = document.getElementById('mlProbLeft');
            if (leftEl) {
                const leftProb = (probs.LEFT || 0) * 100;
                leftEl.textContent = leftProb.toFixed(1) + '%';
                leftEl.style.color = leftProb > 50 ? '#00ff00' : '#888';
            }

            // RIGHT
            const rightEl = document.getElementById('mlProbRight');
            if (rightEl) {
                const rightProb = (probs.RIGHT || 0) * 100;
                rightEl.textContent = rightProb.toFixed(1) + '%';
                rightEl.style.color = rightProb > 50 ? '#00ff00' : '#888';
            }
        } else {
            // No probabilities available
            ['mlProbUp', 'mlProbDown', 'mlProbLeft', 'mlProbRight'].forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.textContent = '-';
                    el.style.color = '#666';
                }
            });
        }

        // Update DQN Q-Values (from Hugging Face Spaces)
        if (prediction.q_values) {
            const qValues = prediction.q_values;
            const maxQ = Math.max(qValues.UP || 0, qValues.DOWN || 0, qValues.LEFT || 0, qValues.RIGHT || 0);

            // Helper function to format Q-value with color
            const formatQValue = (action) => {
                const q = qValues[action] || 0;
                const el = document.getElementById(`qValue${action.charAt(0) + action.slice(1).toLowerCase()}`);
                if (el) {
                    el.textContent = q.toFixed(2);
                    // Color code: highest Q-value = green, others = dimmed
                    if (Math.abs(q - maxQ) < 0.01) {
                        el.style.color = '#00ff00';
                        el.style.textShadow = '0 0 5px #00ff00';
                    } else if (q > 0) {
                        el.style.color = '#88ff88';
                        el.style.textShadow = 'none';
                    } else if (q < 0) {
                        el.style.color = '#ff8888';
                        el.style.textShadow = 'none';
                    } else {
                        el.style.color = '#888';
                        el.style.textShadow = 'none';
                    }
                }
            };

            formatQValue('UP');
            formatQValue('DOWN');
            formatQValue('LEFT');
            formatQValue('RIGHT');

            // Update inference stats
            this.updateInferenceStats(qValues);
        } else {
            // No Q-values available
            ['qValueUp', 'qValueDown', 'qValueLeft', 'qValueRight'].forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.textContent = '-';
                    el.style.color = '#666';
                    el.style.textShadow = 'none';
                }
            });
        }

        // Update action distribution
        this.updateActionDistribution(prediction.action);
    }

    updateInferenceStats(qValues) {
        // Calculate stats
        const qValueArray = [qValues.UP, qValues.DOWN, qValues.LEFT, qValues.RIGHT];
        const avgQ = qValueArray.reduce((a, b) => a + b, 0) / 4;
        const maxQ = Math.max(...qValueArray);

        // Update avg Q-value
        const avgEl = document.getElementById('avgQValue');
        if (avgEl) {
            avgEl.textContent = avgQ.toFixed(2);
            avgEl.style.color = avgQ > 0 ? '#00ff00' : avgQ < 0 ? '#ff8888' : '#888';
        }

        // Update max Q-value
        const maxEl = document.getElementById('maxQValue');
        if (maxEl) {
            maxEl.textContent = maxQ.toFixed(2);
            maxEl.style.color = '#00ff00';
        }

        // Update total predictions
        if (!this.totalPredictions) this.totalPredictions = 0;
        this.totalPredictions++;
        const totalEl = document.getElementById('totalPredictions');
        if (totalEl) {
            totalEl.textContent = this.totalPredictions.toString();
        }
    }

    updateActionDistribution(action) {
        // Track last 20 actions
        if (!this.actionHistory) this.actionHistory = [];
        this.actionHistory.push(action);
        if (this.actionHistory.length > 20) {
            this.actionHistory.shift();
        }

        // Count distribution
        const counts = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
        this.actionHistory.forEach(a => counts[a]++);

        const total = this.actionHistory.length;

        // Update UI for each action
        ['UP', 'DOWN', 'LEFT', 'RIGHT'].forEach(action => {
            const pct = total > 0 ? (counts[action] / total) * 100 : 0;

            const textEl = document.getElementById(`dist${action.charAt(0) + action.slice(1).toLowerCase()}`);
            const barEl = document.getElementById(`dist${action.charAt(0) + action.slice(1).toLowerCase()}Bar`);

            if (textEl) textEl.textContent = pct.toFixed(0) + '%';
            if (barEl) barEl.style.width = pct + '%';
        });
    }

    updateDecisionDetailsUI(prediction) {
        const detailsEl = document.getElementById('decisionDetails');
        if (!detailsEl) return;

        let html = '';

        // Show decision source
        if (prediction.decisionSource) {
            const sourceColor = prediction.decisionSource === 'path_planning' ? '#9966ff' : '#00ffff';
            html += `<div style="margin: 5px 0;">
                <strong style="color: ${sourceColor};">Source:</strong>
                ${prediction.decisionSource === 'path_planning' ? 'Path Planning' : 'Vectorization'}
            </div>`;
        }

        // Show if plan was overridden
        if (prediction.plannedMove) {
            const wasOverridden = prediction.plannedMove !== prediction.action;

            if (wasOverridden) {
                // Plan was overridden - show why
                const plannedScore = prediction.allScores ? prediction.allScores[prediction.plannedMove] : 'N/A';
                const actualScore = prediction.allScores ? prediction.allScores[prediction.action] : 'N/A';

                html += `<div style="margin: 8px 0; padding: 8px; background: #2a1a1a; border-left: 3px solid #ff6600; border-radius: 3px;">
                    <div style="color: #ff6600; font-weight: bold; margin-bottom: 5px;">⚠️ Plan Overridden</div>
                    <div style="margin: 3px 0;">
                        <span style="color: #888;">Planned:</span>
                        <strong style="color: #9966ff;">${prediction.plannedMove}</strong>
                        <span style="color: ${plannedScore < -5 ? '#ff0000' : '#666'};">(${typeof plannedScore === 'number' ? plannedScore.toFixed(1) : plannedScore})</span>
                    </div>
                    <div style="margin: 3px 0;">
                        <span style="color: #888;">Actual:</span>
                        <strong style="color: #00ff00;">${prediction.action}</strong>
                        <span style="color: #00ff00;">(${typeof actualScore === 'number' ? actualScore.toFixed(1) : actualScore})</span>
                    </div>
                    <div style="margin-top: 5px; font-size: 10px; color: #ff6600;">
                        Reason: Planned move too dangerous (score < -5)
                    </div>
                </div>`;
            } else {
                // Following plan
                html += `<div style="margin: 8px 0; padding: 8px; background: #1a1a2a; border-left: 3px solid #9966ff; border-radius: 3px;">
                    <div style="color: #9966ff; font-weight: bold;">✓ Following Plan</div>
                    <div style="margin-top: 5px; font-size: 10px; color: #aaa;">
                        Planned route is safe and optimal
                    </div>
                </div>`;
            }
        }

        // Show exploration/fallback/cached status
        if (prediction.explored) {
            html += `<div style="margin: 5px 0; color: #ffaa00;">
                <strong>🎲 Exploration:</strong> Random move for learning
            </div>`;
        } else if (prediction.cached) {
            html += `<div style="margin: 5px 0; color: #666;">
                <strong>💾 Cached:</strong> Using recent prediction
            </div>`;
        } else if (prediction.fallback) {
            html += `<div style="margin: 5px 0; color: #ff6600;">
                <strong>⚠️ Fallback:</strong> Using heuristics (API unavailable)
            </div>`;
        }

        // Show query strategy (how AI learned)
        if (prediction.queryStrategy) {
            let strategyLabel = '';
            let strategyColor = '#666';
            let strategyIcon = '🔍';

            if (prediction.queryStrategy === 'filtered_strategy') {
                strategyLabel = 'Filtered: Strategy + Success + Context';
                strategyColor = '#9966ff';
                strategyIcon = '🎖️';
            } else if (prediction.queryStrategy === 'filtered_success') {
                strategyLabel = 'Filtered: Successful moves only';
                strategyColor = '#00ff00';
                strategyIcon = '✨';
            } else if (prediction.queryStrategy === 'filtered_powermode') {
                strategyLabel = 'Filtered: Same power mode';
                strategyColor = '#00ffff';
                strategyIcon = '🎯';
            } else if (prediction.queryStrategy === 'unfiltered') {
                strategyLabel = 'Unfiltered: All similar states';
                strategyColor = '#ffaa00';
                strategyIcon = '📊';
            }

            html += `<div style="margin: 8px 0; padding: 6px; background: #1a1a1a; border-left: 3px solid ${strategyColor}; border-radius: 3px;">
                <div style="font-size: 10px; color: ${strategyColor};">
                    ${strategyIcon} <strong>Learning Strategy:</strong> ${strategyLabel}
                </div>
            </div>`;
        }

        // Show metrics if available
        if (prediction.metrics) {
            const m = prediction.metrics;
            html += `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;">
                <div style="font-size: 10px; color: #666;">
                    Score Range: ${m.scoreRange ? m.scoreRange.toFixed(1) : 'N/A'} |
                    Gap: ${m.gapToSecond ? m.gapToSecond.toFixed(1) : 'N/A'}
                </div>
            </div>`;
        }

        detailsEl.innerHTML = html || '<small style="color: #666;">No details available</small>';
    }

    updateCurrentStrategyUI(state) {
        // Calculate current strategy metrics
        if (!window.vectorization) return;

        const enrichedMeta = window.vectorization.calculateEnrichedMetadata(state, 0);

        // Update strategy type
        const strategyEl = document.getElementById('strategyType');
        if (strategyEl) {
            strategyEl.textContent = enrichedMeta.detectedStrategy;

            // Color code by strategy
            const strategyColors = {
                'hunter': '#ff00ff',
                'defensive': '#00ff00',
                'aggressive': '#ff0000',
                'efficient': '#00ffff',
                'survivor': '#ffff00',
                'balanced': '#aaaaaa'
            };
            strategyEl.style.color = strategyColors[enrichedMeta.detectedStrategy] || '#aaa';
        }

        // Update risk level
        const riskEl = document.getElementById('riskLevel');
        const riskBar = document.getElementById('riskBar');
        if (riskEl && riskBar) {
            const riskPercent = Math.round(enrichedMeta.riskLevel * 100);
            riskEl.textContent = riskPercent + '%';
            riskBar.style.width = riskPercent + '%';
        }

        // Update efficiency
        const effEl = document.getElementById('efficiency');
        const effBar = document.getElementById('efficiencyBar');
        if (effEl && effBar) {
            const effPercent = Math.round(enrichedMeta.efficiency * 100);
            effEl.textContent = effPercent + '%';
            effBar.style.width = effPercent + '%';
        }

        // Update aggression
        const aggEl = document.getElementById('aggression');
        const aggBar = document.getElementById('aggressionBar');
        if (aggEl && aggBar) {
            const aggPercent = Math.round(enrichedMeta.aggressionScore * 100);
            aggEl.textContent = aggPercent + '%';
            aggBar.style.width = aggPercent + '%';
        }
    }

    evaluateAIPrediction() {
        // Only evaluate if we have a prediction and previous state
        if (!this.lastAIPrediction || !this.previousAIState || !window.gameStats) {
            return;
        }

        // Calculate what happened during this move
        const scoreDelta = this.aiState.score - this.previousAIState.score;
        const livesDelta = this.aiState.lives - this.previousAIState.lives;
        const pelletsDelta = this.previousAIState.pelletsLeft - this.aiState.pellets.length;

        // Calculate reward (same logic as vectorization.js but simplified)
        let reward = 0;
        let outcome = 'neutral';

        // Death is very bad
        if (livesDelta < 0) {
            reward = -100;
            outcome = 'death';
        }
        // Score increase is good
        else if (scoreDelta > 0) {
            reward = scoreDelta;
            if (scoreDelta >= 200) {
                outcome = 'ghost_eaten'; // Ate a ghost
            } else if (scoreDelta >= 50) {
                outcome = 'power'; // Got power pellet
            } else if (scoreDelta >= 10) {
                outcome = 'pellet'; // Got normal pellet
            }
        }
        // Collected pellet without dying
        else if (pelletsDelta > 0) {
            reward = 10;
            outcome = 'pellet';
        }
        // Check ghost proximity (danger)
        else {
            let minGhostDistance = Infinity;
            for (const ghost of this.aiState.ghosts) {
                const dist = Math.abs(ghost.x - this.aiState.player.x) +
                            Math.abs(ghost.y - this.aiState.player.y);
                minGhostDistance = Math.min(minGhostDistance, dist);
            }

            // Close call
            if (minGhostDistance <= 2 && !this.aiState.powerMode) {
                reward = -10;
                outcome = 'close_call';
            } else {
                // Survived, small positive reward
                reward = 1;
                outcome = 'neutral';
            }
        }

        // Determine if prediction was successful based on reward threshold
        const wasSuccessful = reward > -10; // Failed if reward < -10 (death or close call)

        // Track accuracy with actual confidence value
        window.gameStats.trackPredictionAccuracy(wasSuccessful, this.lastAIPrediction.confidence);

        // Track decision for UI display
        window.gameStats.currentGame.aiPredictions.push({
            timestamp: this.lastAIPrediction.timestamp,
            action: this.lastAIPrediction.action,
            outcome: outcome,
            reward: reward,
            confidence: this.lastAIPrediction.confidence,
            wasSuccessful: wasSuccessful
        });

        // Keep only last 20 predictions for UI
        if (window.gameStats.currentGame.aiPredictions.length > 20) {
            window.gameStats.currentGame.aiPredictions.shift();
        }
    }

    getSerializableState(state) {
        return {
            playerX: state.player.x,
            playerY: state.player.y,
            direction: state.player.direction,
            ghosts: state.ghosts.map(g => ({
                x: g.x,
                y: g.y,
                scared: state.powerMode || false,
                direction: g.direction
            })),
            pellets: state.pellets.map(p => ({ x: p.x, y: p.y })),
            powerPellets: state.powerPellets.map(p => ({ x: p.x, y: p.y })),
            walls: state.walls.map(w => ({ x: w.x, y: w.y })),
            pelletsLeft: state.pellets.length,
            totalPellets: state.totalPellets,
            score: state.score,
            lives: state.lives,
            powerMode: state.powerMode,
            powerModeTimer: state.powerModeTimer
        };
    }

    toggleAI() {
        this.aiEnabled = !this.aiEnabled;
    }
}

// Initialize game when page loads
let game;
window.addEventListener('DOMContentLoaded', () => {
    game = new PacManGame();
});
