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

        // Initialize DQN agent and vectorization
        if (window.dqnAgent) {
            window.dqnAgent.reset();
        }
    }

    createInitialState() {
        return {
            player: {
                x: 10,
                y: 10,
                direction: 'RIGHT',
                mouthOpen: true
            },
            ghosts: [
                { x: 5, y: 5, color: '#ff0000', direction: 'RIGHT' },
                { x: 15, y: 5, color: '#ffb8ff', direction: 'LEFT' },
                { x: 5, y: 15, color: '#00ffff', direction: 'UP' },
                { x: 15, y: 15, color: '#ffb852', direction: 'DOWN' }
            ],
            pellets: this.generatePellets(),
            powerPellets: [
                { x: 2, y: 2 },
                { x: 17, y: 2 },
                { x: 2, y: 17 },
                { x: 17, y: 17 }
            ],
            walls: this.generateWalls(),
            score: 0,
            lives: 3,
            powerMode: false,
            powerModeTimer: 0
        };
    }

    generatePellets() {
        const pellets = [];
        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                if (Math.random() > 0.3) {
                    pellets.push({ x, y });
                }
            }
        }
        return pellets;
    }

    generateWalls() {
        const walls = [];
        // Create a simple maze pattern
        for (let i = 0; i < this.gridSize; i++) {
            if (i % 4 === 0 && i > 0 && i < this.gridSize - 1) {
                for (let j = 2; j < this.gridSize - 2; j += 2) {
                    walls.push({ x: i, y: j });
                }
            }
        }
        return walls;
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            const key = e.key;
            let newDirection = null;

            switch (key) {
                case 'ArrowUp':
                    newDirection = 'UP';
                    break;
                case 'ArrowDown':
                    newDirection = 'DOWN';
                    break;
                case 'ArrowLeft':
                    newDirection = 'LEFT';
                    break;
                case 'ArrowRight':
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

    async vectorizeMove(direction) {
        this.moveCount++;
        document.getElementById('vectorizedMoves').textContent = this.moveCount;

        // Send to vectorization system
        if (window.vectorization) {
            await window.vectorization.vectorizeGameState(
                this.getSerializableState(this.humanState),
                direction
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
        const deltaTime = now - this.lastMoveTime;

        // Update at 10 FPS for movement
        if (deltaTime > 100) {
            this.lastMoveTime = now;

            // Update human player
            this.updatePlayer(this.humanState);
            this.updateGhosts(this.humanState);
            this.checkCollisions(this.humanState);

            // Update AI avatar
            if (this.aiEnabled) {
                this.updateAI();
                this.updatePlayer(this.aiState);
                this.updateGhosts(this.aiState);
                this.checkCollisions(this.aiState);
            }

            // Toggle mouth animation
            this.humanState.player.mouthOpen = !this.humanState.player.mouthOpen;
            this.aiState.player.mouthOpen = !this.aiState.player.mouthOpen;
        }

        // Update power mode timer
        if (this.humanState.powerMode) {
            this.humanState.powerModeTimer--;
            if (this.humanState.powerModeTimer <= 0) {
                this.humanState.powerMode = false;
            }
        }

        if (this.aiState.powerMode) {
            this.aiState.powerModeTimer--;
            if (this.aiState.powerModeTimer <= 0) {
                this.aiState.powerMode = false;
            }
        }
    }

    async updateAI() {
        // Get AI prediction from DQN agent
        if (window.dqnAgent) {
            const prediction = await window.dqnAgent.predictNextMove(
                this.getSerializableState(this.aiState)
            );

            if (prediction && prediction.action) {
                this.aiState.player.direction = prediction.action;

                // Update UI with AI's next move
                document.getElementById('aiNextMove').textContent = prediction.action;
                document.getElementById('aiConfidence').textContent =
                    Math.round(prediction.confidence * 100) + '%';
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
                state.powerModeTimer = 50;
                return false;
            }
            return true;
        });

        // Check ghost collisions
        state.ghosts.forEach(ghost => {
            if (ghost.x === player.x && ghost.y === player.y) {
                if (state.powerMode) {
                    state.score += 200;
                    // Respawn ghost
                    ghost.x = Math.floor(Math.random() * this.gridSize);
                    ghost.y = Math.floor(Math.random() * this.gridSize);
                } else {
                    state.lives--;
                    if (state.lives <= 0) {
                        // Game over
                        console.log('Game Over!');
                    }
                }
            }
        });

        // Update score displays
        if (state === this.humanState) {
            document.getElementById('humanScore').textContent = state.score;
        } else {
            document.getElementById('aiScore').textContent = state.score;
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

        // Draw power pellets
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

        // Draw ghosts
        state.ghosts.forEach(ghost => {
            ctx.fillStyle = state.powerMode ? '#0000ff' : ghost.color;
            this.drawGhost(
                ctx,
                ghost.x * this.cellSize,
                ghost.y * this.cellSize,
                this.cellSize
            );
        });

        // Draw Pac-Man
        ctx.fillStyle = playerColor;
        this.drawPacMan(
            ctx,
            state.player.x * this.cellSize,
            state.player.y * this.cellSize,
            this.cellSize,
            state.player.direction,
            state.player.mouthOpen
        );
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

    getSerializableState(state) {
        return {
            playerX: state.player.x,
            playerY: state.player.y,
            direction: state.player.direction,
            ghosts: state.ghosts.map(g => ({ x: g.x, y: g.y })),
            pelletsLeft: state.pellets.length,
            score: state.score
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
