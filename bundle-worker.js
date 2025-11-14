/**
 * Build Script: Bundle frontend files into worker
 * Creates a dist/ directory with the bundled worker for deployment
 */

const fs = require('fs');
const path = require('path');

// Create dist directory if it doesn't exist
const distDir = path.join(__dirname, 'dist');
if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir);
}

console.log('📦 Building Alvin Pac-Man Worker...\n');

// Read frontend files
console.log('📄 Reading frontend files...');
const html = fs.readFileSync(path.join(__dirname, 'frontend', 'index.html'), 'utf8');
const statisticsJs = fs.readFileSync(path.join(__dirname, 'frontend', 'statistics.js'), 'utf8');
const gameJs = fs.readFileSync(path.join(__dirname, 'frontend', 'game.js'), 'utf8');
const dqnAgentJs = fs.readFileSync(path.join(__dirname, 'frontend', 'dqn-agent.js'), 'utf8');
const vectorizationJs = fs.readFileSync(path.join(__dirname, 'frontend', 'vectorization.js'), 'utf8');

console.log('  ✓ index.html');
console.log('  ✓ statistics.js');
console.log('  ✓ game.js');
console.log('  ✓ dqn-agent.js');
console.log('  ✓ vectorization.js');

// Read worker source file
console.log('\n🔧 Reading worker source...');
const workerCode = fs.readFileSync(path.join(__dirname, 'worker', 'index.js'), 'utf8');
console.log('  ✓ worker/index.js');

// Create bundled worker with static assets appended
const bundledWorker = workerCode + `
// ============================================================================
// STATIC ASSETS - Auto-generated during build
// These constants are injected by the build script (bundle-worker.js)
// Source files: frontend/*.{html,js}
// ============================================================================

const HTML = ${JSON.stringify(html)};
const STATISTICS_JS = ${JSON.stringify(statisticsJs)};
const GAME_JS = ${JSON.stringify(gameJs)};
const DQN_AGENT_JS = ${JSON.stringify(dqnAgentJs)};
const VECTORIZATION_JS = ${JSON.stringify(vectorizationJs)};
`;

// Write bundled worker to dist directory
const outputPath = path.join(distDir, 'index.js');
fs.writeFileSync(outputPath, bundledWorker, 'utf8');

console.log('\n✅ Build complete!');
console.log(`📦 Bundled worker: ${outputPath}`);
console.log(`📊 Bundle size: ${(bundledWorker.length / 1024).toFixed(2)} KB\n`);
