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
const featureEngineeringJs = fs.readFileSync(path.join(__dirname, 'frontend', 'feature-engineering.js'), 'utf8');
const pathPlannerJs = fs.readFileSync(path.join(__dirname, 'frontend', 'path-planner.js'), 'utf8');
const gameJs = fs.readFileSync(path.join(__dirname, 'frontend', 'game.js'), 'utf8');
const dqnAgentJs = fs.readFileSync(path.join(__dirname, 'frontend', 'dqn-agent.js'), 'utf8');
const vectorizationJs = fs.readFileSync(path.join(__dirname, 'frontend', 'vectorization.js'), 'utf8');

console.log('  ✓ index.html');
console.log('  ✓ statistics.js');
console.log('  ✓ feature-engineering.js');
console.log('  ✓ path-planner.js');
console.log('  ✓ game.js');
console.log('  ✓ dqn-agent.js');
console.log('  ✓ vectorization.js');

// Read worker source files
console.log('\n🔧 Reading worker source...');
const workerCode = fs.readFileSync(path.join(__dirname, 'worker', 'index.js'), 'utf8');
const mlPredictorCode = fs.readFileSync(path.join(__dirname, 'worker', 'ml-predictor.js'), 'utf8');
console.log('  ✓ worker/index.js');
console.log('  ✓ worker/ml-predictor.js');

// Create bundled worker with ML predictor and static assets
// Remove the import statement from worker code (we'll inline it)
const workerCodeClean = workerCode.replace(/import\s+\{[^}]+\}\s+from\s+['"]\.\/ml-predictor\.js['"];?\s*/g, '');

// Remove import/export from ML predictor (inline it)
const mlPredictorInline = mlPredictorCode
  .replace(/import\s+[^;]+;?\s*/g, '') // Remove imports
  .replace(/export\s+/g, ''); // Remove exports

const bundledWorker = `
// ============================================================================
// ML PREDICTOR MODULE - Inlined
// ============================================================================
${mlPredictorInline}

// ============================================================================
// MAIN WORKER CODE
// ============================================================================
${workerCodeClean}

// ============================================================================
// STATIC ASSETS - Auto-generated during build
// These constants are injected by the build script (bundle-worker.js)
// Source files: frontend/*.{html,js}
// ============================================================================

const HTML = ${JSON.stringify(html)};
const STATISTICS_JS = ${JSON.stringify(statisticsJs)};
const FEATURE_ENGINEERING_JS = ${JSON.stringify(featureEngineeringJs)};
const PATH_PLANNER_JS = ${JSON.stringify(pathPlannerJs)};
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
