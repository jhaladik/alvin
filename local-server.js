/**
 * Local Development Server
 * Serves the game locally for testing with ML inference server
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
};

const server = http.createServer((req, res) => {
    console.log(`${req.method} ${req.url}`);

    // Default to index.html
    let filePath = req.url === '/' ? '/frontend/index.html' : req.url;

    // Add frontend prefix if not present
    if (!filePath.startsWith('/frontend/')) {
        filePath = '/frontend' + filePath;
    }

    const fullPath = path.join(__dirname, filePath);
    const ext = path.extname(fullPath);
    const contentType = mimeTypes[ext] || 'text/plain';

    fs.readFile(fullPath, (err, data) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('404 Not Found');
            } else {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('500 Internal Server Error');
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        }
    });
});

server.listen(PORT, () => {
    console.log('='.repeat(70));
    console.log('Alvin Pac-Man - Local Development Server');
    console.log('='.repeat(70));
    console.log();
    console.log(`Frontend: http://localhost:${PORT}`);
    console.log('ML Server: http://localhost:5000 (start separately)');
    console.log();
    console.log('Instructions:');
    console.log('1. Start ML inference server: cd ml-training && python inference_server.py');
    console.log('2. Open browser: http://localhost:3000');
    console.log();
    console.log('Press Ctrl+C to stop');
    console.log('='.repeat(70));
});
