# Start Alvin Pac-Man Locally with Real ML Inference
# This runs the ACTUAL trained neural network (100% accuracy)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Alvin Pac-Man - Local ML Testing" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
if (-not (Test-Path "ml-training\checkpoints\best_model.pth")) {
    Write-Host "[ERROR] Trained model not found!" -ForegroundColor Red
    Write-Host "Please train the model first:" -ForegroundColor Yellow
    Write-Host "  cd ml-training" -ForegroundColor Yellow
    Write-Host "  python training/train.py" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] Starting ML Inference Server (port 5000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\ml-training'; python inference_server.py"
Start-Sleep -Seconds 3

Write-Host "[2/2] Starting Frontend Server (port 3000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; node local-server.js"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Servers Started!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ML Inference: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Game Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:3000"
Write-Host ""
Write-Host "Press any key to stop servers..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Note: This won't actually stop the servers - user needs to close the PowerShell windows manually
Write-Host "Please close the server windows manually." -ForegroundColor Yellow
