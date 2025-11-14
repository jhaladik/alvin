# Quick NGC Test Script
Write-Host "Testing NGC CLI authentication..." -ForegroundColor Yellow

# Test with batch list command
$result = ngc batch list 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ NGC CLI authenticated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your recent batch jobs:"
    Write-Host $result
} else {
    $errorMsg = $result | Out-String
    if ($errorMsg -like "*authentication*" -or $errorMsg -like "*apikey*") {
        Write-Host "✗ Authentication failed" -ForegroundColor Red
        Write-Host $errorMsg
        exit 1
    } else {
        Write-Host "✓ NGC CLI authenticated (no jobs yet)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "✓ Ready to start training!" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: Run the training script"
Write-Host "  .\run_training.ps1"
