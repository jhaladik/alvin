# Quick script to add NGC CLI to PATH
# Run this if NGC is already installed but not in PATH

param(
    [string]$NgcPath = "C:\ngc"
)

Write-Host "Adding NGC CLI to PATH..." -ForegroundColor Yellow
Write-Host ""

# Check if NGC CLI exists
if (-not (Test-Path "$NgcPath\ngc.exe")) {
    Write-Host "NGC CLI not found at: $NgcPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please specify the correct path:"
    Write-Host "  .\add_ngc_to_path.ps1 -NgcPath 'C:\path\to\ngc'"
    Write-Host ""
    Write-Host "Or download NGC CLI from:"
    Write-Host "  https://ngc.nvidia.com/setup/installers/cli"
    exit 1
}

Write-Host "Found NGC CLI at: $NgcPath" -ForegroundColor Green

# Add to current session
$env:PATH += ";$NgcPath"
Write-Host "Added to current session PATH" -ForegroundColor Green

# Add permanently to user PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($currentPath -like "*$NgcPath*") {
    Write-Host "Already in permanent PATH" -ForegroundColor Green
} else {
    try {
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$NgcPath", "User")
        Write-Host "Added to permanent PATH" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: You need to restart your terminal for permanent changes to take effect!" -ForegroundColor Yellow
    } catch {
        Write-Host "Failed to update permanent PATH" -ForegroundColor Red
        Write-Host "You may need administrator privileges" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Alternative: Add manually via System Properties > Environment Variables"
        exit 1
    }
}

# Test NGC CLI
Write-Host ""
Write-Host "Testing NGC CLI..." -ForegroundColor Yellow
try {
    $version = ngc --version 2>&1 | Select-Object -First 1
    Write-Host "NGC CLI works! Version: $version" -ForegroundColor Green
} catch {
    Write-Host "NGC CLI found but failed to run" -ForegroundColor Red
    Write-Host "Try restarting your terminal" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Success! NGC CLI is now in your PATH" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Configure NGC: ngc config set"
Write-Host "  2. Or run full setup: .\setup_ngc_complete.ps1"
Write-Host ""
