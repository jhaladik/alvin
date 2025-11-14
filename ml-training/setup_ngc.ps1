# NGC CLI Setup Helper (PowerShell)
# Run this script AFTER downloading NGC CLI from https://ngc.nvidia.com/setup/installers/cli

param(
    [string]$NgcPath = "C:\ngc",
    [string]$ApiKey = ""
)

Write-Host "NGC CLI Setup Helper" -ForegroundColor Blue
Write-Host "===================" -ForegroundColor Blue
Write-Host ""

# Check if NGC CLI exists in the specified path
if (Test-Path "$NgcPath\ngc.exe") {
    Write-Host "✓ Found NGC CLI at: $NgcPath" -ForegroundColor Green
} else {
    Write-Host "✗ NGC CLI not found at: $NgcPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download NGC CLI first:" -ForegroundColor Yellow
    Write-Host "  1. Go to: https://ngc.nvidia.com/setup/installers/cli"
    Write-Host "  2. Download: ngccli_windows.zip"
    Write-Host "  3. Extract to: $NgcPath"
    Write-Host ""
    Write-Host "Then run this script again:"
    Write-Host "  .\setup_ngc.ps1 -NgcPath 'C:\path\to\ngc' -ApiKey 'nvapi-...'"
    exit 1
}

# Add to PATH (current session)
Write-Host ""
Write-Host "Adding NGC CLI to PATH (current session)..." -ForegroundColor Yellow
$env:PATH += ";$NgcPath"

# Verify NGC CLI works
Write-Host "Testing NGC CLI..." -ForegroundColor Yellow
try {
    $version = & "$NgcPath\ngc.exe" --version 2>&1 | Select-Object -First 1
    Write-Host "✓ NGC CLI version: $version" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to run NGC CLI" -ForegroundColor Red
    exit 1
}

# Add to PATH permanently
Write-Host ""
$response = Read-Host "Add to PATH permanently? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$NgcPath*") {
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$NgcPath", "User")
        Write-Host "✓ Added to PATH permanently" -ForegroundColor Green
        Write-Host "  (Restart terminal for changes to take effect)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Already in PATH" -ForegroundColor Green
    }
}

# Configure with API key
Write-Host ""
if ($ApiKey) {
    Write-Host "Configuring NGC CLI with provided API key..." -ForegroundColor Yellow

    # Create config directory
    $configDir = "$env:USERPROFILE\.ngc"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null

    # Create config file
    $config = @"
; WARNING - This is a machine generated file.  Do not edit manually.
; WARNING - To update local config settings, please use the 'ngc config set' command
[CURRENT]
apikey = $ApiKey
format_type = ascii
org =
team =
ace =
"@

    $config | Out-File -FilePath "$configDir\config" -Encoding ASCII -Force
    Write-Host "✓ API key configured" -ForegroundColor Green
} else {
    Write-Host "No API key provided. Running interactive configuration..." -ForegroundColor Yellow
    Write-Host ""
    & "$NgcPath\ngc.exe" config set
}

# Verify authentication
Write-Host ""
Write-Host "Verifying authentication..." -ForegroundColor Yellow
try {
    $whoami = & "$NgcPath\ngc.exe" whoami 2>&1
    if ($LASTEXITCODE -eq 0) {
        $user = ($whoami | Select-String "Name:" | ForEach-Object { $_ -replace "Name:\s*", "" }).Trim()
        Write-Host "✓ Authenticated as: $user" -ForegroundColor Green
    } else {
        throw "Authentication failed"
    }
} catch {
    Write-Host "✗ Authentication failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Get API key from: https://ngc.nvidia.com/setup/api-key" -ForegroundColor Yellow
    Write-Host "Then run: ngc config set" -ForegroundColor Yellow
    exit 1
}

# All done!
Write-Host ""
Write-Host "===================" -ForegroundColor Blue
Write-Host "✓ NGC CLI Setup Complete!" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Blue
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start training: .\run_training.ps1"
Write-Host "  2. Or test NGC CLI: ngc batch list"
Write-Host ""
