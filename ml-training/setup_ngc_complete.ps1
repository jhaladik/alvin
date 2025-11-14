# Complete NGC CLI Setup with Workspace Creation (PowerShell)
# This script will:
# 1. Add NGC CLI to PATH (permanently)
# 2. Configure NGC with API key
# 3. Create a workspace for storing data and models
# 4. Verify everything is working

param(
    [string]$NgcPath = "C:\ngc",
    [string]$ApiKey = "",
    [string]$WorkspaceName = "alvin-pacman"
)

$ErrorActionPreference = "Continue"

# Helper functions
function Print-Banner([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Blue
    Write-Host " $Text" -ForegroundColor Blue
    Write-Host ("=" * 70) -ForegroundColor Blue
    Write-Host ""
}

function Print-Step([string]$Text) {
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Print-Error([string]$Text) {
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Print-Info([string]$Text) {
    Write-Host "ℹ $Text" -ForegroundColor Yellow
}

# Main script
Print-Banner "NGC CLI Complete Setup"

Write-Host "This script will set up NGC CLI with:"
Write-Host "  • NGC CLI in PATH (permanent)"
Write-Host "  • API key configuration"
Write-Host "  • Workspace creation for ML training"
Write-Host ""

# Step 1: Check if NGC CLI exists
Print-Info "Step 1/5: Checking NGC CLI installation..."
Write-Host ""

if (Test-Path "$NgcPath\ngc.exe") {
    Print-Step "Found NGC CLI at: $NgcPath"
} else {
    Print-Error "NGC CLI not found at: $NgcPath"
    Write-Host ""
    Write-Host "Please download NGC CLI first:"
    Write-Host "  1. Go to: https://ngc.nvidia.com/setup/installers/cli"
    Write-Host "  2. Download: ngccli_windows.zip"
    Write-Host "  3. Extract to: $NgcPath (or another location)"
    Write-Host ""
    Write-Host "Then run this script again with the correct path:"
    Write-Host "  .\setup_ngc_complete.ps1 -NgcPath 'C:\path\to\ngc' -ApiKey 'nvapi-...'"
    Write-Host ""
    exit 1
}

# Step 2: Add to PATH permanently
Print-Info "Step 2/5: Adding NGC CLI to PATH..."
Write-Host ""

# Add to current session first
$env:PATH += ";$NgcPath"

# Check if already in permanent PATH
$currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentUserPath -like "*$NgcPath*") {
    Print-Step "NGC CLI already in PATH"
} else {
    try {
        [Environment]::SetEnvironmentVariable("PATH", "$currentUserPath;$NgcPath", "User")
        Print-Step "Added NGC CLI to PATH permanently"
        Print-Info "You may need to restart your terminal for PATH changes to take effect"
    } catch {
        Print-Error "Failed to update PATH (may need admin privileges)"
        Print-Info "You can add manually: System Properties > Environment Variables"
        Write-Host ""
        exit 1
    }
}

# Verify NGC CLI works
Write-Host ""
try {
    $version = & "$NgcPath\ngc.exe" --version 2>&1 | Select-Object -First 1
    Print-Step "NGC CLI version: $version"
} catch {
    Print-Error "Failed to run NGC CLI"
    exit 1
}

# Step 3: Configure API key
Print-Info "Step 3/5: Configuring NGC authentication..."
Write-Host ""

if ($ApiKey) {
    # Use provided API key
    $configDir = "$env:USERPROFILE\.ngc"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null

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
    Print-Step "API key configured"
} else {
    # Interactive configuration
    Print-Info "No API key provided. Starting interactive configuration..."
    Write-Host ""
    Write-Host "To get your API key:"
    Write-Host "  1. Visit: https://ngc.nvidia.com/setup/api-key"
    Write-Host "  2. Generate a new API key"
    Write-Host "  3. Copy the key (nvapi-...)"
    Write-Host ""

    & "$NgcPath\ngc.exe" config set

    if ($LASTEXITCODE -ne 0) {
        Print-Error "Configuration failed"
        exit 1
    }
}

# Step 4: Verify authentication
Print-Info "Step 4/5: Verifying authentication..."
Write-Host ""

try {
    $whoami = & "$NgcPath\ngc.exe" whoami 2>&1
    if ($LASTEXITCODE -eq 0) {
        # Extract user info
        $name = ($whoami | Select-String "Name:" | ForEach-Object { ($_ -replace "Name:\s*", "").Trim() })
        $email = ($whoami | Select-String "Email:" | ForEach-Object { ($_ -replace "Email:\s*", "").Trim() })

        if ($name) {
            Print-Step "Authenticated as: $name ($email)"
        } else {
            Print-Step "Authentication successful"
        }
    } else {
        throw "Authentication failed"
    }
} catch {
    Print-Error "Authentication failed"
    Write-Host ""
    Write-Host "Please check your API key:"
    Write-Host "  1. Get key: https://ngc.nvidia.com/setup/api-key"
    Write-Host "  2. Run: ngc config set"
    Write-Host ""
    exit 1
}

# Step 5: Create workspace
Print-Info "Step 5/5: Setting up workspace..."
Write-Host ""

# Check existing workspaces
try {
    $workspaces = & "$NgcPath\ngc.exe" workspace list 2>&1

    if ($workspaces -like "*$WorkspaceName*") {
        Print-Step "Workspace '$WorkspaceName' already exists"

        # Get workspace details
        $wsInfo = & "$NgcPath\ngc.exe" workspace info "$WorkspaceName" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "Workspace details:"
            Write-Host $wsInfo
        }
    } else {
        Print-Info "Creating workspace: $WorkspaceName"

        # Create workspace
        $createResult = & "$NgcPath\ngc.exe" workspace create --name "$WorkspaceName" 2>&1

        if ($LASTEXITCODE -eq 0) {
            Print-Step "Workspace '$WorkspaceName' created successfully"
        } else {
            # Workspace creation might fail if not available in the plan
            $errorMsg = $createResult | Out-String

            if ($errorMsg -like "*not available*" -or $errorMsg -like "*permission*") {
                Print-Info "Workspace creation not available (may require team/org setup)"
                Print-Info "You can create workspaces manually at: https://ngc.nvidia.com/workspace"
                Write-Host ""
                Write-Host "For now, we'll continue without workspace creation."
                Write-Host "You can still run batch jobs without workspaces."
            } else {
                throw $errorMsg
            }
        }
    }
} catch {
    Print-Info "Workspace feature may not be available on your account"
    Print-Info "Don't worry - batch jobs can still run without workspaces"
    Write-Host ""
}

# Save configuration for scripts
$config = @{
    NgcPath = $NgcPath
    WorkspaceName = $WorkspaceName
    SetupDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}
$config | ConvertTo-Json | Out-File -FilePath ".ngc_config.json" -Encoding utf8
Print-Step "Configuration saved to .ngc_config.json"

# All done!
Write-Host ""
Print-Banner "NGC CLI Setup Complete!"

Write-Host "Configuration:"
Write-Host "  ✓ NGC CLI in PATH: $NgcPath"
Write-Host "  ✓ Authentication configured"
Write-Host "  ✓ Workspace: $WorkspaceName"
Write-Host ""

# Test basic commands
Print-Info "Testing NGC CLI commands..."
Write-Host ""

Write-Host "Available batch jobs:"
try {
    $jobs = & "$NgcPath\ngc.exe" batch list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $jobs
    } else {
        Print-Info "(No batch jobs yet)"
    }
} catch {
    Print-Info "(Error listing jobs - may need org/team setup)"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run training: .\run_training.ps1"
Write-Host "  2. Or test manually: ngc batch list"
Write-Host "  3. View NGC dashboard: https://ngc.nvidia.com"
Write-Host ""

Write-Host "Quick reference commands:"
Write-Host "  ngc batch list              # List batch jobs"
Write-Host "  ngc batch info <job-id>     # Get job details"
Write-Host "  ngc workspace list          # List workspaces"
Write-Host "  ngc registry image list     # List available images"
Write-Host ""
