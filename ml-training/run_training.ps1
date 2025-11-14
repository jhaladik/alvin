# Automated NGC Training Script (PowerShell)
# Submits batch job, monitors progress, downloads results

param(
    [string]$Instance = "dgxa100.40g.1.norm",  # A100 40GB (change to dgxt4.16g.1.norm for budget)
    [string]$Image = "nvcr.io/nvidia/pytorch:23.10-py3"
)

$ErrorActionPreference = "Stop"

# Configuration
$JOB_NAME = "alvin-pacman-training-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$GITHUB_REPO = "https://github.com/jhaladik/alvin.git"

function Print-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Blue
    Write-Host " $Text" -ForegroundColor Blue
    Write-Host ("=" * 70) -ForegroundColor Blue
    Write-Host ""
}

function Print-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Print-Error {
    param([string]$Text)
    Write-Host ""
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Print-Info {
    param([string]$Text)
    Write-Host "ℹ $Text" -ForegroundColor Yellow
}

# Check NGC CLI is installed
function Check-NgcCli {
    Print-Banner "Checking NGC CLI"

    try {
        $version = ngc --version 2>&1 | Select-Object -First 1
        Print-Step "NGC CLI installed: $version"
        return $true
    } catch {
        Print-Error "NGC CLI not found!"
        Write-Host ""
        Write-Host "Install NGC CLI:"
        Write-Host "  1. Download from: https://ngc.nvidia.com/setup/installers/cli"
        Write-Host "  2. Extract to C:\ngc\"
        Write-Host "  3. Add to PATH: `$env:PATH += ';C:\ngc'"
        Write-Host "  4. Run: ngc config set"
        Write-Host ""
        return $false
    }
}

# Check NGC CLI is configured
function Check-NgcConfig {
    try {
        $result = ngc batch list 2>&1
        if ($LASTEXITCODE -ne 0) {
            $errorMsg = $result | Out-String
            if ($errorMsg -like "*authentication*" -or $errorMsg -like "*apikey*" -or $errorMsg -like "*unauthorized*") {
                throw "Authentication failed"
            }
        }
        Print-Step "NGC CLI authenticated"
        return $true
    } catch {
        Print-Error "NGC CLI not configured!"
        Write-Host ""
        Write-Host "Configure NGC CLI:"
        Write-Host "  1. Get API key: https://ngc.nvidia.com/setup/api-key"
        Write-Host "  2. Run: ngc config set"
        Write-Host ""
        return $false
    }
}

# Submit batch job
function Submit-Job {
    Print-Banner "Submitting Training Job"

    Print-Info "Job name: $JOB_NAME"
    Print-Info "Instance: $Instance"
    Print-Info "Image: $Image"
    Write-Host ""

    # Create command to run
    $command = @"
set -e &&
echo '⏳ Cloning repository...' &&
git clone $GITHUB_REPO &&
cd alvin/ml-training &&
echo '⏳ Installing dependencies...' &&
pip install -r requirements.txt -q &&
echo '⏳ Starting training...' &&
python quickstart.py &&
echo '⏳ Copying results...' &&
mkdir -p /results &&
cp -r export /results/ &&
cp -r checkpoints /results/ &&
echo '✓ Training complete!'
"@

    # Submit job
    try {
        $output = ngc batch run `
            --name "$JOB_NAME" `
            --instance "$Instance" `
            --image "$Image" `
            --result /results `
            --total-runtime 3600s `
            --commandline "$command" 2>&1

        # Extract job ID
        $jobId = ($output | Select-String -Pattern "Job Id:\s+(\S+)" | ForEach-Object { $_.Matches.Groups[1].Value })

        if (-not $jobId) {
            # Try alternative pattern
            $jobId = ($output | Select-String -Pattern "\w{16}" | Select-Object -First 1 | ForEach-Object { $_.Matches.Value })
        }

        if (-not $jobId) {
            Print-Error "Failed to submit job!"
            Write-Host $output
            return $null
        }

        Print-Step "Job submitted successfully!"
        Write-Host "  Job ID: $jobId"
        Write-Host "  View at: https://ngc.nvidia.com/jobs/$jobId"
        Write-Host ""

        # Save job ID for later
        $jobId | Out-File -FilePath ".last_job_id" -Encoding utf8
        return $jobId
    } catch {
        Print-Error "Failed to submit job!"
        Write-Host $_.Exception.Message
        return $null
    }
}

# Monitor job progress
function Monitor-Job {
    param([string]$JobId)

    Print-Banner "Monitoring Job Progress"

    Print-Info "Watching job: $JobId"
    Print-Info "Press Ctrl+C to stop monitoring (job will continue running)"
    Write-Host ""

    # Follow logs
    try {
        ngc batch logs "$JobId" --follow
    } catch {
        # User cancelled or error - that's ok
    }

    Write-Host ""
}

# Wait for job completion
function Wait-ForCompletion {
    param([string]$JobId)

    Print-Banner "Waiting for Job Completion"

    while ($true) {
        try {
            $info = ngc batch info "$JobId" 2>&1
            $status = ($info | Select-String "Status:" | ForEach-Object { ($_ -replace "Status:\s*", "").Trim() })

            switch ($status) {
                "FINISHED_SUCCESS" {
                    Print-Step "Job completed successfully!"
                    return $true
                }
                { $_ -in @("FINISHED_FAILURE", "FAILED", "FAILED_RUNONCE") } {
                    Print-Error "Job failed!"
                    Write-Host ""
                    Write-Host "View logs:"
                    Write-Host "  ngc batch logs $JobId"
                    return $false
                }
                { $_ -in @("RUNNING", "QUEUED", "STARTING") } {
                    Write-Host -NoNewline "`rStatus: $status..." -ForegroundColor Yellow
                    Start-Sleep -Seconds 10
                }
                default {
                    Print-Error "Unknown status: $status"
                    return $false
                }
            }
        } catch {
            Print-Error "Failed to check status"
            return $false
        }
    }
}

# Download results
function Download-Results {
    param([string]$JobId)

    Print-Banner "Downloading Results"

    Print-Info "Downloading model and checkpoints..."

    try {
        # Download results
        ngc result download "$JobId" --dest ./results/

        if (Test-Path "./results/$JobId/export/pacman_model.onnx") {
            # Copy to convenient location
            New-Item -ItemType Directory -Path ./export -Force | Out-Null
            Copy-Item "./results/$JobId/export/pacman_model.onnx" ./export/

            $size = (Get-Item ./export/pacman_model.onnx).Length / 1MB
            Print-Step "Model downloaded: export/pacman_model.onnx ($($size.ToString('F2')) MB)"
        } else {
            Print-Error "Model not found in results!"
            return $false
        }

        if (Test-Path "./results/$JobId/checkpoints") {
            New-Item -ItemType Directory -Path ./checkpoints -Force | Out-Null
            Copy-Item -Path "./results/$JobId/checkpoints/*" -Destination ./checkpoints/ -Recurse
            Print-Step "Checkpoints downloaded"
        }

        Write-Host ""
        Print-Step "All results downloaded!"
        Write-Host "  Model: export/pacman_model.onnx"
        Write-Host "  Checkpoints: checkpoints/"
        Write-Host ""
        return $true
    } catch {
        Print-Error "Download failed!"
        Write-Host $_.Exception.Message
        return $false
    }
}

# Cleanup
function Cleanup-Job {
    param([string]$JobId)

    Print-Banner "Cleanup"

    if (-not (Test-Path ".last_job_id")) {
        Print-Info "No job to clean up"
        return
    }

    $response = Read-Host "Do you want to delete the job from NGC? (y/N)"

    if ($response -eq 'y' -or $response -eq 'Y') {
        ngc batch delete "$JobId" --confirm
        Print-Step "Job deleted from NGC"
        Remove-Item .last_job_id
    } else {
        Print-Info "Job kept on NGC (you can delete it later)"
        Print-Info "To delete: ngc batch delete $JobId"
    }
}

# Main execution
function Main {
    Print-Banner "Alvin Pac-Man AI - NGC Automated Training"

    Write-Host "This script will:"
    Write-Host "  1. Submit training job to NGC"
    Write-Host "  2. Monitor progress (15-20 min)"
    Write-Host "  3. Download trained model"
    Write-Host "  4. Cleanup"
    Write-Host ""
    Write-Host "Instance: $Instance"
    Write-Host "Estimated cost: ~`$0.50-2.00 (or free with credits)"
    Write-Host ""

    $response = Read-Host "Continue? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Cancelled"
        return
    }

    # Run pipeline
    if (-not (Check-NgcCli)) { return }
    if (-not (Check-NgcConfig)) { return }

    $jobId = Submit-Job
    if (-not $jobId) { return }

    Write-Host ""
    Write-Host "Options:"
    Write-Host "  1. Monitor logs (streaming)"
    Write-Host "  2. Wait for completion (polling)"
    Write-Host "  3. Continue without monitoring (check back later)"
    Write-Host ""

    $choice = Read-Host "Choose (1/2/3)"

    switch ($choice) {
        "1" {
            Monitor-Job -JobId $jobId
            $success = Wait-ForCompletion -JobId $jobId
        }
        "2" {
            $success = Wait-ForCompletion -JobId $jobId
        }
        "3" {
            Print-Info "Job running in background"
            Print-Info "Check status: ngc batch info $jobId"
            Print-Info "View logs: ngc batch logs $jobId"
            Write-Host ""
            Write-Host "When done, run:"
            Write-Host "  .\download_results.ps1"
            return
        }
        default {
            Print-Error "Invalid choice"
            return
        }
    }

    if (-not $success) { return }

    # Download results
    if (-not (Download-Results -JobId $jobId)) { return }

    # Cleanup
    Cleanup-Job -JobId $jobId

    Print-Banner "✓ TRAINING COMPLETE!"

    Write-Host ""
    Write-Host "Your trained model is ready:"
    Write-Host "  export/pacman_model.onnx"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Integrate model into your app"
    Write-Host "  2. Replace heuristics with ML predictions"
    Write-Host "  3. Test and deploy!"
    Write-Host ""
}

# Run main function
try {
    Main
} catch {
    Write-Host ""
    Write-Host "Interrupted. Job will continue running on NGC." -ForegroundColor Yellow
}
