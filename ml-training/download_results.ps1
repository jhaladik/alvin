# Download Results Helper Script (PowerShell)
# Use this if you chose "Continue without monitoring" during training

param(
    [string]$JobId = ""
)

$ErrorActionPreference = "Stop"

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

# Get job ID
if (-not $JobId) {
    if (Test-Path ".last_job_id") {
        $JobId = Get-Content ".last_job_id" -Raw
        $JobId = $JobId.Trim()
    } else {
        Print-Error "No job ID found!"
        Write-Host ""
        Write-Host "Please provide job ID as argument:"
        Write-Host "  .\download_results.ps1 -JobId <job-id>"
        Write-Host ""
        Write-Host "Or check your jobs:"
        Write-Host "  ngc batch list"
        exit 1
    }
}

Print-Banner "Download NGC Training Results"

Print-Info "Job ID: $JobId"
Write-Host ""

# Check job status
Print-Info "Checking job status..."

try {
    $info = ngc batch info "$JobId" 2>&1
    $status = ($info | Select-String "Status:" | ForEach-Object { ($_ -replace "Status:\s*", "").Trim() })

    switch ($status) {
        "FINISHED_SUCCESS" {
            Print-Step "Job completed successfully!"
        }
        { $_ -in @("RUNNING", "QUEUED", "STARTING") } {
            Print-Error "Job is still running: $status"
            Write-Host ""
            Write-Host "Wait for completion or monitor:"
            Write-Host "  ngc batch logs $JobId --follow"
            exit 1
        }
        { $_ -in @("FINISHED_FAILURE", "FAILED", "FAILED_RUNONCE") } {
            Print-Error "Job failed: $status"
            Write-Host ""
            Write-Host "View logs to diagnose:"
            Write-Host "  ngc batch logs $JobId"
            exit 1
        }
        default {
            Print-Error "Unknown status: $status"
            exit 1
        }
    }
} catch {
    Print-Error "Failed to check job status"
    Write-Host $_.Exception.Message
    exit 1
}

# Download results
Print-Banner "Downloading Results"

Print-Info "Downloading model and checkpoints..."

try {
    ngc result download "$JobId" --dest ./results/

    if (Test-Path "./results/$JobId/export/pacman_model.onnx") {
        # Copy to convenient location
        New-Item -ItemType Directory -Path ./export -Force | Out-Null
        Copy-Item "./results/$JobId/export/pacman_model.onnx" ./export/

        $size = (Get-Item ./export/pacman_model.onnx).Length / 1MB
        Print-Step "Model downloaded: export/pacman_model.onnx ($($size.ToString('F2')) MB)"
    } else {
        Print-Error "Model not found in results!"
        exit 1
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
} catch {
    Print-Error "Download failed!"
    Write-Host $_.Exception.Message
    exit 1
}

# Ask about cleanup
Print-Banner "Cleanup"

$response = Read-Host "Do you want to delete the job from NGC? (y/N)"

if ($response -eq 'y' -or $response -eq 'Y') {
    ngc batch delete "$JobId" --confirm
    Print-Step "Job deleted from NGC"
    if (Test-Path ".last_job_id") {
        Remove-Item ".last_job_id"
    }
} else {
    Print-Info "Job kept on NGC (you can delete it later)"
    Print-Info "To delete: ngc batch delete $JobId"
}

Print-Banner "✓ DOWNLOAD COMPLETE!"

Write-Host ""
Write-Host "Your trained model is ready:"
Write-Host "  export/pacman_model.onnx"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Integrate model into your app"
Write-Host "  2. Replace heuristics with ML predictions"
Write-Host "  3. Test and deploy!"
Write-Host ""
