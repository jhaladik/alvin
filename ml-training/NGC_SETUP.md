# NGC CLI Setup Guide

This guide will help you set up NVIDIA NGC CLI for training your Pac-Man AI model on NVIDIA GPUs.

## Quick Start

### Option 1: Full Automated Setup (Recommended)
```powershell
# Downloads, configures, and creates workspace
.\setup_ngc_complete.ps1 -NgcPath "C:\ngc" -ApiKey "nvapi-YOUR-KEY-HERE"
```

### Option 2: Just Add to PATH
If NGC is already installed but not in PATH:
```powershell
.\add_ngc_to_path.ps1 -NgcPath "C:\ngc"
```

### Option 3: Manual Setup
1. Download NGC CLI from https://ngc.nvidia.com/setup/installers/cli
2. Extract to `C:\ngc\` (or another location)
3. Run: `.\add_ngc_to_path.ps1 -NgcPath "C:\ngc"`
4. Get API key from: https://ngc.nvidia.com/setup/api-key
5. Configure: `ngc config set`

## What is an NGC Workspace?

NGC Workspaces are cloud storage locations for:
- **Datasets**: Training data and game states
- **Models**: Saved models and checkpoints
- **Results**: Training outputs and logs

### Do You Need a Workspace?

**For batch jobs (our use case):** No, workspaces are optional!
- Batch jobs have their own result storage via `--result` flag
- Models are automatically saved to results
- You can download results directly

**When you might want a workspace:**
- Sharing datasets across multiple jobs
- Storing large datasets that don't change
- Collaborating with a team
- Long-term model storage

## Creating a Workspace

### Method 1: Using the Setup Script
```powershell
.\setup_ngc_complete.ps1 -WorkspaceName "alvin-pacman"
```

### Method 2: Using NGC CLI
```powershell
# List existing workspaces
ngc workspace list

# Create new workspace
ngc workspace create --name "alvin-pacman"

# View workspace details
ngc workspace info alvin-pacman
```

### Method 3: Via Web UI
1. Go to https://ngc.nvidia.com/workspace
2. Click "Create Workspace"
3. Enter name: `alvin-pacman`
4. Configure storage size

## Common Issues

### "Workspace not available"
This error means:
- Workspaces require an organization/team setup
- Your account type may not support workspaces
- **Solution**: Don't worry! You can still run batch jobs without workspaces

### "NGC CLI not found"
```powershell
# Check if NGC is installed
where.exe ngc

# If not found, download from:
# https://ngc.nvidia.com/setup/installers/cli

# Then add to PATH:
.\add_ngc_to_path.ps1 -NgcPath "C:\path\to\ngc"
```

### "Authentication failed"
```powershell
# Get new API key
# https://ngc.nvidia.com/setup/api-key

# Reconfigure
ngc config set
```

## Verifying Setup

```powershell
# Check NGC CLI is installed
ngc --version

# Check authentication
ngc whoami

# List workspaces (if available)
ngc workspace list

# List recent batch jobs
ngc batch list
```

## Using Workspaces in Training (Optional)

If you have a workspace, you can mount it in batch jobs:

```powershell
ngc batch run \
  --name "my-training-job" \
  --instance "dgxa100.40g.1.norm" \
  --image "nvcr.io/nvidia/pytorch:23.10-py3" \
  --workspace "alvin-pacman:/workspace:RW" \  # Mount workspace
  --result /results \
  --commandline "python train.py"
```

In the job, your workspace will be available at `/workspace/`

## Next Steps

Once NGC is set up:

1. **Test NGC**: `.\test_ngc.ps1`
2. **Start Training**: `.\run_training.ps1`
3. **Monitor Progress**: Check NGC dashboard at https://ngc.nvidia.com

## Reference

### Important NGC Commands
```powershell
# Batch jobs
ngc batch list                    # List jobs
ngc batch info <job-id>          # Job details
ngc batch logs <job-id>          # View logs
ngc batch delete <job-id>        # Delete job
ngc result download <job-id>     # Download results

# Workspaces (if available)
ngc workspace list               # List workspaces
ngc workspace info <name>        # Workspace details
ngc workspace upload <name> <path>   # Upload files
ngc workspace download <name> <path> # Download files

# Registry
ngc registry image list          # Available Docker images
ngc registry model list          # Available models
```

### File Locations
- **NGC CLI config**: `%USERPROFILE%\.ngc\config`
- **Job results**: `./results/<job-id>/`
- **Local models**: `./export/pacman_model.onnx`
- **Checkpoints**: `./checkpoints/`

## Support

- NGC Documentation: https://docs.nvidia.com/ngc/
- NGC CLI Guide: https://ngc.nvidia.com/setup/cli
- Get API Key: https://ngc.nvidia.com/setup/api-key
- NGC Dashboard: https://ngc.nvidia.com
