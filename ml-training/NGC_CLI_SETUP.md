# NGC CLI Setup Guide

Professional automation using NVIDIA NGC CLI for batch job submission.

## Benefits of NGC CLI

✅ **Faster**: Submit jobs from command line
✅ **Automated**: No manual clicking
✅ **Reproducible**: Scripts you can version control
✅ **Professional**: How it's done in production
✅ **Cheaper**: Batch jobs cost less than interactive workspaces

---

## Step 1: Install NGC CLI (Windows)

### Download NGC CLI

1. Go to: https://ngc.nvidia.com/setup/installers/cli
2. Download **Windows** version: `ngccli_windows.zip`
3. Extract to: `C:\ngc\` (or any folder)

### Add to PATH

**Option A: PowerShell (Recommended)**
```powershell
# Add to PATH temporarily (current session)
$env:PATH += ";C:\ngc"

# Verify installation
ngc --version

# Add to PATH permanently
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\ngc", "User")
```

**Option B: Manual**
1. Search Windows: "Environment Variables"
2. Edit "Path" variable
3. Add: `C:\ngc`
4. Click OK
5. Restart terminal

### Verify Installation

```bash
ngc --version
```

Expected output:
```
NGC CLI 3.XX.X
```

---

## Step 2: Configure NGC CLI

### Get API Key

1. Go to: https://ngc.nvidia.com/setup/api-key
2. Click "Generate API Key"
3. Copy the key (starts with `nvapi-...`)

### Configure CLI

```bash
ngc config set
```

You'll be prompted:
```
Enter API key [****]: nvapi-YOUR-KEY-HERE
Enter CLI output format type [ascii]: ascii
Enter org [no-org]: <press Enter>
Enter team [no-team]: <press Enter>
Enter ace [no-ace]: <press Enter>
```

### Verify Configuration

```bash
ngc whoami
```

Expected output:
```
Name: Your Name
Email: your.email@example.com
```

✅ **Checkpoint**: CLI is configured!

---

## Step 3: Upload Training Code

NGC batch jobs need code in NGC registry or mounted storage.

### Option A: Package as Docker Image (Best)

Create `Dockerfile`:
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Copy training code
WORKDIR /workspace
COPY ml-training/ /workspace/

# Install dependencies
RUN pip install -r requirements.txt

# Set entrypoint
ENTRYPOINT ["python", "quickstart.py"]
```

Build and push:
```bash
# Build image
docker build -t alvin-pacman-training .

# Tag for NGC
docker tag alvin-pacman-training nvcr.io/<your-org>/alvin-pacman-training:latest

# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-api-key>

# Push to NGC
docker push nvcr.io/<your-org>/alvin-pacman-training:latest
```

### Option B: Use Git Clone (Simpler)

The batch job will clone from GitHub directly:
```bash
git clone https://github.com/jhaladik/alvin.git
cd alvin/ml-training
pip install -r requirements.txt
python quickstart.py
```

We'll use this approach! ✓

---

## Step 4: Submit Batch Job

### Create Job Configuration

Save as `ngc-job.yaml`:
```yaml
name: alvin-pacman-training
image: nvcr.io/nvidia/pytorch:23.10-py3
instance: dgxa100.80g.1.norm
commandline: >
  git clone https://github.com/jhaladik/alvin.git &&
  cd alvin/ml-training &&
  pip install -r requirements.txt &&
  python quickstart.py &&
  cp -r export /results/ &&
  cp -r checkpoints /results/
result: /results
workspace: []
ports:
  - containerPort: 6006
    protocol: TCP
    public: false
env:
  - name: NVIDIA_VISIBLE_DEVICES
    value: all
```

### Submit Job

```bash
ngc batch run --config ngc-job.yaml
```

Or directly (no config file):
```bash
ngc batch run \
  --name "alvin-pacman-training-$(date +%Y%m%d-%H%M%S)" \
  --instance dgxa100.80g.1.norm \
  --image "nvcr.io/nvidia/pytorch:23.10-py3" \
  --result /results \
  --commandline "
    git clone https://github.com/jhaladik/alvin.git &&
    cd alvin/ml-training &&
    pip install -r requirements.txt &&
    python quickstart.py &&
    cp -r export /results/ &&
    cp -r checkpoints /results/
  "
```

Expected output:
```
+--------+--------------------------------------+
| Job ID | 1234567890abcdef                     |
| Status | QUEUED                               |
+--------+--------------------------------------+
```

---

## Step 5: Monitor Job

### Check Status

```bash
ngc batch list
```

Output:
```
+-------------------+----------+----------+
| Job Name          | Status   | Progress |
+-------------------+----------+----------+
| alvin-pacman-...  | RUNNING  | 45%      |
+-------------------+----------+----------+
```

### View Logs (Real-time)

```bash
ngc batch logs <job-id> --follow
```

Or:
```bash
ngc batch logs 1234567890abcdef -f
```

You'll see:
```
⏳ Step 1: GPU Verification...
✓ GPU detected: NVIDIA A100-SXM4-40GB

⏳ Step 2: Downloading training data...
✓ Downloaded 946 samples

⏳ Step 3: Extracting features...
...
```

---

## Step 6: Download Results

### List Results

```bash
ngc result list <job-id>
```

### Download Model

```bash
ngc result download <job-id>
```

This downloads to: `./results/<job-id>/`

Model will be at:
```
results/<job-id>/export/pacman_model.onnx
```

---

## Step 7: Cleanup

### Kill Running Job

```bash
ngc batch kill <job-id>
```

### Delete Job

```bash
ngc batch delete <job-id>
```

---

## Automated Script

I'll create `run_training.sh` for full automation!

---

## Instance Types

### GPU Instances

| Instance | GPU | RAM | Cost | Recommended |
|----------|-----|-----|------|-------------|
| `dgxa100.80g.1.norm` | A100 (80GB) | 244GB | $$$ | Best performance |
| `dgxa100.40g.1.norm` | A100 (40GB) | 244GB | $$ | **Recommended** |
| `dgxv100.32g.1.norm` | V100 (32GB) | 244GB | $ | Budget option |
| `dgxt4.16g.1.norm` | T4 (16GB) | 61GB | $ | Slowest |

### Cost Saving Tips

1. Use batch jobs (cheaper than workspaces)
2. Use smaller batch size if using T4
3. Set `--total-runtime` limit
4. Delete jobs after downloading results

---

## Troubleshooting

### "ngc: command not found"

```bash
# Windows PowerShell
$env:PATH += ";C:\ngc"
ngc --version
```

### "Authentication failed"

```bash
# Reconfigure with new API key
ngc config set
```

### "Job failed"

```bash
# View logs to see error
ngc batch logs <job-id>
```

### "Not enough quota"

- Check your NGC credits
- Try smaller instance (T4 instead of A100)
- Wait for quota refresh

---

## Advanced: Custom Docker Image

For faster startup, create custom image with dependencies pre-installed:

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install dependencies
RUN pip install numpy requests onnx onnxruntime tqdm

# Copy code
WORKDIR /workspace
COPY . .

# Entrypoint
CMD ["python", "quickstart.py"]
```

Build and use:
```bash
docker build -t nvcr.io/<org>/alvin:latest .
docker push nvcr.io/<org>/alvin:latest

ngc batch run \
  --image "nvcr.io/<org>/alvin:latest" \
  ...
```

---

## Next Steps

After installing NGC CLI:
1. Run `run_training.sh` (I'll create this)
2. Wait 15 minutes
3. Download results automatically
4. Model ready to deploy!

---

**Ready to install? Follow Step 1 above!**
