#!/bin/bash
#
# Automated NGC Training Script
# Submits batch job, monitors progress, downloads results
#

set -e

# Configuration
JOB_NAME="alvin-pacman-training-$(date +%Y%m%d-%H%M%S)"
INSTANCE="dgxa100.40g.1.norm"  # A100 40GB (change to dgxt4.16g.1.norm for budget)
IMAGE="nvcr.io/nvidia/pytorch:23.10-py3"
GITHUB_REPO="https://github.com/jhaladik/alvin.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}===================================================================${NC}"
}

print_step() {
    echo -e "\n${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "\n${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check NGC CLI is installed
check_ngc_cli() {
    print_banner "Checking NGC CLI"

    if ! command -v ngc &> /dev/null; then
        print_error "NGC CLI not found!"
        echo ""
        echo "Install NGC CLI:"
        echo "  1. Download from: https://ngc.nvidia.com/setup/installers/cli"
        echo "  2. Extract and add to PATH"
        echo "  3. Run: ngc config set"
        echo ""
        exit 1
    fi

    NGC_VERSION=$(ngc --version | head -1)
    print_step "NGC CLI installed: $NGC_VERSION"
}

# Check NGC CLI is configured
check_ngc_config() {
    if ! ngc whoami &> /dev/null; then
        print_error "NGC CLI not configured!"
        echo ""
        echo "Configure NGC CLI:"
        echo "  1. Get API key: https://ngc.nvidia.com/setup/api-key"
        echo "  2. Run: ngc config set"
        echo ""
        exit 1
    fi

    NGC_USER=$(ngc whoami | grep "Name:" | cut -d':' -f2 | xargs)
    print_step "Logged in as: $NGC_USER"
}

# Submit batch job
submit_job() {
    print_banner "Submitting Training Job"

    print_info "Job name: $JOB_NAME"
    print_info "Instance: $INSTANCE"
    print_info "Image: $IMAGE"
    echo ""

    # Create command to run
    COMMAND="
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
    "

    # Submit job
    JOB_OUTPUT=$(ngc batch run \
        --name "$JOB_NAME" \
        --instance "$INSTANCE" \
        --image "$IMAGE" \
        --result /results \
        --total-runtime 3600s \
        --commandline "$COMMAND" \
        2>&1)

    # Extract job ID
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job Id:  )\S+' || echo "$JOB_OUTPUT" | grep -oP '\w{16}')

    if [ -z "$JOB_ID" ]; then
        print_error "Failed to submit job!"
        echo "$JOB_OUTPUT"
        exit 1
    fi

    print_step "Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  View at: https://ngc.nvidia.com/jobs/$JOB_ID"
    echo ""

    # Save job ID for later
    echo "$JOB_ID" > .last_job_id
}

# Monitor job progress
monitor_job() {
    print_banner "Monitoring Job Progress"

    JOB_ID=$(cat .last_job_id)
    print_info "Watching job: $JOB_ID"
    print_info "Press Ctrl+C to stop monitoring (job will continue running)"
    echo ""

    # Follow logs
    ngc batch logs "$JOB_ID" --follow || true

    echo ""
}

# Wait for job completion
wait_for_completion() {
    print_banner "Waiting for Job Completion"

    JOB_ID=$(cat .last_job_id)

    while true; do
        STATUS=$(ngc batch info "$JOB_ID" 2>/dev/null | grep "Status:" | awk '{print $2}' || echo "UNKNOWN")

        case "$STATUS" in
            FINISHED_SUCCESS)
                print_step "Job completed successfully!"
                return 0
                ;;
            FINISHED_FAILURE|FAILED|FAILED_RUNONCE)
                print_error "Job failed!"
                echo ""
                echo "View logs:"
                echo "  ngc batch logs $JOB_ID"
                return 1
                ;;
            RUNNING|QUEUED|STARTING)
                echo -ne "\r${YELLOW}Status: $STATUS...${NC}"
                sleep 10
                ;;
            *)
                print_error "Unknown status: $STATUS"
                return 1
                ;;
        esac
    done
}

# Download results
download_results() {
    print_banner "Downloading Results"

    JOB_ID=$(cat .last_job_id)

    print_info "Downloading model and checkpoints..."

    # Download results
    ngc result download "$JOB_ID" --dest ./results/

    if [ -f "./results/$JOB_ID/export/pacman_model.onnx" ]; then
        # Copy to convenient location
        mkdir -p ./export
        cp "./results/$JOB_ID/export/pacman_model.onnx" ./export/

        MODEL_SIZE=$(du -h ./export/pacman_model.onnx | cut -f1)
        print_step "Model downloaded: export/pacman_model.onnx ($MODEL_SIZE)"
    else
        print_error "Model not found in results!"
        return 1
    fi

    if [ -d "./results/$JOB_ID/checkpoints" ]; then
        mkdir -p ./checkpoints
        cp -r "./results/$JOB_ID/checkpoints/"* ./checkpoints/
        print_step "Checkpoints downloaded"
    fi

    echo ""
    print_step "All results downloaded!"
    echo "  Model: export/pacman_model.onnx"
    echo "  Checkpoints: checkpoints/"
    echo ""
}

# Cleanup
cleanup_job() {
    print_banner "Cleanup"

    if [ ! -f .last_job_id ]; then
        print_info "No job to clean up"
        return 0
    fi

    JOB_ID=$(cat .last_job_id)

    echo "Do you want to delete the job from NGC? (y/N)"
    read -r RESPONSE

    if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
        ngc batch delete "$JOB_ID" --confirm
        print_step "Job deleted from NGC"
        rm .last_job_id
    else
        print_info "Job kept on NGC (you can delete it later)"
        print_info "To delete: ngc batch delete $JOB_ID"
    fi
}

# Main execution
main() {
    print_banner "Alvin Pac-Man AI - NGC Automated Training"

    echo "This script will:"
    echo "  1. Submit training job to NGC"
    echo "  2. Monitor progress (15-20 min)"
    echo "  3. Download trained model"
    echo "  4. Cleanup"
    echo ""
    echo "Instance: $INSTANCE"
    echo "Estimated cost: ~\$0.50-2.00 (or free with credits)"
    echo ""

    read -p "Continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi

    # Run pipeline
    check_ngc_cli
    check_ngc_config
    submit_job

    echo ""
    echo "Options:"
    echo "  1. Monitor logs (streaming)"
    echo "  2. Wait for completion (polling)"
    echo "  3. Continue without monitoring (check back later)"
    echo ""
    read -p "Choose (1/2/3): " -n 1 -r CHOICE
    echo ""

    case "$CHOICE" in
        1)
            monitor_job
            wait_for_completion
            ;;
        2)
            wait_for_completion
            ;;
        3)
            print_info "Job running in background"
            print_info "Check status: ngc batch info $(cat .last_job_id)"
            print_info "View logs: ngc batch logs $(cat .last_job_id)"
            echo ""
            echo "When done, run:"
            echo "  ./download_results.sh"
            exit 0
            ;;
    esac

    # Download results
    download_results

    # Cleanup
    cleanup_job

    print_banner "✓ TRAINING COMPLETE!"

    echo ""
    echo "Your trained model is ready:"
    echo "  export/pacman_model.onnx"
    echo ""
    echo "Next steps:"
    echo "  1. Integrate model into your app"
    echo "  2. Replace heuristics with ML predictions"
    echo "  3. Test and deploy!"
    echo ""
}

# Handle Ctrl+C
trap 'echo -e "\n${YELLOW}Interrupted. Job will continue running on NGC.${NC}"; exit 130' INT

# Run main function
main "$@"
