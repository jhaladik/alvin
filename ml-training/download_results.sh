#!/bin/bash
#
# Download Results Helper Script
# Use this if you chose "Continue without monitoring" during training
#

set -e

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

# Check for job ID
if [ ! -f .last_job_id ]; then
    print_error "No job ID found!"
    echo ""
    echo "Please provide job ID as argument:"
    echo "  ./download_results.sh <job-id>"
    echo ""
    echo "Or check your jobs:"
    echo "  ngc batch list"
    exit 1
fi

JOB_ID=$(cat .last_job_id)

print_banner "Download NGC Training Results"

print_info "Job ID: $JOB_ID"
echo ""

# Check job status
print_info "Checking job status..."
STATUS=$(ngc batch info "$JOB_ID" 2>/dev/null | grep "Status:" | awk '{print $2}' || echo "UNKNOWN")

case "$STATUS" in
    FINISHED_SUCCESS)
        print_step "Job completed successfully!"
        ;;
    RUNNING|QUEUED|STARTING)
        print_error "Job is still running: $STATUS"
        echo ""
        echo "Wait for completion or monitor:"
        echo "  ngc batch logs $JOB_ID --follow"
        exit 1
        ;;
    FINISHED_FAILURE|FAILED|FAILED_RUNONCE)
        print_error "Job failed: $STATUS"
        echo ""
        echo "View logs to diagnose:"
        echo "  ngc batch logs $JOB_ID"
        exit 1
        ;;
    *)
        print_error "Unknown status: $STATUS"
        exit 1
        ;;
esac

# Download results
print_banner "Downloading Results"

print_info "Downloading model and checkpoints..."

ngc result download "$JOB_ID" --dest ./results/

if [ -f "./results/$JOB_ID/export/pacman_model.onnx" ]; then
    # Copy to convenient location
    mkdir -p ./export
    cp "./results/$JOB_ID/export/pacman_model.onnx" ./export/

    MODEL_SIZE=$(du -h ./export/pacman_model.onnx | cut -f1)
    print_step "Model downloaded: export/pacman_model.onnx ($MODEL_SIZE)"
else
    print_error "Model not found in results!"
    exit 1
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

# Ask about cleanup
print_banner "Cleanup"

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

print_banner "✓ DOWNLOAD COMPLETE!"

echo ""
echo "Your trained model is ready:"
echo "  export/pacman_model.onnx"
echo ""
echo "Next steps:"
echo "  1. Integrate model into your app"
echo "  2. Replace heuristics with ML predictions"
echo "  3. Test and deploy!"
echo ""
