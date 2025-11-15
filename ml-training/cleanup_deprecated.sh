#!/bin/bash
# Cleanup Script - Archive Deprecated Files
# Moves old files to deprecated/ directory

echo "======================================================================"
echo "Cleaning Up Deprecated Files"
echo "======================================================================"
echo ""

# Create deprecated directory
mkdir -p deprecated/training_old
mkdir -p deprecated/agents_old
mkdir -p deprecated/docs_old

echo "[+] Created deprecated/ directories"
echo ""

# Move deprecated training scripts
if [ -d "training" ]; then
    echo "[+] Archiving old training scripts..."
    mv training/train.py deprecated/training_old/ 2>/dev/null || true
    mv training/train_dqn.py deprecated/training_old/ 2>/dev/null || true
    mv training/train_rl.py deprecated/training_old/ 2>/dev/null || true
    echo "    → Moved training/*.py to deprecated/training_old/"
fi

# Move deprecated agents
if [ -d "agents" ]; then
    echo "[+] Archiving old agent files..."
    mv agents deprecated/agents_old/ 2>/dev/null || true
    echo "    → Moved agents/ to deprecated/agents_old/"
fi

# Archive outdated docs (optional)
echo "[+] Archiving potentially outdated docs..."
mv RL_TRAINING.md deprecated/docs_old/ 2>/dev/null || true
echo "    → Moved RL_TRAINING.md to deprecated/docs_old/ (will recreate updated version)"

echo ""
echo "======================================================================"
echo "Cleanup Complete!"
echo "======================================================================"
echo ""
echo "Deprecated files moved to: deprecated/"
echo ""
echo "Active files remaining:"
echo "  ✅ train_128.py          - Phase 1 training (128-dim)"
echo "  ✅ train_rl_headless.py  - Phase 2 RL training"
echo "  ✅ export_cloudflare_data.py - Data export"
echo "  ✅ inference_server.py   - HF Spaces inference"
echo "  ✅ models/dqn_network.py - Model architecture"
echo "  ✅ environment/pacman_env.py - Headless environment"
echo ""
echo "Next steps:"
echo "  1. Review deprecated/ directory"
echo "  2. Delete if not needed: rm -rf deprecated/"
echo "  3. Or add to .gitignore: echo 'deprecated/' >> .gitignore"
echo ""
