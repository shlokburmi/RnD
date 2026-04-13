/#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  Raspberry Pi 4 launcher for CNN-SPECK batch processor
#
# Usage (from project directory):
#   chmod +x run.sh      # only needed once
#   ./run.sh             # run interactively  (Ctrl+C to cancel)
#   ./run.sh &           # run in background
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/run.log"
RESULTS="$SCRIPT_DIR/cnnresults.txt"

echo "============================================================"
echo "  CNN-SPECK Raspberry Pi 4 — Batch Runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Logs  : $LOG"
echo "  Results: $RESULTS"
echo "============================================================"

cd "$SCRIPT_DIR"

# Redirect stdout + stderr to log file AND terminal simultaneously
python3 batch_process.py 2>&1 | tee "$LOG"

echo ""
echo "============================================================"
echo "  Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results saved to: $RESULTS"
echo "============================================================"
