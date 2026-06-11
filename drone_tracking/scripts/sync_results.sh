#!/bin/bash
# =============================================================================
# sync_results.sh — Copy WSL results to the Windows OneDrive FYP folder
# Usage: ./sync_results.sh
#
# Results are written to ~/fyp/Results (WSL ext4) during runs because the
# /mnt/c bridge is slow and OneDrive can lock files while syncing mid-run.
# This script mirrors them to the Windows side AFTER runs/batches finish.
# Called automatically at the end of run_config.sh; safe to run manually.
# =============================================================================
set -u

SRC=~/fyp/Results
DST="/mnt/c/Users/fakhe/OneDrive/Desktop/FYP/Results"

[ -d "$SRC" ] || { echo "[sync] nothing to sync ($SRC missing)"; exit 0; }
mkdir -p "$DST"

if command -v rsync >/dev/null 2>&1; then
    rsync -rt --out-format='  %n' "$SRC/" "$DST/"
else
    cp -ruv "$SRC/." "$DST/"
fi
echo "[sync] done → $DST"
