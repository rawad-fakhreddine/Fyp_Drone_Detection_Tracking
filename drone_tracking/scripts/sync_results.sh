#!/bin/bash
# =============================================================================
# sync_results.sh — Copy WSL results to the Windows OneDrive FYP folder
# Usage: ./sync_results.sh
#
# Results are written to ~/fyp/Results (WSL ext4) during runs because the
# /mnt/c bridge is slow and OneDrive can lock files while syncing mid-run.
# This script mirrors them to the Windows side AFTER runs/batches finish.
# Called automatically at the end of run_config.sh; safe to run manually.
#
# 2026-07-12 split (M11.3): OneDrive "Results" is now the CURATED reference
# folder (built from ~/fyp/Results_reference — hand-picked runs, README,
# summary tables). Raw working data mirrors to "Results_raw" instead, so
# batch syncs can never pollute the reference.
# =============================================================================
set -u

SRC=~/fyp/Results
REF=~/fyp/Results_reference
DST_RAW="/mnt/c/Users/fakhe/OneDrive/Desktop/FYP/Results_raw"
DST_REF="/mnt/c/Users/fakhe/OneDrive/Desktop/FYP/Results"

[ -d "$SRC" ] || { echo "[sync] nothing to sync ($SRC missing)"; exit 0; }
mkdir -p "$DST_RAW"

if command -v rsync >/dev/null 2>&1; then
    rsync -rt --exclude 'screenshots' --exclude 'diagnostics' \
          --out-format='  %n' "$SRC/" "$DST_RAW/"
else
    cp -ru "$SRC/." "$DST_RAW/"
fi
echo "[sync] raw done → $DST_RAW"

# curated reference (only if it exists; updated by hand/Claude, never by batches)
if [ -d "$REF" ]; then
    mkdir -p "$DST_REF"
    if command -v rsync >/dev/null 2>&1; then
        rsync -rt --delete --out-format='  %n' "$REF/" "$DST_REF/"
    else
        cp -ru "$REF/." "$DST_REF/"
    fi
    echo "[sync] reference done → $DST_REF"
fi
