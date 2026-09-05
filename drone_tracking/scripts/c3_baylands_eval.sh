#!/usr/bin/env bash
# c3_baylands_eval.sh — evaluate td3_wsCAPS_BEST on BAYLANDS (C1/C2 world),
# moving trajectories T4/T5/T8 x seeds, unified metric. Serial, one sim at a time.
# Halts on real camera degradation (det<0.65). Ledger-skips done cells.
MODEL="${MODEL:-$HOME/fyp/rl/models/td3_wsCAPS_BEST.zip}"
C3="${C3DIR:-$HOME/fyp/Results/comparison_baylands/c3_eval}"
LEDGER="$C3/ledger.txt"; METR="$C3/metrics.tsv"
mkdir -p "$C3/csv"; touch "$LEDGER"
SCR="$HOME/catkin_ws/src/drone_tracking/scripts"
CELLS="${CELLS:-4:43 4:45 5:42 5:43 5:45 8:42 8:43 8:45}"
for CELL in $CELLS; do
  TRAJ="${CELL%%:*}"; SEED="${CELL##*:}"
  LBL="T${TRAJ}_C3_s${SEED}"
  grep -q "^$LBL " "$LEDGER" && { echo "skip $LBL (done)"; continue; }
  echo "=== $LBL  $(date '+%H:%M:%S') ==="
  WORLD=baylands ACCEL_LIMIT=6 \
  MAX_VZ="${MAX_VZ:-2.5}" ACCEL_LIMIT_VZ="${ACCEL_LIMIT_VZ:-6}" \
  bash "$SCR/rl_td3_eval_launch.sh" 150 "$TRAJ" "$SEED" "$MODEL" \
      > "/tmp/c3eval_${LBL}.log" 2>&1
  sleep 3
  CSV="$C3/csv/${LBL}.csv"
  cp "$HOME/flight_log_latest.csv" "$CSV" 2>/dev/null
  LINE=$(python3 "$HOME/fyp/rl/osc_analyze.py" "$CSV" "$LBL" 2>/dev/null | tr '\n' ' ')
  DET=$(echo "$LINE" | grep -oE "det=[0-9.]+" | head -1 | cut -d= -f2)
  HOLD=$(echo "$LINE" | grep -oE "HOLD\(cen&\[5,9\]\)=[0-9.]+" | cut -d= -f2)
  echo "$LBL det=$DET HOLD=$HOLD" >> "$LEDGER"
  { printf '%s\t' "$LBL"; echo "$LINE"; } >> "$METR"
  echo "  -> det=$DET HOLD=$HOLD"
  # degradation canary: det<THRESH = camera starving the policy -> halt for wsl restart.
  # Threshold overridable (DET_HALT); default 0.65 but for all-traj evals set ~0.30 so hard
  # trajectories (T2/3/6/7) with legitimately low det aren't mistaken for a camera leak.
  if python3 -c "import sys; d='$DET'; sys.exit(0 if (d and float(d)<${DET_HALT:-0.65}) else 1)"; then
    echo "  HALT: det=$DET < ${DET_HALT:-0.65} — camera degraded, needs wsl --shutdown" >> "$LEDGER"
    echo "HALT-DEGRADED at $LBL (det=$DET)"; exit 3
  fi
  # deep teardown between cells: releases yolo/gzserver GPU context + waits for VRAM to
  # drain — slows the camera leak so more cells complete before a wsl --shutdown is needed.
  bash "$SCR/deep_teardown.sh" 2>/dev/null
done
echo "C3 BAYLANDS EVAL DONE"
