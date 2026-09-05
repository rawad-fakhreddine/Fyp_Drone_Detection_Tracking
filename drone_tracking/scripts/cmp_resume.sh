#!/usr/bin/env bash
# cmp_resume.sh — resume the C1/C2 comparison matrix (C1-first, 150s, headless).
# Ledger at ~/fyp/Results/comparison_baylands/ledger.txt skips already-done cells,
# so this safely picks up wherever it stopped. Runs the rest of C1, then all C2.
cd ~/catkin_ws/src/drone_tracking/scripts
CELLS=""
for CFG in 1 2; do
  for TRAJ in 1 2 3 4 5 6 7 8; do
    for SEED in 42 43 45 46 47 48 49 50; do
      CELLS="$CELLS $TRAJ:$CFG:$SEED"
    done
  done
done
LOG=/tmp/cmp_resume_$(date +%Y%m%d_%H%M%S).log
CELLS="$CELLS" HEADLESS=1 DUR=150 nohup bash ms6_cmp.sh > "$LOG" 2>&1 &
disown
echo "resumed  PID=$!  log=$LOG"
echo "ledger: ~/fyp/Results/comparison_baylands/ledger.txt (done cells auto-skip)"
