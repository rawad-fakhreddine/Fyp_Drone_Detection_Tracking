#!/usr/bin/env bash
# cmp_resume_c1.sh — C1-ONLY resume (re-run failed/missing C1 cells).
# Does NOT touch C2. Ledger auto-skips already-done C1 cells, so this runs
# only the still-missing C1 cells (the WSL-failed re-runs). Start C2 only
# after C1 reaches 64/64.
cd ~/catkin_ws/src/drone_tracking/scripts
CELLS=""
for TRAJ in 1 2 3 4 5 6 7 8; do
  for SEED in 42 43 45 46 47 48 49 50; do
    CELLS="$CELLS $TRAJ:1:$SEED"
  done
done
LOG=/tmp/cmp_resume_c1_$(date +%Y%m%d_%H%M%S).log
CELLS="$CELLS" HEADLESS=1 DUR=150 nohup bash ms6_cmp.sh > "$LOG" 2>&1 &
disown
echo "resumed C1-only  PID=$!  log=$LOG"
echo "ledger: ~/fyp/Results/comparison_baylands/ledger.txt (done C1 cells auto-skip)"
