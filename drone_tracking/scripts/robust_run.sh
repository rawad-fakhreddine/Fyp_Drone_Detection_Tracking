#!/bin/bash
# robust_run.sh TAG CONFIG TRAJ ZONE SEED DUR — one run with infra-retry.
# Env vars (VX_MODE, KP_VX, ... STRAIGHT_AZ, etc.) are inherited/exported by
# the caller. Retries up to 3x if the launch dies at a startup gate (T1/T6
# TIMEOUT = PX4/Gazebo lockstep flake) rather than producing a mission result.
TAG=$1; CFG=$2; TRAJ=$3; ZONE=$4; SEED=$5; DUR=$6
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEAN=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
LOG="/tmp/run_${TAG}.log"
for attempt in 1 2 3; do
  bash "$CLEAN" >/dev/null 2>&1; sleep 6
  bash "$LS" "$CFG" "$TRAJ" "$ZONE" "$SEED" "$DUR" > "$LOG" 2>&1
  if grep -qE 'ABORTED at|Mission complete' "$LOG"; then
    CSV=$(grep '^\[Save\]' "$LOG" | tail -1 | awk '{print $2}')
    echo "[$TAG] OK (attempt $attempt): $(grep -E 'ABORTED at|Mission complete' "$LOG" | tail -1)"
    echo "$CSV"
    exit 0
  fi
  echo "[$TAG] infra flake attempt $attempt ($(grep -oE 'TIMEOUT' "$LOG" | head -1)) — retrying"
done
echo "[$TAG] FAILED after 3 attempts"
exit 1
