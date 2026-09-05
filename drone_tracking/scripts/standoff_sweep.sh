#!/bin/bash
# standoff_sweep.sh — IBVS HOLD-standoff sweep on the T7 FOV-stress trajectory.
# 4 standoffs {current(3m),5m,6m,7m} x seeds {42,43}, Config 2, Zone 5, headless.
# Each run: launch_stack (env-injected alpha_star/dead_zone/ea_hold) + a parallel
# read-only gt_projection logger -> per-run gt CSV; flight CSV snapshotted too.
# Analysis is done separately (standoff_analyze.py) on these CSVs.
set -u
SWEEPDIR=~/fyp/Results/diagnostics/standoff
mkdir -p "$SWEEPDIR"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
PROG="$SWEEPDIR/progress.log"
DURATION=200
: > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # label alpha_star dead_zone ea_hold seed
  local label=$1 a=$2 d=$3 e=$4 seed=$5
  local tag="T7_s${seed}_${label}"
  local gtcsv="$SWEEPDIR/${tag}_gt.csv"
  local flt="$SWEEPDIR/${tag}_flight.csv"
  local llog="/tmp/standoff_launch_${tag}.log"
  local glog="/tmp/standoff_gt_${tag}.log"
  log ""
  log "===== $(date +%H:%M:%S)  $tag  alpha_star=$a dead=$d ea_hold=$e ====="
  bash "$CLEANUP" >/dev/null 2>&1; sleep 3
  ALPHA_STAR=$a DEAD_ZONE=$d EA_HOLD=$e VIEWER=0 LOSS_TIMEOUT=60 \
    bash "$LS" 2 7 5 "$seed" "$DURATION" > "$llog" 2>&1 &
  local LP=$!
  echo -n "  waiting for /drone_tracking/target_center " | tee -a "$PROG"
  local i
  for i in $(seq 1 260); do
    if timeout 3 rostopic echo -n1 /drone_tracking/target_center >/dev/null 2>&1; then echo " live" | tee -a "$PROG"; break; fi
    if ! kill -0 $LP 2>/dev/null; then echo " (launch exited early!)" | tee -a "$PROG"; break; fi
    sleep 1
  done
  rosrun drone_tracking gt_projection.py --log "$gtcsv" > "$glog" 2>&1 &
  local GP=$!
  wait $LP
  kill -INT $GP 2>/dev/null || true; sleep 2; kill -9 $GP 2>/dev/null || true
  # snapshot the flight CSV this run just produced (newest T7/zone5 in Config2)
  local newest
  newest=$(ls -t ~/fyp/Results/Config2/traj7_zone5_*.csv 2>/dev/null | head -1)
  [ -n "$newest" ] && cp "$newest" "$flt"
  log "  DONE $tag gt_lines=$(wc -l < "$gtcsv" 2>/dev/null) flight=$(basename "${newest:-NONE}")"
}

log "STANDOFF SWEEP START $(date)  DURATION=${DURATION}s LOSS_TIMEOUT=60 Config2 Zone5 T7"
for seed in 42 43; do
  run_one current  0.0067  0.00200 0.01000 "$seed"
  run_one s5       0.00250 0.00075 0.00373 "$seed"
  run_one s6       0.00173 0.00052 0.00258 "$seed"
  run_one s7       0.00127 0.00038 0.00190 "$seed"
done
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "ALL DONE $(date)"
