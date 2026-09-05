#!/bin/bash
# search_latch_ab.sh — A/B validation of the v6.29 SEARCH loss-instant velocity
# latch. base (SEARCH_LATCH=false = blind v6.28 SEARCH) vs latch (true).
# T7/T9 seeds {42,43,44} + T4 s42 no-harm check, Config 2, Zone 5, headless.
# Each run: launch_stack + parallel gt_projection logger; flight CSV snapshotted.
set -u
ABDIR=~/fyp/Results/diagnostics/search_latch
mkdir -p "$ABDIR"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
PROG="$ABDIR/progress.log"
DURATION=200
: > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # traj seed variant(base|latch) latchval(false|true)
  local traj=$1 seed=$2 variant=$3 latchval=$4
  local tag="T${traj}_s${seed}_${variant}"
  local gtcsv="$ABDIR/${tag}_gt.csv"
  local flt="$ABDIR/${tag}_flight.csv"
  local llog="/tmp/ablatch_launch_${tag}.log"
  local glog="/tmp/ablatch_gt_${tag}.log"
  log ""
  log "===== $(date +%H:%M:%S)  $tag  SEARCH_LATCH=$latchval ====="
  bash "$CLEANUP" >/dev/null 2>&1; sleep 3
  SEARCH_LATCH=$latchval VIEWER=0 LOSS_TIMEOUT=60 \
    bash "$LS" 2 "$traj" 5 "$seed" "$DURATION" > "$llog" 2>&1 &
  local LP=$!
  local i
  for i in $(seq 1 260); do
    if timeout 3 rostopic echo -n1 /drone_tracking/target_center >/dev/null 2>&1; then break; fi
    if ! kill -0 $LP 2>/dev/null; then log "  (launch exited early!)"; break; fi
    sleep 1
  done
  rosrun drone_tracking gt_projection.py --log "$gtcsv" > "$glog" 2>&1 &
  local GP=$!
  wait $LP
  kill -INT $GP 2>/dev/null || true; sleep 2; kill -9 $GP 2>/dev/null || true
  local newest
  newest=$(ls -t ~/fyp/Results/Config2/traj${traj}_zone5_*.csv 2>/dev/null | head -1)
  [ -n "$newest" ] && cp "$newest" "$flt"
  # verify the flag actually took (search_latch + LOSS log lines flush at node kill)
  local latchlog
  latchlog=$(grep -c "LOSS — latched" "$(ls -t /tmp/T7_traj${traj}_zone5_*.log /tmp/T7_traj*_zone5_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "?")
  log "  DONE $tag gt_lines=$(wc -l < "$gtcsv" 2>/dev/null) flight=$(basename "${newest:-NONE}")"
}

log "SEARCH-LATCH A/B START $(date)  DURATION=${DURATION}s LOSS_TIMEOUT=60 Config2 Zone5"
for seed in 42 43 44; do
  run_one 7 "$seed" base  false
  run_one 7 "$seed" latch true
done
for seed in 42 43 44; do
  run_one 9 "$seed" base  false
  run_one 9 "$seed" latch true
done
# T4 no-harm check (single seed)
run_one 4 42 base  false
run_one 4 42 latch true
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "ALL DONE $(date)"
