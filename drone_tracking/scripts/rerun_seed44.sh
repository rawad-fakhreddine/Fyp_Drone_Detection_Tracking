#!/bin/bash
# rerun_seed44.sh — re-fly the 3 cells lost to the intermittent takeoff_ready
# gate timeout: T7_s44_base, T9_s44_base, T9_s44_latch. Stale-CSV guard +
# retry (up to 3 attempts) so a takeoff failure doesn't snapshot a prior CSV.
set -u
ABDIR=~/fyp/Results/diagnostics/search_latch
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
PROG="$ABDIR/rerun_progress.log"
DURATION=200
: > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # traj seed variant latchval
  local traj=$1 seed=$2 variant=$3 latchval=$4
  local tag="T${traj}_s${seed}_${variant}"
  local attempt
  for attempt in 1 2 3; do
    local gtcsv="$ABDIR/${tag}_gt.csv"
    local flt="$ABDIR/${tag}_flight.csv"
    local llog="/tmp/ablatch_launch_${tag}.log"
    local glog="/tmp/ablatch_gt_${tag}.log"
    log ""
    log "===== $(date +%H:%M:%S) $tag attempt $attempt SEARCH_LATCH=$latchval ====="
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    local START=$(date +%s)
    SEARCH_LATCH=$latchval VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" 2 "$traj" 5 "$seed" "$DURATION" > "$llog" 2>&1 &
    local LP=$!
    local i
    for i in $(seq 1 260); do
      if timeout 3 rostopic echo -n1 /drone_tracking/target_center >/dev/null 2>&1; then break; fi
      if ! kill -0 $LP 2>/dev/null; then break; fi
      sleep 1
    done
    rosrun drone_tracking gt_projection.py --log "$gtcsv" > "$glog" 2>&1 &
    local GP=$!
    wait $LP
    kill -INT $GP 2>/dev/null || true; sleep 2; kill -9 $GP 2>/dev/null || true
    local newest
    newest=$(ls -t ~/fyp/Results/Config2/traj${traj}_zone5_*.csv 2>/dev/null | head -1)
    # stale guard: only accept a CSV created AFTER this attempt started
    if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$START" ]; then
      cp "$newest" "$flt"
      log "  OK $tag flight=$(basename "$newest") gt_lines=$(wc -l < "$gtcsv" 2>/dev/null)"
      return 0
    fi
    log "  FAIL $tag attempt $attempt (no fresh CSV — takeoff gate?). retrying..."
  done
  log "  GAVE UP $tag after 3 attempts"
  return 1
}

log "RERUN seed44 START $(date)"
run_one 7 44 base  false
run_one 9 44 base  false
run_one 9 44 latch true
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "RERUN DONE $(date)"
