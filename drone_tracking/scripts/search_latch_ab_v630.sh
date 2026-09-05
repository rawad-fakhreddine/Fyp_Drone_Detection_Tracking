#!/bin/bash
# search_latch_ab_v630.sh — A/B validation of the v6.30 TIME-BOUND SEARCH heading
# latch. base (SEARCH_LATCH=false == byte-for-byte v6.28 blind SEARCH) vs new
# (SEARCH_LATCH=true == v6.30 decaying latch). T7/T9 seeds {42,43,THIRD} +
# T4 s42 no-harm, Config 2, Zone 5, headless. Per-cell: cleanup -> launch_stack
# + parallel gt_projection -> snapshot the Config2 CSV with a STALE-CSV mtime
# guard + up to 3 attempts so a takeoff-gate timeout never grabs a prior run.
# THIRD seed passed as $1 (default 44).
set -u
THIRD="${1:-44}"
ABDIR=~/fyp/Results/diagnostics/search_latch_v630
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
  local llog="/tmp/ab630_launch_${tag}.log"
  local glog="/tmp/ab630_gt_${tag}.log"
  local attempt
  for attempt in 1 2 3; do
    log ""
    log "===== $(date +%H:%M:%S)  $tag  SEARCH_LATCH=$latchval  attempt=$attempt ====="
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    local START=$(date +%s)
    SEARCH_LATCH=$latchval VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" 2 "$traj" 5 "$seed" "$DURATION" > "$llog" 2>&1 &
    local LP=$! i
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
    # STALE GUARD: only accept a CSV written AFTER this attempt began
    if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$START" ]; then
      cp "$newest" "$flt"
      log "  OK $tag flight=$(basename "$newest") gt_lines=$(wc -l < "$gtcsv" 2>/dev/null) md5=$(md5sum "$flt" | cut -c1-8)"
      return 0
    fi
    log "  FAIL $tag attempt=$attempt (no fresh CSV — takeoff gate?). retrying..."
  done
  log "  GAVE UP $tag after 3 attempts"
  return 1
}

log "v6.30 TIME-BOUND LATCH A/B START $(date)  seeds {42,43,$THIRD}  DURATION=${DURATION}s LOSS_TIMEOUT=60 Config2 Zone5"
for seed in 42 43 "$THIRD"; do
  run_one 7 "$seed" base  false
  run_one 7 "$seed" latch true
done
for seed in 42 43 "$THIRD"; do
  run_one 9 "$seed" base  false
  run_one 9 "$seed" latch true
done
run_one 4 42 base  false
run_one 4 42 latch true
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "=== MD5 UNIQUENESS CHECK (duplicate = stale snapshot) ==="
md5sum "$ABDIR"/*_flight.csv 2>/dev/null | sort | awk '{c[$1]++; m[$1]=m[$1]" "$2} END{for(k in c) if(c[k]>1) print "DUP",k,m[k]}' | tee -a "$PROG"
log "(no DUP lines above = all snapshots unique)"
log ""
log "v6.30 A/B ALL DONE $(date)"
