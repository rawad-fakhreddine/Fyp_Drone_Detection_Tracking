#!/bin/bash
# tau_sweep.sh — sweep ~search_tau for the v6.30 light SEARCH heading latch.
# For each tau in the list: T7 {42,43,45} + T9 {42,43,45}, Config2 Zone5,
# SEARCH_LATCH=true SEARCH_TAU=tau. Baseline (v6.28) and tau=1.0 are already
# measured (search_latch_v630 dir) — this fills in larger tau. Per-cell stale-
# CSV mtime guard + 3 attempts. Data -> ~/fyp/Results/diagnostics/search_tau/tauN/.
set -u
TAUS="${*:-2 3 4}"
ROOT=~/fyp/Results/diagnostics/search_tau
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
DURATION=200
PROG="$ROOT/progress.log"
mkdir -p "$ROOT"; : > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # tau traj seed
  local tau=$1 traj=$2 seed=$3
  local dir="$ROOT/tau${tau}"; mkdir -p "$dir"
  local tag="T${traj}_s${seed}_latch"
  local gtcsv="$dir/${tag}_gt.csv" flt="$dir/${tag}_flight.csv"
  local llog="/tmp/tausw_launch_tau${tau}_${tag}.log"
  local glog="/tmp/tausw_gt_tau${tau}_${tag}.log"
  local attempt
  for attempt in 1 2 3; do
    log ""
    log "===== $(date +%H:%M:%S)  tau=$tau  $tag  attempt=$attempt ====="
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    local START=$(date +%s)
    SEARCH_LATCH=true SEARCH_TAU=$tau VIEWER=0 LOSS_TIMEOUT=60 \
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
    if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$START" ]; then
      cp "$newest" "$flt"
      # confirm tau actually took
      local tauseen
      tauseen=$(grep -o "tau=[0-9.]*s" "$(ls -t /tmp/T7_traj${traj}_zone5_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
      log "  OK tau=$tau $tag flight=$(basename "$newest") md5=$(md5sum "$flt"|cut -c1-8) banner_$tauseen"
      return 0
    fi
    log "  FAIL tau=$tau $tag attempt=$attempt (takeoff gate?). retrying..."
  done
  log "  GAVE UP tau=$tau $tag"
  return 1
}

log "TAU SWEEP START $(date)  taus={$TAUS}  seeds{42,43,45}  Config2 Zone5 ${DURATION}s LOSS_TIMEOUT=60"
for tau in $TAUS; do
  for seed in 42 43 45; do run_one "$tau" 7 "$seed"; done
  for seed in 42 43 45; do run_one "$tau" 9 "$seed"; done
done
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "=== MD5 UNIQUENESS (dup = stale snapshot) ==="
md5sum "$ROOT"/tau*/*_flight.csv 2>/dev/null | sort | awk '{c[$1]++;m[$1]=m[$1]" "$2} END{for(k in c) if(c[k]>1) print "DUP",k,m[k]}' | tee -a "$PROG"
log "(no DUP lines = all unique)"
log "TAU SWEEP DONE $(date)"
