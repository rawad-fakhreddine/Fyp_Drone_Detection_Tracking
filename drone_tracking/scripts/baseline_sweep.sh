#!/bin/bash
# baseline_sweep.sh — M10.3 baseline diagnostic sweep (DEPLOYED Config 2, gains
# at confirmed-optimal; NOTHING changed in the controller). Produces the
# systematic weakness map + clean multi-trajectory data for ea/ex/ey AUC.
#   Trajectories T2..T9 x seeds {42,43,45} x Zone 5, 200 s, LOSS_TIMEOUT=60.
#   Full flight_logger + gt_projection per run. = 24 runs (~3 h).
# Baseline = NO KP_WZ/KP_Y override (defaults 0.9/1.8), search_latch OFF
# (=byte-for-byte v6.28). T9 is a stress probe (report separately).
# Per-cell stale-CSV mtime guard + 3 attempts + baseline-banner confirmation.
# Data -> ~/fyp/Results/diagnostics/baseline_sweep/T<t>/.
set -u
TRAJS="${TRAJS:-2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-42 43 45}"
ROOT=~/fyp/Results/diagnostics/baseline_sweep
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
DURATION=200
EXPECT_BANNER="Kp_wz=0.90 Kp_y=1.80"   # baseline gains, confirms nothing overridden
PROG="$ROOT/progress.log"
mkdir -p "$ROOT"; : > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # traj seed
  local traj=$1 seed=$2
  local dir="$ROOT/T${traj}"; mkdir -p "$dir"
  local tag="T${traj}_s${seed}"
  local flt="$dir/${tag}_flight.csv" gtcsv="$dir/${tag}_gt.csv"
  local llog="/tmp/blsw_launch_${tag}.log" glog="/tmp/blsw_gt_${tag}.log"
  local attempt
  for attempt in 1 2 3; do
    log ""
    log "===== $(date +%H:%M:%S)  $tag  attempt=$attempt ====="
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    local START=$(date +%s)
    VIEWER=0 LOSS_TIMEOUT=60 \
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
      local banner
      banner=$(grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*" \
        "$(ls -t /tmp/T7_traj${traj}_zone5_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
      if [ -n "$banner" ] && [ "$banner" != "$EXPECT_BANNER" ]; then
        log "  !! BANNER MISMATCH expected baseline '$EXPECT_BANNER' got '$banner' — retrying"
        continue
      fi
      local gtn; gtn=$(wc -l < "$gtcsv" 2>/dev/null || echo 0)
      log "  OK $tag flight=$(basename "$newest") md5=$(md5sum "$flt"|cut -c1-8) gt_rows=$gtn banner[$banner]"
      return 0
    fi
    log "  FAIL $tag attempt=$attempt (takeoff gate?). retrying..."
  done
  log "  GAVE UP $tag"
  return 1
}

log "BASELINE SWEEP START $(date)  trajs={$TRAJS} seeds{$SEEDS} Config2 Zone5 ${DURATION}s LOSS_TIMEOUT=60"
for traj in $TRAJS; do
  for seed in $SEEDS; do run_one "$traj" "$seed"; done
done
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "=== MD5 UNIQUENESS (dup = stale snapshot) ==="
md5sum "$ROOT"/T*/*_flight.csv 2>/dev/null | sort | awk '{c[$1]++;m[$1]=m[$1]" "$2} END{for(k in c) if(c[k]>1) print "DUP",k,m[k]}' | tee -a "$PROG"
log "(no DUP lines = all unique)"
log "BASELINE SWEEP DONE $(date)"
