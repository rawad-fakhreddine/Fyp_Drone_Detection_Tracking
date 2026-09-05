#!/bin/bash
# kp_sweep.sh — M10.3 Kp_wz x Kp_y angular-tracking gain sweep on T7/Zone5.
# Coarse OFAT direction-finder (signed off 2026-06-20):
#   A base  : Kp_wz=0.9  Kp_y=1.8   (fresh paired anchor)
#   B wz^   : Kp_wz=1.5  Kp_y=1.8   (x1.67, isolate yaw centering)
#   C both^ : Kp_wz=1.5  Kp_y=2.7   (yaw + lateral)
# Each cell: T7 {42,43,45}, Config2 Zone5, 200s, LOSS_TIMEOUT=60, VIEWER=0.
# Gains driven by KP_WZ/KP_Y launch env knobs (defaults preserve baseline).
# Only the flight CSV is needed (HOLD%, cmd_wz/cmd_vy saturation, yaw
# zero-crossing, recovery/aborts all come from it) -> no gt_projection.
# Per-cell stale-CSV mtime guard + 3 attempts + banner confirmation.
# Data -> ~/fyp/Results/diagnostics/kp_sweep/<cell>/.
set -u
CELLS="${*:-A B C}"
declare -A KPWZ=( [A]=0.9 [B]=1.5 [C]=1.5 )
declare -A KPY=(  [A]=1.8 [B]=1.8 [C]=2.7 )
ROOT=~/fyp/Results/diagnostics/kp_sweep
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
DURATION=200
PROG="$ROOT/progress.log"
mkdir -p "$ROOT"; : > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

run_one() {   # cell seed
  local cell=$1 seed=$2
  local kpwz=${KPWZ[$cell]} kpy=${KPY[$cell]}
  local dir="$ROOT/$cell"; mkdir -p "$dir"
  local tag="T7_s${seed}"
  local flt="$dir/${tag}_flight.csv"
  local llog="/tmp/kpsw_launch_${cell}_${tag}.log"
  local attempt
  for attempt in 1 2 3; do
    log ""
    log "===== $(date +%H:%M:%S)  cell=$cell (Kp_wz=$kpwz Kp_y=$kpy)  $tag  attempt=$attempt ====="
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    local START=$(date +%s)
    KP_WZ=$kpwz KP_Y=$kpy VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" 2 7 5 "$seed" "$DURATION" > "$llog" 2>&1 &
    local LP=$! i
    for i in $(seq 1 260); do
      if timeout 3 rostopic echo -n1 /drone_tracking/target_center >/dev/null 2>&1; then break; fi
      if ! kill -0 $LP 2>/dev/null; then break; fi
      sleep 1
    done
    wait $LP
    local newest
    newest=$(ls -t ~/fyp/Results/Config2/traj7_zone5_*.csv 2>/dev/null | head -1)
    if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$START" ]; then
      cp "$newest" "$flt"
      local banner
      banner=$(grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*" \
        "$(ls -t /tmp/T7_traj7_zone5_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
      log "  OK $cell $tag flight=$(basename "$newest") md5=$(md5sum "$flt"|cut -c1-8) banner[$banner]"
      # hard fail if the banner gains don't match the requested cell.
      # Node prints %.2f, so normalise the expected string to 2 decimals
      # (else 0.9 != 0.90 trips a false mismatch + needless rerun).
      local exp_banner; exp_banner=$(printf "Kp_wz=%.2f Kp_y=%.2f" "$kpwz" "$kpy")
      if [ "$banner" != "$exp_banner" ]; then
        log "  !! BANNER MISMATCH expected '$exp_banner' got '$banner' — retrying"
        continue
      fi
      return 0
    fi
    log "  FAIL $cell $tag attempt=$attempt (takeoff gate?). retrying..."
  done
  log "  GAVE UP $cell $tag"
  return 1
}

log "KP SWEEP START $(date)  cells={$CELLS}  seeds{42,43,45}  T7 Config2 Zone5 ${DURATION}s LOSS_TIMEOUT=60"
for cell in $CELLS; do
  for seed in 42 43 45; do run_one "$cell" "$seed"; done
done
bash "$CLEANUP" >/dev/null 2>&1
log ""
log "=== MD5 UNIQUENESS (dup = stale snapshot) ==="
md5sum "$ROOT"/*/*_flight.csv 2>/dev/null | sort | awk '{c[$1]++;m[$1]=m[$1]" "$2} END{for(k in c) if(c[k]>1) print "DUP",k,m[k]}' | tee -a "$PROG"
log "(no DUP lines = all unique)"
log "KP SWEEP DONE $(date)"
