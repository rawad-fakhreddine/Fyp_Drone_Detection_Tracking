#!/bin/bash
# vz_sweep.sh — M10.3 vertical-channel gain sweep (OFAT, same structure as the
# Kp_wz/Kp_y horizontal sweep). Config 2, Zone 1, T6+T7, seeds {42,43,45}.
# Cells (treatment = Kp_z; max_vz held at baseline — saturation pre-check showed
# the cap is NOT binding, so cell D is deferred):
#   A baseline  Kp_z=3.0  max_vz=1.5   (fresh paired anchor)
#   B Kp_z x1.5 Kp_z=4.5  max_vz=1.5
#   C Kp_z x2.0 Kp_z=6.0  max_vz=1.5
# = 3 cells x 2 traj x 3 seeds = 18 runs. 200 s, LOSS_TIMEOUT=60, latch OFF.
# Gains in the FILE are untouched; the treatment is injected via the KP_Z env
# (rosparam ~Kp_z). Per run: MD5-uniqueness + banner check (Kp_z matches cell,
# max_vz=1.50, Kp_wz=0.90 Kp_y=1.80 untouched, latch OFF). Records cell->CSV
# mapping to the summary so the analyzer reads by cell, not by guessing newest.
#   bash vz_sweep.sh         # full 18-run sweep
#   bash vz_sweep.sh --dry   # print plan only
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
ZONE=1; DURATION=200; CFG=2
TRAJ=(6 7); SEEDS=(42 43 45)
CELLS=(A B C); declare -A KPZ=([A]=3.0 [B]=4.5 [C]=6.0)
EXPECT_HZ="Kp_wz=0.90 Kp_y=1.80"   # horizontal gains must stay untouched
LOG=~/fyp/Results/diagnostics/vz_sweep/sweep.log
mkdir -p "$(dirname "$LOG")"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
declare -A SEEN_MD5
say(){ echo "$@" | tee -a "$LOG"; }

settle(){   # same hardened teardown as matrix_c1c2 (WSL2 ~7.6 GB; orphan px4)
  bash "$S/cleanup.sh" >/dev/null 2>&1
  pkill -9 -f 'px4-simulator' 2>/dev/null; pkill -9 -f 'px4-' 2>/dev/null
  for port in 4560 4561 11311; do lsof -ti :$port 2>/dev/null | xargs -r kill -9 2>/dev/null; done
  local i procs avail
  for i in $(seq 1 90); do
    procs=$(pgrep -c -f 'gzserver|gzclient|px4|mavros' 2>/dev/null||true)
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${procs:-0}" -eq 0 ] && [ "${avail:-0}" -ge 5000 ]; then break; fi
    sleep 2
  done
  sleep 6
}

run_one(){  # cell traj seed tag
  local cell=$1 traj=$2 seed=$3 tag=$4 kpz=${KPZ[$1]}
  local rdir=~/fyp/Results/Config${CFG}
  say ">>> [$tag] cell $cell  Kp_z=$kpz  T$traj  zone $ZONE  seed $seed  ${DURATION}s"
  if [ $DRY -eq 1 ]; then return 0; fi
  local before csv attempt
  before=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
  csv=""
  for attempt in 1 2 3; do
    settle
    [ "$attempt" -gt 1 ] && say "  (retry $attempt/3 — prior attempt saved no CSV, likely bringup OOM)"
    KP_Z=$kpz SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" "$CFG" "$traj" "$ZONE" "$seed" "$DURATION" >> "$LOG" 2>&1
    csv=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$csv" != "$before" ]; then break; fi
    csv=""
  done
  if [ -z "$csv" ]; then say "  !! FAILED all 3 (no CSV — bringup) $cell T$traj s$seed"; echo "FAILALL $cell T$traj s$seed" >>"$LOG.summary"; return 1; fi
  local md5 banner kpz_b hz latch t7log bok="ok"
  md5=$(md5sum "$csv" | cut -d' ' -f1)
  if [ -n "${SEEN_MD5[$md5]:-}" ]; then say "  !! STALE (md5 dup of ${SEEN_MD5[$md5]})"; echo "STALE $cell T$traj s$seed" >>"$LOG.summary"
  else SEEN_MD5[$md5]="$cell/T$traj/s$seed"; fi
  t7log=$(ls -t /tmp/T7_traj${traj}_zone${ZONE}_*.log 2>/dev/null | head -1)
  banner=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null | head -1)
  kpz_b=$(echo "$banner" | grep -o "Kp_z=[0-9.]*")
  hz=$(echo "$banner" | grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*")
  latch=$(echo "$banner" | grep -o "search_latch=[A-Za-z]*")
  local want_kpz; want_kpz=$(printf "Kp_z=%.2f" "$kpz")
  [ "$kpz_b" != "$want_kpz" ] && { say "  !! Kp_z BANNER MISMATCH: got '$kpz_b' want '$want_kpz'"; bok="BAD"; }
  [ "$hz" != "$EXPECT_HZ" ]   && { say "  !! HORIZONTAL GAIN CHANGED: '$hz'"; bok="BAD"; }
  [ "$latch" != "search_latch=False" ] && { say "  !! LATCH NOT OFF: '$latch'"; bok="BAD"; }
  say "  saved $(basename "$csv")  md5=${md5:0:8}  $kpz_b  banner=$bok"
  echo "OK $cell T$traj s$seed csv=$(basename "$csv") md5=${md5:0:8} kpz=$kpz banner=$bok" >>"$LOG.summary"
}

: > "$LOG"; : > "$LOG.summary"
say "================================================================"
say " M10.3 VERTICAL Kp_z SWEEP   START $(date)"
say " cells A/B/C (Kp_z 3.0/4.5/6.0) x traj{${TRAJ[*]}} x seed{${SEEDS[*]}}"
say " Config $CFG  zone $ZONE  ${DURATION}s  LOSS_TIMEOUT=60  latch OFF  max_vz=1.5"
say "================================================================"
N=0; TOTAL=$(( ${#CELLS[@]} * ${#TRAJ[@]} * ${#SEEDS[@]} ))
for cell in "${CELLS[@]}"; do
  for traj in "${TRAJ[@]}"; do
    for seed in "${SEEDS[@]}"; do
      N=$((N+1)); run_one "$cell" "$traj" "$seed" "$N/$TOTAL"
    done
  done
done
say ""
say "=== VZ SWEEP SUMMARY ($(date)) ==="
cat "$LOG.summary" | tee -a "$LOG"
say "stale: $(grep -c '^STALE' "$LOG.summary"||true)  failed-all-3: $(grep -c '^FAILALL' "$LOG.summary"||true)  banner-bad: $(grep -c 'banner=BAD' "$LOG.summary"||true)"
say "=== M10.3 VZ SWEEP DONE $(date) ==="
