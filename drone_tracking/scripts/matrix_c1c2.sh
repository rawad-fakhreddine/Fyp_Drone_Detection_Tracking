#!/bin/bash
# matrix_c1c2.sh — M11 thesis validation matrix: Config 1 vs Config 2.
# FROZEN SCOPE (do not expand without resetting the matrix):
#   configs : 1 (YOLO+IBVS, raw det, no Kalman) vs 2 (YOLO+Kalman+IBVS)
#   traj    : control-ceiling {T2,T4,T5,T8} + detection-limited {T3,T6,T7}
#   appendix: T9 (stress, reported separately — NEVER a clean A/B number)
#   seeds   : {42,43,45}   zone : 1 (constant across both configs)
#   duration: 200 s        LOSS_TIMEOUT : 60 (recorded-outcome abort)
#   = 2 x 7 x 3 = 42 runs + T9 appendix (2 x 3 = 6) = 48 runs.
# Per run: MD5-uniqueness on the saved CSV (catch stale ~/flight_log_latest
# snapshots) + IBVS banner check (v6.30, Kp_wz=0.90 Kp_y=1.80, latch OFF).
# Config-switching ONLY — no code/gain/trajectory changes. Writes to
# ~/fyp/Results/Config{1,2}/ + consolidated summary.csv (via extract_metrics in
# launch_stack). Unattended; resumable-safe (timestamped CSVs never collide).
#   bash matrix_c1c2.sh           # full 48-run batch
#   bash matrix_c1c2.sh --dry     # print the plan, run nothing
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
ZONE=1; DURATION=200
MAIN_TRAJ=(2 4 5 8 3 6 7)      # control-ceiling first, then detection-limited
APPENDIX_TRAJ=(9)
SEEDS=(42 43 45)
EXPECT="Kp_wz=0.90 Kp_y=1.80"
LOG=~/fyp/Results/diagnostics/gate_c1c2/matrix_run.log
mkdir -p "$(dirname "$LOG")"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
declare -A SEEN_MD5

say(){ echo "$@" | tee -a "$LOG"; }

# Inter-run settle: WSL2 has only ~7.6 GB. launch_stack has no startup cleanup
# and back-to-back runs overlap teardown+bringup -> kernel SIGKILLs the new
# roslaunch during MAVROS init (proven in the takeoff gate: 3/6 alternating
# bringup failures). Clean, wait until heavy procs gone AND RAM recovered.
settle(){
  bash "$S/cleanup.sh" >/dev/null 2>&1
  # cleanup.sh BUG (tracked file, flagged separately): it kills 'bin/px4' and
  # '-x px4' but NOT the 'px4-simulator' SITL module, which orphans and holds
  # TCP 4560 -> blocks EVERY subsequent chaser bringup (caused the 24/24 C1
  # wipeout). Actively KILL the px4 client modules + free the MAVLink/master
  # ports here, then wait until heavy procs are gone AND RAM has recovered.
  pkill -9 -f 'px4-simulator' 2>/dev/null; pkill -9 -f 'px4-' 2>/dev/null
  for port in 4560 4561 11311; do
    lsof -ti :$port 2>/dev/null | xargs -r kill -9 2>/dev/null
  done
  local i procs avail
  for i in $(seq 1 90); do
    procs=$(pgrep -c -f 'gzserver|gzclient|px4|mavros' 2>/dev/null||true)
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${procs:-0}" -eq 0 ] && [ "${avail:-0}" -ge 5000 ]; then break; fi
    sleep 2
  done
  sleep 6
}

run_one(){  # config traj seed tag
  local cfg=$1 traj=$2 seed=$3 tag=$4
  local rdir=~/fyp/Results/Config${cfg}
  say ">>> [$tag] Config $cfg  T$traj  zone $ZONE  seed $seed  ${DURATION}s"
  if [ $DRY -eq 1 ]; then return 0; fi
  # Bounded retry: a bringup OOM SIGKILL produces NO new CSV -> retry (up to 3).
  # A run that reaches the mission ALWAYS saves a CSV (even aborted=1, which is
  # a legitimate recorded outcome, NOT a retry trigger). So "no new CSV" cleanly
  # isolates bringup failures from real results.
  local before csv attempt
  before=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
  csv=""
  for attempt in 1 2 3; do
    settle
    [ "$attempt" -gt 1 ] && say "  (retry $attempt/3 — prior attempt saved no CSV, likely bringup OOM)"
    SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" "$cfg" "$traj" "$ZONE" "$seed" "$DURATION" \
      >> "$LOG" 2>&1
    csv=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$csv" != "$before" ]; then break; fi
    csv=""
  done
  local md5 banner kp latch t7log
  if [ -z "$csv" ]; then say "  !! FAILED all 3 attempts (no CSV — bringup) cfg$cfg T$traj s$seed"; echo "FAILALL c$cfg T$traj s$seed" >>"$LOG.summary"; return 1; fi
  md5=$(md5sum "$csv" | cut -d' ' -f1)
  if [ -n "${SEEN_MD5[$md5]:-}" ]; then
    say "  !! STALE CSV (md5 dup of ${SEEN_MD5[$md5]}): $csv"; echo "STALE c$cfg T$traj s$seed" >>"$LOG.summary"
  else
    SEEN_MD5[$md5]="c$cfg/T$traj/s$seed"
  fi
  t7log=$(ls -t /tmp/T7_traj${traj}_zone${ZONE}_*.log 2>/dev/null | head -1)
  banner=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null | head -1)
  kp=$(echo "$banner" | grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*")
  latch=$(echo "$banner" | grep -o "search_latch=[A-Za-z]*")
  local bok="ok"
  [ "$kp" != "$EXPECT" ] && { say "  !! BANNER Kp MISMATCH: '$kp'"; bok="BAD"; }
  [ "$latch" != "search_latch=False" ] && { say "  !! LATCH NOT OFF: '$latch'"; bok="BAD"; }
  say "  saved $(basename "$csv")  md5=${md5:0:8}  banner=$bok"
  echo "OK c$cfg T$traj s$seed md5=${md5:0:8} banner=$bok" >>"$LOG.summary"
}

: > "$LOG"; : > "$LOG.summary"
say "================================================================"
say " M11 MATRIX  Config 1 vs Config 2   START $(date)"
say " scope: cfg{1,2} x traj{${MAIN_TRAJ[*]}} x seed{${SEEDS[*]}} + T9 appendix"
say " zone $ZONE  ${DURATION}s  LOSS_TIMEOUT=60  latch OFF  (config-switch only)"
say "================================================================"
N=0; TOTAL=$(( 2 * (${#MAIN_TRAJ[@]} + ${#APPENDIX_TRAJ[@]}) * ${#SEEDS[@]} ))
for cfg in 1 2; do
  for traj in "${MAIN_TRAJ[@]}" "${APPENDIX_TRAJ[@]}"; do
    for seed in "${SEEDS[@]}"; do
      N=$((N+1)); run_one "$cfg" "$traj" "$seed" "$N/$TOTAL"
    done
  done
done
say ""
say "=== MATRIX SUMMARY ($(date)) ==="
cat "$LOG.summary" | tee -a "$LOG"
say "stale: $(grep -c '^STALE' "$LOG.summary"||true)  failed-all-3: $(grep -c '^FAILALL' "$LOG.summary"||true)  banner-bad: $(grep -c 'banner=BAD' "$LOG.summary"||true)"
say "=== M11 MATRIX DONE $(date) ==="
