#!/bin/bash
# vz_cellD.sh — M10.3 vertical sweep, cell D follow-on (the outcome-(c) branch).
# T7 cell C saturated (11.4%) under x2 Kp_z, so per the pre-stated protocol we
# test the cap as the lever: max_vz x1.5 = 2.25, Kp_z back at BASELINE 3.0.
# T7 x seeds {42,43,45} = 6... no, 3 runs (T7 is the saturating trajectory; T6
# never saturated at any cell so the cap is irrelevant there). Config 2, Zone 1.
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
ZONE=1; DURATION=200; CFG=2; TRAJ=7; SEEDS=(42 43 45)
KPZ=3.0; MVZ=2.25
EXPECT_HZ="Kp_wz=0.90 Kp_y=1.80"
LOG=~/fyp/Results/diagnostics/vz_sweep/cellD.log
mkdir -p "$(dirname "$LOG")"
declare -A SEEN_MD5
say(){ echo "$@" | tee -a "$LOG"; }
settle(){
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
: > "$LOG"; : > "$LOG.summary"
say "=== M10.3 VZ cell D (max_vz=$MVZ, Kp_z=$KPZ) T7 x {${SEEDS[*]}}  START $(date) ==="
rdir=~/fyp/Results/Config${CFG}
N=0
for seed in "${SEEDS[@]}"; do
  N=$((N+1)); say ">>> [$N/3] cell D  Kp_z=$KPZ max_vz=$MVZ  T$TRAJ  seed $seed"
  before=$(ls -t ${rdir}/traj${TRAJ}_zone${ZONE}_*.csv 2>/dev/null | head -1); csv=""
  for attempt in 1 2 3; do
    settle
    [ "$attempt" -gt 1 ] && say "  (retry $attempt/3)"
    KP_Z=$KPZ MAX_VZ=$MVZ SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" "$CFG" "$TRAJ" "$ZONE" "$seed" "$DURATION" >> "$LOG" 2>&1
    csv=$(ls -t ${rdir}/traj${TRAJ}_zone${ZONE}_*.csv 2>/dev/null | head -1)
    [ -n "$csv" ] && [ "$csv" != "$before" ] && break; csv=""
  done
  if [ -z "$csv" ]; then say "  !! FAILED all 3 (bringup) D s$seed"; echo "FAILALL D T$TRAJ s$seed" >>"$LOG.summary"; continue; fi
  md5=$(md5sum "$csv"|cut -d' ' -f1); bok="ok"
  [ -n "${SEEN_MD5[$md5]:-}" ] && { say "  !! STALE"; echo "STALE D s$seed">>"$LOG.summary"; } || SEEN_MD5[$md5]=1
  t7log=$(ls -t /tmp/T7_traj${TRAJ}_zone${ZONE}_*.log 2>/dev/null|head -1)
  banner=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null|head -1)
  mvz_b=$(echo "$banner"|grep -o "max_vz=[0-9.]*"); kpz_b=$(echo "$banner"|grep -o "Kp_z=[0-9.]*")
  hz=$(echo "$banner"|grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*"); latch=$(echo "$banner"|grep -o "search_latch=[A-Za-z]*")
  [ "$mvz_b" != "max_vz=2.25" ] && { say "  !! max_vz MISMATCH: '$mvz_b'"; bok="BAD"; }
  [ "$kpz_b" != "Kp_z=3.00" ]   && { say "  !! Kp_z not baseline: '$kpz_b'"; bok="BAD"; }
  [ "$hz" != "$EXPECT_HZ" ]      && { say "  !! horiz gain changed: '$hz'"; bok="BAD"; }
  [ "$latch" != "search_latch=False" ] && { say "  !! latch on"; bok="BAD"; }
  say "  saved $(basename "$csv")  md5=${md5:0:8}  $mvz_b $kpz_b  banner=$bok"
  echo "OK D T$TRAJ s$seed csv=$(basename "$csv") md5=${md5:0:8} mvz=$MVZ banner=$bok" >>"$LOG.summary"
done
say "=== cell D DONE $(date) ==="; cat "$LOG.summary"|tee -a "$LOG"
