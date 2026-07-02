#!/bin/bash
# m12_phaseD_verify.sh — M12 Phase D-prep, steps 3 (seed screen) + 4 (smoke gate).
# NOT the campaign. Two blocks:
#   A) SEED SCREEN: seeds {46,47,48}, Config1, T1, Zone1, 70s — takeoff reliability.
#      PASS = the run got past TAKEOFF (CSV has SEARCH/APPROACH/HOLD rows).
#   B) SMOKE GATE: Config1 then Config2, T3, seed42, Zone1, 200s — validates the
#      new logger columns + metric family end-to-end (analysed separately).
# Hardened settle()+retry, banner check (lambda default + horiz gains untouched).
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
ZONE=1
LOG=~/fyp/Results/diagnostics/m12/phaseD_verify.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"; : > "$LOG.summary"
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
# run CFG TRAJ SEED DUR  -> echoes the saved CSV path (or empty on failure)
run_one(){
  local cfg=$1 traj=$2 seed=$3 dur=$4
  local rdir=~/fyp/Results/Config${cfg}
  local before csv
  before=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1); csv=""
  for attempt in 1 2 3; do
    settle
    [ "$attempt" -gt 1 ] && say "    (retry $attempt/3)"
    SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" "$cfg" "$traj" "$ZONE" "$seed" "$dur" >> "$LOG" 2>&1
    csv=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
    [ -n "$csv" ] && [ "$csv" != "$before" ] && break; csv=""
  done
  echo "$csv"
}
# check banner in the freshest IBVS log for this run tag
banner_ok(){
  local traj=$1 t7log b hz lam latch
  t7log=$(ls -t /tmp/T7_traj${traj}_zone${ZONE}_*.log 2>/dev/null|head -1)
  b=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null|head -1)
  hz=$(echo "$b"|grep -o "Kp_wz=0.90 Kp_y=1.80")
  lam=$(echo "$b"|grep -o "lambda=(1.00,1.00,1.00,1.00)")
  latch=$(echo "$b"|grep -o "search_latch=False")
  [ -n "$hz" ] && [ -n "$lam" ] && [ -n "$latch" ] && echo "ok" || echo "BAD($b)"
}
postko_rows(){ # count SEARCH/APPROACH/HOLD rows in a CSV
  [ -f "$1" ] || { echo 0; return; }
  awk -F, 'NR>1 && ($2=="SEARCH"||$2=="APPROACH"||$2=="HOLD"){n++} END{print n+0}' "$1"
}

say "===== M12 PHASE D VERIFY  START $(date) ====="
say ""
say "----- BLOCK A: SEED SCREEN (Config1, T1, Zone1, 70s) -----"
for seed in 46 47 48; do
  say ">> seed $seed"
  csv=$(run_one 1 1 "$seed" 70)
  if [ -z "$csv" ]; then
    say "   seed $seed: NO CSV (takeoff/bringup failed all 3) -> FAIL"
    echo "SEED $seed FAIL no-csv" >>"$LOG.summary"; continue
  fi
  n=$(postko_rows "$csv"); bk=$(banner_ok 1)
  if [ "$n" -gt 0 ]; then
    say "   seed $seed: PASS (post-takeoff rows=$n, banner=$bk) $(basename "$csv")"
    echo "SEED $seed PASS rows=$n banner=$bk csv=$(basename "$csv")" >>"$LOG.summary"
  else
    say "   seed $seed: FAIL (stuck in TAKEOFF, rows=$n) $(basename "$csv")"
    echo "SEED $seed FAIL stuck-takeoff csv=$(basename "$csv")" >>"$LOG.summary"
  fi
done

say ""
say "----- BLOCK B: SMOKE GATE (T3, seed42, Zone1, 200s) -----"
for cfg in 1 2; do
  say ">> Config $cfg  T3 seed42"
  csv=$(run_one "$cfg" 3 42 200)
  if [ -z "$csv" ]; then
    say "   Config $cfg: NO CSV -> smoke FAIL"; echo "SMOKE C$cfg FAIL no-csv" >>"$LOG.summary"; continue
  fi
  n=$(postko_rows "$csv"); bk=$(banner_ok 3)
  say "   Config $cfg: saved $(basename "$csv")  post-takeoff-rows=$n banner=$bk"
  echo "SMOKE C$cfg csv=$(basename "$csv") rows=$n banner=$bk" >>"$LOG.summary"
done
settle
say ""
say "===== M12 PHASE D VERIFY DONE $(date) ====="
cat "$LOG.summary" | tee -a "$LOG"
echo "=== PHASE_D_VERIFY_DONE ==="
