#!/bin/bash
# gate_zone1_takeoff.sh — M11 pre-flight gate: Zone-1 takeoff + banner dry-run.
# Probes BOTH configs x seeds {42,43,45} at Zone 1. For each: brings up the
# full stack (T7, short mission), watches /drone_tracking/takeoff_ready for
# data:True, and captures the IBVS startup banner from the T7 log (the dry-run
# banner check: v6.30 | ... Kp_wz=0.90 Kp_y=1.80 | search_latch=False).
# Takeoff happens BEFORE SEARCH so trajectory is irrelevant; we still probe
# both configs because the prompt flags Config 1 may differ. Records OK/FAIL +
# banner per cell; non-destructive (kills+cleans after each). Probes ALL 6 and
# reports — a FAIL is flagged, not silently skipped.
#   bash gate_zone1_takeoff.sh
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"; CLEANUP="$S/cleanup.sh"
ZONE=1; TRAJ=7; DURATION=8
LOG=~/fyp/Results/diagnostics/gate_c1c2/takeoff_gate.log
mkdir -p "$(dirname "$LOG")"; : > "$LOG"
say(){ echo "$@" | tee -a "$LOG"; }
EXPECT="Kp_wz=0.90 Kp_y=1.80"

# Inter-run settle: WSL2 has only ~7.6 GB; a fresh Gazebo bringup overlapping
# the previous stack's teardown blows past RAM -> kernel SIGKILLs the new
# roslaunch during MAVROS init (the 3/6 alternating failures). Clean, then
# wait until heavy procs are gone AND RAM has recovered, then a buffer.
settle(){
  bash "$CLEANUP" >/dev/null 2>&1
  local i procs avail
  for i in $(seq 1 60); do
    procs=$(pgrep -c -f 'gzserver|gzclient|px4|mavros' 2>/dev/null||true)
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${procs:-0}" -eq 0 ] && [ "${avail:-0}" -ge 4000 ]; then break; fi
    sleep 2
  done
  sleep 4
}

probe(){  # config seed
  local cfg=$1 seed=$2 ok=0 i banner=""
  settle
  say "--- $(date +%H:%M:%S)  Config $cfg  seed $seed  Zone $ZONE ---"
  SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
    bash "$LS" "$cfg" "$TRAJ" "$ZONE" "$seed" "$DURATION" \
    > "/tmp/gate_c${cfg}_s${seed}.log" 2>&1 &
  local LP=$!
  for i in $(seq 1 240); do
    if ! kill -0 $LP 2>/dev/null; then break; fi
    if timeout 2 rostopic echo -n1 /drone_tracking/takeoff_ready 2>/dev/null | grep -q 'data: True'; then
      ok=1; break; fi
    sleep 1
  done
  # stop the run first so Python flushes stdout on exit, THEN read the banner
  kill -INT $LP 2>/dev/null; sleep 3
  local t7log
  t7log=$(ls -t /tmp/T7_traj${TRAJ}_zone${ZONE}_*.log 2>/dev/null | head -1)
  [ -n "$t7log" ] && banner=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null | head -1)
  bash "$CLEANUP" >/dev/null 2>&1
  local kp; kp=$(echo "$banner" | grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*")
  local latch; latch=$(echo "$banner" | grep -o "search_latch=[A-Za-z]*")
  if [ $ok -eq 1 ]; then
    say "  TAKEOFF OK    | banner: $banner"
    if [ "$kp" != "$EXPECT" ]; then say "  !! BANNER Kp MISMATCH: got '$kp' expected '$EXPECT'"; fi
    if [ "$latch" != "search_latch=False" ]; then say "  !! LATCH NOT OFF: got '$latch'"; fi
    echo "OK   c$cfg s$seed  $kp  $latch" >> "$LOG.summary"
  else
    say "  TAKEOFF FAIL  | banner: ${banner:-<none>}  (see /tmp/gate_c${cfg}_s${seed}.log)"
    echo "FAIL c$cfg s$seed" >> "$LOG.summary"
  fi
}

: > "$LOG.summary"
say "=== Zone-1 takeoff gate START $(date) ==="
for cfg in 1 2; do
  for seed in 42 43 45; do
    probe "$cfg" "$seed"
  done
done
say ""
say "=== GATE SUMMARY ==="
cat "$LOG.summary" | tee -a "$LOG"
FAILS=$(grep -c '^FAIL' "$LOG.summary"||true)
say "fails: $FAILS / 6"
say "=== Zone-1 takeoff gate DONE $(date) ==="
