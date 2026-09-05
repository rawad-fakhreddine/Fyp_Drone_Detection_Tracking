#!/usr/bin/env bash
# cmp_babysit.sh — overnight babysitter (v2: detects DEATH *and* HANG).
# Every 12 min: if 128 done -> exit. Else if no progress since last check
# (stall) OR ms6_cmp dead -> kill stale procs (by PID) + relaunch cmp_resume.sh.
# If it relaunches 3x with still no progress (a genuinely stuck cell / hard sim
# degradation needing wsl restart) -> STOP and log NEEDS-ATTENTION for the morning.
LEDGER="$HOME/fyp/Results/comparison_baylands/ledger.txt"
LOG="$HOME/fyp/Results/comparison_baylands/babysit.log"
RESUME="$HOME/catkin_ws/src/drone_tracking/scripts/cmp_resume.sh"
echo "$(date '+%m-%d %H:%M') babysitter v2 started" >> "$LOG"
LAST=-1; STALL=0; RELAUNCHES=0
while true; do
  sleep 720
  N=$(grep -cE "^T[0-9]_C[12]_" "$LEDGER" 2>/dev/null || echo 0)
  A=$(pgrep -f ms6_cmp | wc -l)
  echo "$(date '+%m-%d %H:%M') total=${N}/128 ms6_alive=${A} stall=${STALL} relaunches=${RELAUNCHES}" >> "$LOG"
  if [ "$N" -ge 128 ]; then echo "$(date '+%m-%d %H:%M') ALL 128 DONE" >> "$LOG"; break; fi
  if [ "$N" -gt "$LAST" ]; then LAST=$N; STALL=0; RELAUNCHES=0; continue; fi
  # no progress since last check
  STALL=$((STALL+1))
  if [ "$A" -eq 0 ] || [ "$STALL" -ge 2 ]; then
    if [ "$RELAUNCHES" -ge 3 ]; then
      echo "$(date '+%m-%d %H:%M') NEEDS-ATTENTION: stuck at ${N}/128 after 3 relaunches (likely hard sim degradation -> needs wsl --shutdown). babysitter stopping." >> "$LOG"
      break
    fi
    echo "$(date '+%m-%d %H:%M') stalled/dead at ${N} -> kill stale + relaunch" >> "$LOG"
    P=$(ps -eo pid,args | grep -E "gzserver|px4|mavros|rosmaster|roslaunch|launch_stack|robust_run|ms6_cmp" | grep -v grep | awk '{print $1}')
    [ -n "$P" ] && kill -9 $P 2>/dev/null
    sleep 6
    bash "$RESUME" >> "$LOG" 2>&1
    RELAUNCHES=$((RELAUNCHES+1)); STALL=0
  fi
done
