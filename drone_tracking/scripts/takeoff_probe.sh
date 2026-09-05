#!/bin/bash
# takeoff_probe.sh — find a reliably-takeoff-able 3rd seed in Zone 5.
# Launches the stack (T7 z5), watches /drone_tracking/takeoff_ready for
# data:True, records OK/FAIL, kills+cleans. Takeoff is identical regardless of
# trajectory or search_latch (it happens before SEARCH), so one probe per seed
# settles it for BOTH T7 and T9. Prefers seed 44 (continuity); substitutes
# 45/46 if 44 can't take off cleanly.
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
LOG=~/fyp/Results/diagnostics/search_latch/takeoff_probe.log
mkdir -p "$(dirname "$LOG")"; : > "$LOG"
say(){ echo "$@" | tee -a "$LOG"; }

probe() {   # seed attempts
  local seed=$1 attempts=$2 a
  for a in $(seq 1 "$attempts"); do
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    say "--- $(date +%H:%M:%S) probe seed=$seed attempt=$a ---"
    SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" 2 7 5 "$seed" 8 > "/tmp/probe_${seed}_${a}.log" 2>&1 &
    local LP=$! ok=0 i
    for i in $(seq 1 220); do
      if ! kill -0 $LP 2>/dev/null; then break; fi   # launch_stack exited (TIMEOUT->exit1, or finished)
      if timeout 2 rostopic echo -n1 /drone_tracking/takeoff_ready 2>/dev/null | grep -q 'data: True'; then
        ok=1; break; fi
      sleep 1
    done
    kill -INT $LP 2>/dev/null; sleep 2; bash "$CLEANUP" >/dev/null 2>&1
    if [ $ok -eq 1 ]; then say "  OK seed=$seed took off (attempt $a)"; return 0; fi
    say "  FAIL seed=$seed attempt=$a (no takeoff_ready)"
  done
  return 1
}

say "PROBE START $(date)"
THIRD=""
for s in 44 45 46; do
  if probe "$s" 2; then THIRD=$s; say ">>> SELECTED third seed = $s"; break; fi
  say ">>> seed $s rejected (no clean takeoff in 2 attempts)"
done
[ -z "$THIRD" ] && say ">>> NO clean third seed among 44/45/46"
say "PROBE DONE $(date)  THIRD=$THIRD"
