#!/bin/bash
# takeoff_gate.sh ZONE SEED... — minimal arm+OFFBOARD-takeoff reliability gate
# at an arbitrary spawn zone (Stage-1b of the occlusion-confound test). No
# trajectory; watches /drone_tracking/takeoff_ready for data:True. Takeoff is
# identical regardless of trajectory/latch (it precedes SEARCH). Reports
# PASS/FAIL per seed. 2 attempts/seed. Config 2 baseline, gains untouched.
set -u
ZONE="${1:?usage: takeoff_gate.sh ZONE SEED...}"; shift
SEEDS="$*"; [ -z "$SEEDS" ] && SEEDS="42 43 45"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEANUP=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
LOG=~/fyp/Results/diagnostics/occlusion/takeoff_gate_z${ZONE}.log
mkdir -p "$(dirname "$LOG")"; : > "$LOG"
say(){ echo "$@" | tee -a "$LOG"; }

probe() {   # seed
  local seed=$1 a
  for a in 1 2; do
    bash "$CLEANUP" >/dev/null 2>&1; sleep 3
    say "--- $(date +%H:%M:%S) zone=$ZONE seed=$seed attempt=$a ---"
    SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
      bash "$LS" 2 7 "$ZONE" "$seed" 8 > "/tmp/tkgate_z${ZONE}_${seed}_${a}.log" 2>&1 &
    local LP=$! ok=0 i
    for i in $(seq 1 220); do
      if ! kill -0 $LP 2>/dev/null; then break; fi
      if timeout 2 rostopic echo -n1 /drone_tracking/takeoff_ready 2>/dev/null | grep -q 'data: True'; then
        ok=1; break; fi
      sleep 1
    done
    kill -INT $LP 2>/dev/null; sleep 2; bash "$CLEANUP" >/dev/null 2>&1
    if [ $ok -eq 1 ]; then say "  PASS seed=$seed (attempt $a)"; return 0; fi
    say "  fail seed=$seed attempt=$a (no takeoff_ready)"
  done
  say "  FAIL seed=$seed (no clean takeoff in 2 attempts)"
  return 1
}

say "TAKEOFF GATE START $(date)  zone=$ZONE seeds={$SEEDS}"
PASS=0; TOT=0
for s in $SEEDS; do TOT=$((TOT+1)); probe "$s" && PASS=$((PASS+1)); done
say ""
say "TAKEOFF GATE RESULT zone=$ZONE : $PASS/$TOT seeds clean"
say "TAKEOFF GATE DONE $(date)"
