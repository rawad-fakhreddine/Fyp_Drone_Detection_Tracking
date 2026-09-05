#!/bin/bash
# kp_vx_sweep.sh — Kp_vx direction-finder for the v6.31 distance-domain vx PID.
# T3 / zone 1 / seed 42, Configs 1+2, fixed mission knobs from the 2026-07-07
# session: START_DIST=8, ALPHA_DIST_K=0.084 (Gazebo-calibrated), SIZE_GATE=off,
# LOSS_TIMEOUT=10 (fail-fast testing per Rawad). Kp=2.0 cells already flown
# (run 5 C1 / run 6 C2); this adds Kp {3.0, 4.0} x {C1, C2}.
# After the best Kp is picked: PI test via KI_VX at that Kp (separate runs).
set -u
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
CLEAN=~/catkin_ws/src/drone_tracking/scripts/cleanup.sh
PROG=/tmp/kp_vx_sweep_progress.log
: > "$PROG"
log(){ echo "$@" | tee -a "$PROG"; }

log "KP_VX SWEEP START $(date)  cells: KP{3.0,4.0} x C{1,2}"
for KP in 3.0 4.0; do
  for CFG in 1 2; do
    log ""
    log "=== $(date +%H:%M:%S)  KP=$KP  Config $CFG ==="
    bash "$CLEAN" >/dev/null 2>&1; sleep 3
    VIEWER=1 LOSS_TIMEOUT=10 START_DIST=8 VX_MODE=pid KP_VX=$KP \
      ALPHA_DIST_K=0.084 SIZE_GATE=off \
      bash "$LS" "$CFG" 3 1 42 300 > "/tmp/kpvx_KP${KP}_C${CFG}.log" 2>&1
    log "  $(grep -E 'ABORTED at|Mission complete' "/tmp/kpvx_KP${KP}_C${CFG}.log" | tail -1)"
    log "  $(grep -E '^\[Save\]' "/tmp/kpvx_KP${KP}_C${CFG}.log" | tail -1)"
  done
done
bash "$CLEAN" >/dev/null 2>&1
log ""
log "SWEEP DONE $(date)"
