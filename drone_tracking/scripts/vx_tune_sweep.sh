#!/bin/bash
# vx_tune_sweep.sh — autonomous distance-PID tuning on the CHASE benchmark
# (straight-away target, Config 1 = hardest, no Kalman). Goal: kill the
# 8<->12 m distance surge and hold 8 m smoothly without losing the target.
# Each cell scored on distance std (oscillation), mean (standoff accuracy),
# cmd_vx jitter, and duration/loss. Fixed: exact pitch comp + deriv filter.
set -u
LS=~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh
RUN=~/catkin_ws/src/drone_tracking/scripts/robust_run.sh
SCORE=~/catkin_ws/src/drone_tracking/scripts/score.py
OUT=/tmp/vx_tune_results.txt
: > "$OUT"

# cells: label KP KI KD DLPF
CELLS=(
  "A_kd1.0   2.5 1.2 1.0 0.4"
  "B_kd1.5   2.5 1.2 1.5 0.4"
  "C_kp2kd1  2.0 1.0 1.0 0.4"
  "D_lpf0.3  2.5 1.2 1.0 0.3"
  "E_kp2kd15 2.0 1.0 1.5 0.3"
)

for cell in "${CELLS[@]}"; do
  read lbl KP KI KD DLPF <<< "$cell"
  export VX_MODE=pid KP_VX=$KP KI_VX=$KI KD_VX=$KD D_LPF=$DLPF
  export A_DEC=1.5 D_EMERG=3.0 ALT_FLOOR=11.0 PITCH_COMP=1.0 DERIV_LPF=0.6
  export ALPHA_DIST_K=0.096 SIZE_GATE=off STRAIGHT_AZ=away STRAIGHT_MAX=10000
  export VIEWER=1 LOSS_TIMEOUT=20 START_DIST=8 STUCK_TIMEOUT=8
  CSV=$(bash "$RUN" "vxt_$lbl" 1 3 1 42 120 | tail -1)
  echo "===== $lbl (Kp=$KP Ki=$KI Kd=$KD dlpf=$DLPF) =====" | tee -a "$OUT"
  if [ -f "$CSV" ]; then
    python3 "$SCORE" "$CSV" | tee -a "$OUT"
    python3 - "$CSV" <<PYEOF | tee -a "$OUT"
import csv,statistics as st
rows=[r for r in csv.DictReader(open("$CSV")) if r['phase'] in ('APPROACH','HOLD') and r['true_dist_3d']]
if rows:
    d=[float(r['true_dist_3d']) for r in rows]
    vx=[float(r['cmd_vx']) for r in rows]; dvx=[abs(b-a) for a,b in zip(vx,vx[1:])]
    print("  DIST: mean %.2f std %.2f (osc) range %.1f-%.1f | vx_jit %.2f"%(
      st.mean(d),st.pstdev(d),min(d),max(d),sorted(dvx)[int(.9*len(dvx))] if dvx else 0))
PYEOF
  else
    echo "  FAILED (no CSV)" | tee -a "$OUT"
  fi
done
echo "VX TUNE SWEEP DONE" | tee -a "$OUT"
