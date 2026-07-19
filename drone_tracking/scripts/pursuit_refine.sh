#!/bin/bash
# pursuit_refine.sh — refine the vx pursuit on the CLEAN scenario (north-open,
# straight one-way target, no trees, no reversal). Big fixes already in (gains
# Kp2.5/Ki1.2, yaw-dominant, pitch de-rotation, slew). Goal: smooth the residual
# HOLD vx swing (from noisy raw d_hat) while keeping the gap closed in 6-8 m.
# Cells vary d_lpf (d_hat smoothing), Kd_vx (damping), and Config (Kalman).
set -u
RUN=~/catkin_ws/src/drone_tracking/scripts/robust_run.sh
SCORE=~/catkin_ws/src/drone_tracking/scripts/score.py
OUT=/tmp/pursuit_refine_results.txt
: > "$OUT"

# label CFG D_LPF KD_VX
CELLS=(
  "A_base    1 0.4  1.5"
  "B_lpf55   1 0.55 1.5"
  "C_lpf6kd1 1 0.6  1.0"
  "D_lpf5kd2 1 0.5  2.0"
  "E_C2kal   2 0.4  1.5"
)
for cell in "${CELLS[@]}"; do
  read lbl CFG DLPF KD <<< "$cell"
  export VX_MODE=pid D_LPF=$DLPF KD_VX=$KD A_DEC=2.5 ALT_FLOOR=11.0 PITCH_COMP=1.0 DERIV_LPF=0.6
  export MAX_ACCEL=3.0 MAX_ACCEL_FAST=8.0 KP_VX=2.5 KI_VX=1.2 KP_WZ=2.0 KP_Y=0.4
  export D_HOLD_MIN=6.0 D_HOLD_MAX=8.0 MIN_DIST=4.0 EMERG_VY=2.0 EMERG_BRAKE_VX=4.0 MAX_VX=8.0
  export ALPHA_DIST_K=0.096 SIZE_GATE=off STRAIGHT_AZ=away STRAIGHT_MAX=99999 STRAIGHT_OFFSET=0 SPAWN_YAW=90
  export VIEWER=1 LOSS_TIMEOUT=40 START_DIST=7 STUCK_TIMEOUT=15 SNAP=0
  CSV=$(bash "$RUN" "ref_$lbl" "$CFG" 3 1 42 60 | tail -1)
  echo "===== $lbl (Cfg$CFG d_lpf=$DLPF Kd=$KD) =====" | tee -a "$OUT"
  if [ -f "$CSV" ]; then
    python3 - "$CSV" <<PYEOF | tee -a "$OUT"
import csv,statistics as st
rows=[r for r in csv.DictReader(open("$CSV")) if r['phase'] in ('APPROACH','HOLD') and r['true_dist_3d']]
if rows:
    d=[float(r['true_dist_3d']) for r in rows]
    vx=[float(r['cmd_vx']) for r in rows]; dvx=[abs(b-a) for a,b in zip(vx,vx[1:])]
    inb=100*sum(1 for x in d if 6<=x<=8)/len(d)
    # settled window (2nd half) std = smoothness
    half=d[len(d)//2:]
    print("  dist mean %.1f std %.2f | in6-8 %.0f%% | vx swing(std) %.2f p90step %.2f | closest %.1f"%(
      st.mean(d),st.pstdev(d),inb,st.pstdev(vx),sorted(dvx)[int(.9*len(dvx))] if dvx else 0,min(d)))
PYEOF
  else echo "  FAILED" | tee -a "$OUT"; fi
done
echo "REFINE DONE" | tee -a "$OUT"
