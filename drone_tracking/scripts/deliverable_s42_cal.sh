#!/bin/bash
# deliverable_s42_cal.sh — record all 16 (T1-8 x C1/C2, seed 42, 150 s) on the
# CALIBRATED baseline with the 3-panel MULTIVIEW recording, for the per-traj
# deliverable (mp4 + 3D PNG + Excel). All fixes: k=0.077 + d_lpf=0.5 + uniform
# [6,7] standoff are BAKED defaults; Option B (T6/7) + vertical package (T6/7/8)
# + SPAWN_YAW + traj_track_kp come from the env below. Resume-safe (skip existing
# staged clips). Ledger /tmp/deliverable_s42_results.txt.
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
RUN=~/catkin_ws/src/drone_tracking/scripts/robust_run.sh
STAGE=~/fyp/Results/deliverable_s42
MVDIR=~/fyp/Results/multiview
mkdir -p "$STAGE" "$MVDIR"
LED=/tmp/deliverable_s42_results.txt ; touch "$LED"
MAN="$STAGE/csv_manifest.txt" ; touch "$MAN"
export SPAWN_YAW=90 START_DIST=8 VIEWER=0 SNAP=0 SPECTATOR=1 SPEC_TRACK=1 MULTIVIEW=1 \
       MULTIVIEW_FPS=40 TRAJ_TRACK_KP=1.0 TRAJ_TRACK_KP_Z=2.0 TRAJ_TRACK_CAP=2.5 \
       LEMN_A=13 Z_HOLD=1 LOSS_TIMEOUT=60 STUCK_TIMEOUT=15

env_for_traj(){
  unset KI_Z INT_Z_MAX INT_Z_BLEED MAX_VZ CHASER_ZDN KP_Z MAX_ACCEL_VZ \
        STRAIGHT_AZ STRAIGHT_MAX INCLINE_NO_HREVERSE 2>/dev/null
  export SPAWN_YAW=90
  case $1 in
    2|3) export SPAWN_YAW=96 STRAIGHT_AZ=away STRAIGHT_MAX=99999 ;;
    6|7) export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 \
                KP_Z=6.0 MAX_ACCEL_VZ=6.0 INCLINE_NO_HREVERSE=1 STRAIGHT_AZ=away ;;
    8)   export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 \
                KP_Z=6.0 MAX_ACCEL_VZ=6.0 ;;
  esac
}

for TRAJ in 1 2 3 4 5 6 7 8; do
  for CFG in 1 2; do
    LBL="T${TRAJ}_C${CFG}"
    [ -f "$STAGE/$LBL.mp4" ] && { echo "skip $LBL (exists)" >> "$LED"; continue; }
    env_for_traj "$TRAJ"
    MEM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$MEM" -lt 2400 ]; then
      echo "HALT: memavail ${MEM}MB before $LBL (restart WSL + resume)" >> "$LED"; exit 2; fi
    CSV=$(bash "$RUN" "dl_$LBL" "$CFG" "$TRAJ" 1 42 150 | tail -1)
    sleep 3
    MP4=$(ls -t "$MVDIR"/multiview_Config${CFG}_traj${TRAJ}_zone1_*.mp4 2>/dev/null | head -1)
    SZ=$([ -n "$MP4" ] && stat -c%s "$MP4" || echo 0)
    echo "$LBL CSV=$CSV mp4=${MP4##*/} sz=$((SZ/1048576))MB mem=${MEM}MB" >> "$LED"
    [ "$SZ" -ge 1048576 ] && mv "$MP4" "$STAGE/$LBL.mp4"
    echo "$LBL $CSV" >> "$MAN"
  done
done
echo "DELIVERABLE DONE" >> "$LED"
bash ~/catkin_ws/src/drone_tracking/scripts/cleanup.sh >/dev/null 2>&1
