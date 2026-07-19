#!/bin/bash
# rec16.sh — re-record the 16 demo clips (T1-T8 x C1/C2, seed 42, 200 s) on the
# FINAL matrix code. Same scenario env + gates as the official campaign runner.
# Clips -> ~/fyp/Results_reference/07_Demo_Videos/T{t}_C{c}.mp4 (resume-safe:
# existing clips skip). Ledger /tmp/rec16_results.txt.
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
RUN=~/catkin_ws/src/drone_tracking/scripts/robust_run.sh
VID=~/fyp/Results_reference/07_Demo_Videos
RAWVID=~/fyp/Results/demo_videos
OUT=/tmp/rec16_results.txt
touch "$OUT"
export SPAWN_YAW=90 START_DIST=8 VIEWER=0 SNAP=0 RECORD=1 \
       LOSS_TIMEOUT=60 STUCK_TIMEOUT=15

env_for_traj(){
  unset KI_Z INT_Z_MAX INT_Z_BLEED MAX_VZ CHASER_ZDN \
        D_HOLD_MIN D_HOLD_MAX D_STAR STRAIGHT_AZ STRAIGHT_MAX 2>/dev/null
  case $1 in
    2|3) export STRAIGHT_AZ=away STRAIGHT_MAX=99999 ;;
    5)   export D_HOLD_MIN=9.0 D_HOLD_MAX=11.0 D_STAR=10.0 ;;
    6|7) export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 \
                D_HOLD_MIN=9.0 D_HOLD_MAX=11.0 D_STAR=10.0 STRAIGHT_AZ=away ;;
    8)   export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 \
                D_HOLD_MIN=9.0 D_HOLD_MAX=11.0 D_STAR=10.0 ;;
  esac
}

stats(){ python3 - "$1" <<'PY'
import csv,sys
try:
    rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['phase'] in ('SEARCH','APPROACH','HOLD')]
    fov=[r for r in rows if r.get('in_fov')=='1']
    def pct(rs): return round(100*sum(1 for r in rs if r['raw_det']=='REAL')/max(len(rs),1),1)
    n=len(fov); w=min(800,max(n//3,1))
    h=100*sum(1 for r in rows if r['phase']=='HOLD')/max(len(rows),1)
    d=[float(r['true_dist_3d']) for r in rows if r['true_dist_3d'] not in ('','nan')]
    dur=float(rows[-1]['sim_time'])-float(rows[0]['sim_time'])
    flagged=[r for r in rows if r.get('in_fov') in ('0','1')]
    trk=100*len(fov)/max(len(flagged),1)
    print(pct(fov),pct(fov[:w]),pct(fov[-w:]),round(h,1),round(min(d),2),round(dur),round(trk,1))
except Exception: print(-1,-1,-1,-1,-1,-1,-1)
PY
}

for TRAJ in 1 2 3 4 5 6 7 8; do
  for CFG in 1 2; do
    LBL="T${TRAJ}_C${CFG}"
    [ -f "$VID/$LBL.mp4" ] && continue
    env_for_traj "$TRAJ"
    MEM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$MEM" -lt 2500 ]; then
      echo "HALT: memavail ${MEM}MB before $LBL" >> "$OUT"; exit 2
    fi
    CSV=$(bash "$RUN" "rec_$LBL" "$CFG" "$TRAJ" 1 42 200 | tail -1)
    read DET DET1 DET2 THOLD TCLOSE TDUR TTRK <<< "$(stats "$CSV")"
    # newest mp4 for this config+traj (recorder finalizes on stack teardown)
    sleep 3
    MP4=$(ls -t "$RAWVID"/Config${CFG}_traj${TRAJ}_zone1_*.mp4 2>/dev/null | head -1)
    SZ=$([ -n "$MP4" ] && stat -c%s "$MP4" || echo 0)
    echo "$LBL $CSV det=$DET start=$DET1 end=$DET2 hold=$THOLD track=$TTRK closest=$TCLOSE dur=$TDUR mem=${MEM}MB mp4=${MP4##*/} sz=$((SZ/1048576))MB" >> "$OUT"
    if python3 -c "exit(0 if (0<=$DET2<=$DET1-25) or (0<=$DET<60) else 1)"; then
      echo "HALT: degradation canary ($DET / $DET1->$DET2) at $LBL" >> "$OUT"; exit 3
    fi
    if python3 -c "exit(0 if (0<=$TTRK<85) or (0<$TDUR<195) or (0<$TCLOSE<1.5) else 1)"; then
      echo "HALT-FOR-DISCUSSION: $LBL tracking failure (track=$TTRK closest=$TCLOSE dur=${TDUR}s)" >> "$OUT"; exit 5
    fi
    if [ "$SZ" -lt 1048576 ]; then
      echo "HALT: mp4 missing/tiny for $LBL ($MP4)" >> "$OUT"; exit 6
    fi
    mv "$MP4" "$VID/$LBL.mp4"
  done
done
echo "REC16 DONE" >> "$OUT"
bash ~/catkin_ws/src/drone_tracking/scripts/cleanup.sh >/dev/null 2>&1
cat "$OUT"
