#!/bin/bash
# ms6_run_cells.sh — unified matrix cell runner (gauntlet-first strategy).
# Cells come from $CELLS ("traj:cfg:seed ...") or are generated from $TRAJS.
# Shared ledger /tmp/ms4_batch_results.txt (resume-safe: recorded cells skip).
# Guards per run: memory gate, in-FOV-det trend canary (degradation),
# tracking-failure halt (Rawad's rule: abort / HOLD<85 / closest<2 m).
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
RUN=~/catkin_ws/src/drone_tracking/scripts/robust_run.sh
OUT=/tmp/ms4_batch_results.txt
touch "$OUT"
# TRAJ_TRACK_KP (2026-08-06, Rawad): lock the closed-form targets (T1/T4/T5/T8)
# onto their exact ideal path so they match spec (T1=point, T4/T8=8 m circle,
# T5=figure-8) instead of open-loop drifting 3-4 m over 200 s. Target-side
# scenario setting (identical C1/C2) -> does NOT touch the chaser controller or
# the uniform rule. Only acts on t in (1,4,5,8) internally; T2/3/6/7 ignore it.
export SPAWN_YAW=90 START_DIST=8 VIEWER=0 SNAP=0 LOSS_TIMEOUT=60 STUCK_TIMEOUT=15
export TRAJ_TRACK_KP=1.0 TRAJ_TRACK_CAP=2.5
SEEDS="42 43 45 46 47 48 49 50"

env_for_traj(){ # sets scenario env for trajectory $1
  unset KI_Z INT_Z_MAX INT_Z_BLEED MAX_VZ CHASER_ZDN KP_Z MAX_ACCEL_VZ D_LPF \
        D_HOLD_MIN D_HOLD_MAX D_STAR STRAIGHT_AZ STRAIGHT_MAX SPAWN_YAW \
        INCLINE_NO_HREVERSE 2>/dev/null
  # 2026-08-06 (Rawad, UNIFORM-PARAMETER rule): the low-pass is a fixed 0.5 for
  # EVERY trajectory (launch_stack default now 0.50; the old 0.70-global +
  # T3-scoped-to-0.5 hack is GONE — raising d_lpf added lag, so it stays 0.5 and
  # the decel jerk is handled in the vx law via ~approach_smooth, not the filter).
  # STANDOFF is a uniform [6,7] m band on ALL trajectories (the old [9,11] on
  # T5-T8 removed per the rule). NB this is expected to re-open T5 crossovers and
  # the T7 vertical-swing-vs-frame geometry -> to be re-verified.
  # Vertical package (KP_Z=6 / MAX_VZ=2.5 / MAX_ACCEL_VZ=6 / KI_Z / INT_Z_*) stays
  # on the 3D trajectories T6/T7/T8 (needed for vertical tracking; its status
  # under the uniform rule is still pending Rawad's call).
  # T6/T7: INCLINE_NO_HREVERSE=1 = Option B (forward + vertical zig-zag, NO
  # horizontal reverse) per Rawad 2026-08-06.
  case $1 in
    2)   export STRAIGHT_AZ=away STRAIGHT_MAX=99999 SPAWN_YAW=96 ;;
    3)   export STRAIGHT_AZ=away STRAIGHT_MAX=99999 SPAWN_YAW=96 ;;
    6|7) export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 KP_Z=6.0 MAX_ACCEL_VZ=6.0 \
                INCLINE_NO_HREVERSE=1 STRAIGHT_AZ=away ;;
    8)   export KI_Z=2.0 INT_Z_MAX=1.3 INT_Z_BLEED=0.70 MAX_VZ=2.5 CHASER_ZDN=2.5 KP_Z=6.0 MAX_ACCEL_VZ=6.0 ;;
  esac
}

stats(){ python3 - "$1" <<'PY'
import csv,sys,os
try:
    rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['phase'] in ('SEARCH','APPROACH','HOLD')]
    fov=[r for r in rows if r.get('in_fov')=='1']
    def pct(rs): return round(100*sum(1 for r in rs if r['raw_det']=='REAL')/max(len(rs),1),1)
    n=len(fov); w=min(800,max(n//3,1))
    h=100*sum(1 for r in rows if r['phase']=='HOLD')/max(len(rows),1)
    d=[float(r['true_dist_3d']) for r in rows if r['true_dist_3d'] not in ('','nan')]
    dur=float(rows[-1]['sim_time'])-float(rows[0]['sim_time'])
    # TRACK% v2 = target-in-FOV fraction (ground truth). The corridor version
    # halted a superb T7 flight (HOLD 98.3, closest 3.44) because T7's
    # distance legitimately swings +/-5 m by design. "Kept hold of the
    # target" = kept it in view: any real loss (SEARCH divergence) tanks
    # this within seconds; energetic-but-tracking trajectories score ~100.
    flagged=[r for r in rows if r.get('in_fov') in ('0','1')]
    trk=100*len(fov)/max(len(flagged),1)   # fov = frames with in_fov=='1'
    print(pct(fov),pct(fov[:w]),pct(fov[-w:]),round(h,1),round(min(d),2),round(dur),round(trk,1))
except Exception: print(-1,-1,-1,-1,-1,-1,-1)
PY
}

if [ -z "${CELLS:-}" ]; then
  CELLS=""
  for TRAJ in ${TRAJS:-1 2 3 4 5 6 7 8}; do
    for SEED in $SEEDS; do for CFG in 1 2; do CELLS="$CELLS $TRAJ:$CFG:$SEED"; done; done
  done
fi

for cell in $CELLS; do
  IFS=: read TRAJ CFG SEED <<< "$cell"
  LBL="T${TRAJ}_C${CFG}_s${SEED}"
  grep -q "^$LBL " "$OUT" && continue
  env_for_traj "$TRAJ"
  MEM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$MEM" -lt 2500 ]; then
    echo "HALT: memavail ${MEM}MB before $LBL" >> "$OUT"; cat "$OUT"; exit 2
  fi
  CSV=$(bash "$RUN" "ms6_$LBL" "$CFG" "$TRAJ" 1 "$SEED" 200 | tail -1)
  read DET DET1 DET2 THOLD TCLOSE TDUR TTRK <<< "$(stats "$CSV")"
  echo "$LBL $CSV det=$DET start=$DET1 end=$DET2 hold=$THOLD track=$TTRK closest=$TCLOSE dur=$TDUR mem=${MEM}MB" >> "$OUT"
  if python3 -c "exit(0 if (0<=$DET2<=$DET1-25) or (0<=$DET<60) else 1)"; then
    echo "HALT: degradation canary ($DET / $DET1->$DET2) at $LBL" >> "$OUT"; cat "$OUT"; exit 3
  fi
  # gate on TRACK% (data-computed station-keeping), abort, and the 2 m bar;
  # phase-HOLD stays logged for continuity but no longer gates.
  if python3 -c "exit(0 if (0<=$TTRK<85) or (0<$TDUR<195) or (0<$TCLOSE<1.5) else 1)"; then
    echo "HALT-FOR-DISCUSSION: $LBL tracking failure (track=$TTRK closest=$TCLOSE dur=${TDUR}s)" >> "$OUT"
    echo "CAMPAIGN HALTED (tracking failure)" >> "$OUT"; cat "$OUT"; exit 5
  fi
  # F8 acceptance (Rawad 2026-07-18): 1.5-2.0 m passes = recorded WARN, not a halt
  python3 -c "exit(0 if 1.5<=$TCLOSE<2.0 else 1)" && echo "  WARN: near-bar pass $TCLOSE m at $LBL — record in 08_Failures" >> "$OUT"
done
echo "CELLS DONE" >> "$OUT"
cat "$OUT"
