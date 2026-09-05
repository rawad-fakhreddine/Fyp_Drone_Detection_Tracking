#!/bin/bash
# run_estimation.sh CONFIG TRAJ ZONE SEED RLABEL
# One flight + per-frame estimation logger; CSV -> ~/fyp/Results/diagnostics/Reval/
set -u
CONFIG=$1; TRAJ=$2; ZONE=$3; SEED=$4; RLABEL=$5
OUTDIR=~/fyp/Results/diagnostics/Reval
mkdir -p "$OUTDIR"
OUTCSV="$OUTDIR/T${TRAJ}_s${SEED}_${RLABEL}.csv"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

if [ "$CONFIG" = "1" ]; then TRIG=/drone_tracking/target_center; TMODE=raw
else                        TRIG=/drone_tracking/filtered_target; TMODE=filtered; fi

echo "[est-wrap] CONFIG=$CONFIG T$TRAJ Z$ZONE s$SEED R=$RLABEL trigger=$TMODE -> $OUTCSV"
LLOG="/tmp/estlaunch_T${TRAJ}_s${SEED}_${RLABEL}.log"
GLOG="/tmp/estlogger_T${TRAJ}_s${SEED}_${RLABEL}.log"

VIEWER=0 LOSS_TIMEOUT=60 bash ~/catkin_ws/src/drone_tracking/scripts/launch_stack.sh \
    $CONFIG $TRAJ $ZONE $SEED > "$LLOG" 2>&1 &
LP=$!

echo -n "[est-wrap] waiting for $TRIG "
for i in $(seq 1 220); do
    if timeout 3 rostopic echo -n1 "$TRIG" >/dev/null 2>&1; then echo " live"; break; fi
    if ! kill -0 $LP 2>/dev/null; then echo " (launch exited early!)"; break; fi
    sleep 1; echo -n .
done

rosrun drone_tracking estimation_logger.py --log "$OUTCSV" --trigger $TMODE > "$GLOG" 2>&1 &
GP=$!
echo "[est-wrap] logger pid $GP"

wait $LP                               # launch_stack: mission then cleanup (kills logger)
kill -INT $GP 2>/dev/null || true; sleep 2     # belt-and-suspenders flush
echo "[est-wrap] DONE $OUTCSV lines=$(wc -l < "$OUTCSV" 2>/dev/null)"
grep -h "intrinsics" "$GLOG" 2>/dev/null | head -1
exit 0
