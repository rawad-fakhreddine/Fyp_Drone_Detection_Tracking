#!/bin/bash
# occlusion_record.sh TRAJ [SEED] [ZONE] — M10.3 HEADLESS GT-overlay recording.
# Brings up the full Config-2 stack for ONE hard trajectory, records the
# GT-overlay (YOLO box + ground-truth marker that survives YOLO dropout) to mp4
# and dumps the per-frame overlay CSV — no live window (agent is headless).
# Gains UNTOUCHED; verifies the baseline banner Kp_wz=0.90 Kp_y=1.80 (latch OFF)
# before recording. Outputs:
#   ~/fyp/Results/diagnostics/watch_T<t>.mp4
#   ~/fyp/Results/diagnostics/watch_T<t>_frames.csv
#   bash occlusion_record.sh 3        # T3, seed 42, zone 1
set -u
TRAJ="${1:?usage: occlusion_record.sh TRAJ [SEED] [ZONE]}"
SEED="${2:-42}"
ZONE="${3:-1}"
DURATION="${DURATION:-200}"
OUTDIR="${OUTDIR:-$HOME/fyp/Results/diagnostics}"
MP4="$OUTDIR/watch_T${TRAJ}.mp4"
CSV="$OUTDIR/watch_T${TRAJ}_frames.csv"
EXPECT_BANNER="Kp_wz=0.90 Kp_y=1.80"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
mkdir -p "$OUTDIR"
RP=""
trap 'echo "[rec] cleanup..."; [ -n "$RP" ] && kill -INT "$RP" 2>/dev/null; sleep 2; bash "$S/cleanup.sh" >/dev/null 2>&1' EXIT

echo "=========================================================="
echo " RECORD  T$TRAJ  zone=$ZONE seed=$SEED  Config 2  ${DURATION}s"
echo " mp4 -> $MP4"
echo " csv -> $CSV"
echo "=========================================================="
bash "$S/cleanup.sh" >/dev/null 2>&1; sleep 3

# Stack WITHOUT its own viewer (VIEWER=0). LOSS_TIMEOUT=60 = recorded-outcome abort.
VIEWER=0 LOSS_TIMEOUT=60 \
  bash "$S/launch_stack.sh" 2 "$TRAJ" "$ZONE" "$SEED" "$DURATION" \
  > "/tmp/rec_T${TRAJ}_z${ZONE}_s${SEED}.log" 2>&1 &
LP=$!

# wait for camera feed (and verify the stack didn't die)
echo "[rec] waiting for camera feed (image_raw)..."
for i in $(seq 1 300); do
  if timeout 3 rostopic echo -n1 /iris/usb_cam/image_raw >/dev/null 2>&1; then break; fi
  if ! kill -0 $LP 2>/dev/null; then
    echo "[rec] stack exited early — see /tmp/rec_T${TRAJ}_z${ZONE}_s${SEED}.log"; exit 1; fi
  sleep 1
done

# banner verify (baseline gains, no treatment)
BAN=$(grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*" \
  "$(ls -t /tmp/T7_traj${TRAJ}_zone${ZONE}_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
if [ -n "$BAN" ] && [ "$BAN" != "$EXPECT_BANNER" ]; then
  echo "[rec] !! BANNER MISMATCH: expected '$EXPECT_BANNER' got '$BAN' — aborting"; exit 1
fi
echo "[rec] banner [$BAN] — starting headless recorder"

# Headless recorder in the BACKGROUND; it runs until we SIGINT it.
rosrun drone_tracking gt_overlay_record.py --record "$MP4" --csv "$CSV" \
  > "/tmp/rec_node_T${TRAJ}.log" 2>&1 &
RP=$!

# Wait for the full mission (launch_stack runs the mission then self-cleans).
echo "[rec] recording — waiting for mission to finish..."
wait $LP
echo "[rec] mission done — stopping recorder (flush mp4+csv)..."
kill -INT "$RP" 2>/dev/null
# give the recorder time to release the VideoWriter before teardown
for i in $(seq 1 15); do kill -0 "$RP" 2>/dev/null || break; sleep 1; done
RP=""

echo "=========================================================="
echo " DONE T$TRAJ"
ls -la "$MP4" "$CSV" 2>/dev/null
echo "=========================================================="
