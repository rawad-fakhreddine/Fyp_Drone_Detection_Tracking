#!/bin/bash
# clean_capture.sh TRAJ [SEED] [ZONE] — M10.3 Part B clean-frame capture.
# Brings up the full Config-2 stack headless for ONE trajectory and saves
# UN-annotated camera frames (pristine pixels) + metadata for the offline
# imgsz recovery test. Gains UNTOUCHED; verifies banner Kp_wz=0.90 Kp_y=1.80.
# Outputs: ~/fyp/Results/diagnostics/cap_T<t>/{frames/*.png, clean_capture_meta.csv}
#   bash clean_capture.sh 7        # T7, seed 42, zone 1
set -u
TRAJ="${1:?usage: clean_capture.sh TRAJ [SEED] [ZONE]}"
SEED="${2:-42}"
ZONE="${3:-1}"
DURATION="${DURATION:-200}"
OUTDIR="${OUTDIR:-$HOME/fyp/Results/diagnostics/cap_T${TRAJ}}"
EXPECT_BANNER="Kp_wz=0.90 Kp_y=1.80"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
mkdir -p "$OUTDIR"
RP=""
trap 'echo "[cap] cleanup..."; [ -n "$RP" ] && kill -INT "$RP" 2>/dev/null; sleep 2; bash "$S/cleanup.sh" >/dev/null 2>&1' EXIT

echo "=========================================================="
echo " CLEAN CAPTURE  T$TRAJ  zone=$ZONE seed=$SEED  Config 2  ${DURATION}s"
echo " out -> $OUTDIR"
echo "=========================================================="
bash "$S/cleanup.sh" >/dev/null 2>&1; sleep 3

VIEWER=0 LOSS_TIMEOUT=60 \
  bash "$S/launch_stack.sh" 2 "$TRAJ" "$ZONE" "$SEED" "$DURATION" \
  > "/tmp/cap_T${TRAJ}_z${ZONE}_s${SEED}.log" 2>&1 &
LP=$!

echo "[cap] waiting for camera feed (image_raw)..."
for i in $(seq 1 300); do
  if timeout 3 rostopic echo -n1 /iris/usb_cam/image_raw >/dev/null 2>&1; then break; fi
  if ! kill -0 $LP 2>/dev/null; then
    echo "[cap] stack exited early — see /tmp/cap_T${TRAJ}_z${ZONE}_s${SEED}.log"; exit 1; fi
  sleep 1
done

BAN=$(grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*" \
  "$(ls -t /tmp/T7_traj${TRAJ}_zone${ZONE}_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
if [ -n "$BAN" ] && [ "$BAN" != "$EXPECT_BANNER" ]; then
  echo "[cap] !! BANNER MISMATCH: expected '$EXPECT_BANNER' got '$BAN' — aborting"; exit 1
fi
echo "[cap] banner [$BAN] — starting clean-frame capture"

rosrun drone_tracking clean_capture_record.py --dir "$OUTDIR" \
  > "/tmp/cap_node_T${TRAJ}.log" 2>&1 &
RP=$!

echo "[cap] capturing — waiting for mission to finish..."
wait $LP
echo "[cap] mission done — stopping capture (flush csv)..."
kill -INT "$RP" 2>/dev/null
for i in $(seq 1 15); do kill -0 "$RP" 2>/dev/null || break; sleep 1; done
RP=""

echo "=========================================================="
echo " DONE T$TRAJ"
ls "$OUTDIR/frames" 2>/dev/null | wc -l | xargs echo " frames saved:"
wc -l "$OUTDIR/clean_capture_meta.csv" 2>/dev/null
echo "=========================================================="
