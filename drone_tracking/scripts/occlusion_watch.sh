#!/bin/bash
# occlusion_watch.sh TRAJ [SEED] [ZONE] — M10.3 single interactive watch run.
# Brings up the full Config-2 stack for ONE hard trajectory with the GT-OVERLAY
# viewer (YOLO box + ground-truth marker that survives YOLO dropout) in the
# FOREGROUND so you watch the loss/abort live, one at a time. Gains UNTOUCHED;
# verifies the baseline banner Kp_wz=0.90 Kp_y=1.80 and latch OFF before
# handing you the window. Optional RECORD=/path/feed.mp4 to scrub afterwards.
#   bash occlusion_watch.sh 3            # T3, seed 42, zone 1
#   RECORD=~/t6.mp4 bash occlusion_watch.sh 6
# Press Q in the window to end the watch; cleanup runs on exit.
set -u
TRAJ="${1:?usage: occlusion_watch.sh TRAJ [SEED] [ZONE]}"
SEED="${2:-42}"
ZONE="${3:-1}"
DURATION="${DURATION:-200}"
RECORD="${RECORD:-}"
EXPECT_BANNER="Kp_wz=0.90 Kp_y=1.80"
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
trap 'echo "[watch] cleanup..."; bash "$S/cleanup.sh" >/dev/null 2>&1' EXIT

echo "=========================================================="
echo " WATCH  T$TRAJ  zone=$ZONE seed=$SEED  Config 2  ${DURATION}s"
echo " GT overlay ON (magenta=ground-truth, green=YOLO)  latch OFF"
[ -n "$RECORD" ] && echo " recording -> $RECORD"
echo "=========================================================="
bash "$S/cleanup.sh" >/dev/null 2>&1; sleep 3

# Stack WITHOUT its own viewer (VIEWER=0); we run the gt-overlay viewer instead.
VIEWER=0 LOSS_TIMEOUT=60 \
  bash "$S/launch_stack.sh" 2 "$TRAJ" "$ZONE" "$SEED" "$DURATION" \
  > "/tmp/watch_T${TRAJ}_z${ZONE}_s${SEED}.log" 2>&1 &
LP=$!

# wait for the camera stream (and verify the stack didn't die)
echo "[watch] waiting for camera feed..."
for i in $(seq 1 260); do
  if timeout 3 rostopic echo -n1 /drone_tracking/target_center >/dev/null 2>&1; then break; fi
  if ! kill -0 $LP 2>/dev/null; then echo "[watch] stack exited early — see /tmp/watch_T${TRAJ}_z${ZONE}_s${SEED}.log"; exit 1; fi
  sleep 1
done

# banner verify (baseline gains, no treatment)
BAN=$(grep -o "Kp_wz=[0-9.]* Kp_y=[0-9.]*" \
  "$(ls -t /tmp/T7_traj${TRAJ}_zone${ZONE}_*.log 2>/dev/null | head -1)" 2>/dev/null | head -1)
if [ -n "$BAN" ] && [ "$BAN" != "$EXPECT_BANNER" ]; then
  echo "[watch] !! BANNER MISMATCH: expected baseline '$EXPECT_BANNER' got '$BAN' — aborting"; exit 1
fi
echo "[watch] banner OK [$BAN] — opening overlay window (press Q to end)"

# GT-overlay viewer in the FOREGROUND (this blocks until you press Q or run ends)
REC_ARG=""; [ -n "$RECORD" ] && REC_ARG="--record $RECORD"
rosrun drone_tracking gt_overlay_viewer.py $REC_ARG

echo "[watch] window closed."
