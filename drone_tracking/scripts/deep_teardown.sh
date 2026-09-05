#!/usr/bin/env bash
# deep_teardown.sh — thorough sim teardown that RELEASES the GPU render/YOLO context,
# to slow the camera-detection leak WITHOUT a wsl --shutdown. Root cause of the leak is
# partly orphaned yolo_detection_node + gzserver holding GPU context across launches;
# killing them AND waiting for GPU VRAM to actually free recovers det substantially
# (does NOT fully cure the Mesa render leak, but extends unattended running a lot).
# Usage: source/run between launches, or on a det-drop, before relaunching.
echo "[deep_teardown] killing sim + GPU-context holders..."
for pat in yolo_detection_node gzclient gzserver px4 mavros rosmaster roslaunch launch_stack robust_run rl_td3_eval rl_train_td3; do
  pids=$(pgrep -f "$pat" 2>/dev/null)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
done
sleep 3
# wait for GPU VRAM to drain (context released) — up to 40s
for i in $(seq 1 20); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$USED" ] && break
  if [ "$USED" -lt 300 ]; then echo "[deep_teardown] GPU freed (${USED} MiB) after ${i}x2s"; break; fi
  sleep 2
done
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[deep_teardown] done — GPU used=${USED:-?} MiB, procs=$(pgrep -f 'gzserver|px4|rosmaster' | wc -l)"
