#!/usr/bin/env bash
# rl_sac_train_launch.sh — one-shot SAC training launcher (Config 3, nolockstep)
#
# Usage:
#   SPEED_FACTOR=4 bash rl_sac_train_launch.sh [STEPS] [TRAJ] [SEED]
#
# Defaults: STEPS=30000  TRAJ=4 (orbit)  SEED=42  SPEED_FACTOR=4
#
# What it does:
#   1. Launches the full sim stack (nolockstep, SKIP_IBVS=1, long DURATION)
#   2. Waits for both drones to be at altitude
#   3. Starts rl_train_sac.py --scratch for STEPS env steps
#   4. Cleans up the stack when SAC exits
set -e
STEPS="${1:-30000}"
TRAJ="${2:-4}"
SEED="${3:-42}"
SPEED_FACTOR="${SPEED_FACTOR:-4}"
WORLD="${WORLD:-rl_empty}"

source /opt/ros/noetic/setup.bash
source /home/rawad/catkin_ws/devel/setup.bash

SAC_SAVE="${SAC_SAVE:-~/fyp/rl/models/sac}"
BC_PATH="${BC_PATH:-~/fyp/rl/models/bc_policy_v3.pth}"

echo "================================================================"
echo "  SAC Training | traj=$TRAJ seed=$SEED steps=$STEPS | ${SPEED_FACTOR}x nolockstep"
echo "================================================================"

LAUNCH_LOG="/tmp/rl_stack_$(date +%Y-%m-%d_%H-%M-%S).log"

# --- T1: launch sim stack (no IBVS, no LOSS watchdog, long DURATION) ---
echo "[RL-T1] Launching sim stack (SKIP_IBVS=1, SPEED_FACTOR=${SPEED_FACTOR}x)..."
nohup bash -c "
  source /opt/ros/noetic/setup.bash
  source /home/rawad/catkin_ws/devel/setup.bash
  WORLD=$WORLD HEADLESS=0 VIEWER=0 NO_LOCKSTEP=1 SPEED_FACTOR=$SPEED_FACTOR \
  SKIP_IBVS=1 LOSS_TIMEOUT=3600 DURATION=7200 \
  bash /home/rawad/catkin_ws/src/drone_tracking/scripts/launch_stack.sh 1 $TRAJ 1 $SEED 7200
" > "$LAUNCH_LOG" 2>&1 &
STACK_PID=$!
disown $STACK_PID
echo "  Stack PID=$STACK_PID  log=$LAUNCH_LOG"

# --- T2: wait for both drones at altitude ---
echo "[RL-T2] Waiting for both drones at altitude..."
WAIT_TIMEOUT=300
WAITED=0
until timeout 5 bash -c \
    "rostopic echo -n1 /drone_tracking/takeoff_ready 2>/dev/null | grep -q 'data: True'"; do
    sleep 3; WAITED=$((WAITED+3))
    if [ $WAITED -ge $WAIT_TIMEOUT ]; then
        echo "  ERROR: takeoff_ready TIMEOUT after ${WAIT_TIMEOUT}s — aborting"
        kill $STACK_PID 2>/dev/null; exit 1
    fi
    printf ".";
done
echo " OK"
echo "  Both drones at altitude — starting SAC training"
sleep 2  # let target mover start

# --- T3: launch SAC training ---
SAC_LOG="/tmp/rl_sac_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[RL-T3] Launching rl_train_sac.py --scratch --steps=$STEPS ..."
echo "  SAC log: $SAC_LOG"
rosrun drone_tracking rl_train_sac.py \
    --scratch \
    --steps "$STEPS" \
    --chunk 5000 \
    --save "$SAC_SAVE" \
    --seed "$SEED" \
    --ent-coef 0.02 \
    2>&1 | tee "$SAC_LOG"

echo ""
echo "[RL-T4] SAC training finished. Tearing down sim stack..."
kill $STACK_PID 2>/dev/null || true
sleep 5
# final cleanup in case stack teardown missed anything
source /home/rawad/catkin_ws/src/drone_tracking/scripts/cleanup.sh 2>/dev/null || true
echo "[RL] Done."
