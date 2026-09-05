#!/usr/bin/env bash
# rl_sac_eval_launch.sh — SAC deterministic-eval launcher (Config 3, LOCKSTEP @0.004)
#
# Usage:
#   bash rl_sac_eval_launch.sh [SECS] [TRAJ] [SEED] [MODEL]
#
# Defaults: SECS=120  TRAJ=4 (orbit)  SEED=42  MODEL=~/fyp/rl/models/sac/sac_policy.zip
#
# Mirrors rl_sac_train_launch.sh exactly (stack + hover bridge + takeoff wait) but
# runs rl_eval_sac.py with deterministic=True (mean action, NO exploration),
# reset-free, for a fixed duration. flight_logger keeps writing the standard 65-col
# ~/flight_log_latest.csv so smoothness.py scores it against the IBVS baseline in
# the identical clean-tracking regime.
#
# What it does:
#   1. Launches the full sim stack (lockstep, SKIP_IBVS=1, long DURATION)
#   2. Starts a hover publisher — keeps PX4 in OFFBOARD after takeoff_both exits
#   3. Waits for both drones at altitude
#   4. Runs rl_eval_sac.py (frozen policy) — RL keepalive takes over the topic
#   5. Tears down the stack when eval exits
set -e
SECS="${1:-120}"
TRAJ="${2:-4}"
SEED="${3:-42}"
MODEL="${4:-$HOME/fyp/rl/models/sac/sac_policy.zip}"
WORLD="${WORLD:-rl_empty}"

source /opt/ros/noetic/setup.bash
source /home/rawad/catkin_ws/devel/setup.bash

echo "================================================================"
echo "  SAC Eval | traj=$TRAJ seed=$SEED secs=$SECS | deterministic"
echo "  model=$MODEL"
echo "================================================================"

LAUNCH_LOG="/tmp/rl_stack_$(date +%Y-%m-%d_%H-%M-%S).log"

# --- T1: launch sim stack (lockstep, SKIP_IBVS=1, no LOSS watchdog) ---
echo "[RL-T1] Launching sim stack (SKIP_IBVS=1, LOCKSTEP @0.004)..."
nohup bash -c "
  source /opt/ros/noetic/setup.bash
  source /home/rawad/catkin_ws/devel/setup.bash
  WORLD=$WORLD HEADLESS=1 VIEWER=0 \
  SKIP_IBVS=1 LOSS_TIMEOUT=3600 DURATION=3600 \
  MAX_CLIMB=3.0 HOVER_HOLD_S=1.0 \
  bash /home/rawad/catkin_ws/src/drone_tracking/scripts/launch_stack.sh 1 $TRAJ 1 $SEED 3600
" > "$LAUNCH_LOG" 2>&1 &
STACK_PID=$!
disown $STACK_PID
echo "  Stack PID=$STACK_PID  log=$LAUNCH_LOG"

# --- Hover publisher: keeps chaser in OFFBOARD after takeoff_both.py exits. ---
python3 -u -c "
import time, rospy
from mavros_msgs.msg import PositionTarget
for _ in range(120):
    try:
        rospy.init_node('rl_hover_hold', anonymous=True, disable_signals=True)
        break
    except:
        time.sleep(0.5)
pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
m = PositionTarget()
m.coordinate_frame = 8   # FRAME_LOCAL_NED
m.type_mask = 1479       # velocity only; zero vel = hover
while not rospy.is_shutdown():
    m.header.stamp = rospy.Time(0)
    pub.publish(m)
    time.sleep(0.05)
" &
HOVER_PID=$!
echo "  Hover publisher PID=$HOVER_PID (bridges gap until rl_env.py keepalive starts)"

# --- T2: wait for both drones at altitude ---
echo "[RL-T2] Waiting for both drones at altitude (timeout 360s)..."
if ! timeout 360 python3 - <<'PYEOF'
import sys, rospy
from std_msgs.msg import Bool
rospy.init_node('rl_wait_takeoff', anonymous=True)
try:
    msg = rospy.wait_for_message('/drone_tracking/takeoff_ready', Bool, timeout=350.0)
    sys.exit(0 if msg.data else 1)
except rospy.ROSException:
    sys.exit(1)
PYEOF
then
    echo "  ERROR: takeoff_ready TIMEOUT — aborting"
    kill $STACK_PID 2>/dev/null; exit 1
fi
echo " OK"
echo "  Both drones at altitude — starting SAC eval"
sleep 2  # let target mover start

# --- T3: run deterministic eval ---
EVAL_LOG="/tmp/rl_eval_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[RL-T3] Launching rl_eval_sac.py (deterministic, ${SECS}s)..."
echo "  Eval log: $EVAL_LOG"
PYTHONUNBUFFERED=1 rosrun drone_tracking rl_eval_sac.py \
    _model:="$MODEL" \
    _secs:="$SECS" \
    _n_stack:="${N_STACK:-1}" \
    _max_wz:="${MAX_WZ:-0.5}" \
    _max_vx:="${MAX_VX:-4.0}" \
    _max_vz:="${MAX_VZ:-2.5}" \
    _accel_limit:="${ACCEL_LIMIT:-0.0}" \
    _accel_limit_wz:="${ACCEL_LIMIT_WZ:-${ACCEL_LIMIT:-0.0}}" \
    _accel_limit_vz:="${ACCEL_LIMIT_VZ:-${ACCEL_LIMIT:-0.0}}" \
    _action_lpf:="${ACTION_LPF:-1.0}" \
    _alt_floor:="${ALT_FLOOR:-11.0}" \
    _alt_ceil:="${ALT_CEIL:-22.0}" \
    2>&1 | tee "$EVAL_LOG"

echo ""
echo "[RL-T4] SAC eval finished. Tearing down sim stack..."
kill $STACK_PID 2>/dev/null || true
sleep 5
source /home/rawad/catkin_ws/src/drone_tracking/scripts/cleanup.sh 2>/dev/null || true
echo "[RL] Done. flight log -> ~/flight_log_latest.csv"
