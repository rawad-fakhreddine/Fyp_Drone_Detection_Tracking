#!/usr/bin/env bash
# rl_td3_eval_launch.sh — TD3 deterministic-eval launcher (Config 3, LOCKSTEP @0.004)
#
# Mirrors rl_sac_eval_launch.sh exactly; runs rl_eval_td3.py (deterministic=True). For
# TD3 that IS the trained actor -> the eval scores exactly what was optimised.
#
# Usage:  bash rl_td3_eval_launch.sh [SECS] [TRAJ] [SEED] [MODEL]
# Defaults: SECS=120  TRAJ=4  SEED=42  MODEL=~/fyp/rl/models/td3/td3_policy.zip
set -e
SECS="${1:-120}"
TRAJ="${2:-4}"
SEED="${3:-42}"
MODEL="${4:-$HOME/fyp/rl/models/td3/td3_policy.zip}"
WORLD="${WORLD:-rl_empty}"

source /opt/ros/noetic/setup.bash
source /home/rawad/catkin_ws/devel/setup.bash

echo "================================================================"
echo "  TD3 Eval | traj=$TRAJ seed=$SEED secs=$SECS | deterministic"
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
echo "  Both drones at altitude — starting TD3 eval"
sleep 2  # let target mover start

# --- T3: run deterministic eval ---
EVAL_LOG="/tmp/rl_eval_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[RL-T3] Launching rl_eval_td3.py (deterministic, ${SECS}s)..."
echo "  Eval log: $EVAL_LOG"
PYTHONUNBUFFERED=1 rosrun drone_tracking rl_eval_td3.py \
    _model:="$MODEL" \
    _secs:="$SECS" \
    _n_stack:="${N_STACK:-1}" \
    _max_wz:="${MAX_WZ:-1.0}" \
    _max_vx:="${MAX_VX:-4.0}" \
    _max_vz:="${MAX_VZ:-2.5}" \
    _accel_limit:="${ACCEL_LIMIT:-0.0}" \
    _accel_limit_wz:="${ACCEL_LIMIT_WZ:-${ACCEL_LIMIT:-0.0}}" \
    _accel_limit_vz:="${ACCEL_LIMIT_VZ:-${ACCEL_LIMIT:-0.0}}" \
    _action_lpf:="${ACTION_LPF:-1.0}" \
    2>&1 | tee "$EVAL_LOG"

echo ""
echo "[RL-T4] TD3 eval finished. Tearing down sim stack..."
kill $STACK_PID 2>/dev/null || true
sleep 5
source /home/rawad/catkin_ws/src/drone_tracking/scripts/cleanup.sh 2>/dev/null || true
echo "[RL] Done. flight log -> ~/flight_log_latest.csv"
