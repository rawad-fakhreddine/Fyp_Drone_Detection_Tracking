#!/usr/bin/env bash
# rl_sac_train_launch.sh — SAC training launcher (Config 3, LOCKSTEP @0.004)
#
# Usage:
#   bash rl_sac_train_launch.sh [STEPS] [TRAJ] [SEED]
#   RESUME=1 bash rl_sac_train_launch.sh [STEPS] [TRAJ] [SEED]   # continue from checkpoint
#
# Defaults: STEPS=20000  TRAJ=4 (orbit)  SEED=42  RESUME=0
#
# Session-based training: each run completes ~20k steps within the 7200s stack cap.
# Session 1 (RESUME=0): pure online SAC from scratch (--scratch, ent_coef=auto).
# Session 2+ (RESUME=1): loads sac_policy.zip + sac_replay.pkl and continues.
#
# Speed model (2026-08-23, empirically verified): LOCKSTEP at max_step_size=0.004
# runs at ~1× RTF on i5-13420H/WSL2 — PX4's HIL response (~4ms/step) gates each
# physics step. NEVER set PX4_SIM_SPEED_FACTOR in lockstep — EKF fault → SIGABRT.
#
# What it does:
#   1. Launches the full sim stack (lockstep, SKIP_IBVS=1, long DURATION)
#   2. Starts a hover publisher immediately — keeps PX4 in OFFBOARD after takeoff_both exits
#   3. Waits for both drones to be at altitude
#   4. Starts rl_train_sac.py — RL keepalive takes over the setpoint topic
#   5. Cleans up the stack when SAC exits
set -e
STEPS="${1:-20000}"
TRAJ="${2:-4}"
SEED="${3:-42}"
WORLD="${WORLD:-rl_empty}"
RESUME="${RESUME:-0}"
BC="${BC:-0}"                      # BC=1 -> BC-warm-start + relaxable SACBC anchor (14-dim v4)

source /opt/ros/noetic/setup.bash
source /home/rawad/catkin_ws/devel/setup.bash

SAC_SAVE="${SAC_SAVE:-~/fyp/rl/models/sac}"
BC_PATH="${BC_PATH:-~/fyp/rl/models/bc_policy_v4.pth}"   # v4 = 14-dim explicit-rate clone

if [ "$RESUME" = "1" ]; then
    SESSION_MODE="RESUME from checkpoint"
elif [ "$BC" = "1" ]; then
    SESSION_MODE="BC-warm-start + SACBC anchor (v4 14-dim)"
else
    SESSION_MODE="scratch (online SAC, ent_coef=auto)"
fi
echo "================================================================"
echo "  SAC Training | traj=$TRAJ seed=$SEED steps=$STEPS | $SESSION_MODE"
echo "================================================================"

LAUNCH_LOG="/tmp/rl_stack_$(date +%Y-%m-%d_%H-%M-%S).log"

# --- T1: launch sim stack (lockstep, SKIP_IBVS=1, no LOSS watchdog, long DURATION) ---
echo "[RL-T1] Launching sim stack (SKIP_IBVS=1, LOCKSTEP @0.004)..."
nohup bash -c "
  source /opt/ros/noetic/setup.bash
  source /home/rawad/catkin_ws/devel/setup.bash
  WORLD=$WORLD HEADLESS=1 VIEWER=0 \
  SKIP_IBVS=1 LOSS_TIMEOUT=3600 DURATION=7200 \
  MAX_CLIMB=3.0 HOVER_HOLD_S=1.0 \
  bash /home/rawad/catkin_ws/src/drone_tracking/scripts/launch_stack.sh 1 $TRAJ 1 $SEED 7200
" > "$LAUNCH_LOG" 2>&1 &
STACK_PID=$!
disown $STACK_PID
echo "  Stack PID=$STACK_PID  log=$LAUNCH_LOG"

# --- Hover publisher: keeps chaser in OFFBOARD after takeoff_both.py exits.
# PX4 drops OFFBOARD within 0.5s if no setpoint arrives. IBVS is skipped (RL owns
# the topic), so this bridge publishes zero-velocity (hover) until rl_env.py's
# keepalive takes over. Killed automatically when the stack dies (ROS shutdown). ---
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
m.type_mask = 1479       # ignore pos, acc, yaw → velocity only; zero vel = hover
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
echo "  Both drones at altitude — starting SAC training"
sleep 2  # let target mover start

# --- T3: launch SAC training ---
SAC_LOG="/tmp/rl_sac_$(date +%Y-%m-%d_%H-%M-%S).log"
if [ "$RESUME" = "1" ]; then
    TRAIN_FLAGS="--resume"
    ENT_COEF="${ENT_COEF:-0.1}"   # default 0.1 (prevents collapse from saved low value); overridable
                                  # via env for the BC1 curriculum, which needs LOW exploration
                                  # (ent_coef~0.03) or online SAC washes out the BC band-hold prior.
    echo "[RL-T3] Launching rl_train_sac.py --resume --steps=$STEPS ent_coef=$ENT_COEF ..."
elif [ "$BC" = "1" ]; then
    # BC-warm-start + relaxable SACBC anchor. ent_coef FIXED small (auto washes out the
    # warm-start per 2026-08 note). anchor-ref = same v4 clone; leash w0->0 over bc-anneal.
    TRAIN_FLAGS="--bc $BC_PATH --anchor --anchor-ref $BC_PATH \
        --bc-w0 ${BC_W0:-15} --bc-alpha ${BC_ALPHA:-2.5} --bc-anneal ${BC_ANNEAL:-15000} \
        --log-std ${LOG_STD:--3.0}"
    ENT_COEF="${ENT_COEF:-0.05}"
    echo "[RL-T3] Launching rl_train_sac.py --bc(v4) --anchor --steps=$STEPS ent_coef=$ENT_COEF ..."
else
    TRAIN_FLAGS="--scratch"
    ENT_COEF="auto"  # auto for scratch — targets H=-4
    echo "[RL-T3] Launching rl_train_sac.py --scratch --steps=$STEPS ent_coef=auto ..."
fi
echo "  SAC log: $SAC_LOG"
PYTHONUNBUFFERED=1 rosrun drone_tracking rl_train_sac.py \
    $TRAIN_FLAGS \
    --steps "$STEPS" \
    --chunk 5000 \
    --save "$SAC_SAVE" \
    --seed "$SEED" \
    --ent-coef "$ENT_COEF" \
    --batch-size 256 \
    --episode-secs "${EP_SECS:-25}" \
    --gt-prefill-steps "${GT_PREFILL:-3000}" \
    --n-stack "${N_STACK:-1}" \
    _rew_band_anneal_steps:="${BAND_ANNEAL_STEPS:-0}" \
    _rew_band_hi_start:="${BAND_HI_START:-9.0}" \
    _rew_w_d:="${W_D:-0.5}" \
    _rew_band_bonus:="${BAND_BONUS:-1.5}" \
    _rew_w_alt:="${W_ALT:-0.25}" \
    _rew_close_thresh:="${CLOSE_THRESH:-6.0}" \
    _rew_w_retreat:="${W_RETREAT:-2.0}" \
    _rew_band_pen_cap:="${BAND_PEN_CAP:-1.0}" \
    _max_wz:="${MAX_WZ:-0.5}" \
    _max_vx:="${MAX_VX:-4.0}" \
    _accel_limit:="${ACCEL_LIMIT:-0.0}" \
    _accel_limit_wz:="${ACCEL_LIMIT_WZ:-${ACCEL_LIMIT:-0.0}}" \
    _max_vz:="${MAX_VZ:-2.5}" \
    _alt_floor:="${ALT_FLOOR:-11.0}" \
    _alt_ceil:="${ALT_CEIL:-22.0}" \
    _rew_w_approach:="${W_APPROACH:-1.0}" \
    _rew_approach_cap:="${APPROACH_CAP:-1.5}" \
    _rew_sigma:="${SIGMA:-0.6}" \
    _rew_sigma_far:="${SIGMA_FAR:-0.0}" \
    _rew_w_pursuit_blind:="${W_PURSUIT_BLIND:-0.0}" \
    _rew_pursuit_blind_secs:="${PURSUIT_BLIND_SECS:-1.0}" \
    _rew_pursuit_blind_cap:="${PURSUIT_BLIND_CAP:-1.0}" \
    _rew_w_ex:="${W_EX:-0.0}" \
    _rew_w_ddot:="${W_DDOT:-0.0}" \
    _rew_ddot_ex_gate:="${DDOT_EX_GATE:-0.30}" \
    _rew_ddot_cap:="${DDOT_CAP:-1.0}" \
    _mix_trajs:="${MIX_TRAJS:-}" \
    2>&1 | tee "$SAC_LOG"

echo ""
echo "[RL-T4] SAC training finished. Tearing down sim stack..."
kill $STACK_PID 2>/dev/null || true
sleep 5
# final cleanup in case stack teardown missed anything
source /home/rawad/catkin_ws/src/drone_tracking/scripts/cleanup.sh 2>/dev/null || true
echo "[RL] Done."
