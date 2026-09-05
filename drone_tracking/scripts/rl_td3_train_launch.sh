#!/usr/bin/env bash
# rl_td3_train_launch.sh — TD3 training launcher (Config 3, LOCKSTEP @0.004)
#
# TD3 = deterministic policy gradient. The actor it optimises IS the actor deployed at
# eval -> no stochastic-vs-deterministic gap (SAC's wall this session). Pure RL from
# scratch, NO behaviour cloning. Mirrors rl_sac_train_launch.sh exactly (stack + hover
# bridge + takeoff wait), only the training node + its knobs differ.
#
# Usage:
#   bash rl_td3_train_launch.sh [STEPS] [TRAJ] [SEED]
#   RESUME=1 bash rl_td3_train_launch.sh [STEPS] [TRAJ] [SEED]   # continue from checkpoint
#
# Defaults: STEPS=20000  TRAJ=4 (orbit)  SEED=42  RESUME=0
set -e
STEPS="${1:-20000}"
TRAJ="${2:-4}"
SEED="${3:-42}"
WORLD="${WORLD:-rl_empty}"
RESUME="${RESUME:-0}"

source /opt/ros/noetic/setup.bash
source /home/rawad/catkin_ws/devel/setup.bash

TD3_SAVE="${TD3_SAVE:-~/fyp/rl/models/td3}"

if [ "$RESUME" = "1" ]; then
    SESSION_MODE="RESUME from checkpoint"
else
    SESSION_MODE="scratch (online TD3, deterministic actor + Gaussian expl noise)"
fi
echo "================================================================"
echo "  TD3 Training | traj=$TRAJ seed=$SEED steps=$STEPS | $SESSION_MODE"
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
echo "  Both drones at altitude — starting TD3 training"
sleep 2  # let target mover start

# --- T3: launch TD3 training ---
TD3_LOG="/tmp/rl_td3_$(date +%Y-%m-%d_%H-%M-%S).log"
if [ "$RESUME" = "1" ]; then
    TRAIN_FLAGS="--resume"
    echo "[RL-T3] Launching rl_train_td3.py --resume --steps=$STEPS ..."
else
    TRAIN_FLAGS=""
    echo "[RL-T3] Launching rl_train_td3.py (scratch) --steps=$STEPS ..."
fi
echo "  TD3 log: $TD3_LOG"
PYTHONUNBUFFERED=1 rosrun drone_tracking rl_train_td3.py \
    $TRAIN_FLAGS \
    --steps "$STEPS" \
    --chunk "${CHUNK:-5000}" \
    --save "$TD3_SAVE" \
    --seed "$SEED" \
    --batch-size 256 \
    --episode-secs "${EP_SECS:-35}" \
    --gt-prefill-steps "${GT_PREFILL:-3000}" \
    --offline-pretrain "${OFFLINE_PRETRAIN:-5000}" \
    --n-stack "${N_STACK:-1}" \
    --train-freq "${TRAIN_FREQ:-1}" \
    --gradient-steps "${GRADIENT_STEPS:-1}" \
    --learning-rate "${LEARNING_RATE:-3e-4}" \
    --action-noise "${ACTION_NOISE:-0.20}" \
    --policy-delay "${POLICY_DELAY:-2}" \
    --target-policy-noise "${TARGET_POLICY_NOISE:-0.20}" \
    --caps-lambda-s "${CAPS_LAMBDA_S:-0.0}" \
    --caps-sigma-s "${CAPS_SIGMA_S:-0.05}" \
    --caps-lambda-t "${CAPS_LAMBDA_T:-0.0}" \
    ${BC_ANCHOR:+--bc-anchor "$BC_ANCHOR"} \
    --bc-w0 "${BC_W0:-0.0}" \
    --bc-anneal "${BC_ANNEAL:-20000}" \
    _rew_band_anneal_steps:="${BAND_ANNEAL_STEPS:-0}" \
    _rew_band_hi_start:="${BAND_HI_START:-9.0}" \
    _rew_sigma:="${SIGMA:-0.6}" \
    _rew_A:="${CENTER_A:-3.0}" \
    _rew_sigma_far:="${SIGMA_FAR:-0.0}" \
    _rew_w_ex:="${W_EX:-0.0}" \
    _rew_w_d:="${W_D:-0.5}" \
    _rew_band_bonus:="${BAND_BONUS:-1.5}" \
    _rew_w_alt:="${W_ALT:-1.0}" \
    _rew_alt_cap:="${ALT_CAP:-12.0}" \
    _rew_close_thresh:="${CLOSE_THRESH:-6.0}" \
    _rew_w_retreat:="${W_RETREAT:-2.0}" \
    _rew_band_pen_cap:="${BAND_PEN_CAP:-1.0}" \
    _rew_w_d_close:="${W_D_CLOSE:-0.5}" \
    _rew_band_pen_cap_close:="${BAND_PEN_CAP_CLOSE:-1.0}" \
    _rew_w_ddot:="${W_DDOT:-0.0}" \
    _rew_ddot_ex_gate:="${DDOT_EX_GATE:-0.30}" \
    _rew_w_approach:="${W_APPROACH:-1.0}" \
    _rew_approach_cap:="${APPROACH_CAP:-0.5}" \
    _rew_w_pursuit_blind:="${W_PURSUIT_BLIND:-0.0}" \
    _rew_pursuit_blind_secs:="${PURSUIT_BLIND_SECS:-1.0}" \
    _rew_pursuit_blind_cap:="${PURSUIT_BLIND_CAP:-1.0}" \
    _rew_w_vy_pen:="${W_VY_PEN:-0.0}" \
    _rew_w_fwd:="${W_FWD:-0.0}" \
    _rew_fwd_cap:="${FWD_CAP:-1.5}" \
    _rew_w_match:="${W_MATCH:-0.0}" \
    _rew_match_recede_ref:="${MATCH_RECEDE_REF:-2.0}" \
    _rew_w_acc:="${W_ACC:-0.0}" \
    _rew_acc_cap:="${ACC_CAP:-3.0}" \
    _rew_w_speedmatch:="${W_SPEEDMATCH:-0.0}" \
    _rew_speed_ratio_cap:="${SPEED_RATIO_CAP:-1.2}" \
    _rew_speedmatch_dmin:="${SPEEDMATCH_DMIN:-8.0}" \
    _rew_speedmatch_tmin:="${SPEEDMATCH_TMIN:-0.8}" \
    _rew_speedmatch_floor:="${SPEEDMATCH_FLOOR:-0.6}" \
    _rew_speedmatch_center_sigma:="${SPEEDMATCH_CENTER_SIGMA:-0.0}" \
    _rew_P_lost:="${P_LOST:-0.5}" \
    _rew_w_vel:="${W_VEL:-0.05}" \
    _rew_w_s:="${W_S:-0.20}" \
    _rew_smooth_cap:="${SMOOTH_CAP:-1.0}" \
    _max_wz:="${MAX_WZ:-1.0}" \
    _max_vx:="${MAX_VX:-4.0}" \
    _max_vz:="${MAX_VZ:-2.5}" \
    _accel_limit:="${ACCEL_LIMIT:-0.0}" \
    _accel_limit_wz:="${ACCEL_LIMIT_WZ:-${ACCEL_LIMIT:-0.0}}" \
    _accel_limit_vz:="${ACCEL_LIMIT_VZ:-${ACCEL_LIMIT:-0.0}}" \
    _action_lpf:="${ACTION_LPF:-1.0}" \
    _mix_trajs:="${MIX_TRAJS:-}" \
    2>&1 | tee "$TD3_LOG"

echo ""
echo "[RL-T4] TD3 training finished. Tearing down sim stack..."
kill $STACK_PID 2>/dev/null || true
sleep 5
source /home/rawad/catkin_ws/src/drone_tracking/scripts/cleanup.sh 2>/dev/null || true
echo "[RL] Done."
