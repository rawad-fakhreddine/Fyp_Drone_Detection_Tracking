#!/bin/bash
# =============================================================================
# launch_stack.sh — Run ONE full simulation
# Usage: ./launch_stack.sh CONFIG TRAJ ZONE SEED [DURATION]
#   CONFIG   : 1=YOLO+IBVS  2=YOLO+Kalman+IBVS  3=Full
#   TRAJ     : 1-9
#   ZONE     : 5,6,7,9
#   SEED     : integer
#   DURATION : seconds (default 300)
# =============================================================================
set -u

CONFIG="${1:-}"; TRAJ="${2:-}"; ZONE="${3:-}"; SEED="${4:-}"; DURATION="${5:-300}"
[ -z "$CONFIG" ] || [ -z "$TRAJ" ] || [ -z "$ZONE" ] || [ -z "$SEED" ] && {
    echo "Usage: $0 CONFIG TRAJ ZONE SEED [DURATION]"; exit 1; }

PX4_HOME=~/PX4-Autopilot
PKG_DIR=~/catkin_ws/src/drone_tracking
TARGET_SDF=$PX4_HOME/Tools/sitl_gazebo/models/target_iris_sitl/iris.sdf
MODEL_PATH=~/drone_detection/models/best.pt

[ -f "$MODEL_PATH" ] || { echo "[ERROR] YOLO model missing: $MODEL_PATH"; exit 1; }
[ -f "$TARGET_SDF" ] || { echo "[ERROR] Target SDF missing — run setup_dual_px4.sh first"; exit 1; }

case "$CONFIG" in
    1) CFG_LABEL="Config 1"; USE_KALMAN=false; USE_PPO=false; DET_SRC=raw    ;;
    2) CFG_LABEL="Config 2"; USE_KALMAN=true;  USE_PPO=false; DET_SRC=kalman ;;
    3) CFG_LABEL="Config 3"; USE_KALMAN=true;  USE_PPO=true;  DET_SRC=kalman ;;
    *) echo "[ERROR] CONFIG must be 1, 2 or 3"; exit 1 ;;
esac

TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_TAG="zone${ZONE}_traj${TRAJ}_${TS}"
RESULTS_DIR=~/results/"$CFG_LABEL"
mkdir -p "$RESULTS_DIR"

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source $PX4_HOME/Tools/setup_gazebo.bash $PX4_HOME $PX4_HOME/build/px4_sitl_default 2>/dev/null
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_HOME:$PX4_HOME/Tools/sitl_gazebo

echo ""
echo "================================================================"
echo "  $CFG_LABEL | $RUN_TAG | seed=$SEED | ${DURATION}s"
echo "  Kalman=$USE_KALMAN  PPO=$USE_PPO  DetSrc=$DET_SRC"
echo "================================================================"

wait_topic() {
    local topic="$1" timeout="${2:-30}" elapsed=0
    echo -n "  Waiting for $topic ."
    while ! rostopic list 2>/dev/null | grep -q "^${topic}$"; do
        sleep 1; elapsed=$((elapsed+1)); echo -n "."
        [ $elapsed -ge $timeout ] && { echo " TIMEOUT"; return 1; }
    done
    echo " OK"; return 0
}

# T1 — Gazebo + Chaser PX4 + Chaser MAVROS
echo "[T1] Gazebo + Chaser PX4 + MAVROS..."
roslaunch $PX4_HOME/launch/mavros_posix_sitl.launch \
    vehicle:=iris \
    sdf:=$PX4_HOME/Tools/sitl_gazebo/models/iris_fpv_cam/iris_fpv_cam.sdf \
    world:=$PX4_HOME/Tools/sitl_gazebo/worlds/baylands.world \
    > /tmp/T1_${RUN_TAG}.log 2>&1 &
wait_topic "/mavros/state" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
sleep 5

# T2 — Spawn target + teleport chaser
echo "[T2] Spawning target + positioning chaser (zone=$ZONE seed=$SEED)..."
rosrun drone_tracking random_spawn_target.py _zone:=$ZONE _seed:=$SEED \
    > /tmp/T2_${RUN_TAG}.log 2>&1 &
sleep 4

# T3 — YOLO
echo "[T3] YOLO detection node..."
rosrun drone_tracking yolo_detection_node.py > /tmp/T3_${RUN_TAG}.log 2>&1 &
sleep 3

# T4 — Kalman (configs 2, 3 only)
if [ "$USE_KALMAN" = "true" ]; then
    echo "[T4] Kalman filter..."
    rosrun drone_tracking kalman_filter_node.py > /tmp/T4_${RUN_TAG}.log 2>&1 &
    sleep 2
else
    echo "[T4] Kalman SKIPPED (Config 1)"
fi

# T5 — Target PX4 SITL
echo "[T5] Target PX4 SITL (instance 1)..."
(cd $PX4_HOME && PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_NAME=target_iris \
    ./build/px4_sitl_default/bin/px4 -i 1 -d ./build/px4_sitl_default/etc \
    > /tmp/T5_${RUN_TAG}.log 2>&1) &
sleep 10

# T6 — Target MAVROS
echo "[T6] Target MAVROS..."
roslaunch $PKG_DIR/launch/target_mavros.launch > /tmp/T6_${RUN_TAG}.log 2>&1 &
wait_topic "/target/mavros/state" 30 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
sleep 3

# T7 — IBVS
echo "[T7] IBVS (use_ppo=$USE_PPO detection_source=$DET_SRC)..."
rosrun drone_tracking ibvs_controller_node.py \
    _use_ppo:=$USE_PPO _detection_source:=$DET_SRC \
    > /tmp/T7_${RUN_TAG}.log 2>&1 &
sleep 2

# T8 — PPO (config 3 only)
if [ "$USE_PPO" = "true" ]; then
    echo "[T8] PPO agent..."
    rosrun drone_tracking ppo_agent_node.py > /tmp/T8_${RUN_TAG}.log 2>&1 &
    sleep 2
else
    echo "[T8] PPO SKIPPED"
fi

# T9 — Takeoff both (auto-arms both drones)
echo "[T9] Takeoff both..."
rosrun drone_tracking takeoff_both.py > /tmp/T9_${RUN_TAG}.log 2>&1 &
wait_topic "/drone_tracking/takeoff_ready" 90 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
echo "  Both drones at altitude — mission starting"
sleep 2

# T10 — Target mover
echo "[T10] Target mover (trajectory=$TRAJ seed=$SEED)..."
rosrun drone_tracking target_mover.py _trajectory:=$TRAJ _seed:=$SEED \
    > /tmp/T10_${RUN_TAG}.log 2>&1 &
sleep 1

# T11 — Flight logger
echo "[T11] Flight logger..."
rosrun drone_tracking flight_logger.py > /tmp/T11_${RUN_TAG}.log 2>&1 &

echo ""
echo "  >>> Mission running for ${DURATION}s <<<"
sleep $DURATION
echo "  >>> Mission complete <<<"
echo ""

# Stop logger gracefully first (flush CSV)
echo "[Post] Stopping logger..."
pkill -INT -f flight_logger.py 2>/dev/null
sleep 3

# Save CSV with structured name
CSV_OUT="$RESULTS_DIR/${RUN_TAG}.csv"
if [ -f ~/flight_log_latest.csv ]; then
    cp ~/flight_log_latest.csv "$CSV_OUT"
    echo "[Save] $CSV_OUT"
else
    echo "[WARN] ~/flight_log_latest.csv not found"
fi

# Run analyzer
ANALYSIS_OUT="$RESULTS_DIR/${RUN_TAG}.analysis.txt"
if [ -f "$CSV_OUT" ]; then
    echo "[Analyze] Running analyzer..."
    python3 $PKG_DIR/scripts/analyze_flight_log.py \
        --file "$CSV_OUT" > "$ANALYSIS_OUT" 2>&1
    echo "[Metrics] Extracting summary metrics..."
    python3 $PKG_DIR/scripts/extract_metrics.py \
        --csv "$CSV_OUT" --config $CONFIG \
        --zone $ZONE --traj $TRAJ \
        --seed $SEED --duration $DURATION
fi

# Cleanup
echo "[Cleanup] Tearing down stack..."
bash $PKG_DIR/scripts/cleanup.sh
sleep 3

echo ""
echo "================================================================"
echo "  DONE: $CFG_LABEL / $RUN_TAG"
echo "  CSV:      $CSV_OUT"
echo "  Analysis: $ANALYSIS_OUT"
echo "  Summary:  ~/results/summary.csv"
echo "================================================================"
echo ""
