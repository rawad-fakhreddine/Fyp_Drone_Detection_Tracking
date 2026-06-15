#!/bin/bash
# =============================================================================
# launch_stack.sh — Run ONE full simulation
# Usage: ./launch_stack.sh CONFIG TRAJ ZONE SEED [DURATION]
#   CONFIG   : 1=YOLO+IBVS  2=YOLO+Kalman+IBVS  3=Full
#   TRAJ     : 1-9
#   ZONE     : 5,6,7,9
#   SEED     : integer
#   DURATION : seconds (default 300)
# Env knobs:
#   VIEWER=0        : skip the YOLO debug viewer (default on)
#   START_DIST=N    : spawn separation in m (default: random 8-12)
#   LOSS_TIMEOUT=N  : abort run after N s continuous SEARCH (default 10)
# =============================================================================
set -u

CONFIG="${1:-}"; TRAJ="${2:-}"; ZONE="${3:-}"; SEED="${4:-}"; DURATION="${5:-300}"
[ -z "$CONFIG" ] || [ -z "$TRAJ" ] || [ -z "$ZONE" ] || [ -z "$SEED" ] && {
    echo "Usage: $0 CONFIG TRAJ ZONE SEED [DURATION]"; exit 1; }

PX4_HOME=~/PX4-Autopilot
PKG_DIR=~/catkin_ws/src/drone_tracking
TARGET_SDF=$PX4_HOME/Tools/sitl_gazebo/models/target_iris_sitl/iris.sdf
MODEL_PATH=~/drone_detection/models/best.pt
RESULTS_BASE=~/fyp/Results            # M9.6 step 2: single knob for results location

[ -f "$MODEL_PATH" ] || { echo "[ERROR] YOLO model missing: $MODEL_PATH"; exit 1; }
[ -f "$TARGET_SDF" ] || { echo "[ERROR] Target SDF missing — run setup_dual_px4.sh first"; exit 1; }

case "$CONFIG" in
    1) CFG_LABEL="Config 1"; USE_KALMAN=false; USE_PPO=false; DET_SRC=raw    ;;
    2) CFG_LABEL="Config 2"; USE_KALMAN=true;  USE_PPO=false; DET_SRC=kalman ;;
    3) CFG_LABEL="Config 3"; USE_KALMAN=true;  USE_PPO=true;  DET_SRC=kalman ;;
    *) echo "[ERROR] CONFIG must be 1, 2 or 3"; exit 1 ;;
esac
CFG_DIR="Config${CONFIG}"             # M9.6 step 2: folder name WITHOUT space (path-safe)

TS=$(date +%Y-%m-%d_%H-%M-%S)
# M9.6 step 2: new filename order — trajectory, then zone, then timestamp
RUN_TAG="traj${TRAJ}_zone${ZONE}_${TS}"
RESULTS_DIR="$RESULTS_BASE/$CFG_DIR"
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

# B4: wait for an actual "data: True" message, not just topic existence.
# takeoff_ready is a latched std_msgs/Bool advertised at node start (exists
# immediately) but only published True once both drones reach altitude.
wait_bool_true() {   # wait_bool_true TOPIC TIMEOUT
    local topic="$1" timeout="${2:-90}"
    echo -n "  Waiting for $topic == True ."
    if timeout "$timeout" bash -c \
       "until rostopic echo -n1 $topic 2>/dev/null | grep -q 'data: True'; do sleep 1; echo -n .; done"; then
        echo " OK"; return 0
    else echo " TIMEOUT"; return 1; fi
}

# Sequencing fix (2026-06-11): the stack used to be gated only by wall-clock
# sleeps, but PX4 readiness is a SIM-time property — and sim time freezes
# during the lockstep window (target model spawned, instance 1 not yet
# attached). Racy result: some runs armed fine, others hit "ARM → FAIL"
# because the chaser EKF had only ~8 SIM-seconds after the zone teleport.
# These two gates make every block wait on the actual readiness signal.

wait_fcu() {   # wait_fcu STATE_TOPIC TIMEOUT — MAVROS↔PX4 link is live
    local topic="$1" timeout="${2:-60}" elapsed=0
    echo -n "  Waiting for $topic connected=True ."
    until timeout 3 rostopic echo -n1 "$topic" 2>/dev/null | grep -q "connected: True"; do
        sleep 1; elapsed=$((elapsed+1)); echo -n "."
        [ $elapsed -ge $timeout ] && { echo " TIMEOUT"; return 1; }
    done
    echo " OK"; return 0
}

wait_sim_time() {   # wait_sim_time MIN_SIM_SECS TIMEOUT — EKF settle gate
    local min="$1" timeout="${2:-120}" elapsed=0 simsec=""
    echo -n "  Waiting for sim time >= ${min}s (EKF settle) ."
    while :; do
        simsec=$(timeout 3 rostopic echo -n1 /clock 2>/dev/null | awk '$1=="secs:"{print $2; exit}')
        if [ -n "$simsec" ] && [ "$simsec" -ge "$min" ] 2>/dev/null; then
            echo " OK (sim=${simsec}s)"; return 0
        fi
        sleep 2; elapsed=$((elapsed+2)); echo -n "."
        [ $elapsed -ge $timeout ] && { echo " TIMEOUT (sim=${simsec:-?}s)"; return 1; }
    done
}

# T1 — Gazebo + Chaser PX4 + Chaser MAVROS
# interactive:=false is REQUIRED here: backgrounding roslaunch gives it
# stdin=/dev/null, and the default interactive pxh shell reads EOF the moment
# rcS completes (= when baylands finishes loading) and exits px4 cleanly.
# sitl is required="true" in posix_sitl.launch, so that single EOF tore down
# master + Gazebo and killed every later node (the real cause of the
# "chaser PX4 dies at T5/T10" failures). interactive:=false runs px4 with -d.
echo "[T1] Gazebo + Chaser PX4 + MAVROS..."
roslaunch $PX4_HOME/launch/mavros_posix_sitl.launch \
    vehicle:=iris \
    interactive:=false \
    sdf:=$PX4_HOME/Tools/sitl_gazebo/models/iris_fpv_cam/iris_fpv_cam.sdf \
    world:=$PX4_HOME/Tools/sitl_gazebo/worlds/baylands.world \
    > /tmp/T1_${RUN_TAG}.log 2>&1 &
wait_topic "/mavros/state" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
# Gate: chaser PX4 alive and linked to MAVROS (= baylands fully loaded and the
# sim stepping) BEFORE the zone teleport — topic existence alone fires ~30 s
# too early, while the world is still loading.
wait_fcu "/mavros/state" 90 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
sleep 5

# T2 — Spawn target + teleport chaser
# START_DIST overrides the spawn separation (default -1 → node picks 8-12m)
echo "[T2] Spawning target + positioning chaser (zone=$ZONE seed=$SEED dist=${START_DIST:--1})..."
rosrun drone_tracking random_spawn_target.py _zone:=$ZONE _seed:=$SEED \
    _dist:=${START_DIST:--1} \
    > /tmp/T2_${RUN_TAG}.log 2>&1 &
sleep 4

# T3 — YOLO
echo "[T3] YOLO detection node..."
rosrun drone_tracking yolo_detection_node.py > /tmp/T3_${RUN_TAG}.log 2>&1 &
sleep 3

# TV — YOLO debug viewer (live detection overlay window; VIEWER=0 to disable
# for long unattended batches). Killed by cleanup.sh like every other node.
if [ "${VIEWER:-1}" = "1" ]; then
    echo "[TV] YOLO debug viewer..."
    rosrun drone_tracking yolo_debug_viewer.py > /tmp/TV_${RUN_TAG}.log 2>&1 &
else
    echo "[TV] Viewer SKIPPED (VIEWER=0)"
fi

# T4 — Kalman (configs 2, 3 only)
if [ "$USE_KALMAN" = "true" ]; then
    echo "[T4] Kalman filter..."
    rosrun drone_tracking kalman_filter_node.py > /tmp/T4_${RUN_TAG}.log 2>&1 &
    sleep 2
else
    echo "[T4] Kalman SKIPPED (Config 1)"
fi

# T5 — Target PX4 SITL (instance 1), ATTACHING to the existing Gazebo from T1.
# Uses PX4's own multi-instance Classic form (Tools/gazebo_sitl_multiple_run.sh
# ~L37): dedicated per-instance working dir + -w isolation + explicit rootfs
# "<build>/etc". Ports align: target SDF mavlink_tcp_port=4561 and -i 1 →
# simulator_tcp_port=4560+1=4561. PX4_SIM_MODEL=iris → airframe 10016_iris (the
# only iris airframe in this build; legacy 4001 no longer exists here).
# NOTE (2026-06-10, corrects the earlier "T5 fix" attribution): the recurring
# chaser-PX4 death was NEVER caused by this command — it was the chaser's own
# interactive pxh shell reading EOF (see T1 comment / interactive:=false).
# Verified standalone: this T5 form blocks correctly at "Waiting for simulator
# to accept connection on TCP port 4561" until the target plugin connects.
# This stage is rcS-blocked (lockstep) until then — that is expected.
echo "[T5] Target PX4 SITL (instance 1, attaching to existing Gazebo)..."
PX4_BUILD="$PX4_HOME/build/px4_sitl_default"
TGT_WORKDIR="$PX4_BUILD/instance_1"
mkdir -p "$TGT_WORKDIR"
(cd "$TGT_WORKDIR" && \
 PX4_SIM_MODEL=iris \
 "$PX4_BUILD/bin/px4" -i 1 -d "$PX4_BUILD/etc" -w sitl_iris_1 -s etc/init.d-posix/rcS \
 > /tmp/T5_${RUN_TAG}.log 2>&1) &
sleep 10

# T6 — Target MAVROS
echo "[T6] Target MAVROS..."
roslaunch $PKG_DIR/launch/target_mavros.launch > /tmp/T6_${RUN_TAG}.log 2>&1 &
wait_topic "/target/mavros/state" 30 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
# Gate: target PX4 heartbeating through MAVROS (implies instance 1 attached to
# Gazebo on TCP 4561 and rcS completed — heartbeats only start after that),
# i.e. the lockstep freeze is over and sim time is advancing again.
wait_fcu "/target/mavros/state" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
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

# T9 — Flight logger (B5: started BEFORE takeoff so TAKEOFF + early SEARCH are logged)
echo "[T9] Flight logger..."
rosrun drone_tracking flight_logger.py > /tmp/T9_${RUN_TAG}.log 2>&1 &
sleep 2

# T10 — Takeoff both (auto-arms both drones)
# Gate: PX4 refuses arming until the EKF has re-aligned after the T2 zone
# teleport (a ~200 m position jump during initial alignment). Successful runs
# armed at sim ~24 s+; the crash run tried at sim 8 s and failed 10x, leaving
# the armed target abandoned on the pad. 25 SIM-seconds is the settle floor.
wait_sim_time 25 120 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
echo "[T10] Takeoff both..."
rosrun drone_tracking takeoff_both.py > /tmp/T10_${RUN_TAG}.log 2>&1 &
# B4: block until takeoff_ready publishes data:True (180s > takeoff_both CLIMB_TIMEOUT=150s)
wait_bool_true "/drone_tracking/takeoff_ready" 180 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
echo "  Both drones at altitude — mission starting"
sleep 2

# T11 — Target mover
echo "[T11] Target mover (trajectory=$TRAJ seed=$SEED)..."
rosrun drone_tracking target_mover.py _trajectory:=$TRAJ _seed:=$SEED \
    > /tmp/T11_${RUN_TAG}.log 2>&1 &
sleep 1

# Loss watchdog: if the IBVS phase stays SEARCH for LOSS_TIMEOUT consecutive
# seconds the run is unrecoverable — end it early instead of burning the rest
# of DURATION. Phase is read from the live logger CSV (col 2, 10 Hz).
# Default 10 s = TUNING runs (fail fast). Official --matrix batches export
# LOSS_TIMEOUT=60 via run_config.sh: a 10 s abort censors the recovery-time
# metric and biases against Config 1 (raw mode drops out more, recovers
# slower); a 60 s abort is itself a recorded outcome (aborted=1 in summary).
LOSS_TIMEOUT="${LOSS_TIMEOUT:-10}"
echo ""
echo "  >>> Mission running for ${DURATION}s (loss watchdog ${LOSS_TIMEOUT}s) <<<"
SIM_T0=$(tail -n 1 ~/flight_log_latest.csv 2>/dev/null | cut -d, -f1)
ELAPSED=0; LOST=0; ABORT_REASON=""; ACQUIRED=0
while [ $ELAPSED -lt $DURATION ]; do
    sleep 5; ELAPSED=$((ELAPSED+5))
    PHASE=$(tail -n 1 ~/flight_log_latest.csv 2>/dev/null | cut -d, -f2)
    # First-acquisition grace: SEARCH only counts toward the abort after the
    # run has tracked at least once (APPROACH/HOLD seen) — aborting during
    # initial acquisition would kill runs the controller handles fine.
    case "$PHASE" in APPROACH|HOLD) ACQUIRED=1 ;; esac
    if [ "$PHASE" = "SEARCH" ] && [ $ACQUIRED -eq 1 ]; then
        LOST=$((LOST+5))
    else
        LOST=0
    fi
    if [ $LOST -ge $LOSS_TIMEOUT ]; then
        ABORT_REASON="target lost — SEARCH for ${LOST}s (>= ${LOSS_TIMEOUT}s)"
        break
    fi
done
SIM_T1=$(tail -n 1 ~/flight_log_latest.csv 2>/dev/null | cut -d, -f1)
SIM_SPAN=$(awk -v a="${SIM_T0:-0}" -v b="${SIM_T1:-0}" 'BEGIN{printf "%.0f", b-a}')
if [ -n "$ABORT_REASON" ]; then
    echo "  >>> Mission ABORTED at ${ELAPSED}s: $ABORT_REASON <<<"
else
    echo "  >>> Mission complete <<<"
fi
# Wall-vs-sim visibility: the loop counts WALL seconds against a SIM-second
# DURATION (fine at RTF=1.00) — any future RTF drop shows up right here.
echo "  >>> Mission span: ${ELAPSED}s wall / ${SIM_SPAN}s sim <<<"
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
    [ -n "$ABORT_REASON" ] && \
        echo "RUN ABORTED at ${ELAPSED}s/${DURATION}s: $ABORT_REASON" >> "$ANALYSIS_OUT"
    echo "[Metrics] Extracting summary metrics..."
    ABORTED_FLAG=0; [ -n "$ABORT_REASON" ] && ABORTED_FLAG=1
    python3 $PKG_DIR/scripts/extract_metrics.py \
        --csv "$CSV_OUT" --config $CONFIG \
        --zone $ZONE --traj $TRAJ \
        --seed $SEED --duration $DURATION --aborted $ABORTED_FLAG
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
echo "  Summary:  $RESULTS_BASE/summary.csv"
echo "  (manual runs: ./sync_results.sh to mirror to OneDrive)"
echo "================================================================"
echo ""
