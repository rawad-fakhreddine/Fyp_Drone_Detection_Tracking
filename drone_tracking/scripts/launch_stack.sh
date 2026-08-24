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
#   VIEWER_SCALE=N  : viewer display upscale (default 1.0 = native 640x480)
#   VIEWER_HZ=N     : viewer render rate (default 30 = camera rate)
#   START_DIST=N    : spawn separation in m (default: random 8-12)
#   LOSS_TIMEOUT=N  : abort run after N s continuous SEARCH (default 10)
#   ALPHA_STAR / DEAD_ZONE / EA_HOLD : IBVS standoff knobs (defaults = baseline)
#   VX_MODE=pid     : v6.31 distance-domain vx PID (default legacy = baseline)
#   D_STAR / KP_VX / KI_VX / HOLD_D_TOL : vx PID knobs (8.0 / 1.0 / 0.0 / 1.0)
#   ALPHA_DIST_K=N  : alpha->distance constant (0.0625; measured 0.084 on T3/z1)
#   SIZE_GATE=off   : disable the detector's size/aspect plausibility filter
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
# NO_LOCKSTEP=1: use px4_sitl_nolockstep binary + nolockstep SDF models + PX4_SIM_SPEED_FACTOR.
# C1/C2 runs NEVER set NO_LOCKSTEP (default=0 → px4_sitl_default, unchanged behaviour).
PX4_BUILD_TYPE="px4_sitl_default"
if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
    PX4_BUILD_TYPE="px4_sitl_nolockstep"
    SPEED_FACTOR="${SPEED_FACTOR:-4}"
    export PX4_SIM_SPEED_FACTOR="$SPEED_FACTOR"
    echo "[T1] NO_LOCKSTEP mode: build=$PX4_BUILD_TYPE  speed=${SPEED_FACTOR}x"
fi
source $PX4_HOME/Tools/setup_gazebo.bash $PX4_HOME $PX4_HOME/build/$PX4_BUILD_TYPE 2>/dev/null
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
    # Uses a Python rospy subscriber — more reliable than rostopic|grep when
    # use_sim_time=false (nolockstep), where rostopic echo -n1 can fail to
    # deliver latched messages after the publisher node has exited.
    local topic="$1" timeout="${2:-90}"
    echo -n "  Waiting for $topic == True ."
    if timeout "$timeout" python3 - "$topic" "$timeout" <<'PYEOF'
import sys, rospy
from std_msgs.msg import Bool
rospy.init_node('wait_bool_true', anonymous=True)
try:
    msg = rospy.wait_for_message(sys.argv[1], Bool, timeout=float(sys.argv[2]))
    sys.exit(0 if msg.data else 1)
except rospy.ROSException:
    sys.exit(1)
PYEOF
    then
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
# T0 — Windows power plan -> High performance while the sim runs (fan curve
# ramps under load; restored to Balanced by cleanup.sh). FAN_BOOST=0 disables.
if [ "${FAN_BOOST:-1}" = "1" ]; then
    bash "$PKG_DIR/scripts/fan_boost.sh" on
    echo "[T0] Windows power plan -> High performance (FAN_BOOST=0 to skip)"
fi
# T0b — code provenance: stamp WHICH code + scenario env flies this run
# (Rawad's directive 2026-07-17: revise by fact, not by guessing).
bash "$PKG_DIR/scripts/code_stamp.sh" "$RESULTS_DIR/$RUN_TAG" 2>/dev/null || true

# SPEED_FACTOR>1 (RL training): PX4 lockstep runs the sim faster than real-time so
# each 100 s episode finishes in less wall time. Default unset = 1× = unchanged C1/C2 behaviour.
# NOTE: PX4_SIM_SPEED_FACTOR is for nolockstep ONLY. In lockstep, PX4's hrt tracks
# sim_time from HIL_SENSOR, so scaling the internal clock causes EKF filter faults.
# Lockstep speedup comes from real_time_factor=2 in the world file (Gazebo runs 2×).
if [ -n "${SPEED_FACTOR:-}" ] && [ "${SPEED_FACTOR}" != "1" ]; then
    if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
        export PX4_SIM_SPEED_FACTOR="${SPEED_FACTOR}"
        echo "[T1] SIM SPEED FACTOR = ${SPEED_FACTOR}x (nolockstep wall-clock scaling)"
    else
        echo "[T1] SIM SPEED FACTOR = ${SPEED_FACTOR}x (lockstep faster-than-real-time)"
    fi
fi
# HEADLESS=1 (RL training): no gzclient GUI → frees GPU/CPU, faster sim + env loop.
# The camera SENSOR still renders offscreen in gzserver (YOLO input); only the GUI
# window is dropped. Default off = gui:=true = unchanged C1/C2 behaviour.
GUI_ARG=true
if [ "${HEADLESS:-0}" = "1" ]; then GUI_ARG=false; echo "[T1] HEADLESS (no gzclient GUI — camera sensor still feeds YOLO)"; fi
# WORLD: override to a lighter world for RL training (e.g. WORLD=rl_empty).
# baylands.world = default (C1/C2 baseline); rl_empty = flat + sky only (faster physics).
WORLD_FILE="${WORLD:-baylands}.world"
WORLD_PATH="$PX4_HOME/Tools/sitl_gazebo/worlds/${WORLD_FILE}"
if [ ! -f "$WORLD_PATH" ]; then
    echo "[T1] WARNING: world $WORLD_PATH not found, falling back to baylands.world"
    WORLD_PATH="$PX4_HOME/Tools/sitl_gazebo/worlds/baylands.world"
fi
# Select chaser SDF: nolockstep variant for RL, standard for C1/C2
CHASER_SDF="$PX4_HOME/Tools/sitl_gazebo/models/iris_fpv_cam/iris_fpv_cam.sdf"
if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
    CHASER_SDF="$PX4_HOME/Tools/sitl_gazebo/models/iris_fpv_cam_nolockstep/iris_fpv_cam_nolockstep.sdf"
fi
if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
    # NOLOCKSTEP: split launch to fix TIMESYNC.
    # Root cause: libgazebo_ros_api_plugin.so forces /use_sim_time=true in its Load()
    # regardless of launch file args. MAVROS then timestamps setpoints with sim_time≈0,
    # but PX4 nolockstep uses wall_time (~1.787e9 s) → setpoints appear 56 years stale
    # → rejected → OFFBOARD drops immediately after ARM.
    # Fix: launch Gazebo+PX4 first (no MAVROS), wait for plugin Load() to finish,
    # then override use_sim_time=false, then launch MAVROS with wall-clock time.
    echo "[T1] Nolockstep: Gazebo + Chaser PX4 only (world=$(basename $WORLD_PATH)  speed=${SPEED_FACTOR}x)..."
    roslaunch $PX4_HOME/launch/posix_sitl.launch \
        vehicle:=iris \
        interactive:=false \
        gui:=$GUI_ARG \
        build_type:=$PX4_BUILD_TYPE \
        sdf:=$CHASER_SDF \
        world:=$WORLD_PATH \
        > /tmp/T1_${RUN_TAG}.log 2>&1 &
    # Wait for Gazebo to publish /clock (plugin Load() ran, use_sim_time forced true)
    wait_topic "/clock" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
    sleep 3   # ensure libgazebo_ros_api_plugin Load() has set /use_sim_time=true
    rosparam set /use_sim_time false
    echo "[T1b] Nolockstep: Chaser MAVROS (wall-time mode — matches PX4 hrt_absolute_time)..."
    # Use a dedicated launch file (node.launch directly with our config yaml).
    # px4.launch doesn't expose config_yaml as overridable; node.launch CLI has empty-string quoting issues.
    roslaunch $PKG_DIR/launch/chaser_mavros_nolockstep.launch \
        > /tmp/T1b_${RUN_TAG}.log 2>&1 &
else
    # LOCKSTEP (C1/C2 default): combined launch, unchanged behaviour.
    # Clear chaser eeprom so PX4 boots with GPS defaults (EKF2_AID_MASK=1, HGT_MODE=0).
    # Root cause: nolockstep sessions write EKF2_AID_MASK=264 (EV-only) to eeprom via
    # mavparam. In lockstep, no vision pose is published, so EKF2 starts blind → position
    # invalid → OFFBOARD switch fails → ARM denied ("manual control lost"). The runtime
    # mavparam restore (below) only writes the new value to eeprom for the NEXT boot;
    # it does NOT hot-reload the already-running EKF2 instance. Clearing the eeprom
    # BEFORE PX4 starts is the only reliable fix.
    _CHASER_EEPROM="$HOME/.ros/eeprom/parameters_10016"
    if [ -f "$_CHASER_EEPROM" ]; then
        rm -f "$_CHASER_EEPROM"
        echo "  Cleared chaser eeprom (GPS defaults on boot, no EV poison)"
    fi
    echo "[T1] Gazebo + Chaser PX4 + MAVROS (world=$(basename $WORLD_PATH)  lockstep=1)..."
    roslaunch $PX4_HOME/launch/mavros_posix_sitl.launch \
        vehicle:=iris \
        interactive:=false \
        gui:=$GUI_ARG \
        build_type:=$PX4_BUILD_TYPE \
        sdf:=$CHASER_SDF \
        world:=$WORLD_PATH \
        use_sim_time:=true \
        > /tmp/T1_${RUN_TAG}.log 2>&1 &
fi
# Pin gzserver to P-core threads (0-7 on i5-13420H; 8-11 are E-cores) to keep the
# single-threaded ODE physics loop off the slower E-cores. Scheduling-only — does
# not change lockstep results. Gated to the RL world so C1/C2 baseline runs (baylands)
# are byte-for-byte untouched.
if [ "${WORLD:-}" = "rl_empty" ]; then
    for _i in $(seq 1 20); do
        _GZ_PID=$(pgrep -x gzserver 2>/dev/null | head -1 || true)
        [ -n "$_GZ_PID" ] && break || sleep 1
    done
    if [ -n "${_GZ_PID:-}" ]; then
        taskset -cp 0-7 "$_GZ_PID" >/dev/null 2>&1 \
            && echo "  gzserver (PID=$_GZ_PID) pinned to P-core threads 0-7" || true
    fi
fi
wait_topic "/mavros/state" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
# Gate: chaser PX4 alive and linked to MAVROS (= baylands fully loaded and the
# sim stepping) BEFORE the zone teleport — topic existence alone fires ~30 s
# too early, while the world is still loading.
wait_fcu "/mavros/state" 90 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
sleep 5
# EKF restore: nolockstep sessions bake EKF2_AID_MASK=264 (EV-only, no GPS) and
# EKF2_HGT_MODE=3 (vision height) into the chaser eeprom. In lockstep, no vision
# pose is published → EKF blind → local_position_valid=false → OFFBOARD fails →
# PX4 stays STABILIZED → ARM denied "manual control lost". Restore GPS defaults.
if [ "${NO_LOCKSTEP:-0}" != "1" ]; then
    for _i in 1 2 3; do
        timeout 10 rosrun mavros mavparam set EKF2_AID_MASK 1 >/dev/null 2>&1 \
            && echo "  chaser EKF2_AID_MASK=1 (GPS, lockstep restore)" && break \
            || sleep 3
    done
    for _i in 1 2 3; do
        timeout 10 rosrun mavros mavparam set EKF2_HGT_MODE 0 >/dev/null 2>&1 \
            && echo "  chaser EKF2_HGT_MODE=0 (baro, lockstep restore)" && break \
            || sleep 3
    done
fi

# T2 — Spawn target + teleport chaser
# START_DIST overrides the spawn separation (default -1 → node picks 8-12m)
echo "[T2] Spawning target + positioning chaser (zone=$ZONE seed=$SEED dist=${START_DIST:--1})..."
rosrun drone_tracking random_spawn_target.py _zone:=$ZONE _seed:=$SEED \
    _dist:=${START_DIST:--1} _spawn_yaw:=${SPAWN_YAW:-999} \
    > /tmp/T2_${RUN_TAG}.log 2>&1 &
sleep 4

# T2b — Spectator camera (M-C: static 3rd-person overlook of the arena).
# Env-tunable placement so the view can be adjusted without editing. Optical
# axis = model +X; default aims north (+Y, yaw 1.5708) and ~16 deg down.
if [ "${SPECTATOR:-0}" = "1" ]; then
    echo "[T2b] Spawning spectator camera (static overlook)..."
    rosrun gazebo_ros spawn_model -sdf \
        -file $PKG_DIR/models/spectator_cam.sdf -model spectator_cam \
        -x ${SPEC_X:--5.5} -y ${SPEC_Y:-5} -z ${SPEC_Z:-25} \
        -R 0 -P ${SPEC_PITCH:-0.28} -Y ${SPEC_YAW:-1.5708} \
        > /tmp/T2b_${RUN_TAG}.log 2>&1
    sleep 2
    # SPEC_TRACK=1 → the camera follows the drones (midpoint + offset).
    if [ "${SPEC_TRACK:-0}" = "1" ]; then
        echo "[T2c] Spectator tracker (camera follows the drones)..."
        rosrun drone_tracking spectator_tracker.py \
            _off_x:=${SPEC_OFF_X:--8} _off_y:=${SPEC_OFF_Y:--8} _off_z:=${SPEC_OFF_Z:-9} \
            > /tmp/T2c_${RUN_TAG}.log 2>&1 &
    fi
fi

# T3 — YOLO
# SIZE_GATE=off opens the v3.3 plausibility bounds wide (size/aspect filter
# disabled; persistence gate + conf hysteresis stay). A/B knob — measured
# rejects were 0/100f on T3/zone1, but this lets it be verified per-run.
if [ "${SIZE_GATE:-on}" = "off" ]; then
    echo "[T3] YOLO detection node (SIZE_GATE=off — plausibility filter open)..."
    DET_GATE_ARGS="_min_box_px:=0 _max_box_px:=100000 _aspect_min:=0.001 _aspect_max:=1000"
else
    echo "[T3] YOLO detection node..."
    DET_GATE_ARGS=""
fi
rosrun drone_tracking yolo_detection_node.py $DET_GATE_ARGS > /tmp/T3_${RUN_TAG}.log 2>&1 &
sleep 3

# TV — YOLO debug viewer (live detection overlay window; default OFF so batch
# runs stay headless. VIEWER=1 opens the window). Killed by cleanup.sh.
if [ "${VIEWER:-0}" = "1" ]; then
    echo "[TV] YOLO debug viewer (VIEWER=1)..."
    rosrun drone_tracking yolo_debug_viewer.py \
        _scale:=${VIEWER_SCALE:-1.0} _hz:=${VIEWER_HZ:-30} \
        > /tmp/TV_${RUN_TAG}.log 2>&1 &
else
    echo "[TV] Viewer SKIPPED (VIEWER=0 / default headless)"
fi

# SNAP — headless periodic camera-view snapshots (SNAP=1). Saves annotated
# PNGs every SNAP_INTERVAL (default 3) sim-s into a per-run folder for offline
# visual inspection. Independent of the debug VIEWER window.
if [ "${SNAP:-0}" = "1" ]; then
    SNAP_DIR="$RESULTS_BASE/screenshots/${CFG_DIR}_${RUN_TAG}"
    echo "[SNAP] Snapshot node -> $SNAP_DIR"
    rosrun drone_tracking sim_snapshot.py \
        _snap_dir:="$SNAP_DIR" _snap_interval:=${SNAP_INTERVAL:-3.0} \
        > /tmp/SNAP_${RUN_TAG}.log 2>&1 &
    echo "$SNAP_DIR" > /tmp/last_snap_dir.txt
fi

# RC — headless demo mp4 recorder (RECORD=1). Annotated chaser-camera view
# (raw box/conf/state/phase/FOV/GT-dist), starts at takeoff_ready, finalized
# on teardown. RECORD_SCALE upscales for readability (default 1.5).
if [ "${RECORD:-0}" = "1" ]; then
    REC_DIR="$RESULTS_BASE/demo_videos"
    mkdir -p "$REC_DIR"
    REC_PATH="$REC_DIR/${CFG_DIR}_${RUN_TAG}.mp4"
    echo "[RC] Demo recorder -> $REC_PATH"
    rosrun drone_tracking demo_recorder.py \
        _out:="$REC_PATH" _scale:=${RECORD_SCALE:-1.0} \
        > /tmp/RC_${RUN_TAG}.log 2>&1 &
fi

# MV — 3-panel multiview recorder (MULTIVIEW=1; pair with SPECTATOR=1). Composes
# chaser FOV + spectator cam + live 3-D trajectory into one mp4 (M-C).
if [ "${MULTIVIEW:-0}" = "1" ]; then
    MV_DIR="$RESULTS_BASE/multiview"; mkdir -p "$MV_DIR"
    MV_PATH="$MV_DIR/multiview_${CFG_DIR}_${RUN_TAG}.mp4"
    echo "[MV] Multiview recorder -> $MV_PATH"
    rosrun drone_tracking multiview_recorder.py \
        _out:="$MV_PATH" _fps:=${MULTIVIEW_FPS:-40} \
        > /tmp/MV_${RUN_TAG}.log 2>&1 &
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
PX4_BUILD="$PX4_HOME/build/$PX4_BUILD_TYPE"   # nolockstep binary when NO_LOCKSTEP=1
TGT_WORKDIR="$PX4_BUILD/instance_1"
mkdir -p "$TGT_WORKDIR"
(cd "$TGT_WORKDIR" && \
 PX4_SIM_MODEL=iris \
 "$PX4_BUILD/bin/px4" -i 1 -d "$PX4_BUILD/etc" -w sitl_iris_1 -s etc/init.d-posix/rcS \
 > /tmp/T5_${RUN_TAG}.log 2>&1) &
sleep 10

# T6 — Target MAVROS
if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
    echo "[T6] Target MAVROS (nolockstep — timesync disabled)..."
    roslaunch $PKG_DIR/launch/target_mavros_nolockstep.launch > /tmp/T6_${RUN_TAG}.log 2>&1 &
else
    echo "[T6] Target MAVROS..."
    roslaunch $PKG_DIR/launch/target_mavros.launch > /tmp/T6_${RUN_TAG}.log 2>&1 &
fi
wait_topic "/target/mavros/state" 30 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
# Gate: target PX4 heartbeating through MAVROS (implies instance 1 attached to
# Gazebo on TCP 4561 and rcS completed — heartbeats only start after that),
# i.e. the lockstep freeze is over and sim time is advancing again.
wait_fcu "/target/mavros/state" 60 || { bash $PKG_DIR/scripts/cleanup.sh; exit 1; }
# 2026-07-11 trajectory audit: PX4 default MPC_Z_VEL_MAX_DN=1.0 capped the
# TARGET's descent -> T7 flew 22.7 deg instead of the 35 spec (needs vz=1.72;
# climb legs were fine at 34.5). Raise the target's descent limit to 2.0.
# NB also uncaps T9 evasion dives (fuzzy commands down to -2) -> T9 baseline note.
# retry: the param service is only available once MAVROS finishes its FCU
# param sync (a few seconds after connect); a single immediate call failed.
ZDN_OK=0
for _i in 1 2 3 4 5 6; do
  if timeout 15 rosrun mavros mavparam -n /target/mavros set MPC_Z_VEL_MAX_DN 2.0 >/dev/null 2>&1; then
    ZDN_OK=1; echo "  target MPC_Z_VEL_MAX_DN=2.0 (try $_i)"; break
  fi
  sleep 5
done
[ "$ZDN_OK" = "1" ] || echo "  WARN: target Z_VEL_MAX_DN set failed after retries"
# EKF2 noise scaling for nolockstep speedup.
# In nolockstep, PX4's HRT = wall-clock (gettimeofday), but Gazebo physics runs at
# real_time_factor=SPEED_FACTOR. EKF integrates IMU using wall-clock dt (= sim_dt/SPEED_FACTOR),
# so GPS velocity/position innovations are SPEED_FACTOR× larger than 1× operation.
# Bake SPEED_FACTOR²×default into eeprom via mavparam (eeprom persists; these calls
# are idempotent updates — timeout protects against param-service hang).
if [ "${NO_LOCKSTEP:-0}" = "1" ] && [ -n "${SPEED_FACTOR:-}" ] && [ "${SPEED_FACTOR}" != "1" ]; then
    EKF_V=$(echo "$SPEED_FACTOR * $SPEED_FACTOR * 0.3" | bc)
    EKF_P=$(echo "$SPEED_FACTOR * $SPEED_FACTOR * 0.5" | bc)
    echo "  [EKF] nolockstep speed=${SPEED_FACTOR}x: EKF2_GPS_V_NOISE=$EKF_V  EKF2_GPS_P_NOISE=$EKF_P"
    timeout 15 rosrun mavros mavparam set EKF2_GPS_V_NOISE "$EKF_V" >/dev/null 2>&1 && \
        echo "  chaser EKF2_GPS_V_NOISE=$EKF_V" || echo "  WARN: chaser EKF2_GPS_V_NOISE set failed"
    timeout 15 rosrun mavros mavparam -n /target/mavros set EKF2_GPS_V_NOISE "$EKF_V" >/dev/null 2>&1 && \
        echo "  target EKF2_GPS_V_NOISE=$EKF_V" || echo "  WARN: target EKF2_GPS_V_NOISE set failed"
    timeout 15 rosrun mavros mavparam set EKF2_GPS_P_NOISE "$EKF_P" >/dev/null 2>&1 && \
        echo "  chaser EKF2_GPS_P_NOISE=$EKF_P" || echo "  WARN: chaser EKF2_GPS_P_NOISE set failed"
    timeout 15 rosrun mavros mavparam -n /target/mavros set EKF2_GPS_P_NOISE "$EKF_P" >/dev/null 2>&1 && \
        echo "  target EKF2_GPS_P_NOISE=$EKF_P" || echo "  WARN: target EKF2_GPS_P_NOISE set failed"
    # AID_MASK=264=EV_pos(8)+EV_vel(256). Eeprom bakes this; mavparam belt-and-suspenders.
    # v8(AID=9): GPS vel wrong → "horiz vel not stable". v10(AID=265): GPS+EVvel conflict
    # → "Position estimate error". Fix: AID=264 = EV pos+vel ONLY. No GPS conflict.
    # EV pos=GT exact position; EV vel=finite-diff GT. Both stable at any speed factor.
    timeout 15 rosrun mavros mavparam set EKF2_AID_MASK 264 >/dev/null 2>&1 && \
        echo "  chaser EKF2_AID_MASK=264 (EVpos+EVvel)" || echo "  WARN: chaser EKF2_AID_MASK set failed"
    timeout 15 rosrun mavros mavparam -n /target/mavros set EKF2_AID_MASK 264 >/dev/null 2>&1 && \
        echo "  target EKF2_AID_MASK=264 (EVpos+EVvel)" || echo "  WARN: target EKF2_AID_MASK set failed"
    timeout 15 rosrun mavros mavparam set EKF2_HGT_MODE 3 >/dev/null 2>&1 && \
        echo "  chaser EKF2_HGT_MODE=3 (vision)" || echo "  WARN: chaser EKF2_HGT_MODE set failed"
    timeout 15 rosrun mavros mavparam -n /target/mavros set EKF2_HGT_MODE 3 >/dev/null 2>&1 && \
        echo "  target EKF2_HGT_MODE=3 (vision)" || echo "  WARN: target EKF2_HGT_MODE set failed"
    timeout 15 rosrun mavros mavparam set EKF2_EVV_NOISE 0.1 >/dev/null 2>&1 && \
        echo "  chaser EKF2_EVV_NOISE=0.1" || echo "  WARN: chaser EKF2_EVV_NOISE set failed"
    timeout 15 rosrun mavros mavparam -n /target/mavros set EKF2_EVV_NOISE 0.1 >/dev/null 2>&1 && \
        echo "  target EKF2_EVV_NOISE=0.1" || echo "  WARN: target EKF2_EVV_NOISE set failed"
fi
# T6b — Vision GT bridge (nolockstep only).
# Bridges /gazebo/model_states → /mavros/vision_pose/pose for both drones.
# Publishes RELATIVE positions (spawn-zeroed) so EKF local-frame innovations are <1m.
# (Absolute Gazebo world coords → 30+ m innovations → silently rejected, cs_ev_pos=0.)
# AID_MASK=264=EVpos+EVvel only (no GPS). EV velocity (exact GT) vs GPS vel (half at 2×/4×)
# → GPS-EV conflict on AID=265 → "Position estimate error". AID=264 avoids the conflict.
# HGT_MODE=3: vision height. EVP_NOISE=0.05m, EVV_NOISE=0.1 m/s (tight GT).
if [ "${NO_LOCKSTEP:-0}" = "1" ]; then
    echo "[T6b] Vision GT bridge (Gazebo GT → EKF2 vision fusion)..."
    rosrun drone_tracking gz_gt_to_vision.py > /tmp/T6b_${RUN_TAG}.log 2>&1 &
    sleep 2
else
    # Restore target EKF to GPS defaults (same eeprom-poison risk as chaser)
    for _i in 1 2 3; do
        timeout 10 rosrun mavros mavparam -n /target/mavros set EKF2_AID_MASK 1 >/dev/null 2>&1 \
            && echo "  target EKF2_AID_MASK=1 (GPS, lockstep restore)" && break \
            || sleep 3
    done
    for _i in 1 2 3; do
        timeout 10 rosrun mavros mavparam -n /target/mavros set EKF2_HGT_MODE 0 >/dev/null 2>&1 \
            && echo "  target EKF2_HGT_MODE=0 (baro, lockstep restore)" && break \
            || sleep 3
    done
fi
# CHASER descent cap (T6/T7/T8 vertical trajectories ONLY, env-gated:
# CHASER_ZDN=2.5). NOT unconditional: the 2026-07-11 regression-gate failure
# showed the flat-T3 bench destabilized (pitch +/-10 deg, dist std 1.6) with
# it set — and PX4 persists mavparam writes to eeprom, so it leaked across
# runs. Default = explicitly RESTORE the PX4 default 1.0 every run.
CZDN=${CHASER_ZDN:-1.0}
for _i in 1 2 3 4 5 6; do
  if timeout 15 rosrun mavros mavparam set MPC_Z_VEL_MAX_DN "$CZDN" >/dev/null 2>&1; then
    echo "  chaser MPC_Z_VEL_MAX_DN=$CZDN (try $_i)"; break
  fi
  sleep 5
done
sleep 3

# T7 — IBVS (skipped when SKIP_IBVS=1, e.g. for RL SAC training where rl_train_sac.py owns the setpoint)
if [ "${SKIP_IBVS:-0}" = "1" ]; then
    echo "[T7] IBVS SKIPPED (SKIP_IBVS=1 — RL policy owns the setpoint topic)"
else
echo "[T7] IBVS (use_ppo=$USE_PPO detection_source=$DET_SRC)..."
rosrun drone_tracking ibvs_controller_node.py \
    _use_ppo:=$USE_PPO _detection_source:=$DET_SRC \
    _pred_vx_hold:=${PRED_VX_HOLD:-1} _pred_accel_max:=${PRED_ACCEL_MAX:-1.0} _d_safe_margin:=${D_SAFE_MARGIN:-1.0} \
    _search_latch:=${SEARCH_LATCH:-false} _search_tau:=${SEARCH_TAU:-1.0} \
    _Kp_wz:=${KP_WZ:-2.0} _Kp_y:=${KP_Y:-0.4} \
    _Kp_z:=${KP_Z:-3.0} _max_vz:=${MAX_VZ:-1.5} \
    _Ki_z:=${KI_Z:-0.04} _int_z_max:=${INT_Z_MAX:-0.2} _int_z_bleed:=${INT_Z_BLEED:-1.0} \
    _vz_ff_gain:=${VZ_FF_GAIN:-0.0} _vz_ff_cap:=${VZ_FF_CAP:-1.5} \
    _lambda_x:=${LAMBDA_X:-1.0} _lambda_y:=${LAMBDA_Y:-1.0} \
    _lambda_z:=${LAMBDA_Z:-1.0} _lambda_wz:=${LAMBDA_WZ:-1.0} \
    _lambda_sched:=${LAMBDA_SCHED:-0} _lambda_min:=${LAMBDA_MIN:-0.6} \
    _lambda_e_ref:=${LAMBDA_E_REF:-3.0} _lambda_v_ref:=${LAMBDA_V_REF:-2.0} _lambda_slew:=${LAMBDA_SLEW:-0.03} \
    _alpha_star:=${ALPHA_STAR:-0.0067} _dead_zone:=${DEAD_ZONE:-0.002} \
    _ea_hold:=${EA_HOLD:-0.010} \
    _vx_mode:=${VX_MODE:-pid} _d_star:=${D_STAR:-8.0} \
    _Kp_vx:=${KP_VX:-2.5} _Ki_vx:=${KI_VX:-1.2} _Kd_vx:=${KD_VX:-1.5} _vx_ff:=${VX_FF:-0.0} _vff_lpf:=${VFF_LPF:-0.9} \
    _int_d_max:=${INT_D_MAX:-6.0} _int_band:=${INT_BAND:-2.5} _int_bleed:=${INT_BLEED:-1.0} _int_hold_only:=${INT_HOLD_ONLY:-1} _band_kp:=${BAND_KP:-0.4} \
    _hold_d_tol:=${HOLD_D_TOL:-1.0} _alpha_dist_k:=${ALPHA_DIST_K:-0.077} \
    _d_emerg:=${D_EMERG:-0.0} _d_lpf:=${D_LPF:-0.50} _a_dec:=${A_DEC:-2.0} \
    _d_hold_min:=${D_HOLD_MIN:-6.0} _d_hold_max:=${D_HOLD_MAX:-7.0} \
    _min_dist:=${MIN_DIST:-2.5} _emerg_vy:=${EMERG_VY:-2.0} \
    _emerg_brake_vx:=${EMERG_BRAKE_VX:-4.0} \
    _max_accel:=${MAX_ACCEL:-3.0} _max_accel_fast:=${MAX_ACCEL_FAST:-8.0} \
    _max_accel_vz:=${MAX_ACCEL_VZ:-3.0} \
    _alt_floor:=${ALT_FLOOR:-11.0} \
    _max_vx_retreat_pid:=${MAX_VX_RETREAT_PID:-2.5} _max_vx:=${MAX_VX:-8.0} \
    _pitch_comp:=${PITCH_COMP:-1.3} _deriv_lpf:=${DERIV_LPF:-0.6} \
    > /tmp/T7_${RUN_TAG}.log 2>&1 &
sleep 2
fi  # end SKIP_IBVS guard

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
    _straight_az:=${STRAIGHT_AZ:-random} _straight_max:=${STRAIGHT_MAX:-60} \
    _straight_offset:=${STRAIGHT_OFFSET:-0.0} \
    _traj_track_kp:=${TRAJ_TRACK_KP:-0.0} _traj_track_cap:=${TRAJ_TRACK_CAP:-2.5} \
    _traj_track_kp_z:=${TRAJ_TRACK_KP_Z:-${TRAJ_TRACK_KP:-0.0}} \
    _traj_track_ki_z:=${TRAJ_TRACK_KI_Z:-0.0} _z_hold:=${Z_HOLD:-0} \
    _incline_oneway:=${INCLINE_ONEWAY:-0} _traj_track_lead:=${TRAJ_TRACK_LEAD:-0.0} \
    _incline_no_hreverse:=${INCLINE_NO_HREVERSE:-0} \
    _lemn_a:=${LEMN_A:-8.0} \
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
# When IBVS is skipped (RL training mode) there is no SEARCH phase → disable the
# loss watchdog (otherwise it fires in 10 s and aborts the training run).
[ "${SKIP_IBVS:-0}" = "1" ] && LOSS_TIMEOUT=3600
# Stuck-target watchdog: if the target's Gazebo world position stops changing
# (hit a tree / terrain and wedged) for STUCK_TIMEOUT consecutive seconds, the
# run is dead — abort. Only for MOVING trajectories (T1 is a static hover, so
# a frozen position there is correct). Reads target_wx/wy by header name.
STUCK_TIMEOUT="${STUCK_TIMEOUT:-8}"
MOVING_TRAJ=1; [ "$TRAJ" = "1" ] && MOVING_TRAJ=0
get_tgt_xy() {   # prints "x y" of the target from the last logged row
    python3 -c "
import csv
try:
    rows=list(csv.DictReader(open('$HOME/flight_log_latest.csv')))
    r=rows[-1]; print(r.get('target_wx','') or 'nan', r.get('target_wy','') or 'nan')
except Exception: print('nan nan')
" 2>/dev/null
}
echo ""
echo "  >>> Mission running for ${DURATION}s (loss ${LOSS_TIMEOUT}s, stuck ${STUCK_TIMEOUT}s) <<<"
SIM_T0=$(tail -n 1 ~/flight_log_latest.csv 2>/dev/null | cut -d, -f1)
ELAPSED=0; LOST=0; STUCK=0; ABORT_REASON=""; ACQUIRED=0
PREV_TX=""; PREV_TY=""
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
    # stuck-target check (moving trajectories only, after acquisition)
    if [ $MOVING_TRAJ -eq 1 ] && [ $ACQUIRED -eq 1 ]; then
        read TX TY < <(get_tgt_xy)
        if [ -n "$PREV_TX" ] && [ "$TX" != "nan" ] && [ "$PREV_TX" != "nan" ]; then
            MOVED=$(awk -v a="$TX" -v b="$TY" -v c="$PREV_TX" -v d="$PREV_TY" \
                    'BEGIN{printf "%.2f", sqrt((a-c)^2+(b-d)^2)}')
            # <0.4 m over 5 s = ~stationary (a 3.5 m/s target covers ~17 m)
            if awk -v m="$MOVED" 'BEGIN{exit !(m<0.4)}'; then
                STUCK=$((STUCK+5))
            else
                STUCK=0
            fi
        fi
        PREV_TX="$TX"; PREV_TY="$TY"
        if [ $STUCK -ge $STUCK_TIMEOUT ]; then
            ABORT_REASON="target STUCK ${STUCK}s (>= ${STUCK_TIMEOUT}s) — tree/terrain strike"
            break
        fi
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
