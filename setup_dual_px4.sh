
#!/bin/bash

# setup_dual_px4.sh — One-time setup for dual-PX4 SITL

# Creates a modified iris SDF with instance 1 mavlink ports for the target drone.

#

# Run once:  bash setup_dual_px4.sh

#

# What it does:

#   1. Copies iris.sdf → target_iris_sitl/iris.sdf

#   2. Changes mavlink_udp_port  14560 → 14561

#   3. Changes mavlink_tcp_port  4560  → 4561

#   These port offsets match PX4 SITL instance 1 (-i 1).

set -e

SRC_SDF="$HOME/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf"

DST_DIR="$HOME/PX4-Autopilot/Tools/sitl_gazebo/models/target_iris_sitl"

DST_SDF="$DST_DIR/iris.sdf"

if [ ! -f "$SRC_SDF" ]; then

    echo "ERROR: Source SDF not found at $SRC_SDF"

    exit 1

fi

mkdir -p "$DST_DIR"

cp "$SRC_SDF" "$DST_SDF"

# Patch mavlink ports for instance 1

sed -i 's|<mavlink_udp_port>14560</mavlink_udp_port>|<mavlink_udp_port>14561</mavlink_udp_port>|g' "$DST_SDF"

sed -i 's|<mavlink_tcp_port>4560</mavlink_tcp_port>|<mavlink_tcp_port>4561</mavlink_tcp_port>|g' "$DST_SDF"

echo "=== Created target SDF at $DST_SDF ==="

echo "  mavlink_udp_port: 14561"

echo "  mavlink_tcp_port: 4561"

echo ""

echo "Verify ports:"

grep -n "mavlink_udp_port\|mavlink_tcp_port" "$DST_SDF"

echo ""

echo "=== Done. See terminal sequence below ==="

echo ""

echo "--- TERMINAL SEQUENCE FOR DUAL-PX4 ---"

echo ""

echo "T1  — Gazebo + Chaser PX4 + Chaser MAVROS (existing launch, unchanged)"

echo "T2  — YOLO detection"

echo "T3  — Kalman filter"

echo "T4a — Spawn target:  rosrun gazebo_ros spawn_model -sdf -file $DST_SDF -model target_iris -x 5 -y 0 -z 0.05"

echo "T4b — Target PX4:    cd ~/PX4-Autopilot && PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_NAME=target_iris ./build/px4_sitl_default/bin/px4 -i 1 -d ./build/px4_sitl_default/etc"

echo "T4c — Target MAVROS: roslaunch ~/catkin_ws/src/drone_tracking/launch/target_mavros.launch"

echo "T5  — Takeoff both:  rosrun drone_tracking takeoff_both.py"

echo "T6  — IBVS controller"

echo "T7  — Target mover:  rosrun drone_tracking target_mover.py"

echo "T8  — YOLO debug viewer"

echo "T9  — Flight logger"

