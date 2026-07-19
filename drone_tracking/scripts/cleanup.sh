#!/bin/bash
# cleanup.sh — full teardown of dual-PX4 + Gazebo + MAVROS stack.
# Kills in dependency order, force-kills Gazebo, waits for ports to free.

echo "[cleanup] killing ROS nodes..."
rosnode kill -a 2>/dev/null
sleep 2

echo "[cleanup] killing PX4 instances..."
pkill -9 -f 'bin/px4' 2>/dev/null
pkill -9 -x px4       2>/dev/null

echo "[cleanup] killing MAVROS..."
pkill -9 -f mavros_node 2>/dev/null
pkill -9 -f mavros      2>/dev/null

echo "[cleanup] killing Gazebo (force)..."
killall -9 gzserver gzclient 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null

echo "[cleanup] killing project nodes..."
for n in yolo_detection_node kalman_filter_node ibvs_controller_node \
         target_mover takeoff_both flight_logger ppo_agent_node \
         yolo_debug_viewer random_spawn_target; do
    pkill -9 -f "$n" 2>/dev/null
done

echo "[cleanup] killing roslaunch / roscore..."
pkill -9 -f roslaunch 2>/dev/null
killall -9 roscore rosmaster rosout 2>/dev/null
sleep 2

echo "[cleanup] waiting for ports to release..."
for port in 11311 4560 4561 14540 14541 14560 14561; do
    tries=0
    while lsof -i :$port -sTCP:LISTEN >/dev/null 2>&1 || \
          fuser $port/udp >/dev/null 2>&1; do
        sleep 1; tries=$((tries+1))
        [ $tries -ge 15 ] && { echo "  port $port STUCK"; break; }
    done
done

# Clear gazebo locks / stale logs
rm -rf ~/.gazebo/log/* 2>/dev/null
pkill -9 -f gazebo 2>/dev/null

# B8: log hygiene — purge ROS logs + delete old per-run /tmp logs (>7 days)
rosclean purge -y >/dev/null 2>&1
find /tmp -maxdepth 1 -name 'T*_zone*' -mtime +7 -delete 2>/dev/null

echo "[cleanup] done — stack down, ports checked"

# fan_boost: sim is down -> Windows power plan back to Balanced (quiet fans).
# FAN_KEEP=1 skips the restore (batch runners relaunch within seconds).
if [ "${FAN_KEEP:-0}" != "1" ]; then
    bash "$(dirname "$0")/fan_boost.sh" off 2>/dev/null || true
fi
