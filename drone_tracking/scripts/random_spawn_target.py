#!/usr/bin/env python3
"""
random_spawn_target.py — Spawn both drones at random positions in baylands
===========================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

Picks a random zone in the baylands park, spawns the target drone there
with the target always 3-6m IN FRONT of the chaser's position.

Baylands terrain: X=[-343,343] Y=[-269,269] (from mesh analysis)
Safe spawn zones selected to avoid water edges and ensure flat ground
with diverse backgrounds (trees, paths, structures, open fields).

NOTE: The chaser PX4 always spawns at Gazebo origin (0,0) via the launch
file. This script uses Gazebo's set_model_state service to TELEPORT the
chaser to the chosen zone BEFORE arming. Then spawns the target ahead.

Both PX4 instances will report local_position=(0,0,0) at their spawn
points — this is expected and doesn't affect IBVS (image-space tracking).

Usage:
  # Default: random zone, random heading
  rosrun drone_tracking random_spawn_target.py

  # Specific zone (A-H):
  rosrun drone_tracking random_spawn_target.py _zone:=D

  # Custom target distance:
  rosrun drone_tracking random_spawn_target.py _dist:=5

  # Reproducible position:
  rosrun drone_tracking random_spawn_target.py _seed:=42

Run AFTER:  Terminal 1 (Gazebo + chaser PX4 launched)
Run BEFORE: Terminal 4b (target PX4), Terminal 5 (takeoff_both)
"""

import rospy
import random
import math
import subprocess
import os
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String


# ── Baylands safe spawn zones ─────────────────────────────────────────
# Each zone: (center_x, center_y, description)
# Selected from the baylands terrain mesh to ensure:
#   - Flat ground (no slopes/water)
#   - Interesting backgrounds for YOLO diversity
#   - Well within terrain bounds (conservative margins)
SPAWN_ZONES = {
    'A': (0,     0,    "Origin — open field"),
    'B': (50,    80,   "Tree clusters"),
    'C': (-80,   50,   "Winding paths"),
    'D': (100,  -30,   "Near structures (right)"),
    'E': (-50,  120,   "Tree-lined paths (upper left)"),
    'F': (150,   50,   "Dense trees (right)"),
    'G': (-100, -40,   "Left open area"),
    'H': (30,    -80,   "Lower park area"),
}


def yaw_to_quaternion(yaw):
    """Convert yaw angle (radians) to geometry_msgs/Quaternion."""
    return Quaternion(
        x=0.0, y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


def teleport_model(model_name, x, y, z, yaw):
    """Move an existing Gazebo model to a new pose."""
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name = model_name
        state.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=yaw_to_quaternion(yaw)
        )
        state.reference_frame = "world"
        resp = set_state(state)
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr("[RandomSpawn] Service call failed: %s" % e)
        return False


def spawn_target_model(sdf_path, x, y, z, yaw):
    """Spawn the target_iris model via rosrun gazebo_ros spawn_model."""
    cmd = [
        "rosrun", "gazebo_ros", "spawn_model",
        "-sdf", "-file", sdf_path,
        "-model", "target_iris",
        "-x", "%.3f" % x,
        "-y", "%.3f" % y,
        "-z", "%.3f" % z,
        "-Y", "%.4f" % yaw,
    ]
    rospy.loginfo("[RandomSpawn] Spawning target: %s" % " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        else:
            rospy.logerr("[RandomSpawn] Spawn failed: %s" % result.stderr)
            return False
    except subprocess.TimeoutExpired:
        rospy.logerr("[RandomSpawn] Spawn timed out!")
        return False


def main():
    rospy.init_node('random_spawn_target', anonymous=True)

    # ── Parameters ────────────────────────────────────────────────────
    zone_name = rospy.get_param('~zone', 'random')
    dist      = rospy.get_param('~dist', -1.0)     # -1 = random 3-6m
    seed      = rospy.get_param('~seed', -1)
    jitter    = rospy.get_param('~jitter', 15.0)    # ±meters around zone center

    if seed >= 0:
        random.seed(seed)

    # ── Pick zone ─────────────────────────────────────────────────────
    if zone_name == 'random':
        zone_key = random.choice(list(SPAWN_ZONES.keys()))
    elif zone_name.upper() in SPAWN_ZONES:
        zone_key = zone_name.upper()
    else:
        rospy.logwarn("[RandomSpawn] Unknown zone '%s', picking random" % zone_name)
        zone_key = random.choice(list(SPAWN_ZONES.keys()))

    cx, cy, desc = SPAWN_ZONES[zone_key]

    # Add jitter around zone center
    chaser_x = cx + random.uniform(-jitter, jitter)
    chaser_y = cy + random.uniform(-jitter, jitter)
    chaser_z = 0.3  # slightly above ground
    chaser_yaw = random.uniform(-math.pi, math.pi)

    # ── Target position: always in front of chaser ────────────────────
    if dist < 0:
        target_dist = random.uniform(3.0, 6.0)
    else:
        target_dist = dist

    # "In front" = along chaser's heading direction
    # Add small lateral offset for variety
    lateral_offset = random.uniform(-1.5, 1.5)
    target_x = chaser_x + target_dist * math.cos(chaser_yaw) + \
               lateral_offset * math.sin(chaser_yaw)
    target_y = chaser_y + target_dist * math.sin(chaser_yaw) - \
               lateral_offset * math.cos(chaser_yaw)
    target_z = 0.5
    # Target faces same direction as chaser (will fly away from chaser)
    target_yaw = chaser_yaw

    # ── Find target SDF ──────────────────────────────────────────────
    sdf_candidates = [
        os.path.expanduser(
            "~/PX4-Autopilot/Tools/sitl_gazebo/models/target_iris_sitl/iris.sdf"),
        os.path.expanduser(
            "~/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf"),
    ]
    sdf_path = None
    for p in sdf_candidates:
        if os.path.exists(p):
            sdf_path = p
            break
    if sdf_path is None:
        rospy.logerr("[RandomSpawn] No target SDF found!")
        return

    # ── Print plan ───────────────────────────────────────────────────
    rospy.loginfo("")
    rospy.loginfo("=" * 65)
    rospy.loginfo("[RandomSpawn] Zone %s: %s" % (zone_key, desc))
    rospy.loginfo("=" * 65)
    rospy.loginfo("  Chaser:  x=%.1f  y=%.1f  z=%.1f  yaw=%.0f deg"
                  % (chaser_x, chaser_y, chaser_z, math.degrees(chaser_yaw)))
    rospy.loginfo("  Target:  x=%.1f  y=%.1f  z=%.1f  yaw=%.0f deg"
                  % (target_x, target_y, target_z, math.degrees(target_yaw)))
    rospy.loginfo("  Distance: %.1f m  |  Lateral offset: %.1f m"
                  % (target_dist, lateral_offset))
    rospy.loginfo("  SDF: %s" % os.path.basename(sdf_path))
    rospy.loginfo("=" * 65)

    # ── Step 1: Teleport chaser (already spawned at origin by launch) ─
    rospy.loginfo("[RandomSpawn] Teleporting chaser 'iris' to zone %s..."
                  % zone_key)
    if teleport_model("iris", chaser_x, chaser_y, chaser_z, chaser_yaw):
        rospy.loginfo("[RandomSpawn] Chaser teleported successfully")
    else:
        rospy.logerr("[RandomSpawn] Failed to teleport chaser!")
        return

    rospy.sleep(1.0)  # let Gazebo settle

    # ── Step 2: Spawn target ─────────────────────────────────────────
    rospy.loginfo("[RandomSpawn] Spawning target_iris...")
    if spawn_target_model(sdf_path, target_x, target_y, target_z, target_yaw):
        rospy.loginfo("[RandomSpawn] Target spawned successfully!")
    else:
        rospy.logerr("[RandomSpawn] Failed to spawn target!")
        return

    # ── Publish spawn info (latched) ─────────────────────────────────
    info_pub = rospy.Publisher('/drone_tracking/spawn_info',
                               String, queue_size=1, latch=True)
    info = ("zone=%s chaser=(%.1f,%.1f,yaw=%.0f) target=(%.1f,%.1f) "
            "dist=%.1f" % (zone_key, chaser_x, chaser_y,
                           math.degrees(chaser_yaw),
                           target_x, target_y, target_dist))
    info_pub.publish(String(data=info))

    rospy.loginfo("")
    rospy.loginfo("[RandomSpawn] DONE. Next steps:")
    rospy.loginfo("  1. Start target PX4:  cd ~/PX4-Autopilot/build/px4_sitl_default && "
                  "PX4_SIM_MODEL=iris PX4_SIMULATOR=gazebo MAV_SYS_ID=2 "
                  "./bin/px4 -i 1 -s etc/init.d-posix/rcS")
    rospy.loginfo("  2. Start target MAVROS:  roslaunch drone_tracking "
                  "target_mavros.launch")
    rospy.loginfo("  3. Start takeoff_both + other nodes")
    rospy.loginfo("")

    rospy.sleep(2.0)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass