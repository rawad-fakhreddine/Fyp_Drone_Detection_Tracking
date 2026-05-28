#!/usr/bin/env python3
"""
random_spawn_target.py — v3.0: Exact GPS-surveyed spawn zones
=====================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v2.0 changes:
  - 8 zones now cover the FULL island (both halves, all areas)
  - Jitter reduced ±15m → ±8m (safer near water/tree edges)
  - Spawn z raised 0.3m → 0.5m (clearance for terrain height variation)
  - Zone A relocated from risky origin (0,0) to safer flat area

Baylands terrain:  X=[-343,343]  Y=[-269,269]  (full mesh bounds)
Island land area:  X≈[-220,150]  Y≈[-180,190]  (non-water area)
Model pose offset: z=-1.3m (terrain surface ≈ world z=0)

Coordinate orientation (confirmed from axis inspection):
  +X = south  (toward lower part of image)
  -X = north  (toward upper part of image / structures)
  +Y = east   (toward right of image)
  -Y = west   (toward left / circular gardens / stream)

Zone design rationale:
  A  Center-south open field — safer than origin (terrain z variation at 0,0)
  B  Upper-north structures area — buildings, shelters, parking lot
  C  Upper-center winding paths — confirmed good by testing
  D  Northwest stream + tree-lined paths — western upper area
  E  West circular gardens — spiral garden features, flat ground
  F  Southwest lower gardens — southern circular features
  G  South-center open paths — lower open area
  H  Lower park — confirmed good by testing
"""

import rospy, random, math, subprocess, os
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String


# ── Baylands spawn zones v2.0 ─────────────────────────────────────────────────
# Format: (center_x, center_y, description)
# All zones verified against terrain mesh bounds and screenshot analysis
SPAWN_ZONES = {
    '1': (  -5.5,    40,  "Zone 1 — center area"),
    '2': (-140,     -50,  "Zone 2 — northwest"),
    '3': (-140,     115,  "Zone 3 — north"),
    '4': (-350,      35,  "Zone 4 — far north"),
    '5': (-305,     -75,  "Zone 5 — far northwest"),
    '6': (-200,    -115,  "Zone 6 — west"),
    '7': (  -45,   -170,  "Zone 7 — southwest"),
    '8': (  100,   -285,  "Zone 8 — far south"),
    '9': (  260,   -165,  "Zone 9 — southeast"),
}

# Jitter reduced from ±15m to ±8m — prevents drift toward water edges
DEFAULT_JITTER = 8.0

# Spawn z raised from 0.3 to 0.5 — clearance for terrain height variation
# (model pose is z=-1.3, so terrain surface ≈ world z=0; 0.5m gives safe margin)
SPAWN_Z = 0.5


def yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))


def teleport_model(model_name, x, y, z, yaw):
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name = model_name
        state.pose = Pose(position=Point(x=x, y=y, z=z),
                          orientation=yaw_to_quaternion(yaw))
        state.reference_frame = "world"
        resp = set_state(state)
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr("[RandomSpawn] Service call failed: %s" % e)
        return False


def spawn_target_model(sdf_path, x, y, z, yaw):
    cmd = ["rosrun", "gazebo_ros", "spawn_model",
           "-sdf", "-file", sdf_path,
           "-model", "target_iris",
           "-x", "%.3f" % x, "-y", "%.3f" % y,
           "-z", "%.3f" % z, "-Y", "%.4f" % yaw]
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

    zone_name  = str(rospy.get_param('~zone', 'random'))
    dist       = rospy.get_param('~dist',   -1.0)
    seed       = rospy.get_param('~seed',   -1)
    jitter     = rospy.get_param('~jitter', DEFAULT_JITTER)

    if seed >= 0:
        random.seed(seed)

    # Pick zone
    if zone_name == 'random':
        zone_key = random.choice(list(SPAWN_ZONES.keys()))
    elif zone_name in SPAWN_ZONES:
        zone_key = zone_name.upper()
    else:
        rospy.logwarn("[RandomSpawn] Unknown zone '%s', picking random" % zone_name)
        zone_key = random.choice(list(SPAWN_ZONES.keys()))

    cx, cy, desc = SPAWN_ZONES[zone_key]

    # Chaser position: zone center + jitter
    chaser_x   = cx + random.uniform(-jitter, jitter)
    chaser_y   = cy + random.uniform(-jitter, jitter)
    chaser_z   = SPAWN_Z
    chaser_yaw = random.uniform(-math.pi, math.pi)

    # Target: 3-6m in front of chaser
    target_dist    = dist if dist >= 0 else random.uniform(3.0, 6.0)
    lateral_offset = random.uniform(-1.5, 1.5)
    target_x = chaser_x + target_dist * math.cos(chaser_yaw) + \
               lateral_offset * math.sin(chaser_yaw)
    target_y = chaser_y + target_dist * math.sin(chaser_yaw) - \
               lateral_offset * math.cos(chaser_yaw)
    target_z   = SPAWN_Z
    target_yaw = chaser_yaw

    # Find target SDF
    sdf_candidates = [
        os.path.expanduser(
            "~/PX4-Autopilot/Tools/sitl_gazebo/models/target_iris_sitl/iris.sdf"),
        os.path.expanduser(
            "~/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf"),
    ]
    sdf_path = next((p for p in sdf_candidates if os.path.exists(p)), None)
    if sdf_path is None:
        rospy.logerr("[RandomSpawn] No target SDF found!"); return

    rospy.loginfo("")
    rospy.loginfo("=" * 65)
    rospy.loginfo("[RandomSpawn] v3.0 | Zone %s: %s" % (zone_key, desc))
    rospy.loginfo("=" * 65)
    rospy.loginfo("  Chaser:  x=%.1f  y=%.1f  z=%.1f  yaw=%.0f deg"
                  % (chaser_x, chaser_y, chaser_z, math.degrees(chaser_yaw)))
    rospy.loginfo("  Target:  x=%.1f  y=%.1f  z=%.1f  yaw=%.0f deg"
                  % (target_x, target_y, target_z, math.degrees(target_yaw)))
    rospy.loginfo("  Distance: %.1f m  |  Lateral: %.1f m"
                  % (target_dist, lateral_offset))
    rospy.loginfo("=" * 65)

    rospy.loginfo("[RandomSpawn] Teleporting chaser to zone %s..." % zone_key)
    if not teleport_model("iris", chaser_x, chaser_y, chaser_z, chaser_yaw):
        rospy.logerr("[RandomSpawn] Failed to teleport chaser!"); return

    rospy.sleep(1.0)

    rospy.loginfo("[RandomSpawn] Spawning target_iris...")
    if not spawn_target_model(sdf_path, target_x, target_y, target_z, target_yaw):
        rospy.logerr("[RandomSpawn] Failed to spawn target!"); return

    info_pub = rospy.Publisher('/drone_tracking/spawn_info',
                               String, queue_size=1, latch=True)
    info = ("zone=%s chaser=(%.1f,%.1f,yaw=%.0f) target=(%.1f,%.1f) dist=%.1f"
            % (zone_key, chaser_x, chaser_y, math.degrees(chaser_yaw),
               target_x, target_y, target_dist))
    info_pub.publish(String(data=info))

    rospy.loginfo("")
    rospy.loginfo("[RandomSpawn] DONE. Next steps:")
    rospy.loginfo("  1. Start target PX4 (instance 1)")
    rospy.loginfo("  2. Start target MAVROS")
    rospy.loginfo("  3. Start takeoff_both + other nodes")
    rospy.loginfo("")
    rospy.sleep(2.0)


if __name__ == '__main__':
    try:       main()
    except rospy.ROSInterruptException: pass
