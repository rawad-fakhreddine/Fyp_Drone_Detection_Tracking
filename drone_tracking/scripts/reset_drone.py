#!/usr/bin/env python3
"""
reset_drone.py — teleport chaser back to spawn, disarm, zero state.

Use between flights without restarting Gazebo / PX4.

What this does:
  1. Disarm via /mavros/cmd/arming (skipped if already disarmed)
  2. Teleport the iris model to spawn pose via /gazebo/set_model_state
  3. Send 30 zero TwistStamped commands to drain the setpoint queue
  4. Publish 'RESET' on /drone_tracking/ibvs_phase so any state-aware
     nodes can clear their internal state on next tick

Caveat: PX4 SITL keeps its EKF running. The first arm after reset may
take 2-3 s to converge — this is normal.

Usage: rosrun drone_tracking reset_drone.py
       rosrun drone_tracking reset_drone.py _x:=0 _y:=0 _z:=0.2
"""

import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import TwistStamped, Pose, Twist
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from std_msgs.msg import String


CHASER_MODEL_NAME = "iris"   # default PX4 SITL model name


def wait_for_service(name, timeout=5.0):
    try:
        rospy.wait_for_service(name, timeout=timeout)
        return True
    except rospy.ROSException:
        rospy.logwarn("[Reset] Service %s unavailable" % name)
        return False


def main():
    rospy.init_node('reset_drone')

    # Spawn pose (override via ~x ~y ~z ~yaw params)
    spawn_x   = rospy.get_param('~x',   0.0)
    spawn_y   = rospy.get_param('~y',   0.0)
    spawn_z   = rospy.get_param('~z',   0.2)
    spawn_yaw = rospy.get_param('~yaw', 0.0)
    model     = rospy.get_param('~model', CHASER_MODEL_NAME)

    rospy.loginfo("[Reset] Resetting chaser '%s' to (%.2f, %.2f, %.2f)"
                  % (model, spawn_x, spawn_y, spawn_z))

    # ── 1. Disarm ──────────────────────────────────────────────────────────
    if wait_for_service('/mavros/cmd/arming'):
        try:
            arm = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            arm(False)
            rospy.loginfo("[Reset] Disarm requested")
            rospy.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logwarn("[Reset] Disarm failed: %s" % e)

    # Switch out of OFFBOARD so PX4 doesn't fight the teleport
    if wait_for_service('/mavros/set_mode'):
        try:
            set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            set_mode(custom_mode='AUTO.LOITER')
            rospy.sleep(0.3)
        except rospy.ServiceException as e:
            rospy.logwarn("[Reset] Mode switch failed: %s" % e)

    # ── 2. Teleport via Gazebo ─────────────────────────────────────────────
    if not wait_for_service('/gazebo/set_model_state', timeout=10.0):
        rospy.logerr("[Reset] Cannot reach Gazebo — aborting teleport")
        return

    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    state = ModelState()
    state.model_name = model
    state.pose = Pose()
    state.pose.position.x = spawn_x
    state.pose.position.y = spawn_y
    state.pose.position.z = spawn_z
    # Quaternion from yaw only
    import math
    state.pose.orientation.z = math.sin(spawn_yaw / 2.0)
    state.pose.orientation.w = math.cos(spawn_yaw / 2.0)
    state.twist = Twist()   # zero velocities
    state.reference_frame = 'world'

    try:
        resp = set_state(state)
        if resp.success:
            rospy.loginfo("[Reset] Chaser teleported")
        else:
            rospy.logwarn("[Reset] Teleport reported failure: %s"
                          % resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("[Reset] Teleport service call failed: %s" % e)
        return

    # ── 3. Drain velocity setpoint queue ───────────────────────────────────
    cmd_pub = rospy.Publisher(
        '/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    rospy.sleep(0.2)

    rate = rospy.Rate(20)
    for _ in range(30):
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd_pub.publish(cmd)
        rate.sleep()

    # ── 4. EKF settling pause ──────────────────────────────────────────────
    # PX4's EKF needs time to re-converge on the post-teleport pose.
    # Skipping this leaves the drone with a bad position estimate and
    # makes the next takeoff sluggish or fail outright.
    rospy.loginfo("[Reset] Waiting 6 s for PX4 EKF to re-converge...")
    rospy.sleep(6.0)

    # ── 5. Notify IBVS / loggers via phase topic ───────────────────────────
    phase_pub = rospy.Publisher(
        '/drone_tracking/ibvs_phase', String, queue_size=1, latch=True)
    rospy.sleep(0.2)
    phase_pub.publish(String(data='RESET'))

    rospy.loginfo("[Reset] Done. Ready to re-arm via Terminal 9.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass