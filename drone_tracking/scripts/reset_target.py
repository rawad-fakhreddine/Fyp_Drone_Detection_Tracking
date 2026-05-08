#!/usr/bin/env python3
"""
reset_target.py — teleport target_iris back to its spawn pose.

This script does NOT stop target_mover.py — that's a separate process you
control from T4. What it does is reset the target's pose so a running
target_mover continues its trajectory from a fresh starting point.

If you want to fully reset the trajectory (random_walk seed, figure8
phase, etc), Ctrl-C target_mover.py in T4 and re-run it after this.

Usage:
    rosrun drone_tracking reset_target.py
    rosrun drone_tracking reset_target.py _x:=3 _y:=0 _z:=1.5
    rosrun drone_tracking reset_target.py _model:=target_iris _x:=5
"""

import math
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist


def main():
    rospy.init_node('reset_target')

    model     = rospy.get_param('~model', 'target_iris')
    spawn_x   = rospy.get_param('~x',   3.0)
    spawn_y   = rospy.get_param('~y',   0.0)
    spawn_z   = rospy.get_param('~z',   1.5)
    spawn_yaw = rospy.get_param('~yaw', 0.0)

    rospy.loginfo("[ResetTarget] Resetting '%s' to (%.2f, %.2f, %.2f)"
                  % (model, spawn_x, spawn_y, spawn_z))

    try:
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
    except rospy.ROSException:
        rospy.logerr("[ResetTarget] Gazebo service unavailable")
        return

    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    state = ModelState()
    state.model_name = model
    state.pose = Pose()
    state.pose.position.x = spawn_x
    state.pose.position.y = spawn_y
    state.pose.position.z = spawn_z
    state.pose.orientation.z = math.sin(spawn_yaw / 2.0)
    state.pose.orientation.w = math.cos(spawn_yaw / 2.0)
    state.twist = Twist()
    state.reference_frame = 'world'

    try:
        resp = set_state(state)
        if resp.success:
            rospy.loginfo("[ResetTarget] Target teleported")
        else:
            rospy.logwarn("[ResetTarget] Teleport reported failure: %s"
                          % resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("[ResetTarget] Service call failed: %s" % e)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass