#!/usr/bin/env python3
"""
Bridge Gazebo ground-truth model states → MAVROS vision_pose/pose topics.
Enables EKF2 External Vision fusion alongside GPS to stabilize position estimate
in nolockstep mode (GPS timing mismatch at 2×+ speed causes EKF oscillation).
Uses wall-clock timestamps (rospy.Time.now()) to match PX4 nolockstep HRT.
Only launched by launch_stack.sh when NO_LOCKSTEP=1.

Frame alignment: publishes position RELATIVE to each drone's first received
Gazebo pose (= spawn point ≈ GPS home). Absolute world coords produce 30+ m
innovations which EKF silently rejects (cs_ev_pos stays 0).
"""
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped

_chaser_pub = None
_target_pub = None
_chaser_model = 'iris'
_target_model = 'target_iris'

_chaser_home = None
_target_home = None


def _model_states_cb(msg):
    global _chaser_home, _target_home
    now = rospy.Time.now()  # wall-clock (use_sim_time=false for MAVROS nodes)
    names = msg.name

    try:
        ci = names.index(_chaser_model)
        raw = msg.pose[ci]
        if _chaser_home is None:
            _chaser_home = (raw.position.x, raw.position.y, raw.position.z)
            rospy.loginfo("[GzGtToVision] chaser home latched: x=%.3f y=%.3f z=%.3f",
                          _chaser_home[0], _chaser_home[1], _chaser_home[2])
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = 'map'
        ps.pose.position.x = raw.position.x - _chaser_home[0]
        ps.pose.position.y = raw.position.y - _chaser_home[1]
        ps.pose.position.z = raw.position.z - _chaser_home[2]
        ps.pose.orientation = raw.orientation
        _chaser_pub.publish(ps)
    except ValueError:
        pass

    try:
        ti = names.index(_target_model)
        raw = msg.pose[ti]
        if _target_home is None:
            _target_home = (raw.position.x, raw.position.y, raw.position.z)
            rospy.loginfo("[GzGtToVision] target home latched: x=%.3f y=%.3f z=%.3f",
                          _target_home[0], _target_home[1], _target_home[2])
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = 'map'
        ps.pose.position.x = raw.position.x - _target_home[0]
        ps.pose.position.y = raw.position.y - _target_home[1]
        ps.pose.position.z = raw.position.z - _target_home[2]
        ps.pose.orientation = raw.orientation
        _target_pub.publish(ps)
    except ValueError:
        pass


def main():
    global _chaser_pub, _target_pub, _chaser_model, _target_model

    rospy.init_node('gz_gt_to_vision', anonymous=False)

    _chaser_model = rospy.get_param('~chaser_model', 'iris')
    _target_model = rospy.get_param('~target_model', 'target_iris')

    _chaser_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
    _target_pub = rospy.Publisher('/target/mavros/vision_pose/pose', PoseStamped, queue_size=10)

    rospy.Subscriber('/gazebo/model_states', ModelStates, _model_states_cb, queue_size=1)

    rospy.loginfo("[GzGtToVision] chaser='%s' → /mavros/vision_pose/pose  "
                  "target='%s' → /target/mavros/vision_pose/pose",
                  _chaser_model, _target_model)
    rospy.spin()


if __name__ == '__main__':
    main()
