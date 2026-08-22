#!/usr/bin/env python3
"""
Bridge Gazebo ground-truth model states → MAVROS vision_pose/pose + vision_speed topics.
Enables EKF2 External Vision fusion (AID_MASK=265: GPS+EV_pos+EV_vel) in nolockstep mode.

Root cause addressed: GPS velocity at 2×/4× speed = half actual → EKF v_xy_valid=false →
"horizontal velocity estimate not stable" → ARM blocked. EV velocity (GT, exact) dominates
GPS velocity in EKF (EVV_NOISE=0.1 m/s  ≪  GPS_V_NOISE=1.2 m/s) → v_xy_valid=true → ARM OK.

Frame alignment: publishes position RELATIVE to each drone's first received Gazebo pose
(= spawn point ≈ GPS home). Absolute world coords → 30+ m innovations → silently rejected.

Velocity: finite difference of consecutive EV positions, published to vision_speed/speed_vector.
dt gated at ≥1 ms to avoid division-by-zero at startup.

Only launched by launch_stack.sh when NO_LOCKSTEP=1.
"""
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TwistStamped

_chaser_pos_pub = None
_target_pos_pub = None
_chaser_vel_pub = None
_target_vel_pub = None
_chaser_model = 'iris'
_target_model = 'target_iris'

_chaser_home = None
_target_home = None
_chaser_prev = None   # (rel_pos_tuple, time)
_target_prev = None


def _pub_drone(pos_pub, vel_pub, raw_pose, home, prev_state, now):
    rel_x = raw_pose.position.x - home[0]
    rel_y = raw_pose.position.y - home[1]
    rel_z = raw_pose.position.z - home[2]

    ps = PoseStamped()
    ps.header.stamp = now
    ps.header.frame_id = 'map'
    ps.pose.position.x = rel_x
    ps.pose.position.y = rel_y
    ps.pose.position.z = rel_z
    ps.pose.orientation = raw_pose.orientation
    pos_pub.publish(ps)

    new_prev = ((rel_x, rel_y, rel_z), now)

    if prev_state is not None:
        (px, py, pz), pt = prev_state
        dt = (now - pt).to_sec()
        if dt >= 0.001:
            vs = TwistStamped()
            vs.header.stamp = now
            vs.header.frame_id = 'map'
            vs.twist.linear.x = (rel_x - px) / dt
            vs.twist.linear.y = (rel_y - py) / dt
            vs.twist.linear.z = (rel_z - pz) / dt
            vel_pub.publish(vs)

    return new_prev


def _model_states_cb(msg):
    global _chaser_home, _target_home, _chaser_prev, _target_prev
    now = rospy.Time.now()  # wall-clock (use_sim_time=false for MAVROS nodes)
    names = msg.name

    try:
        ci = names.index(_chaser_model)
        raw = msg.pose[ci]
        if _chaser_home is None:
            _chaser_home = (raw.position.x, raw.position.y, raw.position.z)
            rospy.loginfo("[GzGtToVision] chaser home latched: x=%.3f y=%.3f z=%.3f",
                          _chaser_home[0], _chaser_home[1], _chaser_home[2])
        _chaser_prev = _pub_drone(_chaser_pos_pub, _chaser_vel_pub,
                                  raw, _chaser_home, _chaser_prev, now)
    except ValueError:
        pass

    try:
        ti = names.index(_target_model)
        raw = msg.pose[ti]
        if _target_home is None:
            _target_home = (raw.position.x, raw.position.y, raw.position.z)
            rospy.loginfo("[GzGtToVision] target home latched: x=%.3f y=%.3f z=%.3f",
                          _target_home[0], _target_home[1], _target_home[2])
        _target_prev = _pub_drone(_target_pos_pub, _target_vel_pub,
                                  raw, _target_home, _target_prev, now)
    except ValueError:
        pass


def main():
    global _chaser_pos_pub, _target_pos_pub, _chaser_vel_pub, _target_vel_pub
    global _chaser_model, _target_model

    rospy.init_node('gz_gt_to_vision', anonymous=False)

    _chaser_model = rospy.get_param('~chaser_model', 'iris')
    _target_model = rospy.get_param('~target_model', 'target_iris')

    _chaser_pos_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
    _target_pos_pub = rospy.Publisher('/target/mavros/vision_pose/pose', PoseStamped, queue_size=10)
    _chaser_vel_pub = rospy.Publisher('/mavros/vision_speed/speed_vector', TwistStamped, queue_size=10)
    _target_vel_pub = rospy.Publisher('/target/mavros/vision_speed/speed_vector', TwistStamped, queue_size=10)

    rospy.Subscriber('/gazebo/model_states', ModelStates, _model_states_cb, queue_size=1)

    rospy.loginfo("[GzGtToVision] chaser='%s' → vision_pose + vision_speed  "
                  "target='%s' → /target/mavros vision_pose + vision_speed",
                  _chaser_model, _target_model)
    rospy.spin()


if __name__ == '__main__':
    main()
