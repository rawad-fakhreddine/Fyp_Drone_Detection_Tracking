#!/usr/bin/env python3
"""spectator_tracker.py (M-C) — move the spectator_cam each tick to follow the
drones: keep the midpoint of chaser (iris) + target (target_iris) centred, from
a fixed world-frame offset (behind + above). Gives big, always-in-frame drones
on every trajectory. Publishes /gazebo/set_model_state (model is kinematic).
Params: ~off_x/~off_y/~off_z (camera offset from midpoint, m), ~rate (Hz),
        ~smooth (EMA on the midpoint, 0..1)."""
import rospy
import math
from gazebo_msgs.msg import ModelStates, ModelState
from tf.transformations import quaternion_from_euler


class Tracker:
    def __init__(self):
        rospy.init_node('spectator_tracker', anonymous=True)
        self.ox = float(rospy.get_param('~off_x', -8.0))
        self.oy = float(rospy.get_param('~off_y', -8.0))
        self.oz = float(rospy.get_param('~off_z', 9.0))
        self.smooth = float(rospy.get_param('~smooth', 0.15))  # EMA on midpoint
        self.mid = None
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param('~rate', 30.0))), self.tick)
        rospy.loginfo("[spec_track] following midpoint, offset (%.1f,%.1f,%.1f)"
                      % (self.ox, self.oy, self.oz))
        rospy.spin()

    def cb(self, m):
        try:
            ic = m.name.index('iris')
            it = m.name.index('target_iris')
        except ValueError:
            return
        c, t = m.pose[ic].position, m.pose[it].position
        raw = ((c.x + t.x) / 2.0, (c.y + t.y) / 2.0, (c.z + t.z) / 2.0)
        if self.mid is None:
            self.mid = raw
        else:
            a = self.smooth
            self.mid = tuple(self.mid[i] * (1 - a) + raw[i] * a for i in range(3))

    def tick(self, _evt):
        if self.mid is None:
            return
        mx, my, mz = self.mid
        px, py, pz = mx + self.ox, my + self.oy, mz + self.oz
        dx, dy, dz = mx - px, my - py, mz - pz          # look direction
        yaw = math.atan2(dy, dx)
        pitch = -math.atan2(dz, math.hypot(dx, dy))     # +pitch = look down
        q = quaternion_from_euler(0.0, pitch, yaw)
        ms = ModelState()
        ms.model_name = 'spectator_cam'
        ms.pose.position.x = px
        ms.pose.position.y = py
        ms.pose.position.z = pz
        ms.pose.orientation.x = q[0]
        ms.pose.orientation.y = q[1]
        ms.pose.orientation.z = q[2]
        ms.pose.orientation.w = q[3]
        ms.reference_frame = 'world'
        self.pub.publish(ms)


if __name__ == '__main__':
    try:
        Tracker()
    except rospy.ROSInterruptException:
        pass
