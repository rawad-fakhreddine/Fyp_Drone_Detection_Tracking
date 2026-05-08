#!/usr/bin/env python3
"""
takeoff_node.py — owns the entire arm + takeoff sequence.

**M7.2 update (v3)**: Adds climb-rate ramp during the first 1.5s of
climb. Without it, PX4's cascade controller integrator was empty and
couldn't track the 0.6 m/s step input — drone pitched/rolled for ~3s
while the integrator wound up, then climb began. With ramp, integrator
fills smoothly and climb begins immediately.

Carried over from v2:
  - xy_p_gain reduced 1.5 → 0.6
  - 2s post-arm calm before climbing
  - Origin re-captured AFTER calm
  - Climb rate cap raised to 0.6 m/s
  - Progress watchdog
"""

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool, String


WORLD_VEL_YAW_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW_RATE
)


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class TakeoffNode:

    def __init__(self):
        rospy.init_node('takeoff_node')

        # Parameters
        self.target_altitude = rospy.get_param('~target_altitude', 1.0)
        self.climb_rate_max  = rospy.get_param('~climb_rate_max',  0.6)
        self.climb_rate_min  = rospy.get_param('~climb_rate_min',  0.20)
        self.alt_p_gain      = rospy.get_param('~alt_p_gain',      0.7)
        self.xy_p_gain       = rospy.get_param('~xy_p_gain',       0.6)
        self.xy_v_max        = rospy.get_param('~xy_v_max',        0.2)
        self.ekf_settle_s    = rospy.get_param('~ekf_settle_s',    2.0)
        self.post_arm_calm_s = rospy.get_param('~post_arm_calm_s', 2.0)
        # NEW: climb-rate ramp duration. PX4 cascade controller integrator
        # is empty at climb start; jumping to 0.6 m/s instantly causes
        # pitch/roll oscillation. Ramping over 1.5s lets integrator fill.
        self.climb_ramp_s    = rospy.get_param('~climb_ramp_s',    1.5)
        self.bridge_s        = rospy.get_param('~bridge_s',        1.0)

        # State
        self.state_msg = None
        self.altitude  = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.pose_received = False
        self.locked_yaw = 0.0

        self.cmd_pub = rospy.Publisher(
            '/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.ready_pub = rospy.Publisher(
            '/drone_tracking/takeoff_ready', Bool, queue_size=1, latch=True)
        self.phase_pub = rospy.Publisher(
            '/drone_tracking/ibvs_phase', String, queue_size=1)

        rospy.Subscriber('/mavros/state', State, self.state_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self.pose_cb, queue_size=1)

        self.rate = rospy.Rate(20)

        rospy.loginfo("[Takeoff] v3 — alt=%.2fm, climb_ramp=%.1fs (NEW), "
                      "post_arm_calm=%.1fs"
                      % (self.target_altitude, self.climb_ramp_s,
                         self.post_arm_calm_s))
        self.run()

    def state_cb(self, msg):
        self.state_msg = msg

    def pose_cb(self, msg):
        self.altitude  = msg.pose.position.z
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_yaw = yaw_from_quat(msg.pose.orientation)
        self.pose_received = True

    def _build_msg(self, vx=0.0, vy=0.0, vz=0.0, yaw=None):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = WORLD_VEL_YAW_MASK
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw = float(yaw if yaw is not None else self.locked_yaw)
        return msg

    def stream_zero(self, n):
        for _ in range(n):
            if rospy.is_shutdown():
                return
            self.cmd_pub.publish(self._build_msg())
            self.phase_pub.publish(String(data='TAKEOFF'))
            self.rate.sleep()

    def wait_for_mavros_connection(self, timeout=15.0):
        rospy.loginfo("[Takeoff] Waiting for MAVROS connection...")
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.state_msg is not None and self.state_msg.connected:
                rospy.loginfo("[Takeoff] MAVROS connected (mode=%s, armed=%s)"
                              % (self.state_msg.mode, self.state_msg.armed))
                return True
            self.stream_zero(1)
        rospy.logerr("[Takeoff] MAVROS connection timeout after %.1fs" % timeout)
        return False

    def set_offboard(self, max_attempts=10):
        try:
            rospy.wait_for_service('/mavros/set_mode', timeout=5.0)
        except rospy.ROSException:
            rospy.logerr("[Takeoff] /mavros/set_mode unavailable")
            return False
        set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        for attempt in range(1, max_attempts + 1):
            try:
                set_mode(custom_mode='OFFBOARD')
            except rospy.ServiceException as e:
                rospy.logwarn("[Takeoff] set_mode call failed: %s" % e)
            self.stream_zero(10)
            if self.state_msg and self.state_msg.mode == 'OFFBOARD':
                rospy.loginfo("[Takeoff] OFFBOARD (attempt %d)" % attempt)
                return True
            rospy.logwarn("[Takeoff] mode='%s', retry %d/%d"
                          % (self.state_msg.mode if self.state_msg else '?',
                             attempt, max_attempts))
        rospy.logerr("[Takeoff] OFFBOARD failed after %d attempts" % max_attempts)
        return False

    def arm(self, max_attempts=10):
        try:
            rospy.wait_for_service('/mavros/cmd/arming', timeout=5.0)
        except rospy.ROSException:
            rospy.logerr("[Takeoff] /mavros/cmd/arming unavailable")
            return False
        arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        for attempt in range(1, max_attempts + 1):
            try:
                arm_srv(True)
            except rospy.ServiceException as e:
                rospy.logwarn("[Takeoff] arm call failed: %s" % e)
            self.stream_zero(10)
            if self.state_msg and self.state_msg.armed:
                rospy.loginfo("[Takeoff] Armed (attempt %d)" % attempt)
                return True
            rospy.logwarn("[Takeoff] not armed, retry %d/%d"
                          % (attempt, max_attempts))
        rospy.logerr("[Takeoff] Arm failed after %d attempts." % max_attempts)
        return False

    def post_arm_calm(self):
        rospy.loginfo("[Takeoff] Post-arm calm for %.1fs (let EKF settle)..."
                      % self.post_arm_calm_s)
        self.stream_zero(int(self.post_arm_calm_s * 20))
        rospy.loginfo("[Takeoff] Calm done. Pose=(%.2f, %.2f, %.2f) yaw=%.1f°"
                      % (self.current_x, self.current_y, self.altitude,
                         math.degrees(self.current_yaw)))

    def run(self):
        if not self.wait_for_mavros_connection():
            return

        rospy.loginfo("[Takeoff] Waiting for pose...")
        while not rospy.is_shutdown() and not self.pose_received:
            self.cmd_pub.publish(self._build_msg(yaw=0.0))
            self.phase_pub.publish(String(data='TAKEOFF'))
            self.rate.sleep()

        self.locked_yaw = self.current_yaw

        rospy.loginfo("[Takeoff] Streaming setpoints for OFFBOARD precondition "
                      "(locked_yaw=%.1f°)..." % math.degrees(self.locked_yaw))
        self.stream_zero(60)

        if not self.set_offboard():
            return

        if not self.arm():
            return

        self.post_arm_calm()

        origin_x = self.current_x
        origin_y = self.current_y
        self.locked_yaw = self.current_yaw
        rospy.loginfo("[Takeoff] Origin locked at (%.2f, %.2f), yaw=%.1f°. "
                      "Climbing to %.2fm with %.1fs ramp..."
                      % (origin_x, origin_y, math.degrees(self.locked_yaw),
                         self.target_altitude, self.climb_ramp_s))

        # NEW: climb-rate ramp. Records when climb begins for ramp computation.
        climb_start = rospy.Time.now()

        last_alt = self.altitude
        last_alt_time = rospy.Time.now()
        climb_warned = False
        while not rospy.is_shutdown() and self.altitude < self.target_altitude:
            if not (self.state_msg and self.state_msg.armed):
                rospy.logwarn("[Takeoff] Disarmed during climb — aborting")
                return

            # Compute desired vz from altitude error
            alt_err = self.target_altitude - self.altitude
            vz_target = float(np.clip(alt_err * self.alt_p_gain,
                                      self.climb_rate_min, self.climb_rate_max))

            # NEW: ramp factor. During first climb_ramp_s, scale vz_target
            # by linearly increasing factor (0 → 1). This lets PX4's
            # cascade controller fill its integrator gradually instead of
            # fighting a step input — eliminates the 3-4s startup stall.
            t_into_climb = (rospy.Time.now() - climb_start).to_sec()
            if t_into_climb < self.climb_ramp_s:
                ramp = t_into_climb / self.climb_ramp_s
                vz = vz_target * ramp
            else:
                vz = vz_target

            dx = origin_x - self.current_x
            dy = origin_y - self.current_y
            vx = float(np.clip(dx * self.xy_p_gain, -self.xy_v_max, self.xy_v_max))
            vy = float(np.clip(dy * self.xy_p_gain, -self.xy_v_max, self.xy_v_max))

            self.cmd_pub.publish(self._build_msg(vx=vx, vy=vy, vz=vz))
            self.phase_pub.publish(String(data='TAKEOFF'))

            # Progress watchdog
            now = rospy.Time.now()
            elapsed = (now - last_alt_time).to_sec()
            if elapsed > 1.0:
                dz_dt = (self.altitude - last_alt) / elapsed
                # Only warn AFTER ramp completes (during ramp slow climb is expected)
                if (t_into_climb > self.climb_ramp_s + 1.0
                    and abs(dz_dt) < 0.05 and not climb_warned
                    and self.altitude < 0.8):
                    rospy.logwarn("[Takeoff] Climb stalled at %.2fm "
                                  "(dz/dt=%.3f m/s)" % (self.altitude, dz_dt))
                    climb_warned = True
                last_alt = self.altitude
                last_alt_time = now

            self.rate.sleep()

        if rospy.is_shutdown():
            return

        drift_x = self.current_x - origin_x
        drift_y = self.current_y - origin_y
        rospy.loginfo("[Takeoff] Reached %.2fm. Drift from origin: (%.2f, %.2f). "
                      "Handing off to IBVS." % (self.altitude, drift_x, drift_y))
        self.ready_pub.publish(Bool(data=True))

        rospy.loginfo("[Takeoff] Bridging handoff (%.1fs)..." % self.bridge_s)
        self.stream_zero(int(self.bridge_s * 20))
        self.phase_pub.publish(String(data='TAKEOFF_DONE'))

        rospy.loginfo("[Takeoff] Done.")


if __name__ == '__main__':
    try:
        TakeoffNode()
    except rospy.ROSInterruptException:
        pass