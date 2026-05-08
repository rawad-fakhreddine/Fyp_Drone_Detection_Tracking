#!/usr/bin/env python3
"""
takeoff_both.py — Unified takeoff for chaser + target drones
=============================================================

Arms both drones, enters OFFBOARD mode on both, climbs both to
TAKEOFF_ALT (1.5m), then signals ready so IBVS and target_mover
can take over.

Inherits takeoff_node v3 learnings:
  - Post-arm calm period (2s zero-velocity setpoints)
  - Climb-rate ramp (0 → 0.6 m/s over 1.5s) to avoid PX4 integrator wind-up
  - Origin re-captured after calm period

Namespaces:
  Chaser:  /mavros/...                 (default, instance 0)
  Target:  /target/mavros/...          (instance 1)

Signals:
  /drone_tracking/takeoff_ready         — chaser at altitude
  /drone_tracking/target_takeoff_ready  — target at altitude

Usage:
  rosrun drone_tracking takeoff_both.py
"""

import rospy
import math
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


# ── Type mask for velocity commands (ignore position + acceleration + yaw) ──
LOCAL_VEL_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW
)

TAKEOFF_ALT    = 1.5     # meters — both drones climb to this
ALT_TOLERANCE  = 0.25    # meters — "at altitude" threshold
CALM_DURATION  = 2.0     # seconds — zero-vel setpoints after arming
RAMP_DURATION  = 1.5     # seconds — climb rate ramp 0 → MAX_CLIMB
MAX_CLIMB      = 0.6     # m/s — peak climb rate
XY_P_GAIN      = 0.6     # position hold gain during climb (v3 tuning)
HOVER_HOLD_S   = 3.0     # seconds to keep hovering after signalling ready


class DroneHandle:
    """Manages one drone's MAVROS connection, state, and commands."""

    def __init__(self, ns, label):
        """
        ns: namespace prefix, e.g. '' for chaser, '/target' for target.
        label: human-readable name for logging.
        """
        self.ns = ns
        self.label = label
        pfx = ns + '/mavros' if ns else '/mavros'

        # State
        self.armed = False
        self.mode = ""
        self.connected = False
        self.altitude = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_locked = False

        # Publisher
        self.cmd_pub = rospy.Publisher(
            pfx + '/setpoint_raw/local', PositionTarget, queue_size=1)

        # Subscribers
        rospy.Subscriber(pfx + '/state', State, self._state_cb, queue_size=1)
        rospy.Subscriber(pfx + '/local_position/pose', PoseStamped,
                         self._pose_cb, queue_size=1)

        # Service proxies (wait at most 15s each)
        rospy.loginfo("[TakeoffBoth] %s: waiting for services on %s..." %
                      (label, pfx))
        try:
            rospy.wait_for_service(pfx + '/cmd/arming', timeout=15)
            rospy.wait_for_service(pfx + '/set_mode', timeout=15)
        except rospy.ROSException:
            rospy.logerr("[TakeoffBoth] %s: MAVROS services not available!" % label)
            raise

        self.arm_srv = rospy.ServiceProxy(pfx + '/cmd/arming', CommandBool)
        self.mode_srv = rospy.ServiceProxy(pfx + '/set_mode', SetMode)

        rospy.loginfo("[TakeoffBoth] %s: connected to %s" % (label, pfx))

    def _state_cb(self, msg):
        self.armed = msg.armed
        self.mode = msg.mode
        self.connected = msg.connected

    def _pose_cb(self, msg):
        self.altitude = msg.pose.position.z
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        if not self.origin_locked:
            self.origin_x = self.pos_x
            self.origin_y = self.pos_y

    def lock_origin(self):
        self.origin_x = self.pos_x
        self.origin_y = self.pos_y
        self.origin_locked = True
        rospy.loginfo("[TakeoffBoth] %s: origin locked at (%.2f, %.2f)" %
                      (self.label, self.origin_x, self.origin_y))

    def send_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0,
                 frame=PositionTarget.FRAME_LOCAL_NED):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = frame
        msg.type_mask = LOCAL_VEL_MASK
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate = float(yaw_rate)
        self.cmd_pub.publish(msg)

    def set_offboard(self):
        try:
            resp = self.mode_srv(custom_mode='OFFBOARD')
            rospy.loginfo("[TakeoffBoth] %s: OFFBOARD → %s" %
                          (self.label, "OK" if resp.mode_sent else "FAIL"))
            return resp.mode_sent
        except rospy.ServiceException as e:
            rospy.logwarn("[TakeoffBoth] %s: set_mode failed: %s" %
                          (self.label, e))
            return False

    def arm(self):
        try:
            resp = self.arm_srv(True)
            rospy.loginfo("[TakeoffBoth] %s: ARM → %s" %
                          (self.label, "OK" if resp.success else "FAIL"))
            return resp.success
        except rospy.ServiceException as e:
            rospy.logwarn("[TakeoffBoth] %s: arming failed: %s" %
                          (self.label, e))
            return False

    def at_altitude(self):
        return self.altitude >= (TAKEOFF_ALT - ALT_TOLERANCE)

    def climb_vel(self, ramp_factor):
        """Compute climb velocity with XY position hold."""
        vz = MAX_CLIMB * ramp_factor
        # P-control to hold origin during climb
        vx = -XY_P_GAIN * (self.pos_x - self.origin_x)
        vy = -XY_P_GAIN * (self.pos_y - self.origin_y)
        return vx, vy, vz


class TakeoffBoth:

    def __init__(self):
        rospy.init_node('takeoff_both')

        # Create handles for both drones
        self.chaser = DroneHandle('', 'CHASER')
        self.target = DroneHandle('/target', 'TARGET')

        # Ready signal publishers
        self.chaser_ready_pub = rospy.Publisher(
            '/drone_tracking/takeoff_ready', Bool, queue_size=1, latch=True)
        self.target_ready_pub = rospy.Publisher(
            '/drone_tracking/target_takeoff_ready', Bool, queue_size=1, latch=True)

        self.rate = rospy.Rate(20)
        self.run()

    def _wait_connections(self):
        """Wait for both MAVROS instances to connect."""
        rospy.loginfo("[TakeoffBoth] Waiting for both MAVROS connections...")
        while not rospy.is_shutdown():
            if self.chaser.connected and self.target.connected:
                rospy.loginfo("[TakeoffBoth] Both MAVROS connected")
                return True
            rospy.loginfo_throttle(2,
                "[TakeoffBoth] Waiting... chaser=%s target=%s" %
                (self.chaser.connected, self.target.connected))
            self.rate.sleep()
        return False

    def _pre_arm_stream(self, duration):
        """Send zero-velocity setpoints to both drones (required before OFFBOARD)."""
        rospy.loginfo("[TakeoffBoth] Pre-arm stream for %.1fs..." % duration)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()
            if elapsed >= duration:
                break
            self.chaser.send_vel()
            self.target.send_vel()
            self.rate.sleep()

    def _offboard_and_arm(self):
        """Set OFFBOARD mode and arm both drones."""
        # OFFBOARD first (PX4 requires setpoint stream before OFFBOARD)
        rospy.loginfo("[TakeoffBoth] Setting OFFBOARD on both...")
        for attempt in range(5):
            ok_c = self.chaser.mode == 'OFFBOARD' or self.chaser.set_offboard()
            ok_t = self.target.mode == 'OFFBOARD' or self.target.set_offboard()
            # Keep streaming while retrying
            self.chaser.send_vel()
            self.target.send_vel()
            if ok_c and ok_t:
                break
            rospy.sleep(0.5)

        # Calm period — zero velocity for 2s before arming
        rospy.loginfo("[TakeoffBoth] Post-OFFBOARD calm period (%.1fs)..." %
                      CALM_DURATION)
        self._pre_arm_stream(CALM_DURATION)

        # Arm both
        rospy.loginfo("[TakeoffBoth] Arming both drones...")
        for attempt in range(10):
            if not self.chaser.armed:
                self.chaser.arm()
            if not self.target.armed:
                self.target.arm()
            # Keep streaming setpoints + wait for state callback to update
            for _ in range(10):
                self.chaser.send_vel()
                self.target.send_vel()
                self.rate.sleep()
            if self.chaser.armed and self.target.armed:
                rospy.loginfo("[TakeoffBoth] Both armed!")
                break
            rospy.loginfo("[TakeoffBoth] Arm attempt %d: chaser=%s target=%s" %
                          (attempt+1, self.chaser.armed, self.target.armed))

        if not self.chaser.armed or not self.target.armed:
            rospy.logerr("[TakeoffBoth] Failed to arm! chaser=%s target=%s" %
                         (self.chaser.armed, self.target.armed))
            return False

        # Lock origins after arming calm
        self.chaser.lock_origin()
        self.target.lock_origin()

        rospy.loginfo("[TakeoffBoth] Both drones armed and OFFBOARD")
        return True

    def _climb(self):
        """Ramp-climb both drones to TAKEOFF_ALT."""
        rospy.loginfo("[TakeoffBoth] Climbing to %.1fm (ramp %.1fs)..." %
                      (TAKEOFF_ALT, RAMP_DURATION))
        t0 = rospy.Time.now()
        chaser_done = False
        target_done = False

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()

            # Ramp factor: 0→1 over RAMP_DURATION
            ramp = min(1.0, elapsed / RAMP_DURATION)

            # Chaser
            if not chaser_done:
                vx, vy, vz = self.chaser.climb_vel(ramp)
                self.chaser.send_vel(vx, vy, vz)
                if self.chaser.at_altitude():
                    chaser_done = True
                    rospy.loginfo("[TakeoffBoth] CHASER at %.2fm ✓" %
                                  self.chaser.altitude)
            else:
                # Hold altitude
                vx, vy, _ = self.chaser.climb_vel(0.0)
                self.chaser.send_vel(vx, vy, 0.0)

            # Target
            if not target_done:
                vx, vy, vz = self.target.climb_vel(ramp)
                self.target.send_vel(vx, vy, vz)
                if self.target.at_altitude():
                    target_done = True
                    rospy.loginfo("[TakeoffBoth] TARGET at %.2fm ✓" %
                                  self.target.altitude)
            else:
                vx, vy, _ = self.target.climb_vel(0.0)
                self.target.send_vel(vx, vy, 0.0)

            if chaser_done and target_done:
                rospy.loginfo("[TakeoffBoth] Both at altitude!")
                return True

            # Timeout after 30s
            if elapsed > 30.0:
                rospy.logerr("[TakeoffBoth] Climb timeout! chaser=%.1fm target=%.1fm"
                             % (self.chaser.altitude, self.target.altitude))
                return False

            self.rate.sleep()

        return False

    def _signal_ready(self):
        """Publish ready signals and continue hovering."""
        rospy.loginfo("[TakeoffBoth] *** BOTH DRONES READY — signalling ***")
        self.chaser_ready_pub.publish(Bool(data=True))
        self.target_ready_pub.publish(Bool(data=True))

        # Keep hovering for a few seconds so IBVS + target_mover can take over
        rospy.loginfo("[TakeoffBoth] Holding hover for %.1fs then yielding..." %
                      HOVER_HOLD_S)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - t0).to_sec()
            if elapsed >= HOVER_HOLD_S:
                break

            # Simple hover: zero velocity with XY hold
            vx_c, vy_c, _ = self.chaser.climb_vel(0.0)
            self.chaser.send_vel(vx_c, vy_c, 0.0)

            vx_t, vy_t, _ = self.target.climb_vel(0.0)
            self.target.send_vel(vx_t, vy_t, 0.0)

            self.rate.sleep()

        rospy.loginfo("[TakeoffBoth] Takeoff node complete. "
                      "IBVS owns chaser, target_mover owns target. "
                      "Ctrl+C to exit.")

        # Keep alive but stop publishing — other nodes have taken over
        rospy.spin()

    def run(self):
        if not self._wait_connections():
            return
        self._pre_arm_stream(3.0)
        if not self._offboard_and_arm():
            return
        if not self._climb():
            return
        self._signal_ready()


if __name__ == '__main__':
    try:
        TakeoffBoth()
    except rospy.ROSInterruptException:
        pass