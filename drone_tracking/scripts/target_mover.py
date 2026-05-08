#!/usr/bin/env python3
"""
target_mover.py — v7.1  (MAVROS velocity commands, real PX4 flight)
====================================================================

Same smooth curved motion model as v7, but the target drone is now a
real PX4 SITL instance (instance 1) controlled via MAVROS velocity
commands in /target/mavros/ namespace.

Key differences from v7 (teleport):
  - No more SetModelState — target flies with real aerodynamics
  - Reads current position from /target/mavros/local_position/pose
  - Sends velocity commands to /target/mavros/setpoint_raw/local
  - Uses FRAME_LOCAL_NED (world frame) since our velocity model is
    in world frame (speed * cos/sin of heading)
  - Yaw controlled via yaw_rate field (= omega, our heading rate)
  - STABILIZE rise done via vz velocity commands (no teleport)
  - Waits for /drone_tracking/target_takeoff_ready from takeoff_both

Motion model (unchanged from v7):
  - Heading rate (omega): resampled every [5,15]s, constant between → arcs
  - Speed: resampled every [5,15]s, EMA-smoothed, range [0.2, 0.8] m/s
  - Vertical speed (vz): resampled every [5,15]s, altitude-biased, EMA-smoothed
  - Soft repulsion beyond 60m from origin
  - Soft altitude attraction to [7,11]m band

State machine:
  WAITING (for target_takeoff_ready signal from takeoff_both)
  → RISING (climb from 1.5m to 6m via vz commands, chaser HOLD trigger)
  → SETTLING (hover at 6m for 3s)
  → MOVING (smooth curved flight via velocity commands)
  → END (600s cap)

Usage:
  rosrun drone_tracking target_mover.py

Prerequisites:
  - PX4 instance 1 running and connected
  - /target/mavros running
  - takeoff_both.py has armed and flown target to 1.5m
"""

import math
import random
import rospy
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool


# ── Velocity command type mask ────────────────────────────────────────────
# Command: velocity (vx, vy, vz) + yaw_rate
# Ignore: position, acceleration, yaw
VEL_YR_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW
)


class TargetMover:

    # ── Altitude parameters ───────────────────────────────────────────────
    RISE_TO_Z      = 6.0       # target rises to this altitude after takeoff
    RISE_VZ        = 0.5       # m/s climb rate during rise
    RISE_TOLERANCE = 0.3       # m — "at altitude" threshold
    STABILIZE_TIME = 3.0       # seconds to hover after reaching altitude
    MAX_TIME       = 600.0

    # ── Soft repulsion (XY) ───────────────────────────────────────────────
    SOFT_RADIUS          = 60.0
    HARD_RADIUS          = 100.0
    MAX_REPULSION_OMEGA  = 30.0    # deg/s

    # ── Soft altitude attraction ──────────────────────────────────────────
    Z_FLOOR       = 3.0
    Z_CEIL        = 15.0
    Z_TARGET_LO   = 7.0
    Z_TARGET_HI   = 11.0
    Z_BIAS_GAIN   = 0.15

    # ── Speed parameters ──────────────────────────────────────────────────
    SPEED_MIN      = 0.2
    SPEED_MAX      = 0.8
    SPEED_EMA      = 0.05

    # ── Heading rate ──────────────────────────────────────────────────────
    OMEGA_MAX_DEG  = 15.0

    # ── Vertical speed ────────────────────────────────────────────────────
    VZ_MIN         = -0.4
    VZ_MAX         =  0.4
    VZ_EMA         = 0.05

    # ── Resample intervals ────────────────────────────────────────────────
    INTERVAL_MIN   = 5.0
    INTERVAL_MAX   = 15.0

    # ── Chaser safety ─────────────────────────────────────────────────────
    SAFETY_RADIUS  = 4.0

    def __init__(self):
        rospy.init_node('target_mover')

        # ── Current state from MAVROS ─────────────────────────────────────
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.yaw = 0.0
        self.got_pose = False

        # ── Chaser position (from chaser's pose) ─────────────────────────
        self.chaser_x = 0.0
        self.chaser_y = 0.0

        # ── Phase state ───────────────────────────────────────────────────
        self.phase = "WAITING"
        self.takeoff_ready = False
        self.chaser_phase = "UNKNOWN"
        self.rise_start_time = None
        self.settle_start_time = None
        self.motion_start_time = None

        # ── Motion state (same as v7) ─────────────────────────────────────
        self.heading = 0.0
        self.speed = 0.4
        self.speed_target = 0.4
        self.omega = 0.0
        self.vz = 0.0
        self.vz_target = 0.0
        self.next_omega_time = 0.0
        self.next_speed_time = 0.0
        self.next_vz_time    = 0.0

        # ── ROS interfaces ────────────────────────────────────────────────
        self.cmd_pub = rospy.Publisher(
            '/target/mavros/setpoint_raw/local', PositionTarget, queue_size=1)

        rospy.Subscriber('/target/mavros/local_position/pose', PoseStamped,
                         self.target_pose_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self.chaser_pose_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase', String,
                         self.phase_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_takeoff_ready', Bool,
                         self.takeoff_ready_cb, queue_size=1)

        rospy.loginfo("[TargetMover] v7.1 MAVROS-based | soft repulsion "
                      "R_soft=%.0fm R_hard=%.0fm | speed=[%.1f,%.1f] "
                      "omega_max=%.0f°/s vz=[%.1f,%.1f] | "
                      "Z_target=[%.0f,%.0f]m"
                      % (self.SOFT_RADIUS, self.HARD_RADIUS,
                         self.SPEED_MIN, self.SPEED_MAX,
                         self.OMEGA_MAX_DEG,
                         self.VZ_MIN, self.VZ_MAX,
                         self.Z_TARGET_LO, self.Z_TARGET_HI))

        self.rate = rospy.Rate(50)
        self.run()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def target_pose_cb(self, msg):
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z
        # Extract yaw from quaternion
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.got_pose = True

    def chaser_pose_cb(self, msg):
        self.chaser_x = msg.pose.position.x
        self.chaser_y = msg.pose.position.y

    def phase_cb(self, msg):
        self.chaser_phase = msg.data

    def takeoff_ready_cb(self, msg):
        if msg.data and not self.takeoff_ready:
            rospy.loginfo("[TargetMover] Target takeoff ready signal received")
            self.takeoff_ready = True

    # ── Velocity command helper ───────────────────────────────────────────

    def send_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        """Send world-frame velocity command to target drone."""
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = VEL_YR_MASK
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate = float(yaw_rate)
        self.cmd_pub.publish(msg)

    # ── Motion model (identical to v7) ────────────────────────────────────

    def _random_interval(self):
        return random.uniform(self.INTERVAL_MIN, self.INTERVAL_MAX)

    def _sample_omega(self):
        max_rad = math.radians(self.OMEGA_MAX_DEG)
        return random.uniform(-max_rad, max_rad)

    def _sample_speed(self):
        return random.uniform(self.SPEED_MIN, self.SPEED_MAX)

    def _sample_vz(self, current_z):
        base = random.uniform(self.VZ_MIN, self.VZ_MAX)
        if current_z < self.Z_TARGET_LO:
            bias = self.Z_BIAS_GAIN * (self.Z_TARGET_LO - current_z)
            return base + bias
        elif current_z > self.Z_TARGET_HI:
            bias = self.Z_BIAS_GAIN * (self.Z_TARGET_HI - current_z)
            return base + bias
        return base

    def _repulsion_omega(self, x, y):
        dist = math.hypot(x, y)
        if dist < self.SOFT_RADIUS or dist < 0.1:
            return 0.0
        t = min(1.0, (dist - self.SOFT_RADIUS) /
                max(self.HARD_RADIUS - self.SOFT_RADIUS, 1.0))
        angle_to_origin = math.atan2(-y, -x)
        diff = angle_to_origin - self.heading
        diff = math.atan2(math.sin(diff), math.cos(diff))
        max_omega = math.radians(self.MAX_REPULSION_OMEGA)
        correction = t * max_omega * (2.0 / math.pi) * diff
        return max(-max_omega, min(max_omega, correction))

    def _altitude_bias_vz(self, current_z):
        if current_z < self.Z_FLOOR + 1.0:
            return 0.3 * (self.Z_FLOOR + 1.0 - current_z)
        elif current_z > self.Z_CEIL - 1.0:
            return -0.3 * (current_z - (self.Z_CEIL - 1.0))
        return 0.0

    def _chaser_avoidance_omega(self, x, y):
        dx, dy = self.chaser_x - x, self.chaser_y - y
        dist = math.hypot(dx, dy)
        if dist > self.SAFETY_RADIUS * 2.0 or dist < 0.1:
            return 0.0
        angle_to_chaser = math.atan2(dy, dx)
        diff = angle_to_chaser - self.heading
        diff = math.atan2(math.sin(diff), math.cos(diff))
        if abs(diff) > math.pi / 2.0:
            return 0.0
        urgency = 1.0 - (dist / (self.SAFETY_RADIUS * 2.0))
        max_avoid = math.radians(25.0)
        return -math.copysign(urgency * max_avoid, diff)

    def _compute_moving_velocity(self, dt, elapsed):
        """Compute velocity for MOVING phase. Returns (vx, vy, vz, yaw_rate)."""

        # ── Resample if timers expired ────────────────────────────────────
        if elapsed >= self.next_omega_time:
            self.omega = self._sample_omega()
            self.next_omega_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New omega=%.1f°/s (next in %.1fs)"
                          % (math.degrees(self.omega),
                             self.next_omega_time - elapsed))

        if elapsed >= self.next_speed_time:
            self.speed_target = self._sample_speed()
            self.next_speed_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New speed_target=%.2f m/s (next in %.1fs)"
                          % (self.speed_target,
                             self.next_speed_time - elapsed))

        if elapsed >= self.next_vz_time:
            self.vz_target = self._sample_vz(self.pos_z)
            self.next_vz_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New vz_target=%+.2f m/s at z=%.1fm "
                          "(next in %.1fs)"
                          % (self.vz_target, self.pos_z,
                             self.next_vz_time - elapsed))

        # ── EMA smooth speed and vz ───────────────────────────────────────
        alpha_s = 1.0 - math.exp(-self.SPEED_EMA * dt * 50.0)
        self.speed += alpha_s * (self.speed_target - self.speed)

        alpha_v = 1.0 - math.exp(-self.VZ_EMA * dt * 50.0)
        self.vz += alpha_v * (self.vz_target - self.vz)

        # ── Compute effective heading rate ────────────────────────────────
        effective_omega = self.omega
        effective_omega += self._repulsion_omega(self.pos_x, self.pos_y)
        effective_omega += self._chaser_avoidance_omega(self.pos_x, self.pos_y)

        # ── Update heading (internal tracker — drone yaw follows via yaw_rate) ──
        self.heading += effective_omega * dt
        self.heading = math.atan2(math.sin(self.heading),
                                  math.cos(self.heading))

        # ── Compute world-frame velocity ──────────────────────────────────
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        vz = self.vz + self._altitude_bias_vz(self.pos_z)

        # ── Yaw rate: steer drone yaw toward heading ──────────────────────
        # The drone's actual yaw may lag behind our heading model.
        # Use a P-controller to align drone yaw with desired heading,
        # plus feed-forward from omega.
        yaw_err = self.heading - self.yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        yaw_rate = effective_omega + 1.5 * yaw_err  # P gain for yaw tracking

        # Clamp velocities for safety
        speed_limit = 1.0
        vx = max(-speed_limit, min(speed_limit, vx))
        vy = max(-speed_limit, min(speed_limit, vy))
        vz = max(-0.6, min(0.6, vz))
        yaw_rate = max(-0.8, min(0.8, yaw_rate))

        return vx, vy, vz, yaw_rate

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        dt = 1.0 / 50.0

        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # ── WAITING: hover until takeoff_both signals ready ───────────
            if self.phase == "WAITING":
                # Don't publish anything — takeoff_both owns the target
                if self.takeoff_ready:
                    rospy.loginfo("[TargetMover] Takeoff ready — waiting for "
                                  "chaser HOLD to begin rising")
                    self.phase = "WAIT_HOLD"

            # ── WAIT_HOLD: hover until chaser enters HOLD ─────────────────
            elif self.phase == "WAIT_HOLD":
                self.send_vel(0.0, 0.0, 0.0)  # hover

                if self.chaser_phase == "HOLD":
                    self.rise_start_time = now
                    self.heading = self.yaw  # sync heading to current yaw
                    self.phase = "RISING"
                    rospy.loginfo("[TargetMover] Chaser HOLD — rising from "
                                  "%.1fm to %.1fm"
                                  % (self.pos_z, self.RISE_TO_Z))

            # ── RISING: climb to RISE_TO_Z ────────────────────────────────
            elif self.phase == "RISING":
                alt_err = self.RISE_TO_Z - self.pos_z

                if alt_err > self.RISE_TOLERANCE:
                    # Climb with P-controlled vz
                    vz = min(self.RISE_VZ, alt_err * 0.5)
                    self.send_vel(0.0, 0.0, vz)
                else:
                    # At altitude — start settling
                    self.settle_start_time = now
                    self.phase = "SETTLING"
                    rospy.loginfo("[TargetMover] At %.1fm — settling for %.1fs"
                                  % (self.pos_z, self.STABILIZE_TIME))

                # If chaser loses HOLD during rise, go back to waiting
                if self.chaser_phase != "HOLD":
                    rospy.logwarn("[TargetMover] Lost HOLD during rise — "
                                  "back to WAIT_HOLD")
                    self.phase = "WAIT_HOLD"

            # ── SETTLING: hover at altitude for STABILIZE_TIME ────────────
            elif self.phase == "SETTLING":
                self.send_vel(0.0, 0.0, 0.0)  # hover

                if (now - self.settle_start_time).to_sec() >= self.STABILIZE_TIME:
                    # Initialise motion state
                    self.heading = self.yaw
                    self.speed = 0.4
                    self.speed_target = 0.4
                    self.omega = 0.0
                    self.vz = 0.0
                    self.vz_target = 0.0
                    self.next_omega_time = 0.0
                    self.next_speed_time = 0.0
                    self.next_vz_time = 0.0

                    self.motion_start_time = now
                    self.phase = "MOVING"
                    rospy.loginfo("[TargetMover] *** STARTING SMOOTH MOTION "
                                  "at z=%.1fm ***" % self.pos_z)

            # ── MOVING: smooth curved flight ──────────────────────────────
            elif self.phase == "MOVING":
                elapsed = (now - self.motion_start_time).to_sec()
                vx, vy, vz, yr = self._compute_moving_velocity(dt, elapsed)
                self.send_vel(vx, vy, vz, yr)

                dist = math.hypot(self.pos_x, self.pos_y)
                rospy.loginfo_throttle(5,
                    "[TargetMover] v7.1 | t=%.1fs pos=(%.1f,%.1f,%.1f) "
                    "hdg=%.0f° yaw=%.0f° spd=%.2f vz=%+.2f dist=%.0fm"
                    % (elapsed, self.pos_x, self.pos_y, self.pos_z,
                       math.degrees(self.heading), math.degrees(self.yaw),
                       self.speed, self.vz, dist))

                if elapsed >= self.MAX_TIME:
                    rospy.loginfo("[TargetMover] Max time — hovering")
                    self.phase = "END"

            # ── END: hover in place ───────────────────────────────────────
            elif self.phase == "END":
                self.send_vel(0.0, 0.0, 0.0)
                rospy.loginfo_throttle(10, "[TargetMover] END. Ctrl+C to exit.")

            self.rate.sleep()


if __name__ == '__main__':
    try:
        TargetMover()
    except rospy.ROSInterruptException:
        pass