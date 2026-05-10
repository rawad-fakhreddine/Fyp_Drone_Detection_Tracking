#!/usr/bin/env python3
"""
target_mover.py — v7.3  (extended region, difficulty modes, evasion)
=====================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v7.3 changes:
  1. Extended region: SOFT_RADIUS=200m, HARD_RADIUS=300m
     Baylands terrain is ~686x538m — target roams freely across the park.
  2. Extended MAX_TIME: 1200s (20 minutes per run)
  3. Difficulty param: _difficulty:= (easy/medium/hard/extreme)
  4. Active evasion (hard/extreme): target steers away when chaser is close
     This creates the pursuit-evasion scenario central to the FYP.
  5. Inspired by literature (Pereira 2021, Tuncer 2023):
     - Smooth arcs with random resampling (random waypoint behavior)
     - Forward/lateral/diagonal motion patterns via heading + omega
     - Pursuit-evasion with active escape maneuvers

Usage:
  rosrun drone_tracking target_mover.py                       # medium (default)
  rosrun drone_tracking target_mover.py _difficulty:=hard
  rosrun drone_tracking target_mover.py _difficulty:=extreme
"""

import math
import random
import rospy
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool


VEL_YR_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW
)


DIFFICULTY_PARAMS = {
    'easy': {
        'speed_min': 0.2, 'speed_max': 0.4,
        'omega_max_deg': 8.0,
        'vz_min': -0.2, 'vz_max': 0.2,
        'interval_min': 8.0, 'interval_max': 20.0,
        'speed_ema': 0.03,
        'evasion_enabled': False,
        'evasion_radius': 0.0,
        'evasion_boost': 0.0,
        'speed_limit': 0.6,
    },
    'medium': {
        'speed_min': 0.3, 'speed_max': 0.6,
        'omega_max_deg': 15.0,
        'vz_min': -0.4, 'vz_max': 0.4,
        'interval_min': 5.0, 'interval_max': 15.0,
        'speed_ema': 0.05,
        'evasion_enabled': False,
        'evasion_radius': 0.0,
        'evasion_boost': 0.0,
        'speed_limit': 1.0,
    },
    'hard': {
        'speed_min': 0.4, 'speed_max': 0.8,
        'omega_max_deg': 25.0,
        'vz_min': -0.5, 'vz_max': 0.5,
        'interval_min': 3.0, 'interval_max': 10.0,
        'speed_ema': 0.08,
        'evasion_enabled': True,
        'evasion_radius': 6.0,
        'evasion_boost': 0.3,
        'speed_limit': 1.2,
    },
    'extreme': {
        'speed_min': 0.5, 'speed_max': 1.0,
        'omega_max_deg': 35.0,
        'vz_min': -0.6, 'vz_max': 0.6,
        'interval_min': 2.0, 'interval_max': 7.0,
        'speed_ema': 0.10,
        'evasion_enabled': True,
        'evasion_radius': 8.0,
        'evasion_boost': 0.5,
        'speed_limit': 1.5,
    },
}


class TargetMover:

    # ── Altitude ──────────────────────────────────────────────────────
    RISE_TO_Z      = 6.0
    RISE_VZ        = 0.5
    RISE_TOLERANCE = 0.3
    STABILIZE_TIME = 3.0
    MAX_TIME       = 1200.0     # 20 minutes

    # ── Extended region (covers baylands park: ~686x538m) ─────────────
    SOFT_RADIUS          = 200.0     # free roaming up to 200m from spawn
    HARD_RADIUS          = 300.0     # hard repulsion beyond 300m
    MAX_REPULSION_OMEGA  = 30.0

    # ── Altitude attraction ───────────────────────────────────────────
    Z_FLOOR       = 3.0
    Z_CEIL        = 15.0
    Z_TARGET_LO   = 7.0
    Z_TARGET_HI   = 11.0
    Z_BIAS_GAIN   = 0.15

    # ── Chaser safety ─────────────────────────────────────────────────
    SAFETY_RADIUS  = 4.0

    def __init__(self):
        rospy.init_node('target_mover')

        self.difficulty = rospy.get_param('~difficulty', 'medium')
        if self.difficulty not in DIFFICULTY_PARAMS:
            rospy.logwarn("[TargetMover] Unknown difficulty '%s', using 'medium'"
                          % self.difficulty)
            self.difficulty = 'medium'
        p = DIFFICULTY_PARAMS[self.difficulty]

        self.SPEED_MIN      = p['speed_min']
        self.SPEED_MAX      = p['speed_max']
        self.OMEGA_MAX_DEG  = p['omega_max_deg']
        self.VZ_MIN         = p['vz_min']
        self.VZ_MAX         = p['vz_max']
        self.INTERVAL_MIN   = p['interval_min']
        self.INTERVAL_MAX   = p['interval_max']
        self.SPEED_EMA      = p['speed_ema']
        self.VZ_EMA         = p['speed_ema']
        self.evasion_enabled = p['evasion_enabled']
        self.evasion_radius  = p['evasion_radius']
        self.evasion_boost   = p['evasion_boost']
        self.speed_limit     = p['speed_limit']

        self.pos_x = self.pos_y = self.pos_z = 0.0
        self.yaw = 0.0
        self.got_pose = False
        self.chaser_x = self.chaser_y = 0.0

        self.phase = "WAITING"
        self.takeoff_ready = False
        self.chaser_phase = "UNKNOWN"
        self.rise_start_time = None
        self.settle_start_time = None
        self.motion_start_time = None

        self.heading = 0.0
        self.speed = 0.4
        self.speed_target = 0.4
        self.omega = 0.0
        self.vz = 0.0
        self.vz_target = 0.0
        self.next_omega_time = 0.0
        self.next_speed_time = 0.0
        self.next_vz_time    = 0.0

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

        rospy.loginfo("[TargetMover] v7.3 difficulty=%s | speed=[%.1f,%.1f] "
                      "omega_max=%.0f deg/s vz=[%.1f,%.1f] | "
                      "evasion=%s (R=%.1fm boost=%.1f) | "
                      "region=%.0fm max_time=%.0fs"
                      % (self.difficulty,
                         self.SPEED_MIN, self.SPEED_MAX,
                         self.OMEGA_MAX_DEG,
                         self.VZ_MIN, self.VZ_MAX,
                         "ON" if self.evasion_enabled else "OFF",
                         self.evasion_radius, self.evasion_boost,
                         self.SOFT_RADIUS, self.MAX_TIME))

        self.rate = rospy.Rate(50)
        self.run()

    def target_pose_cb(self, msg):
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z
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

    def send_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = VEL_YR_MASK
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate = float(yaw_rate)
        self.cmd_pub.publish(msg)

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

    def _evasion(self, x, y):
        """Active evasion: steer away from chaser and boost speed."""
        if not self.evasion_enabled:
            return 0.0, 0.0

        dx = self.chaser_x - x
        dy = self.chaser_y - y
        dist = math.hypot(dx, dy)

        if dist > self.evasion_radius or dist < 0.1:
            return 0.0, 0.0

        urgency = 1.0 - (dist / self.evasion_radius)
        urgency = urgency ** 1.5

        angle_to_chaser = math.atan2(dy, dx)
        away_heading = angle_to_chaser + math.pi
        diff = away_heading - self.heading
        diff = math.atan2(math.sin(diff), math.cos(diff))

        max_evasion_omega = math.radians(40.0)
        evasion_omega = urgency * max_evasion_omega * (2.0 / math.pi) * diff
        evasion_omega = max(-max_evasion_omega, min(max_evasion_omega,
                                                     evasion_omega))
        speed_boost = urgency * self.evasion_boost

        if urgency > 0.5:
            rospy.loginfo_throttle(2,
                "[TargetMover] EVADING! dist=%.1fm urgency=%.2f "
                "omega=%+.0f deg/s boost=+%.2f m/s"
                % (dist, urgency, math.degrees(evasion_omega), speed_boost))

        return evasion_omega, speed_boost

    def _compute_moving_velocity(self, dt, elapsed):
        if elapsed >= self.next_omega_time:
            self.omega = self._sample_omega()
            self.next_omega_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New omega=%.1f deg/s (next in %.1fs)"
                          % (math.degrees(self.omega),
                             self.next_omega_time - elapsed))

        if elapsed >= self.next_speed_time:
            self.speed_target = self._sample_speed()
            self.next_speed_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New speed=%.2f m/s (next in %.1fs)"
                          % (self.speed_target,
                             self.next_speed_time - elapsed))

        if elapsed >= self.next_vz_time:
            self.vz_target = self._sample_vz(self.pos_z)
            self.next_vz_time = elapsed + self._random_interval()
            rospy.loginfo("[TargetMover] New vz=%+.2f m/s at z=%.1fm"
                          % (self.vz_target, self.pos_z))

        alpha_s = 1.0 - math.exp(-self.SPEED_EMA * dt * 50.0)
        self.speed += alpha_s * (self.speed_target - self.speed)
        alpha_v = 1.0 - math.exp(-self.VZ_EMA * dt * 50.0)
        self.vz += alpha_v * (self.vz_target - self.vz)

        effective_omega = self.omega
        effective_omega += self._repulsion_omega(self.pos_x, self.pos_y)
        effective_omega += self._chaser_avoidance_omega(self.pos_x, self.pos_y)

        evasion_omega, speed_boost = self._evasion(self.pos_x, self.pos_y)
        effective_omega += evasion_omega

        self.heading += effective_omega * dt
        self.heading = math.atan2(math.sin(self.heading),
                                  math.cos(self.heading))

        effective_speed = self.speed + speed_boost
        vx = effective_speed * math.cos(self.heading)
        vy = effective_speed * math.sin(self.heading)
        vz = self.vz + self._altitude_bias_vz(self.pos_z)

        yaw_err = self.heading - self.yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        yaw_rate = effective_omega + 1.5 * yaw_err

        vx = max(-self.speed_limit, min(self.speed_limit, vx))
        vy = max(-self.speed_limit, min(self.speed_limit, vy))
        vz = max(-0.6, min(0.6, vz))
        yaw_rate = max(-0.8, min(0.8, yaw_rate))

        return vx, vy, vz, yaw_rate

    def run(self):
        dt = 1.0 / 50.0

        while not rospy.is_shutdown():
            now = rospy.Time.now()

            if self.phase == "WAITING":
                if self.takeoff_ready:
                    rospy.loginfo("[TargetMover] Takeoff ready — waiting for "
                                  "chaser HOLD")
                    self.phase = "WAIT_HOLD"

            elif self.phase == "WAIT_HOLD":
                self.send_vel(0.0, 0.0, 0.0)
                if self.chaser_phase == "HOLD":
                    self.rise_start_time = now
                    self.heading = self.yaw
                    self.phase = "RISING"
                    rospy.loginfo("[TargetMover] Chaser HOLD — rising to %.1fm"
                                  % self.RISE_TO_Z)

            elif self.phase == "RISING":
                alt_err = self.RISE_TO_Z - self.pos_z
                if alt_err > self.RISE_TOLERANCE:
                    vz = min(self.RISE_VZ, alt_err * 0.5)
                    self.send_vel(0.0, 0.0, vz)
                else:
                    self.settle_start_time = now
                    self.phase = "SETTLING"
                    rospy.loginfo("[TargetMover] At %.1fm — settling %.1fs"
                                  % (self.pos_z, self.STABILIZE_TIME))

                if self.chaser_phase != "HOLD":
                    rospy.logwarn("[TargetMover] Lost HOLD — back to WAIT_HOLD")
                    self.phase = "WAIT_HOLD"

            elif self.phase == "SETTLING":
                self.send_vel(0.0, 0.0, 0.0)
                if (now - self.settle_start_time).to_sec() >= self.STABILIZE_TIME:
                    self.heading = self.yaw
                    self.speed = (self.SPEED_MIN + self.SPEED_MAX) / 2.0
                    self.speed_target = self.speed
                    self.omega = 0.0
                    self.vz = 0.0
                    self.vz_target = 0.0
                    self.next_omega_time = 0.0
                    self.next_speed_time = 0.0
                    self.next_vz_time = 0.0
                    self.motion_start_time = now
                    self.phase = "MOVING"
                    rospy.loginfo("[TargetMover] *** %s MOTION at z=%.1fm ***"
                                  % (self.difficulty.upper(), self.pos_z))

            elif self.phase == "MOVING":
                elapsed = (now - self.motion_start_time).to_sec()
                vx, vy, vz, yr = self._compute_moving_velocity(dt, elapsed)
                self.send_vel(vx, vy, vz, yr)

                dist_origin = math.hypot(self.pos_x, self.pos_y)
                chaser_dist = math.hypot(self.chaser_x - self.pos_x,
                                         self.chaser_y - self.pos_y)
                rospy.loginfo_throttle(5,
                    "[TargetMover] v7.3 %s | t=%.0fs pos=(%.0f,%.0f,%.1f) "
                    "hdg=%.0f spd=%.2f vz=%+.2f | chaser=%.1fm origin=%.0fm"
                    % (self.difficulty, elapsed,
                       self.pos_x, self.pos_y, self.pos_z,
                       math.degrees(self.heading), self.speed, self.vz,
                       chaser_dist, dist_origin))

                if elapsed >= self.MAX_TIME:
                    rospy.loginfo("[TargetMover] Max time (%.0fs) — hovering"
                                  % self.MAX_TIME)
                    self.phase = "END"

            elif self.phase == "END":
                self.send_vel(0.0, 0.0, 0.0)
                rospy.loginfo_throttle(10, "[TargetMover] END.")

            self.rate.sleep()


if __name__ == '__main__':
    try:
        TargetMover()
    except rospy.ROSInterruptException:
        pass