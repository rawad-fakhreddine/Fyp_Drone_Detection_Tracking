#!/usr/bin/env python3
"""
ibvs_controller_node.py  —  v6.13  (3m hold + faster closing)
==============================================================

Changes from v6.12:
  * alpha_star: 0.0038 → 0.0067   (hold ≈4m → ≈3m)
  * K_far: 10.0 → 14.0            (faster closing — vx ≈ 0.6m/s when target far)
  * err_a_max: 0.012 → 0.018      (proportional to new alpha_star)
  * HOLD entry err_a: 0.0025 → 0.005 (proportional)
  * PPO clamp: [0.002, 0.008] → [0.003, 0.012]

Diagnosis: M7.2 v6.12 produced cmd_vx=0.34 m/s mean during HOLD with target
at ~6m. Target moves at 0.4-0.7 m/s. Chaser couldn't close the gap because
its forward speed at typical err_a level was below target speed. Raising
K_far to 14 produces vx≈0.6 m/s at typical err_a, enough to close.

Carried over: BODY-FRAME publisher, max_vx=0.7, max_vy=0.85.
"""

import rospy
import math
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import Bool, String


BODY_VEL_TYPE_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW
)


class IBVSController:

    def __init__(self):
        rospy.init_node('ibvs_controller_node')

        self.USE_PPO = False

        # Camera
        self.img_w  = 640.0
        self.img_h  = 480.0
        self.img_cx = self.img_w / 2.0
        self.img_cy = self.img_h / 2.0
        self.area_norm = self.img_w * self.img_h
        self.pitch_compensation_gain = 0.8

        # ── Setpoints (M7.2 v6.13: 3m hold) ─────────────────────────────────
        # alpha_star = 0.0067 corresponds to ~3m chaser-to-target distance
        self.x_star     = 0.0
        self.y_star     = 0.0
        self.alpha_star = 0.0067
        self.lam        = 0.5

        # ── Distance gains (M7.2 v6.13: K_far raised) ──────────────────────
        # K_far = 14 produces vx = 14·sqrt(0.005)·0.65 ≈ 0.64 m/s at typical
        # err_a in HOLD. Was 10 → produced only 0.46 m/s, slower than target's
        # 0.4-0.7 m/s speed regimes. Now chaser can actually close the gap.
        self.K_far  = 14.0
        self.K_near = 6.0

        # Y / Z / yaw PID
        self.Kp_y,  self.Ki_y,  self.Kd_y   = 1.4, 0.05, 0.3
        # M7.2 v6.14: Kp_z raised 1.2 → 1.8 to track fast vertical target motion.
        # At err_y=0.15, old: vz=0.12 m/s (too slow). New: vz=0.18 m/s.
        self.Kp_z,  self.Ki_z,  self.Kd_z   = 1.8, 0.04, 0.5
        self.Kp_wz, self.Ki_wz, self.Kd_wz  = 0.9, 0.0,  0.15

        # Velocity limits
        self.max_vx         = 0.70
        self.max_vx_retreat = 0.50
        self.max_vy         = 0.85
        # M7.2 v6.14: max_vz raised 0.5 → 0.8. Target's climb_dive mode hits
        # 0.7 m/s vertical, chaser needs margin to actually catch up.
        self.max_vz         = 0.8
        self.max_wz         = 0.4

        # Phase thresholds
        # M7.2 v6.15: min_altitude_safe raised 1.0 → 6.0. Chaser climbs to 6m
        # in SEARCH (above tree canopy), then any descent during APPROACH/HOLD
        # is clamped at 6m. Prevents tree collisions when target descends.
        self.min_altitude_safe = 1.0
        self.alpha_min_valid   = 0.0005

        # ── Error saturation (M7.2 v6.13: rescaled) ────────────────────────
        # err_a_max = 0.018: at alpha_star=0.0067, caps err_a at +0.018 →
        # alpha=0.025 (chaser ~1.5m away, must retreat hard).
        self.err_x_max = 0.8
        self.err_y_max = 0.8
        self.err_a_max = 0.018

        # Integral windup limits
        self.int_y_max = 0.2
        self.int_z_max = 0.2

        # Detection timeouts
        self.detection_timeout = 3.0
        self.stale_timeout     = 1.5
        self.ppo_timeout       = 2.0

        # PRED-mode gain
        self.pred_gain_scale = 0.7

        # APPROACH startup ramp
        self.APPROACH_RAMP_S = 2.0
        self.approach_start_time = None

        # Recovery sub-state
        self.recovery_duration   = 2.0
        self.recovery_start_time = None

        # Velocity smoothing
        self.vel_smooth_normal   = 0.5
        self.vel_smooth_reversal = 0.1

        # Internal state
        self.cx = self.cy = None
        self.alpha = 0.0
        self.got_real_detection = False
        self.is_prediction      = False
        self.last_real_detection_time = None
        self.armed = False
        self.altitude     = 0.0
        self.current_pitch = 0.0
        self.takeoff_ready = False
        self.phase = "TAKEOFF"

        # PID history
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.int_err_y  = 0.0
        self.int_err_z  = 0.0

        # PPO timing
        self.last_ppo_time = None

        # Velocity smoothing memory
        self.prev_vx = self.prev_vy = self.prev_vz = self.prev_wz = 0.0

        # ROS
        self.cmd_pub = rospy.Publisher(
            '/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.active_pub = rospy.Publisher(
            '/drone_tracking/ibvs_active', Bool, queue_size=1)
        self.phase_pub = rospy.Publisher(
            '/drone_tracking/ibvs_phase', String, queue_size=1)

        rospy.Subscriber('/drone_tracking/filtered_target', Point,
                         self.detection_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints', Quaternion,
                         self.setpoints_cb, queue_size=1)
        rospy.Subscriber('/mavros/state', State,
                         self.state_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self.pose_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready', Bool,
                         self.takeoff_ready_cb, queue_size=1)

        self.dt   = 1.0 / 20.0
        self.rate = rospy.Rate(20)

        rospy.loginfo("[IBVS] v6.15 started (PPO=%s) "
                      "alpha_star=%.4f hold≈3m | max_vx=%.2f max_vz=%.2f Kp_z=%.1f "
                      "min_alt=%.1fm"
                      % ("ON" if self.USE_PPO else "OFF",
                         self.alpha_star, self.max_vx, self.max_vz,
                         self.Kp_z, self.min_altitude_safe))
        rospy.loginfo("[IBVS] Publishing to /mavros/setpoint_raw/local "
                      "(FRAME_BODY_NED, mask=0x%X)" % BODY_VEL_TYPE_MASK)
        rospy.loginfo("[IBVS] Waiting for takeoff_node to signal ready...")
        self.run()

    def state_cb(self, msg):
        self.armed = msg.armed

    def pose_cb(self, msg):
        self.altitude = msg.pose.position.z
        q = msg.pose.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            self.current_pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.current_pitch = math.asin(sinp)

    def takeoff_ready_cb(self, msg):
        if msg.data and not self.takeoff_ready:
            rospy.loginfo("[IBVS] Takeoff complete signal received")
            self.takeoff_ready = True

    def detection_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.got_real_detection = False
            self.is_prediction      = False
            return

        if self.phase in ("TAKEOFF", "DISARMED"):
            return

        self.cx = msg.x
        self.cy = msg.y
        self.alpha = np.clip(abs(msg.z) / self.area_norm, 0.0, 1.0)

        if msg.z > 0:
            self.got_real_detection = True
            self.is_prediction      = False
            self.last_real_detection_time = rospy.Time.now()
        else:
            self.got_real_detection = False
            self.is_prediction      = True

    def setpoints_cb(self, msg):
        if not self.USE_PPO:
            return
        self.x_star     = np.clip(float(msg.x), -0.3, 0.3)
        self.y_star     = np.clip(float(msg.y), -0.3, 0.3)
        # PPO clamp rescaled: [0.003, 0.012] gives PPO authority over
        # ~2.3m to ~4.7m chaser-to-target distance.
        self.alpha_star = np.clip(float(msg.z), 0.003, 0.012)
        self.lam        = np.clip(float(msg.w), 0.3, 1.0)
        self.last_ppo_time = rospy.Time.now()

    def time_since_detection(self):
        if self.last_real_detection_time is None:
            return float('inf')
        return (rospy.Time.now() - self.last_real_detection_time).to_sec()

    def ppo_is_active(self):
        if not self.USE_PPO or self.last_ppo_time is None:
            return False
        return (rospy.Time.now() - self.last_ppo_time).to_sec() < self.ppo_timeout

    def reset_pid(self):
        self.prev_err_x = self.prev_err_y = 0.0
        self.int_err_y  = self.int_err_z  = 0.0

    def in_recovery(self):
        if self.recovery_start_time is None:
            return False
        if (rospy.Time.now() - self.recovery_start_time).to_sec() > self.recovery_duration:
            self.recovery_start_time = None
            rospy.loginfo("[IBVS] Recovery complete")
            return False
        return True

    def approach_ramp_factor(self):
        if self.approach_start_time is None:
            return 1.0
        elapsed = (rospy.Time.now() - self.approach_start_time).to_sec()
        if elapsed >= self.APPROACH_RAMP_S:
            self.approach_start_time = None
            return 1.0
        return elapsed / self.APPROACH_RAMP_S

    def smooth(self, prev_v, new_v):
        if prev_v * new_v < 0.0 and abs(new_v) > 0.05:
            s = self.vel_smooth_reversal
        else:
            s = self.vel_smooth_normal
        return s * prev_v + (1.0 - s) * new_v

    def _build_body_vel_msg(self, vx=0.0, vy=0.0, vz=0.0, wz=0.0):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        msg.type_mask = BODY_VEL_TYPE_MASK
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate   = float(wz)
        return msg

    def compute_velocities(self, gain_scale=1.0):
        err_x = (self.cx - self.img_cx) / self.img_cx - self.x_star
        err_y_raw = (self.cy - self.img_cy) / self.img_cy - self.y_star
        err_y = err_y_raw - self.current_pitch * self.pitch_compensation_gain
        err_a = self.alpha - self.alpha_star

        err_x = np.clip(err_x, -self.err_x_max, self.err_x_max)
        err_y = np.clip(err_y, -self.err_y_max, self.err_y_max)
        err_a = np.clip(err_a, -self.err_a_max, self.err_a_max)

        d_err_x = (err_x - self.prev_err_x) / self.dt
        d_err_y = (err_y - self.prev_err_y) / self.dt

        self.int_err_y = np.clip(self.int_err_y + err_x * self.dt,
                                 -self.int_y_max, self.int_y_max)
        self.int_err_z = np.clip(self.int_err_z + err_y * self.dt,
                                 -self.int_z_max, self.int_z_max)

        self.prev_err_x = err_x
        self.prev_err_y = err_y

        if self.ppo_is_active():
            lam_gain = 0.3 + 0.7 * self.lam
        else:
            lam_gain = 0.65
        gain = gain_scale * lam_gain

        if self.in_recovery():
            vx = 0.0
        elif err_a < 0.0:
            vx = self.K_far * np.sqrt(-err_a) * gain
        else:
            vx = -self.K_near * np.sqrt(err_a) * gain
        vx = np.clip(vx, -self.max_vx_retreat, self.max_vx)

        vy = -gain * (self.Kp_y  * err_x + self.Ki_y * self.int_err_y +
                      self.Kd_y  * d_err_x)
        vz = -gain * (self.Kp_z  * err_y + self.Ki_z * self.int_err_z +
                      self.Kd_z  * d_err_y)
        wz = -gain * (self.Kp_wz * err_x + self.Kd_wz * d_err_x)

        vy = np.clip(vy, -self.max_vy, self.max_vy)
        vz = np.clip(vz, -self.max_vz, self.max_vz)
        wz = np.clip(wz, -self.max_wz, self.max_wz)

        vx = self.smooth(self.prev_vx, vx)
        vy = self.smooth(self.prev_vy, vy)
        vz = self.smooth(self.prev_vz, vz)
        wz = self.smooth(self.prev_wz, wz)

        self.prev_vx, self.prev_vy = vx, vy
        self.prev_vz, self.prev_wz = vz, wz
        return vx, vy, vz, wz

    def run(self):
        while not rospy.is_shutdown():
            cmd_vx = cmd_vy = cmd_vz = cmd_wz = 0.0
            publish_cmd = True

            if not self.armed:
                self.phase = "DISARMED"
                publish_cmd = False

            elif self.phase == "DISARMED":
                rospy.loginfo("[IBVS] Armed — waiting for takeoff_node")
                self.phase = "TAKEOFF"
                publish_cmd = False

            elif self.phase == "TAKEOFF":
                if self.takeoff_ready:
                    rospy.loginfo("[IBVS] Entering SEARCH")
                    self.phase = "SEARCH"
                publish_cmd = False

            elif self.phase == "SEARCH":
                # M7.2 v6.15: climb cap raised 0.30 → 0.50 since target altitude
                # is now 6m (vs old 1m), need faster ascent to avoid long SEARCH.
                alt_err = self.min_altitude_safe - self.altitude
                cmd_vz = float(np.clip(alt_err * 0.3, -0.20, 0.30))
                if self.got_real_detection and self.alpha > self.alpha_min_valid:
                    rospy.loginfo("[IBVS] Target acquired α=%.4f → APPROACH (ramping)"
                                  % self.alpha)
                    self.reset_pid()
                    self.approach_start_time = rospy.Time.now()
                    self.phase = "APPROACH"

            elif self.phase == "APPROACH":
                det_age = self.time_since_detection()
                if det_age > self.detection_timeout:
                    rospy.logwarn("[IBVS] Lost target (%.1fs) → SEARCH" % det_age)
                    self.reset_pid()
                    self.phase = "SEARCH"
                elif det_age > self.stale_timeout:
                    pass
                elif self.got_real_detection:
                    ramp = self.approach_ramp_factor()
                    cmd_vx, cmd_vy, cmd_vz, cmd_wz = self.compute_velocities(
                        gain_scale=ramp)

                    err_x = abs((self.cx - self.img_cx) / self.img_cx - self.x_star)
                    err_y = abs((self.cy - self.img_cy) / self.img_cy - self.y_star)
                    err_a = abs(self.alpha - self.alpha_star)
                    # HOLD entry err_a threshold rescaled with new alpha_star
                    if err_x < 0.12 and err_y < 0.12 and err_a < 0.005:
                        rospy.loginfo("[IBVS] Centered → HOLD")
                        self.phase = "HOLD"
                elif self.is_prediction:
                    cmd_vx, cmd_vy, cmd_vz, cmd_wz = self.compute_velocities(
                        gain_scale=self.pred_gain_scale)

            elif self.phase == "HOLD":
                det_age = self.time_since_detection()
                if det_age > self.detection_timeout:
                    rospy.logwarn("[IBVS] Lost in HOLD → APPROACH")
                    self.reset_pid()
                    self.phase = "APPROACH"
                    self.recovery_start_time = rospy.Time.now()
                elif det_age > self.stale_timeout:
                    rospy.logwarn_throttle(1, "[IBVS] HOLD stale (%.1fs) — hover"
                                           % det_age)
                elif self.got_real_detection:
                    cmd_vx, cmd_vy, cmd_vz, cmd_wz = self.compute_velocities(
                        gain_scale=1.0)
                elif self.is_prediction:
                    cmd_vx, cmd_vy, cmd_vz, cmd_wz = self.compute_velocities(
                        gain_scale=self.pred_gain_scale)

            if (publish_cmd and self.armed and self.altitude < 0.5
                    and self.phase not in ("TAKEOFF", "DISARMED")):
                cmd_vz = max(cmd_vz, 0.3)

            if publish_cmd:
                msg = self._build_body_vel_msg(cmd_vx, cmd_vy, cmd_vz, cmd_wz)
                self.cmd_pub.publish(msg)
            self.active_pub.publish(Bool(data=(self.phase in ("APPROACH", "HOLD"))))
            self.phase_pub.publish(String(data=self.phase))

            if self.phase in ("APPROACH", "HOLD") and self.cx is not None:
                err_x_v = (self.cx - self.img_cx) / self.img_cx - self.x_star
                err_y_v = (self.cy - self.img_cy) / self.img_cy - self.y_star
                err_a_v = self.alpha - self.alpha_star
                rospy.loginfo_throttle(2,
                    "[IBVS] %s%s | ex=%.3f ey=%.3f ea=%.4f α=%.4f pitch=%.1f° | "
                    "vx=%.2f vy=%.2f vz=%.2f wz=%.2f | alt=%.1f det=%s"
                    % (self.phase, " (REC)" if self.in_recovery() else "",
                       err_x_v, err_y_v, err_a_v, self.alpha,
                       math.degrees(self.current_pitch),
                       cmd_vx, cmd_vy, cmd_vz, cmd_wz,
                       self.altitude,
                       "REAL" if self.got_real_detection else
                       "PRED" if self.is_prediction else "NONE"))

            self.rate.sleep()


if __name__ == '__main__':
    IBVSController()