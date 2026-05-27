#!/usr/bin/env python3
"""
kalman_filter_node.py  —  M9.7
================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

State vector: [cx, cy, alpha, vx, vy, valpha]

M9.6 changes (on top of M9.5):
  * Q_vel cx/cy: 6.0 → 3.0
    With Q_vel=6.0 the filter explained every YOLO measurement jump as
    "velocity changed" rather than "noise", causing the filtered cx/cy
    to track raw measurements almost 1:1 (p50 jitter unchanged after
    filtering). Halving Q_vel forces the velocity estimate to change
    more slowly, so the kinematic prediction stays smooth and the
    correction step actually attenuates measurement noise.
    Tradeoff: ~0.5-1 frame extra lag on sharp target accelerations —
    acceptable because the maneuver system ramps via EMA anyway.

  * New topic: /drone_tracking/kalman_velocity  (Point: vx, vy, valpha)
    Published alongside /drone_tracking/filtered_target so the IBVS
    SEARCH phase can extrapolate where the target went after loss.

M9.5 changes (preserved):
  * R_pos: 3.0 → 6.0 for cx/cy — aggressively smooth YOLO bbox bounce

M9.4 changes (preserved):
  * Q_vel: 4.0 → 6.0  — (now lowered to 3.0 in M9.6)
  * R_pos: 1.0 → 3.0  — (now raised further to 6.0 in M9.5)
  * PIXEL_JUMP_OUTLIER: 200 → 120 px
  * velocity_damping: 0.92 → 0.88

M9.3 changes (preserved):
  * Position-jump outlier rejection (>120px → reject)
  * MAX_CONSECUTIVE_REJECTIONS: 5 → 8
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point


class KalmanFilterNode:

    # Alpha-collapse outlier (unchanged from M7)
    OUTLIER_PREV_ALPHA_PIX = 3000.0
    OUTLIER_NEW_ALPHA_PIX  = 900.0

    # M9.4: tightened from 200px — catches grey-object false positives
    PIXEL_JUMP_OUTLIER = 120.0

    # M9.3: increased escape hatch
    MAX_CONSECUTIVE_REJECTIONS = 8

    def __init__(self):
        rospy.init_node('kalman_filter_node')

        self.x = np.zeros(6)
        self.P = np.eye(6) * 500.0

        dt = 1.0 / 20.0
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # M9.6: Q_vel cx/cy 6.0 → 3.0 — smoother cx/cy velocity estimate
        # M9.5: R_pos raised to 6.0 — smooth YOLO bbox bounce
        self.R = np.diag([6.0, 6.0, 5.0])
        self.Q = np.diag([0.5, 0.5, 3.0, 5.0, 5.0, 3.0])

        # M9.4: faster velocity decay during dropout
        self.velocity_damping = 0.88

        self.initialized            = False
        self.dropout_count          = 0
        self.max_dropout            = 30
        self.consecutive_rejections = 0

        self.pub = rospy.Publisher(
            '/drone_tracking/filtered_target', Point, queue_size=1)
        self.vel_pub = rospy.Publisher(
            '/drone_tracking/kalman_velocity', Point, queue_size=1)
        rospy.Subscriber(
            '/drone_tracking/target_center', Point, self.callback)

        rospy.loginfo("[Kalman] M9.6 | R_pos=6.0 R_alpha=5.0 | Q_vel=5.0 | "
                      "jump=%.0fpx | damp=%.2f"
                      % (self.PIXEL_JUMP_OUTLIER, self.velocity_damping))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def force_reinit_alpha(self, msg):
        """Break out of alpha-collapse rejection lock."""
        self.x[2] = msg.z
        self.x[5] = 0.0
        self.P[2, 2] = 500.0
        self.P[5, 5] = 500.0
        self.consecutive_rejections = 0
        rospy.logwarn("[Kalman] Alpha-escape after %d rejections — "
                      "alpha reset to %.0f px"
                      % (self.MAX_CONSECUTIVE_REJECTIONS, msg.z))

    def _is_outlier(self, msg):
        """
        Returns (is_outlier, reason_str).

        Rule 1 — Alpha collapse (M7, unchanged):
          Previous alpha was large (>3000px) AND new alpha is tiny (<900px).

        Rule 2 — Position jump (M9.3, tightened in M9.4):
          cx or cy changed by more than PIXEL_JUMP_OUTLIER pixels in one step.
          Catches tree/ground false positives and grey-object misdetections.
        """
        prev_alpha  = self.x[2]
        pixel_jump  = np.hypot(msg.x - self.x[0], msg.y - self.x[1])
        cx_jump     = abs(msg.x - self.x[0])
        cy_jump     = abs(msg.y - self.x[1])

        alpha_outlier = (prev_alpha > self.OUTLIER_PREV_ALPHA_PIX
                         and msg.z < self.OUTLIER_NEW_ALPHA_PIX)

        pos_outlier = (pixel_jump > self.PIXEL_JUMP_OUTLIER
                       and self.consecutive_rejections < self.MAX_CONSECUTIVE_REJECTIONS)

        if alpha_outlier:
            return True, ("alpha-collapse prev=%.0f→%.0f px jump=%.0f"
                          % (prev_alpha, msg.z, pixel_jump))
        if pos_outlier:
            return True, ("pos-jump cx=%.0f cy=%.0f (%.0fpx > %.0f limit)"
                          % (cx_jump, cy_jump, pixel_jump, self.PIXEL_JUMP_OUTLIER))
        return False, ""

    def callback(self, msg):
        is_dropout = np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z)

        if not is_dropout:
            if not self.initialized:
                self.x[0:3] = [msg.x, msg.y, msg.z]
                self.x[3:6] = 0.0
                self.initialized            = True
                self.dropout_count          = 0
                self.consecutive_rejections = 0
                rospy.loginfo("[Kalman] Initialized: cx=%.1f cy=%.1f alpha=%.1f"
                              % (msg.x, msg.y, msg.z))
                self.predict()
                self.update(np.array([msg.x, msg.y, msg.z]))

            else:
                outlier, reason = self._is_outlier(msg)

                if outlier and self.consecutive_rejections >= self.MAX_CONSECUTIVE_REJECTIONS:
                    self.force_reinit_alpha(msg)
                    self.dropout_count = 0
                    self.predict()
                    self.update(np.array([msg.x, msg.y, msg.z]))
                    is_dropout = False

                elif outlier:
                    rospy.logwarn_throttle(
                        1, "[Kalman] Outlier REJECTED (%d/%d): %s"
                        % (self.consecutive_rejections + 1,
                           self.MAX_CONSECUTIVE_REJECTIONS, reason))
                    self.consecutive_rejections += 1
                    self.dropout_count          += 1
                    self.x[3] *= self.velocity_damping
                    self.x[4] *= self.velocity_damping
                    self.x[5] *= self.velocity_damping
                    if self.dropout_count >= self.max_dropout:
                        self.x[3] = self.x[4] = self.x[5] = 0.0
                    self.predict()
                    is_dropout = True

                else:
                    self.dropout_count          = 0
                    self.consecutive_rejections = 0
                    self.predict()
                    self.update(np.array([msg.x, msg.y, msg.z]))

        else:
            self.dropout_count += 1
            if not self.initialized:
                out = Point()
                out.x = out.y = out.z = float('nan')
                self.pub.publish(out)
                return
            self.x[3] *= self.velocity_damping
            self.x[4] *= self.velocity_damping
            self.x[5] *= self.velocity_damping
            if self.dropout_count >= self.max_dropout:
                self.x[3] = self.x[4] = self.x[5] = 0.0
                rospy.logwarn_throttle(2, "[Kalman] Long dropout (%d) — vel zeroed"
                                       % self.dropout_count)
            self.predict()

        self.x[0] = np.clip(self.x[0], 0.0, 640.0)
        self.x[1] = np.clip(self.x[1], 0.0, 480.0)
        self.x[2] = np.clip(self.x[2], 0.0, 307200.0)

        # Publish filtered position
        out = Point()
        out.x = self.x[0]
        out.y = self.x[1]
        out.z = self.x[2] if not is_dropout else -self.x[2]
        self.pub.publish(out)

        # M9.6: publish velocity state for IBVS SEARCH direction
        vel = Point()
        vel.x = float(self.x[3])
        vel.y = float(self.x[4])
        vel.z = float(self.x[5])
        self.vel_pub.publish(vel)


if __name__ == '__main__':
    KalmanFilterNode()
    rospy.spin()