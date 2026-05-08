#!/usr/bin/env python3
"""
kalman_filter_node.py  (M7 patch — rejection-rule escape)
---------------------------------------------------------
State vector: [cx, cy, alpha, vx, vy, valpha]

Changes from previous version:
  * Renamed the "close-range collapse" rule to "outlier_reject" — it
    fires on any sudden bbox shrinkage, not just close range.
  * Added MAX_CONSECUTIVE_REJECTIONS escape: after N rejected real
    frames in a row, force-accept the next measurement and re-init
    the alpha component. Prevents the 6.8s lock observed at t=190s.
  * Counter is reset to 0 on every clean update.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point


class KalmanFilterNode:

    # Outlier rule thresholds (kept from previous version)
    OUTLIER_PREV_ALPHA_PIX = 3000.0   # only check when prev alpha is large
    OUTLIER_NEW_ALPHA_PIX  = 900.0    # raw must be smaller than this to fire
    PIXEL_JUMP_PIX         = 100.0    # logged but not used to gate rejection

    # NEW: escape hatch
    MAX_CONSECUTIVE_REJECTIONS = 5    # ≈ 0.17 s at 30 Hz; break the lock

    def __init__(self):
        rospy.init_node('kalman_filter_node')

        # State
        self.x = np.zeros(6)
        self.P = np.eye(6) * 500.0

        # Transition matrix
        dt = 1.0 / 20.0
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix (observe cx, cy, alpha)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Noise
        self.R = np.diag([1.0, 1.0, 5.0])
        self.Q = np.diag([1.0, 1.0, 2.0, 0.5, 0.5, 0.5])

        # Velocity damping during dropout
        self.velocity_damping = 0.85

        # Internal state
        self.initialized              = False
        self.dropout_count            = 0
        self.max_dropout              = 30
        self.consecutive_rejections   = 0   # NEW

        # ROS
        self.pub = rospy.Publisher(
            '/drone_tracking/filtered_target', Point, queue_size=1)
        rospy.Subscriber(
            '/drone_tracking/target_center', Point, self.callback)

        rospy.loginfo("[Kalman] Filter node started.")

    # ── Predict / Update ───────────────────────────────────────────────────

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    # ── Re-initialise α-component after a forced escape ────────────────────

    def force_reinit_alpha(self, msg):
        """
        Break out of a rejection lock. Reset α and α-rate to fresh
        measurement, but keep cx/cy state — those usually weren't lying.
        Inflate P[2,2] so the next few updates pull α toward truth fast.
        """
        self.x[2] = msg.z
        self.x[5] = 0.0
        self.P[2, 2] = 500.0
        self.P[5, 5] = 500.0
        self.consecutive_rejections = 0
        rospy.logwarn(
            "[Kalman] Escape after %d consecutive rejections — "
            "α reset to %.0f px"
            % (self.MAX_CONSECUTIVE_REJECTIONS, msg.z))

    # ── Callback ───────────────────────────────────────────────────────────

    def callback(self, msg):
        is_dropout = np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z)

        if not is_dropout:
            if not self.initialized:
                # First detection — initialise state
                self.x[0:3] = [msg.x, msg.y, msg.z]
                self.x[3:6] = 0.0
                self.initialized = True
                self.dropout_count = 0
                self.consecutive_rejections = 0
                rospy.loginfo(
                    "[Kalman] Initialized from first detection: "
                    "cx=%.1f cy=%.1f alpha=%.1f" % (msg.x, msg.y, msg.z))
                self.predict()
                self.update(np.array([msg.x, msg.y, msg.z]))
            else:
                # Outlier check
                prev_alpha = self.x[2]
                outlier = (prev_alpha > self.OUTLIER_PREV_ALPHA_PIX
                           and msg.z   < self.OUTLIER_NEW_ALPHA_PIX)
                pixel_jump = np.hypot(msg.x - self.x[0], msg.y - self.x[1])

                # Escape hatch: too many rejections in a row → force-accept
                if outlier and self.consecutive_rejections >= self.MAX_CONSECUTIVE_REJECTIONS:
                    self.force_reinit_alpha(msg)
                    self.dropout_count = 0
                    self.predict()
                    self.update(np.array([msg.x, msg.y, msg.z]))
                    is_dropout = False

                elif outlier:
                    rospy.logwarn_throttle(
                        1, "[Kalman] Outlier rejected (%d/%d): "
                        "prev_alpha=%.0f → raw_alpha=%.0f  pixel_jump=%.0f"
                        % (self.consecutive_rejections + 1,
                           self.MAX_CONSECUTIVE_REJECTIONS,
                           prev_alpha, msg.z, pixel_jump))
                    self.consecutive_rejections += 1
                    self.dropout_count          += 1
                    self.x[3] *= self.velocity_damping
                    self.x[4] *= self.velocity_damping
                    self.x[5] *= self.velocity_damping
                    if self.dropout_count >= self.max_dropout:
                        self.x[3] = self.x[4] = self.x[5] = 0.0
                    self.predict()
                    is_dropout = True   # publish as predicted

                else:
                    # Normal update — clear all rejection state
                    self.dropout_count          = 0
                    self.consecutive_rejections = 0
                    self.predict()
                    self.update(np.array([msg.x, msg.y, msg.z]))

        else:
            # YOLO dropout (NaN)
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
                rospy.logwarn_throttle(
                    2, "[Kalman] Long dropout (%d steps) — velocities zeroed"
                    % self.dropout_count)
            self.predict()

        # Clip state to image bounds
        self.x[0] = np.clip(self.x[0], 0.0, 640.0)
        self.x[1] = np.clip(self.x[1], 0.0, 480.0)
        self.x[2] = np.clip(self.x[2], 0.0, 307200.0)

        # Publish (z>0 = real, z<0 = predicted)
        out = Point()
        out.x = self.x[0]
        out.y = self.x[1]
        out.z = self.x[2] if not is_dropout else -self.x[2]
        self.pub.publish(out)


if __name__ == '__main__':
    KalmanFilterNode()
    rospy.spin()