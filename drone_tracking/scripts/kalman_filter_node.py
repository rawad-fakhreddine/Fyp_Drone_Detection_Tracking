#!/usr/bin/env python3
"""
kalman_filter_node.py  —  M9.3  (position-jump outlier rejection)
=================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

State vector: [cx, cy, alpha, vx, vy, valpha]

M9.3 changes (on top of M9.2):
  * Added POSITION-JUMP outlier rejection:
    If cx or cy changes by more than PIXEL_JUMP_OUTLIER (200px) in one
    frame, the measurement is rejected as a false positive (tree / ground).
    This is independent of the existing alpha-collapse rule.

    Root cause: at Z_FLOOR=6m, target was descending to tree level due
    to a WAIT_HOLD bug. YOLO detected trees with cx jumps of 142-426px.
    The old Kalman accepted these because alpha didn't collapse.

    Threshold: 200px (3× the 60px max seen in normal fast tracking,
    safely below the 426px tree jumps).

  * MAX_CONSECUTIVE_REJECTIONS increased: 5 → 8
    Prevents early escape from a sustained occlusion (tree in background)
    while keeping the 5-step alpha-collapse escape for close-range events.

  * All M9.2 changes preserved:
    Q_vel=4.0/4.0/2.0, Q_pos=2.0/2.0/3.0, velocity_damping=0.92
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point


class KalmanFilterNode:

    # Alpha-collapse outlier (unchanged from M7)
    OUTLIER_PREV_ALPHA_PIX = 3000.0
    OUTLIER_NEW_ALPHA_PIX  = 900.0

    # M9.3: position-jump outlier
    PIXEL_JUMP_OUTLIER = 120.0   # px — rejects tree false positives (426px)
                                  # safe for fast tracking (max ~60px normal)

    # M9.3: increased escape hatch for sustained occlusion
    MAX_CONSECUTIVE_REJECTIONS = 8   # M9.2 was 5

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

        # M9.2 noise tuning (fast-target calibrated)
        self.R = np.diag([3.0, 3.0, 5.0])
        self.Q = np.diag([2.0, 2.0, 3.0, 6.0, 6.0, 3.0])

        self.velocity_damping = 0.88   # M9.2

        self.initialized            = False
        self.dropout_count          = 0
        self.max_dropout            = 30
        self.consecutive_rejections = 0

        self.pub = rospy.Publisher(
            '/drone_tracking/filtered_target', Point, queue_size=1)
        rospy.Subscriber(
            '/drone_tracking/target_center', Point, self.callback)

        rospy.loginfo("[Kalman] M9.3 started — "
                      "alpha-collapse + position-jump (>%.0fpx) rejection"
                      % self.PIXEL_JUMP_OUTLIER)

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
          Catches close-range bbox collapse.

        Rule 2 — Position jump (M9.3, new):
          cx or cy changed by more than PIXEL_JUMP_OUTLIER pixels in one step.
          Catches tree/ground false positives.
          NOT fired if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS
          (escape hatch still works).
        """
        prev_alpha  = self.x[2]
        pixel_jump  = np.hypot(msg.x - self.x[0], msg.y - self.x[1])
        cx_jump     = abs(msg.x - self.x[0])
        cy_jump     = abs(msg.y - self.x[1])

        # Rule 1: alpha collapse
        alpha_outlier = (prev_alpha > self.OUTLIER_PREV_ALPHA_PIX
                         and msg.z < self.OUTLIER_NEW_ALPHA_PIX)

        # Rule 2: position jump (only when not already in escape mode)
        pos_outlier = (pixel_jump > self.PIXEL_JUMP_OUTLIER
                       and self.consecutive_rejections < self.MAX_CONSECUTIVE_REJECTIONS)

        if alpha_outlier:
            return True, ("alpha-collapse prev=%.0f→%.0f px jump=%.0f"
                          % (prev_alpha, msg.z, pixel_jump))
        if pos_outlier:
            return True, ("pos-jump cx=%.0f cy=%.0f (%.0fpx total > %.0f limit)"
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

                # Alpha-collapse escape hatch (force-accept after N rejections)
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

        out = Point()
        out.x = self.x[0]
        out.y = self.x[1]
        out.z = self.x[2] if not is_dropout else -self.x[2]
        self.pub.publish(out)


if __name__ == '__main__':
    KalmanFilterNode()
    rospy.spin()