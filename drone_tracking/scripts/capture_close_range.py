#!/usr/bin/env python3
"""
capture_close_range.py — v2 (init-order fix)
"""

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge


class CloseRangeCapture:
    SAVE_DIR         = os.path.expanduser("~/drone_detection/capture")
    DEBUG_DIR        = os.path.expanduser("~/drone_detection/capture_debug")
    SAVE_INTERVAL    = 0.2
    TRAIN_PHASES     = ("APPROACH", "HOLD")
    DEBUG_PHASES     = ("TAKEOFF", "SEARCH")
    DARK_FRAME_LIMIT = 10

    def __init__(self):
        rospy.init_node('close_range_capture')

        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.DEBUG_DIR, exist_ok=True)

        existing_train = [f for f in os.listdir(self.SAVE_DIR)
                          if f.startswith("close_range_") and f.endswith(".jpg")]
        self.train_idx = len(existing_train)

        existing_debug = [f for f in os.listdir(self.DEBUG_DIR)
                          if f.startswith("debug_") and f.endswith(".jpg")]
        self.debug_idx = len(existing_debug)

        self.bridge = CvBridge()
        self.last_save_time = rospy.Time(0)
        self.current_phase = "UNKNOWN"

        self.consecutive_dark_frames = 0
        self.is_dark = False

        # ── Stats MUST be initialized before subscribers ──
        self.n_train_saved   = 0
        self.n_saved_dark    = 0
        self.n_skip_phase    = 0
        self.n_skip_longdark = 0

        rospy.loginfo("[Capture] TRAIN dir has %d existing files, new start at %d"
                      % (len(existing_train), self.train_idx))
        rospy.loginfo("[Capture] DEBUG dir has %d existing files, new start at %d"
                      % (len(existing_debug), self.debug_idx))
        rospy.loginfo("[Capture] TRAIN → %s" % self.SAVE_DIR)
        rospy.loginfo("[Capture] DEBUG → %s" % self.DEBUG_DIR)
        rospy.loginfo("[Capture] Rate: %.1f Hz, dark limit: %d frames (no frame cap)"
                      % (1.0 / self.SAVE_INTERVAL, self.DARK_FRAME_LIMIT))

        rospy.Subscriber('/iris/usb_cam/image_raw', Image,
                         self.image_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase', String,
                         self.phase_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/filtered_target', Point,
                         self.detection_cb, queue_size=1)

    def phase_cb(self, msg):
        self.current_phase = msg.data

    def detection_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.consecutive_dark_frames += 1
        elif msg.z > 0:
            if self.is_dark:
                rospy.loginfo("[Capture] Detection recovered after %d dark frames"
                              % self.consecutive_dark_frames)
            self.consecutive_dark_frames = 0
            self.is_dark = False
        else:
            self.consecutive_dark_frames += 1

        if self.consecutive_dark_frames >= self.DARK_FRAME_LIMIT and not self.is_dark:
            self.is_dark = True
            rospy.logwarn("[Capture] Entering DARK mode — %d consecutive no-det frames"
                          % self.consecutive_dark_frames)

    def image_cb(self, msg):
        is_train_phase = self.current_phase in self.TRAIN_PHASES
        is_debug_phase = self.current_phase in self.DEBUG_PHASES

        if not (is_train_phase or is_debug_phase):
            self.n_skip_phase += 1
            return

        now = rospy.Time.now()
        if (now - self.last_save_time).to_sec() < self.SAVE_INTERVAL:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if is_train_phase:
            if self.is_dark:
                self.n_skip_longdark += 1
                return
            filename = os.path.join(
                self.SAVE_DIR, "close_range_%04d.jpg" % self.train_idx)
            cv2.imwrite(filename, frame)
            if self.consecutive_dark_frames > 0:
                self.n_saved_dark += 1
            self.train_idx += 1
            self.n_train_saved += 1
        else:
            filename = os.path.join(
                self.DEBUG_DIR, "debug_%04d.jpg" % self.debug_idx)
            cv2.imwrite(filename, frame)
            self.debug_idx += 1

        self.last_save_time = now

        rospy.loginfo_throttle(2,
            "[Capture] train=%d (dropout=%d) debug=%d | phase=%s | skipped: other=%d dark=%d"
            % (self.n_train_saved, self.n_saved_dark, self.debug_idx,
               self.current_phase, self.n_skip_phase, self.n_skip_longdark))


if __name__ == '__main__':
    try:
        CloseRangeCapture()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass