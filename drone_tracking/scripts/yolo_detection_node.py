#!/usr/bin/env python3
"""
yolo_detection_node.py — v3.2  (YOLOv8s deploy + device/conf params + fps log)
==============================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v3.2 changes from v3.1 (M9.6 step 2):
  Deployed YOLOv8s (best.pt = best_v4s.pt, ~11.1M params; v4n kept as
  best_v4n.pt rollback). Added ~device rosparam (default 'cuda', auto-fallback
  to 'cpu' with a logwarn if CUDA is unavailable) passed explicitly to the
  model call, and ~conf rosparam (default 0.35 — unchanged behaviour). Logs the
  resolved device + model file at startup and a rolling-average fps every 100
  processed frames. Detection logic itself is unchanged from v3.1.

v3.1 changes from v3:
  Added ALPHA_MEDIAN_WINDOW = 5 rolling median on bbox area (p.z).
  Root cause fixed: YOLO outputs bbox areas like 1200→990→1179 for a drone
  at constant speed because Gazebo's mesh rendering varies frame-to-frame
  (aliasing, light angle) causing NMS to shift the tight bbox by ±5-15px.
  A 5-frame median rejects transient spikes without adding lag, since:
    - Constant speed → area changes slowly → median = true value
    - Single spike (1200,1000,990,1010,980) → median = 1000 (correct)
    - Actual acceleration → monotonic area change → median tracks correctly
  cx/cy are NOT smoothed here — Kalman filter handles those.
  This runs at ~19fps; 5 frames = 0.26s window, imperceptible lag for control.
"""

import rospy
import time
import torch
from collections import deque
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode:
    # Rolling median window for bbox area only
    ALPHA_MEDIAN_WINDOW = 5

    def __init__(self):
        rospy.init_node('yolo_detection_node')

        self.bridge = CvBridge()

        # v3.2: model file + ~device rosparam (default 'cuda', auto-fallback cpu)
        self._model_path = '/home/rawad/drone_detection/models/best.pt'
        requested_dev    = str(rospy.get_param('~device', 'cuda'))
        if requested_dev.startswith('cuda') and not torch.cuda.is_available():
            rospy.logwarn("[YOLO] ~device='%s' requested but CUDA is "
                          "unavailable — falling back to CPU." % requested_dev)
            self.device = 'cpu'
        else:
            self.device = requested_dev
        self.model = YOLO(self._model_path)

        # Class filter: drone only (index 0 in unified dataset)
        self.target_class   = 0
        # v3.2: ~conf rosparam (default 0.35 — same value as the v3 constant)
        self.conf_threshold = float(rospy.get_param('~conf', 0.35))

        # Rolling median buffer for bbox area (alpha = w*h in pixels)
        self._alpha_buf = deque(maxlen=self.ALPHA_MEDIAN_WINDOW)

        # v3.2: rolling-average fps reporting (logged every _fps_window frames)
        self._fps_window  = 100
        self._frame_count = 0
        self._fps_t0      = time.time()

        self.pub = rospy.Publisher(
            '/drone_tracking/target_center', Point, queue_size=1)
        self.box_pub = rospy.Publisher(
            '/drone_tracking/target_box', Quaternion, queue_size=1)
        rospy.Subscriber(
            '/iris/usb_cam/image_raw', Image, self.callback, queue_size=1)

        rospy.loginfo("[YOLO] v3.2 | model=%s | device=%s | conf=%.2f | "
                      "class=%d | alpha_median=%d frames"
                      % (self._model_path, self.device, self.conf_threshold,
                         self.target_class, self.ALPHA_MEDIAN_WINDOW))

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=[self.target_class],
            device=self.device,
            verbose=False)

        p = Point()
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Select HIGHEST-CONFIDENCE box (v3 change — not boxes[0])
            confidences = boxes.conf.cpu().numpy()
            best_idx    = int(confidences.argmax())
            box         = boxes[best_idx]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Raw centroid (not smoothed — Kalman handles this)
            raw_cx    = float((x1 + x2) / 2)
            raw_cy    = float((y1 + y2) / 2)
            raw_alpha = float((x2 - x1) * (y2 - y1))

            # v3.1: 5-frame rolling median on bbox area only
            self._alpha_buf.append(raw_alpha)
            smooth_alpha = float(np.median(self._alpha_buf))

            p.x = raw_cx        # centroid X — unsmoothed, Kalman handles it
            p.y = raw_cy        # centroid Y — unsmoothed, Kalman handles it
            p.z = smooth_alpha  # bbox area  — median-smoothed here
            # Publish actual w,h for debug viewer (other nodes unaffected)
            self.box_pub.publish(Quaternion(x=raw_cx, y=raw_cy,
                z=float(x2-x1), w=float(y2-y1)))

        else:
            # No detection — clear buffer so old values don't contaminate
            # re-acquisition (target reappears at different distance)
            self._alpha_buf.clear()
            p.x = float('nan')
            p.y = float('nan')
            p.z = float('nan')

        self.pub.publish(p)

        # v3.2: rolling-average fps every _fps_window processed frames
        self._frame_count += 1
        if self._frame_count % self._fps_window == 0:
            now = time.time()
            dt  = now - self._fps_t0
            fps = (self._fps_window / dt) if dt > 0 else 0.0
            rospy.loginfo("[YOLO] v3.2 | device=%s | rolling fps=%.1f "
                          "(last %d frames)"
                          % (self.device, fps, self._fps_window))
            self._fps_t0 = now


if __name__ == '__main__':
    YoloNode()
    rospy.spin()