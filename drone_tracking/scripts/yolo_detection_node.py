#!/usr/bin/env python3
"""
yolo_detection_node.py — v3.4  (conf hysteresis + plausibility + persistence)
==============================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v3.4 changes from v3.3 (M10.1 — phantom-lock prep):
  Confidence hysteresis on the YOLO call. The single fixed conf (0.35) let
  weak background features (tree line) clear the bar and, given enough frames,
  satisfy the persistence gate → phantom lock. Now state-dependent:
    ~conf         (0.35) — conf_track,   used while TRACKING.
    ~conf_acquire (0.55) — used while LOST/CONFIRMING (initial + re-acquire).
  A high bar to (re)acquire rejects weak phantoms; the sensitive bar is only
  used once a real lock is established. Stacks with the persistence gate
  (acquisition now needs high-conf AND ~persist_frames persistent frames).
  Identical in raw and kalman configs (gates which detections exist, not the
  coordinates of accepted ones) — ablation integrity preserved.

v3.3 changes from v3.2 (M9.6 — T7 phantom-chase fix):
  During the T7 stress run (zone 7, seed 42) YOLO latched a false positive on
  the tree line ("DRONE 27x8") while the chaser was in SEARCH; IBVS chased the
  phantom, the phase left SEARCH (resetting the loss watchdog), and recovery
  failed. False positives corrupt SEARCH, the watchdog, and detection metrics.
  Two gates added between YOLO output and publishing — this is detection-stack
  logic, NOT smoothing: it gates WHICH detections exist; coordinates of
  accepted detections pass through untouched, so Config-1 (raw) and Config-2
  (kalman) see identical behavior and ablation integrity is preserved.

  A. PLAUSIBILITY FILTER (every frame, anchored to the target's true size —
     iris from target_iris_sitl/iris.sdf: rotor-tip-to-tip 0.52-0.77 m,
     body height 0.11 m; camera f=277 px (live camera_info, 2026-06-15 —
     was mis-stated as 307) → w_px(Z) = 277*W/Z):
       ~min_box_px (3)   : w below the px width of the narrow 0.52 m span at
                           Z_max≈60 m (277*0.52/60≈2.4) = clutter speck
       ~max_box_px (300) : w above the px width of the 0.77 m diagonal at
                           Z_min≈0.8 m (277*0.77/0.8≈267) = nonsense
                           (bounds [3,300] px still valid under f=277 — both
                            limits bracket the re-derived range with margin)
       ~aspect_min/~aspect_max ([0.8, 6.0]) : w/h band (iris side-on is
                           wide+flat ~3; head-on ~1.5)
     Rejected boxes are never published; per-100-frame rejection counts are
     logged with the fps line. Candidates are tried in confidence order, so a
     plausible 2nd-best box can still win over an implausible best box.

  B. ACQUISITION PERSISTENCE (state machine):
       TRACKING   — plausibility-filtered detections pass through as before.
                    No accepted detection for > ~lost_after (1.0 s) → LOST.
       LOST       — a candidate must appear in >= ~persist_frames (3)
                    consecutive frames with centroid jump < ~persist_max_jump
                    (80 px) between frames before publishing resumes
                    (single-frame phantom hits can no longer capture the
                    controller). While unconfirmed, the no-detection NaN
                    point is published (same signal as a YOLO miss).
     Node starts in LOST, so the FIRST acquisition also needs persistence.

  Publishes /drone_tracking/detector_status (String, every frame):
  "STATE,conf,w,h" — TRACKING / LOST / CONFIRMING k/N — for the debug
  viewer's [GATED] tag. Downstream control topics are unchanged.

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
import math
import torch
from collections import deque
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode:
    # Rolling median window for bbox area only
    ALPHA_MEDIAN_WINDOW = 5

    def __init__(self):
        rospy.init_node('yolo_detection_node')

        self.bridge = CvBridge()

        # v3.2: model file + ~device rosparam (default 'cuda', auto-fallback cpu)
        self._model_path = '/home/rawad/drone_detection/models/best_v41s.pt'
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
        # v3.4: confidence hysteresis — high bar to (re)acquire, sensitive to track
        self.conf_track   = float(rospy.get_param('~conf', 0.35))         # while TRACKING
        self.conf_acquire = float(rospy.get_param('~conf_acquire', 0.55)) # while LOST/CONFIRMING

        # v3.3 A: plausibility bounds (px) — anchored to iris true size, f=277
        self.min_box_px = float(rospy.get_param('~min_box_px', 3.0))
        self.max_box_px = float(rospy.get_param('~max_box_px', 300.0))
        self.aspect_min = float(rospy.get_param('~aspect_min', 0.8))
        self.aspect_max = float(rospy.get_param('~aspect_max', 6.0))

        # v3.3 B: acquisition persistence
        self.persist_frames   = int(rospy.get_param('~persist_frames', 3))
        self.persist_max_jump = float(rospy.get_param('~persist_max_jump', 80.0))
        self.lost_after       = float(rospy.get_param('~lost_after', 1.0))
        self._tracking        = False          # start LOST: first acquisition
        self._last_accept_t   = None           #   also needs persistence
        self._streak          = 0
        self._streak_cx       = None
        self._streak_cy       = None

        # v3.3: per-window rejection counters (reported with the fps line)
        self._rej_size = self._rej_aspect = self._rej_persist = 0

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
        self.status_pub = rospy.Publisher(
            '/drone_tracking/detector_status', String, queue_size=1)
        rospy.Subscriber(
            '/iris/usb_cam/image_raw', Image, self.callback, queue_size=1)

        rospy.loginfo("[YOLO] v3.4 | model=%s | device=%s | conf track/acquire=%.2f/%.2f"
                      " | class=%d | alpha_median=%d frames"
                      % (self._model_path, self.device,
                         self.conf_track, self.conf_acquire,
                         self.target_class, self.ALPHA_MEDIAN_WINDOW))
        rospy.loginfo("[YOLO] v3.4 gate | box w in [%.0f, %.0f] px | w/h in "
                      "[%.1f, %.1f] | persist %d frames < %.0f px jump | "
                      "lost after %.1f s"
                      % (self.min_box_px, self.max_box_px,
                         self.aspect_min, self.aspect_max,
                         self.persist_frames, self.persist_max_jump,
                         self.lost_after))

    # ── v3.3 A: plausibility filter ──────────────────────────────────────
    def _plausible(self, w, h):
        """True if the box could be the iris target; count rejects by reason."""
        if w < self.min_box_px or w > self.max_box_px:
            self._rej_size += 1
            return False
        if h <= 0 or not (self.aspect_min <= w / h <= self.aspect_max):
            self._rej_aspect += 1
            return False
        return True

    # ── v3.3 B: persistence gate. Returns True when publishing is allowed. ──
    def _persistence_ok(self, cx, cy, now):
        if self._tracking:
            return True
        # LOST/CONFIRMING: candidate must persist across consecutive frames
        if (self._streak_cx is not None and
                math.hypot(cx - self._streak_cx, cy - self._streak_cy)
                >= self.persist_max_jump):
            self._streak = 0                    # jumped — restart with this box
        self._streak += 1
        self._streak_cx, self._streak_cy = cx, cy
        if self._streak >= self.persist_frames:
            self._tracking = True
            self._streak = 0
            self._streak_cx = self._streak_cy = None
            rospy.loginfo("[YOLO] v3.3 gate: acquisition CONFIRMED "
                          "(%d consecutive frames) — TRACKING" % self.persist_frames)
            return True
        self._rej_persist += 1
        return False

    def _no_accept(self, now):
        """Bookkeeping for a frame with no accepted detection."""
        if (self._tracking and self._last_accept_t is not None and
                now - self._last_accept_t > self.lost_after):
            self._tracking = False
            self._streak = 0
            self._streak_cx = self._streak_cy = None
            rospy.logwarn("[YOLO] v3.3 gate: no accepted detection for %.1f s "
                          "— LOST (re-acquisition needs persistence)"
                          % self.lost_after)
        if not self._tracking and self._streak > 0:
            self._streak = 0                    # streak broken by a miss
            self._streak_cx = self._streak_cy = None

    def _status_str(self, conf, w, h):
        if self._tracking:
            state = "TRACKING"
        elif self._streak > 0:
            state = "CONFIRMING %d/%d" % (self._streak, self.persist_frames)
        else:
            state = "LOST"
        return "%s,%.2f,%.0f,%.0f" % (state, conf, w, h)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now = rospy.Time.now().to_sec()

        results = self.model(
            frame,
            conf=(self.conf_track if self._tracking else self.conf_acquire),
            classes=[self.target_class],
            device=self.device,
            verbose=False)

        p = Point()
        boxes = results[0].boxes

        # v3.3: best PLAUSIBLE box, tried in confidence order (was: best box)
        cand = None
        if len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy()
            for idx in np.argsort(-confidences):
                x1, y1, x2, y2 = boxes[int(idx)].xyxy[0].cpu().numpy()
                w, h = float(x2 - x1), float(y2 - y1)
                if self._plausible(w, h):
                    cand = (float((x1 + x2) / 2), float((y1 + y2) / 2),
                            w, h, float(confidences[int(idx)]))
                    break

        published = False
        last_conf, last_w, last_h = float('nan'), 0.0, 0.0
        if cand is not None:
            raw_cx, raw_cy, w, h, conf = cand
            last_conf, last_w, last_h = conf, w, h
            if self._persistence_ok(raw_cx, raw_cy, now):
                raw_alpha = w * h
                # v3.1: 5-frame rolling median on bbox area only
                self._alpha_buf.append(raw_alpha)
                smooth_alpha = float(np.median(self._alpha_buf))

                p.x = raw_cx        # centroid X — unsmoothed, Kalman handles it
                p.y = raw_cy        # centroid Y — unsmoothed, Kalman handles it
                p.z = smooth_alpha  # bbox area  — median-smoothed here
                # Publish actual w,h for debug viewer (other nodes unaffected)
                self.box_pub.publish(Quaternion(x=raw_cx, y=raw_cy, z=w, w=h))
                self._last_accept_t = now
                published = True

        if not published:
            if cand is None:
                self._no_accept(now)
            # No accepted detection — clear buffer so old values don't
            # contaminate re-acquisition (target reappears at different
            # distance). Gated/unconfirmed frames publish the same NaN
            # no-detection signal as a YOLO miss (downstream unchanged).
            self._alpha_buf.clear()
            p.x = float('nan')
            p.y = float('nan')
            p.z = float('nan')

        self.pub.publish(p)
        self.status_pub.publish(String(data=self._status_str(
            last_conf, last_w, last_h)))

        # v3.2: rolling-average fps every _fps_window processed frames
        self._frame_count += 1
        if self._frame_count % self._fps_window == 0:
            dt  = time.time() - self._fps_t0
            fps = (self._fps_window / dt) if dt > 0 else 0.0
            rospy.loginfo("[YOLO] v3.3 | device=%s | rolling fps=%.1f | "
                          "rejects/100f: size=%d aspect=%d persist=%d"
                          % (self.device, fps, self._rej_size,
                             self._rej_aspect, self._rej_persist))
            self._rej_size = self._rej_aspect = self._rej_persist = 0
            self._fps_t0 = time.time()


if __name__ == '__main__':
    YoloNode()
    rospy.spin()
