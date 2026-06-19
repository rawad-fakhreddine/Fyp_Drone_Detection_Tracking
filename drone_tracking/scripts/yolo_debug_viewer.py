#!/usr/bin/env python3
"""yolo_debug_viewer.py — live overlay of YOLO detections with REAL bbox dimensions

v1.3 (2026-06-17): GUI moved to the MAIN thread (FOV-window fix).
  ROOT CAUSE of "VIEWER=1 but no window appeared": v1.1/v1.2 called
  cv2.imshow()/cv2.waitKey() inside the ROS subscriber callback (img_cb),
  which rospy runs on a SEPARATE thread while the main thread sat in
  rospy.spin(). Under the OpenCV 4.13 Qt5 backend, GUI calls off the main
  thread render unreliably and emit "QObject::killTimer: Timers cannot be
  stopped from another thread" — with NO Python exception, so the TV log
  looked clean and the window silently never showed.
  Fix: the callbacks now only STORE the latest frame/box/status (thread-safe
  numpy work). All cv2 GUI calls run in a main-thread render loop (run()).
  The window is also created EAGERLY with a "waiting for camera" placeholder
  so it appears the instant the node starts, before the first frame.

v1.2 (2026-06-11): detection rate is now a ROLLING window over the last
ROLL_WINDOW (300) frames instead of cumulative since node start (a long clean
HOLD no longer hides a current dropout). Shows current box confidence and
w x h px (from /drone_tracking/detector_status, v3.3 node), and a [GATED]
tag while the detector gate is in LOST/CONFIRMING state — i.e. boxes YOLO
sees but the persistence gate is withholding from the controller.

v1.1 (2026-06-10): stale-box handling — a box older than BOX_TIMEOUT s is no
longer drawn (and no longer counted as a detection), so dropouts are visible
instead of a frozen rectangle and the on-screen detection rate stays honest.
Launched automatically by launch_stack.sh (stage TV; VIEWER=0 to disable;
default is now OFF so batch runs stay headless — VIEWER=1 opens the window).
"""
import rospy, cv2
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5   # s of sim time without a new box before the overlay clears
ROLL_WINDOW = 300   # frames in the rolling detection-rate window (v1.2)
WIN = 'YOLO Debug'


class Viewer:
    def __init__(self):
        rospy.init_node('yolo_debug_viewer')
        self.bridge = CvBridge()
        self.frame = None                       # latest cv2 image (set in img_cb)
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.roll = deque(maxlen=ROLL_WINDOW)   # v1.2: 1=detected, 0=not
        self.status = ""                        # v3.3 "STATE,conf,w,h"
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.loginfo("YOLO viewer v1.3 started (GUI on main thread) — press Q in window to quit")

    # ── callbacks: STORE only, no GUI calls (they run off the main thread) ──
    def box_cb(self, msg):
        self.last_box = msg
        self.last_box_time = rospy.Time.now()

    def status_cb(self, msg):
        self.status = msg.data

    def img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, "[viewer] imgmsg_to_cv2 failed: %r" % e)
            return
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
        self.roll.append(1 if fresh else 0)
        self.frame = img

    # ── render: runs on the MAIN thread ──
    def _draw(self, frame):
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)

        state, conf, sw, sh = "", "", "", ""
        if self.status:
            parts = self.status.split(',')
            if len(parts) == 4:
                state, conf, sw, sh = parts
        gated = state not in ("", "TRACKING")

        if fresh:
            cx, cy = int(self.last_box.x), int(self.last_box.y)
            w,  h  = int(self.last_box.z), int(self.last_box.w)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"DRONE {w}x{h}"
            if conf and conf != 'nan':
                label += f" conf={conf}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO DETECTION", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if gated:
            tag = f"[GATED] {state}"
            if conf and conf != 'nan':
                tag += f"  cand {sw}x{sh} conf={conf}"
            cv2.putText(frame, tag, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        rate = 100.0 * sum(self.roll) / max(len(self.roll), 1)
        cv2.putText(frame, f"Detection rate: {rate:.1f}% (last {len(self.roll)} frames)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def run(self):
        # Eager window so it appears immediately, before the first camera frame.
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        base = np.zeros((480, 640, 3), np.uint8)
        # WALL-CLOCK render loop. Deliberately NOT rospy.Rate(): /use_sim_time
        # is true under PX4 SITL and sim time FREEZES during the T5 lockstep
        # window (target instance attaching). A sim-time sleep would stall the
        # Qt event pump there -> window goes unresponsive and WSLg closes it.
        # cv2.waitKey(33) both pumps the GUI and paces the loop in wall time,
        # so the window stays alive and responsive regardless of sim time.
        while not rospy.is_shutdown():
            frame = self.frame
            if frame is not None:
                disp = self._draw(frame.copy())
            else:
                disp = base.copy()
                cv2.putText(disp, "Waiting for camera feed...", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WIN, disp)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        if not rospy.is_shutdown():
            rospy.signal_shutdown('user quit')


if __name__ == '__main__':
    Viewer().run()
