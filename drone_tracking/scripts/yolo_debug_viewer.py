#!/usr/bin/env python3
"""yolo_debug_viewer.py — live overlay of YOLO detections with REAL bbox dimensions

v1.2 (2026-06-11): detection rate is now a ROLLING window over the last
ROLL_WINDOW (300) frames instead of cumulative since node start (a long clean
HOLD no longer hides a current dropout). Shows current box confidence and
w x h px (from /drone_tracking/detector_status, v3.3 node), and a [GATED]
tag while the detector gate is in LOST/CONFIRMING state — i.e. boxes YOLO
sees but the persistence gate is withholding from the controller.

v1.1 (2026-06-10): stale-box handling — a box older than BOX_TIMEOUT s is no
longer drawn (and no longer counted as a detection), so dropouts are visible
instead of a frozen rectangle and the on-screen detection rate stays honest.
Launched automatically by launch_stack.sh (stage TV; VIEWER=0 to disable).
"""
import rospy, cv2
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5   # s of sim time without a new box before the overlay clears
ROLL_WINDOW = 300   # frames in the rolling detection-rate window (v1.2)


class Viewer:
    def __init__(self):
        rospy.init_node('yolo_debug_viewer')
        self.bridge = CvBridge()
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.roll = deque(maxlen=ROLL_WINDOW)   # v1.2: 1=detected, 0=not
        self.status = ""                        # v3.3 "STATE,conf,w,h"
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.loginfo("YOLO viewer v1.2 started — press Q in window to quit")

    def box_cb(self, msg):
        self.last_box = msg
        self.last_box_time = rospy.Time.now()

    def status_cb(self, msg):
        self.status = msg.data

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
        self.roll.append(1 if fresh else 0)

        # v1.2: detector gate state ("STATE,conf,w,h" from node v3.3)
        state, conf, sw, sh = "", "", "", ""
        if self.status:
            parts = self.status.split(',')
            if len(parts) == 4:
                state, conf, sw, sh = parts
        gated = state not in ("", "TRACKING")

        if fresh:
            cx, cy = int(self.last_box.x), int(self.last_box.y)
            w,  h  = int(self.last_box.z), int(self.last_box.w)
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"DRONE {w}x{h}"
            if conf and conf != 'nan':
                label += f" conf={conf}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            cv2.putText(frame, "NO DETECTION", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if gated:
            # YOLO may be seeing a candidate (conf/w/h shown) but the v3.3
            # gate is withholding it from the controller
            tag = f"[GATED] {state}"
            if conf and conf != 'nan':
                tag += f"  cand {sw}x{sh} conf={conf}"
            cv2.putText(frame, tag, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        rate = 100.0 * sum(self.roll) / max(len(self.roll), 1)
        cv2.putText(frame, f"Detection rate: {rate:.1f}% (last {len(self.roll)} frames)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow('YOLO Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('user quit')

if __name__ == '__main__':
    Viewer()
    rospy.spin()
    cv2.destroyAllWindows()
