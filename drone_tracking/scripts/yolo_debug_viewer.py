#!/usr/bin/env python3
"""yolo_debug_viewer.py — live overlay of YOLO detections with REAL bbox dimensions

v1.1 (2026-06-10): stale-box handling — a box older than BOX_TIMEOUT s is no
longer drawn (and no longer counted as a detection), so dropouts are visible
instead of a frozen rectangle and the on-screen detection rate stays honest.
Launched automatically by launch_stack.sh (stage TV; VIEWER=0 to disable).
"""
import rospy, cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5   # s of sim time without a new box before the overlay clears


class Viewer:
    def __init__(self):
        rospy.init_node('yolo_debug_viewer')
        self.bridge = CvBridge()
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.det_count = 0
        self.total_frames = 0
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.loginfo("YOLO viewer started — press Q in window to quit")

    def box_cb(self, msg):
        self.last_box = msg
        self.last_box_time = rospy.Time.now()

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.total_frames += 1
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
        if fresh:
            cx, cy = int(self.last_box.x), int(self.last_box.y)
            w,  h  = int(self.last_box.z), int(self.last_box.w)
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"DRONE {w}x{h}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            self.det_count += 1
        else:
            cv2.putText(frame, "NO DETECTION", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        rate = 100.0 * self.det_count / max(self.total_frames, 1)
        cv2.putText(frame, f"Detection rate: {rate:.1f}% ({self.det_count}/{self.total_frames})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow('YOLO Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('user quit')

if __name__ == '__main__':
    Viewer()
    rospy.spin()
    cv2.destroyAllWindows()
