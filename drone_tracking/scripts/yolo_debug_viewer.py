#!/usr/bin/env python3
"""yolo_debug_viewer.py — live overlay of YOLO detections with REAL bbox dimensions"""
import rospy, cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge

class Viewer:
    def __init__(self):
        rospy.init_node('yolo_debug_viewer')
        self.bridge = CvBridge()
        self.last_box = None
        self.det_count = 0
        self.total_frames = 0
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.loginfo("YOLO viewer started — press Q in window to quit")

    def box_cb(self, msg):
        self.last_box = msg

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.total_frames += 1
        if self.last_box:
            cx, cy = int(self.last_box.x), int(self.last_box.y)
            w,  h  = int(self.last_box.z), int(self.last_box.w)
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"DRONE {w}x{h}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            self.det_count += 1
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