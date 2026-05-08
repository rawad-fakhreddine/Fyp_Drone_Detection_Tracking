#!/usr/bin/env python3
"""
yolo_detection_node.py — v3 deployment
  - Loads best.pt (currently YOLO v3)
  - Selects highest-confidence box (not boxes[0])
  - Class filter: only 'drone' class (index 0)
  - Confidence threshold 0.35 (raised from 0.20 for v3:
    v3 has Precision=1.000 and Recall=0.992 across all scenarios,
    so weak detections are more likely false positives than missed drones)
  - verbose=False (clean logs)
  - NaN publishing on no-detection (IBVS v6.3 stale-detection hover relies on this)
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node')

        self.bridge = CvBridge()
        self.model = YOLO('/home/rawad/drone_detection/models/best.pt')

        # Unified dataset classes: ["drone", "bird", "aircraft"]
        # We only want drone (class index 0)
        self.target_class = 0

        # v3 raised threshold from 0.20 → 0.35
        # v2 needed 0.20 because of hard-case uncertainty (39% miss rate on debug frames)
        # v3 is precise enough that weak detections are usually false positives
        self.conf_threshold = 0.35

        self.pub = rospy.Publisher(
            '/drone_tracking/target_center', Point, queue_size=1)
        rospy.Subscriber(
            '/iris/usb_cam/image_raw', Image, self.callback, queue_size=1)

        rospy.loginfo("[YOLO] Detection node started (conf=%.2f, class=%d)"
                      % (self.conf_threshold, self.target_class))

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run inference with conf threshold and class filter
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=[self.target_class],
            verbose=False)

        p = Point()
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Select HIGHEST-CONFIDENCE box (not boxes[0])
            confidences = boxes.conf.cpu().numpy()
            best_idx = int(confidences.argmax())
            box = boxes[best_idx]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            p.x = float((x1 + x2) / 2)
            p.y = float((y1 + y2) / 2)
            p.z = float((x2 - x1) * (y2 - y1))
        else:
            p.x = float('nan')
            p.y = float('nan')
            p.z = float('nan')

        self.pub.publish(p)


if __name__ == '__main__':
    YoloNode()
    rospy.spin()