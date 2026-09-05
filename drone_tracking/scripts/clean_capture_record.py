#!/usr/bin/env python3
"""clean_capture_record.py — M10.3 Part B: CLEAN-frame capture for the imgsz
==============================================================================
recovery test. Unlike gt_overlay_record.py, this writes the *un-annotated*
camera frame to PNG (no GT marker burned on the target), so offline YOLO
re-inference at higher imgsz sees pristine pixels. Per saved frame it dumps a
metadata row that pairs the clean PNG with the live detector outcome + the
ground-truth projection, so we can isolate IN-FRAME RAW-MISS frames offline.

Saves every frame whose GT projects INSIDE the FOV (the only frames where an
in-frame miss is even possible). HOLD/APPROACH/SEARCH all included.

Reuses GTProjector from gt_projection.py (no edits). Intrinsics from the live
camera_info (fx=277.19); SDF-hfov fallback only after a grace window, with a
loud warning (the fx=457 fallback was the bug caught on the first T3 run).

Outputs:
  <dir>/frames/<t>.png            clean BGR frame, native 640x480
  <dir>/clean_capture_meta.csv    one row per saved frame:
    t, png, yolo_detected, det_state, det_w, gt_u, gt_v, gt_in_frame,
    separation, depth, est_target_px

  yolo_detected = 1 iff target_center is a fresh non-NaN point (controller saw
                  a detection); det_state from /detector_status distinguishes
                  raw-miss (LOST) from gate-withheld (CONFIRMING).
  est_target_px = fx * 0.5 / depth   (apparent iris width at that range)

Usage:
  rosrun drone_tracking clean_capture_record.py --dir ~/fyp/Results/diagnostics/cap_T7
"""
import argparse, csv, math, os, sys
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_projection import GTProjector, intrinsics_from_hfov

INTR_GRACE_FRAMES = 90   # wait this many frames for camera_info before SDF fallback
DET_FRESH_S = 0.30       # target_center older than this => treat as no detection


class CleanCapture(object):
    def __init__(self, outdir, target='target_iris'):
        self.bridge = CvBridge()
        self.outdir = outdir
        self.framedir = os.path.join(outdir, 'frames')
        os.makedirs(self.framedir, exist_ok=True)
        self.csv_path = os.path.join(outdir, 'clean_capture_meta.csv')
        self.csv = open(self.csv_path, 'w', newline='')
        self.w = csv.writer(self.csv)
        self.w.writerow(['t', 'png', 'yolo_detected', 'det_state', 'det_w',
                         'gt_u', 'gt_v', 'gt_in_frame', 'separation',
                         'depth', 'est_target_px'])
        self.target = target
        self.proj = None
        self.intr_src = 'waiting'
        self.ms = None
        self.chaser_name = None
        self.det_pt = None
        self.det_stamp = 0.0
        self.det_status = 'LOST,0,0,0'
        self.n = 0
        self.saved = 0

        rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo,
                         self.cam_info_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self.ms_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_center', Point,
                         self.det_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String,
                         self.status_cb, queue_size=1)
        rospy.Subscriber('/iris/usb_cam/image_raw', Image,
                         self.img_cb, queue_size=1)
        rospy.loginfo('[clean_cap] capturing IN-FOV frames -> %s', self.framedir)

    def cam_info_cb(self, msg):
        if self.intr_src != 'camera_info' and msg.K and msg.K[0] > 0:
            self.proj = GTProjector(msg.K[0], msg.K[4], msg.K[2], msg.K[5],
                                    msg.width, msg.height)
            self.intr_src = 'camera_info'
            rospy.loginfo('[clean_cap] intrinsics from camera_info: fx=%.2f '
                          '%dx%d', msg.K[0], msg.width, msg.height)

    def ms_cb(self, msg):
        self.ms = msg
        if self.chaser_name is None:
            for nm in msg.name:
                if 'iris' in nm and nm != self.target and not nm.startswith('target'):
                    self.chaser_name = nm
                    break

    def det_cb(self, msg):
        self.det_pt = msg
        self.det_stamp = rospy.Time.now().to_sec()

    def status_cb(self, msg):
        self.det_status = msg.data

    def _ensure_proj(self):
        if self.proj is None and self.n >= INTR_GRACE_FRAMES:
            fx, fy, cx, cy = intrinsics_from_hfov()
            self.proj = GTProjector(fx, fy, cx, cy)
            self.intr_src = 'sdf_hfov'
            rospy.logwarn('[clean_cap] !! camera_info never arrived; SDF-hfov '
                          'fallback fx=%.1f (vertical/edge calls approximate)', fx)

    def _poses(self):
        if self.ms is None or self.chaser_name is None:
            return None
        try:
            ci = self.ms.name.index(self.chaser_name)
            ti = self.ms.name.index(self.target)
        except ValueError:
            return None
        cp = self.ms.pose[ci].position
        cq = self.ms.pose[ci].orientation
        tp = self.ms.pose[ti].position
        return ((cp.x, cp.y, cp.z), (cq.x, cq.y, cq.z, cq.w), (tp.x, tp.y, tp.z))

    def img_cb(self, msg):
        self.n += 1
        self._ensure_proj()
        if self.proj is None:
            return
        poses = self._poses()
        if poses is None:
            return
        gt = self.proj.compute(*poses)
        if not gt['in_fov']:
            return  # only in-FOV frames can have an in-frame miss

        now = rospy.Time.now().to_sec()
        fresh = (self.det_pt is not None and (now - self.det_stamp) < DET_FRESH_S)
        yolo_det = int(fresh and not math.isnan(self.det_pt.x))
        st = self.det_status.split(',')
        det_state = st[0].split()[0] if st and st[0] else 'LOST'
        det_w = float(st[2]) if len(st) > 2 else 0.0

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        t = msg.header.stamp.to_sec() or now
        png = os.path.join(self.framedir, '%010.3f.png' % t)
        cv2.imwrite(png, frame)
        self.saved += 1
        est_px = (self.proj.fx * 0.5 / gt['depth']) if gt['depth'] > 0 else float('nan')
        self.w.writerow(['%.3f' % t, os.path.basename(png), yolo_det, det_state,
                         '%.0f' % det_w, '%.1f' % gt['u'], '%.1f' % gt['v'], 1,
                         '%.2f' % gt['true_dist'], '%.2f' % gt['depth'],
                         '%.1f' % est_px])
        if self.saved % 100 == 0:
            self.csv.flush()
            rospy.loginfo('[clean_cap] saved=%d frames (intr=%s)',
                          self.saved, self.intr_src)

    def close(self):
        try:
            self.csv.flush(); self.csv.close()
        except Exception:
            pass
        rospy.loginfo('[clean_cap] DONE saved=%d in-FOV frames -> %s',
                      self.saved, self.csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--target', default='target_iris')
    args, _ = ap.parse_known_args()
    rospy.init_node('clean_capture_record', anonymous=True)
    cap = CleanCapture(os.path.expanduser(args.dir), args.target)
    rospy.on_shutdown(cap.close)
    rospy.spin()


if __name__ == '__main__':
    main()
