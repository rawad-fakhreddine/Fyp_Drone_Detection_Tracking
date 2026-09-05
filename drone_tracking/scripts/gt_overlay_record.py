#!/usr/bin/env python3
"""gt_overlay_record.py — HEADLESS GT-overlay recorder (mp4 + per-frame CSV).
==============================================================================
M10.3 visual FOV-loss diagnostic, headless variant of gt_overlay_viewer.
DISPLAY/ANALYSIS ONLY — subscribes; never publishes to or modifies the flight
pipeline (kalman / ibvs / detection). Untracked catkin tool.

Identical overlay geometry to gt_overlay_viewer (same gt_projection.GTProjector,
same green YOLO box / magenta GT cross / yellow edge-arrow + axis label / HUD),
but instead of imshow it:
  1. writes every annotated frame to an mp4 (--record), so Rawad can scrub the
     exact loss moment, and
  2. dumps one CSV row per camera frame (--csv) with the overlay-derived data
     the offline loss-onset analyzer consumes.

All rendering and writing happens in the image callback (a single ROS
subscriber thread → serialized); the other callbacks store-only. So no GUI, no
main-thread requirement, runs head-less.

CSV columns:
  t, yolo_detected, gt_u, gt_v, gt_in_frame, exit_edge, axis,
  du, dv, separation, depth, bbox_area
    t .............. sim clock (use_sim_time)
    yolo_detected .. 1 if a fresh REAL YOLO detection (target_center non-nan,
                     age < BOX_TIMEOUT), else 0
    gt_u, gt_v ..... ground-truth target projected pixel (nan if behind camera)
    gt_in_frame .... 1 if 0<=u<w and 0<=v<h and in front, else 0
    exit_edge ...... "" in-frame; else TOP/BOTTOM/LEFT/RIGHT combo, or BEHIND
    axis ........... ey(vert) | ex(horiz) | "" — which axis carries the exit
    du, dv ......... gt offset from frame centre (px); nan if behind
    separation ..... true 3-D chaser→target distance (m)
    depth .......... camera-forward distance Xb (m)
    bbox_area ...... last fresh YOLO box w*h (px^2); nan if no fresh box

Usage (launched by occlusion_record.sh alongside the stack):
  rosrun drone_tracking gt_overlay_record.py --record /path/feed.mp4 --csv /path/frames.csv
"""
import os, sys, math, argparse
import rospy, cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Quaternion, Point
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_projection import GTProjector, intrinsics_from_hfov, IMG_W, IMG_H

BOX_TIMEOUT = 0.5      # s — a detection older than this counts as "lost"
ROLL_WINDOW = 300
VIDEO_FPS   = 20.0
# Grace before falling back to SDF intrinsics: camera_info (fx=277.19, the value
# the canonical gt CSVs used) normally lands within ~1 frame of the first image,
# but the very first image can beat it. Wait this many frames for the real
# intrinsics before resorting to the SDF 70-deg fallback (fx=457), which would
# scale the GT projection wrong (~1.65x) and falsely inflate edge-departure.
INTR_GRACE_FRAMES = 90


class GTRecorder:
    def __init__(self, record, csv, target='target_iris'):
        rospy.init_node('gt_overlay_record', anonymous=True)
        self.bridge = CvBridge()
        self.target_name = target
        self.record_path = record
        self.writer = None
        self.proj = None
        self.intr_src = 'waiting'
        self.ms = None
        self.chaser_name = None
        # YOLO detection state (from target_center: REAL/NONE) + box geom
        self.det_real = False
        self.det_time = rospy.Time(0)
        self.box_wh = None
        self.box_time = rospy.Time(0)
        self.cx = self.cy = float('nan')
        self.status = ""
        self.roll = []
        self.n = 0
        os.makedirs(os.path.dirname(os.path.abspath(csv)), exist_ok=True)
        self.f = open(csv, 'w', buffering=1)
        self.f.write('t,yolo_detected,gt_u,gt_v,gt_in_frame,exit_edge,axis,'
                     'du,dv,separation,depth,bbox_area\n')
        rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, self.cam_info_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.ms_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_center', Point, self.center_cb, queue_size=2)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        # image LAST so the others are wired before frames flow
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("[gt_record] headless recorder up -> mp4=%s csv=%s", record, csv)

    # ── store-only callbacks ──
    def cam_info_cb(self, msg):
        # install / upgrade to the real intrinsics (overrides any SDF fallback)
        if self.intr_src != 'camera_info' and msg.K and msg.K[0] > 0:
            self.proj = GTProjector(msg.K[0], msg.K[4], msg.K[2], msg.K[5],
                                    msg.width, msg.height)
            self.intr_src = 'camera_info'
            rospy.loginfo("[gt_record] intrinsics from camera_info fx=%.2f %dx%d",
                          msg.K[0], msg.width, msg.height)

    def ms_cb(self, msg):
        self.ms = msg
        if self.chaser_name is None:
            for nm in msg.name:
                if 'iris' in nm and nm != self.target_name and not nm.startswith('target'):
                    self.chaser_name = nm
                    rospy.loginfo("[gt_record] chaser='%s' target='%s'", nm, self.target_name)
                    break

    def center_cb(self, msg):
        real = not (math.isnan(msg.x) or math.isnan(msg.z))
        self.det_real = real
        self.det_time = rospy.Time.now()
        if real:
            self.cx, self.cy = msg.x, msg.y

    def box_cb(self, msg):
        self.box_wh = (msg.z, msg.w)     # z=w, w=h
        self.box_time = rospy.Time.now()

    def status_cb(self, msg):
        self.status = msg.data

    # ── ground-truth projection of the current target ──
    def _gt(self):
        if self.proj is None:
            # wait for the real camera_info; only fall back after the grace window
            if self.n < INTR_GRACE_FRAMES:
                return None
            fx, fy, cx, cy = intrinsics_from_hfov()
            self.proj = GTProjector(fx, fy, cx, cy)
            self.intr_src = 'sdf_hfov'
            rospy.logwarn("[gt_record] no camera_info after %d frames — SDF fallback fx=%.2f",
                          self.n, fx)
        if self.ms is None or self.chaser_name is None:
            return None
        try:
            ci = self.ms.name.index(self.chaser_name)
            ti = self.ms.name.index(self.target_name)
        except ValueError:
            return None
        cp = self.ms.pose[ci].position
        cq = self.ms.pose[ci].orientation
        tp = self.ms.pose[ti].position
        return self.proj.compute((cp.x, cp.y, cp.z),
                                 (cq.x, cq.y, cq.z, cq.w),
                                 (tp.x, tp.y, tp.z))

    # ── render + record + log (MAIN/only worker: the image callback) ──
    def img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, "[gt_record] cv_bridge failed: %r" % e)
            return
        now = rospy.Time.now()
        h, w = frame.shape[:2]
        ccx, ccy = w / 2.0, h / 2.0

        yolo_fresh = (self.det_real and (now - self.det_time).to_sec() < BOX_TIMEOUT)
        box_fresh = (self.box_wh is not None and (now - self.box_time).to_sec() < BOX_TIMEOUT)
        bbox_area = float('nan')
        if box_fresh:
            bbox_area = float(self.box_wh[0]) * float(self.box_wh[1])

        # detector status parse (for [GATED] tag)
        state = conf = sw = sh = ""
        if self.status:
            p = self.status.split(',')
            if len(p) == 4:
                state, conf, sw, sh = p
        gated = state not in ("", "TRACKING")

        # YOLO box overlay
        if box_fresh and yolo_fresh and not math.isnan(self.cx):
            bx, by = int(self.cx), int(self.cy)
            bw, bh = int(self.box_wh[0]), int(self.box_wh[1])
            cv2.rectangle(frame, (bx - bw // 2, by - bh // 2),
                          (bx + bw // 2, by + bh // 2), (0, 255, 0), 2)
            lbl = "DRONE %dx%d" % (bw, bh)
            if conf and conf != 'nan':
                lbl += " conf=%s" % conf
            cv2.putText(frame, lbl, (bx - bw // 2, by - bh // 2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO YOLO DETECTION", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ground-truth overlay + CSV fields
        gt = self._gt()
        gt_u = gt_v = du = dv = sep = depth = float('nan')
        in_frame = 0
        exit_edge = ""
        axis = ""
        hud2 = ""
        if gt is not None:
            sep = gt['true_dist']; depth = gt['depth']
            u, v = gt['u'], gt['v']
            behind = (depth <= self.proj.near) or math.isnan(u) or math.isnan(v)
            if behind:
                exit_edge = "BEHIND"
                cv2.putText(frame, "TARGET BEHIND CAMERA", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                gt_u, gt_v = u, v
                du, dv = (u - ccx), (v - ccy)
                if gt['in_fov']:
                    in_frame = 1
                    iu, iv = int(round(u)), int(round(v))
                    cv2.drawMarker(frame, (iu, iv), (255, 0, 255), cv2.MARKER_CROSS, 26, 2)
                    cv2.circle(frame, (iu, iv), 3, (255, 0, 255), -1)
                    cv2.putText(frame, "GT", (iu + 10, iv - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                else:
                    edges = []
                    if v < 0:  edges.append("TOP")
                    if v >= h: edges.append("BOTTOM")
                    if u < 0:  edges.append("LEFT")
                    if u >= w: edges.append("RIGHT")
                    exit_edge = "/".join(edges)
                    axis = "ey(vert)" if (v < 0 or v >= h) else "ex(horiz)"
                    cu = min(max(u, 6), w - 6); cv = min(max(v, 6), h - 6)
                    cv2.arrowedLine(frame, (int(ccx), int(ccy)), (int(cu), int(cv)),
                                    (0, 255, 255), 2, tipLength=0.06)
                    cv2.drawMarker(frame, (int(cu), int(cv)), (0, 255, 255),
                                   cv2.MARKER_TRIANGLE_UP, 22, 3)
                    cv2.putText(frame, "TARGET OFF-FRAME %s  axis=%s" % (exit_edge, axis),
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                owner = "ex(horiz)" if abs(du) >= abs(dv) else "ey(vert)"
                hud2 = ("sep=%.1fm depth=%.1fm  du=%+.0f dv=%+.0f -> %s%s"
                        % (sep, depth, du, dv, owner, "" if in_frame else "  [OFF]"))

        # HUD
        self.roll.append(1 if yolo_fresh else 0)
        if len(self.roll) > ROLL_WINDOW:
            self.roll.pop(0)
        rate = 100.0 * sum(self.roll) / max(len(self.roll), 1)
        cv2.putText(frame, "Detection rate: %.1f%% (last %d)  t=%.1f" % (rate, len(self.roll), rospy.get_time()),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if gated:
            cv2.putText(frame, "[GATED] %s" % state, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if hud2:
            cv2.putText(frame, hud2, (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        # write video
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.record_path, fourcc, VIDEO_FPS, (w, h))
            rospy.loginfo("[gt_record] recording -> %s (%dx%d)", self.record_path, w, h)
        self.writer.write(frame)

        # write CSV row
        def fmt(x):
            return 'nan' if (isinstance(x, float) and math.isnan(x)) else ('%.2f' % x)
        self.f.write('%.3f,%d,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s\n' % (
            rospy.get_time(), 1 if yolo_fresh else 0,
            fmt(gt_u), fmt(gt_v), in_frame, exit_edge, axis,
            fmt(du), fmt(dv), fmt(sep), fmt(depth),
            'nan' if math.isnan(bbox_area) else '%.0f' % bbox_area))
        self.n += 1

    def on_shutdown(self):
        try:
            self.f.flush(); self.f.close()
        except Exception:
            pass
        if self.writer is not None:
            self.writer.release()
        rospy.loginfo("[gt_record] wrote %d frames (intrinsics: %s) -> %s",
                      self.n, self.intr_src, self.record_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', required=True, help='output mp4 path')
    ap.add_argument('--csv', required=True, help='output per-frame CSV path')
    ap.add_argument('--target', default='target_iris')
    args, _ = ap.parse_known_args()
    GTRecorder(args.record, args.csv, target=args.target)
    rospy.spin()
