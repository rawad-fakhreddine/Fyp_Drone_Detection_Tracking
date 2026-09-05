#!/usr/bin/env python3
"""gt_overlay_viewer.py — live FPV viewer with YOLO box + GROUND-TRUTH overlay.
==============================================================================
M10.3 visual FOV-loss diagnostic (untracked catkin tool — DISPLAY ONLY, never
publishes to or modifies the flight pipeline). Same window/threading model as
yolo_debug_viewer v1.3 (eager window, all cv2 calls on the MAIN thread, wall-
clock waitKey pacing so the Qt pump survives the T5 sim-time freeze).

What it adds over yolo_debug_viewer: it draws the GROUND-TRUTH target position
projected into the frame using gt_projection.GTProjector (the exact geometry
the gt CSV logs), so the target stays visible AFTER YOLO drops it. When the
target leaves the image it is pinned to the nearest edge with an arrow + an
"OFF <EDGE>" label naming the departure axis (ex = left/right, ey = top/bottom)
— the discriminator between horizontal departure (T3), vertical departure
(T6/T7), and shrink/too-far. A center->target tracer shows the departure
direction; the HUD prints separation, depth, in/out-FOV, and which axis owns
the offset. Optional --record writes the displayed feed to an mp4 for
frame-by-frame scrubbing of the loss moment.

Overlay legend:
  GREEN  rectangle .... live YOLO detection (the controller's input)
  MAGENTA cross ....... ground-truth target (projected; always shown)
  YELLOW edge arrow ... target is OFF-FRAME, pinned to nearest edge
  cyan/red HUD text ... detection rate, separation, departure axis

Usage (normally launched by occlusion_watch.sh alongside the stack):
  rosrun drone_tracking gt_overlay_viewer.py [--record /path/feed.mp4]
"""
import os, sys, math, argparse
import rospy, cv2
import numpy as np
from collections import deque
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
from cv_bridge import CvBridge

# import the SHARED ground-truth projector (ROS-free class) from the sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_projection import GTProjector, intrinsics_from_hfov, IMG_W, IMG_H

BOX_TIMEOUT = 0.5
ROLL_WINDOW = 300
WIN = 'GT Overlay — YOLO(green) + ground-truth(magenta)'


class GTOverlay:
    def __init__(self, target='target_iris', record=None):
        rospy.init_node('gt_overlay_viewer', anonymous=True)
        self.bridge = CvBridge()
        self.target_name = target
        self.frame = None
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.roll = deque(maxlen=ROLL_WINDOW)
        self.status = ""
        self.proj = None
        self.intr_src = 'waiting'
        self.ms = None
        self.chaser_name = None
        self.record_path = record
        self.writer = None
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, self.cam_info_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.ms_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.loginfo("gt_overlay_viewer started (GUI main thread) — press Q to quit")

    # ── callbacks: STORE only (no GUI calls off the main thread) ──
    def box_cb(self, msg):
        self.last_box = msg
        self.last_box_time = rospy.Time.now()

    def status_cb(self, msg):
        self.status = msg.data

    def cam_info_cb(self, msg):
        if self.proj is None and msg.K and msg.K[0] > 0:
            self.proj = GTProjector(msg.K[0], msg.K[4], msg.K[2], msg.K[5],
                                    msg.width, msg.height)
            self.intr_src = 'camera_info'
            rospy.loginfo("[overlay] intrinsics from camera_info fx=%.2f", msg.K[0])

    def ms_cb(self, msg):
        self.ms = msg
        if self.chaser_name is None:
            for nm in msg.name:
                if 'iris' in nm and nm != self.target_name and not nm.startswith('target'):
                    self.chaser_name = nm
                    rospy.loginfo("[overlay] chaser='%s' target='%s'", nm, self.target_name)
                    break

    def img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, "[overlay] imgmsg_to_cv2 failed: %r" % e)
            return
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
        self.roll.append(1 if fresh else 0)
        self.frame = img

    # ── ground-truth projection of the current target into the frame ──
    def _gt(self):
        if self.proj is None:
            fx, fy, cx, cy = intrinsics_from_hfov()
            self.proj = GTProjector(fx, fy, cx, cy)
            self.intr_src = 'sdf_hfov'
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

    # ── render (MAIN thread) ──
    def _draw(self, frame):
        h, w = frame.shape[:2]
        ccx, ccy = w // 2, h // 2

        # YOLO detection (controller input)
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
        state, conf, sw, sh = "", "", "", ""
        if self.status:
            parts = self.status.split(',')
            if len(parts) == 4:
                state, conf, sw, sh = parts
        gated = state not in ("", "TRACKING")
        if fresh:
            bx, by = int(self.last_box.x), int(self.last_box.y)
            bw, bh = int(self.last_box.z), int(self.last_box.w)
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

        # ground-truth overlay (always, even when YOLO has dropped it)
        gt = self._gt()
        hud2 = ""
        if gt is not None:
            sep = gt['true_dist']; depth = gt['depth']
            u, v = gt['u'], gt['v']
            behind = (depth <= self.proj.near) or math.isnan(u) or math.isnan(v)
            if behind:
                cv2.putText(frame, "TARGET BEHIND CAMERA", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif gt['in_fov']:
                iu, iv = int(round(u)), int(round(v))
                cv2.drawMarker(frame, (iu, iv), (255, 0, 255),
                               cv2.MARKER_CROSS, 26, 2)
                cv2.circle(frame, (iu, iv), 3, (255, 0, 255), -1)
                cv2.putText(frame, "GT", (iu + 10, iv - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                # OFF-FRAME: clamp to nearest edge, arrow + departure-axis label
                cu = min(max(u, 6), w - 6); cv = min(max(v, 6), h - 6)
                icu, icv = int(cu), int(cv)
                cv2.arrowedLine(frame, (ccx, ccy), (icu, icv),
                                (0, 255, 255), 2, tipLength=0.06)
                cv2.drawMarker(frame, (icu, icv), (0, 255, 255),
                               cv2.MARKER_TRIANGLE_UP, 22, 3)
                edges = []
                if v < 0:       edges.append("TOP")
                if v >= h:      edges.append("BOTTOM")
                if u < 0:       edges.append("LEFT")
                if u >= w:      edges.append("RIGHT")
                axis = "ey(vert)" if (v < 0 or v >= h) else "ex(horiz)"
                cv2.putText(frame, "TARGET OFF-FRAME %s  axis=%s"
                            % ("/".join(edges), axis), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            # departure-axis read: which offset from center dominates
            if not behind:
                du, dv = (u - ccx), (v - ccy)
                owner = "ex(horiz)" if abs(du) >= abs(dv) else "ey(vert)"
                hud2 = ("sep=%.1fm depth=%.1fm  off-center du=%+.0f dv=%+.0f -> %s"
                        % (sep, depth, du, dv, owner))
            else:
                hud2 = "sep=%.1fm depth=%.1fm" % (sep, depth)

        # HUD
        rate = 100.0 * sum(self.roll) / max(len(self.roll), 1)
        cv2.putText(frame, "Detection rate: %.1f%% (last %d)" % (rate, len(self.roll)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if gated:
            tag = "[GATED] %s" % state
            if conf and conf != 'nan':
                tag += "  cand %sx%s conf=%s" % (sw, sh, conf)
            cv2.putText(frame, tag, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if hud2:
            cv2.putText(frame, hud2, (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        return frame

    def _record(self, disp):
        if self.record_path is None:
            return
        if self.writer is None:
            h, w = disp.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.record_path, fourcc, 20.0, (w, h))
            rospy.loginfo("[overlay] recording -> %s", self.record_path)
        self.writer.write(disp)

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        base = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        while not rospy.is_shutdown():
            frame = self.frame
            if frame is not None:
                disp = self._draw(frame.copy())
            else:
                disp = base.copy()
                cv2.putText(disp, "Waiting for camera feed...", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self._record(disp)
            cv2.imshow(WIN, disp)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        if self.writer is not None:
            self.writer.release()
            rospy.loginfo("[overlay] recording saved -> %s", self.record_path)
        cv2.destroyAllWindows()
        if not rospy.is_shutdown():
            rospy.signal_shutdown('user quit')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='target_iris')
    ap.add_argument('--record', default=None, help='optional mp4 output path')
    args, _ = ap.parse_known_args()
    GTOverlay(target=args.target, record=args.record).run()
