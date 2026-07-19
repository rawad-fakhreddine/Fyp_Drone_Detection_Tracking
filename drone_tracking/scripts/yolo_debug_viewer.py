#!/usr/bin/env python3
"""yolo_debug_viewer.py — live overlay of YOLO detections with REAL bbox dimensions

v1.6 (2026-07-07): ground-truth FOV indicator + %% (Gazebo).
  * Rotates the chaser->target LOS into the camera frame (chaser quaternion
    from /gazebo/model_states; 5 deg down mount per fpv_cam.sdf; half-angles
    from the LIVE fx=277 optics: az ±49.1 deg, el ±40.9 deg) and shows
    IN FOV / OUT OF FOV (with az/el when out) + %% of frames in FOV since
    takeoff — visibility truth, independent of YOLO and of the HOLD metric.

v1.5 (2026-07-07): detection %% counted from TAKEOFF COMPLETE, single total.
  * The rolling last-300-frames rate is REMOVED from the overlay; one number
    remains: detected/total camera frames, counted only once
    /drone_tracking/takeoff_ready publishes True (both drones at altitude).
    Frames during world-load/takeoff no longer pollute the statistic. The
    window still opens at node start (same launch order); only the counter
    is gated — equivalent to launching the viewer after takeoff.

v1.4 (2026-07-07): display resolution + rate + mission-total stats + GT distance.
  * ~scale (default 2.0): display upscale — 640x480 camera -> 1280x960 window.
    DISPLAY-ONLY (cv2.resize after the frame is stored); the camera/native
    inference resolution is untouched, so detection behavior is identical.
    Overlays are drawn AFTER the resize at scaled coordinates (crisp text).
  * ~hz (default 60): render-loop rate (was fixed ~30 via waitKey(33)).
  * CUMULATIVE detection %: detected/total camera frames since node start,
    shown next to the v1.2 rolling-window rate ("% YOLO detection in total
    frames" — the whole-mission number, not just the last 300 frames).
  * Gazebo ground-truth separation: subscribes /gazebo/model_states and shows
    the chaser-target distance as dx,dy,dz + 3D norm (world frame) — live
    verification of the HOLD standoff (d*=8 m mission).

v1.3 (2026-06-17): GUI moved to the MAIN thread (FOV-window fix).
  ROOT CAUSE of "VIEWER=1 but no window appeared": v1.1/v1.2 called
  cv2.imshow()/cv2.waitKey() inside the ROS subscriber callback (img_cb),
  which rospy runs on a SEPARATE thread while the main thread sat in
  rospy.spin(). Under the OpenCV 4.13 Qt5 backend, GUI calls off the main
  thread render unreliably — with NO Python exception, so the TV log looked
  clean and the window silently never showed.
  Fix: callbacks only STORE state; all cv2 GUI calls run in a main-thread
  render loop (run()). Window created EAGERLY with a placeholder.

v1.2 (2026-06-11): rolling detection rate (last 300 frames), box conf + w x h
  from /drone_tracking/detector_status, [GATED] tag while the v3.3 gate
  withholds candidates.

v1.1 (2026-06-10): stale-box handling (BOX_TIMEOUT) + NO DETECTION banner.
Launched automatically by launch_stack.sh (stage TV; VIEWER=1 to enable).
"""
import rospy, cv2, math
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5   # s of sim time without a new box before the overlay clears
WIN = 'YOLO Debug'


class Viewer:
    def __init__(self):
        rospy.init_node('yolo_debug_viewer')
        self.bridge = CvBridge()
        # v1.4.1: defaults match the chaser camera EXACTLY (fpv_cam.sdf:
        # 640x480 @ 30 Hz). scale=1.0 shows native pixels 1:1 — any resize
        # (up or down) softens the image; >1.0 is opt-in for a bigger window.
        self.scale = float(rospy.get_param('~scale', 1.0))   # v1.4 display upscale
        hz = float(rospy.get_param('~hz', 30.0))             # v1.4 render rate
        self.delay_ms = max(1, int(1000.0 / max(hz, 1.0)))
        self.frame = None                       # latest cv2 image (set in img_cb)
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.tot_frames = 0                     # v1.5: counted from takeoff only
        self.det_frames = 0
        self.takeoff = False                    # v1.5: gates the counters
        self.status = ""                        # v3.3 "STATE,conf,w,h"
        # v1.4: Gazebo ground-truth separation (world frame)
        self.gz_dx = self.gz_dy = self.gz_dz = float('nan')
        # v1.6.1: display-side EMA on the drawn box (raw YOLO boxes jump
        # +-1-2 px/frame from Gazebo render aliasing — visually noisy at
        # 10-40 px sizes). DISPLAY ONLY: control topics untouched.
        # ~box_smooth = weight on the previous drawn box (0 = raw).
        self.box_smooth = float(rospy.get_param('~box_smooth', 0.0))  # v1.6.2: RAW by default — the 0.6 EMA made the C1 box look filtered/delayed (Rawad); ~box_smooth>0 to re-enable
        self._disp_box = None                   # smoothed (cx,cy,w,h)
        # v1.6: ground-truth FOV state (camera: fx=277, 5 deg down mount)
        self.in_fov = None                      # None until gazebo data
        self.fov_az = self.fov_el = float('nan')
        self.fov_in = 0                         # frames in FOV since takeoff
        self.fov_tot = 0
        self.CAM_PITCH = 0.0873
        self.HALF_H = math.atan(320.0 / 277.19)
        self.HALF_V = math.atan(240.0 / 277.19)
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready', Bool, self.takeoff_cb, queue_size=1)
        rospy.loginfo("YOLO viewer v1.6 started (scale=%.1f, %.0f Hz render, "
                      "det%%+FOV%% count from takeoff) — press Q to quit"
                      % (self.scale, hz))

    # ── callbacks: STORE only, no GUI calls (they run off the main thread) ──
    def box_cb(self, msg):
        self.last_box = msg
        self.last_box_time = rospy.Time.now()

    def status_cb(self, msg):
        self.status = msg.data

    def takeoff_cb(self, msg):
        if msg.data and not self.takeoff:
            self.takeoff = True
            rospy.loginfo("[viewer] takeoff complete — detection %% counter started")

    def gazebo_cb(self, msg):
        try:
            ic = msg.name.index('iris')
            it = msg.name.index('target_iris')
        except ValueError:
            return
        self.gz_dx = msg.pose[it].position.x - msg.pose[ic].position.x
        self.gz_dy = msg.pose[it].position.y - msg.pose[ic].position.y
        self.gz_dz = msg.pose[it].position.z - msg.pose[ic].position.z
        # v1.6: LOS -> body (chaser quaternion) -> camera (5 deg down mount)
        q = msg.pose[ic].orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        dx, dy, dz = self.gz_dx, self.gz_dy, self.gz_dz
        bx = ((1-2*(qy*qy+qz*qz))*dx + (2*(qx*qy+qz*qw))*dy + (2*(qx*qz-qy*qw))*dz)
        by = ((2*(qx*qy-qz*qw))*dx + (1-2*(qx*qx+qz*qz))*dy + (2*(qy*qz+qx*qw))*dz)
        bz = ((2*(qx*qz+qy*qw))*dx + (2*(qy*qz-qx*qw))*dy + (1-2*(qx*qx+qy*qy))*dz)
        cp, sp = math.cos(self.CAM_PITCH), math.sin(self.CAM_PITCH)
        xc, yc, zc = cp*bx - sp*bz, by, sp*bx + cp*bz
        self.fov_az = math.atan2(yc, xc)
        self.fov_el = math.atan2(zc, math.hypot(xc, yc))
        self.in_fov = (xc > 0 and abs(self.fov_az) <= self.HALF_H
                       and abs(self.fov_el) <= self.HALF_V)

    def img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, "[viewer] imgmsg_to_cv2 failed: %r" % e)
            return
        if self.takeoff:   # v1.5: count only after both drones are at altitude
            fresh = (self.last_box is not None and
                     (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)
            self.tot_frames += 1
            if fresh:
                self.det_frames += 1
            if self.in_fov is not None:   # v1.6: GT FOV %, same gating
                self.fov_tot += 1
                if self.in_fov:
                    self.fov_in += 1
        self.frame = img

    # ── render: runs on the MAIN thread ──
    def _draw(self, frame):
        s = self.scale
        if s != 1.0:   # v1.4: upscale FIRST, draw overlays at scaled coords
            frame = cv2.resize(frame, None, fx=s, fy=s,
                               interpolation=cv2.INTER_LINEAR)
        fresh = (self.last_box is not None and
                 (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)

        state, conf, sw, sh = "", "", "", ""
        if self.status:
            parts = self.status.split(',')
            if len(parts) == 4:
                state, conf, sw, sh = parts
        gated = state not in ("", "TRACKING")

        if fresh:
            # v1.6.1: smooth the DISPLAYED box (control topics untouched)
            raw = (self.last_box.x, self.last_box.y,
                   self.last_box.z, self.last_box.w)
            a = self.box_smooth
            if self._disp_box is None or a <= 0:
                self._disp_box = raw
            else:
                p = self._disp_box
                # re-seed instead of gliding across a real jump (>40 px)
                if abs(raw[0]-p[0]) > 40 or abs(raw[1]-p[1]) > 40:
                    self._disp_box = raw
                else:
                    self._disp_box = tuple(a*pv + (1-a)*rv
                                           for pv, rv in zip(p, raw))
            bx, bY, bw, bh = self._disp_box
            cx, cy = int(bx * s), int(bY * s)
            w,  h  = int(bw * s), int(bh * s)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "DRONE %dx%d" % (int(bw), int(bh))
            if conf and conf != 'nan':
                label += f" conf={conf}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            self._disp_box = None   # v1.6.1: re-seed after a dropout
            cv2.putText(frame, "NO DETECTION", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if gated:
            tag = f"[GATED] {state}"
            if conf and conf != 'nan':
                tag += f"  cand {sw}x{sh} conf={conf}"
            cv2.putText(frame, tag, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # v1.5: single total detection rate, counted from takeoff complete
        if self.takeoff:
            cum = 100.0 * self.det_frames / max(self.tot_frames, 1)
            txt = "YOLO det: %.1f%% of %d frames (since takeoff)" % (cum, self.tot_frames)
        else:
            txt = "YOLO det: -- (waiting for takeoff)"
        cv2.putText(frame, txt,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # v1.6: ground-truth FOV status + running % (Gazebo, YOLO-independent)
        if self.in_fov is not None:
            pct = 100.0 * self.fov_in / max(self.fov_tot, 1)
            if self.in_fov:
                ftxt, fcol = "TARGET IN FOV", (0, 255, 0)
            else:
                ftxt, fcol = ("OUT OF FOV az=%+.0f el=%+.0f"
                              % (math.degrees(self.fov_az),
                                 math.degrees(self.fov_el))), (0, 0, 255)
            if self.takeoff:
                ftxt += "  |  %.1f%% in FOV" % pct
            cv2.putText(frame, ftxt, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcol, 2)

        # v1.4: Gazebo ground-truth separation (world frame, target - chaser)
        if not math.isnan(self.gz_dx):
            d3 = math.sqrt(self.gz_dx**2 + self.gz_dy**2 + self.gz_dz**2)
            cv2.putText(frame,
                        "GZ dist: %.2fm  dx=%+.1f dy=%+.1f dz=%+.1f"
                        % (d3, self.gz_dx, self.gz_dy, self.gz_dz),
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        return frame

    def run(self):
        # Eager window so it appears immediately, before the first camera frame.
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        base = np.zeros((int(480 * self.scale), int(640 * self.scale), 3), np.uint8)
        # WALL-CLOCK render loop. Deliberately NOT rospy.Rate(): /use_sim_time
        # is true under PX4 SITL and sim time FREEZES during the T5 lockstep
        # window (target instance attaching). A sim-time sleep would stall the
        # Qt event pump there -> window goes unresponsive and WSLg closes it.
        # cv2.waitKey(delay_ms) both pumps the GUI and paces the loop in wall
        # time, so the window stays alive regardless of sim time. (v1.4: delay
        # from ~hz, default 60 Hz — was fixed 33 ms ~ 30 Hz.)
        while not rospy.is_shutdown():
            frame = self.frame
            if frame is not None:
                disp = self._draw(frame.copy())
            else:
                disp = base.copy()
                cv2.putText(disp, "Waiting for camera feed...", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WIN, disp)
            if cv2.waitKey(self.delay_ms) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        if not rospy.is_shutdown():
            rospy.signal_shutdown('user quit')


if __name__ == '__main__':
    Viewer().run()
