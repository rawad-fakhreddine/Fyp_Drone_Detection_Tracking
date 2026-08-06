#!/usr/bin/env python3
"""multiview_recorder.py (M-C) — 3-panel demo recorder into a single mp4:
   [ CHASER FOV + YOLO overlay | SPECTATOR camera | live 3-D TRAJECTORY ]
Sim-time sampled (x1 playback), starts on takeoff, closes on shutdown.
Params: ~out, ~fps (20), ~fov_topic, ~spec_topic."""
import rospy, cv2, numpy as np, threading, math
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BOX_TIMEOUT = 0.5
PANEL_H = 420


class MultiRec:
    def __init__(self):
        rospy.init_node('multiview_recorder', anonymous=True)
        self.out = rospy.get_param('~out', '/tmp/multiview.mp4')
        self.fps = float(rospy.get_param('~fps', 20.0))
        self.render_every = max(1, int(self.fps * 1.2))   # redraw 3-D ~0.8 Hz
        # keeps the 3-D matplotlib redraw from blocking the FOV/spectator capture
        # loop, so the left (FOV) panel keeps up with the camera
        fov_topic = rospy.get_param('~fov_topic', '/iris/usb_cam/image_raw')
        spec_topic = rospy.get_param('~spec_topic',
                                     '/spectator_cam/spectator_cam/image_raw')
        self.br = CvBridge()
        self.fov = None
        self.spec = None
        self.box = None
        self.box_t = rospy.Time(0)
        self.phase = ""
        self.status = ""                       # detector_status "STATE,conf,w,h"
        self.in_fov = None
        self.fov_az = self.fov_el = float('nan')
        self.det_f = self.tot_f = 0            # detection %% counters (from takeoff)
        self.fov_in = self.fov_tot = 0        # GT FOV %% counters
        self.CAM_PITCH = 0.0873               # 5 deg down mount (matches viewer)
        self.HALF_H = math.atan(320.0 / 277.19)
        self.HALF_V = math.atan(240.0 / 277.19)
        self.takeoff = False
        self.tpath = []
        self.cpath = []
        self.plot_img = None
        self.tickn = 0
        self.writer = None
        self.n = 0
        self.rec_start = self.last_t = None   # for x1-playback fps correction
        self._c = self._t = None
        self.fig = None; self.ax = None       # created in the render thread
        rospy.Subscriber(fov_topic, Image, self.fov_cb, queue_size=1)
        rospy.Subscriber(spec_topic, Image, self.spec_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase', String,
                         lambda m: setattr(self, 'phase', m.data), queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String,
                         lambda m: setattr(self, 'status', m.data), queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gz_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready', Bool, self.takeoff_cb, queue_size=1)
        threading.Thread(target=self._render_loop, daemon=True).start()
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.tick)
        rospy.on_shutdown(self.finish)
        rospy.loginfo("[multi_rec] -> %s (fov %s | spec %s)" % (self.out, fov_topic, spec_topic))
        rospy.spin()

    def _render_loop(self):
        # render the 3-D panel in the BACKGROUND at ~8 Hz so it stays smooth
        # without blocking the camera-capture loop (the left/FOV panel)
        self.fig = plt.figure(figsize=(5.0, 4.2))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.tight_layout()               # ONCE — per-frame tight_layout is the fps killer
        r = rospy.Rate(30)                     # 3-D redraw rate (then render-time limited)
        while not rospy.is_shutdown():
            if self.takeoff and len(self.tpath) > 1:
                try:
                    self.plot_img = self._render3d()
                except Exception:
                    pass
            r.sleep()

    def fov_cb(self, m):
        try:
            self.fov = self.br.imgmsg_to_cv2(m, 'bgr8')
        except Exception:
            pass

    def spec_cb(self, m):
        try:
            self.spec = self.br.imgmsg_to_cv2(m, 'bgr8')
        except Exception:
            pass

    def box_cb(self, m):
        self.box = m
        self.box_t = rospy.Time.now()

    def takeoff_cb(self, m):
        if m.data and not self.takeoff:
            self.takeoff = True
            rospy.loginfo("[multi_rec] takeoff — RECORDING")

    def gz_cb(self, m):
        try:
            ic = m.name.index('iris')
            it = m.name.index('target_iris')
        except ValueError:
            return
        c, t = m.pose[ic].position, m.pose[it].position
        self._c = (c.x, c.y, c.z)
        self._t = (t.x, t.y, t.z)
        # GT in-FOV (LOS -> chaser body -> camera, 5 deg down mount) — matches viewer
        q = m.pose[ic].orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        dx, dy, dz = t.x - c.x, t.y - c.y, t.z - c.z
        bx = (1-2*(qy*qy+qz*qz))*dx + 2*(qx*qy+qz*qw)*dy + 2*(qx*qz-qy*qw)*dz
        by = 2*(qx*qy-qz*qw)*dx + (1-2*(qx*qx+qz*qz))*dy + 2*(qy*qz+qx*qw)*dz
        bz = 2*(qx*qz+qy*qw)*dx + 2*(qy*qz-qx*qw)*dy + (1-2*(qx*qx+qy*qy))*dz
        cp, sp = math.cos(self.CAM_PITCH), math.sin(self.CAM_PITCH)
        xc, yc, zc = cp*bx - sp*bz, by, sp*bx + cp*bz
        self.fov_az = math.atan2(yc, xc)
        self.fov_el = math.atan2(zc, math.hypot(xc, yc))
        self.in_fov = (xc > 0 and abs(self.fov_az) <= self.HALF_H
                       and abs(self.fov_el) <= self.HALF_V)

    def tick(self, _evt):
        if not self.takeoff or self.fov is None:
            return
        # detection % + GT FOV % counters (from takeoff, like the debug viewer)
        self.tot_f += 1
        if self.box is not None and (rospy.Time.now() - self.box_t).to_sec() < BOX_TIMEOUT:
            self.det_f += 1
        if self.in_fov is not None:
            self.fov_tot += 1
            if self.in_fov:
                self.fov_in += 1
        if self._t is not None and self.n % 3 == 0:
            self.tpath.append(self._t)
            self.cpath.append(self._c)
        fov = self._draw_fov(self.fov.copy())
        spec = self.spec.copy() if self.spec is not None else np.zeros((480, 640, 3), np.uint8)
        plot = self.plot_img if self.plot_img is not None else np.zeros((PANEL_H, 500, 3), np.uint8)
        frame = self._compose(fov, spec, plot)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.out, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            if not self.writer.isOpened():
                rospy.logerr("[multi_rec] cannot open writer %s" % self.out)
                rospy.signal_shutdown("writer failed")
                return
        self.writer.write(frame)
        if self.rec_start is None:
            self.rec_start = rospy.Time.now()
        self.last_t = rospy.Time.now()
        self.n += 1

    def _draw_fov(self, f):
        fresh = self.box is not None and (rospy.Time.now() - self.box_t).to_sec() < BOX_TIMEOUT
        state, conf, sw, sh = "", "", "", ""
        if self.status:
            p = self.status.split(',')
            if len(p) == 4:
                state, conf, sw, sh = p
        gated = state not in ("", "TRACKING")
        # detection box (+ conf)
        if fresh:
            b = self.box
            cx, cy, w, h = int(b.x), int(b.y), int(b.z), int(b.w)
            cv2.rectangle(f, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (0, 255, 0), 2)
            lab = "DRONE %dx%d" % (w, h)
            if conf and conf != 'nan':
                lab += " conf=%s" % conf
            cv2.putText(f, lab, (cx-w//2, cy-h//2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(f, "NO DETECTION", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if gated:
            cv2.putText(f, "[GATED] %s" % state, (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        # YOLO detection % (since takeoff) + GT FOV % + phase (like the debug viewer)
        det = 100.0 * self.det_f / max(self.tot_f, 1)
        cv2.putText(f, "YOLO det: %.1f%% (%d f)" % (det, self.tot_f), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.in_fov is not None:
            pct = 100.0 * self.fov_in / max(self.fov_tot, 1)
            if self.in_fov:
                ft, fc = "IN FOV  |  %.1f%% in FOV" % pct, (0, 220, 0)
            else:
                ft, fc = ("OUT OF FOV az=%+.0f el=%+.0f | %.1f%%"
                          % (math.degrees(self.fov_az), math.degrees(self.fov_el), pct)), (0, 0, 255)
            cv2.putText(f, ft, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fc, 2)
        if self.phase:
            cv2.putText(f, self.phase, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # inter-drone distance + dx/dy/dz (bottom of the FOV panel)
        if self._t is not None and self._c is not None:
            dx = self._t[0] - self._c[0]
            dy = self._t[1] - self._c[1]
            dz = self._t[2] - self._c[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            cv2.putText(f, "dist %.2f m   dx %+.1f  dy %+.1f  dz %+.1f m" % (dist, dx, dy, dz),
                        (10, f.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return f

    def _render3d(self):
        ax = self.ax
        ax.clear()
        tp = list(self.tpath); cp = list(self.cpath)   # snapshot (main thread appends)
        m = min(len(tp), len(cp))
        if m > 1:
            t = np.array(tp[:m])
            c = np.array(cp[:m])
            # cap render cost: downsample the growing path so draw time stays
            # bounded (keeps the 3-D fps high late in a long run); last point kept.
            if m > 600:
                st = m // 600
                t = np.vstack([t[::st], t[-1]])
                c = np.vstack([c[::st], c[-1]])
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color='#1f77b4', lw=1.4, label='target')
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color='#d62728', lw=1.4, label='chaser')
            ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], color='#1f77b4', s=30)
            ax.scatter(c[-1, 0], c[-1, 1], c[-1, 2], color='#d62728', s=30)
            ax.legend(loc='upper left', fontsize=7)
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Y (m)', fontsize=7)
        ax.set_zlabel('Z (m)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=-58)
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

    def _compose(self, fov, spec, plot3d):
        def rz(img, H):
            hh, ww = img.shape[:2]
            return cv2.resize(img, (int(ww * H / hh), H))

        def label(img, txt):
            cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(img, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return img
        a = label(rz(fov, PANEL_H), "CHASER FOV (YOLO)")
        b = label(rz(spec, PANEL_H), "SPECTATOR CAM")
        c = label(rz(plot3d, PANEL_H), "3D TRAJECTORY (target=blue chaser=red)")
        return np.hstack([a, b, c])

    def finish(self):
        if self.writer is not None:
            self.writer.release()
            # rewrite at the fps actually achieved so playback is x1 real-time
            if self.rec_start is not None and self.last_t is not None and self.n > 5:
                dur = (self.last_t - self.rec_start).to_sec()
                actual = self.n / dur if dur > 0.1 else self.fps
                if abs(actual - self.fps) / self.fps > 0.12:
                    self._reencode(actual)
        rospy.loginfo("[multi_rec] saved %s (%d frames)" % (self.out, self.n))

    def _reencode(self, fps):
        import os
        cap = cv2.VideoCapture(self.out)
        w, h = int(cap.get(3)), int(cap.get(4))
        tmp = self.out + '.tmp.mp4'
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            vw.write(fr)
        cap.release(); vw.release()
        try:
            os.replace(tmp, self.out)
            rospy.loginfo("[multi_rec] re-encoded to %.1f fps (x1 real-time)" % fps)
        except Exception:
            pass


if __name__ == '__main__':
    try:
        MultiRec()
    except rospy.ROSInterruptException:
        pass
