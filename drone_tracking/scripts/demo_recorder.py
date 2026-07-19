#!/usr/bin/env python3
"""demo_recorder.py v1.1 — headless mp4 recorder of the annotated chaser view.

Records what the debug viewer shows (RAW box + conf + detector state + GT
FOV/separation + IBVS phase) straight to an .mp4, for supervisor demo clips.
Recording STARTS at takeoff_ready==True ("after the takeoff") and stops when
the node is killed (cleanup.sh / launch_stack teardown) — rospy.on_shutdown
finalizes the file. No GUI, no display smoothing (box is the raw detection,
viewer v1.6.2 semantics), safe to run alongside VIEWER=1.

v1.1 (Rawad's demo spec): TRUE x1 PLAYBACK — frames are sampled by a SIM-TIME
rospy.Timer at exactly ~fps (default 20) and the file is stamped the same
fps, so 60 sim-seconds = 60 video-seconds regardless of the camera/inference
rate (v1.0 wrote per camera callback ~22 fps into a 30 fps file = 1.35x
fast). Default ~scale is now 1.0 = native 640x480, "as the real one".

Params: ~out (mp4 path, required) · ~scale (default 1.0) · ~fps (default 20).
Launch knob: RECORD=1 [RECORD_SCALE] in launch_stack.sh (stage RC).
"""
import math
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5


class DemoRecorder:
    CAM_PITCH = math.radians(5.0)
    HALF_H = math.radians(49.0)   # fx=277 -> ~98 deg hfov
    HALF_V = math.radians(40.9)

    def __init__(self):
        rospy.init_node('demo_recorder', anonymous=True)
        self.out_path = rospy.get_param('~out')
        self.scale = float(rospy.get_param('~scale', 1.0))
        self.fps = float(rospy.get_param('~fps', 20.0))
        self.bridge = CvBridge()
        self.writer = None
        self.takeoff = False
        self.frames = 0
        self.frame = None            # latest camera image (BGR)
        self.last_box = None
        self.last_box_time = rospy.Time(0)
        self.status = ""
        self.phase = ""
        self.gz_dx = self.gz_dy = self.gz_dz = float('nan')
        self.in_fov = None
        self.fov_az = self.fov_el = 0.0

        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase', String, self.phase_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready', Bool, self.takeoff_cb, queue_size=1)
        # SIM-TIME sampler: exactly self.fps frames per sim-second -> the file
        # (stamped self.fps) plays back at true x1 speed. Timer only fires once
        # sim time advances, which is fine: recording waits for takeoff anyway.
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.tick)
        rospy.on_shutdown(self.finish)
        rospy.loginfo("[demo_rec] v1.1 -> %s (scale %.1f, %g fps sim-time = x1 playback)"
                      % (self.out_path, self.scale, self.fps))
        rospy.spin()

    def box_cb(self, m):
        self.last_box = m
        self.last_box_time = rospy.Time.now()

    def status_cb(self, m): self.status = m.data
    def phase_cb(self, m): self.phase = m.data

    def takeoff_cb(self, m):
        if m.data and not self.takeoff:
            self.takeoff = True
            rospy.loginfo("[demo_rec] takeoff complete — RECORDING")

    def img_cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def gazebo_cb(self, msg):
        try:
            ic = msg.name.index('iris')
            it = msg.name.index('target_iris')
        except ValueError:
            return
        self.gz_dx = msg.pose[it].position.x - msg.pose[ic].position.x
        self.gz_dy = msg.pose[it].position.y - msg.pose[ic].position.y
        self.gz_dz = msg.pose[it].position.z - msg.pose[ic].position.z
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

    def tick(self, _evt):
        if not self.takeoff or self.frame is None:
            return
        frame = self._draw(self.frame.copy())
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            if not self.writer.isOpened():
                rospy.logerr("[demo_rec] cannot open writer %s" % self.out_path)
                rospy.signal_shutdown("writer failed")
                return
        self.writer.write(frame)
        self.frames += 1

    def _draw(self, frame):
        s = self.scale
        if s != 1.0:
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
            b = self.last_box
            cx, cy = int(b.x * s), int(b.y * s)
            w, h = int(b.z * s), int(b.w * s)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "DRONE %dx%d" % (int(b.z), int(b.w))
            if conf and conf != 'nan':
                label += " conf=%s" % conf
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "NO DETECTION", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        if gated:
            cv2.putText(frame, "[GATED] %s" % state, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        if self.phase:
            cv2.putText(frame, "PHASE: %s" % self.phase, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if self.in_fov is not None:
            if self.in_fov:
                ftxt, fcol = "TARGET IN FOV", (0, 255, 0)
            else:
                ftxt, fcol = ("OUT OF FOV az=%+.0f el=%+.0f"
                              % (math.degrees(self.fov_az),
                                 math.degrees(self.fov_el))), (0, 0, 255)
            cv2.putText(frame, ftxt, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, fcol, 1)

        if not math.isnan(self.gz_dx):
            d3 = math.sqrt(self.gz_dx**2 + self.gz_dy**2 + self.gz_dz**2)
            cv2.putText(frame,
                        "GZ dist: %.2fm  dx=%+.1f dy=%+.1f dz=%+.1f"
                        % (d3, self.gz_dx, self.gz_dy, self.gz_dz),
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        return frame

    def finish(self):
        if self.writer is not None:
            self.writer.release()
            rospy.loginfo("[demo_rec] saved %d frames (%.1f s) -> %s"
                          % (self.frames, self.frames / self.fps, self.out_path))


if __name__ == '__main__':
    DemoRecorder()
