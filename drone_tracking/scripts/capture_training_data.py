#!/usr/bin/env python3
"""
capture_training_data.py — v3.1 (always capture, 3Hz, empty labels)
=====================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v3.1 changes from v3.0:
  * No phase filter — captures in ALL phases (TAKEOFF, APPROACH, HOLD, SEARCH)
    so negative examples (no target visible) are collected naturally
  * SAVE_INTERVAL_S 0.15 → 0.333 (3 Hz)
  * max_frames default 800 → 1000
  * When target out of FOV: saves image with EMPTY label file
    (empty label = no target = _empty bucket in sort_v4_by_scenario.py)

USAGE:
  rosrun drone_tracking capture_training_data.py
  rosrun drone_tracking capture_training_data.py _max_frames:=1500
"""

import os, math, glob, rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
from cv_bridge import CvBridge

# ── Camera intrinsics (iris_fpv_cam, 640×480) ─────────────────────────────────
IMG_W = 640;  IMG_H = 480
FOV_H = math.radians(65.2);  FOV_V = math.radians(51.2)
FX = (IMG_W / 2.0) / math.tan(FOV_H / 2.0)
FY = (IMG_H / 2.0) / math.tan(FOV_V / 2.0)
CX = IMG_W / 2.0;  CY = IMG_H / 2.0

# ── Drone physical size (iris in Gazebo) ──────────────────────────────────────
DRONE_HALF_SPAN   = 0.28
DRONE_HALF_HEIGHT = 0.07
DRONE_CORNERS = np.array([
    [ DRONE_HALF_SPAN,  DRONE_HALF_SPAN,  DRONE_HALF_HEIGHT],
    [ DRONE_HALF_SPAN, -DRONE_HALF_SPAN,  DRONE_HALF_HEIGHT],
    [-DRONE_HALF_SPAN,  DRONE_HALF_SPAN,  DRONE_HALF_HEIGHT],
    [-DRONE_HALF_SPAN, -DRONE_HALF_SPAN,  DRONE_HALF_HEIGHT],
    [ DRONE_HALF_SPAN,  DRONE_HALF_SPAN, -DRONE_HALF_HEIGHT],
    [ DRONE_HALF_SPAN, -DRONE_HALF_SPAN, -DRONE_HALF_HEIGHT],
    [-DRONE_HALF_SPAN,  DRONE_HALF_SPAN, -DRONE_HALF_HEIGHT],
    [-DRONE_HALF_SPAN, -DRONE_HALF_SPAN, -DRONE_HALF_HEIGHT],
], dtype=np.float64)

# ── Capture settings ──────────────────────────────────────────────────────────
SAVE_INTERVAL_S = 0.333   # v3.1: 3 Hz (was 0.15 = ~6.7 Hz)
MIN_BBOX_PX     = 6
MAX_BBOX_FRAC   = 0.90

# ── Output paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser('~/drone_detection/capture_v4')
RAW_DIR  = os.path.join(BASE_DIR, 'raw')
IMG_DIR  = os.path.join(RAW_DIR, 'images')
LBL_DIR  = os.path.join(RAW_DIR, 'labels')


def _safe_start_index(directory):
    existing = glob.glob(os.path.join(directory, 'f*.jpg'))
    if not existing:
        return 0
    indices = []
    for f in existing:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:    indices.append(int(stem[1:]))
        except: pass
    return max(indices) + 1 if indices else 0


class GTCapture:
    def __init__(self):
        rospy.init_node('capture_training_data')

        self.max_frames = int(rospy.get_param('~max_frames', 1000))

        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LBL_DIR, exist_ok=True)
        cls_path = os.path.join(BASE_DIR, 'classes.txt')
        if not os.path.exists(cls_path):
            with open(cls_path, 'w') as f: f.write('drone\n')

        self.frame_idx   = _safe_start_index(IMG_DIR)
        self.saved_count = 0
        self.last_save_t = 0.0

        self.chaser_x = self.chaser_y = self.chaser_z = 0.
        self.chaser_yaw = self.chaser_pitch = 0.
        self.target_x = self.target_y = self.target_z = 0.
        self._got_chaser = self._got_target = False
        self.ibvs_phase = 'UNKNOWN'
        self.bridge = CvBridge()

        rospy.Subscriber('/mavros/local_position/pose',
                         PoseStamped, self._chaser_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states',
                         ModelStates, self._gazebo_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase',
                         String, self._phase_cb, queue_size=1)
        rospy.Subscriber('/iris/usb_cam/image_raw',
                         Image, self._image_cb, queue_size=1)

        rospy.loginfo("[GTCapture] v3.1 | 3Hz | MAX=%d | Start index=%d"
                      % (self.max_frames, self.frame_idx))
        rospy.loginfo("[GTCapture] Output: %s" % RAW_DIR)
        rospy.loginfo("[GTCapture] Capturing ALL phases (empty labels = negative examples)")
        rospy.spin()

    def _chaser_cb(self, m):
        self.chaser_x = m.pose.position.x
        self.chaser_y = m.pose.position.y
        self.chaser_z = m.pose.position.z
        q = m.pose.orientation
        self.chaser_yaw   = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
        sp = 2*(q.w*q.y - q.z*q.x)
        self.chaser_pitch = math.copysign(math.pi/2, sp) if abs(sp)>=1 else math.asin(sp)
        self._got_chaser  = True

    def _gazebo_cb(self, m):
        try:
            i = m.name.index('target_iris')
            p = m.pose[i].position
            self.target_x = p.x; self.target_y = p.y; self.target_z = p.z
            self._got_target = True
        except Exception: pass

    def _phase_cb(self, m): self.ibvs_phase = m.data

    def _image_cb(self, msg):
        if self.saved_count >= self.max_frames:
            rospy.loginfo("[GTCapture] Done — %d frames saved. Shutting down."
                          % self.saved_count)
            rospy.signal_shutdown("capture complete")
            return

        # Wait for pose data to arrive (brief startup delay)
        if not self._got_chaser or not self._got_target:
            return

        now = rospy.Time.now().to_sec()
        if now - self.last_save_t < SAVE_INTERVAL_S:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("[GTCapture] Image error: %s" % e); return

        fname    = 'f%06d' % self.frame_idx
        img_path = os.path.join(IMG_DIR, fname + '.jpg')
        lbl_path = os.path.join(LBL_DIR, fname + '.txt')

        cv2.imwrite(img_path, frame)

        # GT bbox — save label if target visible, empty file if not
        result = self._compute_gt_bbox()
        if result is not None:
            cx_n, cy_n, w_n, h_n = result
            with open(lbl_path, 'w') as f:
                f.write('0 %.6f %.6f %.6f %.6f\n' % (cx_n, cy_n, w_n, h_n))
        else:
            open(lbl_path, 'w').close()   # empty = no target in FOV → _empty bucket

        self.frame_idx   += 1
        self.saved_count += 1
        self.last_save_t  = now

        if self.saved_count % 100 == 0 or self.saved_count == 1:
            if result is not None:
                d    = math.sqrt((self.target_x-self.chaser_x)**2 +
                                 (self.target_y-self.chaser_y)**2 +
                                 (self.target_z-self.chaser_z)**2)
                area = cx_n*0 + w_n*IMG_W * h_n*IMG_H   # reuse w_n/h_n
                area = w_n*IMG_W * h_n*IMG_H
                rospy.loginfo("[GTCapture] %d/%d | phase=%s d=%.1fm "
                              "area=%.0fpx² bbox=(%.3f,%.3f,%.3f,%.3f)" %
                              (self.saved_count, self.max_frames,
                               self.ibvs_phase, d, area,
                               cx_n, cy_n, w_n, h_n))
            else:
                rospy.loginfo("[GTCapture] %d/%d | phase=%s EMPTY (target out of FOV)" %
                              (self.saved_count, self.max_frames, self.ibvs_phase))

    def _compute_gt_bbox(self):
        dx = self.target_x - self.chaser_x
        dy = self.target_y - self.chaser_y
        dz = self.target_z - self.chaser_z
        d  = math.sqrt(dx**2 + dy**2 + dz**2)
        if d < 0.5: return None

        cy_ = math.cos(self.chaser_yaw);  sy_ = math.sin(self.chaser_yaw)
        cp_ = math.cos(self.chaser_pitch); sp_ = math.sin(self.chaser_pitch)

        def world_to_cam(wx, wy, wz):
            bx  =  cy_*wx + sy_*wy
            by_ = -sy_*wx + cy_*wy
            bz_ =  wz
            cam_z =  cp_*bx + sp_*bz_
            cam_x = -by_
            cam_y = -(sp_*bx - cp_*bz_)
            return cam_x, cam_y, cam_z

        cx__, cy__, cz__ = world_to_cam(dx, dy, dz)
        if cz__ < 0.2: return None
        u_c = FX*(cx__/cz__) + CX
        v_c = FY*(cy__/cz__) + CY
        if not (0 < u_c < IMG_W and 0 < v_c < IMG_H): return None

        us, vs = [], []
        for corner in DRONE_CORNERS:
            cx2, cy2, cz2 = world_to_cam(dx+corner[0], dy+corner[1], dz+corner[2])
            if cz2 < 0.1: continue
            us.append(FX*(cx2/cz2) + CX)
            vs.append(FY*(cy2/cz2) + CY)
        if len(us) < 4: return None

        u_min, u_max = min(us), max(us)
        v_min, v_max = min(vs), max(vs)
        bw = u_max - u_min
        bh = v_max - v_min

        if bw < MIN_BBOX_PX or bh < MIN_BBOX_PX: return None
        if bw/IMG_W > MAX_BBOX_FRAC or bh/IMG_H > MAX_BBOX_FRAC: return None

        return ((u_min+u_max)/2/IMG_W, (v_min+v_max)/2/IMG_H,
                bw/IMG_W, bh/IMG_H)


if __name__ == '__main__':
    try:    GTCapture()
    except rospy.ROSInterruptException: pass
