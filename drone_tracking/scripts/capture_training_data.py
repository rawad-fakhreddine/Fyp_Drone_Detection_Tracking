#!/usr/bin/env python3
"""
capture_training_data.py — v2.0 (GT-labeled, safe indexing, backup-aware)
=========================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

Captures frames during dual-SITL simulation and saves them with
ground-truth YOLO labels computed from Gazebo 3D geometry.

KEY FIXES vs v1 (capture_close_range.py):
  * Safe indexing: uses max_existing_index + 1 (not len()) to avoid
    overwriting frames when there are numbering gaps.
  * Separate images/ and labels/ subdirs (not flat like old capture/).
  * Trajectory tag in filename (T1_, T2_, ...) for bucket sorting later.
  * GT labels from Gazebo geometry → consistent bbox sizes at same distance.

OUTPUT STRUCTURE:
  ~/drone_detection/capture_v4/
    raw/
      images/   ← T1_f000000.jpg, T2_f000003.jpg, ...
      labels/   ← T1_f000000.txt, T2_f000003.txt, ...
    classes.txt ← always just "drone"

DOES IT OVERRIDE? NO — uses max(existing_index) + 1 so gaps are safe.

GITHUB: This script lives in catkin_ws (push ✓).
        ~/drone_detection/capture_v4/ is DATA (never push ✗).

USAGE:
  # Run full pipeline first (both drones flying), then in new terminal:
  rosrun drone_tracking capture_training_data.py _traj:=1   # T1: hover
  rosrun drone_tracking capture_training_data.py _traj:=2   # T2: slow straight
  rosrun drone_tracking capture_training_data.py _traj:=3   # T3: fast
  rosrun drone_tracking capture_training_data.py _traj:=4   # T4: circle
  rosrun drone_tracking capture_training_data.py _traj:=5   # T5: lemniscate
  rosrun drone_tracking capture_training_data.py _traj:=6   # T6: incline
  rosrun drone_tracking capture_training_data.py _traj:=7   # T7: steep incline
  rosrun drone_tracking capture_training_data.py _traj:=8   # T8: helix
  rosrun drone_tracking capture_training_data.py _traj:=9   # T9: fuzzy evasion
"""

import os, math, glob, rospy, cv2, numpy as np
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
from cv_bridge import CvBridge

# ── Camera intrinsics (iris_fpv_cam, 640×480) ─────────────────────────────────
IMG_W = 640;  IMG_H = 480
FOV_H = math.radians(65.2);  FOV_V = math.radians(51.2)
FX = (IMG_W / 2.0) / math.tan(FOV_H / 2.0)   # ≈ 500 px
FY = (IMG_H / 2.0) / math.tan(FOV_V / 2.0)   # ≈ 500 px
CX = IMG_W / 2.0;  CY = IMG_H / 2.0

# ── Drone physical size (iris in Gazebo) ──────────────────────────────────────
DRONE_HALF_SPAN   = 0.28   # m  (motor-to-motor diagonal half)
DRONE_HALF_HEIGHT = 0.07   # m  (body half-height)
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
MAX_FRAMES      = 800    # frames per trajectory run (stop after this)
SAVE_INTERVAL_S = 0.15   # min seconds between saves (~7 fps save rate)
MIN_BBOX_PX     = 6      # skip if projected box < 6px (too far/tiny)
MAX_BBOX_FRAC   = 0.90   # skip if box > 90% of image (too close/filling frame)
IBVS_SAVE_PHASES = {"APPROACH", "HOLD"}

# ── Output paths ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.expanduser('~/drone_detection/capture_v4')
RAW_DIR   = os.path.join(BASE_DIR, 'raw')
IMG_DIR   = os.path.join(RAW_DIR, 'images')
LBL_DIR   = os.path.join(RAW_DIR, 'labels')


def _safe_start_index(directory, pattern='*.jpg'):
    """
    Returns max_existing_index + 1 (NOT len).
    Safe even when there are gaps in numbering (e.g., 0006, 0008, skipped 0007).
    This prevents OVERWRITING existing files on re-run.
    """
    existing = glob.glob(os.path.join(directory, pattern))
    if not existing:
        return 0
    indices = []
    for f in existing:
        stem = os.path.splitext(os.path.basename(f))[0]
        digits = ''.join(filter(str.isdigit, stem.split('_')[-1]))
        try:
            indices.append(int(digits))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


class GTCapture:
    def __init__(self):
        rospy.init_node('capture_training_data')

        self.traj = int(rospy.get_param('~traj', 0))
        self.tag  = 'T%d' % self.traj

        # Create output dirs
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LBL_DIR, exist_ok=True)

        # Write classes.txt (single class: drone)
        cls_path = os.path.join(BASE_DIR, 'classes.txt')
        if not os.path.exists(cls_path):
            with open(cls_path, 'w') as f: f.write('drone\n')

        # Safe index: won't overwrite even if gaps exist in numbering
        # Searches for files matching this trajectory tag
        tag_pattern = '%s_f*.jpg' % self.tag
        self.frame_idx   = _safe_start_index(IMG_DIR, tag_pattern)
        self.saved_count = 0
        self.last_save_t = 0.0

        # State
        self.chaser_x=self.chaser_y=self.chaser_z=0.
        self.chaser_yaw=self.chaser_pitch=0.
        self.target_x=self.target_y=self.target_z=0.
        self._got_chaser=self._got_target=False
        self.ibvs_phase='UNKNOWN'
        self.bridge=CvBridge()

        rospy.Subscriber('/mavros/local_position/pose',
                         PoseStamped, self._chaser_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states',
                         ModelStates, self._gazebo_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase',
                         String, self._phase_cb, queue_size=1)
        rospy.Subscriber('/iris/usb_cam/image_raw',
                         Image, self._image_cb, queue_size=1)

        rospy.loginfo("[GTCapture] v2.0 | Traj=%s | MAX=%d | Start index=%d"
                      % (self.tag, MAX_FRAMES, self.frame_idx))
        rospy.loginfo("[GTCapture] Output: %s" % RAW_DIR)
        rospy.spin()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _chaser_cb(self, m):
        self.chaser_x=m.pose.position.x
        self.chaser_y=m.pose.position.y
        self.chaser_z=m.pose.position.z
        q=m.pose.orientation
        self.chaser_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2))
        sp=2*(q.w*q.y-q.z*q.x)
        self.chaser_pitch=(math.copysign(math.pi/2,sp) if abs(sp)>=1 else math.asin(sp))
        self._got_chaser=True

    def _gazebo_cb(self, m):
        try:
            i=m.name.index('target_iris')
            p=m.pose[i].position
            self.target_x=p.x; self.target_y=p.y; self.target_z=p.z
            self._got_target=True
        except: pass

    def _phase_cb(self, m): self.ibvs_phase=m.data

    def _image_cb(self, msg):
        if self.saved_count >= MAX_FRAMES:
            rospy.loginfo("[GTCapture] Done — saved %d/%d frames for %s"
                          % (self.saved_count, MAX_FRAMES, self.tag))
            rospy.signal_shutdown("capture complete")
            return

        if not self._got_chaser or not self._got_target: return
        if self.ibvs_phase not in IBVS_SAVE_PHASES: return

        now = rospy.Time.now().to_sec()
        if now - self.last_save_t < SAVE_INTERVAL_S: return

        result = self._compute_gt_bbox()
        if result is None: return
        cx_n, cy_n, w_n, h_n = result

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("[GTCapture] Image error: %s" % e); return

        # Build filename: T1_f000042 (trajectory + global index)
        fname  = '%s_f%06d' % (self.tag, self.frame_idx)
        img_path = os.path.join(IMG_DIR, fname + '.jpg')
        lbl_path = os.path.join(LBL_DIR, fname + '.txt')

        cv2.imwrite(img_path, frame)
        with open(lbl_path, 'w') as f:
            f.write('0 %.6f %.6f %.6f %.6f\n' % (cx_n, cy_n, w_n, h_n))

        self.frame_idx  += 1
        self.saved_count += 1
        self.last_save_t = now

        if self.saved_count % 100 == 0 or self.saved_count == 1:
            d = math.sqrt((self.target_x-self.chaser_x)**2+
                          (self.target_y-self.chaser_y)**2+
                          (self.target_z-self.chaser_z)**2)
            area = w_n*IMG_W * h_n*IMG_H
            rospy.loginfo("[GTCapture] %d/%d | d=%.1fm area=%.0fpx² bbox=(%.3f,%.3f,%.3f,%.3f)"
                          % (self.saved_count, MAX_FRAMES, d, area, cx_n, cy_n, w_n, h_n))

    # ── GT bbox projection from Gazebo geometry ────────────────────────────────
    def _compute_gt_bbox(self):
        dx=self.target_x-self.chaser_x
        dy=self.target_y-self.chaser_y
        dz=self.target_z-self.chaser_z
        d=math.sqrt(dx**2+dy**2+dz**2)
        if d < 0.5: return None

        cy_=math.cos(self.chaser_yaw);  sy_=math.sin(self.chaser_yaw)
        cp_=math.cos(self.chaser_pitch); sp_=math.sin(self.chaser_pitch)

        def world_to_cam(wx, wy, wz):
            bx =  cy_*wx + sy_*wy
            by_= -sy_*wx + cy_*wy
            bz_= wz
            cam_z =  cp_*bx + sp_*bz_
            cam_x = -by_
            cam_y = -(sp_*bx - cp_*bz_)
            return cam_x, cam_y, cam_z

        # Check center is in front
        cx__, cy__, cz__ = world_to_cam(dx, dy, dz)
        if cz__ < 0.2: return None
        u_c = FX*(cx__/cz__)+CX
        v_c = FY*(cy__/cz__)+CY
        if not (0 < u_c < IMG_W and 0 < v_c < IMG_H): return None

        # Project 8 drone corners
        us, vs = [], []
        for corner in DRONE_CORNERS:
            cx2, cy2, cz2 = world_to_cam(dx+corner[0], dy+corner[1], dz+corner[2])
            if cz2 < 0.1: continue
            us.append(FX*(cx2/cz2)+CX)
            vs.append(FY*(cy2/cz2)+CY)
        if len(us) < 4: return None

        u_min,u_max=min(us),max(us)
        v_min,v_max=min(vs),max(vs)
        bw=u_max-u_min; bh=v_max-v_min

        if bw < MIN_BBOX_PX or bh < MIN_BBOX_PX: return None
        if bw/IMG_W > MAX_BBOX_FRAC or bh/IMG_H > MAX_BBOX_FRAC: return None

        return ((u_min+u_max)/2/IMG_W, (v_min+v_max)/2/IMG_H,
                bw/IMG_W, bh/IMG_H)


if __name__=='__main__':
    try: GTCapture()
    except rospy.ROSInterruptException: pass