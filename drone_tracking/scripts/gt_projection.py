#!/usr/bin/env python3
"""
gt_projection.py — ground-truth target projection into the chaser FPV camera.
==============================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

READ-ONLY MEASUREMENT UTILITY. Subscribes only; never publishes to or modifies
the flight pipeline (kalman / ibvs / detection). It is the SHARED ground-truth
projector behind both measure_yolo_noise.py (R characterization) and
diagnose_dropouts.py (dropout cause classification).

Two faces:
  1. GTProjector (importable class) — pure geometry, no ROS.
  2. Logger node (run this file) — logs one CSV row per camera frame with the
     projected target, the live YOLO detection, and the detector gate state,
     all on the sim clock, for offline analysis by the two tools above.

TRANSFORM CHAIN (verified empirically by the u/v residual sanity check —
measure_yolo_noise warns if |mean residual| is large, which a flipped axis
would cause):
  world target pos (gazebo/model_states: 'target_iris')
   → relative to camera world pose = chaser_pose ⊕ cam_offset
        cam_offset from fpv_cam.sdf: xyz (0.12, 0, -0.03), rpy (0, 0.0873, 0)
        (the fpv_cam is fixed-jointed to iris_chaser::base_link; pitch 0.0873
         rad ≈ 5° nose-down)
   → camera BODY frame (Gazebo camera convention: +X forward, +Y left, +Z up)
        b = R_cam_world^T · (p_target_world - p_cam_world)
   → optical pixel (image: u right, v down):
        u = cx - fx·(Yb/Xb)      v = cy - fy·(Zb/Xb)      depth = Xb
  Intrinsics come from the live /iris/usb_cam/camera_info K matrix when
  available (reported at startup), else from the SDF horizontal_fov fallback.

ALPHA UNITS: the pipeline's "alpha" is bbox AREA in raw pixels (w_px·h_px) —
the same units the Kalman ingests and that R=[6,6,5] is expressed in. The
detection node 5-frame-medians it before publishing, so the logged yolo alpha
is the smoothed value the Kalman actually receives (correct target for R).
projected_alpha uses a point-size model (TARGET_SIZE_M, default 0.5 m): it is
a geometry estimate, so residual_alpha MEAN is dominated by the size model
while its per-distance-bin VARIANCE is the true area noise (what R needs). The
cx/cy residuals are size-model-independent and fully trustworthy.
"""
import math
import numpy as np

# ── Camera mounting (from fpv_cam.sdf <pose> rel. to iris base_link) ──────────
CAM_OFFSET_XYZ = (0.12, 0.0, -0.03)
CAM_OFFSET_RPY = (0.0, 0.0873, 0.0)      # ~5° pitch down
TARGET_SIZE_M  = 0.5                      # representative iris span (task spec)
IMG_W, IMG_H   = 640, 480
SDF_HFOV       = 1.2217                   # rad (70°) — fallback intrinsics
NEAR_PLANE     = 0.1


def quat_to_R(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def intrinsics_from_hfov(hfov=SDF_HFOV, w=IMG_W, h=IMG_H):
    """Gazebo camera: fx = w / (2 tan(hfov/2)); fy=fx; principal point centred."""
    fx = w / (2.0 * math.tan(hfov / 2.0))
    return fx, fx, (w - 1) / 2.0, (h - 1) / 2.0


class GTProjector:
    """Pure-geometry ground-truth projector. No ROS."""

    def __init__(self, fx, fy, cx, cy, w=IMG_W, h=IMG_H, near=NEAR_PLANE,
                 cam_off_xyz=CAM_OFFSET_XYZ, cam_off_rpy=CAM_OFFSET_RPY,
                 target_size=TARGET_SIZE_M):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.w, self.h, self.near = w, h, near
        self.t_off = np.array(cam_off_xyz, dtype=float)
        self.R_off = rpy_to_R(*cam_off_rpy)
        self.S = target_size

    def compute(self, chaser_pos, chaser_quat, target_pos):
        pc = np.array(chaser_pos, dtype=float)
        pt = np.array(target_pos, dtype=float)
        Rc = quat_to_R(*chaser_quat)
        p_cam = pc + Rc @ self.t_off
        R_cam = Rc @ self.R_off
        d_world = pt - p_cam
        true_dist = float(np.linalg.norm(d_world))
        Xb, Yb, Zb = R_cam.T @ d_world          # camera body frame
        in_front = Xb > self.near
        if in_front:
            u = self.cx - self.fx * (Yb / Xb)
            v = self.cy - self.fy * (Zb / Xb)
            pw = self.fx * self.S / Xb
            ph = self.fy * self.S / Xb
            alpha_pix = pw * ph
        else:
            u = v = pw = ph = alpha_pix = float('nan')
        in_fov = bool(in_front and 0 <= u < self.w and 0 <= v < self.h)
        return dict(
            in_fov=in_fov, u=u, v=v, depth=float(Xb), true_dist=true_dist,
            alpha_pix=alpha_pix, proj_w=pw, proj_h=ph,
            azimuth_deg=math.degrees(math.atan2(-Yb, Xb)),     # +right
            elev_deg=math.degrees(math.atan2(-Zb, math.hypot(Xb, Yb))))  # +down


# ─────────────────────────────────────────────────────────────────────────────
#  Logger node (only imported/used when this file is run directly)
# ─────────────────────────────────────────────────────────────────────────────
def _run_logger():
    import os, argparse, rospy
    from gazebo_msgs.msg import ModelStates
    from sensor_msgs.msg import CameraInfo
    from geometry_msgs.msg import Point
    from std_msgs.msg import String

    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='output CSV path')
    ap.add_argument('--target', default='target_iris')
    ap.add_argument('--target_size', type=float, default=TARGET_SIZE_M)
    args, _ = ap.parse_known_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    rospy.init_node('gt_projection_logger', anonymous=True)

    state = dict(proj=None, ms=None, det_status='LOST,0,0,0', n=0,
                 chaser_name=None, intr_src='waiting')

    def cam_info_cb(msg):
        if state['proj'] is None and msg.K and msg.K[0] > 0:
            fx, fy = msg.K[0], msg.K[4]
            cx, cy = msg.K[2], msg.K[5]
            state['proj'] = GTProjector(fx, fy, cx, cy, msg.width, msg.height,
                                        target_size=args.target_size)
            state['intr_src'] = 'camera_info'
            rospy.loginfo("[gt_proj] intrinsics from camera_info: "
                          "fx=%.2f fy=%.2f cx=%.2f cy=%.2f %dx%d",
                          fx, fy, cx, cy, msg.width, msg.height)

    def ms_cb(msg):
        state['ms'] = msg
        if state['chaser_name'] is None:
            for nm in msg.name:
                if 'iris' in nm and nm != args.target and not nm.startswith('target'):
                    state['chaser_name'] = nm
                    rospy.loginfo("[gt_proj] chaser model = '%s' | target = '%s' "
                                  "| all models: %s", nm, args.target,
                                  ', '.join(msg.name))
                    break

    def status_cb(msg):
        state['det_status'] = msg.data

    rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, cam_info_cb, queue_size=1)
    rospy.Subscriber('/gazebo/model_states', ModelStates, ms_cb, queue_size=1)
    rospy.Subscriber('/drone_tracking/detector_status', String, status_cb, queue_size=1)

    f = open(args.log, 'w', buffering=1)            # line-buffered: survives kill
    f.write('sim_time,in_fov,proj_u,proj_v,proj_alpha_pix,proj_w_px,proj_h_px,'
            'true_dist,depth_cam,azimuth_deg,elev_deg,'
            'yolo_cx,yolo_cy,yolo_alpha_pix,yolo_det,'
            'det_state,det_conf,det_w,det_h\n')

    def det_cb(msg):
        # one camera frame per /drone_tracking/target_center message
        if state['proj'] is None:
            # fall back to SDF intrinsics so we never silently drop frames
            fx, fy, cx, cy = intrinsics_from_hfov()
            state['proj'] = GTProjector(fx, fy, cx, cy, target_size=args.target_size)
            state['intr_src'] = 'sdf_hfov'
            rospy.logwarn("[gt_proj] no camera_info yet — using SDF intrinsics "
                          "fx=%.2f cx=%.2f", fx, cx)
        ms, proj = state['ms'], state['proj']
        if ms is None or state['chaser_name'] is None:
            return
        try:
            ci = ms.name.index(state['chaser_name'])
            ti = ms.name.index(args.target)
        except ValueError:
            return
        cp = ms.pose[ci].position
        cq = ms.pose[ci].orientation
        tp = ms.pose[ti].position
        r = proj.compute((cp.x, cp.y, cp.z), (cq.x, cq.y, cq.z, cq.w),
                         (tp.x, tp.y, tp.z))

        det = 'NONE' if (math.isnan(msg.x) or math.isnan(msg.z)) else 'REAL'
        ycx = float('nan') if det == 'NONE' else msg.x
        ycy = float('nan') if det == 'NONE' else msg.y
        yal = float('nan') if det == 'NONE' else msg.z
        st = state['det_status'].split(',')
        ds = st[0] if st else ''
        dc = st[1] if len(st) > 1 else '0'
        dw = st[2] if len(st) > 2 else '0'
        dh = st[3] if len(st) > 3 else '0'

        t = rospy.get_time()                        # sim clock (use_sim_time)
        f.write('%.3f,%d,%.2f,%.2f,%.1f,%.2f,%.2f,%.3f,%.3f,%.2f,%.2f,'
                '%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                    t, int(r['in_fov']), r['u'], r['v'], r['alpha_pix'],
                    r['proj_w'], r['proj_h'], r['true_dist'], r['depth'],
                    r['azimuth_deg'], r['elev_deg'],
                    ('%.2f' % ycx) if det == 'REAL' else 'nan',
                    ('%.2f' % ycy) if det == 'REAL' else 'nan',
                    ('%.1f' % yal) if det == 'REAL' else 'nan',
                    det, ds, dc, dw, dh))
        state['n'] += 1

    rospy.Subscriber('/drone_tracking/target_center', Point, det_cb, queue_size=5)

    def on_shutdown():
        try:
            f.flush(); f.close()
        except Exception:
            pass
        rospy.loginfo("[gt_proj] logged %d frames (intrinsics: %s) -> %s",
                      state['n'], state['intr_src'], args.log)

    rospy.on_shutdown(on_shutdown)
    rospy.loginfo("[gt_proj] logging to %s — waiting for stack topics ...", args.log)
    rospy.spin()


if __name__ == '__main__':
    _run_logger()
