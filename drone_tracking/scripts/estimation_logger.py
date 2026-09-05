#!/usr/bin/env python3
"""
estimation_logger.py — per-frame Kalman estimation-quality logger.
==============================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

READ-ONLY diagnostic node (R-validation deliverable). Subscribes only; never
publishes to the flight pipeline. Reuses gt_projection.GTProjector for the
shared ground-truth projection (live camera_info intrinsics).

Logs ONE CSV row per camera frame at DETECTION rate (~19 Hz), each row pairing:
  - true (cx,cy,alpha)      ground truth  (gt_projection: proj_u, proj_v, proj_alpha_pix)
  - raw YOLO (cx,cy,alpha)  the measurement (/drone_tracking/target_center)
  - kalman (cx,cy,alpha)    the estimate (/drone_tracking/filtered_target)
  - is_dropout              filtered frame was a prediction (kalman z<0) — Config 2;
                            or (Config 1) no real detection this frame.

Trigger:
  --trigger filtered  (Config 2): row written on each /drone_tracking/filtered_target
       (the last link in the chain → raw for this frame already received). is_dropout
       from the sign of filtered z (<0 = predicted-through / outlier-rejected).
  --trigger raw       (Config 1, no Kalman): row written on each target_center;
       filtered columns = nan; is_dropout = (no REAL detection this frame).

ALPHA UNITS / caveat (see gt_projection docstring): proj_alpha uses a 0.5 m
point-size model, so it carries a constant size-model BIAS. cx/cy truth is
geometry-only and trustworthy. For alpha the bias-removed STD of (est-true) is
the meaningful discriminator (analyze_estimation.py reports both RMS and std).
"""
import os, math, argparse
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import String

from gt_projection import GTProjector, intrinsics_from_hfov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='output CSV path')
    ap.add_argument('--trigger', choices=['filtered', 'raw'], default='filtered')
    ap.add_argument('--target', default='target_iris')
    ap.add_argument('--target_size', type=float, default=0.5)
    args, _ = ap.parse_known_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    rospy.init_node('estimation_logger', anonymous=True)

    S = dict(proj=None, ms=None, chaser=None, det_status='LOST,0,0,0',
             raw=None, filt=None, n=0, intr='waiting')

    def cam_cb(msg):
        if S['proj'] is None and msg.K and msg.K[0] > 0:
            S['proj'] = GTProjector(msg.K[0], msg.K[4], msg.K[2], msg.K[5],
                                    msg.width, msg.height,
                                    target_size=args.target_size)
            S['intr'] = 'camera_info'
            rospy.loginfo("[est] intrinsics from camera_info: fx=%.2f fy=%.2f "
                          "cx=%.2f cy=%.2f %dx%d", msg.K[0], msg.K[4],
                          msg.K[2], msg.K[5], msg.width, msg.height)

    def ms_cb(msg):
        S['ms'] = msg
        if S['chaser'] is None:
            for nm in msg.name:
                if 'iris' in nm and nm != args.target and not nm.startswith('target'):
                    S['chaser'] = nm
                    rospy.loginfo("[est] chaser='%s' target='%s' models=%s",
                                  nm, args.target, ', '.join(msg.name))
                    break

    def status_cb(msg):
        S['det_status'] = msg.data

    def raw_cb(msg):
        # raw YOLO measurement: NaN x/z => no detection this frame
        if math.isnan(msg.x) or math.isnan(msg.z):
            S['raw'] = None
        else:
            S['raw'] = (msg.x, msg.y, msg.z)
        if args.trigger == 'raw':
            write_row()

    def filt_cb(msg):
        # kalman output: z<0 => predicted-through / outlier-rejected (is_dropout)
        if math.isnan(msg.x) or math.isnan(msg.z):
            S['filt'] = None            # uninitialised — treated as dropout, nan est
        else:
            is_drop = msg.z < 0.0
            S['filt'] = (msg.x, msg.y, abs(msg.z), is_drop)
        if args.trigger == 'filtered':
            write_row()

    rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, cam_cb, queue_size=1)
    rospy.Subscriber('/gazebo/model_states', ModelStates, ms_cb, queue_size=1)
    rospy.Subscriber('/drone_tracking/detector_status', String, status_cb, queue_size=1)
    rospy.Subscriber('/drone_tracking/target_center', Point, raw_cb, queue_size=5)
    rospy.Subscriber('/drone_tracking/filtered_target', Point, filt_cb, queue_size=5)

    f = open(args.log, 'w', buffering=1)        # line-buffered: survives -9 kill
    f.write('sim_time,in_fov,true_dist,proj_u,proj_v,proj_alpha,'
            'raw_cx,raw_cy,raw_alpha,raw_det,'
            'filt_cx,filt_cy,filt_alpha,is_dropout,det_state\n')

    def write_row():
        if S['proj'] is None:
            return   # wait for camera_info — NEVER SDF-fallback (real fx=277, not 457)
        ms, proj = S['ms'], S['proj']
        if ms is None or S['chaser'] is None:
            return
        try:
            ci = ms.name.index(S['chaser']); ti = ms.name.index(args.target)
        except ValueError:
            return
        cp, cq, tp = ms.pose[ci].position, ms.pose[ci].orientation, ms.pose[ti].position
        r = proj.compute((cp.x, cp.y, cp.z), (cq.x, cq.y, cq.z, cq.w),
                         (tp.x, tp.y, tp.z))

        raw = S['raw']
        if raw is None:
            rcx = rcy = ral = 'nan'; rdet = 'NONE'
        else:
            rcx = '%.2f' % raw[0]; rcy = '%.2f' % raw[1]; ral = '%.1f' % raw[2]; rdet = 'REAL'

        if args.trigger == 'filtered':
            filt = S['filt']
            if filt is None:
                fcx = fcy = fal = 'nan'; isd = 1
            else:
                fcx = '%.2f' % filt[0]; fcy = '%.2f' % filt[1]; fal = '%.1f' % filt[2]
                isd = 1 if filt[3] else 0
        else:   # raw trigger (Config 1, no kalman)
            fcx = fcy = fal = 'nan'
            isd = 0 if rdet == 'REAL' else 1

        ds = (S['det_status'].split(',') + ['', '', '', ''])[0]
        t = rospy.get_time()
        f.write('%.3f,%d,%.3f,%.2f,%.2f,%.1f,%s,%s,%s,%s,%s,%s,%s,%d,%s\n' % (
            t, int(r['in_fov']), r['true_dist'], r['u'], r['v'], r['alpha_pix'],
            rcx, rcy, ral, rdet, fcx, fcy, fal, isd, ds))
        S['n'] += 1

    def on_shutdown():
        try:
            f.flush(); f.close()
        except Exception:
            pass
        rospy.loginfo("[est] logged %d frames (intrinsics:%s) -> %s",
                      S['n'], S['intr'], args.log)

    rospy.on_shutdown(on_shutdown)
    rospy.loginfo("[est] trigger=%s logging -> %s (waiting for stack) ...",
                  args.trigger, args.log)
    rospy.spin()


if __name__ == '__main__':
    main()
