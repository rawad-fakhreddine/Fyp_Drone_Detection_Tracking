#!/usr/bin/env python3
"""
flight_logger.py — M12 (v2)
===========================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

M12 Phase D-prep (v2): C1-vs-C2 measurement hardening.
  * Rate 10 Hz -> 20 Hz: the IBVS controller runs at 20 Hz, so jerk (2nd
    difference of the commanded velocity) computed from a 10 Hz log ALIASES.
    Logging at 20 Hz captures every setpoint -> jerk/derivative metrics are
    computed at the controller's native step. All existing metrics are ratio-
    or sim_time-based (rate-independent), so this is backward-compatible;
    extract_metrics derives dt from consecutive sim_time deltas, not the rate.
  * New END-appended columns from /drone_tracking/ibvs_errors (published by
    ibvs_controller_node): ex_ctrl, ey_ctrl, ea_ctrl, dex, dey, dea, ctrl_state.
    These are the controller's EXACT internal signals and are per-config-correct
    (raw stream in C1, filtered in C2) — unlike the legacy ex/ey/ea columns,
    which are derived from /filtered_target and are therefore 0 in C1 (Kalman
    node is not launched in Config 1). Use ex_ctrl/dex/... for C1-vs-C2 work.
    ctrl_state = 1 REAL / 2 PRED / 0 no-detection-this-cycle.

M9.6 step 3 (IBVS v6.26 P1 guard): new 'emerg' column ('1'/'0', appended at
  END for back-compat) sampled from /drone_tracking/emergency_brake — records
  when the emergency brake override is engaged. EMERGENCY is a flag, NOT a
  phase (the watchdog and HOLD% metrics parse the phase string unchanged).

M9.6 (B5): logging rate 4 Hz -> 10 Hz (denser sampling for the derivative-noise
  metrics in the Q sweep). Logger is now launched BEFORE takeoff_both so the
  TAKEOFF phase and early SEARCH are captured. No rate-dependent logic exists
  (every column is an instantaneous sample), so the rate change is safe.

M9.3 addition (on top of M8.2):
  - world_alt_err = chaser_wz - target_wz  (Gazebo world frame)
    Replaces the broken cross-frame MAVROS alt_err (pos_z - target_pz)
    which had a +0.5m spawn-offset bias.

All previous columns preserved (backward-compatible).
Output: ~/flight_log_latest.csv
"""

import rospy, math, csv, os
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import String, Float32MultiArray, Bool
from gazebo_msgs.msg import ModelStates


class FlightLogger:
    def __init__(self):
        rospy.init_node('flight_logger')

        # ── Chaser state ──────────────────────────────────────────────
        self.pos_x = self.pos_y = self.pos_z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.cmd_vx = self.cmd_vy = self.cmd_vz = self.cmd_wz = 0.0
        self.cmd_frame = 0
        self.phase = "UNKNOWN"
        self.armed = False
        self.emerg = False   # v6.26 emergency brake engaged flag

        # ── M12 Phase D-prep: controller's exact internal error+derivatives ──
        self.ex_ctrl = self.ey_ctrl = self.ea_ctrl = float('nan')
        self.dex = self.dey = self.dea = float('nan')
        self.ctrl_state = 0.0   # 1 REAL / 2 PRED / 0 no-detection this cycle

        # ── Target state (M7.3) ──────────────────────────────────────
        self.target_px = float('nan')
        self.target_py = float('nan')
        self.target_pz = float('nan')
        self.target_cmd_vx = float('nan')
        self.target_cmd_vy = float('nan')
        self.target_cmd_vz = float('nan')
        self.target_cmd_wz = float('nan')
        self.got_target_pose = False
        self.got_target_cmd  = False

        # ── Gazebo world positions (M8.2) ─────────────────────────────
        self.chaser_wx = float('nan')
        self.chaser_wy = float('nan')
        self.chaser_wz = float('nan')
        self.target_wx = float('nan')
        self.target_wy = float('nan')
        self.target_wz = float('nan')
        self.got_gazebo = False
        # ── 2026-07-07: ground-truth in-FOV flag ─────────────────────
        # Chaser orientation from Gazebo (same frame as the positions), LOS
        # rotated world->body->camera. Camera model per the 2026-06 gt work:
        # fx=277.19 is what the live optics actually follow (SDF hfov 70 deg is
        # NOT rendered), mount pitched 5 deg down (fpv_cam.sdf pose 0.0873).
        self.chaser_qx = self.chaser_qy = float('nan')
        self.chaser_qz = self.chaser_qw = float('nan')
        self.CAM_PITCH = 0.0873                       # rad, mount pitch (down)
        self.HALF_H = math.atan(320.0 / 277.19)       # ±49.1 deg
        self.HALF_V = math.atan(240.0 / 277.19)       # ±40.9 deg

        # ── RAW YOLO ─────────────────────────────────────────────────
        self.raw_cx = self.raw_cy = 0.0
        self.raw_alpha_pix = 0.0
        self.raw_det = "NONE"

        # ── FILTERED Kalman ──────────────────────────────────────────
        self.flt_cx = self.flt_cy = 0.0
        self.flt_alpha_pix = 0.0
        self.flt_det = "NONE"

        # ── PPO outputs (M8) ─────────────────────────────────────────
        self.ppo_alpha_star = float('nan')
        self.ppo_lambda     = float('nan')
        self.got_ppo = False

        # ── Fuzzy state (M8.2) ────────────────────────────────────────
        self.fuzzy_d           = float('nan')
        self.fuzzy_ddot        = float('nan')
        self.fuzzy_phi         = float('nan')
        self.fuzzy_perturb_deg = float('nan')
        self.fuzzy_speed       = float('nan')
        self.fuzzy_omega       = float('nan')
        self.fuzzy_vz          = float('nan')
        self.got_fuzzy = False

        # ── Image params ─────────────────────────────────────────────
        self.img_cx    = 320.0
        self.img_cy    = 240.0
        self.area_norm = 640.0 * 480.0
        self.target_alpha = 0.0067

        # ── CSV ──────────────────────────────────────────────────────
        self.csv_path = os.path.expanduser("~/flight_log_latest.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer   = csv.writer(self.csv_file)
        self.writer.writerow([
            'sim_time', 'phase', 'armed',
            'pos_x', 'pos_y', 'pos_z',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_wz', 'cmd_frame',
            'raw_cx', 'raw_cy', 'raw_alpha', 'raw_det',
            'flt_cx', 'flt_cy', 'flt_alpha', 'flt_det',
            'alpha_delta', 'ex', 'ey', 'ea',
            # M7.3 target columns
            'target_px', 'target_py', 'target_pz',
            'target_cmd_vx', 'target_cmd_vy', 'target_cmd_vz', 'target_cmd_wz',
            'dist_3d',           # legacy broken cross-frame — kept for backward compat
            # M8 PPO columns
            'ppo_alpha_star', 'ppo_lambda',
            # M8.2 true distance + fuzzy state
            'true_dist_3d',      # Gazebo world-frame 3D distance (accurate)
            'fuzzy_d',           # fuzzy input: 3D distance (m)
            'fuzzy_ddot',        # fuzzy input: closing rate (m/s)
            'fuzzy_phi',         # fuzzy input: FOV exposure (0-1)
            'fuzzy_perturb_deg', # OU heading perturbation (deg)
            'fuzzy_speed',       # fuzzy output: commanded speed (m/s)
            'fuzzy_omega',       # fuzzy output: turn rate (rad/s)
            'fuzzy_vz',          # fuzzy output: vertical rate (m/s)
            # M9.3 world-frame altitude error
            'world_alt_err',     # chaser_wz - target_wz (Gazebo world frame, no spawn bias)
            # M9.6 step 3: IBVS v6.26 emergency brake flag (appended at END)
            'emerg',             # '1' while the P1 emergency brake override is engaged
            # ── M12 Phase D-prep: controller's exact signals (END-appended) ──
            'ex_ctrl', 'ey_ctrl', 'ea_ctrl',  # errors AS THE CONTROLLER SAW THEM (per-config)
            'dex', 'dey', 'dea',              # error derivatives (headline C1-vs-C2 signal)
            'ctrl_state',                     # 1 REAL / 2 PRED / 0 none this cycle
            # ── 2026-07-07: per-axis Gazebo separation (world frame, END-appended)
            'gz_dx', 'gz_dy', 'gz_dz',        # target - chaser (m); norm = true_dist_3d
            # ── 2026-07-07: ground-truth camera-FOV geometry ──
            'in_fov',                         # '1' target inside camera FOV (Gazebo GT)
            'fov_az_deg', 'fov_el_deg',       # LOS angles in the camera frame
            # ── 2026-07-07: absolute Gazebo world positions (trajectory GT) ──
            'chaser_wx', 'chaser_wy', 'chaser_wz',
            'target_wx', 'target_wy', 'target_wz',
        ])

        # ── Subscribers ──────────────────────────────────────────────
        rospy.Subscriber('/mavros/local_position/pose',    PoseStamped,      self.pose_cb,        queue_size=1)
        rospy.Subscriber('/mavros/setpoint_raw/local',     PositionTarget,   self.cmd_cb,         queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase',     String,           self.phase_cb,       queue_size=1)
        rospy.Subscriber('/mavros/state',                  State,            self.state_cb,       queue_size=1)
        rospy.Subscriber('/drone_tracking/target_center',  Point,            self.raw_cb,         queue_size=1)
        rospy.Subscriber('/drone_tracking/filtered_target',Point,            self.flt_cb,         queue_size=1)
        rospy.Subscriber('/target/mavros/local_position/pose', PoseStamped,  self.target_pose_cb, queue_size=1)
        rospy.Subscriber('/target/mavros/setpoint_raw/local',  PositionTarget,self.target_cmd_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints', Quaternion,       self.ppo_cb,         queue_size=1)
        rospy.Subscriber('/gazebo/model_states',           ModelStates,      self.gazebo_cb,      queue_size=1)
        rospy.Subscriber('/drone_tracking/target_fuzzy_state', Float32MultiArray, self.fuzzy_cb,  queue_size=1)
        rospy.Subscriber('/drone_tracking/emergency_brake', Bool,                self.emerg_cb,   queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_errors',    Float32MultiArray,    self.ibvs_err_cb, queue_size=1)

        self.rate = rospy.Rate(20)   # M12: match the 20 Hz controller (anti-alias jerk)
        rospy.loginfo("[FlightLogger] M12 v2 started (20 Hz) | CSV: %s" % self.csv_path)
        self.run()

    # ── Callbacks ────────────────────────────────────────────────────

    def pose_cb(self, msg):
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z
        q = msg.pose.orientation
        sinr = 2.0*(q.w*q.x + q.y*q.z); cosr = 1.0 - 2.0*(q.x*q.x + q.y*q.y)
        self.roll  = math.atan2(sinr, cosr)
        sinp = 2.0*(q.w*q.y - q.z*q.x)
        self.pitch = math.asin(max(-1, min(1, sinp)))
        siny = 2.0*(q.w*q.z + q.x*q.y); cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.yaw   = math.atan2(siny, cosy)

    def cmd_cb(self, msg):
        self.cmd_vx = msg.velocity.x; self.cmd_vy = msg.velocity.y
        self.cmd_vz = msg.velocity.z; self.cmd_wz = msg.yaw_rate
        self.cmd_frame = msg.coordinate_frame

    def phase_cb(self, msg): self.phase = msg.data
    def state_cb(self, msg): self.armed = msg.armed
    def emerg_cb(self, msg): self.emerg = msg.data

    def ibvs_err_cb(self, msg):
        d = msg.data
        if len(d) >= 7:
            self.ex_ctrl, self.ey_ctrl, self.ea_ctrl = d[0], d[1], d[2]
            self.dex, self.dey, self.dea = d[3], d[4], d[5]
            self.ctrl_state = d[6]

    def raw_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.raw_det = "NONE"; return
        self.raw_cx = msg.x; self.raw_cy = msg.y
        self.raw_alpha_pix = abs(msg.z)
        self.raw_det = "REAL" if msg.z > 0 else "NONE"

    def flt_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.flt_det = "NONE"; return
        self.flt_cx = msg.x; self.flt_cy = msg.y
        self.flt_alpha_pix = abs(msg.z)
        if   msg.z > 0: self.flt_det = "REAL"
        elif msg.z < 0: self.flt_det = "PRED"
        else:           self.flt_det = "NONE"

    def target_pose_cb(self, msg):
        self.target_px = msg.pose.position.x
        self.target_py = msg.pose.position.y
        self.target_pz = msg.pose.position.z
        self.got_target_pose = True

    def target_cmd_cb(self, msg):
        self.target_cmd_vx = msg.velocity.x; self.target_cmd_vy = msg.velocity.y
        self.target_cmd_vz = msg.velocity.z; self.target_cmd_wz = msg.yaw_rate
        self.got_target_cmd = True

    def ppo_cb(self, msg):
        self.ppo_alpha_star = msg.z; self.ppo_lambda = msg.w; self.got_ppo = True

    def gazebo_cb(self, msg):
        try:
            ic = msg.name.index('iris')
            self.chaser_wx = msg.pose[ic].position.x
            self.chaser_wy = msg.pose[ic].position.y
            self.chaser_wz = msg.pose[ic].position.z
            q = msg.pose[ic].orientation
            self.chaser_qx, self.chaser_qy = q.x, q.y
            self.chaser_qz, self.chaser_qw = q.z, q.w
        except (ValueError, IndexError): pass
        try:
            it = msg.name.index('target_iris')
            self.target_wx = msg.pose[it].position.x
            self.target_wy = msg.pose[it].position.y
            self.target_wz = msg.pose[it].position.z
            self.got_gazebo = True
        except (ValueError, IndexError): pass

    def fuzzy_cb(self, msg):
        if len(msg.data) >= 7:
            self.fuzzy_d           = msg.data[0]
            self.fuzzy_ddot        = msg.data[1]
            self.fuzzy_phi         = msg.data[2]
            self.fuzzy_perturb_deg = msg.data[3]
            self.fuzzy_speed       = msg.data[4]
            self.fuzzy_omega       = msg.data[5]
            self.fuzzy_vz          = msg.data[6]
            self.got_fuzzy         = True

    # ── Main loop ────────────────────────────────────────────────────

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()

            raw_alpha   = self.raw_alpha_pix / self.area_norm
            flt_alpha   = self.flt_alpha_pix / self.area_norm
            alpha_delta = abs(flt_alpha - raw_alpha)

            ex = (self.flt_cx - self.img_cx) / self.img_cx if self.flt_det != "NONE" else 0.0
            ey = (self.flt_cy - self.img_cy) / self.img_cy if self.flt_det != "NONE" else 0.0
            ea = flt_alpha - self.target_alpha              if self.flt_det != "NONE" else 0.0

            # Legacy dist_3d (broken cross-frame — kept for backward compat)
            if self.got_target_pose:
                dx = self.pos_x - self.target_px
                dy = self.pos_y - self.target_py
                dz = self.pos_z - self.target_pz
                dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                dist_3d = float('nan')

            # True Gazebo world-frame distance (M8.2)
            # 2026-07-07: per-axis components kept (gz_d* columns, target-chaser)
            if self.got_gazebo:
                gz_dx = self.target_wx - self.chaser_wx
                gz_dy = self.target_wy - self.chaser_wy
                gz_dz = self.target_wz - self.chaser_wz
                true_dist_3d  = math.sqrt(gz_dx*gz_dx + gz_dy*gz_dy + gz_dz*gz_dz)
                world_alt_err = self.chaser_wz - self.target_wz   # M9.3
            else:
                gz_dx = gz_dy = gz_dz = float('nan')
                true_dist_3d  = float('nan')
                world_alt_err = float('nan')

            # Ground-truth in-FOV (2026-07-07): LOS world->body (chaser
            # quaternion from Gazebo) -> camera (5 deg down mount), then
            # compare az/el against the fx=277 half-angles.
            in_fov = ''; fov_az = fov_el = float('nan')
            if self.got_gazebo and not math.isnan(self.chaser_qw):
                qx, qy, qz, qw = (self.chaser_qx, self.chaser_qy,
                                  self.chaser_qz, self.chaser_qw)
                # body = R(q)^T · rel  (rows of R^T = columns of R)
                bx = ((1 - 2*(qy*qy + qz*qz)) * gz_dx +
                      (2*(qx*qy + qz*qw))     * gz_dy +
                      (2*(qx*qz - qy*qw))     * gz_dz)
                by = ((2*(qx*qy - qz*qw))     * gz_dx +
                      (1 - 2*(qx*qx + qz*qz)) * gz_dy +
                      (2*(qy*qz + qx*qw))     * gz_dz)
                bz = ((2*(qx*qz + qy*qw))     * gz_dx +
                      (2*(qy*qz - qx*qw))     * gz_dy +
                      (1 - 2*(qx*qx + qy*qy)) * gz_dz)
                cp, sp = math.cos(self.CAM_PITCH), math.sin(self.CAM_PITCH)
                xc = cp*bx - sp*bz          # camera frame (5 deg down mount)
                yc = by
                zc = sp*bx + cp*bz
                fov_az = math.atan2(yc, xc)
                fov_el = math.atan2(zc, math.hypot(xc, yc))
                in_fov = '1' if (xc > 0 and abs(fov_az) <= self.HALF_H
                                 and abs(fov_el) <= self.HALF_V) else '0'

            def f(v, fmt='%.3f'):
                return fmt % v if not math.isnan(v) else ''

            self.writer.writerow([
                '%.3f' % now, self.phase, int(self.armed),
                '%.3f' % self.pos_x, '%.3f' % self.pos_y, '%.3f' % self.pos_z,
                '%.1f' % math.degrees(self.roll),
                '%.1f' % math.degrees(self.pitch),
                '%.1f' % math.degrees(self.yaw),
                '%.3f' % self.cmd_vx, '%.3f' % self.cmd_vy,
                '%.3f' % self.cmd_vz, '%.3f' % self.cmd_wz,
                '%d'   % self.cmd_frame,
                '%.1f' % self.raw_cx, '%.1f' % self.raw_cy,
                '%.5f' % raw_alpha, self.raw_det,
                '%.1f' % self.flt_cx, '%.1f' % self.flt_cy,
                '%.5f' % flt_alpha, self.flt_det,
                '%.5f' % alpha_delta,
                '%.3f' % ex, '%.3f' % ey, '%.5f' % ea,
                f(self.target_px)      if self.got_target_pose else '',
                f(self.target_py)      if self.got_target_pose else '',
                f(self.target_pz)      if self.got_target_pose else '',
                f(self.target_cmd_vx)  if self.got_target_cmd  else '',
                f(self.target_cmd_vy)  if self.got_target_cmd  else '',
                f(self.target_cmd_vz)  if self.got_target_cmd  else '',
                f(self.target_cmd_wz)  if self.got_target_cmd  else '',
                '%.2f' % dist_3d       if not math.isnan(dist_3d)      else '',
                '%.5f' % self.ppo_alpha_star if self.got_ppo else '',
                '%.3f' % self.ppo_lambda     if self.got_ppo else '',
                '%.2f' % true_dist_3d        if self.got_gazebo else '',
                f(self.fuzzy_d,    '%.2f')   if self.got_fuzzy else '',
                f(self.fuzzy_ddot, '%.3f')   if self.got_fuzzy else '',
                f(self.fuzzy_phi,  '%.3f')   if self.got_fuzzy else '',
                f(self.fuzzy_perturb_deg, '%.1f') if self.got_fuzzy else '',
                f(self.fuzzy_speed, '%.3f')  if self.got_fuzzy else '',
                f(self.fuzzy_omega, '%.3f')  if self.got_fuzzy else '',
                f(self.fuzzy_vz,   '%.3f')   if self.got_fuzzy else '',
                '%.3f' % world_alt_err if not math.isnan(world_alt_err) else '',  # M9.3
                '1' if self.emerg else '0',   # M9.6 step 3: v6.26 emergency brake
                # ── M12 Phase D-prep: controller's exact internal signals ──
                f(self.ex_ctrl, '%.4f'), f(self.ey_ctrl, '%.4f'), f(self.ea_ctrl, '%.5f'),
                f(self.dex, '%.4f'), f(self.dey, '%.4f'), f(self.dea, '%.5f'),
                '%.0f' % self.ctrl_state,
                # ── 2026-07-07: per-axis Gazebo separation (target - chaser) ──
                f(gz_dx), f(gz_dy), f(gz_dz),
                # ── 2026-07-07: ground-truth camera-FOV geometry ──
                in_fov,
                f(math.degrees(fov_az), '%.1f') if not math.isnan(fov_az) else '',
                f(math.degrees(fov_el), '%.1f') if not math.isnan(fov_el) else '',
                # ── absolute Gazebo world positions (trajectory ground truth) ──
                f(self.chaser_wx), f(self.chaser_wy), f(self.chaser_wz),
                f(self.target_wx), f(self.target_wy), f(self.target_wz),
            ])
            self.csv_file.flush()
            self.rate.sleep()

        self.csv_file.close()
        rospy.loginfo("[FlightLogger] CSV closed: %s" % self.csv_path)


if __name__ == '__main__':
    try:
        FlightLogger()
    except rospy.ROSInterruptException:
        pass