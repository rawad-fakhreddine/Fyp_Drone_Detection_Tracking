#!/usr/bin/env python3
"""
flight_logger.py — M8.2 Fuzzy-aware logger (4 Hz)
===================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

M8.2 additions:
  - Subscribes to /drone_tracking/target_fuzzy_state (Float32MultiArray)
    for 7 fuzzy debug values: d, d_dot, phi, perturb_deg, speed, omega, vz
  - Subscribes to /gazebo/model_states for TRUE world-frame dist_3d
    (fixes the broken cross-frame MAVROS distance from M7.3)
  - New columns: true_dist_3d, fuzzy_d, fuzzy_ddot, fuzzy_phi,
                 fuzzy_perturb_deg, fuzzy_speed, fuzzy_omega, fuzzy_vz

M8 columns preserved: ppo_alpha_star, ppo_lambda
All M7.3 columns preserved (backward-compatible with old CSVs).

Output: ~/flight_log_latest.csv
"""

import rospy, math, csv, os
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import String, Float32MultiArray
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

        # ── Gazebo world positions (M8.2) — true dist_3d ─────────────
        self.chaser_wx = float('nan')
        self.chaser_wy = float('nan')
        self.chaser_wz = float('nan')
        self.target_wx = float('nan')
        self.target_wy = float('nan')
        self.target_wz = float('nan')
        self.got_gazebo = False

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
        # [d, d_dot, phi, perturb_deg, f_speed, f_omega, f_vz]
        self.fuzzy_d           = float('nan')
        self.fuzzy_ddot        = float('nan')
        self.fuzzy_phi         = float('nan')
        self.fuzzy_perturb_deg = float('nan')
        self.fuzzy_speed       = float('nan')
        self.fuzzy_omega       = float('nan')
        self.fuzzy_vz          = float('nan')
        self.got_fuzzy = False

        # ── Image / target params ────────────────────────────────────
        self.img_cx      = 320.0
        self.img_cy      = 240.0
        self.area_norm   = 640.0 * 480.0
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
            'dist_3d',          # legacy (broken cross-frame MAVROS) — kept for backward compat
            # M8 PPO columns
            'ppo_alpha_star', 'ppo_lambda',
            # M8.2 true distance + fuzzy state columns
            'true_dist_3d',     # Gazebo world-frame distance (accurate)
            'fuzzy_d',          # fuzzy input 1: 3D distance (m)
            'fuzzy_ddot',       # fuzzy input 2: closing rate (m/s)
            'fuzzy_phi',        # fuzzy input 3: FOV exposure (0-1)
            'fuzzy_perturb_deg',# OU heading perturbation (deg)
            'fuzzy_speed',      # fuzzy output 1: commanded speed (m/s)
            'fuzzy_omega',      # fuzzy output 2: turn rate (rad/s)
            'fuzzy_vz',         # fuzzy output 3: vertical rate (m/s)
        ])

        # ── Subscribers ──────────────────────────────────────────────
        # Chaser
        rospy.Subscriber('/mavros/local_position/pose',  PoseStamped,
                         self.pose_cb,   queue_size=1)
        rospy.Subscriber('/mavros/setpoint_raw/local',   PositionTarget,
                         self.cmd_cb,    queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase',   String,
                         self.phase_cb,  queue_size=1)
        rospy.Subscriber('/mavros/state', State,
                         self.state_cb,  queue_size=1)
        rospy.Subscriber('/drone_tracking/target_center', Point,
                         self.raw_cb,    queue_size=1)
        rospy.Subscriber('/drone_tracking/filtered_target', Point,
                         self.flt_cb,    queue_size=1)
        # Target (M7.3)
        rospy.Subscriber('/target/mavros/local_position/pose', PoseStamped,
                         self.target_pose_cb, queue_size=1)
        rospy.Subscriber('/target/mavros/setpoint_raw/local',  PositionTarget,
                         self.target_cmd_cb,  queue_size=1)
        # PPO (M8)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints', Quaternion,
                         self.ppo_cb,    queue_size=1)
        # Gazebo world positions (M8.2) — true dist_3d
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self.gazebo_cb, queue_size=1)
        # Fuzzy state from target_mover (M8.2)
        rospy.Subscriber('/drone_tracking/target_fuzzy_state', Float32MultiArray,
                         self.fuzzy_cb,  queue_size=1)

        self.rate = rospy.Rate(4)
        rospy.loginfo("[FlightLogger] M8.2 Fuzzy-aware logger started (4 Hz)")
        rospy.loginfo("[FlightLogger] CSV: %s" % self.csv_path)
        rospy.loginfo("[FlightLogger] target_alpha=%.4f" % self.target_alpha)
        self.run()

    # ── Chaser callbacks ─────────────────────────────────────────────

    def pose_cb(self, msg):
        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z
        q = msg.pose.orientation
        sinr = 2.0*(q.w*q.x + q.y*q.z)
        cosr = 1.0 - 2.0*(q.x*q.x + q.y*q.y)
        self.roll  = math.atan2(sinr, cosr)
        sinp = 2.0*(q.w*q.y - q.z*q.x)
        self.pitch = math.asin(max(-1, min(1, sinp)))
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.yaw   = math.atan2(siny, cosy)

    def cmd_cb(self, msg):
        self.cmd_vx    = msg.velocity.x
        self.cmd_vy    = msg.velocity.y
        self.cmd_vz    = msg.velocity.z
        self.cmd_wz    = msg.yaw_rate
        self.cmd_frame = msg.coordinate_frame

    def phase_cb(self, msg):
        self.phase = msg.data

    def state_cb(self, msg):
        self.armed = msg.armed

    def raw_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.raw_det = "NONE"; return
        self.raw_cx         = msg.x
        self.raw_cy         = msg.y
        self.raw_alpha_pix  = abs(msg.z)
        self.raw_det        = "REAL" if msg.z > 0 else "NONE"

    def flt_cb(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            self.flt_det = "NONE"; return
        self.flt_cx        = msg.x
        self.flt_cy        = msg.y
        self.flt_alpha_pix = abs(msg.z)
        if   msg.z > 0: self.flt_det = "REAL"
        elif msg.z < 0: self.flt_det = "PRED"
        else:           self.flt_det = "NONE"

    # ── Target callbacks (M7.3) ──────────────────────────────────────

    def target_pose_cb(self, msg):
        self.target_px = msg.pose.position.x
        self.target_py = msg.pose.position.y
        self.target_pz = msg.pose.position.z
        self.got_target_pose = True

    def target_cmd_cb(self, msg):
        self.target_cmd_vx = msg.velocity.x
        self.target_cmd_vy = msg.velocity.y
        self.target_cmd_vz = msg.velocity.z
        self.target_cmd_wz = msg.yaw_rate
        self.got_target_cmd = True

    # ── PPO callback (M8) ────────────────────────────────────────────

    def ppo_cb(self, msg):
        self.ppo_alpha_star = msg.z
        self.ppo_lambda     = msg.w
        self.got_ppo        = True

    # ── Gazebo callback (M8.2) — true world positions ─────────────────

    def gazebo_cb(self, msg):
        try:
            ic = msg.name.index('iris')
            self.chaser_wx = msg.pose[ic].position.x
            self.chaser_wy = msg.pose[ic].position.y
            self.chaser_wz = msg.pose[ic].position.z
        except (ValueError, IndexError):
            pass
        try:
            it = msg.name.index('target_iris')
            self.target_wx = msg.pose[it].position.x
            self.target_wy = msg.pose[it].position.y
            self.target_wz = msg.pose[it].position.z
            self.got_gazebo = True
        except (ValueError, IndexError):
            pass

    # ── Fuzzy state callback (M8.2) ───────────────────────────────────

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

            # Legacy dist_3d (broken cross-frame MAVROS — kept for backward compat)
            if self.got_target_pose:
                dx = self.pos_x - self.target_px
                dy = self.pos_y - self.target_py
                dz = self.pos_z - self.target_pz
                dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                dist_3d = float('nan')

            # True world-frame dist_3d from Gazebo (M8.2)
            if self.got_gazebo:
                dx = self.chaser_wx - self.target_wx
                dy = self.chaser_wy - self.target_wy
                dz = self.chaser_wz - self.target_wz
                true_dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                true_dist_3d = float('nan')

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
                # M7.3 target columns
                f(self.target_px) if self.got_target_pose else '',
                f(self.target_py) if self.got_target_pose else '',
                f(self.target_pz) if self.got_target_pose else '',
                f(self.target_cmd_vx) if self.got_target_cmd else '',
                f(self.target_cmd_vy) if self.got_target_cmd else '',
                f(self.target_cmd_vz) if self.got_target_cmd else '',
                f(self.target_cmd_wz) if self.got_target_cmd else '',
                '%.2f' % dist_3d if not math.isnan(dist_3d) else '',
                # M8 PPO columns
                '%.5f' % self.ppo_alpha_star if self.got_ppo else '',
                '%.3f' % self.ppo_lambda     if self.got_ppo else '',
                # M8.2 true distance + fuzzy state
                '%.2f' % true_dist_3d        if self.got_gazebo else '',
                f(self.fuzzy_d,    '%.2f')   if self.got_fuzzy else '',
                f(self.fuzzy_ddot, '%.3f')   if self.got_fuzzy else '',
                f(self.fuzzy_phi,  '%.3f')   if self.got_fuzzy else '',
                f(self.fuzzy_perturb_deg, '%.1f') if self.got_fuzzy else '',
                f(self.fuzzy_speed, '%.3f')  if self.got_fuzzy else '',
                f(self.fuzzy_omega, '%.3f')  if self.got_fuzzy else '',
                f(self.fuzzy_vz,   '%.3f')   if self.got_fuzzy else '',
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