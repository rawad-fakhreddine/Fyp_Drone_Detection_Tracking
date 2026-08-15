#!/usr/bin/env python3
"""
rl_env.py — RL Milestone (Config 3) · Step 1: environment skeleton + observation builder.

DroneTrackingEnv (gymnasium.Env): the RL policy replaces the IBVS CONTROL block only —
YOLO perception stays. This file implements the LOCKED observation design
(FYP/RL/observation/Observation_Design.docx) and the action interface
(FYP/RL/action/Action_Design.docx). Reward/reset are Step 3/4 stubs.

Observation (single frame, 13 values, normalized; frame-stacking N=4 is done by the
SB3 VecFrameStack wrapper, NOT here):
    [ ex, ey_c, d_hat, dex, dey, dd, w, h, conf, t_since_det, pitch, roll,
      a_prev(4) ]                    -> total 16 floats (a_prev = 4 axes)
  - ex   = (cx-320)/320
  - ey_c = pitch-compensated vertical error (same math as ibvs_controller_node:
           beta = atan2(cy-240, F_PX);  ey_c = (F_PX/240)*(beta + pitch_comp*pitch))
  - d_hat = sqrt(alpha_dist_k / alpha), alpha = box_area/(640*480), k = 0.077 (calibrated)
  - dropout rule (locked): FREEZE last valid ex/ey/d/w/h; rates -> 0; conf -> 0;
    t_since_det rises (clipped at 1 s).
Action: [vx, vy, vz, wz] in [-1,1], scaled to caps -> /mavros/setpoint_raw/local
        (FRAME_BODY_NED velocities) — the exact IBVS output interface.

Modes (rosparam ~mode):
  probe  : READ-ONLY. Print the live observation ~2 Hz while the normal stack flies.
           Validates the obs against a real IBVS flight before any learning.
  record : READ-ONLY. Write (obs, IBVS-action) pairs at 20 Hz to CSV for the
           behaviour-cloning warm-start (captures IBVS cmds from setpoint_raw/local).
  env    : library use — import DroneTrackingEnv; step() publishes velocities.

Usage:
  rosrun drone_tracking rl_env.py _mode:=probe
  rosrun drone_tracking rl_env.py _mode:=record _out:=~/rl_demos/T3_C1_s42.csv
"""
import os, sys, csv, math, time
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import PositionTarget
from gazebo_msgs.msg import ModelStates

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:            # probe/record work without gymnasium installed
    _GYM = False

IMG_W, IMG_H = 640.0, 480.0
IMG_CX, IMG_CY = 320.0, 240.0
F_PX = 277.19                  # live camera_info fx (same constant as IBVS)
AREA_NORM = IMG_W * IMG_H

OBS_NAMES = ["ex","ey","d_hat","dex","dey","dd","w","h","conf","t_nodet",
             "pitch","roll","a_vx","a_vy","a_vz","a_wz"]
OBS_DIM = len(OBS_NAMES)


class ObsBuilder(object):
    """Assembles the locked observation vector from the live ROS topics.
    Read-only: subscribes, never publishes."""

    def __init__(self):
        self.k          = float(rospy.get_param("~alpha_dist_k", 0.077))
        self.pitch_comp = float(rospy.get_param("~pitch_comp", 1.3))   # launch default
        self.rate_lpf   = float(rospy.get_param("~rate_lpf", 0.6))     # EMA on rates
        # raw detection state
        self.cx = self.cy = float('nan'); self.alpha = 0.0
        self.box_w = self.box_h = 0.0; self.conf = 0.0
        self.valid = False
        self.t_last_det = None
        # chaser attitude
        self.pitch = self.roll = 0.0
        # frozen-last-valid features (dropout rule) + rates
        self.f_ex = self.f_ey = 0.0; self.f_d = 0.0; self.f_w = self.f_h = 0.0
        self.dex = self.dey = self.dd = 0.0
        self._prev = None                                  # (t, ex, ey, d)
        # last action (captured IBVS cmd in probe/record; policy action in env mode)
        self.a_prev = np.zeros(4, dtype=np.float32)
        self.caps = np.array([float(rospy.get_param("~max_vx", 8.0)),
                              float(rospy.get_param("~max_vy", 1.2)),
                              float(rospy.get_param("~max_vz", 2.5)),
                              float(rospy.get_param("~max_wz", 0.5))], dtype=np.float32)
        # ground truth (record extras / future reward — NEVER enters the obs)
        self.true_dist = float('nan')
        self._names_idx = None

        rospy.Subscriber('/drone_tracking/target_center', Point, self._center_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self._box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self._status_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._gz_cb, queue_size=2)

    # ---- callbacks -------------------------------------------------------
    def _center_cb(self, m):
        if math.isnan(m.x):                       # detector's explicit no-detection point
            self.valid = False; return
        self.cx, self.cy = float(m.x), float(m.y)
        self.alpha = abs(float(m.z)) / AREA_NORM
        self.valid = self.alpha > 1e-9
        if self.valid:
            self.t_last_det = rospy.Time.now()

    def _box_cb(self, m):
        self.box_w, self.box_h = float(m.z), float(m.w)

    def _status_cb(self, m):
        try:
            parts = m.data.split(',')
            self.conf = float(parts[1]) if len(parts) > 1 else 0.0
        except (ValueError, IndexError):
            self.conf = 0.0

    def _pose_cb(self, m):
        q = m.pose.orientation
        sp = 2.0*(q.w*q.y - q.z*q.x)
        self.pitch = math.copysign(math.pi/2, sp) if abs(sp) >= 1 else math.asin(sp)
        sr = 2.0*(q.w*q.x + q.y*q.z); cr = 1.0 - 2.0*(q.x*q.x + q.y*q.y)
        self.roll = math.atan2(sr, cr)

    def _gz_cb(self, m):
        if self._names_idx is None:
            ch = tg = None
            for i, n in enumerate(m.name):
                if 'iris' in n and 'target' not in n and ch is None: ch = i
                if 'target' in n: tg = i
            if ch is None or tg is None: return
            self._names_idx = (ch, tg)
        ch, tg = self._names_idx
        try:
            c, t = m.pose[ch].position, m.pose[tg].position
            self.true_dist = math.sqrt((c.x-t.x)**2 + (c.y-t.y)**2 + (c.z-t.z)**2)
        except IndexError:
            self._names_idx = None

    # ---- assembly --------------------------------------------------------
    def _t_since_det(self):
        if self.t_last_det is None: return 1.0
        return min((rospy.Time.now() - self.t_last_det).to_sec(), 1.0)

    def build(self):
        """Returns (obs16 normalized float32, raw dict for logging/printing)."""
        now = rospy.Time.now().to_sec()
        if self.valid:
            ex = (self.cx - IMG_CX) / IMG_CX
            beta = math.atan2(self.cy - IMG_CY, F_PX)
            ey = (F_PX / IMG_CY) * (beta + self.pitch_comp * self.pitch)
            d = math.sqrt(self.k / max(self.alpha, 1e-9))
            d = min(d, 40.0)
            if self._prev is not None:
                tp, pex, pey, pd = self._prev
                dt = max(now - tp, 1e-3)
                a = self.rate_lpf
                self.dex = a*self.dex + (1-a)*(ex - pex)/dt
                self.dey = a*self.dey + (1-a)*(ey - pey)/dt
                self.dd  = a*self.dd  + (1-a)*(d  - pd )/dt
            self._prev = (now, ex, ey, d)
            self.f_ex, self.f_ey, self.f_d = ex, ey, d
            self.f_w, self.f_h = self.box_w, self.box_h
            conf = self.conf
        else:
            # LOCKED dropout rule: freeze position features, zero rates, conf=0
            self.dex = self.dey = self.dd = 0.0
            self._prev = None
            conf = 0.0

        t_nd = self._t_since_det()
        obs = np.array([
            np.clip(self.f_ex, -1.5, 1.5),
            np.clip(self.f_ey, -1.5, 1.5),
            np.clip(self.f_d / 20.0, 0.0, 2.0),
            np.clip(self.dex / 2.0, -1.0, 1.0),
            np.clip(self.dey / 2.0, -1.0, 1.0),
            np.clip(self.dd  / 5.0, -1.0, 1.0),
            np.clip(self.f_w / 100.0, 0.0, 3.0),
            np.clip(self.f_h / 100.0, 0.0, 3.0),
            np.clip(conf, 0.0, 1.0),
            t_nd,
            np.clip(self.pitch / 0.5, -1.0, 1.0),
            np.clip(self.roll  / 0.5, -1.0, 1.0),
            self.a_prev[0], self.a_prev[1], self.a_prev[2], self.a_prev[3],
        ], dtype=np.float32)
        raw = dict(cx=self.cx, cy=self.cy, alpha=self.alpha, d_hat=self.f_d,
                   valid=int(self.valid), conf=conf, t_nodet=t_nd,
                   pitch_deg=math.degrees(self.pitch), roll_deg=math.degrees(self.roll),
                   true_dist=self.true_dist)
        return obs, raw


class IbvsActionTap(object):
    """Captures the live IBVS velocity commands (probe/record modes) so a_prev and
    the BC action labels are the REAL controller output, normalized by the caps."""
    def __init__(self, builder):
        self.b = builder
        rospy.Subscriber('/mavros/setpoint_raw/local', PositionTarget, self._cb, queue_size=1)
    def _cb(self, m):
        a = np.array([m.velocity.x, m.velocity.y, m.velocity.z, m.yaw_rate], dtype=np.float32)
        self.b.a_prev = np.clip(a / self.b.caps, -1.0, 1.0)


if _GYM:
    class DroneTrackingEnv(gym.Env):
        """Gymnasium wrapper. step() publishes body-frame velocities (the IBVS output
        interface). reward/terminated are Step-3/4 stubs; reset is reset-free for now
        (returns the current obs — target re-randomization comes in Step 3)."""
        metadata = {"render_modes": []}

        def __init__(self, ctrl_hz=20.0):
            super().__init__()
            self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.ob = ObsBuilder()
            self.dt = 1.0 / ctrl_hz
            self.pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
            self._msg = PositionTarget()
            self._msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            self._msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                                   PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                                   PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                                   PositionTarget.IGNORE_YAW)

        def step(self, action):
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            v = a * self.ob.caps
            self._msg.header.stamp = rospy.Time.now()
            self._msg.velocity.x, self._msg.velocity.y, self._msg.velocity.z = float(v[0]), float(v[1]), float(v[2])
            self._msg.yaw_rate = float(v[3])
            self.pub.publish(self._msg)
            self.ob.a_prev = a
            rospy.sleep(self.dt)
            obs, raw = self.ob.build()
            reward = 0.0            # Step 4 (Reward_Design.docx)
            terminated = False      # Step 4: collision / sustained-loss
            truncated = False       # Step 4: fixed episode time (Option A)
            return obs, reward, terminated, truncated, raw

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            obs, raw = self.ob.build()
            return obs, raw


# --------------------------- probe / record mains -----------------------------
def _run_probe(ob):
    tap = IbvsActionTap(ob)
    rospy.loginfo("[rl_env] PROBE mode — read-only; printing obs at 2 Hz. Ctrl-C to stop.")
    rate = rospy.Rate(2)
    hdr = " ".join(f"{n:>8s}" for n in OBS_NAMES)
    n = 0
    while not rospy.is_shutdown():
        obs, raw = ob.build()
        if n % 10 == 0:
            print("\n" + hdr + "   | d_hat  valid true_dist")
        print(" ".join(f"{v:8.3f}" for v in obs) +
              f"   | {raw['d_hat']:5.2f}  {raw['valid']}     {raw['true_dist']:5.2f}")
        n += 1
        rate.sleep()


def _run_record(ob):
    tap = IbvsActionTap(ob)
    out = os.path.expanduser(rospy.get_param("~out", "~/rl_demos/demo.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, 'w', newline='')
    wcsv = csv.writer(f)
    wcsv.writerow(['sim_time'] + OBS_NAMES +
                  ['act_vx','act_vy','act_vz','act_wz','valid','d_hat','true_dist'])
    rospy.loginfo("[rl_env] RECORD mode -> %s (20 Hz, read-only). Ctrl-C to stop.", out)
    rate = rospy.Rate(20)
    rows = 0
    while not rospy.is_shutdown():
        obs, raw = ob.build()
        a = ob.a_prev            # normalized IBVS action captured from the setpoint topic
        wcsv.writerow([f"{rospy.Time.now().to_sec():.3f}"] +
                      [f"{v:.5f}" for v in obs] + [f"{v:.5f}" for v in a] +
                      [raw['valid'], f"{raw['d_hat']:.3f}", f"{raw['true_dist']:.3f}"])
        rows += 1
        if rows % 600 == 0:
            f.flush(); rospy.loginfo("[rl_env] %d rows", rows)
        rate.sleep()
    f.close()
    rospy.loginfo("[rl_env] saved %d rows -> %s", rows, out)


if __name__ == "__main__":
    rospy.init_node("rl_env", anonymous=True)
    mode = rospy.get_param("~mode", "probe")
    ob = ObsBuilder()
    if mode == "record":
        _run_record(ob)
    else:
        _run_probe(ob)
