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
import os, sys, csv, math, time, threading
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
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

# 2026-08-18 (supervisor): EXPLICIT-RATE observation — the finite-difference rates
# dex/dey/dd carry the temporal info, so pitch/roll are DROPPED and frame-stacking
# is OFF (single frame). a_prev kept for action continuity + the smoothness reward.
OBS_NAMES = ["ex","ey","d_hat","dex","dey","dd","w","h","conf","t_nodet",
             "a_vx","a_vy","a_vz","a_wz"]
OBS_DIM = len(OBS_NAMES)

# Reward params (FYP/RL/reward/Reward_Design.docx). Defaults are START values, all
# rosparam-overridable. Design principles baked in here:
#  (1) BOUNDED, COMPARABLE SCALE (2026-08-24): every per-step term is clipped to ~[-1,1]
#      so no single term dominates the critic's target. The OLD reward left the out-of-band
#      distance penalty UNBOUNDED (−16.5/step at d=40 m) and terminals at −100/−150 (~100×
#      the per-step scale) — that scale gap made Q-values hard to fit and stalled/degraded
#      learning. Now per-step total ≈ [−2,+2]; terminals are the largest but only ~7× that.
#      NB: this CHANGES reward semantics → a replay buffer built with the old reward must NOT
#      be reused; train fresh (--scratch) after this change.
#  (2) per-frame P_lost is small (0.5); the real loss deterrent is the SUSTAINED-loss terminal.
#  (3) collision terminal (P_safe) > sustained-loss terminal (P_lost_final): when a loss looks
#      inevitable the agent must NEVER score better by RAMMING (−P_safe) than by losing it
#      (−P_lost_final). The 2 m safety bar in reward form.
#  (4) the centering bonus (+A) is paid ONLY on a valid detection, so a frozen (lost) frame
#      can't collect it. A tracked in-band step (+2) dominates either terminal.
REWARD_DEFAULTS = dict(
    A=1.0, sigma=0.5,          # centering (valid only): exp bump in [0,1], peaks +A when centered.
                               # sigma WIDENED 0.3→0.5 (2026-08-24 eval): at 0.3 the bump was ~0 for
                               # |ex|>0.5, so a target drifting to the frame edge got NO pull-back
                               # gradient before YOLO lost it (eval: 31% detection, mean|ex|=0.69).
                               # 0.5 gives centering gradient across the whole frame.
    w_d=0.5, band_lo=6.0, band_hi=7.0,   # band: +1 inside [6,7] m; linear penalty outside...
    band_pen_cap=1.0,          # ...clipped to −band_pen_cap so a far target can't drown out centering
    w_s=0.05, smooth_cap=1.0,  # smoothness −w_s·‖Δa‖², clipped to −smooth_cap
    w_approach=2.0, approach_cap=1.0,    # approach +w_approach·Δd when closing outside band, clip +approach_cap
    w_vel=0.05,                # anti-hover: small reward for any nonzero action magnitude
    w_alt=0.0, alt_cap=4.0,    # altitude-match (GT, always-on): −w_alt·min(|cz−tz|,alt_cap).
                               # DISABLED (w_alt=0, 2026-08-24): ey ALREADY carries the vertical
                               # "target above/below" signal and is now vision-gated + rewarded via
                               # centering, so ey should hold altitude on its own. Kept as a tunable
                               # crutch: re-enable (w_alt≈0.25) ONLY if the ey-alone test shows the
                               # policy still sinks (blind-recovery is the only thing ey can't do).
    P_lost=0.5,                # per-frame keep-in-view penalty (detection lost)
    d_min=2.0, P_safe=15.0,    # collision: d_true<d_min → terminal −P_safe (bounded; > loss terminal)
    loss_secs=5.0, P_lost_final=10.0,    # sustained loss>5s → terminal −P_lost_final
)

def compute_reward(p, a, prev_a, ex, ey, d_true, valid, t_lost, d_prev=None, alt_gap=None):
    """Shaped reward + terminal flag. p = params dict (see REWARD_DEFAULTS). GT (d_true,
    t_lost) is legal here — reward is training-only. Returns (reward, terminated, reason).

    VISION-FIRST (2026-08-24 root fix): centering, band AND approach are ALL paid ONLY on a
    valid detection. The only way to earn reward is to keep the target in the camera. This
    closes the loophole where the GT band reward paid for the right distance even while BLIND
    — which let the policy sink below the target (target leaves the top of frame → no ey
    signal → stuck low, blind, but still collecting distance credit). Now a blind step scores
    only −P_lost, so keeping the target framed (hence matching altitude + yaw) is non-optional.
    All per-step terms are BOUNDED to ~[-1,1] so the reward stays on one scale."""
    if valid:
        # centering: exp bump in [0,1], peaks +A when centered
        r_center = p['A'] * math.exp(-(ex*ex + ey*ey) / (p['sigma']**2))
        # band: +1 inside [6,7]; bounded linear penalty outside (clipped to −band_pen_cap)
        bcap = p.get('band_pen_cap', 1.0)
        if math.isnan(d_true):        r_band = 0.0
        elif d_true < p['band_lo']:   r_band = max(-bcap, -p['w_d'] * (p['band_lo'] - d_true))
        elif d_true > p['band_hi']:   r_band = max(-bcap, -p['w_d'] * (d_true - p['band_hi']))
        else:                         r_band = 1.0
        # approach: bounded positive signal for closing distance when outside band
        r_approach = 0.0
        if (d_prev is not None and not math.isnan(d_true) and not math.isnan(d_prev)
                and d_true > p['band_hi']):
            r_approach = min(p.get('approach_cap', 1.0), p['w_approach'] * max(0.0, d_prev - d_true))
    else:
        # BLIND: target not in the camera → NO centering/band/approach credit. GT distance must
        # not pay while the target is unseen, or the policy loiters at the right range blind.
        r_center = r_band = r_approach = 0.0
    # anti-hover + smoothness are action-shaping, applied every step.
    r_vel = p.get('w_vel', 0.0) * float(np.linalg.norm(np.asarray(a, dtype=np.float32)))
    da = np.asarray(a) - np.asarray(prev_a)
    r_smooth = max(-p.get('smooth_cap', 1.0), -p['w_s'] * float(np.dot(da, da)))
    r_lost = -p['P_lost'] if valid == 0 else 0.0
    # altitude-match (GT, ALWAYS-ON incl. blind): the recovery gradient that pulls the chaser
    # back to the target's altitude so the target re-enters the frame. alt_gap = |cz − tz|.
    r_alt = 0.0
    if alt_gap is not None and not math.isnan(alt_gap):
        r_alt = -p.get('w_alt', 0.0) * min(abs(alt_gap), p.get('alt_cap', 4.0))
    reward = r_center + r_band + r_approach + r_vel + r_smooth + r_lost + r_alt
    terminated = False; reason = ""
    if (not math.isnan(d_true)) and d_true < p['d_min']:
        reward -= p['P_safe']; terminated = True; reason = "collision"   # GT safety, even if blind
    elif t_lost > p['loss_secs']:
        reward -= p['P_lost_final']; terminated = True; reason = "lost"
    return reward, terminated, reason


class ObsBuilder(object):
    """Assembles the locked observation vector from the live ROS topics.
    Read-only: subscribes, never publishes."""

    def __init__(self):
        self.k          = float(rospy.get_param("~alpha_dist_k", 0.077))
        self.pitch_comp = float(rospy.get_param("~pitch_comp", 1.3))   # launch default
        self.rate_lpf   = float(rospy.get_param("~rate_lpf", 0.6))     # EMA on rates
        # zero_aprev: feed ZEROS into the 4 a_prev obs slots so the policy CANNOT see
        # its own last action -> removes the positive-feedback runaway channel (the 2026-
        # 08-17 SAC finding). self.a_prev still tracks the REAL last action for the
        # smoothness reward (that path is unaffected). Default False = v1 obs unchanged;
        # SAC v2 passes ~zero_aprev:=true and uses a clone RETRAINED with a_prev zeroed.
        self.zero_aprev = bool(rospy.get_param("~zero_aprev", False))
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
        # a_prev = the PREVIOUS action carried in the observation (policy action in
        # env mode; managed by the record loop in record mode). latest_cmd = the
        # freshest IBVS command captured by the tap — kept SEPARATE from a_prev so
        # the recorded label is never leaked into the observation (see _run_record).
        self.a_prev = np.zeros(4, dtype=np.float32)
        self.latest_cmd = np.zeros(4, dtype=np.float32)
        self.caps = np.array([float(rospy.get_param("~max_vx", 8.0)),
                              float(rospy.get_param("~max_vy", 1.2)),
                              float(rospy.get_param("~max_vz", 2.5)),
                              float(rospy.get_param("~max_wz", 0.5))], dtype=np.float32)
        # ground truth (record extras / reward / recovery scaffolding — NEVER in the obs)
        self.true_dist = float('nan')
        self.rel_w = (float('nan'), float('nan'), float('nan'))  # target−chaser, world XYZ
        self.chaser_yaw = 0.0                                    # chaser heading (world)
        self.chaser_alt = 0.0                                    # chaser ENU z (m), for altitude safety

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
        # exact model names, matching flight_logger.gazebo_cb ('iris' / 'target_iris')
        try:
            ic = m.name.index('iris'); it = m.name.index('target_iris')
        except ValueError:
            return
        try:
            c, t = m.pose[ic].position, m.pose[it].position
            self.true_dist = math.sqrt((c.x-t.x)**2 + (c.y-t.y)**2 + (c.z-t.z)**2)
            # GT relative vector (world) + chaser yaw — TRAINING-ONLY, used by _recover to
            # re-acquire a target lost in ANY direction (never enters the observation).
            self.rel_w = (t.x - c.x, t.y - c.y, t.z - c.z)
            self.chaser_alt = float(c.z)
            q = m.pose[ic].orientation
            self.chaser_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        except IndexError:
            pass

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
            conf = self.conf if not math.isnan(self.conf) else 0.0
        else:
            # Explicit-rate dropout (supervisor 2026-08-23): freeze rates at last known
            # values instead of zeroing. Network retains last velocity direction during
            # YOLO blackout. _prev kept so rates resume correctly on re-acquisition.
            conf = 0.0

        if self.t_last_det is None:
            t_since_raw = 999.0
        else:
            t_since_raw = (rospy.Time.now() - self.t_last_det).to_sec()
        t_nd = min(t_since_raw, 1.0)
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
            # pitch/roll REMOVED (explicit-rate obs, 2026-08-18). a_prev slots: zeroed
            # when ~zero_aprev; the real self.a_prev still drives the smoothness reward.
            *(np.zeros(4, dtype=np.float32) if self.zero_aprev else self.a_prev[:4]),
        ], dtype=np.float32)
        raw = dict(cx=self.cx, cy=self.cy, alpha=self.alpha, d_hat=self.f_d,
                   valid=int(self.valid), conf=conf, t_nodet=t_nd, t_since_raw=t_since_raw,
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
        self.b.latest_cmd = np.clip(a / self.b.caps, -1.0, 1.0)   # NOT a_prev (no label leak)


if _GYM:
    class DroneTrackingEnv(gym.Env):
        """Gymnasium wrapper. step() publishes body-frame velocities (the IBVS output
        interface). reward/terminated are Step-3/4 stubs; reset is reset-free for now
        (returns the current obs — target re-randomization comes in Step 3)."""
        metadata = {"render_modes": []}

        def __init__(self, ctrl_hz=20.0, episode_secs=None):
            super().__init__()
            self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.ob = ObsBuilder()
            self.dt = 1.0 / ctrl_hz
            # Episode length (Option A): truncate at max_episode_secs. RESET IS
            # RESET-FREE — no teleport of either drone (avoids the PX4 EKF instability
            # that teleports caused historically); the target keeps flying, an episode
            # is just a bounded time-window, and truncation is bootstrapped (not terminal).
            self.max_episode_secs = float(episode_secs if episode_secs is not None
                                          else rospy.get_param("~episode_secs", 100.0))
            self._ep_t0 = None
            self._ep_steps = 0
            # first-reset handoff window: HOVER (keepalive) for this long so IBVS can be
            # killed WITHOUT the policy ever stepping against it. The overlap-fight — the
            # RL policy stepping aggressively while a live IBVS also commands — drove the
            # target off-frame in every failed run; hovering during the kill removes it.
            self._first_reset = True
            self._handoff_wait = float(rospy.get_param("~handoff_wait", 6.0))
            self._prev_d_true = float('nan')   # for approach reward
            # reward params (rosparam-overridable; defaults = REWARD_DEFAULTS)
            self._rp = {k: float(rospy.get_param("~rew_" + k, v))
                        for k, v in REWARD_DEFAULTS.items()}
            self.pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
            self._msg = PositionTarget()
            self._msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # world-z avoids tilt-drift
            self._msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                                   PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                                   PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                                   PositionTarget.IGNORE_YAW)
            # control-rate clock: rospy.Rate ABSORBS the SAC gradient-step time so the
            # effective loop stays ~20 Hz (a bare sleep(dt) would drift to ~13-15 Hz once
            # a gradient step is added on top of every env step).
            self._rate = rospy.Rate(ctrl_hz)
            # OFFBOARD keepalive: PX4 drops OFFBOARD after ~0.5 s without a setpoint.
            # A daemon thread republishes the last command at 20 Hz so the gaps during
            # SAC gradient steps, checkpoint/replay-buffer saves (seconds) and the
            # reset()-recovery wait never starve the stream. step()/reset() only UPDATE
            # self._msg (under the lock); the thread guarantees continuity.
            self._msg_lock = threading.Lock()
            self._alive = True
            # RL always runs with SKIP_IBVS=1 — no IBVS on the setpoint topic.
            # Start keepalive immediately so hover setpoints are published from env
            # creation onward. Without this, the gap between takeoff_both.py exiting
            # and reset() being called (~2-10s) causes PX4 to time out OFFBOARD and
            # land the drone before a single training step runs.
            self._ka_active = True
            self._ka = threading.Thread(target=self._keepalive, daemon=True)
            self._ka.start()

        def _keepalive(self):
            r = rospy.Rate(20)
            while self._alive and not rospy.is_shutdown():
                if self._ka_active:
                    with self._msg_lock:
                        self._msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
                        self.pub.publish(self._msg)
                r.sleep()

        def _set_cmd(self, vx, vy, vz, wz):
            """Atomically update + publish the setpoint the keepalive thread holds."""
            with self._msg_lock:
                self._msg.velocity.x = float(vx); self._msg.velocity.y = float(vy)
                self._msg.velocity.z = float(vz); self._msg.yaw_rate = float(wz)
                self._msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
                self.pub.publish(self._msg)      # immediate; thread republishes between steps

        def _set_world_cmd(self, vx_ned, vy_ned, vz_ned, wz):
            """Publish a one-shot FRAME_LOCAL_NED setpoint (used by _recover only).
            Bypasses EKF yaw dependency: _recover computes velocities in ENU world frame
            and converts to NED; PX4 applies them directly without body→world rotation."""
            m = PositionTarget()
            m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                           PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                           PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                           PositionTarget.IGNORE_YAW)
            m.velocity.x = float(vx_ned); m.velocity.y = float(vy_ned)
            m.velocity.z = float(vz_ned); m.yaw_rate = float(wz)
            m.header.stamp = rospy.Time(0)
            self.pub.publish(m)

        def close(self):
            self._alive = False

        def step(self, action):
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            prev_a = np.asarray(self.ob.a_prev, dtype=np.float32).copy()   # a_(t-1) for smoothness
            d_prev = self._prev_d_true                                       # for approach reward
            v = a * self.ob.caps
            # Convert body-frame action → world LOCAL frame so vz is world-z (avoids tilt-drift).
            # PROBE-VERIFIED 2026-08-24: setpoint_raw/local is ENU — velocity.x→world-x,
            # velocity.y→world-y, velocity.z→Up. Map body (fwd=v[0], right=v[1]) to world;
            # chaser_yaw is the ENU heading (CCW from world-x). The old code put the world-y
            # component into velocity.x (x/y swapped) → chaser drove ~90° off, flew away.
            yaw = self.ob.chaser_yaw
            c_y, s_y = math.cos(yaw), math.sin(yaw)
            vx_ned = v[0] * c_y + v[1] * s_y   # velocity.x (world-x) = fwd·cos + right·sin
            vy_ned = v[0] * s_y - v[1] * c_y   # velocity.y (world-y) = fwd·sin − right·cos
            # Altitude envelope [8m floor, 22m ceiling] — both clamp the policy action.
            # SIGN CONVENTION (verified live 2026-08-24): /mavros/setpoint_raw/local is ENU
            # on the ROS side — MAVROS converts ENU→NED internally, so vz POSITIVE = UP and
            # vz NEGATIVE = DOWN. The old code assumed NED (positive=down) and inverted every
            # altitude clamp: the ceiling forced vz=+2.5 to "descend" → drone climbed to 378 m.
            ALT_FLOOR = 11.0   # raised 8→11 (2026-08-24): target orbits at ~14 m (its Z_FLOOR=12);
                               # an 8 m floor let the chaser sink ~6 m below → target left the top
                               # of the camera FOV → no ey signal → stuck low. 11 m keeps the chaser
                               # in the band where the target stays framable and ey stays informative.
            ALT_CEIL  = 22.0
            alt = self.ob.chaser_alt
            if alt < ALT_FLOOR:
                climb = min(self.ob.caps[2], 1.5 * max(1.0, ALT_FLOOR - alt))    # +vz = UP
                if v[2] < climb:
                    v = v.copy(); v[2] = float(climb)
                rospy.logwarn_throttle(2.0, "[rl_env] alt-floor: z=%.2fm forcing vz=%.2f", alt, v[2])
            elif alt > ALT_CEIL:
                descend = max(-self.ob.caps[2], -1.0 * max(1.0, alt - ALT_CEIL))  # -vz = DOWN
                if v[2] > descend:
                    v = v.copy(); v[2] = float(descend)
                rospy.logwarn_throttle(2.0, "[rl_env] alt-ceiling: z=%.2fm forcing vz=%.2f", alt, v[2])
            self._set_cmd(vx_ned, vy_ned, v[2], v[3])
            self.ob.a_prev = a                       # closed loop: a_prev = own action
            self._rate.sleep()                       # paced 20 Hz (absorbs gradient-step time)
            obs, raw = self.ob.build()
            self._prev_d_true = raw['true_dist']     # update for next step
            self._ep_steps += 1
            elapsed = (rospy.Time.now() - self._ep_t0).to_sec() if self._ep_t0 else 0.0
            # altitude gap = |target_z − chaser_z| from GT rel vector (training-only reward input)
            _rz = self.ob.rel_w[2]
            _alt_gap = abs(_rz) if not math.isnan(_rz) else None
            reward, terminated, reason = compute_reward(
                self._rp, a, prev_a, float(obs[0]), float(obs[1]),
                raw['true_dist'], raw['valid'], raw['t_since_raw'], d_prev=d_prev, alt_gap=_alt_gap)
            truncated = (not terminated) and (elapsed >= self.max_episode_secs)  # Option A
            if alt < 10.0:
                rospy.logwarn_throttle(1.0, "[rl_env] LOW ALT in step: z=%.2fm vz_cmd=%.2f ep_step=%d",
                                       alt, v[2], self._ep_steps)
            if terminated or truncated:
                # Stop any descending action before SB3's gradient step runs.
                # The keepalive publishes the last _msg for the duration of learn(); hover avoids drift.
                self._set_cmd(0, 0, 0, 0)
            if terminated:                        # diagnostic: which terminal + where
                rospy.loginfo("[rl_env] TERMINAL %s after %d steps: d_true=%.2f t_lost=%.1f ex=%.2f ey=%.2f alt=%.2f",
                              reason, self._ep_steps, raw['true_dist'], raw['t_since_raw'],
                              float(obs[0]), float(obs[1]), alt)
            raw['elapsed'] = elapsed; raw['ep_steps'] = self._ep_steps
            raw['reward'] = reward; raw['term_reason'] = reason
            return obs, reward, terminated, truncated, raw

        def _request_offboard(self):
            """Request OFFBOARD mode via MAVROS. Needed when SKIP_IBVS=1: without a
            prior setpoint stream, PX4 drops OFFBOARD within 0.5s of last setpoint,
            so by the time reset() is called the drone is in HOLD. Stream setpoints
            first for 1s, THEN request mode change (PX4 rejects the switch if no
            recent setpoints exist)."""
            try:
                # pre-stream setpoints so PX4 accepts the mode request
                r = rospy.Rate(20)
                for _ in range(20):   # 1s @ 20 Hz
                    self._set_cmd(0, 0, 0, 0)
                    r.sleep()
                set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                resp = set_mode(custom_mode='OFFBOARD')
                if resp.mode_sent:
                    rospy.loginfo("[rl_env] OFFBOARD mode requested OK")
                else:
                    rospy.logwarn("[rl_env] OFFBOARD request returned mode_sent=False")
            except Exception as e:
                rospy.logwarn("[rl_env] OFFBOARD request failed: %s", e)

        def reset(self, seed=None, options=None):
            # RESET-FREE: no teleport. But EVERY episode must start from a VALID, SAFE
            # tracking state. Unlike the BC teacher, a learning policy ends episodes on a
            # terminal (collision / sustained-loss) — so without recovery the next
            # episode would start already-lost or already-too-close and instantly
            # re-terminate, flooding the replay buffer with 1-step −P episodes. _recover()
            # holds a hover, waits for a detection, gently retreats if too close, and
            # slow yaw-sweeps if still lost. It is TRAINING SCAFFOLDING (IBVS SEARCH is
            # offline during RL) that runs ONLY between episodes → it never enters an
            # observation the policy learns from, so the RL controller stays pure.
            super().reset(seed=seed)
            self._ka_active = True            # training starts now -> take over the stream
            if self._first_reset:
                self._first_reset = False
                # Re-request OFFBOARD when SKIP_IBVS=1: OFFBOARD drops within 0.5s of last
                # setpoint; with no IBVS, the mode was lost by the time reset() is called.
                # The service call streams setpoints first (PX4 requires a live stream to
                # accept the OFFBOARD request), then switches the mode.
                self._request_offboard()
                # hold a pure hover while IBVS is killed — the policy must NOT step yet
                rospy.loginfo("[rl_env] HANDOFF: hovering %.1fs — kill IBVS now.", self._handoff_wait)
                t0 = rospy.Time.now()
                while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < self._handoff_wait:
                    self._set_cmd(0, 0, 0, 0)
                    rospy.sleep(0.05)
            self._recover()
            # Always reset the detection timer at episode start. Two failure modes:
            # (1) t_last_det=None (never detected): t_since_raw=999 → instant "lost" terminal.
            # (2) t_last_det=old (from prior episode that ended lost): episode 2+ starts
            #     with t_since_raw already > loss_secs → 1-step episode loop.
            # Resetting to now gives EVERY episode loss_secs wall-seconds to acquire.
            _det_age = (float('inf') if self.ob.t_last_det is None
                        else (rospy.Time.now() - self.ob.t_last_det).to_sec())
            if _det_age > 1.0:
                self.ob.t_last_det = rospy.Time.now()
                rospy.loginfo("[rl_env] reset t_last_det at episode start (was %.1fs old)", _det_age)
            self.ob.a_prev = np.zeros(4, dtype=np.float32)   # fresh smoothness baseline
            self._prev_d_true = float('nan')                  # reset approach reward baseline
            self._ep_t0 = rospy.Time.now()
            self._ep_steps = 0
            obs, raw = self.ob.build()
            return obs, raw

        def _rearm_and_takeoff(self, target_alt=14.0):
            """Re-arm and climb back to target_alt using POSITION setpoints.

            Velocity setpoints work from ground when armed+OFFBOARD (verified live).
            The keepalive is already streaming at 20 Hz (_ka_active=True), so OFFBOARD
            is already satisfied. Sequence: OFFBOARD → arm → climb velocity → wait.
            """
            rospy.logwarn("[rl_env] CRASH RECOVERY: chaser at alt=%.2fm — re-arming", self.ob.chaser_alt)
            r = rospy.Rate(20)

            # 1. Request OFFBOARD (keepalive is already streaming hover setpoints).
            try:
                set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                resp = set_mode(custom_mode='OFFBOARD')
                rospy.loginfo("[rl_env] OFFBOARD re-request: mode_sent=%s", resp.mode_sent)
            except Exception as e:
                rospy.logwarn("[rl_env] OFFBOARD re-request failed: %s", e)

            # 2. Arm.
            try:
                arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
                resp = arm_srv(True)
                rospy.loginfo("[rl_env] arm: success=%s", resp.success)
            except Exception as e:
                rospy.logwarn("[rl_env] arm failed: %s", e)

            # 3. Climb: update keepalive msg to vz=+1.5 (ENU via MAVROS: positive = upward).
            with self._msg_lock:
                self._msg.velocity.x = 0.0
                self._msg.velocity.y = 0.0
                self._msg.velocity.z = 1.5
                self._msg.yaw_rate = 0.0
                self._msg.header.stamp = rospy.Time(0)

            # 4. Wait for altitude (max 40 s).
            t0 = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < 40.0:
                if self.ob.chaser_alt >= target_alt - 2.0:
                    rospy.loginfo("[rl_env] crash recovery done: alt=%.1fm", self.ob.chaser_alt)
                    break
                r.sleep()
            else:
                rospy.logwarn("[rl_env] crash recovery timed out; alt=%.1fm", self.ob.chaser_alt)

            # 5. Hover.
            with self._msg_lock:
                self._msg.velocity.z = 0.0
                self._msg.header.stamp = rospy.Time(0)

        def _recover(self, timeout=15.0):
            """Establish a YOLO-detectable start before each episode (see reset()).
            GT-DRIVEN scaffolding: drives chaser to within 12 m of target (wide enough
            to be achievable on T4 orbit at 2 m/s — the old [5.5,7.5]m band settle was
            geometrically impossible on an orbiting target and caused every recovery to
            time out, leaving the drone at d=15m pointing the wrong way).
            Phase 2 yaw-aligns toward GT target and waits for YOLO confirmation (8s).
            Falls back to hover/yaw-scan if GT is missing."""
            # If drone crashed (altitude < 2m), re-arm and take off before proceeding.
            if self.ob.chaser_alt < 2.0:
                self._rearm_and_takeoff()
            # Warn if we entered _recover() at a dangerously low altitude (crash investigation).
            if self.ob.chaser_alt < 10.0:
                rospy.logwarn("[rl_env] _recover entered at LOW alt=%.2fm — policy descended into floor", self.ob.chaser_alt)
            lo, hi = self._rp['band_lo'], self._rp['band_hi']
            d_star = 0.5 * (lo + hi)
            SETTLE_DIST = 12.0   # accept "close enough" rather than in-band; YOLO detects ≤12m
            SETTLE_MIN  = 5.5    # but NEVER settle inside near-collision range — retreat if closer.
                                 # Without this, after a collision terminal (d≈2m) _recover saw
                                 # 2<12 → "settled" → exited → step 1 re-collided → 1-step loop that
                                 # deadlocked online training (685 collisions, ep_len≈5). Now a too-
                                 # close start falls through to the reposition block (mag<0 → retreat).
            r = rospy.Rate(20); t0 = rospy.Time.now(); settle = 0
            while not rospy.is_shutdown():
                el = (rospy.Time.now() - t0).to_sec()
                if el > timeout:
                    rospy.logwarn("[rl_env] _recover timed out (%.0fs); proceeding to yaw-align", timeout)
                    break
                td = self.ob.true_dist
                rel = self.ob.rel_w
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, dz = rel
                    rng = math.hypot(dx_w, dy_w)
                    yaw = self.ob.chaser_yaw
                    dbeta = math.atan2(dy_w, dx_w) - yaw
                    dbeta = math.atan2(math.sin(dbeta), math.cos(dbeta))
                    # wide settle: accept SETTLE_MIN ≤ d < SETTLE_DIST (T4 orbit can't hold
                    # [5.5,7.5]m). Too-close (d<SETTLE_MIN) is NOT framed → retreat below.
                    d_ref = td if not math.isnan(td) else rng
                    framed_gt = (SETTLE_MIN <= d_ref < SETTLE_DIST)
                else:
                    dbeta = 0.0; rng = float('nan'); framed_gt = False
                framed = framed_gt or (self.ob.valid and not math.isnan(td)
                                       and lo - 0.4 <= td <= hi + 0.4)
                if framed:
                    self._set_cmd(0, 0, 0, 0); settle += 1
                    if settle >= 3:          # ~0.15 s at distance (reduced: yaw no longer gating)
                        break
                    r.sleep(); continue
                settle = 0
                # --- GT reposition: LOCAL_NED via keepalive (correct fix) ---
                # Must use LOCAL_NED so vz is always world-z regardless of drone tilt.
                # Body-frame (BODY_NED) at 4 m/s forward causes ~15-20° bank → vz=0 in
                # body frame has a downward world component → chaser descends and crashes.
                # Approach: update _msg directly to LOCAL_NED so the keepalive thread
                # continuously republishes the approach command (no cancellation).
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, dz = rel
                    if rng > 1e-3:
                        mag = max(-4.0, min(4.0, 1.5 * (rng - d_star)))
                        vx_ned = mag * dx_w / rng   # velocity.x → world-x (dx_w); ENU probe-verified
                        vy_ned = mag * dy_w / rng   # velocity.y → world-y (dy_w)
                    else:
                        vx_ned = vy_ned = 0.0
                    vz_ned = max(-2.5, min(2.5, 0.8 * dz))   # ENU: +vz=UP; dz=ENU up → climb toward target
                    # Altitude floor in _recover(): identical to the one in step().
                    # Critical: without this, a chaser that DROPPED below target altitude keeps
                    # sinking (vz toward target < 0) for the full 15-s timeout, crashing.
                    _alt = self.ob.chaser_alt
                    if _alt < 11.0:   # match step() ALT_FLOOR (raised 8→11, keeps target framable)
                        _climb = min(2.5, 1.5 * max(1.0, 11.0 - _alt))   # +vz = UP
                        if vz_ned < _climb:
                            vz_ned = _climb
                            rospy.logwarn_throttle(2.0,
                                "[rl_env] _recover alt-floor: z=%.2fm forcing vz_ned=%.2f",
                                _alt, vz_ned)
                    wz = max(-1.0, min(1.0, 1.5 * dbeta))   # ENU yaw_rate = CCW+; turn toward target
                    with self._msg_lock:
                        self._msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                        self._msg.velocity.x = float(vx_ned)
                        self._msg.velocity.y = float(vy_ned)
                        self._msg.velocity.z = float(vz_ned)
                        self._msg.yaw_rate = float(wz)
                        self._msg.header.stamp = rospy.Time(0)
                        self.pub.publish(self._msg)
                else:
                    self._set_cmd(0, 0, 0, 0.4 if el > 1.0 else 0.0)
                r.sleep()

            # --- Phase 2: yaw-align + YOLO wait (max 5s) ---
            yaw_t0 = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - yaw_t0).to_sec() < 5.0:
                if self.ob.valid:
                    rospy.loginfo("[rl_env] _recover: YOLO confirmed after yaw-align")
                    break
                rel = self.ob.rel_w
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, _ = rel
                    yaw = self.ob.chaser_yaw
                    dbeta = math.atan2(dy_w, dx_w) - yaw
                    dbeta = math.atan2(math.sin(dbeta), math.cos(dbeta))
                    wz = max(-1.0, min(1.0, 1.5 * dbeta))   # ENU yaw_rate = CCW+; turn toward target
                    self._set_cmd(0, 0, 0, wz)   # yaw only
                else:
                    self._set_cmd(0, 0, 0, 0.4)  # no GT → slow spin
                r.sleep()
            if not self.ob.valid:
                rospy.logwarn("[rl_env] _recover: YOLO not detecting after yaw-align; starting anyway")
            self._set_cmd(0, 0, 0, 0)   # LOCAL_NED zeros = world-frame hover


# --------------------------- policy runner (Step 2) ---------------------------
def _run_policy(ob, bc_path, run_secs):
    """CLOSED-LOOP: the BC policy flies the drone (replaces IBVS). Publishes body
    velocities at 20 Hz, a_prev = its OWN last action, frame-stack N=4 built here.
    This is the copycat test — a pure a_prev-echo would drift immediately.
    NB: no external safety filter (per design). A test-only hover-on-lost guard
    prevents a runaway if the target is lost > 3 s; it is NOT part of the controller."""
    import torch
    from collections import deque
    from rl_train_bc import BCPolicy
    ck = torch.load(bc_path, map_location='cpu', weights_only=False)
    net = BCPolicy(obs_dim=64); net.load_state_dict(ck['state_dict']); net.eval()
    caps = np.array(ck['caps'], dtype=np.float32)
    pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
    msg = PositionTarget(); msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
    msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                     PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                     PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                     PositionTarget.IGNORE_YAW)
    rospy.loginfo("[rl_env] POLICY mode — BC (%s) flying for %.0fs. No external filter.",
                  os.path.basename(bc_path), run_secs)
    r0 = rospy.Rate(20)
    while not rospy.is_shutdown() and not ob.valid:   # wait for first detection
        r0.sleep()
    stack = deque([ob.build()[0]] * 4, maxlen=4)
    a = np.zeros(4, dtype=np.float32)
    rate = rospy.Rate(20); t0 = rospy.Time.now(); n = 0; lost_since = None
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < run_secs:
        ob.a_prev = a                          # closed loop: a_prev = own last action
        obs, raw = ob.build()
        stack.append(obs)
        x = np.concatenate(stack).astype(np.float32)[None]      # (1,64) oldest..newest
        with torch.no_grad():
            a = net(torch.from_numpy(x))[0].numpy()
        # test-only safety (NOT the controller): hover if target lost > 3 s
        if ob.valid:
            lost_since = None
        elif lost_since is None:
            lost_since = rospy.Time.now()
        if lost_since is not None and (rospy.Time.now() - lost_since).to_sec() > 3.0:
            v = np.zeros(4, dtype=np.float32); tag = "LOST->HOVER"
        else:
            v = np.clip(a, -1.0, 1.0) * caps; tag = ""
        msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
        msg.velocity.x, msg.velocity.y, msg.velocity.z = float(v[0]), float(v[1]), float(v[2])
        msg.yaw_rate = float(v[3]); pub.publish(msg)
        if n % 20 == 0:                        # ~1 Hz status
            print(f"t+{(rospy.Time.now()-t0).to_sec():4.0f}s  ex{obs[0]:+.3f} ey{obs[1]:+.3f}"
                  f"  d_hat{raw['d_hat']:5.2f} true{raw['true_dist']:5.2f}"
                  f"  v[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f},{v[3]:+.3f}] {tag}", flush=True)
        n += 1; rate.sleep()
    for _ in range(10):                        # leave a clean hover (avoid failsafe jerk)
        msg.header.stamp = rospy.Time(0)       # Time(0) → MAVROS uses FCU time
        msg.velocity.x = msg.velocity.y = msg.velocity.z = 0.0; msg.yaw_rate = 0.0
        pub.publish(msg); rate.sleep()
    rospy.signal_shutdown("policy run done")


# --------------------------- probe / record mains -----------------------------
def _run_probe(ob):
    tap = IbvsActionTap(ob)
    secs = float(rospy.get_param("~probe_secs", 12.0))   # self-terminate (clean flush)
    rospy.loginfo("[rl_env] PROBE mode — read-only; obs at 2 Hz for %.0f s.", secs)
    rate = rospy.Rate(2)
    hdr = " ".join(f"{n:>8s}" for n in OBS_NAMES)
    n = 0; t0 = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < secs:
        ob.a_prev = ob.latest_cmd            # display the live IBVS command in a_* cols
        obs, raw = ob.build()
        if n % 10 == 0:
            print("\n" + hdr + "   | d_hat  valid true_dist", flush=True)
        print(" ".join(f"{v:8.3f}" for v in obs) +
              f"   | {raw['d_hat']:5.2f}  {raw['valid']}     {raw['true_dist']:5.2f}", flush=True)
        n += 1
        rate.sleep()
    rospy.signal_shutdown("probe done")


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
    prev = np.zeros(4, dtype=np.float32)     # a_(t-1): the PREVIOUS IBVS command
    while not rospy.is_shutdown():
        ob.a_prev = prev                     # obs carries a_(t-1); NOT the label -> no leak
        obs, raw = ob.build()
        label = ob.latest_cmd.copy()         # a_t: the CURRENT command = the decision to imitate
        wcsv.writerow([f"{rospy.Time.now().to_sec():.3f}"] +
                      [f"{v:.5f}" for v in obs] + [f"{v:.5f}" for v in label] +
                      [raw['valid'], f"{raw['d_hat']:.3f}", f"{raw['true_dist']:.3f}"])
        prev = label                         # advance a_(t-1)
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
    elif mode == "policy":
        _run_policy(ob,
                    os.path.expanduser(rospy.get_param("~bc", "~/fyp/rl/models/bc_policy_v2.pth")),
                    float(rospy.get_param("~run_secs", 30.0)))
    else:
        _run_probe(ob)
