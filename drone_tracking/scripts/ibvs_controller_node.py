#!/usr/bin/env python3
"""
ibvs_controller_node.py  —  v6.30  (time-bound SEARCH heading latch)
=======================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v6.30 — time-bound SEARCH heading latch, light single-knob form (2026-06-19):
  v6.29 latch won on deterministic escape (T7) but backfired on reactive
  evasion (T9): committing to the loss-instant heading FOREVER flies the
  chaser the wrong way when the target reverses mid-gap. Fix (deliberately
  minimal): scale the latched heading by ONE smooth decay weight
  w(t)=exp(-t/tau), t = time since loss. w=1 at the loss instant and fades
  smoothly toward 0, so the heading is trusted early (helps T7) and abandoned
  before a mid-gap reverse can be committed to (protects T9). One knob:
  ~search_tau (default 1.0 s; larger = trust the heading longer). One safety:
  the per-step heading extrapolation is clamped to SEARCH_STEP_CLAMP_PX so a
  large v0 can't slam the predicted position to the frame edge. The Stage-2
  sweep center is recomputed each tick from the decayed v0, so it too fades to
  a neutral sweep. SEARCH-ONLY and gated by ~search_latch: with the flag off,
  svx=live _kf_vx=0 (Kalman-zeroed by 1.5 s) -> behavior is byte-for-byte
  v6.28; HOLD/APPROACH command output is untouched; Config 1 has no
  kalman_velocity -> v0=0 -> graceful blind search.

v6.29 — SEARCH loss-instant velocity latch (2026-06-19):
  Root cause found: the v6.22 "velocity-predicted SEARCH" always read ZERO
  velocity. The Kalman hard-zeroes its velocity state at max_dropout=30
  frames = 1.5 s of dropout, but IBVS only enters SEARCH at
  detection_timeout = 3.0 s after loss and reads _kf_vx THEN — by which time
  it is 0. So _search_base_cwz=0, _pred_cx froze at the last position, and
  Stage 2 was a blind ±30° sweep around heading 0. Offline damping/Q/R/
  standoff sweeps were all flat because none of them touch this.
  Fix: latch v0=(_kf_vx,_kf_vy) from the LAST tracking frame (while the
  estimate is still valid) and use that frozen heading for the Stage-1
  position extrapolation and the Stage-2 sweep center, plus extrapolate the
  last-known position forward by v0 over the APPROACH->SEARCH gap. Gated by
  ~search_latch (default True). SEARCH-ONLY: HOLD/APPROACH command output is
  byte-for-byte unchanged (the latch update never feeds the control law);
  with the flag off, behavior is identical to v6.28 (svx = live _kf_vx = 0 at
  3 s). Config 1 (no kalman_velocity) keeps v0=0 -> graceful blind-search
  fallback. Logs v0 at loss and at SEARCH start for verification.

v6.28 — SEARCH recovery speeds (2026-06-15):
  Phantom-lock / T7 prep: SEARCH ran slower (Stage 1 1.0, Stage 2 2.0 m/s)
  than the fastest target (T3/T7 = 3.5 m/s), so once the target was lost the
  separation diverged and the 2-stage search could never recover. Raised
  SEARCH speeds to Stage 1 2.5, Stage 2 4.5 m/s (Stage 2 = max_vx; SEARCH
  cvx is published un-clamped) so search can out-run a fleeing target. Stage
  logic, yaw sweep, Kalman prediction and the emergency guard are unchanged.

v6.27 — pursuit speed headroom (2026-06-11):
  Stress-trio T3 (straight, target 3.5 m/s): HOLD 9.8%, separation grew
  past 20 m, watchdog abort at 57 s sim — chaser max_vx (3.5) EQUALLED the
  target speed, so closure was geometrically impossible. Structural, not
  tunable by gains. max_vx 3.5 -> 4.5 (~30% speed advantage), now the
  ~max_vx rosparam. PX4 side checked: MPC_XY_VEL_MAX = 12.0 m/s (build
  default, no init.d-posix override) — 4.5 is not clipped. Intended side
  effect: the v6.26 emergency brake commands vx = -max_vx while engaged,
  so brake authority rises to -4.5 too. max_vx_retreat (0.50) deliberately
  NOT changed here — decide after re-run emerg data. Control law untouched.

v6.26 — P1 chaser-target collision guard (2026-06-11):
  Collision forensics (T2 z5 seed42 run): the chaser closed from 2.1 m to
  0.40 m over ~6 s with cmd_vx PINNED at -0.50 the whole time — the brake
  branch was working but is clamped by max_vx_retreat=0.50 m/s, so a 1 m/s
  target closing on the chaser out-runs the brake. Kalman was NOT a main
  contributor (3 collapse-rejection rows near the peak; PRED bridging
  behaved correctly).

  Guard (sits ABOVE the control law — normal-region behavior unchanged):
    ALPHA_EMERGENCY      = 0.033  (~alpha_emergency rosparam)
      chosen from data: 3.6x the healthy-HOLD alpha ceiling (max 0.0091
      across T1/T2 healthy HOLD), ~2x the HOLD band ceiling
      (alpha_star+ea_hold = 0.0167), 10x below the collision peak (0.353);
      crossed 5.2 s before closest approach in the recorded collision.
    ALPHA_EMERGENCY_EXIT = 0.7 x ALPHA_EMERGENCY  (hysteresis, no chatter)
  While engaged: vx = -max_vx (full braking; bypasses vel_smooth AND the
  max_vx_retreat clamp — smoothing/clamping a brake defeats it); vy/vz/wz
  keep tracking. Target lost while engaged -> release, normal SEARCH logic
  (never brake blind). Engaged state on /drone_tracking/emergency_brake
  (flight_logger 'emerg' column). Identical in raw/kalman modes
  (ablation-safe: uses the same alpha the controller already consumes).

v6.22 — SEARCH phase v2:
  Problem: SEARCH at 0.3 m/s could not recover from separations >30m
           (target moves at 2-4 m/s, chaser at 0.3 m/s → gap only grows).

  Solution: 2-stage escalating search using Kalman velocity prediction:

  Stage 1 (0–3 s):  1.0 m/s forward.
    Direction: yaw/vz toward the PREDICTED target position, extrapolated
    from the last known position + Kalman velocity state (vx, vy px/s).
    This is far more informative than "last_cx" alone because the
    Kalman velocity already encodes where the target was GOING.

  Stage 2 (3 s+):   2.0 m/s forward.
    Direction: slow ±30° yaw sweep (sin wave, ~16 s period) centred on
    the velocity-derived heading at the moment of loss.
    This covers cases where the target changed direction after loss.

  On re-detection in either stage → APPROACH with ramp (unchanged).

  Other v6.22 changes:
    min_altitude_safe:  1.0 → 13.0 m  (SEARCH fallback climb target,
                        matches new Z_FLOOR=12m from target_mover v10.3)
    Subscribes to /drone_tracking/kalman_velocity (new M9.6 topic)

v6.21 changes (preserved):
  K_far=35, Kd_a=150, smooth=0.15, Kp_z=2.5, pitch_comp=0.4,
  directional SEARCH memory (last_cx/cy)

v6.19 changes (preserved):
  alpha-rate feedforward vx, max_vx=3.0, dead zone ±0.002
"""

import rospy, math
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import Bool, String

BODY_VEL_TYPE_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW)

class IBVSController:
    def __init__(self):
        rospy.init_node('ibvs_controller_node')
        self.USE_PPO = rospy.get_param("~use_ppo", True)
        self.detection_source = rospy.get_param("~detection_source", "kalman")  # 'kalman' or 'raw'
        self.img_w=640.; self.img_h=480.
        self.img_cx=self.img_w/2.; self.img_cy=self.img_h/2.
        self.area_norm=self.img_w*self.img_h
        self.pitch_compensation_gain=0.4
        self.x_star=0.; self.y_star=0.; self.alpha_star=0.0067; self.lam=0.5

        # ── Distance gains ────────────────────────────────────────────
        self.K_far  = 35.0
        self.K_near = 6.0
        self.Kd_a   = 150.0
        self.ff_max = 1.5
        self.DEAD_ZONE = 0.002

        # Y / Z / yaw PID
        # M10.3: Kp_y / Kp_wz rosparam-overridable for the angular-tracking gain
        # sweep. Defaults = current hardcoded values -> a plain run is byte-for-
        # byte baseline. Kd_y/Kd_wz stay fixed; raising Kp alone lowers the
        # damping ratio, so the sweep gates on a yaw-oscillation (zero-crossing)
        # disqualifier. No treatment gain is baked into the file.
        self.Kp_y=float(rospy.get_param("~Kp_y",1.8)); self.Ki_y=0.05; self.Kd_y=0.3
        # M10.3 vertical-channel probe: Kp_z rosparam-overridable (default =
        # current 3.0 -> plain run is byte-for-byte baseline, same discipline as
        # Kp_wz/Kp_y). Kd_z stays fixed; raising Kp_z alone lowers the damping
        # ratio, so the sweep gates on an ey zero-crossing (oscillation) guard.
        self.Kp_z=float(rospy.get_param("~Kp_z",3.0)); self.Ki_z=0.04; self.Kd_z=0.5
        self.Kp_wz=float(rospy.get_param("~Kp_wz",0.9)); self.Ki_wz=0.; self.Kd_wz=0.15

        # Velocity limits
        # v6.27: max_vx 3.5 -> 4.5 (~max_vx rosparam). T3 proved 3.5 gives
        # ZERO closure on a 3.5 m/s target; interception needs a speed
        # advantage (~30%). PX4 MPC_XY_VEL_MAX=12 (build default) won't clip.
        self.max_vx=float(rospy.get_param("~max_vx",4.5)); self.max_vx_retreat=0.50
        # M10.3: max_vz rosparam-overridable (default = current 1.5 -> baseline).
        # cell D lever if the vz saturation pre-check shows the cap is binding.
        self.max_vy=1.20; self.max_vz=float(rospy.get_param("~max_vz",1.5)); self.max_wz=0.5

        # v6.26: P1 emergency brake guard (override above the control law)
        self.ALPHA_EMERGENCY      = float(rospy.get_param("~alpha_emergency", 0.033))
        self.ALPHA_EMERGENCY_EXIT = float(rospy.get_param(
            "~alpha_emergency_exit", 0.7*self.ALPHA_EMERGENCY))
        self.emergency_engaged = False
        self._emerg_count = 0

        # v6.22: min_altitude_safe raised to match new Z_FLOOR=12m
        self.min_altitude_safe=13.0; self.alpha_min_valid=0.0005
        self.err_x_max=0.8; self.err_y_max=0.8; self.err_a_max=0.018
        self.int_y_max=0.2; self.int_z_max=0.2
        self.detection_timeout=3.0; self.stale_timeout=1.5; self.ppo_timeout=2.0
        self.pred_gain_scale=0.7
        self.APPROACH_RAMP_S=2.0; self.approach_start_time=None
        self.recovery_duration=2.0; self.recovery_start_time=None
        self.vel_smooth_normal=0.15; self.vel_smooth_reversal=0.1

        self.cx=self.cy=None; self.alpha=0.; self.last_cx=self.last_cy=None
        self.got_real_detection=False; self.is_prediction=False
        self.last_real_detection_time=None
        self.armed=False; self.altitude=0.; self.current_pitch=0.
        self.takeoff_ready=False; self.phase="TAKEOFF"
        self.prev_err_x=self.prev_err_y=0.
        self.prev_err_a=0.
        self.int_err_y=self.int_err_z=0.
        self.last_ppo_time=None
        self.prev_vx=self.prev_vy=self.prev_vz=self.prev_wz=0.

        # v6.22: Kalman velocity state + SEARCH phase variables
        self._kf_vx=0.0; self._kf_vy=0.0       # image-space velocity (px/s)
        self._search_elapsed=0.0                 # time spent in current SEARCH
        self._pred_cx=self.img_cx                # predicted target cx during SEARCH
        self._pred_cy=self.img_cy                # predicted target cy during SEARCH
        self._search_base_cwz=0.0               # yaw bias from velocity at loss
        # v6.29: latch the loss-instant velocity for SEARCH heading. The Kalman
        # zeroes its velocity at max_dropout=30 frames (1.5 s), but SEARCH only
        # reads it at detection_timeout=3.0 s -> it was always 0, so SEARCH flew
        # blind. We latch (_kf_vx,_kf_vy) from the LAST tracking frame (while
        # still valid) and use that frozen v0 for the SEARCH heading. SEARCH-
        # only: HOLD/APPROACH output untouched. Config 1 has no kalman_velocity
        # so v0 stays 0 -> graceful fallback to blind search.
        # Default OFF: the tau sweep (2026-06-19) showed the latch never lifts the
        # T7 gate (its bottleneck is FOV/closure, not SEARCH heading) and only
        # helps reactive evasion (T9) at tau=1.0; default-on would regress the
        # deterministic gate + matrix. Kept as an opt-in evasion-recovery aid.
        self._search_latch=bool(rospy.get_param("~search_latch", False))
        self._kf_vx_latched=0.0; self._kf_vy_latched=0.0
        self._was_tracking=False
        # v6.30 (light): time-bound the latched heading with ONE smooth decay.
        # The heading weight is w(t)=exp(-t/tau), t = time since loss: full at
        # the loss instant, fading smoothly toward 0. tau is the single knob
        # (~search_tau, larger = trust the heading longer). T7 (deterministic
        # escape) benefits from the early-high weight; T9 (reactive evasion) is
        # protected because the heading fades before a mid-gap reverse can be
        # committed to. ONE clamp constant bounds the per-step extrapolation so
        # a large v0 can't slam the predicted position to the frame edge.
        self._search_tau=float(rospy.get_param("~search_tau", 1.0))
        self.SEARCH_STEP_CLAMP_PX=12.0   # max |per-tick heading extrapolation| (px)

        self.cmd_pub=rospy.Publisher('/mavros/setpoint_raw/local',PositionTarget,queue_size=1)
        self.active_pub=rospy.Publisher('/drone_tracking/ibvs_active',Bool,queue_size=1)
        self.phase_pub=rospy.Publisher('/drone_tracking/ibvs_phase',String,queue_size=1)
        self.emerg_pub=rospy.Publisher('/drone_tracking/emergency_brake',Bool,queue_size=1)
        det_topic = '/drone_tracking/filtered_target' if self.detection_source == 'kalman' else '/drone_tracking/target_center'
        rospy.Subscriber(det_topic, Point, self.detection_cb, queue_size=1)
        rospy.loginfo("[IBVS] detection_source=%s (subscribing to %s)" % (self.detection_source, det_topic))
        rospy.Subscriber('/drone_tracking/kalman_velocity',Point,self.kf_vel_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints',Quaternion,self.setpoints_cb,queue_size=1)
        rospy.Subscriber('/mavros/state',State,self.state_cb,queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose',PoseStamped,self.pose_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready',Bool,self.takeoff_ready_cb,queue_size=1)

        self.dt=1./20.; self.rate=rospy.Rate(20)
        rospy.loginfo("[IBVS] v6.30 | K_far=%.0f Kd_a=%.0f ff_max=%.1f max_vx=%.1f dead=%.3f | Kp_wz=%.2f Kp_y=%.2f Kp_z=%.2f max_vz=%.2f | search_latch=%s tau=%.2fs step_clamp=%.0fpx"
                      %(self.K_far,self.Kd_a,self.ff_max,self.max_vx,self.DEAD_ZONE,self.Kp_wz,self.Kp_y,self.Kp_z,self.max_vz,self._search_latch,
                        self._search_tau,self.SEARCH_STEP_CLAMP_PX))
        rospy.loginfo("[IBVS] v6.26 emergency brake: engage a>%.4f, release a<%.4f (P1 guard)"
                      %(self.ALPHA_EMERGENCY,self.ALPHA_EMERGENCY_EXIT))
        self.run()

    def state_cb(self,m): self.armed=m.armed
    def pose_cb(self,m):
        self.altitude=m.pose.position.z
        q=m.pose.orientation;sp=2*(q.w*q.y-q.z*q.x)
        self.current_pitch=math.copysign(math.pi/2,sp) if abs(sp)>=1 else math.asin(sp)
    def takeoff_ready_cb(self,m):
        if m.data and not self.takeoff_ready:
            rospy.loginfo("[IBVS] Takeoff complete");self.takeoff_ready=True
    def detection_cb(self,m):
        if np.isnan(m.x) or np.isnan(m.y) or np.isnan(m.z):
            self.got_real_detection=False;self.is_prediction=False;return
        if self.phase in ("TAKEOFF","DISARMED"):return
        self.cx=m.x;self.cy=m.y;self.alpha=np.clip(abs(m.z)/self.area_norm,0.,1.)
        if m.z>0:
            self.got_real_detection=True;self.is_prediction=False
            self.last_real_detection_time=rospy.Time.now()
            self.last_cx=m.x; self.last_cy=m.y
        else: self.got_real_detection=False;self.is_prediction=True
    def kf_vel_cb(self,m):
        """Receives Kalman velocity state for SEARCH direction prediction."""
        self._kf_vx=float(m.x); self._kf_vy=float(m.y)
    def _heading_weight(self):
        """v6.30 (light): single smooth confidence decay for the latched heading.
        w(t)=exp(-t/tau), t = time since loss. w=1 at the loss instant, fading
        toward 0 so a stale heading is abandoned rather than chased."""
        if self._search_tau<=0.0: return 0.0
        return float(math.exp(-self.time_since_detection()/self._search_tau))

    def _srch_v(self):
        """Velocity used for SEARCH heading. v6.30 (light): the latched loss-
        instant v0 scaled by the time-decay weight w(t). With search_latch off,
        returns the live (already-zeroed) estimate -> svx=0 -> byte-for-byte
        v6.28. Config 1 has no kalman_velocity -> v0=0 -> graceful blind search."""
        if not self._search_latch:
            return self._kf_vx, self._kf_vy
        w=self._heading_weight()
        return self._kf_vx_latched*w, self._kf_vy_latched*w
    def setpoints_cb(self,m):
        if not self.USE_PPO:return
        self.x_star=np.clip(float(m.x),-.3,.3);self.y_star=np.clip(float(m.y),-.3,.3)
        self.alpha_star=np.clip(float(m.z),.003,.020)
        self.lam=np.clip(float(m.w),.3,1.);self.last_ppo_time=rospy.Time.now()
    def time_since_detection(self):
        if self.last_real_detection_time is None:return float('inf')
        return (rospy.Time.now()-self.last_real_detection_time).to_sec()
    def ppo_is_active(self):
        if not self.USE_PPO or self.last_ppo_time is None:return False
        return (rospy.Time.now()-self.last_ppo_time).to_sec()<self.ppo_timeout
    def reset_pid(self):
        self.prev_err_x=self.prev_err_y=self.prev_err_a=0.
        self.int_err_y=self.int_err_z=0.
    def in_recovery(self):
        if self.recovery_start_time is None:return False
        if (rospy.Time.now()-self.recovery_start_time).to_sec()>self.recovery_duration:
            self.recovery_start_time=None;return False
        return True
    def approach_ramp_factor(self):
        if self.approach_start_time is None:return 1.
        e=(rospy.Time.now()-self.approach_start_time).to_sec()
        if e>=self.APPROACH_RAMP_S:self.approach_start_time=None;return 1.
        return e/self.APPROACH_RAMP_S
    def smooth(self,pv,nv):
        s=self.vel_smooth_reversal if pv*nv<0 and abs(nv)>.05 else self.vel_smooth_normal
        return s*pv+(1-s)*nv
    def _build_body_vel_msg(self,vx=0.,vy=0.,vz=0.,wz=0.):
        m=PositionTarget();m.header.stamp=rospy.Time.now()
        m.coordinate_frame=PositionTarget.FRAME_BODY_NED;m.type_mask=BODY_VEL_TYPE_MASK
        m.velocity.x=float(vx);m.velocity.y=float(vy);m.velocity.z=float(vz);m.yaw_rate=float(wz)
        return m

    def compute_velocities(self,gain_scale=1.):
        ex=(self.cx-self.img_cx)/self.img_cx-self.x_star
        ey_raw=(self.cy-self.img_cy)/self.img_cy-self.y_star
        ey=ey_raw-self.current_pitch*self.pitch_compensation_gain
        ea=self.alpha-self.alpha_star
        ex=np.clip(ex,-self.err_x_max,self.err_x_max)
        ey=np.clip(ey,-self.err_y_max,self.err_y_max)
        ea=np.clip(ea,-self.err_a_max,self.err_a_max)

        dex=(ex-self.prev_err_x)/self.dt
        dey=(ey-self.prev_err_y)/self.dt
        dea=(ea-self.prev_err_a)/self.dt
        self.prev_err_x=ex; self.prev_err_y=ey; self.prev_err_a=ea

        self.int_err_y=np.clip(self.int_err_y+ex*self.dt,-self.int_y_max,self.int_y_max)
        self.int_err_z=np.clip(self.int_err_z+ey*self.dt,-self.int_z_max,self.int_z_max)

        lam_gain=(.4+.6*self.lam) if self.ppo_is_active() else .70
        gain=gain_scale*lam_gain

        # PD control on alpha with adaptive feedforward
        if self.in_recovery():
            vx=0.
        elif ea < -self.DEAD_ZONE:
            vx_p = self.K_far * np.sqrt(-ea - self.DEAD_ZONE) * gain
            ff = 0.0
            if dea < -0.0005:
                ff = min(self.ff_max, self.Kd_a * (-dea) * gain)
            vx = vx_p + ff
        elif ea > self.DEAD_ZONE:
            vx = -self.K_near * np.sqrt(ea - self.DEAD_ZONE) * gain
        else:
            vx = 0.
        vx=np.clip(vx,-self.max_vx_retreat,self.max_vx)

        vy=-gain*(self.Kp_y*ex+self.Ki_y*self.int_err_y+self.Kd_y*dex)
        vz=-gain*(self.Kp_z*ey+self.Ki_z*self.int_err_z+self.Kd_z*dey)
        wz=-gain*(self.Kp_wz*ex+self.Kd_wz*dex)
        vy=np.clip(vy,-self.max_vy,self.max_vy)
        vz=np.clip(vz,-self.max_vz,self.max_vz)
        wz=np.clip(wz,-self.max_wz,self.max_wz)
        vx=self.smooth(self.prev_vx,vx);vy=self.smooth(self.prev_vy,vy)
        vz=self.smooth(self.prev_vz,vz);wz=self.smooth(self.prev_wz,wz)
        self.prev_vx=vx;self.prev_vy=vy;self.prev_vz=vz;self.prev_wz=wz
        return vx,vy,vz,wz

    def run(self):
        while not rospy.is_shutdown():
            cvx=cvy=cvz=cwz=0.; pub=True
            # v6.29: latch velocity WHILE actively tracking (used only by SEARCH).
            # This update never feeds the control law, so HOLD/APPROACH output is
            # unchanged. On the tracking->loss transition, freeze v0 and log it.
            if self.got_real_detection and self.phase in ("APPROACH","HOLD"):
                self._kf_vx_latched=self._kf_vx; self._kf_vy_latched=self._kf_vy
                self._was_tracking=True
            elif self._was_tracking and not self.got_real_detection:
                self._was_tracking=False
                if self._search_latch:
                    rospy.loginfo("[IBVS] LOSS — latched v0=(%.1f,%.1f) px/s for SEARCH heading"
                                  %(self._kf_vx_latched,self._kf_vy_latched))
            if not self.armed: self.phase="DISARMED"; pub=False
            elif self.phase=="DISARMED": self.phase="TAKEOFF"; pub=False
            elif self.phase=="TAKEOFF":
                if self.takeoff_ready:
                    self.phase="SEARCH"
                    self._search_elapsed=0.0
                    self._pred_cx=self.img_cx; self._pred_cy=self.img_cy
                    self._search_base_cwz=0.0
                pub=False

            elif self.phase=="SEARCH":
                # v6.22: 2-stage velocity-predicted search
                self._search_elapsed += self.dt
                if self._search_elapsed < 3.0:
                    # Stage 1: 2.5 m/s toward Kalman-predicted target position (v6.28)
                    cvx = 2.5
                    if self.last_cx is not None:
                        ex_s=(self._pred_cx-self.img_cx)/self.img_cx
                        ey_s=(self._pred_cy-self.img_cy)/self.img_cy
                        cwz=float(np.clip(-0.4*ex_s,-self.max_wz,self.max_wz))
                        cvz=float(np.clip(-0.4*ey_s,-0.40,0.40))
                        # Extrapolate predicted position each tick along the
                        # time-decayed latched v0 (v6.30 light). The per-step
                        # displacement is clamped so a large v0 can't slam the
                        # prediction to the frame edge.
                        svx,svy=self._srch_v()
                        c=self.SEARCH_STEP_CLAMP_PX
                        self._pred_cx=float(np.clip(
                            self._pred_cx+np.clip(svx*self.dt,-c,c), 0., self.img_w))
                        self._pred_cy=float(np.clip(
                            self._pred_cy+np.clip(svy*self.dt,-c,c), 0., self.img_h))
                    else:
                        # No detection memory yet — climb to safe altitude
                        cvz=float(np.clip(
                            (self.min_altitude_safe-self.altitude)*0.3, -.20, .30))
                else:
                    # Stage 2: 4.5 m/s + slow yaw sweep ±30° around velocity heading (v6.28)
                    cvx = 4.5
                    sweep = 0.25 * math.sin(self._search_elapsed * 0.4)  # ~16s period
                    # v6.30: recompute the heading bias from the TIME-WEIGHTED v0
                    # every tick — as confidence decays the center -> 0, leaving a
                    # neutral sweep. flag-off: svx=0 -> base=0 (== v6.28).
                    svx,_=self._srch_v()
                    base_cwz=float(np.clip(-0.2*svx/self.img_cx,-self.max_wz,self.max_wz))
                    cwz = float(np.clip(base_cwz + sweep, -self.max_wz, self.max_wz))
                    if self.last_cy is not None:
                        ey_s=(self.last_cy-self.img_cy)/self.img_cy
                        cvz=float(np.clip(-0.3*ey_s,-0.40,0.40))
                if self.got_real_detection and self.alpha>self.alpha_min_valid:
                    self.reset_pid(); self.approach_start_time=rospy.Time.now()
                    self.phase="APPROACH"
                    rospy.loginfo("[IBVS] Re-acquired a=%.4f after %.1fs SEARCH → APPROACH"
                                  %(self.alpha, self._search_elapsed))

            elif self.phase=="APPROACH":
                da=self.time_since_detection()
                if da>self.detection_timeout:
                    self.reset_pid(); self.phase="SEARCH"
                    self._search_elapsed=0.0
                    # v6.30 (light): seed SEARCH at the last-known position (no
                    # one-shot jump) and let the clamped per-step extrapolation
                    # carry it along the time-decayed heading. With search_latch
                    # off, svx=live _kf_vx=0 -> seed == last_cx -> identical to
                    # v6.28. The sweep center starts along the decayed heading.
                    svx,svy=self._srch_v()
                    self._pred_cx=float(self.last_cx) if self.last_cx is not None else self.img_cx
                    self._pred_cy=float(self.last_cy) if self.last_cy is not None else self.img_cy
                    self._search_base_cwz=float(np.clip(
                        -0.2*svx/self.img_cx, -self.max_wz, self.max_wz))
                    if self._search_latch:
                        rospy.loginfo("[IBVS] SEARCH start: v0=(%.1f,%.1f) px/s w=%.2f base_cwz=%.2f"
                                      %(svx,svy,self._heading_weight(),self._search_base_cwz))
                elif da>self.stale_timeout: pass
                elif self.got_real_detection:
                    cvx,cvy,cvz,cwz=self.compute_velocities(gain_scale=self.approach_ramp_factor())
                    ex_=abs((self.cx-self.img_cx)/self.img_cx-self.x_star)
                    ey_=abs((self.cy-self.img_cy)/self.img_cy-self.y_star)
                    ea_=abs(self.alpha-self.alpha_star)
                    if ex_<.12 and ey_<.12 and ea_<.010:
                        self.phase="HOLD";rospy.loginfo("[IBVS] Centered → HOLD")
                elif self.is_prediction:
                    cvx,cvy,cvz,cwz=self.compute_velocities(gain_scale=self.pred_gain_scale)

            elif self.phase=="HOLD":
                da=self.time_since_detection()
                if da>self.detection_timeout:
                    self.reset_pid();self.phase="APPROACH";self.recovery_start_time=rospy.Time.now()
                elif da>self.stale_timeout:
                    rospy.logwarn_throttle(1,"[IBVS] HOLD stale %.1fs"%da)
                elif self.got_real_detection:
                    cvx,cvy,cvz,cwz=self.compute_velocities(gain_scale=1.)
                elif self.is_prediction:
                    cvx,cvy,cvz,cwz=self.compute_velocities(gain_scale=self.pred_gain_scale)

            # ── v6.26: P1 EMERGENCY BRAKE GUARD — override ABOVE the control law.
            # The normal brake branch saturates at max_vx_retreat=0.50 m/s; a
            # target closing faster than that out-runs it (recorded collision:
            # 0.40 m separation). Above ALPHA_EMERGENCY, vx is forced to full
            # reverse, bypassing both vel_smooth and the retreat clamp (a
            # smoothed/clamped brake defeats its purpose). vy/vz/wz keep
            # tracking so the camera stays on the target. Hysteresis exit
            # prevents chatter at the boundary. If the target is lost while
            # engaged, release and let normal SEARCH logic take over — never
            # brake blind.
            if self.phase in ("APPROACH","HOLD") and (self.got_real_detection or self.is_prediction):
                if not self.emergency_engaged and self.alpha > self.ALPHA_EMERGENCY:
                    self.emergency_engaged=True; self._emerg_count+=1
                    rospy.logwarn("[IBVS] EMERGENCY BRAKE ENGAGED #%d a=%.4f > %.4f"
                                  %(self._emerg_count,self.alpha,self.ALPHA_EMERGENCY))
                elif self.emergency_engaged and self.alpha < self.ALPHA_EMERGENCY_EXIT:
                    self.emergency_engaged=False
                    rospy.loginfo("[IBVS] Emergency brake RELEASED a=%.4f < %.4f"
                                  %(self.alpha,self.ALPHA_EMERGENCY_EXIT))
                if self.emergency_engaged:
                    cvx=-self.max_vx
                    self.prev_vx=cvx   # keep smoother memory consistent for release
            elif self.emergency_engaged:
                self.emergency_engaged=False
                rospy.logwarn("[IBVS] Emergency brake RELEASED (target lost) — SEARCH takes over")
            self.emerg_pub.publish(Bool(data=self.emergency_engaged))

            if pub and self.armed and self.altitude<0.5 and self.phase not in ("TAKEOFF","DISARMED"):
                cvz=max(cvz,.3)
            if pub: self.cmd_pub.publish(self._build_body_vel_msg(cvx,cvy,cvz,cwz))
            self.active_pub.publish(Bool(data=self.phase in ("APPROACH","HOLD")))
            self.phase_pub.publish(String(data=self.phase))
            if self.phase in ("APPROACH","HOLD") and self.cx is not None:
                ev=self.alpha-self.alpha_star
                dea_v=(ev-self.prev_err_a)/self.dt if hasattr(self,'_last_ea_log') else 0.
                self._last_ea_log=ev
                ppo_s=" PPO a*=%.4f lam=%.2f"%(self.alpha_star,self.lam) if self.ppo_is_active() else ""
                rospy.loginfo_throttle(2,
                    "[IBVS] %s ea=%.4f dea=%.4f a=%.4f vx=%.2f alt=%.1f det=%s%s"
                    %(self.phase,ev,dea_v,self.alpha,cvx,self.altitude,
                      "REAL" if self.got_real_detection else "PRED" if self.is_prediction else "NONE",ppo_s))
            self.rate.sleep()

if __name__=='__main__': IBVSController()