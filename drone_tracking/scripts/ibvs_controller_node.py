#!/usr/bin/env python3
"""
ibvs_controller_node.py  —  v6.31  (distance-domain vx PID, default-OFF)
=======================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v6.31 — distance-domain vx PID + standoff rosparams (2026-07-07):
  * ~alpha_star / ~ea_hold / ~dead_zone rosparams (defaults = baseline
    0.0067/0.010/0.002 -> plain run byte-for-byte v6.30).
  * ~vx_mode ('legacy' default | 'pid'): pid replaces the K_far/K_near
    sqrt-of-ea vx law with a P(I) controller in METERS:
      d_hat = sqrt(alpha_dist_k/alpha)   (pinhole: alpha ~ k/d^2, k=0.0625
              fit through the standoff-sweep cells 3/5/6/7 m)
      e_d   = d_hat - d_star             (~d_star, default 8.0 m)
      vx    = (Kp_vx*e_d + Ki_vx*int(e_d)) * gain   (then lambda_x + clamps
              + vel_smooth as before; Ki_vx=0 default = pure P)
    HOLD entry in pid mode gates on |e_d| < ~hold_d_tol (1.0 m) instead of
    the ea band. Known P-only property: a constant-velocity target needs
    vx=v_t at steady state -> e_ss = v_t/(0.70*Kp_vx) extra standoff; if T3
    shows it, enable Ki_vx (PI). vy/vz/wz PIDs, SEARCH, emergency brake all
    untouched.

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
from std_msgs.msg import Bool, String, Float32MultiArray

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
        self.pitch_compensation_gain=0.4   # legacy (unused after 2026-07-08 fix)
        # exact pitch de-rotation: scale on the body-pitch term (1.0 = full
        # geometric stabilization). ~pitch_comp rosparam, sweepable.
        self.pitch_comp=float(rospy.get_param("~pitch_comp",1.0))
        self.F_PX=277.19   # live camera focal length (px) — camera_info fx
        # derivative low-pass weight (EMA on prev derivative). 0 = raw finite
        # diff (legacy); 0.6 ~= 2-tick smoothing. Cuts Kd-amplified cmd jitter.
        self.deriv_lpf=float(rospy.get_param("~deriv_lpf",0.6))
        self.x_star=0.; self.y_star=0.; self.lam=0.5
        # v6.31: HOLD-standoff knobs rosparam-overridable (defaults = current
        # baseline -> plain run byte-for-byte). Launch knobs ALPHA_STAR/EA_HOLD.
        self.alpha_star=float(rospy.get_param("~alpha_star",0.0067))
        self.ea_hold=float(rospy.get_param("~ea_hold",0.010))

        # ── Distance gains ────────────────────────────────────────────
        self.K_far  = 35.0
        self.K_near = 6.0
        self.Kd_a   = 150.0
        self.ff_max = 1.5
        self.DEAD_ZONE = float(rospy.get_param("~dead_zone",0.002))

        # ── v6.31 distance-domain vx PID (default OFF -> legacy sqrt law) ────
        # ~vx_mode 'legacy' keeps the K_far/K_near sqrt-of-ea law byte-for-byte.
        # 'pid': vx = (Kp_vx*e_d + Ki_vx*int(e_d))*gain with e_d = d_hat-d_star,
        # d_hat = sqrt(alpha_dist_k/alpha). Pinhole model alpha ~ k/d^2; k=0.0625
        # is the exact fit through the standoff-sweep cells (0.0067@3m,
        # 0.0025@5m, 0.00173@6m, 0.00127@7m). Error is LINEAR IN METERS, so one
        # gain works from spawn range to standoff (P on ea has no far-field
        # dynamic range at d*=8: |ea| is bounded by alpha_star~0.001).
        # P first, Ki_vx=0 default. A constant-velocity target (T3, 3.5 m/s)
        # leaves a P-only steady-state gap e_ss = v_t/(Kp_vx*0.70) — measuring
        # that gap IS the PI trigger criterion.
        self.vx_mode=str(rospy.get_param("~vx_mode","legacy"))
        self.d_star=float(rospy.get_param("~d_star",8.0))
        # HOLD BAND (2026-07-08, user spec): hold anywhere in [d_hold_min,
        # d_hold_max]=[6,8] m instead of a single point. INSIDE the band the
        # chaser COASTS at the target-matched speed (no push, no stop) — this
        # kills the advance-stop lurch (the old code zeroed vx in the dead zone).
        # Outside the band it gently corrects back. Below min_dist it is a hard
        # safety floor (brake + dodge).
        self.d_hold_min=float(rospy.get_param("~d_hold_min",6.0))
        self.d_hold_max=float(rospy.get_param("~d_hold_max",8.0))
        self.min_dist=float(rospy.get_param("~min_dist",4.0))
        # R1b: safety-ceiling bias budget — d_hat's measured close-range
        # over-read (~0.9 m, k-calibration + alpha nonlinearity, both configs)
        self.d_safe_margin=float(rospy.get_param("~d_safe_margin",1.0))
        # Distance-PID gains — defaults = sweep winner (cell B, 2026-07-08):
        # Kp 2.5 / Ki 1.2 / Kd 1.5 killed the 8<->12 m surge (HOLD 99.4%, det
        # 94.9%, dist std 1.62, closest 7.3 m) on the chase benchmark.
        self.Kp_vx=float(rospy.get_param("~Kp_vx",2.5))
        self.Ki_vx=float(rospy.get_param("~Ki_vx",1.2))
        # Kd on the closing rate d(d_hat)/dt: the loop carries ~1 s of lag
        # (detector median + Kalman + d_lpf + vel_smooth + PX4 velocity
        # response), so P-only surges advance-stop near Kp>=3. Damping trims
        # vx while closing fast and vanishes at steady chase (e_ss untouched).
        self.Kd_vx=float(rospy.get_param("~Kd_vx",1.5))
        # Approach envelope (0 = off): vx <= v_target_est + sqrt(2*a_dec*e_d).
        # Measured on the 300 s C2 PID run: cmd_vx stayed 4.50 down to 8.6 m
        # and reversal episodes overshot the 8 m setpoint to 1.75/1.58/0.90 m.
        # The envelope is the decel-feasible closing speed: loose when far
        # (fast approach kept), converging to the TARGET's speed at e_d=0
        # (matched-speed arrival). v_target_est = gap-rate + own vx, so a
        # target turning toward the chaser collapses the cap immediately.
        self.a_dec=float(rospy.get_param("~a_dec",0.0))
        # 2026-08-06 CALIBRATION 0.096 -> 0.077: measured alpha*d_true^2 vs Gazebo
        # GT (48k detected frames) = ~0.077 in the 5-8 m band; 0.096 made d_hat
        # over-read ~13% (+0.5 m) so the chaser held at true ~5.5 m believing 6 m.
        # 0.077 -> d_hat honest -> holds at true 6-7 m + decel cap engages at the
        # right distance (shallower, safer arrival). Validated T1-T8: custody 100%,
        # HOLD 95-98%, closest safer on 7/8; safety ceiling reads honest d_hat (more
        # conservative). This is THE smooth-deceleration fix (near-band gadgets
        # approach_smooth/decel_damp were dead ends: d_hat bias was the root cause).
        self.alpha_dist_k=float(rospy.get_param("~alpha_dist_k",0.077))
        # dead_d 0.05 (was 0.2): a P law is gentle near zero error by itself;
        # the wide dead zone caused a stop-go limit cycle (vx=0 inside, kick
        # outside) — the run-4 chaser surge the user saw.
        self.dead_d=float(rospy.get_param("~dead_d",0.05))      # m, vx dead zone
        self.hold_d_tol=float(rospy.get_param("~hold_d_tol",1.0)) # m, HOLD entry
        self.int_e_d=0.0
        # int_d_max 3.0 -> 6.0: effective Ki*clamp*gain was 1.2*3.0*0.70 = 2.52
        # m/s < the 3.5 m/s target cruise, so INSIDE the band the integrator
        # could never supply the matched speed -> the chaser always coasted too
        # slow -> gap grew -> Kp re-accelerated == the decel/reaccel hunt the
        # user diagnosed at 6-7 m. At 6.0 the ceiling is 1.2*6.0*0.70 = 5.0 m/s,
        # so the integrator CAN charge to the 3.5 cruise and truly hold station.
        self.int_d_max=float(rospy.get_param("~int_d_max",6.0))   # integrator clamp (m*s)
        # Conditional integration (anti-windup): only charge the integrator when
        # |e_d| < int_band. During the far sprint (large e_d) Kp+the decel cap
        # drive the approach and the integrator stays ~0, so it does NOT wind up
        # and overshoot inward on arrival; it charges to the cruise speed only
        # as the chaser settles near the hold point.
        self.int_band=float(rospy.get_param("~int_band",2.5))     # m
        # Phantom-cruise bleed factor (per 20 Hz tick) applied when the chaser is
        # inside the band-top yet still closing (integrator over-charged from the
        # static-target approach). 0.90 -> ~0.12/s decay: kills the phantom cruise
        # fast enough to stop the descent overshoot, releases the moment closing
        # stops (d_rate>=-0.1) so the held cruise value is preserved for the chase.
        self.int_bleed=float(rospy.get_param("~int_bleed",0.90))
        # HOLD-only integration (phantom-cruise root fix): when 1, the integrator
        # charges ONLY in HOLD, never during the APPROACH descent. On a static /
        # late-launching target the descent then runs on pure-P + the decel cap
        # and eases to a STOP at the band (vx->0), instead of the integrator
        # over-charging to ~3.5 m/s cruise and diving ~2 m past it. Cruise is
        # built once station is held and the target actually moves.
        self.int_hold_only=int(rospy.get_param("~int_hold_only",0))
        # SOFT-CENTERING inside the hold band (user request 2026-07-11: the gap
        # free-wanders across the full 1 m dead band, true dist 5.5-6.5 m).
        # Gentle P pull toward the band CENTRE applied ONLY while inside the
        # band (e_d==0 there, so this fills the dead zone with a weak slope
        # instead of flat nothing). kappa must stay << Kp_vx or the in-band
        # push/stop hunt the dead band was built to kill comes back. 0 = off.
        self.band_kp=float(rospy.get_param("~band_kp",0.0))
        self.d_hat=float('nan')                 # live distance estimate (m)
        # d_hat validity floor. NOT alpha_min_valid (0.0005 <-> ~11 m): first
        # C1/T3 flight ABORTED because d_hat froze at ~11 m -> e_d capped ~3 m
        # -> vx capped ~2.2 m/s < the 3.5 m/s target. 5e-5 <-> ~35 m; d_hat
        # clamped to d_hat_max so a 1-px glitch can't command a runaway sprint.
        self.alpha_d_min=float(rospy.get_param("~alpha_d_min",5e-5))
        self.d_hat_max=40.0
        # EMA low-pass on d_hat: even after the detector's 5-frame area median,
        # small boxes (10-20 px) step alpha ~5%/frame (p90, measured run 3) ->
        # ~0.24 m d_hat jumps -> visible vx surging at Kp>=2. Weight on the
        # PREVIOUS estimate; 0.4 ~= 2-tick lag, cuts step noise ~x0.6.
        self.d_lpf=float(rospy.get_param("~d_lpf",0.4))
        self._prev_d_hat=float('nan'); self._d_rate=0.0   # filtered d(d_hat)/dt
        self._dbg_edd=self._dbg_vxraw=self._dbg_cap=0.0   # vx-pipeline instrument
        # Velocity FEEDFORWARD (2026-07-09, user-diagnosed hunting): at station
        # the old law slowed (integrator eased) -> gap grew -> re-accelerated ->
        # decel/reaccel LIMIT CYCLE. Fix: estimate the target's LOS speed
        # v_t = chaser_speed + gap_rate = prev_vx + d_rate (the two swings cancel
        # so it's ~constant even while hunting), smooth it, and command it
        # DIRECTLY so the chaser cruises at the target speed; Kp only trims the
        # gap. ~vx_ff (1=on default), ~vff_lpf EMA weight.
        # DEFAULT-OFF (2026-07-09): the self-referential v_t=prev_vx+d_rate feeds
        # back through prev_vx and, with no proportional pushback before the hard
        # emergency-brake wall, bounced the chaser between a target-speed sprint
        # and a -4 m/s brake slam (vx std 4.0, dist 3-15 m, emerg 6%). REJECTED.
        # The raised+conditionally-integrated PI below is the proper velocity
        # feedforward (the integrator converges to the target's cruise speed).
        self.vx_ff=float(rospy.get_param("~vx_ff",0.0))
        self.vff_lpf=float(rospy.get_param("~vff_lpf",0.9))
        self.v_ff=0.0
        # Coast-through-misses (2026-07-08): YOLO drops frames, and in Config 1
        # (no Kalman PRED) the controller had nothing to act on -> cmd_vx=0 on
        # every miss, so the chaser STUTTERED instead of closing the gap and
        # fell behind a fast target. Instead, hold the last tracking command,
        # gently decayed, for up to ~miss_hold_frames so the chaser keeps
        # approaching smoothly through dropouts. Config 2 rarely needs it
        # (Kalman predicts); identical code in both -> ablation-safe.
        self._last_track_cmd=(0.,0.,0.,0.)
        self._miss_frames=0
        self.miss_hold_frames=int(rospy.get_param("~miss_hold_frames",25))
        self.miss_decay=float(rospy.get_param("~miss_decay",0.97))
        # Acceleration (slew-rate) limit on the NORMAL command (2026-07-08).
        # The user-observed loop: fast accel/decel swings the chaser attitude
        # -> the camera pitches -> the label box shifts -> the controller
        # reacts -> more accel. Capping how fast vx/vy can change keeps the
        # attitude (and camera) gentle, breaking the loop at its source. The
        # emergency brake is applied AFTER this and bypasses it (must stay
        # instant). ~max_accel in m/s^2 (0 = off). 3.0 -> 0.15 m/s per 20 Hz tick.
        self.max_accel=float(rospy.get_param("~max_accel",3.0))
        # faster vx acceleration when catching up from far (target beyond band);
        # keeps the smooth limit near/inside the band.
        self.max_accel_fast=float(rospy.get_param("~max_accel_fast",8.0))
        # vz-specific accel limit (2026-08-02): the vertical channel needs to
        # REVERSE fast to track a bouncing target (az↑ ⟺ HOLD↑ on the zig-zag).
        # Default = max_accel (byte-for-byte) unless raised. Higher = faster vz.
        self.max_accel_vz=float(rospy.get_param("~max_accel_vz",self.max_accel))
        self._pub_vx=self._pub_vy=self._pub_vz=0.0

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
        # M11.3 vertical carrier (2026-07-12, defaults = legacy no-op): the vz
        # channel was pure P+D in practice — Ki_z*int_z_max*gain = 0.0056 m/s,
        # nothing against true-T7's 1.72 m/s legs, so sustained climb/descent
        # tracking required ey_ss = 1.72/(gain*Kp_z) = 0.82 — beyond the 0.8
        # clip = frame edge (measured riding +0.24/-0.19 mean on desc/climb).
        # Same defect class as the vx integrator ceiling. A REAL integrator
        # (e.g. Ki_z=2.0, int_z_max=1.8 -> ceiling 2.52 m/s) carries the leg
        # rate at ey~0; int_z_bleed<1 discharges it fast when ey flips sign
        # (T7's bounce is a +/-1.72 square wave, ~6.4 s half-period — a stale
        # integral after each reversal is the failure mode to guard).
        self.Kp_z=float(rospy.get_param("~Kp_z",3.0))
        self.Ki_z=float(rospy.get_param("~Ki_z",0.04)); self.Kd_z=0.5
        self.int_z_bleed=float(rospy.get_param("~int_z_bleed",1.0))
        # 2026-08-04 vertical-velocity FEEDFORWARD (Task #8): on a bouncing/
        # climbing target the integrator recharges slowly at each vertical
        # reversal -> the target rides ~0.20 of frame off-centre (|ey_c|~0.20 on
        # T7). Anticipate the target's TRUE vertical speed = d_hat * angular
        # drift rate: vz_ff = Kff_z * d_hat * dey (same sign/place as Kd_z*dey,
        # but distance-scaled so far-target angular drift maps to real m/s).
        # Default 0.0 = byte-for-byte OFF; ~vz_ff_gain rosparam / VZ_FF_GAIN knob.
        self.vz_ff_gain=float(rospy.get_param("~vz_ff_gain",0.0))
        self.vz_ff_cap=float(rospy.get_param("~vz_ff_cap",1.5))  # |feedforward| clamp (m/s)
        self.Kp_wz=float(rospy.get_param("~Kp_wz",0.9)); self.Ki_wz=0.; self.Kd_wz=0.15

        # Velocity limits
        # v6.27: max_vx 3.5 -> 4.5 (~max_vx rosparam). T3 proved 3.5 gives
        # ZERO closure on a 3.5 m/s target; interception needs a speed
        # advantage (~30%). PX4 MPC_XY_VEL_MAX=12 (build default) won't clip.
        self.max_vx=float(rospy.get_param("~max_vx",4.5)); self.max_vx_retreat=0.50
        # pid-mode retreat authority. MODERATE (2.5) not full (4.5): full-speed
        # retreat oscillated (over-retreat -> re-approach limit cycle, measured
        # min-sep 1.49 m + HOLD 16%). 2.5 lets the PID back off smoothly without
        # the limit cycle; the latched dodge handles genuine head-on passes.
        self.max_vx_retreat_pid=float(rospy.get_param("~max_vx_retreat_pid",2.5))
        # M10.3: max_vz rosparam-overridable (default = current 1.5 -> baseline).
        # cell D lever if the vz saturation pre-check shows the cap is binding.
        self.max_vy=1.20; self.max_vz=float(rospy.get_param("~max_vz",1.5)); self.max_wz=0.5
        self.emerg_vy=float(rospy.get_param("~emerg_vy",2.0))   # dodge speed at a pass
        # emergency brake reverse speed — DECOUPLED from max_vx so the chase
        # top speed can be raised (to close on the target) without making the
        # brake a violent slam. Stays moderate for smoothness.
        self.emerg_brake_vx=float(rospy.get_param("~emerg_brake_vx",4.5))

        # ── M12 Phase A: per-axis output multipliers (adaptive-λ knobs) ──────
        # Generalizes the single forward-aggression `gain` scalar into one
        # multiplier per channel, applied as a gain on each channel's
        # control-law output BEFORE its safety clamp + smoothing (so the max_v*
        # caps and vel_smooth stay authoritative regardless of λ). rosparam
        # ~lambda_* default 1.0 -> a plain run is byte-for-byte v6.30 (x*1.0==x).
        # These are the knobs the Phase E adaptive-λ oracle sweep schedules per
        # trajectory. Supersedes the deprecated PPO-era self.lam scheduler
        # (Config 3 parked since M9.6).
        self.lambda_x =float(rospy.get_param("~lambda_x", 1.0))
        self.lambda_y =float(rospy.get_param("~lambda_y", 1.0))
        self.lambda_z =float(rospy.get_param("~lambda_z", 1.0))
        self.lambda_wz=float(rospy.get_param("~lambda_wz",1.0))
        self.BASE_GAIN=0.70   # was lam_gain's (deployed) use_ppo=false constant

        # v6.26: P1 emergency brake guard (override above the control law)
        self.ALPHA_EMERGENCY      = float(rospy.get_param("~alpha_emergency", 0.033))
        self.ALPHA_EMERGENCY_EXIT = float(rospy.get_param(
            "~alpha_emergency_exit", 0.7*self.ALPHA_EMERGENCY))
        self.emergency_engaged = False
        self._emerg_count = 0
        self._emerg_t0 = None    # engage time — min-hold prevents brake chatter
        self._dodge_sign = 1.0   # lateral dodge direction, LATCHED at engage
        # pid-mode distance trigger for the same guard (0 = off). The alpha
        # trigger (0.033 ~ 1.6 m at k=0.084) fires too late at 3.5 m/s closure
        # (recorded 0.69-0.80 m through-passes); d_hat lets it fire at range.
        # Release needs BOTH alpha below exit AND d_hat past 1.3x the trigger.
        self.d_emerg = float(rospy.get_param("~d_emerg", 0.0))
        # tree-strike guard: baylands trees ~10 m, drones fly ~14 m. If the
        # chaser descends below ~alt_floor while chasing a dipping target it
        # clips canopy. Below the floor, downward vz is blocked and a gentle
        # climb is injected. 0 = off (keeps legacy runs byte-for-byte).
        self.alt_floor = float(rospy.get_param("~alt_floor", 0.0))

        # v6.22: min_altitude_safe raised to match new Z_FLOOR=12m
        self.min_altitude_safe=13.0; self.alpha_min_valid=0.0005
        self.err_x_max=0.8; self.err_y_max=0.8; self.err_a_max=0.018
        self.int_y_max=0.2
        # M11.3: int_z_max rosparam (default = legacy 0.2 no-op). Size with
        # Ki_z so gain*Ki_z*int_z_max >= the leg rate to carry (1.72 on T7).
        self.int_z_max=float(rospy.get_param("~int_z_max",0.2))
        self.detection_timeout=3.0; self.stale_timeout=1.5; self.ppo_timeout=2.0
        self.pred_gain_scale=0.7
        # ── F1 fix (2026-07-14, Results_reference/08_Failures F1): while the
        # controller is consuming KF PREDICTIONS (flt z<0), cmd_vx may never
        # rise above its value at the last REAL detection (+0.3 margin).
        # Predictions steer centering (vy/vz/wz untouched) but never fuel a
        # closure sprint: on T1/C2/s43 the KF's alpha-velocity (trained by a
        # retreat) extrapolated the target away during a 1.5 s dropout and the
        # law floored vx to 8 m/s blind through the true target (0.34 m pass).
        # Inert in Config 1 (raw source never sets is_prediction). Audit of 39
        # C2 flights: rule would have engaged in 7 (incl. the failure + the
        # T7 dropout-sprints); 32 provably unchanged.
        self.pred_vx_hold=int(rospy.get_param("~pred_vx_hold",1))
        self.pred_accel_max=float(rospy.get_param("~pred_accel_max",1.0))
        self._pred_vx_cap=0.0
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
        # M12 Phase D-prep: publish the controller's EXACT internal error +
        # derivative signals so flight_logger records the per-config-correct
        # values (raw stream in C1, filtered in C2). data =
        # [ex,ey,ea, dex,dey,dea, ctrl_state] where ctrl_state = 1 REAL / 2 PRED
        # / 0 no-detection-this-cycle. Purely observational — does not touch the
        # control law (Phase A byte-for-byte gate re-verified).
        self.err_pub=rospy.Publisher('/drone_tracking/ibvs_errors',Float32MultiArray,queue_size=1)
        self.ex_c=self.ey_c=self.ea_c=0.0; self.dex_c=self.dey_c=self.dea_c=0.0
        det_topic = '/drone_tracking/filtered_target' if self.detection_source == 'kalman' else '/drone_tracking/target_center'
        rospy.Subscriber(det_topic, Point, self.detection_cb, queue_size=1)
        rospy.loginfo("[IBVS] detection_source=%s (subscribing to %s)" % (self.detection_source, det_topic))
        rospy.Subscriber('/drone_tracking/kalman_velocity',Point,self.kf_vel_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints',Quaternion,self.setpoints_cb,queue_size=1)
        rospy.Subscriber('/mavros/state',State,self.state_cb,queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose',PoseStamped,self.pose_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready',Bool,self.takeoff_ready_cb,queue_size=1)

        self.dt=1./20.; self.rate=rospy.Rate(20)
        # NB "Kp_z=... max_vz=..." stay adjacent — m12_campaign_run.sh banner
        # freeze-check greps that contiguous pair; carrier fields append after.
        rospy.loginfo("[IBVS] v6.31 | K_far=%.0f Kd_a=%.0f ff_max=%.1f max_vx=%.1f dead=%.3f | Kp_wz=%.2f Kp_y=%.2f Kp_z=%.2f max_vz=%.2f Ki_z=%.2f int_z_max=%.2f int_z_bleed=%.2f | lambda=(%.2f,%.2f,%.2f,%.2f) | search_latch=%s tau=%.2fs step_clamp=%.0fpx"
                      %(self.K_far,self.Kd_a,self.ff_max,self.max_vx,self.DEAD_ZONE,self.Kp_wz,self.Kp_y,self.Kp_z,self.max_vz,
                        self.Ki_z,self.int_z_max,self.int_z_bleed,
                        self.lambda_x,self.lambda_y,self.lambda_z,self.lambda_wz,self._search_latch,
                        self._search_tau,self.SEARCH_STEP_CLAMP_PX))
        rospy.loginfo("[IBVS] v6.31 standoff/vx: a*=%.5f ea_hold=%.5f | vx_mode=%s d*=%.1fm Kp_vx=%.2f Ki_vx=%.2f Kd_vx=%.2f a_dec=%.2f k=%.4f dead_d=%.2fm hold_tol=%.2fm d_emerg=%.1f"
                      %(self.alpha_star,self.ea_hold,self.vx_mode,self.d_star,
                        self.Kp_vx,self.Ki_vx,self.Kd_vx,self.a_dec,self.alpha_dist_k,
                        self.dead_d,self.hold_d_tol,self.d_emerg))
        rospy.loginfo("[IBVS] v6.26 emergency brake: engage a>%.4f, release a<%.4f (P1 guard) | pred_vx_hold=%d (F1 guard)"
                      %(self.ALPHA_EMERGENCY,self.ALPHA_EMERGENCY_EXIT,self.pred_vx_hold))
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
        self.int_e_d=0.   # v6.31 vx distance integrator
        self.d_hat=float('nan')   # re-seed from fresh alpha after a loss
        self._prev_d_hat=float('nan'); self._d_rate=0.0
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
        # ── Pitch-stabilized vertical error (EXACT geometry, 2026-07-08) ──────
        # Hard accel/decel pitches the quad up to ~38 deg; the fixed forward
        # camera then sees the target shift vertically, injecting a spurious ey
        # that made vz thrash. Convert the pixel to a true angle below boresight
        # (atan, valid at large angles) and de-rotate by the SIGNED body pitch
        # into a gravity-stabilized frame — corrects both nose-up (decel/brake)
        # and nose-down (accel). Scaled by F/cy0=1.15 back into legacy ey units.
        # (Tried upstream in the detection node but YOLO starves its pose cb.)
        beta=math.atan2(self.cy-self.img_cy, self.F_PX)   # rad below boresight
        ey=(self.F_PX/self.img_cy)*(beta+self.pitch_comp*self.current_pitch)-self.y_star
        ea=self.alpha-self.alpha_star
        ex=np.clip(ex,-self.err_x_max,self.err_x_max)
        ey=np.clip(ey,-self.err_y_max,self.err_y_max)
        ea=np.clip(ea,-self.err_a_max,self.err_a_max)

        # Filtered derivatives (2026-07-08): raw finite differences at 20 Hz
        # turn ~0.15 ey detection noise into ~3.0 in dey, and Kd_z pumps that
        # into cmd_vz (measured p90 vz step 1.28 even with ey stable). An EMA
        # low-pass on the derivative (~deriv_lpf weight on the previous value)
        # kills the amplified noise; Config 2's Kalman de-noises upstream too.
        dl=self.deriv_lpf
        dex=dl*self.dex_c+(1-dl)*(ex-self.prev_err_x)/self.dt
        dey=dl*self.dey_c+(1-dl)*(ey-self.prev_err_y)/self.dt
        dea=dl*self.dea_c+(1-dl)*(ea-self.prev_err_a)/self.dt
        self.prev_err_x=ex; self.prev_err_y=ey; self.prev_err_a=ea
        # M12: cache exact internal error+derivative signals for /ibvs_errors
        self.ex_c=ex; self.ey_c=ey; self.ea_c=ea
        self.dex_c=dex; self.dey_c=dey; self.dea_c=dea

        self.int_err_y=np.clip(self.int_err_y+ex*self.dt,-self.int_y_max,self.int_y_max)
        # M11.3 carrier discharge: when ey flips against the stored integral
        # (leg reversal on T7's vertical bounce), a plain integrator un-winds
        # only at |ey|·dt — seconds of wrong-side vz. Multiplicative bleed
        # (0.90/tick @20 Hz ≈ 88%/s decay) dumps the stale carrier fast while
        # leaving steady-leg charging untouched. Default 1.0 = off = legacy.
        if self.int_z_bleed<1.0 and ey*self.int_err_z<0.0:
            self.int_err_z*=self.int_z_bleed
        self.int_err_z=np.clip(self.int_err_z+ey*self.dt,-self.int_z_max,self.int_z_max)

        # M12 Phase A: PPO-era lam_gain scheduler DEPRECATED (Config 3 parked;
        # ppo_is_active() is False in every deployed config, where lam_gain
        # collapsed to the 0.70 base-aggression constant). Per-axis aggression
        # is now the lambda_x/y/z/wz output multipliers applied below.
        gain=gain_scale*self.BASE_GAIN

        # vx: v6.31 distance-domain PID (vx_mode=pid) or legacy sqrt-of-ea law
        if self.vx_mode=="pid":
            # R2 (2026-07-15, F1 source fix): the distance belief updates ONLY
            # from REAL detections. On KF PRED frames the extrapolated alpha
            # walks (F1: believed distance inflated 8.8->11.5 m in 1.5 s and
            # the law sprinted through the true target) — predictions may
            # steer the camera, but never teach the distance. d_hat freezes at
            # the last measured value; _d_rate decays toward 0 with it. No-op
            # in Config 1 (raw source has no PRED frames).
            if self.alpha>=self.alpha_d_min and self.got_real_detection:
                d_new=min(math.sqrt(self.alpha_dist_k/max(self.alpha,1e-9)),
                          self.d_hat_max)
                self.d_hat=(d_new if math.isnan(self.d_hat)
                            else self.d_lpf*self.d_hat+(1.-self.d_lpf)*d_new)
            # BAND error: 0 inside [d_hold_min, d_hold_max], else signed
            # distance OUTSIDE the band (>0 too far, <0 too close).
            d=self.d_hat
            if math.isnan(d):            e_d=0.0
            elif d>self.d_hold_max:      e_d=d-self.d_hold_max
            elif d<self.d_hold_min:      e_d=d-self.d_hold_min
            else:                        e_d=0.0
            # filtered closing rate for the Kd term (EMA on the raw diff)
            if not (math.isnan(self.d_hat) or math.isnan(self._prev_d_hat)):
                self._d_rate=0.7*self._d_rate+0.3*(self.d_hat-self._prev_d_hat)/self.dt
            self._prev_d_hat=self.d_hat
            if self.in_recovery():
                vx=0.
            else:
                # Conditional integration (anti-windup): charge the integrator
                # ONLY near the setpoint (|e_d|<int_band). Far out, the sprint is
                # driven by Kp+the decel cap and the integrator stays ~0 (no
                # windup -> no arrival overshoot); near station it charges up to
                # the target's cruise speed so the chaser MATCHES it and holds
                # (fixes the decel/reaccel hunt). Inside a band e_d=0 -> the
                # integrator simply HOLDS its charged cruise value = smooth coast.
                # int_hold_only RACE FIX (8-seed campaign, 2026-07-11): pure
                # HOLD-only gating deadlocked 3/16 flights (both configs) — if
                # the target launches BEFORE the chaser first reaches the band,
                # HOLD is never entered, the integrator never charges, and the
                # pure-P chase locks at its steady-state gap ~2 m above the band
                # (vx==target speed, no closure, HOLD 0% forever). Discriminator
                # between that stalled chase and the phantom-cruise descent is
                # the GAP RATE: descending on a static target the gap CLOSES
                # (d_rate<0 -> keep blocking, no dive); a stalled/losing chase
                # has e_d>0 with the gap steady/opening (d_rate>-0.1) -> the
                # integrator MUST charge (classic integrate-when-not-improving).
                allow_int=((not self.int_hold_only) or (self.phase=="HOLD")
                           or (e_d>0 and self._d_rate>-0.1))
                if allow_int and abs(e_d) < self.int_band:
                    self.int_e_d=np.clip(self.int_e_d+e_d*self.dt,
                                         -self.int_d_max,self.int_d_max)
                # F7 (2026-07-15): DISCHARGE-ONLY below the band floor. A
                # positive cruise residual while TOO CLOSE is always stale
                # (its job is matching a fleeing target's speed); left frozen
                # it can exactly cancel the retreat P-term and park the chaser
                # in a stable equilibrium 0.5 m under the band (T1/C2/s45:
                # perfect tracking, HOLD 0% for 204 s). Draining it lets the
                # retreat win and the chaser re-enters the band. Never charges
                # (floored at 0) -> phantom-cruise + race-deadlock fixes intact.
                elif e_d<0.0 and self.int_e_d>0.0:
                    self.int_e_d=max(self.int_e_d+e_d*self.dt,0.0)
                # PHANTOM-CRUISE BLEED (2026-07-09, descent-overshoot root fix):
                # during the initial approach the integrator charges up toward the
                # target's cruise speed BEFORE the (still-static) target launches,
                # so the chaser carries ~3.5 m/s of phantom cruise that drives it
                # DOWN through the band (dip to ~5 m), then over-backs-off (bounce
                # to ~9 m). Signature of over-drive = inside the band-top yet STILL
                # CLOSING (on a matched chase d_rate~0 here). When we see it, decay
                # the integrator so the descent eases to a stop AT the band instead
                # of plunging through — and it damps a fast catch-up entry too.
                # Fires only when d_rate<-0.1 (genuinely closing); a legitimate
                # catch-up sits at e_d>0 (d>band top) so it is untouched.
                # GATED TO APPROACH ONLY: the bleed must NEVER fire in steady HOLD
                # (there d_rate ticks below -0.1 constantly from d_hat noise, and
                # bleeding the cruise integrator on every such tick drained it and
                # brought the hunt back — dist std 1.46, vx std 2.11). During the
                # initial descent the phase is APPROACH, so this kills the phantom
                # cruise exactly when it forms and is OFF once station is held.
                if (self.phase=="APPROACH" and not math.isnan(self.d_hat)
                        and self.d_hat<=self.d_hold_max and self._d_rate<-0.1):
                    self.int_e_d*=self.int_bleed
                # Fade the Kd (closing-rate) term OUT inside the band: it is
                # needed for approach damping (large e_d) but INSIDE the band it
                # just reacts to d_hat noise and swings vx -> pitch swing ->
                # visible FOV oscillation. kd_fade = |e_d| ramped over 1 m, so
                # Kd is full when approaching, ~0 when holding station (steady
                # vx -> steady pitch -> steady view).
                kd_fade=min(1.0, abs(e_d)/1.0)
                if self.vx_ff>0.0:
                    # target LOS speed estimate (prev_vx+d_rate), heavily
                    # smoothed -> the steady cruise speed the chaser should hold
                    vt=self.prev_vx+self._d_rate
                    self.v_ff=self.vff_lpf*self.v_ff+(1.0-self.vff_lpf)*vt
                    self.v_ff=float(np.clip(self.v_ff,0.0,self.max_vx))
                    # feedforward cruise + Kp gap-trim + faded-Kd approach damping
                    # (Ki dropped: the feedforward replaces the hunting integrator)
                    vx=self.v_ff+(self.Kp_vx*e_d+self.Kd_vx*self._d_rate*kd_fade)*gain
                else:
                    vx=(self.Kp_vx*e_d+self.Ki_vx*self.int_e_d
                        +self.Kd_vx*self._d_rate*kd_fade)*gain
                    # soft-centering: weak pull toward band centre, in-band only
                    # (e_d==0 exactly when d is inside [d_hold_min, d_hold_max])
                    if self.band_kp>0.0 and e_d==0.0 and not math.isnan(d):
                        centre=0.5*(self.d_hold_min+self.d_hold_max)
                        vx+=self.band_kp*(d-centre)*gain
                self._dbg_edd=e_d; self._dbg_vxraw=vx; self._dbg_cap=self.v_ff  # instrument
                # DECEL-FEASIBLE CLOSING CAP, referenced to the band FLOOR
                # (d_hold_min), active whenever CLOSING and still above the floor
                # — INCLUDING inside the band. Previously gated on e_d>0 (band top)
                # so it switched OFF the moment the chaser crossed into the band;
                # on a fast catch-up it then barrelled through to ~4.9 m before the
                # slew limit could stop it (the residual transient overshoot). Now
                # the cap keeps the closing speed on a sqrt profile that reaches
                # the target's cruise exactly AT the floor -> the chaser eases into
                # the band with ~zero relative speed, no overshoot, no bounce.
                # Baseline = the integrator's LEARNED cruise (Ki*int*gain), which
                # is STABLE (unlike prev_vx+d_rate, which dips during a fast close
                # and made the cap over-brake). margin measured to the floor.
                if self.a_dec>0.0 and self._d_rate<0.2 and self.d_hat>self.d_hold_min:
                    v_cruise=self.Ki_vx*self.int_e_d*gain   # learned target speed
                    # Aim the decel profile at the band CENTRE, not the floor, so
                    # the chaser reaches cruise speed 0.5 m ABOVE the floor -> the
                    # residual momentum/lag lands it ~inside the band instead of
                    # plunging ~1 m past it. Between centre and floor margin=0 ->
                    # cap=v_cruise, so closing is held at the target's speed (no
                    # excess) and the descent eases to a stop instead of diving.
                    cap_ctr=0.5*(self.d_hold_min+self.d_hold_max)
                    margin=max(self.d_hat-cap_ctr,0.0)
                    cap=v_cruise+math.sqrt(2.0*self.a_dec*margin)
                    self._dbg_cap=cap
                    vx=min(vx,cap)
                # R1 (2026-07-15, the s46 fix): ABSOLUTE stopping-distance
                # ceiling to the SAFETY floor (min_dist). Unlike the band cap
                # above, NO d_rate gate — no estimate can switch it off (s46:
                # box noise made d_rate read 'opening' mid-dive, the cap
                # disengaged, and a 5.5 m/s approach passed the static target
                # at 1.93 m). Physics only: never fly faster than a_dec can
                # stop before min_dist. Binds only close-in (at d_hat 15 m the
                # ceiling is 7.1; at 20 m it exceeds max_vx) so catch sprints
                # are untouched; negative (retreat) vx is never affected.
                # R1b (2026-07-17, T1/C2/s46 closest=1.87 WITH the ceiling
                # binding): d_hat OVER-READS ~0.9 m at close range (the known
                # k-calibration bias, both configs) so the envelope's promised
                # floor physically landed ~1 m low. The margin budgets the
                # guard for its own sensor's bias: stop by min_dist TRUE even
                # when d_hat flatters the distance by up to ~d_safe_margin.
                # R1c (2026-07-17, T4/C2/s46 closest=1.76 with R1b binding):
                # the envelope must stop the GAP, not just the chaser. T4's
                # orbit swept the target INWARD at ~2 m/s during the catch —
                # both closing means the floor arrives meters early. Subtract
                # the target's own closing speed (measured gap rate minus our
                # commanded speed, clamped >=0: a fleeing target relaxes
                # nothing). Static targets (T1): v_tgt~0 -> identical to R1b.
                if self.a_dec>0.0 and not math.isnan(self.d_hat):
                    margin=max(self.d_hat-(self.min_dist+self.d_safe_margin),0.0)
                    v_tgt=max(-self._d_rate-max(self.prev_vx,0.0),0.0)
                    vx=min(vx,max(math.sqrt(2.0*self.a_dec*margin)-v_tgt,0.0))
        elif self.in_recovery():
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
        vx=vx*self.lambda_x   # M12 λ_x (before safety clamp; λ=1 -> identity)
        # v6.31: SYMMETRIC retreat in pid mode. The legacy -max_vx_retreat=0.5
        # clamp (a P1-era artifact of the sqrt law) meant the PID could only
        # back away at 0.5 m/s, so any faster closure needed the hard emergency
        # brake — which then stuck engaged and wrecked the track. With full
        # retreat authority (~max_vx_retreat_pid, default = max_vx) the distance
        # PID decelerates and backs off SMOOTHLY on its own; the emergency brake
        # reverts to a rare last resort. Legacy mode keeps the 0.5 clamp.
        if self.vx_mode=="pid":
            vx=np.clip(vx,-self.max_vx_retreat_pid,self.max_vx)
        else:
            vx=np.clip(vx,-self.max_vx_retreat,self.max_vx)

        vy=-gain*(self.Kp_y*ex+self.Ki_y*self.int_err_y+self.Kd_y*dex)
        # vertical-velocity feedforward (Task #8): distance-scaled anticipation of
        # the target's vertical motion so vz doesn't wait for the integrator to
        # recharge at a bounce reversal. dey is already deriv_lpf-smoothed; guard
        # nan d_hat and clamp so it can't blow up at range.
        vz_ff=0.0
        if self.vz_ff_gain>0.0 and not math.isnan(self.d_hat):
            vz_ff=float(np.clip(self.vz_ff_gain*self.d_hat*dey,-self.vz_ff_cap,self.vz_ff_cap))
        vz=-gain*(self.Kp_z*ey+self.Ki_z*self.int_err_z+self.Kd_z*dey+vz_ff)
        wz=-gain*(self.Kp_wz*ex+self.Kd_wz*dex)
        vy=vy*self.lambda_y; vz=vz*self.lambda_z; wz=wz*self.lambda_wz  # M12 λ (before clamp)
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
                    # 2026-08-03 HOLD-label fix: gate on the pitch-compensated
                    # tracking errors the controller actually drives (ex_c/ey_c),
                    # NOT raw pixel offsets. On the fast T3 chase the steady
                    # nose-down cruise pitch biases raw ey to ~0.24 (> the 0.12
                    # gate) while ey_c stays ~0.03 in-band. HOLD latches on entry,
                    # so on seeds where raw-ey never dips under 0.12 the run reads
                    # 0 % HOLD despite holding perfectly (the "T3 0%-HOLD"
                    # artifact: 2/8 seeds, band-occ 93-94 % identical to the rest).
                    # ex_c == raw ex (no horizontal pitch coupling), so C1/C2 and
                    # all non-pitched trajectories are byte-unaffected.
                    ex_=abs(self.ex_c)
                    ey_=abs(self.ey_c)
                    # v6.31: HOLD distance criterion — meters in pid mode,
                    # ea band (rosparam ~ea_hold, default .010) in legacy mode
                    if self.vx_mode=="pid":
                        d_ok=(not math.isnan(self.d_hat) and
                              self.d_hold_min<=self.d_hat<=self.d_hold_max)
                    else:
                        d_ok=abs(self.alpha-self.alpha_star)<self.ea_hold
                    if ex_<.12 and ey_<.12 and d_ok:
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

            # ── Coast-through-misses: keep approaching smoothly on YOLO drops ──
            # If tracking this cycle, remember the command and reset the miss
            # counter. If this cycle is a MISS (no real detection, no Kalman
            # prediction) while in APPROACH/HOLD, coast on the last command,
            # gently decayed, for up to miss_hold_frames instead of dropping to
            # 0 — so the chaser keeps closing the gap through dropouts rather
            # than stuttering and falling behind (user-reported, 2026-07-08).
            if self.phase in ("APPROACH","HOLD"):
                if self.got_real_detection or self.is_prediction:
                    self._last_track_cmd=(cvx,cvy,cvz,cwz); self._miss_frames=0
                elif (self.time_since_detection()<self.detection_timeout
                      and self._miss_frames<self.miss_hold_frames):
                    self._miss_frames+=1
                    dcy=self.miss_decay**self._miss_frames
                    lc=self._last_track_cmd
                    cvx,cvy,cvz,cwz=lc[0]*dcy,lc[1]*dcy,lc[2]*dcy,lc[3]*dcy
                    self.prev_vx=cvx   # keep smoother continuous on re-acquire

            # ── F1 guard v2: RATE-LIMITED acceleration on prediction ──────────
            # v1 (hard freeze) fixed the T1 blind-sprint but cost T7 (3/3 s45
            # repeats 39.7-67.4 vs 98.0: T7 dropouts coincide with the target
            # genuinely fleeing on a leg — freezing vx loses the chase). v2:
            # while on PRED, vx may RAMP at pred_accel_max (1 m/s^2) above its
            # last-REAL value — enough to chase a fleeing leg through a short
            # dropout, but a blind 8 m/s sprint would need >5 s of continuous
            # prediction (the KF stops predicting at 1.5 s). F1 replay: cap =
            # 1.2+0.3+1.0*1.5 = 3.0 m/s vs the 8.0 that caused the 0.34 m pass.
            if self.pred_vx_hold and self.phase in ("APPROACH","HOLD"):
                if self.got_real_detection:
                    self._pred_vx_cap=cvx          # vx at the last REAL frame
                elif self.is_prediction:
                    t_pred=(rospy.Time.now()-self.last_real_detection_time).to_sec()
                    cap=self._pred_vx_cap+0.3+self.pred_accel_max*max(t_pred,0.0)
                    if cvx>cap: cvx=cap            # slowing always stays free

            # Slew-rate limit the NORMAL command (breaks the accel->pitch->box
            # ->controller loop). Applied in APPROACH/HOLD only; the emergency
            # brake below bypasses it. dv cap = max_accel*dt per tick.
            if self.max_accel>0.0 and self.phase in ("APPROACH","HOLD"):
                dv=self.max_accel*self.dt
                # vx: allow FAST acceleration when the target is far (catch up
                # to close the gap the user wants reduced) but the gentle limit
                # near/inside the band (smooth hold). Otherwise the slew limit
                # throttled the post-reversal sprint and it never closed.
                far=(self.vx_mode=="pid" and not math.isnan(self.d_hat)
                     and self.d_hat>self.d_hold_max+1.0)
                dvx=(self.max_accel_fast if far else self.max_accel)*self.dt
                cvx=float(np.clip(cvx,self._pub_vx-dvx,self._pub_vx+dvx))
                cvy=float(np.clip(cvy,self._pub_vy-dv,self._pub_vy+dv))
                dvz=self.max_accel_vz*self.dt   # vz can reverse faster (bounce tracking)
                cvz=float(np.clip(cvz,self._pub_vz-dvz,self._pub_vz+dvz))

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
            # ── IMMINENT-COLLISION AVOIDANCE (v6.31 redesign) ──────────────
            # Engage only when the target is BOTH near AND closing — a near-
            # but-receding target needs no brake (old version stuck engaged
            # for 14 s). Response: full reverse vx + a LATCHED lateral dodge
            # (direction fixed at engage so it can't flip and thrash, and
            # because a head-on target is dead-centre where ex_c~0). The dodge
            # is the primary avoidance: braking cannot out-retreat a 3.5 m/s
            # head-on closer, but a sidestep clears the path. Release when
            # separating, 0.5 s min-hold.
            if self.phase in ("APPROACH","HOLD") and (self.got_real_detection or self.is_prediction):
                d_valid=(self.vx_mode=="pid" and not math.isnan(self.d_hat))
                # Dodge only when GENUINELY close AND closing — firing at 8 m on
                # any fast gap-shrink (even with lateral clearance) over-fired
                # 19 s and held the chaser back. Trigger within min_dist+1.5 m
                # (~5.5 m) closing >0.8 m/s -> ~1.5 m of lead before the floor.
                attack = (d_valid and self.d_hat < self.min_dist+1.5
                          and self._d_rate < -0.8)
                floor  = d_valid and self.d_hat < self.min_dist
                a_trig = self.alpha > self.ALPHA_EMERGENCY
                d_trig = attack or floor
                d_clear= ((not d_valid) or self.d_hat > self.min_dist+2.5
                          or self._d_rate > 0.3)
                held_long_enough=(self._emerg_t0 is not None and
                                  (rospy.Time.now()-self._emerg_t0).to_sec()>=0.5)
                if not self.emergency_engaged and (a_trig or d_trig):
                    self.emergency_engaged=True; self._emerg_count+=1
                    self._emerg_t0=rospy.Time.now()
                    # latch dodge direction: toward the frame side the target
                    # is NOT on (or a default when dead-centre)
                    self._dodge_sign = 1.0 if self.ex_c>=0 else -1.0
                    rospy.logwarn("[IBVS] AVOID ENGAGED #%d a=%.4f d_hat=%.1f d_rate=%.1f dodge=%+d"
                                  %(self._emerg_count,self.alpha,self.d_hat,
                                    self._d_rate,int(self._dodge_sign)))
                elif (self.emergency_engaged and held_long_enough
                      and self.alpha < self.ALPHA_EMERGENCY_EXIT and d_clear):
                    self.emergency_engaged=False
                    rospy.loginfo("[IBVS] Avoid RELEASED a=%.4f d_hat=%.1f"
                                  %(self.alpha,self.d_hat))
                if self.emergency_engaged:
                    cvx=-self.emerg_brake_vx   # decoupled from max_vx: raising
                    self.prev_vx=cvx           # chase speed must not make the
                    #                            brake violent (stays smooth)
                    if self.vx_mode=="pid":
                        # STRONGER dodge (emerg_vy > max_vy) — the sidestep is
                        # the primary clearance-maker at a head-on pass; more
                        # lateral speed × the early trigger = wider miss.
                        cvy=self._dodge_sign*self.emerg_vy   # committed sidestep
                        self.prev_vy=cvy
            elif self.emergency_engaged:
                self.emergency_engaged=False
                rospy.logwarn("[IBVS] Emergency brake RELEASED (target lost) — SEARCH takes over")
            self.emerg_pub.publish(Bool(data=self.emergency_engaged))

            if pub and self.armed and self.altitude<0.5 and self.phase not in ("TAKEOFF","DISARMED"):
                cvz=max(cvz,.3)
            # tree-strike guard: hold above the canopy floor while tracking
            if (self.alt_floor>0.0 and self.armed
                    and self.phase in ("APPROACH","HOLD","SEARCH")):
                if self.altitude<self.alt_floor:
                    # block descent + gentle proportional climb back up
                    cvz=max(cvz, min(0.6, 0.5*(self.alt_floor-self.altitude)))
            # track what was actually published so the slew limiter resumes
            # smoothly after an emergency-brake override
            self._pub_vx=cvx; self._pub_vy=cvy; self._pub_vz=cvz
            if pub: self.cmd_pub.publish(self._build_body_vel_msg(cvx,cvy,cvz,cwz))
            self.active_pub.publish(Bool(data=self.phase in ("APPROACH","HOLD")))
            self.phase_pub.publish(String(data=self.phase))
            # M12 Phase D-prep: emit exact internal errors + derivatives.
            # ctrl_state = what the controller acted on THIS cycle (1 REAL /
            # 2 PRED / 0 none). ex_c..dea_c carry the last compute_velocities
            # values; a cycle with ctrl_state=0 leaves them stale (masked out
            # downstream via ctrl_state).
            _cs = 1.0 if self.got_real_detection else (2.0 if self.is_prediction else 0.0)
            self.err_pub.publish(Float32MultiArray(data=[
                self.ex_c,self.ey_c,self.ea_c,self.dex_c,self.dey_c,self.dea_c,_cs]))
            if self.phase in ("APPROACH","HOLD") and self.cx is not None:
                ev=self.alpha-self.alpha_star
                dea_v=(ev-self.prev_err_a)/self.dt if hasattr(self,'_last_ea_log') else 0.
                self._last_ea_log=ev
                ppo_s=" PPO a*=%.4f lam=%.2f"%(self.alpha_star,self.lam) if self.ppo_is_active() else ""
                rospy.loginfo_throttle(1,
                    "[IBVS] %s d_hat=%.1f e_d=%.1f | vx_raw=%.1f cap=%.1f d_rate=%.2f prevvx=%.1f -> cvx=%.2f | wz=%.2f vy=%.2f"
                    %(self.phase,self.d_hat,self._dbg_edd,self._dbg_vxraw,self._dbg_cap,
                      self._d_rate,self.prev_vx,cvx,cwz,cvy))
            self.rate.sleep()

if __name__=='__main__': IBVSController()