#!/usr/bin/env python3
"""
target_mover.py  —  v10.10  Trajectory-audit fixes (T2/T3 one-way, T6/T7 shuttle, T8 periodic)
=====================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v10.6 changes (on top of v10.5):
  B1 — fuzzy altitude-escape boost was INVERTED. It fired at phi>0.40, but its
       gain au=(phi-.60)/.40 is negative for phi in (0.40,0.60), so it drove the
       target TOWARD the chaser's altitude AND overwrote all 9 fuzzy altitude
       rules. Threshold raised to phi>0.60 (au now in [0,1]); for phi<=0.60 the
       fuzzy vz rules apply untouched. Deliberate behaviour change.
  SPRINT MF rescale finished: the sampled speed universe _SP is [1.0,3.5], but
       SPRINT was trap(3.20,3.60,4.00,4.00) — its plateau (3.6–4.0) sat OUTSIDE
       the universe, so its max membership inside [1.0,3.5] was only 0.75,
       silently capping top escape speed. Now SPRINT=trap(3.00,3.35,3.50,3.50);
       FAST tri(2.10,2.80,3.50) already fit. (v10.5 intent was "[1,4]→[1,3.5],
       MFs rescaled" but only the universe was rescaled — this finishes it.)
  NOTE: the T9 baseline RESETS after v10.6 (B1 + SPRINT change behaviour) — re-run
       the 2×T9 {42,43} stress baseline before any comparison.

v10.3 changes (on top of v10.2):
  SPEED RANGE [1.0 → 3.5 m/s]:
    Universe is the sampled array _SP = [1.0, 3.5] (61 points). Defuzzified
    output additionally clamped at 3.5. Floor stays at 1.0 (universe minimum).
    MFs (v10.6): SLOW  tri(1.00,1.20,1.50)  — near floor
         NORMAL tri(1.30,1.80,2.30)
         FAST   tri(2.10,2.80,3.50)
         SPRINT trap(3.00,3.35,3.50,3.50)  — v10.6: rescaled to fit [1.0,3.5]
    Note: high omega rules allow large heading changes; this naturally
    caps net ground displacement while keeping the chaser challenged.

  ALTITUDE RANGE EXTENDED:
    Z_FLOOR:   10.0 → 12.0 m  (clears tall trees)
    Z_CEIL:    20.0 → 24.0 m  (more vertical escape room)
    RISE_TO_Z: 12.0 → 14.0 m  (target cruises higher)
    RISE_VZ:   0.8  → 1.2 m/s (faster initial climb)
    STABILIZE_TIME: 3.0 → 1.5 s (less SETTLING pause at altitude)
    vz clamp:  ±1.5 → ±2.0 m/s

  VERTICAL MANEUVERABILITY:
    VERT_B:   [0.40,0.80] → [0.60,1.20] m/s  (stronger altitude weave)
    VERT_T:   [5.0, 10.0] → [4.0, 8.0]  s   (faster vertical cycles)
    vz_intensity CLOSE: [0.8,1.2] → [1.0,1.5]
    vz_intensity MID:   [1.0,1.5] → [1.2,2.0]
    vz_intensity FAR:   [1.2,2.0] → [1.5,2.5]
    Altitude escape boost multiplier: 1.0 → 1.3

  PHI-ADAPTIVE MANEUVER INTERVALS (4-tier):
    d<3m:   base [1.0,2.0]s  — very close, panic mode
    d<5m:   base [1.5,3.0]s
    d<8m:   base [2.5,4.5]s
    d>=8m:  base [3.5,6.0]s
    phi_factor = 0.6 + 0.7*(1-phi):
      phi=1.0 (centered): ×0.6 → fastest maneuver changes
      phi=0.5 (partial):  ×0.95 → normal
      phi=0.0 (escaped):  ×1.3  → maintain current escape heading

  DIRECTIONAL VARIETY (no vx reduction):
    OU_SIGMA_MAX: 0.70 → 1.00  (wider random heading swings)
    OU_LIMIT:     π/3 → π/2.5  (±60° → ±72° heading envelope)
    LAT_A_MIN: 40° → 50°  (higher baseline lateral weave)
    LAT_A_MAX: 75° → 90°  (wider max lateral weave)
    Maneuver weights: VX 8%→4%, VY 18%→22%, VY_VZ 16%→20%

v10.2 changes (preserved):
  Always-Escape fuzzy rules, 7-axis maneuver SM, EDGE→FAST

Phase flow: WAITING → RISING → SETTLING → MOVING
"""

import math, random, rospy
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool, Float32MultiArray
from gazebo_msgs.msg import ModelStates

VEL_YR_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW)
# velocity XY + POSITION Z (+ yaw_rate): PX4 holds the exact altitude so the
# target can't drift on the flat trajectories (2026-08-10, ~z_hold).
POSZ_VELXY_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_VZ |
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
    PositionTarget.IGNORE_YAW)

# ═══════════════════════════════════════════════════════════════════════
#  FUZZY ESCAPER  —  v10.6  (speed universe [1.0, 3.5])
# ═══════════════════════════════════════════════════════════════════════
class FuzzyEscaper:
    _SP=[1.00+i*2.5/60 for i in range(61)]   # [1.0 .. 3.5]
    _OM=[0.00+i*1.20/60 for i in range(61)]
    _VZ=[-1.00+i*2./60 for i in range(61)]
    @staticmethod
    def _trap(x,a,b,c,d):
        if x<a or x>d:return 0.
        if b<=x<=c:return 1.
        if x<b:return (x-a)/(b-a) if b-a>1e-9 else 1.
        return (d-x)/(d-c) if d-c>1e-9 else 1.
    @staticmethod
    def _tri(x,a,b,c):
        if x<=a or x>=c:return 0.
        if x<=b:return (x-a)/(b-a) if b-a>1e-9 else 1.
        return (c-x)/(c-b) if c-b>1e-9 else 1.
    def _mu_d(s,d):
        t,r=s._trap,s._tri
        return {'VERY_CLOSE':t(d,0,0,1.5,3),'CLOSE':r(d,1.5,3,4.5),
                'MEDIUM':r(d,3,6,9),'FAR':r(d,6,9,12),
                'VERY_FAR':t(d,10,15,1e9,1e9)}
    def _mu_v(s,dv):
        t,r=s._trap,s._tri
        return {'FAST_CLOSING':t(dv,.35,.50,1e9,1e9),'CLOSING':r(dv,.05,.25,.45),
                'STABLE':r(dv,-.15,0,.15),'RECEDING':t(dv,-1e9,-1e9,-.10,.05)}
    def _mu_phi(s,phi):
        t,r=s._trap,s._tri
        return {'OUTSIDE':t(phi,0,0,.15,.30),'EDGE':r(phi,.20,.40,.60),
                'PARTIAL':r(phi,.50,.65,.80),'CENTERED':t(phi,.70,.85,1,1)}
    def _mu_dz(s,dz):
        t,r=s._trap,s._tri
        return {'MUCH_ABOVE':t(dz,3,5,1e9,1e9),'ABOVE':r(dz,1,2.5,4),
                'SAME':r(dz,-2,0,2),'BELOW':r(dz,-4,-2.5,-1),
                'MUCH_BELOW':t(dz,-1e9,-1e9,-5,-3)}
    def _mf_speed(s,v,t):
        T,R=s._trap,s._tri
        # v10.6: speed universe [1.0, 3.5]; SPRINT rescaled to fit (was 3.2,3.6,4,4)
        return {'SLOW':R(v,1.00,1.20,1.50),'NORMAL':R(v,1.30,1.80,2.30),
                'FAST':R(v,2.10,2.80,3.50),'SPRINT':T(v,3.00,3.35,3.50,3.50)}[t]
    def _mf_omega(s,v,t):
        T,R=s._trap,s._tri
        return {'NONE':T(v,0,0,.05,.12),'GENTLE':R(v,.08,.20,.35),
                'MODERATE':R(v,.28,.48,.68),'SHARP':R(v,.58,.75,.92),
                'MAX':T(v,.85,.98,1.20,1.20)}[t]
    def _mf_vz(s,v,t):
        T,R=s._trap,s._tri
        return {'FAST_DIVE':T(v,-1,-1,-.65,-.40),'DIVE':R(v,-.60,-.35,-.10),
                'HOLD':R(v,-.12,0,.12),'CLIMB':R(v,.10,.35,.60),
                'FAST_CLIMB':T(v,.40,.65,1,1)}[t]
    def _defuzz(s,act,univ,mf):
        n=d=0.
        for x in univ:
            a=max(min(a_,mf(x,t_)) for t_,a_ in act.items()); n+=x*a; d+=a
        return n/d if d>1e-9 else (univ[0]+univ[-1])*.5
    def infer(s,d,d_dot,phi,dz):
        D,V,F,Z=s._mu_d(d),s._mu_v(d_dot),s._mu_phi(phi),s._mu_dz(dz)
        sp={k:0. for k in ('SLOW','NORMAL','FAST','SPRINT')}
        om={k:0. for k in ('NONE','GENTLE','MODERATE','SHARP','MAX')}
        vz={k:0. for k in ('FAST_DIVE','DIVE','HOLD','CLIMB','FAST_CLIMB')}
        def f(s_,sp_=None,om_=None,vz_=None):
            if sp_:sp[sp_]=max(sp[sp_],s_)
            if om_:om[om_]=max(om[om_],s_)
            if vz_:vz[vz_]=max(vz[vz_],s_)

        # ── v10.2: ALWAYS ESCAPE — intensity = phi × closeness ────────
        f(D['VERY_CLOSE'],sp_='SPRINT',om_='MAX')
        f(min(D['CLOSE'],V['FAST_CLOSING']),sp_='SPRINT',om_='SHARP')
        f(min(D['CLOSE'],F['CENTERED']),sp_='SPRINT',om_='SHARP')
        f(min(D['CLOSE'],F['PARTIAL']),sp_='SPRINT',om_='MODERATE')
        f(min(D['CLOSE'],F['EDGE']),sp_='FAST',om_='MODERATE')
        f(min(D['MEDIUM'],V['FAST_CLOSING']),sp_='SPRINT',om_='MODERATE')
        f(min(D['MEDIUM'],F['CENTERED']),sp_='SPRINT',om_='SHARP')
        f(min(D['MEDIUM'],F['PARTIAL']),sp_='FAST',om_='MODERATE')
        f(min(D['MEDIUM'],F['EDGE']),sp_='FAST',om_='MODERATE')
        f(min(D['FAR'],V['FAST_CLOSING']),sp_='FAST',om_='GENTLE')
        f(min(D['FAR'],F['CENTERED']),sp_='FAST',om_='MODERATE')
        f(min(D['FAR'],F['PARTIAL']),sp_='NORMAL',om_='GENTLE')
        f(min(D['FAR'],F['EDGE']),sp_='FAST',om_='GENTLE')
        f(min(D['FAR'],V['RECEDING']),sp_='NORMAL',om_='GENTLE')
        f(D['VERY_FAR'],sp_='SLOW',om_='NONE')
        f(F['OUTSIDE'],sp_='FAST',om_='GENTLE')
        f(min(V['FAST_CLOSING'],F['CENTERED']),sp_='SPRINT',om_='MAX')
        f(min(D['VERY_FAR'],F['CENTERED']),sp_='NORMAL',om_='GENTLE')
        f(min(F['CENTERED'],V['STABLE']),sp_='FAST',om_='SHARP')
        f(min(F['CENTERED'],V['CLOSING']),sp_='SPRINT',om_='MAX')

        # ── Altitude escape rules ─────────────────────────────────────
        f(Z['MUCH_ABOVE'],vz_='FAST_DIVE')
        f(min(D['CLOSE'],Z['ABOVE']),vz_='DIVE')
        f(min(D['MEDIUM'],Z['ABOVE']),vz_='DIVE')
        f(Z['SAME'],vz_='HOLD')
        f(min(D['CLOSE'],Z['BELOW']),vz_='CLIMB')
        f(min(D['MEDIUM'],Z['BELOW']),vz_='CLIMB')
        f(Z['MUCH_BELOW'],vz_='FAST_CLIMB')
        f(min(D['VERY_CLOSE'],Z['MUCH_ABOVE']),vz_='FAST_DIVE')
        f(min(D['VERY_CLOSE'],Z['MUCH_BELOW']),vz_='FAST_CLIMB')

        return (s._defuzz(sp,s._SP,s._mf_speed),
                s._defuzz(om,s._OM,s._mf_omega),
                s._defuzz(vz,s._VZ,s._mf_vz))


# ═══════════════════════════════════════════════════════════════════════
#  TARGET MOVER  —  v10.6
# ═══════════════════════════════════════════════════════════════════════
class TargetMover:
    RISE_TO_Z=14.0; RISE_VZ=1.2; RISE_TOLERANCE=0.3; STABILIZE_TIME=1.5; MAX_TIME=300.0
    Z_FLOOR=12.0; Z_CEIL=24.0; Z_CLAMP_MARGIN=1.5
    SOFT_RADIUS=45.0; HARD_RADIUS=120.0; MAX_REPULSION_OMEGA=15.0
    SAFETY_RADIUS=3.0; BLEND_MAX_DIST=12.0
    STRAIGHT_MAX_DIST=60.0; TRAJ2_SPEED=1.0; TRAJ3_SPEED=3.5
    INCLINE_MAX_DIST=60.0   # v10.10: T6/T7 horizontal shuttle half-length (m)
    CIRCLE_R=8.; CIRCLE_T=25.; LEMN_A=8.; LEMN_T=40.
    INCLINE_A_SPEED=2.; INCLINE_A_SLOPE=15.; INCLINE_B_SPEED=3.; INCLINE_B_SLOPE=35.
    HELIX_R=8.; HELIX_T=25.; HELIX_VZ=0.15; HELIX_HALF_TIME=50.
    OU_THETA=0.40; OU_SIGMA_MAX=1.00; OU_LIMIT=math.pi/2.5; OU_PHI_URGENCY=0.75
    FOV_H_HALF=math.radians(32.6); FOV_V_HALF=math.radians(25.6)
    CHASER_MAX_YAW=0.4
    LAT_A_MIN=math.radians(50.); LAT_A_MAX=math.radians(90.)
    LAT_T_MIN=3.; LAT_T_MAX=7.
    VERT_B_MIN=0.60; VERT_B_MAX=1.20; VERT_T_MIN=4.; VERT_T_MAX=8.

    def __init__(self):
        rospy.init_node('target_mover')
        self.trajectory=int(rospy.get_param('~trajectory',9))
        # v10.7: T2/T3 straight-line azimuth mode. 'random' (default) =
        # baseline seeded draw; 'away' = directly away from the chaser
        # (world bearing chaser->target at trajectory start); a number =
        # fixed world azimuth in degrees.
        self._straight_az=str(rospy.get_param('~straight_az','random'))
        # v10.10 (trajectory audit, Rawad decision): official T2/T3 = ONE-WAY
        # straight (never reverses) with SHORTER mission durations so the path
        # fits the island — the old 60 m shuttle default was a containment
        # hack whose reversal flew the target back THROUGH the chaser.
        # ~straight_max=60 remains available as a diagnostic override.
        self._straight_max=float(rospy.get_param('~straight_max',99999.0))
        # v10.9: lateral offset between the out/return legs (racetrack) so the
        # target does not fly back THROUGH the chaser. 0 = old exact reversal.
        self._straight_offset=float(rospy.get_param('~straight_offset',0.0))
        self._lat_target=self._straight_offset/2.0
        # M-B trajectory-tracking feedback (2026-07-24): pull the REAL flown path
        # onto the EXACT formula path. v = formula_feedforward + Kp*(ideal-actual).
        # 0.0 = OFF = byte-for-byte legacy open-loop (unchanged matrix behaviour).
        self._traj_track_kp=float(rospy.get_param('~traj_track_kp',0.0))
        self._traj_track_cap=float(rospy.get_param('~traj_track_cap',2.5))
        # T5 lemniscate half-width (m), env-tunable (default = class LEMN_A=8).
        # Larger -> wider figure-8 so the chaser covers a bigger region.
        self.LEMN_A=float(rospy.get_param('~lemn_a',self.LEMN_A))
        # separate ALTITUDE-hold gain for the path-lock (2026-08-10): a P pull has
        # steady-state error against the target's slow vz drift; a stronger Z gain
        # removes the ~0.4 m climb on the flat trajectories (T1/T2/T3/T4/T5).
        # Default = traj_track_kp (no change unless set higher).
        self._traj_track_kp_z=float(rospy.get_param('~traj_track_kp_z',self._traj_track_kp))
        # integral on the altitude error -> zeros the steady-state climb (a P pull
        # alone leaves offset against the constant drift). Clamped. Default 0 = off.
        self._traj_track_ki_z=float(rospy.get_param('~traj_track_ki_z',0.0))
        self._track_int_z=0.0
        # ~z_hold (2026-08-10): on the FLAT trajectories (T1-T5) command the Z as a
        # POSITION setpoint (PX4 altitude hold) instead of velocity -> the target
        # cannot drift/climb. Default 0 = velocity Z (byte-compat).
        self._z_hold=int(rospy.get_param('~z_hold',0))
        # M-B: T6/T7 as a SINGLE straight inclined leg (no vertical bounce, no
        # horizontal shuttle). 0 = legacy bounce/shuttle. NB one-way climbs out
        # of bounds fast (T7 ~1.72 m/s up), so pair with a SHORT DURATION.
        self._incline_oneway=int(rospy.get_param('~incline_oneway',0))
        # Option B (2026-08-02): T6/T7 fly FORWARD without reversing horizontally,
        # keeping the vertical zig-zag (high-low). 0 = legacy horizontal SHUTTLE.
        # NB forward-only leaves the island in ~25-30 s at 3 m/s -> use a SHORT
        # duration or a larger zone.
        self._incline_no_hreverse=int(rospy.get_param('~incline_no_hreverse',0))
        # M-B lead compensation: aim the tracking feedback at the ideal point
        # this many seconds in the FUTURE, to cancel the drone's velocity-tracking
        # lag (the phase lag that makes the orbit radius read a few % too big).
        self._traj_track_lead=float(rospy.get_param('~traj_track_lead',0.0))
        seed = rospy.get_param('~seed', -1)
        if seed >= 0:
            random.seed(int(seed))
            rospy.loginfo("[TargetMover] Seeded with %d" % int(seed))
        if self.trajectory not in range(1,12): self.trajectory=9
        self.fuzzy=FuzzyEscaper()
        self.pos_x=self.pos_y=self.pos_z=0.; self.yaw=0.; self.got_pose=False
        self.chaser_x=self.chaser_y=self.chaser_z=0.; self.chaser_yaw=None
        self.world_x=self.world_y=self.world_z=0.
        self.chaser_world_x=self.chaser_world_y=self.chaser_world_z=0.
        self._got_world_pos=False
        self._warned_no_world_pos=False   # B6: warn-once guard
        self.phase="WAITING"; self.takeoff_ready=False; self.chaser_phase="UNKNOWN"
        self.rise_start_time=self.settle_start_time=self.motion_start_time=None
        self._traj_init_done=False; self._traj_azimuth=self._traj_phase0=0.
        self._straight_dir=1.; self._straight_start_x=self._straight_start_y=0.
        self._straight_start_z=0.; self._z_rate=0.; self._prev_pz=0.
        self._incline_vz_dir=1.
        self._rw_wps=[]; self._rw_i=0; self._rw_speeds=[2.2]   # M-E random waypoints
        self.heading=0.; self.speed_ema=0.5; self.vz_ema=0.
        self.hdg_perturb=0.; self._last_phi_high=False
        self._prev_d=None; self._d_dot_ema=0.

        # v10.6: maneuver state machine — 7 axes, phi+distance adaptive
        self._maneuver_timer=0.; self._maneuver_interval=3.0
        self._maneuver_type='VX'
        self._vy_sign=1.0;  self._vy_intensity=1.0;  self._vy_escape=0.
        self._vz_sign=1.0;  self._vz_intensity=1.0;  self._vz_escape=0.

        self._lat_half_t=0.; self._lat_cycle=0
        self._lat_A=math.radians(60.); self._lat_T=5.
        self._vert_half_t=0.; self._vert_cycle=0
        self._vert_B=0.8; self._vert_Tz=6.
        self._current_weave_offset=0.

        self.cmd_pub=rospy.Publisher(
            '/target/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.fuzzy_pub=rospy.Publisher(
            '/drone_tracking/target_fuzzy_state', Float32MultiArray, queue_size=1)
        self.target_phase_pub=rospy.Publisher(
            '/drone_tracking/target_phase', String, queue_size=1)

        rospy.Subscriber('/target/mavros/local_position/pose',
                         PoseStamped, self._target_pose_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose',
                         PoseStamped, self._chaser_pose_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase',
                         String, self._phase_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_takeoff_ready',
                         Bool, self._takeoff_ready_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states',
                         ModelStates, self._gazebo_states_cb, queue_size=1)

        N={1:"Static Hover",2:"Slow Straight",3:"Fast Straight",4:"Circle",
           5:"Lemniscate",6:"Incline Med",7:"Incline Hard",8:"Helix",
           9:"Fuzzy+Weave v10.6",10:"Random Waypoints",11:"Random Wpts+Speed"}
        rospy.loginfo("[TargetMover] v10.11 | T%d: %s | Z=[%.0f,%.0f] rise=%.0f | track_kp=%.2f"
                      % (self.trajectory, N[self.trajectory],
                         self.Z_FLOOR, self.Z_CEIL, self.RISE_TO_Z, self._traj_track_kp))
        self.rate=rospy.Rate(50); self._run()

    # ── Callbacks ─────────────────────────────────────────────────────
    def _target_pose_cb(self,m):
        self.pos_x=m.pose.position.x; self.pos_y=m.pose.position.y
        self.pos_z=m.pose.position.z
        q=m.pose.orientation
        self.yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2))
        self.got_pose=True

    def _chaser_pose_cb(self,m):
        self.chaser_x=m.pose.position.x; self.chaser_y=m.pose.position.y
        self.chaser_z=m.pose.position.z
        q=m.pose.orientation
        self.chaser_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2))

    def _phase_cb(self,m): self.chaser_phase=m.data

    def _takeoff_ready_cb(self,m):
        if m.data and not self.takeoff_ready:
            rospy.loginfo("[TargetMover] Takeoff ready"); self.takeoff_ready=True

    def _gazebo_states_cb(self,m):
        try:
            i=m.name.index('iris'); p=m.pose[i].position
            self.chaser_world_x=p.x; self.chaser_world_y=p.y; self.chaser_world_z=p.z
            q=m.pose[i].orientation
            self.chaser_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2))
        except: pass
        try:
            i=m.name.index('target_iris'); p=m.pose[i].position
            self.world_x=p.x; self.world_y=p.y; self.world_z=p.z
            self._got_world_pos=True
        except: pass
        # B6: the matching above is silent on failure. If the target world pose
        # never resolves, the fuzzy escaper computes distance from zeros. Warn
        # ONCE if still unresolved ~5s into MOVING (matching logic unchanged).
        if (not self._got_world_pos and not self._warned_no_world_pos
                and self.phase=="MOVING" and self.motion_start_time is not None
                and (rospy.Time.now()-self.motion_start_time).to_sec() > 5.0):
            self._warned_no_world_pos=True
            rospy.logwarn("[TargetMover] No world pose for target_iris/iris from "
                          "/gazebo/model_states — fuzzy distance invalid. "
                          "Check model names.")

    def send_vel(self,vx=0.,vy=0.,vz=0.,yr=0.,pz=None):
        m=PositionTarget(); m.header.stamp=rospy.Time.now()
        m.coordinate_frame=PositionTarget.FRAME_LOCAL_NED
        m.velocity.x=float(vx); m.velocity.y=float(vy); m.yaw_rate=float(yr)
        if pz is None:
            m.type_mask=VEL_YR_MASK; m.velocity.z=float(vz)
        else:                              # velocity XY + hold this exact altitude
            m.type_mask=POSZ_VELXY_MASK; m.position.z=float(pz)
        self.cmd_pub.publish(m)

    # ── Trajectory init/dispatch ──────────────────────────────────────
    def _init_trajectory(self):
        self._traj_azimuth=random.uniform(0,2*math.pi)
        # v10.7: optional T2/T3 azimuth override. The random draw above is
        # ALWAYS consumed first, so every other seeded value stays identical
        # per seed (same discipline as the v3.2 CLEAR_VIEW_YAW remap).
        # v10.10: azimuth steering extended to T6/T7 (their shuttle azimuth was
        # an uncontrollable random draw -> could point into the tree line)
        if self.trajectory in (2,3,6,7) and self._straight_az!='random':
            if self._straight_az=='away':
                if self._got_world_pos:
                    self._traj_azimuth=math.atan2(
                        self.world_y-self.chaser_world_y,
                        self.world_x-self.chaser_world_x)
                    rospy.loginfo("[TargetMover] v10.7 straight_az=away -> "
                                  "%.0f deg world (directly away from chaser)"
                                  %math.degrees(self._traj_azimuth))
                else:
                    rospy.logwarn("[TargetMover] straight_az=away but no world "
                                  "pose yet — keeping the random azimuth")
            else:
                try:
                    self._traj_azimuth=math.radians(float(self._straight_az))
                    rospy.loginfo("[TargetMover] v10.7 straight_az fixed at "
                                  "%s deg (world)"%self._straight_az)
                except ValueError:
                    rospy.logwarn("[TargetMover] bad ~straight_az '%s' — "
                                  "keeping the random azimuth"%self._straight_az)
        self._traj_phase0=random.uniform(0,2*math.pi)
        self._straight_dir=1.
        self._straight_start_x=self.pos_x; self._straight_start_y=self.pos_y
        self._straight_start_z=self.pos_z   # v10.7: z-hold reference
        self._prev_pz=self.pos_z; self._z_rate=0.
        self._incline_vz_dir=1.
        if self.trajectory==9:
            self.heading=self.yaw; self.speed_ema=0.5; self.vz_ema=0.
            self.hdg_perturb=0.; self._prev_d=None; self._d_dot_ema=0.
            self._lat_half_t=0.; self._lat_cycle=0
            self._lat_A=math.radians(random.uniform(50,90))
            self._lat_T=random.uniform(3,7)
            self._vert_half_t=0.; self._vert_cycle=0
            self._vert_B=random.uniform(self.VERT_B_MIN, self.VERT_B_MAX)
            self._vert_Tz=random.uniform(self.VERT_T_MIN, self.VERT_T_MAX)
        if self.trajectory in (10,11):
            # M-E: seeded RANDOM waypoint path — reproducible per seed, NOT
            # reactive to the chaser (unlike fuzzy T9). Fly to a sequence of
            # random points around the spawn; loop them. T10 = constant speed;
            # T11 = a RANDOM speed per segment too.
            sx,sy=self._straight_start_x,self._straight_start_y
            box=float(rospy.get_param('~rw_box',14.0))
            zlo,zhi=self.Z_FLOOR+2.0,self.Z_CEIL-2.0
            n=int(rospy.get_param('~rw_points',10))
            self._rw_wps=[(sx+random.uniform(-box,box),
                           sy+random.uniform(-box,box),
                           random.uniform(zlo,zhi)) for _ in range(n)]
            self._rw_i=0
            if self.trajectory==11:               # random speed per segment
                smin=float(rospy.get_param('~rw_speed_min',1.0))
                smax=float(rospy.get_param('~rw_speed_max',2.8))
                self._rw_speeds=[random.uniform(smin,smax) for _ in range(n)]
            else:                                 # constant speed
                self._rw_speeds=[float(rospy.get_param('~rw_speed',2.2))]*n
        self._traj_init_done=True

    def _get_traj_vel(self,elapsed,dt):
        if not self._traj_init_done: self._init_trajectory()
        t=self.trajectory
        if   t==1: vx,vy,vz,wz=0.,0.,0.,0.
        elif t==2: vx,vy,vz,wz=self._ts(self.TRAJ2_SPEED)
        elif t==3: vx,vy,vz,wz=self._ts(self.TRAJ3_SPEED)
        elif t==4: vx,vy,vz,wz=self._tc(elapsed)
        elif t==5: vx,vy,vz,wz=self._tl(elapsed)
        elif t==6: vx,vy,vz,wz=self._ti(self.INCLINE_A_SPEED,self.INCLINE_A_SLOPE)
        elif t==7: vx,vy,vz,wz=self._ti(self.INCLINE_B_SPEED,self.INCLINE_B_SLOPE)
        elif t==8: vx,vy,vz,wz=self._th(elapsed)
        elif t in (10,11): vx,vy,vz,wz=self._tr()
        elif t==9: return self._compute_fuzzy_velocity(dt)
        else: return 0.,0.,0.,0.
        # M-B trajectory-tracking position feedback: add Kp*(ideal-actual) so the
        # REAL path fits the EXACT formula. Applied only to the pure-parametric
        # trajectories whose ideal position is closed-form (T1 hold, T4 circle,
        # T5 figure-8, T8 helix-xy). T2/T3/T6/T7 already track spec (they are
        # position-referenced), so they are left byte-for-byte unchanged.
        # (2026-08-10) T2/T3 straight lines added to the closed-form path-lock so
        # they fly a clean level line (kills the open-loop climb + lateral wobble).
        if self._traj_track_kp>0.0 and t in (1,2,3,4,5,8):
            ix,iy,iz=self._ideal_pos(elapsed)
            kp,kpz,cap=self._traj_track_kp,self._traj_track_kp_z,self._traj_track_cap
            vx+=max(-cap,min(cap,kp*(ix-self.pos_x)))
            vy+=max(-cap,min(cap,kp*(iy-self.pos_y)))
            if t in (1,2,3,4,5):           # flat trajectories: hold start altitude
                ez=iz-self.pos_z
                if self._traj_track_ki_z>0.0:
                    self._track_int_z=max(-2.5,min(2.5,self._track_int_z+ez*dt))
                vz+=max(-cap,min(cap,kpz*ez+self._traj_track_ki_z*self._track_int_z))
        return vx,vy,vz,wz

    def _ideal_pos(self,e):
        """Exact formula position (target local frame) at MOVING-elapsed e, for
        the closed-form trajectories, anchored at the MOVING-start position.
        Derived by integrating the formula velocity from e=0 (pos==anchor)."""
        t=self.trajectory
        e=e+self._traj_track_lead          # lead-compensate the plant lag
        sx,sy,sz=self._straight_start_x,self._straight_start_y,self._straight_start_z
        if t in (4,8):
            R=self.CIRCLE_R if t==4 else self.HELIX_R
            T=self.CIRCLE_T if t==4 else self.HELIX_T
            w=2*math.pi/T; p0=self._traj_phase0
            return (sx+R*(math.cos(w*e+p0)-math.cos(p0)),
                    sy+R*(math.sin(w*e+p0)-math.sin(p0)), sz)
        if t==5:
            w=2*math.pi/self.LEMN_T
            return (sx+self.LEMN_A*math.sin(w*e),
                    sy+0.5*self.LEMN_A*math.sin(2*w*e), sz)
        if t in (2,3):                     # straight line: start + speed*t along az
            spd=self.TRAJ2_SPEED if t==2 else self.TRAJ3_SPEED
            az=self._traj_azimuth
            return (sx+spd*e*math.cos(az), sy+spd*e*math.sin(az), sz)
        return sx,sy,sz                    # T1 hold at anchor

    def _tr(self):
        # M-E random-waypoint law: steer toward the current random waypoint at
        # rw_speed; advance (and loop) when within 1.5 m. Uses only own position,
        # so it is deterministic per seed and NOT reactive to the chaser.
        if not self._rw_wps:
            return 0., 0., 0., 0.
        wx, wy, wz = self._rw_wps[self._rw_i]
        dx, dy, dz = wx - self.pos_x, wy - self.pos_y, wz - self.pos_z
        if math.sqrt(dx*dx + dy*dy + dz*dz) < 1.5:
            self._rw_i = (self._rw_i + 1) % len(self._rw_wps)
            wx, wy, wz = self._rw_wps[self._rw_i]
            dx, dy, dz = wx - self.pos_x, wy - self.pos_y, wz - self.pos_z
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < 1e-3:
            return 0., 0., 0., 0.
        s = self._rw_speeds[self._rw_i]
        vz = max(-1.5, min(1.5, s * dz / d))
        return s * dx / d, s * dy / d, vz, 0.

    # ── Trajectories 1-8 ─────────────────────────────────────────────
    def _ts(self,spd):
        az=self._traj_azimuth
        # along-track distance travelled from the leg start
        dx=self.pos_x-self._straight_start_x; dy=self.pos_y-self._straight_start_y
        along=dx*math.cos(az)+dy*math.sin(az)
        if abs(along)>self._straight_max and self._straight_dir==1.:
            self._straight_dir=-1.
            # v10.9: on reversal, offset the return leg LATERALLY so the target
            # does NOT retrace straight back THROUGH the chaser (that head-on
            # pass made a 6-8 m band + 4 m floor physically impossible). It flips
            # between +/- straight_offset/2, forming a racetrack of parallel
            # lines. straight_offset=0 -> exact old back-and-forth.
            self._lat_target=-self._lat_target
        elif abs(along)<2. and self._straight_dir==-1.:
            self._straight_dir=1.
            self._lat_target=-self._lat_target
        v=spd*self._straight_dir
        # perpendicular steering toward the current lateral offset line
        px,py=-math.sin(az),math.cos(az)                 # unit perpendicular
        lat_pos=dx*px+dy*py
        v_perp=max(-1.0,min(1.0,0.6*(self._lat_target-lat_pos)))
        vx=v*math.cos(az)+v_perp*px
        vy=v*math.sin(az)+v_perp*py
        # v10.8: DAMPED (PD) altitude hold (kills the ~0.7 m altitude wave).
        z_err=self._straight_start_z-self.pos_z
        self._z_rate=0.6*self._z_rate+0.4*(self.pos_z-self._prev_pz)/0.02
        self._prev_pz=self.pos_z
        vz=max(-0.7,min(0.7, 0.8*z_err - 0.5*self._z_rate))
        return vx,vy,vz,0.

    def _tc(self,e):
        w=2*math.pi/self.CIRCLE_T; p=w*e+self._traj_phase0
        return -self.CIRCLE_R*w*math.sin(p),self.CIRCLE_R*w*math.cos(p),0.,0.

    def _tl(self,e):
        w=2*math.pi/self.LEMN_T
        return self.LEMN_A*w*math.cos(w*e),self.LEMN_A*w*math.cos(2*w*e),0.,0.

    def _ti(self,spd,sl_deg):
        # v10.10 (trajectory audit): T6/T7 horizontal leg is now a SHUTTLE
        # bounded at INCLINE_MAX_DIST (they flew one-way on a random azimuth
        # and left the island inside a 300 s run — T1-T8 have NO boundary
        # repulsion, that layer is T9-only). Vertical bounce unchanged.
        sl=math.radians(sl_deg); lat=spd*math.cos(sl); vb=spd*math.sin(sl)
        if self._incline_oneway:            # Option A: forward + climb, then level
            az=self._traj_azimuth           # climb to the ceiling, then cruise fwd
            vz=vb if self.pos_z<self.Z_CEIL-1 else 0.0
            return lat*math.cos(az), lat*math.sin(az), vz, 0.
        if self._incline_vz_dir>0 and self.pos_z>=self.Z_CEIL-1:
            self._incline_vz_dir=-1.
        elif self._incline_vz_dir<0 and self.pos_z<=self.Z_FLOOR+1:
            self._incline_vz_dir=1.
        az=self._traj_azimuth
        if self._incline_no_hreverse:       # Option B: forward + vertical zig-zag
            return (lat*math.cos(az), lat*math.sin(az),
                    vb*self._incline_vz_dir, 0.)
        dx=self.pos_x-self._straight_start_x; dy=self.pos_y-self._straight_start_y
        along=dx*math.cos(az)+dy*math.sin(az)
        if abs(along)>self.INCLINE_MAX_DIST and self._straight_dir==1.:
            self._straight_dir=-1.
        elif abs(along)<2. and self._straight_dir==-1.:
            self._straight_dir=1.
        return (lat*math.cos(az)*self._straight_dir,
                lat*math.sin(az)*self._straight_dir,
                vb*self._incline_vz_dir, 0.)

    def _th(self,e):
        # v10.10 (trajectory audit): vz was a single up-then-down keyed on the
        # hardcoded HELIX_HALF_TIME=50 s — after ~130 s the target sat on the
        # Z floor and T8 degenerated into a FLAT circle (a T4 duplicate) for
        # the rest of the run. Now a continuous triangle-wave bounce between
        # the Z bounds, matching the spec "continuous ascent+descent".
        w=2*math.pi/self.HELIX_T; p=w*e+self._traj_phase0
        vx=-self.HELIX_R*w*math.sin(p); vy=self.HELIX_R*w*math.cos(p)
        if self._incline_vz_dir>0 and self.pos_z>=self.Z_CEIL-1:
            self._incline_vz_dir=-1.
        elif self._incline_vz_dir<0 and self.pos_z<=self.Z_FLOOR+1:
            self._incline_vz_dir=1.
        return vx,vy,self.HELIX_VZ*self._incline_vz_dir,0.

    # ── Fuzzy helpers ─────────────────────────────────────────────────
    def _get_3d_distance(self):
        return math.sqrt(
            (self.world_x-self.chaser_world_x)**2 +
            (self.world_y-self.chaser_world_y)**2 +
            (self.world_z-self.chaser_world_z)**2)

    def _get_closing_rate(self,d):
        if self._prev_d is None: self._prev_d=d; return 0.
        raw=(self._prev_d-d)/(1./50.)
        self._d_dot_ema=.85*self._d_dot_ema+.15*raw
        self._prev_d=d; return self._d_dot_ema

    def _get_fov_exposure(self):
        if self.chaser_yaw is None: return 0.5
        dx=self.world_x-self.chaser_world_x
        dy=self.world_y-self.chaser_world_y
        dz=self.world_z-self.chaser_world_z
        gd=math.hypot(dx,dy)
        b=math.atan2(dy,dx)
        ha=abs(math.atan2(math.sin(b-self.chaser_yaw),math.cos(b-self.chaser_yaw)))
        va=abs(math.atan2(dz,gd)) if gd>0.1 else 0.
        return max(0,1-ha/self.FOV_H_HALF)*max(0,1-va/self.FOV_V_HALF)

    def _update_hdg_perturb(self,d,dt,phi):
        sc=max(min(1.,d/self.BLEND_MAX_DIST),phi)
        self.hdg_perturb+=(-self.OU_THETA*self.hdg_perturb*dt +
                            self.OU_SIGMA_MAX*sc*random.gauss(0,math.sqrt(dt)))
        if phi>self.OU_PHI_URGENCY:
            u=(phi-self.OU_PHI_URGENCY)/(1-self.OU_PHI_URGENCY)
            tp=self.OU_LIMIT*math.copysign(1,self.hdg_perturb if self.hdg_perturb else 1)
            self.hdg_perturb+=u*.6*(tp-self.hdg_perturb)*dt*50
        self.hdg_perturb=max(-self.OU_LIMIT,min(self.OU_LIMIT,self.hdg_perturb))

    def _compute_desired_heading(self):
        dx=self.world_x-self.chaser_world_x
        dy=self.world_y-self.chaser_world_y
        if math.hypot(dx,dy)<0.1: return self.heading
        a=math.atan2(dy,dx)
        return math.atan2(math.sin(a+self.hdg_perturb),math.cos(a+self.hdg_perturb))

    def _clamp_vz(self,vz):
        m=self.Z_CLAMP_MARGIN
        if vz<0 and self.pos_z<self.Z_FLOOR+m:
            vz*=max(0,(self.pos_z-self.Z_FLOOR)/m)
        if vz>0 and self.pos_z>self.Z_CEIL-m:
            vz*=max(0,(self.Z_CEIL-self.pos_z)/m)
        return vz

    def _repulsion_omega(self,x,y):
        d=math.hypot(x,y)
        if d<=self.SOFT_RADIUS or d<0.1: return 0.
        t=min(1,(d-self.SOFT_RADIUS)/max(self.HARD_RADIUS-self.SOFT_RADIUS,1))
        df=math.atan2(math.sin(math.atan2(-y,-x)-self.heading),
                      math.cos(math.atan2(-y,-x)-self.heading))
        mo=math.radians(self.MAX_REPULSION_OMEGA)
        return max(-mo,min(mo,t*mo*(2/math.pi)*df))

    def _chaser_avoidance_omega(self):
        dx=self.chaser_world_x-self.world_x; dy=self.chaser_world_y-self.world_y
        d=math.hypot(dx,dy)
        if d>self.SAFETY_RADIUS*2 or d<0.1: return 0.
        df=math.atan2(math.sin(math.atan2(dy,dx)-self.heading),
                      math.cos(math.atan2(dy,dx)-self.heading))
        if abs(df)>math.pi/2: return 0.
        return -math.copysign((1-d/(self.SAFETY_RADIUS*2))*math.radians(25),df)

    def _anti_headon_omega(self):
        dx=self.chaser_world_x-self.world_x; dy=self.chaser_world_y-self.world_y
        d=math.hypot(dx,dy)
        if d>25 or d<0.1: return 0.
        df=math.atan2(math.sin(math.atan2(dy,dx)-self.heading),
                      math.cos(math.atan2(dy,dx)-self.heading))
        c=math.radians(60)
        if abs(df)>c: return 0.
        return -math.copysign((1-abs(df)/c)*(.3+.7*(1-d/25))*math.radians(30),df)

    # ── Weave (distance-adaptive) ──────────────────────────────────────
    def _lat_weave_offset(self,dt,d):
        self._lat_half_t+=dt
        if self._lat_half_t>=self._lat_T:
            self._lat_half_t-=self._lat_T; self._lat_cycle+=1
            d_scale=min(1.,max(0.,(d-3.)/5.))
            base_A=self.LAT_A_MIN+d_scale*(self.LAT_A_MAX-self.LAT_A_MIN)
            self._lat_A=base_A+math.radians(random.uniform(-8,8))
            self._lat_A=max(self.LAT_A_MIN,min(self.LAT_A_MAX,self._lat_A))
            self._lat_T=random.uniform(self.LAT_T_MIN,self.LAT_T_MAX)
        phase=math.pi*self._lat_half_t/self._lat_T
        sign=1. if self._lat_cycle%2==0 else -1.
        offset=sign*self._lat_A*math.sin(phase)
        self._current_weave_offset=offset
        return offset

    def _vert_weave_vz(self,dt):
        self._vert_half_t+=dt
        if self._vert_half_t>=self._vert_Tz:
            self._vert_half_t-=self._vert_Tz; self._vert_cycle+=1
            self._vert_B=random.uniform(self.VERT_B_MIN,self.VERT_B_MAX)
            self._vert_Tz=random.uniform(self.VERT_T_MIN,self.VERT_T_MAX)
        phase=math.pi*self._vert_half_t/self._vert_Tz
        sign=1. if self._vert_cycle%2==0 else -1.
        return sign*self._vert_B*math.sin(phase)

    # ── Fuzzy velocity engine — v10.6 ─────────────────────────────────
    def _compute_fuzzy_velocity(self,dt):
        d      = self._get_3d_distance()
        d_dot  = self._get_closing_rate(d)
        phi    = self._get_fov_exposure()
        delta_z= self.chaser_world_z - self.world_z
        f_speed, f_omega, f_vz = self.fuzzy.infer(d, d_dot, phi, delta_z)

        # Altitude escape boost when chaser clearly has us in FOV.
        # B1 FIX (was phi>0.40): with the 0.40 threshold, au=(phi-.60)/.40 was
        # NEGATIVE for phi in (0.40,0.60), flipping the escape direction TOWARD
        # the chaser AND overwriting all 9 fuzzy altitude rules. Threshold 0.60
        # makes au in [0,1]; for phi<=0.60 the fuzzy vz rules apply untouched.
        if phi > 0.60:
            au = (phi - .60) / .40
            if abs(delta_z) > .5:
                ed = -math.copysign(1., delta_z)
            else:
                ed = 1. if (self.Z_CEIL-self.pos_z) >= (self.pos_z-self.Z_FLOOR) else -1.
            f_vz = ed * 1.3 * au                  # overwrite only when boost active

        # Heading update
        self._update_hdg_perturb(d, dt, phi)
        wo = self._lat_weave_offset(dt, d)
        base = self._compute_desired_heading()
        target_hdg = math.atan2(math.sin(base+wo), math.cos(base+wo))
        he = math.atan2(math.sin(target_hdg-self.heading),
                        math.cos(target_hdg-self.heading))
        self.heading += max(-f_omega*dt, min(f_omega*dt, he))
        self.heading += (self._repulsion_omega(self.pos_x, self.pos_y) +
                         self._chaser_avoidance_omega() +
                         self._anti_headon_omega()) * dt
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))

        # ── v10.6: 4-tier distance + phi-urgency adaptive maneuvers ───
        self._maneuver_timer += dt
        if self._maneuver_timer >= self._maneuver_interval:
            self._maneuver_timer = 0.
            # 4-tier distance-based base interval
            if d < 3.0:
                base_int = random.uniform(2.0, 3.0)
                self._vy_intensity = random.uniform(1.0, 1.5)
                self._vz_intensity = random.uniform(1.0, 1.5)
            elif d < 5.0:
                base_int = random.uniform(2.5, 4.0)
                self._vy_intensity = random.uniform(1.2, 1.8)
                self._vz_intensity = random.uniform(1.2, 1.8)
            elif d < 8.0:
                base_int = random.uniform(3.5, 5.5)
                self._vy_intensity = random.uniform(1.5, 2.5)
                self._vz_intensity = random.uniform(1.5, 2.5)
            else:
                base_int = random.uniform(4.5, 7.0)
                self._vy_intensity = random.uniform(2.5, 3.5)
                self._vz_intensity = random.uniform(1.5, 2.5)
            # Phi-urgency: faster maneuver changes when chaser has target in FOV
            phi_factor = 0.6 + 0.7 * (1.0 - phi)
            self._maneuver_interval = max(0.8, base_int * phi_factor)
            # Pick escape axis — favour lateral and combined
            types  = ['VX',  'VY',  'VZ',  'VX_VY', 'VY_VZ', 'VX_VZ', 'VX_VY_VZ']
            weights= [0.04,  0.22,  0.10,   0.22,    0.20,    0.08,    0.14]
            self._maneuver_type = random.choices(types, weights=weights)[0]
            self._vy_sign = random.choice([-1., 1.])
            self._vz_sign = random.choice([-1., 1.])
            rospy.loginfo(
                "[TargetMover] v10.6 Maneuver: %-10s d=%.1fm phi=%.2f vy=%+.1f vz=%+.1f int=%.1fs"
                % (self._maneuver_type, d, phi,
                   self._vy_sign * self._vy_intensity,
                   self._vz_sign * self._vz_intensity,
                   self._maneuver_interval))

        # EMA-ramp vy and vz strafe toward targets
        phi_scale = max(0.5, phi)
        vy_target  = (self._vy_sign * self._vy_intensity * phi_scale
                      if 'VY' in self._maneuver_type else 0.)
        vz_m_target= (self._vz_sign * self._vz_intensity * phi_scale
                      if 'VZ' in self._maneuver_type else 0.)
        self._vy_escape += 0.08 * (vy_target    - self._vy_escape)
        self._vz_escape += 0.10 * (vz_m_target  - self._vz_escape)

        # Speed: floor 1.0 m/s, clamp 3.5 m/s (v10.6)
        effective_speed = max(f_speed, 1.00)
        effective_speed = min(effective_speed, 3.5)
        self.speed_ema += 0.10 * (effective_speed - self.speed_ema)

        # Vertical: fuzzy vz + weave + maneuver vz dodge
        vz_combined = f_vz + self._vert_weave_vz(dt)
        self.vz_ema += 0.12 * (self._clamp_vz(vz_combined) - self.vz_ema)

        # Final velocities
        vx = self.speed_ema * math.cos(self.heading)
        vy = self.speed_ema * math.sin(self.heading) + self._vy_escape
        vz = self.vz_ema + self._vz_escape

        # Z safety clamps
        if self.pos_z < self.Z_FLOOR + 0.4 and vz < 0.: vz = 0.20
        if self.pos_z > self.Z_CEIL  - 0.4 and vz > 0.: vz = -0.20

        # Yaw rate
        ye = math.atan2(math.sin(self.heading-self.yaw),
                        math.cos(self.heading-self.yaw))
        ys = math.copysign(1., he) if abs(he) > 0.01 else 0.
        yr = ys * f_omega * .3 + 1.5 * ye

        # Velocity clamps (v10.6: vx/vy ±3.5, vz ±2.0)
        vx = max(-3.5, min(3.5, vx))
        vy = max(-3.5, min(3.5, vy))
        vz = max(-2.0, min(2.0, vz))
        yr = max(-.8,  min(.8,  yr))

        # Publish fuzzy state for analyzer
        fs = Float32MultiArray()
        fs.data = [float(d), float(d_dot), float(phi),
                   float(math.degrees(self.hdg_perturb)),
                   float(f_speed), float(f_omega), float(f_vz)]
        self.fuzzy_pub.publish(fs)
        return vx, vy, vz, yr

    # ── Main loop ──────────────────────────────────────────────────────
    def _run(self):
        dt = 1. / 50.
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.phase == "WAITING":
                if self.takeoff_ready:
                    self.rise_start_time = now
                    self.heading = self.yaw
                    self.phase = "RISING"
                    rospy.loginfo("[TargetMover] Takeoff done — rising to %.1fm"
                                  % self.RISE_TO_Z)
            elif self.phase == "RISING":
                ae = self.RISE_TO_Z - self.pos_z
                if ae > self.RISE_TOLERANCE:
                    self.send_vel(0.2*math.cos(self.heading),
                                  0.2*math.sin(self.heading),
                                  min(self.RISE_VZ, ae*.5))
                else:
                    self.settle_start_time = now; self.phase = "SETTLING"
                    rospy.loginfo("[TargetMover] At %.1fm — settling" % self.pos_z)
            elif self.phase == "SETTLING":
                self.send_vel(0, 0, 0)
                if (now-self.settle_start_time).to_sec() >= self.STABILIZE_TIME:
                    self.phase = "MOVING"; self.motion_start_time = now
                    rospy.loginfo("[TargetMover] *** T%d STARTED z=%.1f ***"
                                  % (self.trajectory, self.pos_z))
            elif self.phase == "MOVING":
                elapsed = (now - self.motion_start_time).to_sec()
                vx, vy, vz, yr = self._get_traj_vel(elapsed, dt)
                # z_hold: on the flat trajectories command the moving-start altitude
                # as a POSITION setpoint so PX4 holds it (no drift). vz is ignored.
                pz = (self._straight_start_z
                      if (self._z_hold and self.trajectory in (1,2,3,4,5)) else None)
                self.send_vel(vx, vy, vz, yr, pz=pz)
                rospy.loginfo_throttle(5,
                    "[TargetMover] T%d t=%.0fs pos=(%.1f,%.1f,%.1f) v=(%.1f,%.1f,%+.1f)"
                    % (self.trajectory, elapsed,
                       self.pos_x, self.pos_y, self.pos_z, vx, vy, vz))
                if elapsed >= self.MAX_TIME: self.phase = "END"
            elif self.phase == "END":
                self.send_vel(0, 0, 0)
                rospy.loginfo_throttle(10, "[TargetMover] END.")
            self.target_phase_pub.publish(String(data=self.phase))
            self.rate.sleep()


if __name__ == '__main__':
    try:    TargetMover()
    except rospy.ROSInterruptException: pass