#!/usr/bin/env python3
"""
target_mover.py  —  v10.2  Always-Escape + 7-Axis Maneuver System
===================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v10.2 changes (on top of v10.1):
  FUZZY RULES — "Always Escape" principle:
    Previous: target slowed to SLOW/NONE at EDGE/FAR → contradiction
              (target escapes to edge, then stops, lets chaser re-center)
    New:      target ALWAYS escapes at ≥ NORMAL speed except VERY_FAR
              Intensity = phi × closeness:
                CENTERED+CLOSE  → SPRINT/MAX     (maximum urgency)
                CENTERED+MEDIUM → SPRINT/SHARP   (high urgency)
                CENTERED+FAR    → FAST/MODERATE  (moderate urgency)
                EDGE+CLOSE      → FAST/MODERATE  (still escaping)
                EDGE+MEDIUM     → NORMAL/MODERATE (edge = partial success)
                EDGE+FAR        → NORMAL/GENTLE  (far+edge = success, maintain)
                OUTSIDE         → FAST/GENTLE    (escaped, keep going!)
                VERY_FAR        → SLOW/NONE      (only stop here for re-engage)

  MANEUVER SYSTEM — 7-axis distance-adaptive:
    7 escape methods: VX, VY, VZ, VX_VY, VY_VZ, VX_VZ, VX_VY_VZ
    Distance-adaptive intensity:
      CLOSE (d<5m):   vy=[1.0,1.5]  vz=[0.5,1.0]  interval=[2,4]s
      MEDIUM (5-8m):  vy=[1.5,2.0]  vz=[0.8,1.2]  interval=[2,5]s
      FAR (d>8m):     vy=[2.0,3.0]  vz=[1.0,1.5]  interval=[3,6]s
    Both vy_escape and vz_escape EMA-ramped (smooth, no stops)
    Removed _fov_min_speed (was binary sprint/drift source)

v10.1 changes (preserved):
  Floor: 0.70→1.00 m/s, SPRINT center 2.80→3.20, vy=[1.5,2.0]

v10.0 changes (preserved):
  Maneuver state machine added, vy strafe, speed EMA 0.20→0.10

v9.9 changes (preserved):
  Z_FLOOR=10.0, RISE_TO_Z=12.0, Z_CEIL=20.0

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

# ═══════════════════════════════════════════════════════════════════════
#  FUZZY ESCAPER  —  v10.2  (Always-Escape rules)
# ═══════════════════════════════════════════════════════════════════════
class FuzzyEscaper:
    _SP=[0.50+i*3./60 for i in range(61)]
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
        # v10.1: raised NORMAL center 1.20→1.40, FAST 2.00→2.30, SPRINT 2.80→3.20
        return {'SLOW':R(v,.50,.70,.90),'NORMAL':R(v,.80,1.40,1.80),
                'FAST':R(v,1.60,2.30,3.00),'SPRINT':T(v,2.80,3.20,3.80,3.80)}[t]
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
        # Only VERY_FAR uses SLOW (re-engagement). All others: NORMAL minimum.
        f(D['VERY_CLOSE'],sp_='SPRINT',om_='MAX')                           # d<3m: max panic
        f(min(D['CLOSE'],V['FAST_CLOSING']),sp_='SPRINT',om_='SHARP')      # closing fast
        f(min(D['CLOSE'],F['CENTERED']),sp_='SPRINT',om_='SHARP')          # close+centered
        f(min(D['CLOSE'],F['PARTIAL']),sp_='SPRINT',om_='MODERATE')        # close+partial: upgraded
        f(min(D['CLOSE'],F['EDGE']),sp_='FAST',om_='MODERATE')             # close+edge: was NORMAL
        f(min(D['MEDIUM'],V['FAST_CLOSING']),sp_='SPRINT',om_='MODERATE')  # upgraded from FAST
        f(min(D['MEDIUM'],F['CENTERED']),sp_='SPRINT',om_='SHARP')         # med+centered
        f(min(D['MEDIUM'],F['PARTIAL']),sp_='FAST',om_='MODERATE')         # med+partial: +omega
        f(min(D['MEDIUM'],F['EDGE']),sp_='NORMAL',om_='MODERATE')          # was SLOW: fixed
        f(min(D['FAR'],V['FAST_CLOSING']),sp_='FAST',om_='GENTLE')         # upgraded from NORMAL
        f(min(D['FAR'],F['CENTERED']),sp_='FAST',om_='MODERATE')           # upgraded from NORMAL
        f(min(D['FAR'],F['PARTIAL']),sp_='NORMAL',om_='GENTLE')            # was SLOW: fixed
        f(min(D['FAR'],F['EDGE']),sp_='NORMAL',om_='GENTLE')               # was SLOW: fixed
        f(min(D['FAR'],V['RECEDING']),sp_='NORMAL',om_='GENTLE')           # was SLOW: fixed
        f(D['VERY_FAR'],sp_='SLOW',om_='NONE')                             # ONLY slow: re-engage
        f(F['OUTSIDE'],sp_='FAST',om_='GENTLE')                            # escaped: keep going!
        f(min(V['FAST_CLOSING'],F['CENTERED']),sp_='SPRINT',om_='MAX')     # emergency sprint
        f(min(D['VERY_FAR'],F['CENTERED']),sp_='NORMAL',om_='GENTLE')      # very far+centered: was SLOW
        f(min(F['CENTERED'],V['STABLE']),sp_='FAST',om_='SHARP')           # centered+stable: escape
        f(min(F['CENTERED'],V['CLOSING']),sp_='SPRINT',om_='MAX')          # centered+closing: max

        # ── Altitude escape rules (unchanged) ─────────────────────────
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
#  TARGET MOVER  —  v10.2
# ═══════════════════════════════════════════════════════════════════════
class TargetMover:
    RISE_TO_Z=12.0; RISE_VZ=0.8; RISE_TOLERANCE=0.3; STABILIZE_TIME=3.0; MAX_TIME=300.0
    Z_FLOOR=10.0; Z_CEIL=20.0; Z_CLAMP_MARGIN=1.5
    SOFT_RADIUS=45.0; HARD_RADIUS=120.0; MAX_REPULSION_OMEGA=15.0
    SAFETY_RADIUS=3.0; BLEND_MAX_DIST=12.0
    STRAIGHT_MAX_DIST=60.0; TRAJ2_SPEED=1.0; TRAJ3_SPEED=3.5
    CIRCLE_R=8.; CIRCLE_T=25.; LEMN_A=8.; LEMN_T=40.
    INCLINE_A_SPEED=2.; INCLINE_A_SLOPE=15.; INCLINE_B_SPEED=3.; INCLINE_B_SLOPE=35.
    HELIX_R=8.; HELIX_T=25.; HELIX_VZ=0.15; HELIX_HALF_TIME=50.
    OU_THETA=0.40; OU_SIGMA_MAX=0.70; OU_LIMIT=math.pi/3.0; OU_PHI_URGENCY=0.75
    FOV_H_HALF=math.radians(32.6); FOV_V_HALF=math.radians(25.6)
    CHASER_MAX_YAW=0.4
    LAT_A_MIN=math.radians(40.); LAT_A_MAX=math.radians(75.)
    LAT_T_MIN=3.; LAT_T_MAX=7.
    VERT_B_MIN=0.40; VERT_B_MAX=0.80; VERT_T_MIN=5.; VERT_T_MAX=10.

    def __init__(self):
        rospy.init_node('target_mover')
        self.trajectory=int(rospy.get_param('~trajectory',9))
        if self.trajectory not in range(1,10): self.trajectory=9
        self.fuzzy=FuzzyEscaper()
        self.pos_x=self.pos_y=self.pos_z=0.; self.yaw=0.; self.got_pose=False
        self.chaser_x=self.chaser_y=self.chaser_z=0.; self.chaser_yaw=None
        self.world_x=self.world_y=self.world_z=0.
        self.chaser_world_x=self.chaser_world_y=self.chaser_world_z=0.
        self._got_world_pos=False
        self.phase="WAITING"; self.takeoff_ready=False; self.chaser_phase="UNKNOWN"
        self.rise_start_time=self.settle_start_time=self.motion_start_time=None
        self._traj_init_done=False; self._traj_azimuth=self._traj_phase0=0.
        self._straight_dir=1.; self._straight_start_x=self._straight_start_y=0.
        self._incline_vz_dir=1.
        self.heading=0.; self.speed_ema=0.5; self.vz_ema=0.
        self.hdg_perturb=0.; self._last_phi_high=False
        self._prev_d=None; self._d_dot_ema=0.

        # v10.2: maneuver state machine — 7 axes, distance-adaptive
        self._maneuver_timer=0.; self._maneuver_interval=3.0
        self._maneuver_type='VX'
        self._vy_sign=1.0;  self._vy_intensity=1.0;  self._vy_escape=0.
        self._vz_sign=1.0;  self._vz_intensity=0.8;  self._vz_escape=0.

        self._lat_half_t=0.; self._lat_cycle=0
        self._lat_A=math.radians(55.); self._lat_T=5.
        self._vert_half_t=0.; self._vert_cycle=0
        self._vert_B=0.5; self._vert_Tz=7.
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
           9:"Fuzzy+Weave v10.2"}
        rospy.loginfo("[TargetMover] v10.2 | T%d: %s | Z=[%.0f,%.0f] rise=%.0f"
                      % (self.trajectory, N[self.trajectory],
                         self.Z_FLOOR, self.Z_CEIL, self.RISE_TO_Z))
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

    def send_vel(self,vx=0.,vy=0.,vz=0.,yr=0.):
        m=PositionTarget(); m.header.stamp=rospy.Time.now()
        m.coordinate_frame=PositionTarget.FRAME_LOCAL_NED
        m.type_mask=VEL_YR_MASK
        m.velocity.x=float(vx); m.velocity.y=float(vy)
        m.velocity.z=float(vz); m.yaw_rate=float(yr)
        self.cmd_pub.publish(m)

    # ── Trajectory init/dispatch ──────────────────────────────────────
    def _init_trajectory(self):
        self._traj_azimuth=random.uniform(0,2*math.pi)
        self._traj_phase0=random.uniform(0,2*math.pi)
        self._straight_dir=1.
        self._straight_start_x=self.pos_x; self._straight_start_y=self.pos_y
        self._incline_vz_dir=1.
        if self.trajectory==9:
            self.heading=self.yaw; self.speed_ema=0.5; self.vz_ema=0.
            self.hdg_perturb=0.; self._prev_d=None; self._d_dot_ema=0.
            self._lat_half_t=0.; self._lat_cycle=0
            self._lat_A=math.radians(random.uniform(40,75))
            self._lat_T=random.uniform(3,7)
            self._vert_half_t=0.; self._vert_cycle=0
            self._vert_B=random.uniform(.4,.8)
            self._vert_Tz=random.uniform(5,10)
        self._traj_init_done=True

    def _get_traj_vel(self,elapsed,dt):
        if not self._traj_init_done: self._init_trajectory()
        t=self.trajectory
        if   t==1: return 0.,0.,0.,0.
        elif t==2: return self._ts(self.TRAJ2_SPEED)
        elif t==3: return self._ts(self.TRAJ3_SPEED)
        elif t==4: return self._tc(elapsed)
        elif t==5: return self._tl(elapsed)
        elif t==6: return self._ti(self.INCLINE_A_SPEED,self.INCLINE_A_SLOPE)
        elif t==7: return self._ti(self.INCLINE_B_SPEED,self.INCLINE_B_SLOPE)
        elif t==8: return self._th(elapsed)
        elif t==9: return self._compute_fuzzy_velocity(dt)
        return 0.,0.,0.,0.

    # ── Trajectories 1-8 ─────────────────────────────────────────────
    def _ts(self,spd):
        dx=self.pos_x-self._straight_start_x; dy=self.pos_y-self._straight_start_y
        if math.hypot(dx,dy)>self.STRAIGHT_MAX_DIST and self._straight_dir==1.:
            self._straight_dir=-1.
        elif math.hypot(dx,dy)<2. and self._straight_dir==-1.:
            self._straight_dir=1.
        v=spd*self._straight_dir
        return v*math.cos(self._traj_azimuth),v*math.sin(self._traj_azimuth),0.,0.

    def _tc(self,e):
        w=2*math.pi/self.CIRCLE_T; p=w*e+self._traj_phase0
        return -self.CIRCLE_R*w*math.sin(p),self.CIRCLE_R*w*math.cos(p),0.,0.

    def _tl(self,e):
        w=2*math.pi/self.LEMN_T
        return self.LEMN_A*w*math.cos(w*e),self.LEMN_A*w*math.cos(2*w*e),0.,0.

    def _ti(self,spd,sl_deg):
        sl=math.radians(sl_deg); lat=spd*math.cos(sl); vb=spd*math.sin(sl)
        if self._incline_vz_dir>0 and self.pos_z>=self.Z_CEIL-1:
            self._incline_vz_dir=-1.
        elif self._incline_vz_dir<0 and self.pos_z<=self.Z_FLOOR+1:
            self._incline_vz_dir=1.
        return (lat*math.cos(self._traj_azimuth),
                lat*math.sin(self._traj_azimuth),
                vb*self._incline_vz_dir, 0.)

    def _th(self,e):
        w=2*math.pi/self.HELIX_T; p=w*e+self._traj_phase0
        vx=-self.HELIX_R*w*math.sin(p); vy=self.HELIX_R*w*math.cos(p)
        vz=self.HELIX_VZ if e<self.HELIX_HALF_TIME else -self.HELIX_VZ
        if self.pos_z>=self.Z_CEIL and vz>0:  vz=0.
        if self.pos_z<=self.Z_FLOOR and vz<0: vz=0.
        return vx,vy,vz,0.

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

    # ── Fuzzy velocity engine — v10.2 ─────────────────────────────────
    def _compute_fuzzy_velocity(self,dt):
        d      = self._get_3d_distance()
        d_dot  = self._get_closing_rate(d)
        phi    = self._get_fov_exposure()
        delta_z= self.chaser_world_z - self.world_z
        f_speed, f_omega, f_vz = self.fuzzy.infer(d, d_dot, phi, delta_z)

        # Altitude escape boost when chaser is looking
        if phi > 0.60:
            au = (phi - .60) / .40
            if abs(delta_z) > .5:
                ed = -math.copysign(1., delta_z)
            else:
                ed = 1. if (self.Z_CEIL-self.pos_z) >= (self.pos_z-self.Z_FLOOR) else -1.
            f_vz = ed * 1.0 * au

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

        # ── v10.2: 7-axis maneuver state machine — distance-adaptive ──
        self._maneuver_timer += dt
        if self._maneuver_timer >= self._maneuver_interval:
            self._maneuver_timer = 0.
            # Distance-adaptive interval and intensity
            if d < 5.0:
                self._maneuver_interval = random.uniform(2., 4.)
                self._vy_intensity = random.uniform(1.0, 1.5)
                self._vz_intensity = random.uniform(0.5, 1.0)
            elif d < 8.0:
                self._maneuver_interval = random.uniform(2., 5.)
                self._vy_intensity = random.uniform(1.5, 2.0)
                self._vz_intensity = random.uniform(0.8, 1.2)
            else:
                self._maneuver_interval = random.uniform(3., 6.)
                self._vy_intensity = random.uniform(2.0, 3.0)
                self._vz_intensity = random.uniform(1.0, 1.5)
            # Pick one of 7 escape axes
            types  = ['VX',  'VY',  'VZ',  'VX_VY', 'VY_VZ', 'VX_VZ', 'VX_VY_VZ']
            weights= [0.08,  0.18,  0.10,   0.22,    0.16,    0.10,    0.16]
            self._maneuver_type = random.choices(types, weights=weights)[0]
            self._vy_sign = random.choice([-1., 1.])
            self._vz_sign = random.choice([-1., 1.])
            rospy.loginfo(
                "[TargetMover] v10.2 Maneuver: %-10s d=%.1fm vy=%+.1f vz=%+.1f int=%.1fs"
                % (self._maneuver_type, d,
                   self._vy_sign * self._vy_intensity,
                   self._vz_sign * self._vz_intensity,
                   self._maneuver_interval))

        # EMA-ramp vy and vz strafe toward targets
        phi_scale = max(0.5, phi)
        vy_target  = (self._vy_sign * self._vy_intensity * phi_scale
                      if 'VY' in self._maneuver_type else 0.)
        vz_m_target= (self._vz_sign * self._vz_intensity * phi_scale
                      if 'VZ' in self._maneuver_type else 0.)
        self._vy_escape += 0.08 * (vy_target    - self._vy_escape)   # ~0.25s ramp
        self._vz_escape += 0.06 * (vz_m_target  - self._vz_escape)   # ~0.33s ramp

        # Speed: always ≥ 1.0 m/s floor, EMA 0.10 for smooth transitions
        effective_speed = max(f_speed, 1.00)
        effective_speed = min(effective_speed, 3.5)
        self.speed_ema += 0.10 * (effective_speed - self.speed_ema)

        # Vertical: fuzzy vz + weave + maneuver vz dodge
        vz_combined = f_vz + self._vert_weave_vz(dt)
        self.vz_ema += 0.12 * (self._clamp_vz(vz_combined) - self.vz_ema)

        # Final velocities
        vx = self.speed_ema * math.cos(self.heading)
        vy = self.speed_ema * math.sin(self.heading) + self._vy_escape
        vz = self.vz_ema + self._vz_escape   # fuzzy vz + maneuver vz

        # Z safety clamps (after adding vz_escape)
        if self.pos_z < self.Z_FLOOR + 0.4 and vz < 0.: vz = 0.20
        if self.pos_z > self.Z_CEIL  - 0.4 and vz > 0.: vz = -0.20

        # Yaw rate
        ye = math.atan2(math.sin(self.heading-self.yaw),
                        math.cos(self.heading-self.yaw))
        ys = math.copysign(1., he) if abs(he) > 0.01 else 0.
        yr = ys * f_omega * .3 + 1.5 * ye

        # Velocity clamps
        vx = max(-3.5, min(3.5, vx))
        vy = max(-3.5, min(3.5, vy))
        vz = max(-1.5, min(1.5, vz))
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
                self.send_vel(vx, vy, vz, yr)
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