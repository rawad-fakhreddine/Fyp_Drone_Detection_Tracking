#!/usr/bin/env python3
"""
ibvs_controller_node.py  —  v6.25  (2-stage velocity-predicted SEARCH)
=======================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

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
        self.Kp_y=1.8; self.Ki_y=0.05; self.Kd_y=0.3
        self.Kp_z=3.0; self.Ki_z=0.04; self.Kd_z=0.5
        self.Kp_wz=0.9; self.Ki_wz=0.; self.Kd_wz=0.15

        # Velocity limits
        self.max_vx=3.5; self.max_vx_retreat=0.50
        self.max_vy=1.20; self.max_vz=1.5; self.max_wz=0.5

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

        self.cmd_pub=rospy.Publisher('/mavros/setpoint_raw/local',PositionTarget,queue_size=1)
        self.active_pub=rospy.Publisher('/drone_tracking/ibvs_active',Bool,queue_size=1)
        self.phase_pub=rospy.Publisher('/drone_tracking/ibvs_phase',String,queue_size=1)
        det_topic = '/drone_tracking/filtered_target' if self.detection_source == 'kalman' else '/drone_tracking/target_center'
        rospy.Subscriber(det_topic, Point, self.detection_cb, queue_size=1)
        rospy.loginfo("[IBVS] detection_source=%s (subscribing to %s)" % (self.detection_source, det_topic))
        rospy.Subscriber('/drone_tracking/kalman_velocity',Point,self.kf_vel_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_setpoints',Quaternion,self.setpoints_cb,queue_size=1)
        rospy.Subscriber('/mavros/state',State,self.state_cb,queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose',PoseStamped,self.pose_cb,queue_size=1)
        rospy.Subscriber('/drone_tracking/takeoff_ready',Bool,self.takeoff_ready_cb,queue_size=1)

        self.dt=1./20.; self.rate=rospy.Rate(20)
        rospy.loginfo("[IBVS] v6.25 | K_far=%.0f Kd_a=%.0f ff_max=%.1f max_vx=%.1f dead=%.3f"
                      %(self.K_far,self.Kd_a,self.ff_max,self.max_vx,self.DEAD_ZONE))
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
                    # Stage 1: 1.0 m/s toward Kalman-predicted target position
                    cvx = 1.0
                    if self.last_cx is not None:
                        ex_s=(self._pred_cx-self.img_cx)/self.img_cx
                        ey_s=(self._pred_cy-self.img_cy)/self.img_cy
                        cwz=float(np.clip(-0.4*ex_s,-self.max_wz,self.max_wz))
                        cvz=float(np.clip(-0.4*ey_s,-0.40,0.40))
                        # Extrapolate predicted position each tick
                        self._pred_cx=float(np.clip(
                            self._pred_cx+self._kf_vx*self.dt, 0., self.img_w))
                        self._pred_cy=float(np.clip(
                            self._pred_cy+self._kf_vy*self.dt, 0., self.img_h))
                    else:
                        # No detection memory yet — climb to safe altitude
                        cvz=float(np.clip(
                            (self.min_altitude_safe-self.altitude)*0.3, -.20, .30))
                else:
                    # Stage 2: 2.0 m/s + slow yaw sweep ±30° around velocity heading
                    cvx = 2.0
                    sweep = 0.25 * math.sin(self._search_elapsed * 0.4)  # ~16s period
                    cwz = float(np.clip(
                        self._search_base_cwz + sweep, -self.max_wz, self.max_wz))
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
                    self._pred_cx=float(self.last_cx) if self.last_cx is not None else self.img_cx
                    self._pred_cy=float(self.last_cy) if self.last_cy is not None else self.img_cy
                    self._search_base_cwz=float(np.clip(
                        -0.2*self._kf_vx/self.img_cx, -self.max_wz, self.max_wz))
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