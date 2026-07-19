#!/usr/bin/env python3
"""sim_snapshot.py — EVENT-TRIGGERED snapshots of the chaser's annotated camera
view, for offline visual inspection of critical moments.

Instead of only fixed-interval frames, it captures the FOV the instant something
notable happens, so oscillation/brake/loss events are actually caught:

  OSC   — command jerk: |Δcmd_vx| or |Δcmd_vz| between ticks exceeds a threshold
          (the chaser is surging / the control is oscillating)
  BRAKE — emergency brake engaged (rising edge)
  LOST  — detection just dropped (box went stale)
  NEAR  — Gazebo separation fell below near_thresh (collision risk)
  t     — slow periodic baseline for context

Each PNG is labelled with the trigger, sim time, phase, det box, cmd, and the
Gazebo dx/dy/dz. Headless (cv2.imwrite). Launched by launch_stack when SNAP=1.

Params: ~snap_dir, ~snap_interval (baseline s, default 6), ~vx_jerk (0.8),
        ~vz_jerk (0.5), ~near_thresh (3.0), ~cooldown (0.6 s per event type)
"""
import rospy, cv2, os, math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import String, Bool
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge

BOX_TIMEOUT = 0.5


class Snapshot:
    def __init__(self):
        rospy.init_node('sim_snapshot')
        self.bridge = CvBridge()
        self.snap_dir = os.path.expanduser(
            rospy.get_param('~snap_dir', '~/fyp/Results/screenshots/run'))
        os.makedirs(self.snap_dir, exist_ok=True)
        self.interval   = float(rospy.get_param('~snap_interval', 6.0))
        self.vx_jerk    = float(rospy.get_param('~vx_jerk', 0.8))
        self.vz_jerk    = float(rospy.get_param('~vz_jerk', 0.5))
        self.near_thresh= float(rospy.get_param('~near_thresh', 3.0))
        self.cooldown   = float(rospy.get_param('~cooldown', 0.6))

        self.frame = None
        self.last_box = None; self.last_box_time = rospy.Time(0)
        self.status = ""; self.phase = "?"
        self.gz = (float('nan'),)*3
        self.cmd = (0., 0., 0.)      # vx, vy, vz
        self.prev_cmd = (0., 0., 0.)
        self.emerg = False; self.prev_emerg = False
        self.det_fresh = True; self.prev_fresh = True
        self.last_base_t = None
        self.last_fire = {}          # event -> sim time of last capture
        # burst: on an OSC event, save the next N CONSECUTIVE camera frames so
        # the box motion vs the drone is visible frame-by-frame (oscillation).
        self.burst_n   = int(rospy.get_param('~burst_n', 8))
        self._burst_left = 0
        self._burst_tag = ""
        self._burst_i = 0

        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self.box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/ibvs_phase', String, self.phase_cb, queue_size=1)
        rospy.Subscriber('/mavros/setpoint_raw/local', PositionTarget, self.cmd_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/emergency_brake', Bool, self.emerg_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gz_cb, queue_size=1)
        rospy.loginfo("[snapshot] event-triggered -> %s", self.snap_dir)

    def box_cb(self, m): self.last_box = m; self.last_box_time = rospy.Time.now()
    def status_cb(self, m): self.status = m.data
    def phase_cb(self, m): self.phase = m.data
    def cmd_cb(self, m): self.cmd = (m.velocity.x, m.velocity.y, m.velocity.z)
    def emerg_cb(self, m): self.emerg = m.data

    def gz_cb(self, m):
        try:
            ic = m.name.index('iris'); it = m.name.index('target_iris')
        except ValueError:
            return
        self.gz = (m.pose[it].position.x - m.pose[ic].position.x,
                   m.pose[it].position.y - m.pose[ic].position.y,
                   m.pose[it].position.z - m.pose[ic].position.z)

    def _fire_ok(self, ev, now):
        return (ev not in self.last_fire or
                now - self.last_fire[ev] >= self.cooldown)

    def img_cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, "[snapshot] cv_bridge: %r" % e)
            return
        now = rospy.Time.now().to_sec()
        self.det_fresh = (self.last_box is not None and
                          (rospy.Time.now() - self.last_box_time).to_sec() < BOX_TIMEOUT)

        # ── event detection ──────────────────────────────────────────────
        ev = None
        if self.emerg and not self.prev_emerg:
            ev = "BRAKE"
        elif (not self.det_fresh) and self.prev_fresh and self.phase in ("APPROACH","HOLD"):
            ev = "LOST"
        elif not math.isnan(self.gz[0]) and \
                math.sqrt(sum(v*v for v in self.gz)) < self.near_thresh:
            ev = "NEAR"
        elif (abs(self.cmd[0]-self.prev_cmd[0]) > self.vx_jerk or
              abs(self.cmd[2]-self.prev_cmd[2]) > self.vz_jerk) and \
                self.phase in ("APPROACH","HOLD"):
            ev = "OSC"

        # burst in progress: save every consecutive frame until exhausted
        if self._burst_left > 0:
            self._save(now, "%s%d" % (self._burst_tag, self._burst_i))
            self._burst_i += 1
            self._burst_left -= 1
        elif ev == "OSC" and self._fire_ok(ev, now):
            # start a burst of consecutive frames to SEE the box oscillate
            self.last_fire[ev] = now
            self._burst_left = self.burst_n
            self._burst_tag = "OSC@%d_" % int(now)
            self._burst_i = 0
        elif ev is not None and self._fire_ok(ev, now):
            self.last_fire[ev] = now
            self._save(now, ev)
        elif self.last_base_t is None or now - self.last_base_t >= self.interval:
            self.last_base_t = now
            self._save(now, "t")

        self.prev_cmd = self.cmd; self.prev_emerg = self.emerg
        self.prev_fresh = self.det_fresh

    def _save(self, simt, ev):
        if self.frame is None:
            return
        f = self.frame.copy()
        if self.det_fresh:
            cx, cy = int(self.last_box.x), int(self.last_box.y)
            w, h = int(self.last_box.z), int(self.last_box.w)
            cv2.rectangle(f, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (0, 255, 0), 2)
            cv2.putText(f, "%dx%d" % (w, h), (cx-w//2, cy-h//2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(f, "NO DET", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        col = {"OSC": (0, 165, 255), "BRAKE": (0, 0, 255),
               "LOST": (0, 0, 255), "NEAR": (0, 0, 255)}.get(ev, (0, 255, 255))
        cv2.putText(f, "t=%.0fs %s %s vx%.1f vz%.1f" % (
            simt, self.phase, ev, self.cmd[0], self.cmd[2]), (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        if not math.isnan(self.gz[0]):
            d3 = math.sqrt(sum(v*v for v in self.gz))
            cv2.putText(f, "GZ %.1fm dx%+.1f dy%+.1f dz%+.1f" % (
                d3, self.gz[0], self.gz[1], self.gz[2]),
                (10, f.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 200, 0), 2)
        tag = ev if ev != "t" else "t"
        cv2.imwrite(os.path.join(self.snap_dir, "t%03d_%s.png" % (int(simt), tag)), f)


if __name__ == '__main__':
    Snapshot()
    rospy.spin()
