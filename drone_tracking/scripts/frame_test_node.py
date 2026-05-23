#!/usr/bin/env python3
"""
frame_test_node.py - Instrumented version with robust OFFBOARD recovery.

Tests /mavros/setpoint_raw/local + FRAME_BODY_NED with:
  - 10Hz CSV logging of pose, yaw, velocity throughout each trial
  - Pre-trial stationary check (|vel| < 0.05 m/s before start)
  - Yaw IGNORED during velocity command (isolates from yaw-tracking)
  - Echo of first published message for sanity check
  - Auto-recovers OFFBOARD if takeoff_node exited and PX4 dropped mode

CSVs saved to ~/flight_logs/frame_test_<timestamp>_T<n>.csv
"""
import rospy
import math
import os
import csv
import threading
from datetime import datetime
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode, CommandBool
from nav_msgs.msg import Odometry

TEST_VX        = 0.3
TEST_DURATION  = 4.0
HOVER_SETTLE   = 3.0
BRAKE_TIME     = 1.5
YAW_TOL_RAD    = 0.05
STATIONARY_VEL = 0.05
STATIONARY_WAIT_MAX = 8.0
LOG_HZ         = 10.0
RATE_HZ        = 20.0

LOG_DIR = os.path.expanduser('~/flight_logs')
os.makedirs(LOG_DIR, exist_ok=True)
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')


class FrameTest:
    def __init__(self):
        rospy.init_node('frame_test_node')
        self.pose = None
        self.twist = None
        self.state = None
        self.have_odom = False

        self.raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/mavros/state', State, self.state_cb)

        # Background hover stream — keeps OFFBOARD alive throughout
        self._hover_msg_lock = threading.Lock()
        self._hover_msg = None
        self._hover_thread = None
        self._hover_run = False

        # Wait briefly for first odom so we can construct hover msg
        rospy.loginfo("[FRAME_TEST] waiting for first odom...")
        t0 = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and not self.have_odom:
            if (rospy.Time.now() - t0).to_sec() > 10.0:
                rospy.logerr("[FRAME_TEST] no odom received, exiting")
                raise SystemExit(1)
            rate.sleep()
        rospy.loginfo("[FRAME_TEST] odom OK, pose=(%.2f,%.2f,%.2f)",
                      self.pose.position.x, self.pose.position.y, self.pose.position.z)

        self.rate = rospy.Rate(RATE_HZ)
        rospy.loginfo("[FRAME_TEST] node started (instrumented, run_ts=%s)", RUN_TS)

    def odom_cb(self, msg):
        self.pose = msg.pose.pose
        self.twist = msg.twist.twist
        self.have_odom = True

    def state_cb(self, msg):
        self.state = msg

    @staticmethod
    def yaw_from_quat(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def vel_norm(self):
        if self.twist is None:
            return float('inf')
        return math.hypot(math.hypot(self.twist.linear.x, self.twist.linear.y),
                          self.twist.linear.z)

    def _make_pos_yaw_msg(self, x, y, z, yaw_rad):
        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.yaw = yaw_rad
        return msg

    def _make_body_vel_msg(self, vx, vy, vz):
        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE
        )
        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz
        return msg

    def echo_msg(self, msg, label):
        rospy.loginfo("[FRAME_TEST] %s msg: coord_frame=%d type_mask=0x%X(%d) "
                      "vel=(%.2f,%.2f,%.2f) pos=(%.2f,%.2f,%.2f) yaw=%.2f",
                      label, msg.coordinate_frame, msg.type_mask, msg.type_mask,
                      msg.velocity.x, msg.velocity.y, msg.velocity.z,
                      msg.position.x, msg.position.y, msg.position.z, msg.yaw)

    # ---- Background hover stream ----
    def _hover_loop(self):
        rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown() and self._hover_run:
            with self._hover_msg_lock:
                if self._hover_msg is not None:
                    self._hover_msg.header.stamp = rospy.Time.now()
                    self.raw_pub.publish(self._hover_msg)
            rate.sleep()

    def start_hover_stream(self):
        with self._hover_msg_lock:
            self._hover_msg = self._make_pos_yaw_msg(
                self.pose.position.x, self.pose.position.y, self.pose.position.z,
                self.yaw_from_quat(self.pose.orientation))
        self._hover_run = True
        self._hover_thread = threading.Thread(target=self._hover_loop, daemon=True)
        self._hover_thread.start()
        rospy.loginfo("[FRAME_TEST] background hover stream started")

    def stop_hover_stream(self):
        self._hover_run = False
        if self._hover_thread is not None:
            self._hover_thread.join(timeout=1.0)
        rospy.loginfo("[FRAME_TEST] background hover stream stopped")

    def update_hover_msg(self, msg):
        """Atomically swap what the background thread publishes."""
        with self._hover_msg_lock:
            self._hover_msg = msg

    # ---- OFFBOARD recovery ----
    def ensure_offboard_armed(self, timeout=30.0):
        """Stream hover, then if needed, request OFFBOARD + arm via services."""
        rospy.loginfo("[FRAME_TEST] ensuring OFFBOARD + armed...")
        self.start_hover_stream()
        rospy.sleep(1.0)  # give PX4 a stream to latch onto

        try:
            rospy.wait_for_service('/mavros/set_mode', timeout=5.0)
            rospy.wait_for_service('/mavros/cmd/arming', timeout=5.0)
        except rospy.ROSException as e:
            rospy.logerr("[FRAME_TEST] mavros services unavailable: %s", e)
            return False
        set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

        t0 = rospy.Time.now()
        last_attempt = rospy.Time.now() - rospy.Duration(2.0)
        while not rospy.is_shutdown():
            if self.state is not None:
                if self.state.mode == 'OFFBOARD' and self.state.armed:
                    rospy.loginfo("[FRAME_TEST] OFFBOARD + armed OK (mode=%s, armed=%s)",
                                  self.state.mode, self.state.armed)
                    return True
                # Throttle service-call attempts to once per 2s
                if (rospy.Time.now() - last_attempt).to_sec() > 2.0:
                    last_attempt = rospy.Time.now()
                    if self.state.mode != 'OFFBOARD':
                        rospy.loginfo("[FRAME_TEST] requesting OFFBOARD (current=%s)",
                                      self.state.mode)
                        try:
                            set_mode(custom_mode='OFFBOARD')
                        except rospy.ServiceException as e:
                            rospy.logwarn("[FRAME_TEST] set_mode failed: %s", e)
                    elif not self.state.armed:
                        rospy.loginfo("[FRAME_TEST] requesting arm")
                        try:
                            arming(True)
                        except rospy.ServiceException as e:
                            rospy.logwarn("[FRAME_TEST] arming failed: %s", e)
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logerr("[FRAME_TEST] timeout: state=%s",
                             self.state.mode if self.state else None)
                return False
            self.rate.sleep()

    # ---- Trial helpers (unchanged logic, but no need to publish themselves
    #      — they update_hover_msg() and the bg thread handles it) ----
    def hold_pose_yaw(self, x, y, z, yaw_rad, duration):
        self.update_hover_msg(self._make_pos_yaw_msg(x, y, z, yaw_rad))
        rospy.sleep(duration)

    def yaw_align(self, target_yaw_rad, max_wait=15.0):
        rospy.loginfo("[FRAME_TEST] aligning yaw to %.1f deg", math.degrees(target_yaw_rad))
        x0 = self.pose.position.x
        y0 = self.pose.position.y
        z0 = self.pose.position.z
        self.update_hover_msg(self._make_pos_yaw_msg(x0, y0, z0, target_yaw_rad))
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            cur_yaw = self.yaw_from_quat(self.pose.orientation)
            yaw_err = math.atan2(math.sin(target_yaw_rad - cur_yaw),
                                 math.cos(target_yaw_rad - cur_yaw))
            if abs(yaw_err) < YAW_TOL_RAD:
                rospy.loginfo("[FRAME_TEST] yaw aligned (cur=%.1f deg)", math.degrees(cur_yaw))
                return True
            if (rospy.Time.now() - t0).to_sec() > max_wait:
                rospy.logwarn("[FRAME_TEST] yaw align timeout")
                return False
            self.rate.sleep()

    def wait_stationary(self):
        rospy.loginfo("[FRAME_TEST] waiting for drone to settle (|vel| < %.2f)...", STATIONARY_VEL)
        x0 = self.pose.position.x
        y0 = self.pose.position.y
        z0 = self.pose.position.z
        yaw0 = self.yaw_from_quat(self.pose.orientation)
        self.update_hover_msg(self._make_pos_yaw_msg(x0, y0, z0, yaw0))
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            v = self.vel_norm()
            if v < STATIONARY_VEL:
                rospy.loginfo("[FRAME_TEST] settled (|vel|=%.3f)", v)
                return True
            if (rospy.Time.now() - t0).to_sec() > STATIONARY_WAIT_MAX:
                rospy.logwarn("[FRAME_TEST] settle timeout (|vel|=%.3f)", v)
                return False
            self.rate.sleep()

    def run_trial(self, label, target_yaw_deg):
        rospy.loginfo("=" * 60)
        rospy.loginfo("[FRAME_TEST] %s: yaw=%d deg, BODY vx=+%.2f for %.1fs",
                      label, target_yaw_deg, TEST_VX, TEST_DURATION)
        rospy.loginfo("=" * 60)

        target_yaw_rad = math.radians(target_yaw_deg)
        if not self.yaw_align(target_yaw_rad):
            return None

        self.hold_pose_yaw(self.pose.position.x, self.pose.position.y,
                           self.pose.position.z, target_yaw_rad, HOVER_SETTLE)
        if not self.wait_stationary():
            rospy.logwarn("[FRAME_TEST] proceeding despite settle timeout")

        x0, y0, z0 = self.pose.position.x, self.pose.position.y, self.pose.position.z
        yaw_start = self.yaw_from_quat(self.pose.orientation)
        v_start = self.vel_norm()
        rospy.loginfo("[FRAME_TEST] START pos=(%.3f,%.3f,%.3f) yaw=%.1f |vel|=%.3f",
                      x0, y0, z0, math.degrees(yaw_start), v_start)

        csv_path = os.path.join(LOG_DIR, f'frame_test_{RUN_TS}_{label}.csv')
        rospy.loginfo("[FRAME_TEST] logging to %s", csv_path)
        f = open(csv_path, 'w', newline='')
        w = csv.writer(f)
        w.writerow(['t_rel', 'pos_x', 'pos_y', 'pos_z', 'yaw_deg',
                    'vel_x', 'vel_y', 'vel_z', 'cmd_phase'])

        # Switch background thread to BODY-frame velocity
        vel_msg = self._make_body_vel_msg(TEST_VX, 0.0, 0.0)
        self.echo_msg(vel_msg, f'{label} VEL_CMD')
        self.update_hover_msg(vel_msg)

        log_period = 1.0 / LOG_HZ
        t_start = rospy.Time.now()
        t_end = t_start + rospy.Duration(TEST_DURATION)
        next_log = t_start
        log_rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            now = rospy.Time.now()
            if now >= next_log:
                t_rel = (now - t_start).to_sec()
                yaw_now = self.yaw_from_quat(self.pose.orientation)
                w.writerow([f'{t_rel:.3f}',
                            f'{self.pose.position.x:.4f}',
                            f'{self.pose.position.y:.4f}',
                            f'{self.pose.position.z:.4f}',
                            f'{math.degrees(yaw_now):.2f}',
                            f'{self.twist.linear.x:.4f}',
                            f'{self.twist.linear.y:.4f}',
                            f'{self.twist.linear.z:.4f}',
                            'CMD'])
                next_log += rospy.Duration(log_period)
            log_rate.sleep()

        # Brake — switch bg msg to zero body vel
        self.update_hover_msg(self._make_body_vel_msg(0.0, 0.0, 0.0))
        t_brake_end = rospy.Time.now() + rospy.Duration(BRAKE_TIME)
        while not rospy.is_shutdown() and rospy.Time.now() < t_brake_end:
            now = rospy.Time.now()
            if now >= next_log:
                t_rel = (now - t_start).to_sec()
                yaw_now = self.yaw_from_quat(self.pose.orientation)
                w.writerow([f'{t_rel:.3f}',
                            f'{self.pose.position.x:.4f}',
                            f'{self.pose.position.y:.4f}',
                            f'{self.pose.position.z:.4f}',
                            f'{math.degrees(yaw_now):.2f}',
                            f'{self.twist.linear.x:.4f}',
                            f'{self.twist.linear.y:.4f}',
                            f'{self.twist.linear.z:.4f}',
                            'BRAKE'])
                next_log += rospy.Duration(log_period)
            log_rate.sleep()

        f.close()

        x1, y1, z1 = self.pose.position.x, self.pose.position.y, self.pose.position.z
        yaw_end = self.yaw_from_quat(self.pose.orientation)
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        yaw_drift = math.degrees(math.atan2(math.sin(yaw_end - yaw_start),
                                            math.cos(yaw_end - yaw_start)))
        rospy.loginfo("[FRAME_TEST] END   pos=(%.3f,%.3f,%.3f) yaw=%.1f",
                      x1, y1, z1, math.degrees(yaw_end))
        rospy.loginfo("[FRAME_TEST] DELTA dx=%+.3f dy=%+.3f dz=%+.3f yaw_drift=%+.1f deg",
                      dx, dy, dz, yaw_drift)
        rospy.loginfo("[FRAME_TEST] CSV: %s", csv_path)

        # Return bg to a safe pose-hold at current location
        self.update_hover_msg(self._make_pos_yaw_msg(x1, y1, z1, yaw_end))

        return {'label': label, 'yaw_deg': target_yaw_deg,
                'yaw_start_rad': yaw_start, 'yaw_end_rad': yaw_end,
                'yaw_drift_deg': yaw_drift,
                'dx': dx, 'dy': dy, 'dz': dz,
                'csv': csv_path}

    def diagnose(self, results):
        print()
        print("=" * 78)
        print(f"FRAME TEST RESULTS  (instrumented, run_ts={RUN_TS})")
        print("=" * 78)
        expected = TEST_VX * TEST_DURATION
        print(f"{'Trial':<6} {'YawCmd':<7} {'YawStart':<9} {'YawEnd':<8} {'YawDrift':<9} "
              f"{'dx':<9} {'dy':<9} {'dz':<9}")
        for r in results:
            print(f"{r['label']:<6} {r['yaw_deg']:<7} "
                  f"{math.degrees(r['yaw_start_rad']):<9.1f} "
                  f"{math.degrees(r['yaw_end_rad']):<8.1f} "
                  f"{r['yaw_drift_deg']:<+9.1f} "
                  f"{r['dx']:<+9.3f} {r['dy']:<+9.3f} {r['dz']:<+9.3f}")
        print("-" * 78)
        for r in results:
            yaw = r['yaw_start_rad']
            body_x, body_y = expected * math.cos(yaw), expected * math.sin(yaw)
            world_x, world_y = expected, 0.0
            err_body = math.hypot(r['dx'] - body_x, r['dy'] - body_y)
            err_world = math.hypot(r['dx'] - world_x, r['dy'] - world_y)
            print(f"  {r['label']}: body-pred=({body_x:+.2f},{body_y:+.2f}) err={err_body:.2f}  "
                  f"world-pred=({world_x:+.2f},{world_y:+.2f}) err={err_world:.2f}  "
                  f"observed=({r['dx']:+.3f},{r['dy']:+.3f})")
        print("-" * 78)
        print("CSV files for inspection:")
        for r in results:
            print(f"  {r['csv']}")
        print("=" * 78)

    def run(self):
        if not self.ensure_offboard_armed():
            rospy.logerr("[FRAME_TEST] could not ensure OFFBOARD/armed, aborting")
            self.stop_hover_stream()
            return
        try:
            r1 = self.run_trial('T1', 0)
            if r1 is None:
                rospy.logerr("[FRAME_TEST] T1 failed, aborting")
                return
            rospy.loginfo("[FRAME_TEST] inter-trial settle...")
            self.hold_pose_yaw(self.pose.position.x, self.pose.position.y,
                               self.pose.position.z,
                               self.yaw_from_quat(self.pose.orientation), 3.0)
            r2 = self.run_trial('T2', 90)
            if r2 is None:
                rospy.logerr("[FRAME_TEST] T2 failed, aborting")
                return
            rospy.loginfo("[FRAME_TEST] final hover hold...")
            self.hold_pose_yaw(self.pose.position.x, self.pose.position.y,
                               self.pose.position.z,
                               self.yaw_from_quat(self.pose.orientation), 3.0)
            self.diagnose([r1, r2])
        finally:
            self.stop_hover_stream()


if __name__ == '__main__':
    try:
        FrameTest().run()
    except rospy.ROSInterruptException:
        pass