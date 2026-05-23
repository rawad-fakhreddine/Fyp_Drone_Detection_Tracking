#!/usr/bin/env python3
"""
arm_offboard.py — stream setpoints, switch to OFFBOARD, arm, with retries.

Why a script not a one-liner:
  * Verifies mode actually changed (the service can succeed but the mode
    not stick if setpoints aren't streaming continuously).
  * Verifies armed status actually changed.
  * Continues streaming setpoints during mode-switch attempts so PX4
    doesn't drop OFFBOARD between calls.
  * Retries each step up to 5 times with informative error messages.

Usage:
    rosrun drone_tracking arm_offboard.py
"""

import rospy
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import TwistStamped


def main():
    rospy.init_node('arm_offboard')

    # Subscribe to /mavros/state so we can verify mode/armed changes
    current = {'mode': '', 'armed': False, 'connected': False}

    def state_cb(msg):
        current['mode']      = msg.mode
        current['armed']     = msg.armed
        current['connected'] = msg.connected
    rospy.Subscriber('/mavros/state', State, state_cb)

    pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',
                          TwistStamped, queue_size=1)
    rate = rospy.Rate(20)

    # Wait for MAVROS to connect to PX4
    print('Waiting for MAVROS-PX4 connection...')
    timeout = rospy.Time.now() + rospy.Duration(10)
    while not current['connected'] and rospy.Time.now() < timeout:
        rate.sleep()
    if not current['connected']:
        print('ERROR: MAVROS not connected to PX4. Check Terminal 1.')
        return 1
    print(f'Connected. Current mode={current["mode"]} armed={current["armed"]}')

    # Stream zero setpoints for 3 seconds (PX4 OFFBOARD precondition)
    print('Streaming setpoints for 3 s to satisfy OFFBOARD precondition...')
    for _ in range(60):
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        pub.publish(cmd)
        rate.sleep()

    # Wait for services
    rospy.wait_for_service('/mavros/set_mode', timeout=5.0)
    rospy.wait_for_service('/mavros/cmd/arming', timeout=5.0)
    set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    arm      = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

    # Switch to OFFBOARD with retry-and-verify
    for attempt in range(1, 6):
        try:
            set_mode(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print(f'  set_mode call failed: {e}')
        # Continue streaming setpoints during the switch
        for _ in range(10):
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            pub.publish(cmd)
            rate.sleep()
        if current['mode'] == 'OFFBOARD':
            print(f'OFFBOARD confirmed (attempt {attempt})')
            break
        print(f'  attempt {attempt}: mode is "{current["mode"]}", retrying...')
    else:
        print('ERROR: failed to enter OFFBOARD after 5 attempts.')
        print('  Most likely cause: setpoint stream interrupted, or')
        print('  PX4 rejected OFFBOARD due to preflight failure.')
        return 1

    # Arm with retry-and-verify
    for attempt in range(1, 6):
        try:
            arm(True)
        except rospy.ServiceException as e:
            print(f'  arm call failed: {e}')
        # Keep streaming during arm attempts
        for _ in range(10):
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            pub.publish(cmd)
            rate.sleep()
        if current['armed']:
            print(f'Armed (attempt {attempt})')
            break
        print(f'  attempt {attempt}: not armed, retrying...')
    else:
        print('ERROR: failed to arm after 5 attempts.')
        print('  Most likely cause: PX4 EKF not converged.')
        print('  Wait 10 more seconds and re-run, or restart simulation.')
        return 1

    # Hold for a moment so takeoff_node has armed status before we exit
    print('Streaming setpoints for 1 more second to bridge handoff...')
    for _ in range(20):
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        pub.publish(cmd)
        rate.sleep()

    print('Done. takeoff_node should now be climbing.')
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        if exit_code:
            raise SystemExit(exit_code)
    except rospy.ROSInterruptException:
        pass