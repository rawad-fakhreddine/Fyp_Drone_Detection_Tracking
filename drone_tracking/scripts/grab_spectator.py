#!/usr/bin/env python3
"""grab_spectator.py N OUTPREFIX [TOPIC] — save N frames (spaced ~4 s) from a
camera topic to OUTPREFIX_i.png. Used to check spectator-camera placement."""
import rospy, cv2, sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/spec'
TOPIC = sys.argv[3] if len(sys.argv) > 3 else '/spectator_cam/image_raw'
br = CvBridge()
latest = [None]
saved = [0]


def cb(m):
    latest[0] = m


def grab(_evt):
    if latest[0] is None:
        return
    img = br.imgmsg_to_cv2(latest[0], 'bgr8')
    p = '%s_%d.png' % (OUT, saved[0])
    cv2.imwrite(p, img)
    rospy.loginfo('[grab] saved %s' % p)
    saved[0] += 1
    if saved[0] >= N:
        rospy.signal_shutdown('done')


rospy.init_node('grab_spectator', anonymous=True)
rospy.Subscriber(TOPIC, Image, cb, queue_size=1)
rospy.Timer(rospy.Duration(4.0), grab)
rospy.spin()
