import rospy
import numpy as np

import sys
# adding to the system path
sys.path.insert(0, '/home/prashanth/Thesis/Calibration/')

from common import config
from robot.robot import Panda
from common import utils

utils.set_up_terminal_for_key_check()
rospy.init_node('record')
ros_rate = rospy.Rate(hz=1)

robot = Panda()
demo_poses = []
print('Recording started... Enter \'x\' to stop...')
try:
    while not rospy.is_shutdown():

        # camera data
        #camera.capture_cv_image(resize_image=False, show_image=True, show_big_image=True)
        # Robot data once camera has a frame
        endpoint_pose = robot.arm_group.get_current_pose().pose
        demo_poses.append(endpoint_pose)

        # Check if the demo has ended
        if utils.check_for_key('x'):
            print('###### Recording stopped ######')
            np.save('trial_code/demo_poses.npy', demo_poses)
            break
        # Sleep until the next loop
        ros_rate.sleep()

finally:
    rospy.signal_shutdown('Done')
    utils.reset_terminal()
