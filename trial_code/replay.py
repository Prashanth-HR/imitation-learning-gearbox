import rospy
import numpy as np
import sys
import traceback
# adding to the system path
sys.path.insert(0, '/home/prashanth/Thesis/Calibration/')

from robot.robot import Panda
from common import config
from common import utils

rospy.init_node('replay')
robot = Panda()

ros_rate = rospy.Rate(hz=1)
demo_poses = np.load('trial_code/demo_poses.npy', allow_pickle=True)
print('No of demos recorded '+ str(demo_poses.size))

try:
    print('### Replay started ###')
    
    for pose in demo_poses:
        robot.arm_group.go(pose)
        #ros_rate.sleep()
    print('### Replay Ended ###')
except Exception:
    print(traceback.format_exc()) 
finally:
    rospy.signal_shutdown('')