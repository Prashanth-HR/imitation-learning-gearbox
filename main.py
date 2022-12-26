from robot.camera import Camera
import rospy
import numpy as np
from common import config
from common import utils
from robot.robot import Panda
from threading import Thread

rospy.init_node('camera_test')
camera = Camera()
robot = Panda()

# Set up the terminal for user input
utils.set_up_terminal_for_key_check()
arm_group = robot.arm_group
arm_group.set_max_velocity_scaling_factor(0.1)
arm_group.set_max_acceleration_scaling_factor(0.1)

status = False
i = 0
print('###### Enter x to stop ######')
thread = Thread(target=robot.move_to_neutral) 

while not rospy.is_shutdown():
    
    #bottleneck_pose_vector_vertical = np.load('../Data/' + str(config.TASK_NAME) + '/Single_Demo/Raw/bottleneck_pose_vector_vertical.npy')
    camera.capture_cv_image(resize_image=False, show_image=True, show_big_image=True)
    if i == 0:
        thread.start()
    i+=1
    ros_rate = rospy.Rate(config.CONTROL_RATE)
    #ros_rate.sleep()
    if utils.check_for_key('x') or not thread.is_alive():
        utils.reset_terminal()
        rospy.signal_shutdown('User Terminate')
camera.shutdown()
utils.reset_terminal()
print(status)