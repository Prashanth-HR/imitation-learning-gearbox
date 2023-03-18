from robot.camera import Camera
import rospy
import traceback
import numpy as np
from common import config
from common import utils
from robot.robot import Panda

rospy.init_node('camera_test')
camera = Camera()
robot = Panda()

# Set up the terminal for user input
utils.set_up_terminal_for_key_check()
arm_group = robot.arm_group
# arm_group.set_max_velocity_scaling_factor(0.1)
# arm_group.set_max_acceleration_scaling_factor(0.1)

print('### Move to Neutral ###')
robot.move_to_neutral()

ros_rate = rospy.Rate(config.CONTROL_RATE)

try:
    print('Ready for demo ...')
    print('####### Press r to record #######')
    while not rospy.is_shutdown():
        pass
        if utils.check_for_key('r'):
            break


    print('###### Enter x to stop ######')
    # Record poses
    demo_poses = []
    step_num = 1
    while not rospy.is_shutdown():

        # camera data
        #camera.capture_cv_image(resize_image=False, show_image=True, show_big_image=True)
        # Robot data once camera has a frame
        endpoint_pose = robot.arm_group.get_current_pose().pose
        demo_poses.append(endpoint_pose)

        # Check if the demo has ended
        if utils.check_for_key('x'):
            print('###### Recording done ######')
            np.save('demo_poses.npy', demo_poses)
            break
        else:
            step_num += 1
        # Sleep until the next loop
        
        ros_rate.sleep()


    print('### Move to Neutral ###')
    robot.move_to_neutral()
    # Replay Demo
    print('###### Enter p to play recording ######')
    #robot.arm_group.set_pose_targets(demo_poses)
    for demo in demo_poses:
        robot.arm_group.go(demo)
    status = False

    while not rospy.is_shutdown():
        if utils.check_for_key('p'):
            print('#### Moving to pose ####')
            for demo in demo_poses:
                status =robot.arm_group.go(demo)
        if status == True:
            print('##### Replay Done #####')
            break
except Exception:
    print(traceback.format_exc())
finally:   
    # Shutdown
    camera.shutdown()
    rospy.signal_shutdown('Done')
    utils.reset_terminal()
