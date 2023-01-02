import numpy as np
import cv2
import rospy
import PyKDL

from robot.robot import Panda
from robot.camera import Camera
from robot import kdl_utils
from common import utils
from common import config


class SingleDemoCollectorShort:

    def __init__(self, task_name):
        # Define class parameters from constructor arguments
        self.task_name = task_name
        # Add other class parameters which will be defined later
        self.ros_rate = None
        self.sawyer = None
        self.camera = None

    def run(self):
        # Initialise ROS
        print('Initialising ROS ...')
        rospy.init_node('single_demo_collector_short')
        self.ros_rate = rospy.Rate(config.CONTROL_RATE)
        rospy.on_shutdown(self._shutdown)
        print('\tROS initialised.')
        # Initialise the robot and camera
        print('Initialising Panda ...')
        self.sawyer = Panda()
        print('\tSawyer initialised.')
        print('Initialising Camera ...')
        self.camera = Camera()
        print('\tCamera initialised.')
        # Set up the terminal for user input
        utils.set_up_terminal_for_key_check()
        # First, move the robot to its initial pose
        print('Move to init pose')
        self.sawyer.move_to_joint_angles(config.ROBOT_INIT_JOINT_ANGLES)
        #self.sawyer.move_to_pose(config.ROBOT_INIT_POSE)
        # Wait until the user has specified the bottleneck pose
        self._request_bottleneck_pose()
        # Collect the demo from the human
        demo_poses = self._request_demo()
        
        # Set the new bottleneck pose to be the one at the beginning of the new pose vectors, after the chopping off
        bottleneck_pose = demo_poses[0]
        print('### Bottleneck pose ###')
        print(bottleneck_pose)
        # Then create the vertical bottleneck pose
        bottleneck_pose_vertical = kdl_utils.create_vertical_pose_from_x_y_z_theta(bottleneck_pose.p[0], bottleneck_pose.p[1], bottleneck_pose.p[2], bottleneck_pose.M.GetRPY()[2])
        # And then create the transformation, in end-effector frame, between the vertical and demonstration bottleneck
        bottleneck_transformation_vertical_to_demo = bottleneck_pose_vertical.Inverse() * bottleneck_pose
        # Save the data
        self._save_data(bottleneck_pose, bottleneck_pose_vertical, bottleneck_transformation_vertical_to_demo, demo_poses)

    # Function to allow the user to specify the bottleneck pose
    def _request_bottleneck_pose(self):
        # Indicate that the robot is ready for the next demo
        print('Move robot to bottleneck, and press \'b\' to confirm bottleneck.')
        print('Make sure that the bottleneck is slightly above the true bottleneck, and also well above any surface of the object. ')
        print('Press \'g\' to open/close the gripper.')
        self.sawyer.set_light_colour('hand', 'yellow')
        # Wait until 'b' is pressed
        while not rospy.is_shutdown():
            self.camera.show_big_live_image()
            if utils.check_for_key('b'):
                break
            self.camera.show_big_live_image()
            if utils.check_for_key('g'):
                self.sawyer.switch_gripper()

    # Collect a single demo
    def _request_demo(self):
        # Indicate that the robot is ready for the next demo
        print('Ready for demo ...')
        print('####### Press r to record #######')
        # Wait until the next demo starts
        while not rospy.is_shutdown():
            pass
            if utils.check_for_key('r'):
                break
        if rospy.is_shutdown():
            return
        # Start recording the demo
        print('\tRecording demo ...')
        
        demo_poses = self._record_demo()
        
        return demo_poses

    # Record the demo, which is called once movement has first been detected
    def _record_demo(self):
        demo_poses = []
        step_num = 1
        print('####### Press x to end record #######')
        while not rospy.is_shutdown():
            # Get the pose of the robot's endpoint
            endpoint_pose = self.sawyer.get_endpoint_pose()
            demo_poses.append(endpoint_pose)
            # Check if the demo has ended
            if utils.check_for_key('x'):
                return demo_poses
            else:
                step_num += 1
            # Sleep until the next loop
            self.ros_rate.sleep()

    # Save the demo data
    def _save_data(self, bottleneck_pose, bottleneck_pose_vertical, bottleneck_transformation_vertical_to_demo, demo_poses):
        print('Saving data ...')
        data_directory = '../Data/' + str(self.task_name) + '/Single_Demo/Raw'
        utils.create_or_clear_directory(data_directory)
        bottleneck_pose_vector = kdl_utils.create_vector_from_pose(bottleneck_pose)
        np.save(data_directory + '/bottleneck_pose_vector.npy', bottleneck_pose_vector)
        bottleneck_pose_vector_vertical = kdl_utils.create_vector_from_pose(bottleneck_pose_vertical)
        np.save(data_directory + '/bottleneck_pose_vector_vertical.npy', bottleneck_pose_vector_vertical)
        bottleneck_transformation_vector = kdl_utils.create_vector_from_pose(bottleneck_transformation_vertical_to_demo)
        transformation_path = data_directory + '/bottleneck_transformation_vector_vertical_to_demo.npy'
        np.save(transformation_path, bottleneck_transformation_vector)
        demo_poses_path = data_directory + '/demo_pose_vectors.npy'
        np.save(demo_poses_path, demo_poses)
        print('\tData saved.')

    # Function that is called when Control-C is pressed
    def _shutdown(self):
        print('\nShutting down ...')
        # Reset the terminal back to the original user input mode
        utils.reset_terminal()
        print('\tShutdown finished.\n')
