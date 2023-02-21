import numpy as np
import cv2
import rospy
import PyKDL

from Robot.sawyer import Sawyer
from Robot.camera import Camera
from Robot import kdl_utils
from Common import utils
from Common import config


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
        print('Initialising Sawyer ...')
        self.sawyer = Sawyer()
        print('\tSawyer initialised.')
        print('Initialising Camera ...')
        self.camera = Camera()
        print('\tCamera initialised.')
        # Set up the terminal for user input
        utils.set_up_terminal_for_key_check()
        # First, move the robot to its initial pose
        self.sawyer.move_to_joint_angles(config.ROBOT_INIT_JOINT_ANGLES)
        #self.sawyer.move_to_pose(config.ROBOT_INIT_POSE)
        # Wait until the user has specified the bottleneck pose
        self._request_bottleneck_pose()
        # Collect the demo from the human
        demo_poses, demo_velocity_vectors = self._request_demo()
        # Chop off the first 90 examples (3 seconds), because there is usually vibration at the start due to fighting against the robot
        demo_poses = demo_poses[90:]
        demo_velocity_vectors = demo_velocity_vectors[90:]
        # Chop of the first few examples where there is zero velocity
        num_sequential_moving_examples = 0
        speed_threshold = 0.01
        for example_num in range(len(demo_velocity_vectors)):
            speed = np.linalg.norm(demo_velocity_vectors[example_num])
            if speed > speed_threshold:
                num_sequential_moving_examples += 1
            else:
                num_sequential_moving_examples = 0
            if num_sequential_moving_examples > 10:
                demo_start_index = example_num - 10
                break
        demo_poses = demo_poses[demo_start_index:]
        demo_velocity_vectors = demo_velocity_vectors[demo_start_index:]
        # Set the new bottleneck pose to be the one at the beginning of the new pose vectors, after the chopping off
        bottleneck_pose = demo_poses[0]
        # Then create the vertical bottleneck pose
        bottleneck_pose_vertical = kdl_utils.create_vertical_pose_from_x_y_z_theta(bottleneck_pose.p[0], bottleneck_pose.p[1], bottleneck_pose.p[2], bottleneck_pose.M.GetRPY()[2])
        # And then create the transformation, in end-effector frame, between the vertical and demonstration bottleneck
        bottleneck_transformation_vertical_to_demo = bottleneck_pose_vertical.Inverse() * bottleneck_pose
        # Save the data
        self._save_data(bottleneck_pose, bottleneck_pose_vertical, bottleneck_transformation_vertical_to_demo, demo_velocity_vectors)

    # Function to allow the user to specify the bottleneck pose
    def _request_bottleneck_pose(self):
        # Indicate that the robot is ready for the next demo
        print('Move robot to bottleneck, and press \'b\' to confirm bottleneck.')
        print('Make sure that the bottleneck is slightly above the true bottleneck, and also well above any surface of the object. ')
        print('Press \'g\' to open/close the gripper.')
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

        print('####### Press r to start recording #######')
        # Wait until the next demo starts
        while not rospy.is_shutdown():
            pass
            if utils.check_for_key('r'):
                print('####### Press x to end record #######')
                break
        if rospy.is_shutdown():
            return
        # Start recording the demo
        print('\tRecording demo ...')
        demo_poses, demo_velocities = self._record_demo()
        
        # Return the data
        return demo_poses, demo_velocities

    # Record the demo, which is called once movement has first been detected
    def _record_demo(self):
        demo_poses = []
        demo_velocities = []
        step_num = 1
        while not rospy.is_shutdown():
            # Get the pose of the robot's endpoint
            endpoint_pose = self.sawyer.get_endpoint_pose()
            demo_poses.append(endpoint_pose)
            # Get the velocity of the robot's endpoint
            translation_velocity, rotation_velocity = self.sawyer.get_endpoint_velocity_in_endpoint_frame()
            velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[0], rotation_velocity[1], rotation_velocity[2]])
            demo_velocities.append(velocity_vector)
            # Check if the demo has ended
            if utils.check_for_key('x'):
                return demo_poses, demo_velocities
            else:
                step_num += 1
            # Sleep until the next loop
            self.ros_rate.sleep()

    # Save the demo data
    def _save_data(self, bottleneck_pose, bottleneck_pose_vertical, bottleneck_transformation_vertical_to_demo, demo_velocity_vectors):
        print('Saving data ...')
        data_directory = '../Data/' + str(self.task_name) + '/Single_Demo/Raw'
        utils.create_or_clear_directory(data_directory)
        bottleneck_pose_vector = kdl_utils.create_vector_from_pose(bottleneck_pose)
        np.save(data_directory + '/bottleneck_pose_vector.npy', bottleneck_pose_vector)
        bottleneck_pose_vector_vertical = kdl_utils.create_vector_from_pose(bottleneck_pose_vertical)
        np.save(data_directory + '/bottleneck_pose_vector_vertical.npy', bottleneck_pose_vector_vertical)
        velocities_path = data_directory + '/demo_velocity_vectors.npy'
        np.save(velocities_path, demo_velocity_vectors)
        bottleneck_transformation_vector = kdl_utils.create_vector_from_pose(bottleneck_transformation_vertical_to_demo)
        transformation_path = data_directory + '/bottleneck_transformation_vector_vertical_to_demo.npy'
        np.save(transformation_path, bottleneck_transformation_vector)
        print('\tData saved.')

    # Function that is called when Control-C is pressed
    def _shutdown(self):
        print('\nShutting down ...')
        # Reset the terminal back to the original user input mode
        utils.reset_terminal()
        print('\tShutdown finished.\n')
