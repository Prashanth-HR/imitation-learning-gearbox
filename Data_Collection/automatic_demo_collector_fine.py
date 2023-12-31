import time

import numpy as np
import rospy
import cv2
import PyKDL

from Common import utils
from Common import config
from Robot.sawyer import Sawyer
from Robot.camera import Camera
from Robot import kdl_utils


class AutomaticDemoCollectorFine:

    def __init__(self, task_name, total_num_trajectories, num_timesteps_per_image, max_translation_speed, max_rotation_speed):
        # Assign variables from the arguments
        self.task_name = task_name
        self.total_num_trajectories = total_num_trajectories
        self.num_timesteps_per_image = num_timesteps_per_image
        self.max_translation_speed = max_translation_speed
        self.max_rotation_speed = max_rotation_speed
        # Define other variables based on the above
        # Create variables which will be defined later
        self.data_directory = None
        self.ros_rate = None
        self.is_ros_running = None
        self.sawyer = None
        self.camera = None
        self.control_time_step = None
        self.bottleneck_pose = None

        # track image number
        self.image_number = 0


    def run(self):
        # Set up the directories
        self.data_directory = '../Data/' + str(self.task_name) + '/Automatic_Fine/Raw'
        utils.create_or_clear_directory(self.data_directory)
        utils.create_or_clear_directory(self.data_directory + '/Images')
        # Initialise ROS
        print('Initialising ROS ...')
        rospy.init_node('demo_collector')
        self.ros_rate = rospy.Rate(config.CONTROL_RATE)
        self.is_ros_running = True
        rospy.on_shutdown(self._shutdown)
        print('\tROS initialised.')
        # Initialise the robot and camera
        print('Initialising Sawyer ...')
        self.sawyer = Sawyer()
        print('\tSawyer initialised.')
        print('Initialising Camera ...')
        self.camera = Camera()
        print('\tCamera initialised.')
        # Load the bottleneck pose
        bottleneck_pose_vector_vertical = np.load('../Data/' + str(self.task_name) + '/Single_Demo/Raw/bottleneck_pose_vector_vertical.npy')
        bottleneck_pose_vertical = kdl_utils.create_pose_from_vector(bottleneck_pose_vector_vertical)
        bottleneck_pose_vertical_3dof = kdl_utils.create_pose_3dof_from_pose(bottleneck_pose_vertical)
        bottleneck_height = bottleneck_pose_vector_vertical[2]
        # First, move the robot to its initial pose
        self.sawyer.move_to_joint_angles(config.ROBOT_INIT_JOINT_ANGLES)
        # First, move the robot to the bottleneck
        self.sawyer.move_to_pose(bottleneck_pose_vertical)
        # Loop over trajectories
        endpoint_pose_vectors_list = []
        trajectory_start_indices = []
        trajectory_lengths = []
        example_trajectory_indices = []
        trajectory_num = 0
        trajectory_start_index = 0
        num_examples_so_far = 0
        self.control_time_step = 0
        delta_x = delta_y = 0.02
        delta_theta = 0.4
        while self.is_ros_running and trajectory_num < self.total_num_trajectories:
            print('Trajectory ' + str(trajectory_num+1) + ' / ' + str(self.total_num_trajectories))
            # Randomly sample a target pose
            x = bottleneck_pose_vertical_3dof[0] + np.random.uniform(-delta_x, delta_x)
            y = bottleneck_pose_vertical_3dof[1] + np.random.uniform(-delta_y, delta_y)
            theta = bottleneck_pose_vertical_3dof[2] + np.random.uniform(-delta_theta, delta_theta)
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(x, y, bottleneck_height, theta)
            # Move the robot to the target pose, whilst capturing images and 3dof poses
            trajectory_images_list, trajectory_endpoint_pose_vectors_list = self._move_to_target_and_record_data(target_pose)
            # Save the images
            #self._save_trajectory_images(trajectory_images_list, num_examples_so_far)
            num_examples_so_far += len(trajectory_images_list)
            # Add this data to the list
            endpoint_pose_vectors_list += trajectory_endpoint_pose_vectors_list
            # Update the auxiliary data
            trajectory_start_indices.append(trajectory_start_index)
            trajectory_num_examples = len(trajectory_images_list)
            trajectory_lengths.append(trajectory_num_examples)
            example_trajectory_indices.append(trajectory_num * trajectory_num_examples)
            # Update the counters
            trajectory_start_index += trajectory_num_examples
            trajectory_num += 1
        # Save the dataset
        self._save_non_image_data(endpoint_pose_vectors_list, trajectory_lengths, trajectory_start_indices, example_trajectory_indices)

    def _move_to_target_and_record_data(self, target_pose):
        images_list = []
        endpoint_pose_vectors_list = []
        while self.is_ros_running:
            # Capture an image and add to the list
            image = self.camera.capture_cv_image(resize_image=False, show_image=False, show_big_image=False)
            #images_list.append(image)
            image_path = self.data_directory + '/Images/image_' + str(self.image_number) + '.png'
            cv2.imwrite(image_path, image)
            self.image_number += 1
            # Has to be removed later just a placeholder to get no of images captured in 1 trajectory
            images_list.append(image_path)

            # Get the pose of the robot's endpoint
            endpoint_pose = self.sawyer.get_endpoint_pose()
            endpoint_pose_vector = kdl_utils.create_vector_from_pose(endpoint_pose)
            endpoint_pose_vectors_list.append(endpoint_pose_vector)
            # Move towards the target
            has_reached_goal = self.sawyer.move_towards_pose(target_pose, self.max_translation_speed * 0.2, self.max_rotation_speed * 0.2)
            # Sleep until the next loop
            self.ros_rate.sleep()
            # Update the control time step
            self.control_time_step += 1
            # If the endpoint has reached the goal, break
            if has_reached_goal:
                break
        return images_list, endpoint_pose_vectors_list

    def _save_trajectory_images(self, trajectory_images, example_start_num):
        print('\nSaving trajectory images ...')
        num_examples = len(trajectory_images)
        for trajectory_example_num in range(num_examples):
            cv_image = trajectory_images[trajectory_example_num]
            image_path = self.data_directory + '/Images/image_' + str(trajectory_example_num + example_start_num) + '.png'
            cv2.imwrite(image_path, cv_image)
        print('\tImages saved.')

    def _save_non_image_data(self, endpoint_pose_vectors_list, trajectory_lengths, trajectory_start_indices, example_trajectory_indices):
        print('\nSaving non-image data ...')
        assert(len(trajectory_lengths) == len(trajectory_start_indices))
        np.save(self.data_directory + '/endpoint_pose_vectors.npy', endpoint_pose_vectors_list)
        np.save(self.data_directory + '/trajectory_lengths.npy', trajectory_lengths)
        np.save(self.data_directory + '/trajectory_start_indices.npy', trajectory_start_indices)
        np.save(self.data_directory + '/example_trajectory_indices.npy', example_trajectory_indices)
        print('\tData saved.')

    # Function that is called when Control-C is pressed
    # It is better to use is_ros_running rather than rospy.is_shutdown(), because sometimes rospy.is_shutdown() isn't triggered (e.g. if you do Control-C outside of the main ROS control loop, such as with doing position control with Intera, it does not flag that ROS has been shutdown until it is too late)
    def _shutdown(self):
        print('\nShutting down ...')
        self.is_ros_running = False
        print('Shut down complete.\n')
