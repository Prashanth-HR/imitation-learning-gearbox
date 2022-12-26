import numpy as np
import rospy

from controllers.coarse_controller import CoarseController
from controllers.correction_controller import CorrectionController
from robot import kdl_utils
from common import utils


class CoarseToFineController:

    def __init__(self, task_name, sawyer, camera, ros_rate, num_training_trajectories):
        # Set variables from the constructor arguments
        self.task_name = task_name
        self.sawyer = sawyer
        self.camera = camera
        self.ros_rate = ros_rate
        # Set other variables
        self.max_velocity_scale = 0.05
        self.max_accletation_scale = 0.2
        # Create the coarse controller
        self.coarse_controller = CoarseController(task_name=task_name, sawyer=sawyer, camera=camera, ros_rate=ros_rate, num_training_trajectories=num_training_trajectories)
        # Create the correction controller
        self.correction_controller = CorrectionController(task_name=task_name, sawyer=sawyer, camera=camera, ros_rate=ros_rate, num_training_trajectories=num_training_trajectories)
        # Load the bottleneck data
        bottleneck_path = '../Data/' + str(self.task_name) + '/Single_Demo/Raw/bottleneck_pose_vector_vertical.npy'
        bottleneck_pose_vector = np.load(bottleneck_path)
        self.bottleneck_pose = kdl_utils.create_pose_from_vector(bottleneck_pose_vector)
        transformation_path = '../Data/' + str(self.task_name) + '/Single_Demo/Raw/bottleneck_transformation_vector_vertical_to_demo.npy'
        bottleneck_transformation_vector = np.load(transformation_path)
        self.bottleneck_transformation_vertical_to_demo = kdl_utils.create_pose_from_vector(bottleneck_transformation_vector)
        self.bottleneck_height = bottleneck_pose_vector[2]
        # Load the demo velocities
        self.demo_velocities = np.load('../Data/' + str(task_name) + '/Single_Demo/Raw/demo_velocity_vectors.npy')
        # Setup ROS
        self.is_ros_running = True
        rospy.on_shutdown(self._shutdown)

    def run_episode(self, estimation_method, use_correction):
        # Move to the vertical bottleneck
        print('COARSE')
        self.coarse_controller.run_episode(estimation_method, self.bottleneck_height)
        for _ in range(5):
            self.ros_rate.sleep()
        # Optionally apply the correction
        print('CORRECTION')
        if use_correction:
            self.correction_controller.run(self.bottleneck_height)
        for _ in range(5):
            self.ros_rate.sleep()
        # Move to the SE(3) bottleneck from the demo
        print('FINE')
        demo_bottleneck = self.sawyer.get_endpoint_pose() * self.bottleneck_transformation_vertical_to_demo
        self.sawyer.move_to_pose(demo_bottleneck)
        # Execute the demo velocities
        num_steps = len(self.demo_velocities)
        step_num = 0
        print('Press < q > to end episode ...')
        while not utils.check_for_key('q') and step_num < num_steps and self.is_ros_running:
            self.sawyer.set_endpoint_velocity_in_endpoint_frame(self.demo_velocities[step_num])
            self.ros_rate.sleep()
            step_num += 1
        for _ in range(5):
            self.ros_rate.sleep()
        step_num = 0
        # Optionally play the demo in reverse
        if 0:
            while not utils.check_for_key('q') and step_num < num_steps and self.is_ros_running:
                self.sawyer.set_endpoint_velocity_in_endpoint_frame(-self.demo_velocities[num_steps - 1 - step_num])
                self.ros_rate.sleep()
                step_num += 1

    # Function that is called when Control-C is pressed
    # It is better to use is_ros_running rather than rospy.is_shutdown(), because sometimes rospy.is_shutdown() isn't triggered (e.g. if you do Control-C outside of the main ROS control loop, such as with doing position control with Intera, it does not flag that ROS has been shutdown until it is too late)
    def _shutdown(self):
        print('\nControl-C detected: Shutting down ...')
        utils.reset_terminal()
        self.is_ros_running = False
        print('Shut down complete.\n')
