import rospy

from common import utils
from common import config
from robot.robot import Panda
from robot.camera import Camera
from controllers.coarse_to_fine_controller import CoarseToFineController


class CoarseToFineTester:

    def __init__(self):
        # Set up ROS
        print('Initialising ROS ...')
        rospy.init_node('coarse_policy_tester')
        self.ros_rate = rospy.Rate(30)
        self.is_ros_running = True
        rospy.on_shutdown(self._shutdown)
        print('\tROS initialised')

        # Initialise the robot and camera
        print('Initialising Camera ...')
        self.camera = Camera()
        print('\tCamera initialised.')
        print('Initialising Panda ...')
        self.sawyer = Panda()
        print('\tSawyer initialised.')

    def run_episodes(self, task_name, estimation_method, use_correction):
        utils.set_up_terminal_for_key_check()
        # Create the coarse controller
        coarse_to_fine_controller = CoarseToFineController(task_name, self.sawyer, self.camera, self.ros_rate, config.NO_OF_TRAJECTORIES)
        # Loop over episodes
        while self.is_ros_running:
            print('New episode')
            # Move the robot to the starting joint angles
            self.sawyer.move_to_joint_angles(config.ROBOT_INIT_JOINT_ANGLES)
            # Move to the starting pose
            self.sawyer.move_to_pose(config.ROBOT_INIT_POSE)
            print('Press n for next episode.')
            while not utils.check_for_key('n'):
                pass
            # Run an episode using the coarse controller
            coarse_to_fine_controller.run_episode(estimation_method, use_correction)

    # Function that is called when Control-C is pressed
    # It is better to use is_ros_running rather than rospy.is_shutdown(), because sometimes rospy.is_shutdown() isn't triggered (e.g. if you do Control-C outside of the main ROS control loop, such as with doing position control with Intera, it does not flag that ROS has been shutdown until it is too late)
    def _shutdown(self):
        print('\nControl-C detected: Shutting down ...')
        utils.reset_terminal()
        self.is_ros_running = False
        print('Shut down complete.\n')
