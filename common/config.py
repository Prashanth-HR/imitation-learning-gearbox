import numpy as np
from robot import kdl_utils


# TASK NAME
TASK_NAME = 'Gear Assembly'

# TASK SPACE
TASK_SPACE_WIDTH = 0.3
TASK_SPACE_ANGLE = 0.25 * np.pi
DEMO_START_MID_POS = [0.3, 0, 0.59]
DEMO_START_MID_ORI = [np.pi, 0, 0]
TASK_SPACE_MIN_HEIGHT = -0.04

# ROBOT CONTROL
CONTROL_RATE = 30
ROBOT_INIT_JOINT_ANGLES = [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483]
ROBOT_INIT_POSE = kdl_utils.create_pose_from_pos_ori_euler(DEMO_START_MID_POS, DEMO_START_MID_ORI)
MAX_VELOCITY_SCALE = 0.1
MAX_ACCLERATION_SCALE = 0.1

# TRAINING
MINIBATCH_SIZE = 32
NO_OF_TRAJECTORIES = 50

# IMAGE
ORIGINAL_IMAGE_SIZE = 512
RESIZED_IMAGE_SIZE = 64
