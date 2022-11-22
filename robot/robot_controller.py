import sys
import copy
import rospy
import moveit_commander

class Panda:

    def __init__(self):
        self.is_gripper_open = True
        pass

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def switch_gripper(self):
        if self.is_gripper_open:
            self.close_gripper()
        else:
            self.open_gripper()

    def move_to_joint_angles(self, joint_angles):
        pass

    def move_to_pose(self, pose):
        pass

    def get_endpoint_pose(self):
        pass

    def get_joint_angles(self):
        pass

    def get_endpoint_velocity_in_endpoint_frame(self):
        pass

    def get_endpoint_velocity_in_base_frame(self):
        pass

    def set_endpoint_velocity_in_base_frame(self, endpoint_velocity_vector):
        pass

    def set_endpoint_velocity_in_endpoint_frame(self, endpoint_velocity_vector):
        pass

    def move_towards_pose(self, target_pose, max_translation_speed, max_rotation_speed):
        pass

    def set_joint_velocities(self, joint_velocities):
        pass

    def set_light_colour(self, light, colour):
        pass


    #####################
    # PRIVATE FUNCTIONS #
    #####################
    def _get_joint_velocities(self):
        pass

    def _compute_jacobian(self):
        pass

    def _is_moving(self):
        pass

    def _cuff_button_callback(self, button_value):
        pass

    def _lower_button_callback(self, button_value):
        pass

    def _blink_all_lights(self):
        pass



