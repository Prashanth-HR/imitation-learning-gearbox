import rospy
from panda_robot import PandaArm
from franka_interface import GripperInterface, ArmInterface
import PyKDL
import kdl_utils


class Panda:

    def __init__(self):
        super(Panda, self).__init__()
        rospy.init_node("panda_demo")        
        panda_robot = PandaArm()
        
        self.robot = panda_robot
        self.gripper : GripperInterface = panda_robot.get_gripper()
        self.is_gripper_open = True


    def open_gripper(self):
        self.gripper.open()
        self.is_gripper_open = True

    def close_gripper(self, wait = False):
        self.gripper.close()
        self.is_gripper_open = False

    def switch_gripper(self):
        if self.gripper.calibrate:
            self.close_gripper()
        else:
            self.open_gripper()

    def move_to_joint_angles(self, joint_angles):
        self.robot.move_to_joint_position(joint_angles=joint_angles)
        

    def move_to_pose(self, pose):
        # Get the position and orientation components
        position = pose.p
        orientation = pose.M.GetQuaternion()
        self.robot.move_to_cartesian_pose(pos=position, ori=orientation)
        

    def get_endpoint_pose(self):
        joint_angles = self.get_joint_angles()
        joint_angles_kdl = kdl_utils.create_kdl_array_from_joint_angles(joint_angles)
        endpoint_pose = PyKDL.Frame()
        self.forward_kinematics_solver.JntToCart(joint_angles_kdl, endpoint_pose)
        return endpoint_pose

    def get_joint_angles(self):
        return self.robot.angles() # returns an array of angles 
        # robot.joint_angles() gets List[dict(Joint_name : angle)]

    def get_endpoint_velocity_in_endpoint_frame(self):
        # Get the current joint velocities
        joint_velocities_array = self._get_joint_velocities()
        # Get the current Jacobian
        jacobian = self._compute_jacobian()
        # Compute the endpoint's velocity in the base frame
        joint_velocities_transposed = joint_velocities_array.reshape(-1, 1)
        endpoint_velocity = jacobian * joint_velocities_transposed
        # Get the translation and rotation components
        translation_velocity = PyKDL.Vector(endpoint_velocity[0], endpoint_velocity[1], endpoint_velocity[2])
        rotation_velocity = PyKDL.Vector(endpoint_velocity[3], endpoint_velocity[4], endpoint_velocity[5])
        # Return these components
        return translation_velocity, rotation_velocity

    def get_endpoint_velocity_in_base_frame(self):
        self.robot.set_joint_velocities()

    def set_endpoint_velocity_in_base_frame(self, endpoint_velocity_vector):
        pass

    def set_endpoint_velocity_in_endpoint_frame(self, endpoint_velocity_vector):
        pass

    def move_towards_pose(self, target_pose, max_translation_speed, max_rotation_speed):
        self.robot.set_arm_speed(max_translation_speed)
        self.robot.set_joint_position_speed(max_rotation_speed) # not needed, since the above covers it.
        position = target_pose.p
        orientation = target_pose.M.GetQuaternion()
        self.robot.move_to_cartesian_pose(pos=position, ori=orientation)

    def set_joint_velocities(self, joint_velocities):
        velocity_list = [(joint_name, joint_velocities[i]) for i, joint_name in enumerate(self.robot.joint_names())]
        velocity_command = dict(velocity_list)
        self.robot.set_joint_velocities(velocities=velocity_command)

    def set_light_colour(self, light, colour):
        pass


    #####################
    # PRIVATE FUNCTIONS #
    #####################
    def _get_joint_velocities(self):
        self.robot.joint_velocities()
        

    def _compute_jacobian(self):
        self.robot.jacobian()

    def _is_moving(self):
        pass

    def _cuff_button_callback(self, button_value):
        pass

    def _lower_button_callback(self, button_value):
        pass

    def _blink_all_lights(self):
        pass

    ###################
    # Extrs Functions #
    ###################

    def move_to_neutral(self, wait = True):
        pass


def main():
    panda = Panda()
    panda.open_gripper()
    #print(rospy.get_param(param_name=''))


if __name__ == "__main__":
    main()