import numpy as np
import rospy
from panda_robot import PandaArm
from franka_interface import ArmInterface, GripperInterface
import PyKDL
import quaternion

class Sawyer:

    def __init__(self):

        #PandaArm.__init__(self)
        #rospy.init_node("panda_demo") # initialise ros node
        self.robot = PandaArm()
        self.gripper = self.robot.get_gripper()
        self.move_group_inerface = self.robot.get_movegroup_interface()
        self.control_manager = self.robot.get_controller_manager()

        self.is_gripper_open = True
        # set the max wait-time for controller
        #self.robot.set_command_timeout(1.0)

    def open_gripper(self):
        self.gripper.open()
        self.is_gripper_open = True

    def close_gripper(self):
        self.gripper.close()
        self.is_gripper_open = False

    def switch_gripper(self):
        if self.is_gripper_open:
            self.close_gripper()
        else:
            self.open_gripper()

    def move_to_joint_angles(self, joint_angles):
        #print('Move to joint angles')
        self.robot.move_to_joint_position(joint_angles)

    def move_to_pose(self, pose):
        #print('move to pose')
        pos = [*pose.p]
        ori = quaternion.from_float_array(pose.M.GetQuaternion())
        self.robot.move_to_cartesian_pose(pos, ori)
        while self._is_moving():
            pass

    def get_endpoint_pose(self):
        p , q = self.robot.ee_pose()
        frame = PyKDL.Frame(PyKDL.Rotation.Quaternion(*q.components), PyKDL.Vector(*p))
        return frame

    def get_joint_angles(self):
        return self.robot.angles()

    def get_endpoint_velocity_in_endpoint_frame(self):
        ee_linear, ee_angular = self.get_endpoint_velocity_in_base_frame()
        ee_pose = self.get_endpoint_pose()
        ee_pose_Rot = ee_pose.M

        ee_linear_ee = ee_pose_Rot.Inverse() * ee_linear
        ee_angular_ee = ee_pose_Rot.Inverse() * ee_angular

        return ee_linear_ee, ee_angular_ee 

    def get_endpoint_velocity_in_base_frame(self):
        linear, angular = self.robot.ee_velocity()
        return np.array([*linear,*angular])

    def set_endpoint_velocity_in_base_frame(self, endpoint_velocity_vector):
        # convert this cartesian velocity to joint velocity
        jacobian = self._compute_jacobian()
        jacobian_inverse = np.linalg.pinv(jacobian)
        joint_velocities = np.dot(jacobian_inverse, endpoint_velocity_vector)
        joint_velocities = np.squeeze(np.array(joint_velocities))
        self.robot.exec_velocity_cmd(joint_velocities)
        
    def set_endpoint_velocity_in_endpoint_frame(self, endpoint_velocity_vector):
        print('Called : set_endpoint_velocity_in_endpoint_frame()')
        pass

    
    def move_towards_pose(self, target_frame, max_translation_speed, max_rotation_speed):
        current_frame = self.get_endpoint_pose()
        transformation_from_current_to_target = current_frame.Inverse() * target_frame
        # The axis-angle representation is relative to the init frame, i.e. the end-effector's own frame
        current_to_target_axis_angle_endpoint_frame = transformation_from_current_to_target.M.GetRot()
        # Convert this so that it is relative to the base frame
        current_to_target_axis_angle = current_frame.M * current_to_target_axis_angle_endpoint_frame
        current_to_target_translation = target_frame.p - current_frame.p
        # Determine the time to the target, so that neither the translation or rotation speed limits are exceeded
        current_to_target_rotation_angle = np.linalg.norm(np.array([current_to_target_axis_angle[0], current_to_target_axis_angle[1], current_to_target_axis_angle[2]]))
        time_using_rotation_speed = current_to_target_rotation_angle / max_rotation_speed
        current_to_target_translation_distance = np.linalg.norm(np.array([current_to_target_translation[0], current_to_target_translation[1], current_to_target_translation[2]]))
        time_using_translation_speed = current_to_target_translation_distance / max_translation_speed
        time_to_target = max(time_using_rotation_speed, time_using_translation_speed)
        # If the time to the target is less than half a timestep, then don't bother moving, and return to say that the target has been reached
        print('Time to target: {}'.format(time_to_target))
        if time_to_target < 1.0/30.0 :
            print('Stopping active controller')
            # active_controller = self.control_manager.current_controller
            # self.control_manager.stop_controller(active_controller)
            self.robot.exit_control_mode()
            return True
        # Otherwise, move the robot towards the target
        else:
            # Then, determine how much to rotate / translate at each time step
            translation_velocity = current_to_target_translation / time_to_target
            rotation_velocity = current_to_target_axis_angle / time_to_target
            velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[0], rotation_velocity[1], rotation_velocity[2]])
            # Apply this velocity vector
            self.set_endpoint_velocity_in_base_frame(velocity_vector)
        
        return False
    
    def set_joint_velocities(self, joint_velocities):
        self.exec_velocity_cmd(joint_velocities)

    def set_light_colour(self, clr1, clr2):
        pass

    def _get_joint_velocities(self):
        return self.robot.velocities()

    def _compute_jacobian(self):
        joint_angles = self.get_joint_angles()
        return self.robot.jacobian(joint_angles)

    def _is_moving(self):
        joint_velocities = self._get_joint_velocities()
        max_velocity = np.max(np.abs(joint_velocities))
        if max_velocity > 0.01:
            return True
        else:
            return False

def main():
    panda = Sawyer()
    res = panda.move_to_neutral()
    print(res)
    #print(rospy.get_param(param_name=''))


if __name__ == "__main__":
    main()