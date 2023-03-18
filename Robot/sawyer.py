import numpy as np
import rospy
from panda_robot import PandaArm
from franka_interface import ArmInterface, GripperInterface
from franka_interface.robot_enable import RobotEnable
import PyKDL
import quaternion
import tf_conversions
import copy

class Sawyer:

    def __init__(self):

        #PandaArm.__init__(self)
        #rospy.init_node("panda_demo") # initialise ros node
        self.robot = PandaArm()
        self.robot_enable = RobotEnable()
        self.gripper = self.robot.get_gripper()
        self.move_group_inerface = self.robot.get_movegroup_interface()
        self.control_manager = self.robot.get_controller_manager()

        self.initialize()

    def initialize(self):
        self.is_gripper_open = True
        # enable the robot if it is in error state
        self.robot_enable.enable() 
        # set the max wait-time for controller
        self.robot.set_command_timeout(3.0)

    def open_gripper(self):
        status = self.gripper.open()
        if status:
            self.is_gripper_open = True
        return status

    def close_gripper(self):
        status = self.gripper.close()
        if status:
            self.is_gripper_open = False
        return status

    def switch_gripper(self):
        if self.is_gripper_open:
            return self.close_gripper()
        else:
            return self.open_gripper()

    def move_to_joint_angles(self, joint_angles):
        self.robot.move_to_joint_position(joint_angles)

    def move_to_pose(self, pose):
        pos = [*pose.p]
        ori = quaternion.from_float_array(pose.M.GetQuaternion())
        self.robot.move_to_cartesian_pose(pos, ori)
        while self.is_moving():
            pass

    def get_endpoint_pose(self):
        p , q = self.robot.ee_pose()
        frame = PyKDL.Frame(PyKDL.Rotation.Quaternion(*q.components), PyKDL.Vector(*p))
        return frame

    def get_joint_angles(self):
        return self.robot.angles()

    def get_joint_efforts(self):
        return self.robot.efforts()

    def get_endpoint_effort_in_base_frame(self):
        endpoint_effort_dist = self.robot.endpoint_effort()
        return endpoint_effort_dist['force'].tolist(), endpoint_effort_dist['torque'].tolist()

    def get_endpoint_velocity_in_endpoint_frame(self):
        ee_linear, ee_angular = self.get_endpoint_velocity_in_base_frame()
        ee_pose = self.get_endpoint_pose()
        ee_pose_Rot = ee_pose.M

        ee_linear_ee = ee_pose_Rot.Inverse() * ee_linear
        ee_angular_ee = ee_pose_Rot.Inverse() * ee_angular

        return ee_linear_ee, ee_angular_ee 

    def get_endpoint_velocity_in_base_frame(self):
        linear, angular = self.robot.ee_velocity()
        return PyKDL.Vector(*linear), PyKDL.Vector(*angular)

    def set_endpoint_velocity_in_base_frame(self, endpoint_velocity_vector):
        # convert this cartesian velocity to joint velocity
        jacobian = self.compute_jacobian()
        jacobian_inverse = np.linalg.pinv(jacobian)
        joint_velocities = np.dot(jacobian_inverse, endpoint_velocity_vector)
        joint_velocities = np.squeeze(np.array(joint_velocities))
        self.set_joint_velocities(joint_velocities)
        
    def set_endpoint_velocity_in_endpoint_frame(self, endpoint_velocity_vector):
        # Get the rotation between the base frame and the endpoint frame
        endpoint_pose = self.get_endpoint_pose()
        endpoint_rotation_matrix = endpoint_pose.M
        # Rotate the endpoint velocity from the endpoint frame to the base frame
        endpoint_translation_velocity = PyKDL.Vector(endpoint_velocity_vector[0], endpoint_velocity_vector[1],  endpoint_velocity_vector[2])
        endpoint_rotation_velocity = PyKDL.Vector(endpoint_velocity_vector[3], endpoint_velocity_vector[4], endpoint_velocity_vector[5])
        endpoint_translation_velocity_in_base_frame = endpoint_rotation_matrix * endpoint_translation_velocity
        endpoint_rotation_velocity_in_base_frame = endpoint_rotation_matrix * endpoint_rotation_velocity
        endpoint_velocity_vector_in_base_frame = np.array(
            [endpoint_translation_velocity_in_base_frame[0], endpoint_translation_velocity_in_base_frame[1],
             endpoint_translation_velocity_in_base_frame[2], endpoint_rotation_velocity_in_base_frame[0],
             endpoint_rotation_velocity_in_base_frame[1], endpoint_rotation_velocity_in_base_frame[2]])
        # set endpoint velocity vector in base frame
        self.set_endpoint_velocity_in_base_frame(endpoint_velocity_vector_in_base_frame)
        

    
    def move_towards_pose(self, target_frame, max_translation_speed, max_rotation_speed):
        current_frame = self.get_endpoint_pose()
        transformation_from_current_to_target = current_frame.Inverse() * target_frame
        # The axis-angle representation is relative to the init frame, i.e. the end-effector's own frame
        current_to_target_axis_angle_endpoint_frame = transformation_from_current_to_target.M.GetRot()
        # Convert this so that it is relative to the base frame
        current_to_target_axis_angle = current_frame.M * current_to_target_axis_angle_endpoint_frame
        current_to_target_translation = target_frame.p - current_frame.p
        #print('translation:{}, rotation:{}'.format(current_to_target_translation, current_to_target_axis_angle))
        # Determine the time to the target, so that neither the translation or rotation speed limits are exceeded
        current_to_target_rotation_angle = np.linalg.norm(np.array([current_to_target_axis_angle[0], current_to_target_axis_angle[1], current_to_target_axis_angle[2]]))
        time_using_rotation_speed = current_to_target_rotation_angle / max_rotation_speed
        current_to_target_translation_distance = np.linalg.norm(np.array([current_to_target_translation[0], current_to_target_translation[1], current_to_target_translation[2]]))
        time_using_translation_speed = current_to_target_translation_distance / max_translation_speed
        
        # Need to comment this when roattion velocity works
        time_to_target = max(time_using_rotation_speed, time_using_translation_speed)
        # If the time to the target is less than half a timestep, then don't bother moving, and return to say that the target has been reached
        #print('Time to target: {}'.format(time_to_target))            
        if time_to_target <  1.0 / 30.0:
            self.stop()
            return True
        # Otherwise, move the robot towards the target
        else:
            # Then, determine how much to rotate / translate at each time step
            translation_velocity = current_to_target_translation / time_to_target
            rotation_velocity = current_to_target_axis_angle / time_to_target
            
            velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[2], rotation_velocity[1], rotation_velocity[0]])
            self.set_endpoint_velocity_in_base_frame(velocity_vector)
        
        return False
    
    def set_joint_velocities(self, joint_velocities):
        self.robot.exec_velocity_cmd(joint_velocities)

    def set_light_colour(self, clr1, clr2):
        pass

    def get_joint_velocities(self, include_gripper: bool = False):
        return self.robot.velocities(include_gripper)

    def compute_jacobian(self):
        #joint_angles = self.get_joint_angles()
        return self.robot.jacobian()

    def is_moving(self):
        joint_velocities = self.get_joint_velocities()
        max_velocity = np.max(np.abs(joint_velocities))
        if max_velocity > 0.01:
            return True
        else:
            return False

    def stop(self):
        '''
        Stop the robot and set the controller to default controller
        '''
        self.set_joint_velocities([0.0] *7)
        while True:
            dq_d = np.array(self.robot.dq_d)
            if all(dq_d == 0.0):
                self.robot.set_command_timeout(0.05)
                rospy.sleep(0.5)
                self.robot.set_command_timeout(2.0)
                break

    def _interpolate_(self, current_frame, traget_frame, num=5):
        '''
        Interpolates b/w the source and targer frane by number(num) of samples.
            Parameters:
                source_frame(PyKDL.Frame)
                target_frame(PyKDL.Frame)
            Returns:
                waypoints([PyKDL.Frame]): returns the list of interpolated frames including target.

        '''
        current_position = current_frame.p
        current_orientation = current_frame.M.GetQuaternion()
        target_position = traget_frame.p
        target_orientation = traget_frame.M.GetQuaternion()

        samples = np.linspace([*current_position], [*target_position], num=num)
        waypoints = []
        for sample in samples:
            wpose = tf_conversions.toMsg(current_frame)
            wpose.position.x = sample[0]
            wpose.position.y = sample[1]
            wpose.position.z = sample[2]
            waypoints.append(copy.deepcopy(wpose))

        # add final position and orientation
        final_pose = tf_conversions.toMsg(current_frame)
        final_pose.position.x = target_position[0]
        final_pose.position.y = target_position[1]
        final_pose.position.z = target_position[2]
        final_pose.orientation.x = target_orientation[0]
        final_pose.orientation.y = target_orientation[1]
        final_pose.orientation.z = target_orientation[2]
        final_pose.orientation.w = target_orientation[3]
        waypoints.append(copy.deepcopy(final_pose))
        
        return waypoints


def main():
    panda = Sawyer()
    panda.robot.move_to_neutral()
    panda.switch_gripper()


if __name__ == "__main__":
    main()