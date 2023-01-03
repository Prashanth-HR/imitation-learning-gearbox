import sys

import moveit_commander
import numpy as np
import PyKDL
import rospy
from moveit_commander import (MoveGroupCommander, PlanningSceneInterface,
                              RobotCommander, RobotTrajectory)
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from urdf_parser_py.urdf import URDF

from robot import kdl_parser, kdl_utils

class Panda:

    def __init__(self):
        super(Panda, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("robot_controller", anonymous=True)        
        
        robot : RobotCommander = moveit_commander.RobotCommander()
        scene : PlanningSceneInterface = moveit_commander.PlanningSceneInterface()
        
        
        arm_group : MoveGroupCommander = moveit_commander.MoveGroupCommander('panda_arm')
        gripper_group : MoveGroupCommander = moveit_commander.MoveGroupCommander('panda_hand')
        

        joint_subscriber = rospy.Subscriber("/joint_states", JointState, self._callbackJointState)
        
        
        # Control variables
        self.robot = robot
        self.scene = scene
        self.arm_group = arm_group
        self.gripper_group = gripper_group

        # Monitoted robot states
        self.robot_position = []
        self.robot_velocity = []
        self.robot_effort = []

        # URDF parser
        self.urdf = URDF.from_parameter_server(key='robot_description')
        
        # KDL
        self.kdl_tree = kdl_parser.kdl_tree_from_urdf_model(self.urdf)
        self.base_link = self.urdf.get_root()
        self.tip_link = 'panda_hand'
        self.arm_chain = self.kdl_tree.getChain(self.base_link, self.tip_link)
        self.forward_kinematics_solver = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        self.forward_velocity_solver = PyKDL.ChainFkSolverVel_recursive(self.arm_chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(self.arm_chain)
        self.ik_solver = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        
        # ToDo - get gripper stare and check for above threshold valve to decide open or closed
        self.is_gripper_open = True

    def _callbackJointState(self, data):
        self.robot_position = data.position
        self.robot_velocity = data.velocity
        self.robot_effort = data.effort

    def open_gripper(self, wait = True):
        self.gripper_group.set_named_target('open')
        status = self.gripper_group.go(wait = wait)
        if status == True : 
            self.is_gripper_open = True
        return status

    def close_gripper(self, wait = True):
        self.gripper_group.set_named_target('close')
        status = self.gripper_group.go(wait = wait)
        if status == True : 
            self.is_gripper_open = True
        return status

    def switch_gripper(self):
        if self.is_gripper_open:
            self.close_gripper()
        else:
            self.open_gripper()

    def move_to_joint_angles(self, joint_angles, wait = True):
        arm_group = self.arm_group
        arm_group.set_joint_value_target(joint_angles)
        #move_group.go(joint_angles, True)
        status = arm_group.go(wait = wait)
        arm_group.stop()
        return status
        

    def move_to_pose(self, pose, wait=True):
        # Get the position and orientation components
        position = pose.p
        orientation = pose.M.GetQuaternion()

        arm_group = self.arm_group
        pose = arm_group.get_current_pose().pose

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        # print('#### Target Pose ####')
        # print(pose)
        arm_group.set_pose_target(pose=pose)
        status = arm_group.go(wait)

        arm_group.stop()
        arm_group.clear_pose_targets()
        
        return status
        

    def get_endpoint_pose(self):
        joint_angles = self.get_joint_angles()
        # print('## Joint Angles ###')
        # print(joint_angles)
        joint_angles_kdl = kdl_utils.create_kdl_array_from_joint_angles(joint_angles)
        # print('## Joint Angles KDL ###')
        # print(joint_angles_kdl)
        endpoint_pose = PyKDL.Frame()
        # print('## End pose init ###')
        # print(endpoint_pose)
        self.forward_kinematics_solver.JntToCart(joint_angles_kdl, endpoint_pose)
        # print('## End pose final ###')
        # print(endpoint_pose)
        return endpoint_pose

    def get_joint_angles(self):
        return self.arm_group.get_current_joint_values()

    def get_endpoint_velocity_in_endpoint_frame(self):
        
        pass

    def get_endpoint_velocity_in_base_frame(self):
        pass

    def set_endpoint_velocity_in_base_frame(self, endpoint_velocity_vector):
        # Need ro work with RobotTrajectory
        pass

    def set_endpoint_velocity_in_endpoint_frame(self, endpoint_velocity_vector):
        pass

    def move_towards_pose(self, target_pose, max_velocity_scale=0.1, max_accletation_scale=0.1, wait=True):
        arm_group = self.arm_group
        # arm_group.set_max_velocity_scaling_factor(max_velocity_scale)
        # arm_group.set_max_acceleration_scaling_factor(max_accletation_scale)
        # Get the position and orientation components
        position = target_pose.p
        orientation = target_pose.M.GetQuaternion()

        arm_group = self.arm_group
        pose = arm_group.get_current_pose().pose

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        # print('#### Target Pose ####')
        # print(pose)

        # To plan the cartesian uncomment below'
        (plan, fraction) = arm_group.compute_cartesian_path([pose],1.0, 0.0)
        plan = arm_group.retime_trajectory(arm_group.get_current_state(), plan, max_velocity_scale, max_accletation_scale)
        status = arm_group.execute(plan, wait)
        
        # To plan the joint motion uncommnet below
        # arm_group.set_pose_target(pose=pose)
        # status = arm_group.go(wait)
        # arm_group.stop()
        # arm_group.clear_pose_targets()
        
        return status


    def set_joint_velocities(self, joint_velocities):
        joint_state = self.arm_group.get_current_state()
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['joint_a', 'joint_b']
        joint_state.position = [0.17, 0.34]
        self.arm_group.set_joint_value_target(joint_state)

    def set_light_colour(self, light, colour):
        pass


    #####################
    # PRIVATE FUNCTIONS #
    #####################
    def _get_joint_velocities(self):
        return self.robot_velocity

    def _get_joint_efforts(self):
        return self.robot_effort

    def _compute_jacobian(self):
        return self.arm_group.get_jacobian_matrix()

    def _is_moving(self):
        joint_velocities = self._get_joint_velocities()
        max_velocity = np.max(np.abs(joint_velocities))
        if max_velocity > 0.01:
            return True
        else:
            return False

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
        self.arm_group.set_named_target("ready")
        return self.arm_group.go(wait)


def main():
    panda = Panda()
    res = panda.move_to_neutral()
    print(res)
    #print(rospy.get_param(param_name=''))


if __name__ == "__main__":
    main()