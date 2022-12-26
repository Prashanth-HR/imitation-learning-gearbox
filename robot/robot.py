import sys
import copy
import rospy
import moveit_commander
from moveit_commander import Grasp, JointState, PlanningSceneInterface, RobotCommander, MoveGroupCommander

import moveit_msgs.msg
import geometry_msgs.msg
from rospy_message_converter import message_converter
from moveit_msgs.msg import PlanningScene
from franka_core_msgs.msg import JointCommand
from franka_core_msgs.msg import RobotState, EndPointState

#from kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from robot import kdl_utils
from robot import kdl_parser
import PyKDL
#import kdl_utils

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class Panda:

    def __init__(self):
        super(Panda, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("robot_controller", anonymous=True)        
        
        robot : RobotCommander = moveit_commander.RobotCommander()
        scene : PlanningSceneInterface = moveit_commander.PlanningSceneInterface()
        
        arm_group : MoveGroupCommander = moveit_commander.MoveGroupCommander('panda_arm')
        gripper_group : MoveGroupCommander = moveit_commander.MoveGroupCommander('panda_hand')

        # We can get a list of all the groups in the robot:
        # print("============ Available Planning Groups:", robot.get_group_names())
        # ['panda_arm', 'panda_hand', 'panda_manipulator']
        

        # Sometimes for debugging it is useful to print the entire state of the
        # print("============ Printing robot state")
        # print(robot.get_current_state())
        
        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.arm_group = arm_group
        self.gripper_group = gripper_group

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
        
        #self.group_names = group_names
        #self.panda_robot = panda_robot
        self.is_gripper_open = True
        

    def open_gripper(self, wait = False):
        self.gripper_group.set_named_target('open')
        self.gripper_group.go(wait = wait)
        self.is_gripper_open = True

    def close_gripper(self, wait = False):
        self.gripper_group.set_named_target('close')
        self.gripper_group.go(wait = wait)
        self.is_gripper_open = False

    def switch_gripper(self):
        if self.is_gripper_open:
            self.close_gripper()
        else:
            self.open_gripper()

    def move_to_joint_angles(self, joint_angles, wait = False):
        arm_group = self.arm_group
        arm_group.set_joint_value_target(joint_angles)
        #move_group.go(joint_angles, True)
        arm_group.go(wait = wait)
        arm_group.stop()

        # For testing:
        current_joints = arm_group.get_current_joint_values()
        return all_close(joint_angles, current_joints, 0.01)
        

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
        pass

    def set_light_colour(self, light, colour):
        pass


    #####################
    # PRIVATE FUNCTIONS #
    #####################
    def _get_joint_velocities(self):
        joint_velocities_dict = self.arm_group.joint_velocities()
        pass

    def _compute_jacobian(self):
        return self.arm_group.get_jacobian_matrix()

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
        """
            Send arm group to neutral pose defined using named state in urdf.

            :param wait: if set to True, blocks till execution is complete
            :type wait: bool
        """
        self.arm_group.set_named_target("ready")
        return self.arm_group.go(wait)


def main():
    panda = Panda()
    res = panda.move_to_neutral()
    print(res)
    #print(rospy.get_param(param_name=''))


if __name__ == "__main__":
    main()