import numpy as np
import PyKDL


def create_pose_from_pos_ori_euler(position, orientation):
    kdl_position = PyKDL.Vector(position[0], position[1], position[2])
    kdl_orientation = PyKDL.Rotation.RPY(orientation[0], orientation[1], orientation[2])
    pose = PyKDL.Frame(kdl_orientation, kdl_position)
    return pose


def create_pose_from_vector(pose_vector):
    kdl_position = PyKDL.Vector(pose_vector[0], pose_vector[1], pose_vector[2])
    kdl_orientation = PyKDL.Rotation.Quaternion(pose_vector[3], pose_vector[4], pose_vector[5], pose_vector[6])
    pose = PyKDL.Frame(kdl_orientation, kdl_position)
    return pose


def create_vector_from_pose(pose):
    position = pose.p
    rotation = pose.M.GetQuaternion()
    pose_vector = np.array([position[0], position[1], position[2], rotation[0], rotation[1], rotation[2], rotation[3]], dtype=np.float32)
    return pose_vector


def create_vertical_pose_from_x_y_z_theta(x, y, z, theta):
    orientation = PyKDL.Rotation.RPY(np.pi, 0, theta)
    position = PyKDL.Vector(x, y, z)
    pose = PyKDL.Frame(orientation, position)
    return pose


def create_vertical_pose_from_pose(pose):
    orientation = PyKDL.Rotation.RPY(np.pi, 0, pose.M.GetRPY()[2])
    position = pose.p
    pose_vertical = PyKDL.Frame(orientation, position)
    return pose_vertical


def create_pose_3dof_from_pose(pose):
    pose_3dof = np.array([pose.p[0], pose.p[1], pose.M.GetRPY()[2]])
    return pose_3dof


# Convert an array of angles to an array in the KDL format, so that KDL can use this data in its calculations
def create_kdl_array_from_joint_angles(joint_angles):
    kdl_array = PyKDL.JntArray(7)
    for joint_num in range(7):
        kdl_array[joint_num] = joint_angles[joint_num]
    return kdl_array


def create_numpy_matrix_from_kdl_array(kdl_array):
    matrix = np.mat(np.zeros((kdl_array.rows(), kdl_array.columns())))
    for i in range(kdl_array.rows()):
        for j in range(kdl_array.columns()):
            matrix[i, j] = kdl_array[i, j]
    return matrix
