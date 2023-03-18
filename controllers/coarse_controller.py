import pickle
import traceback

import numpy as np
import rospy
import torch
from threading import Thread

from common import config, utils
from robot import kdl_utils
from training.networks import ImageToPoseNetworkCoarse
from sklearn.neural_network import MLPRegressor


# CoarseController allows us to perform an episode of coarse control, for a particular task
class CoarseController:

    def __init__(self, task_name, sawyer, camera, ros_rate, num_training_trajectories):
        self.task_name = task_name
        self.sawyer = sawyer
        self.camera = camera
        self.ros_rate = ros_rate
        self.num_training_trajectories = num_training_trajectories
        self.max_velocity_scale = config.MAX_VELOCITY_SCALE
        self.max_accleration_scale = config.MAX_ACCLERATION_SCALE
        self.num_uncertainty_samples = 1000
        self.is_ros_running = True
        self.image_to_pose_network = None
        self.validation_error_variance = None
        self.pose_to_uncertainty_regressor : MLPRegressor = None
        rospy.on_shutdown(self._shutdown)

        # Load the network
        self.image_to_pose_network = ImageToPoseNetworkCoarse(self.task_name, self.num_training_trajectories)
        self.image_to_pose_network.load()
        self.image_to_pose_network.eval()
        validation_error_path = '../Networks/' + str(self.task_name) + '/pose_to_uncertainty_validation_error.npy'
        validation_error = np.load(validation_error_path)
        self.validation_error_variance = np.square(validation_error)

        # Load the uncertainty predictor
        filename = '../Data/' + str(self.task_name) + '/Automatic_Coarse/Pose_To_Uncertainty_Predictor/regressor_' + str(self.num_training_trajectories) + '.pickle'
        with open(filename, 'rb') as handle:
            self.pose_to_uncertainty_regressor = pickle.load(handle)

        bottleneck_path = '../Data/' + str(self.task_name) + '/Single_Demo/Raw/bottleneck_pose_vector_vertical.npy'
        bottleneck_pose_vector = np.load(bottleneck_path)
        self.bottleneck_height = bottleneck_pose_vector[2]

    # This runs a single episode of coarse control, using a particular estimation method
    def run_episode(self, estimation_method, bottleneck_pose=None):
        # Run a test episode with the specified method
        bottleneck_height = self.bottleneck_height
        if estimation_method == 'oracle':
            self.run_episode_oracle(bottleneck_height, bottleneck_pose)
        elif estimation_method == 'current_image':
            self.run_episode_current_image(bottleneck_height)
        elif estimation_method == 'first_image':
            self.run_episode_first_image(bottleneck_height)
        elif estimation_method == 'best_image_dropout':
            self.run_episode_best_image_dropout(bottleneck_height)
        elif estimation_method == 'best_image_predicted':
            self.run_episode_best_image_predicted(bottleneck_height)
        elif estimation_method == 'batch':
            self.run_episode_batch(bottleneck_height)
        elif estimation_method == 'batch_with_dropout_uncertainty':
            self.run_episode_batch_with_dropout_uncertainty(bottleneck_height)
        elif estimation_method == 'batch_with_predicted_uncertainty':
            self.run_episode_batch_with_predicted_uncertainty(bottleneck_height)
        elif estimation_method == 'filtering_with_static_uncertainty':
            self.run_episode_filtering_with_static_uncertainty(bottleneck_height)
        elif estimation_method == 'filtering_with_dropout_uncertainty':
            self.run_episode_filtering_with_dropout_uncertainty(bottleneck_height)
        elif estimation_method == 'filtering_with_predicted_uncertainty':
            self.run_episode_filtering_with_predicted_uncertainty(bottleneck_height)

    def run_episode_oracle(self, bottleneck_height, bottleneck_pose):
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        while not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Calculate the latest estimate of the bottleneck pose
            estimated_bottleneck_pose_3dof = kdl_utils.create_pose_3dof_from_pose(bottleneck_pose)
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Compute the target pose in the live frame
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            # Send the robot towards the target pose
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_current_image(self, bottleneck_height):
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Calculate the current prediction of the bottleneck pose
            predicted_bottleneck_pose_3dof = self._predict_bottleneck_pose_3dof()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Calculate the latest estimate of the bottleneck pose
            estimated_bottleneck_pose_3dof = predicted_bottleneck_pose_3dof
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Compute the target pose in the live frame
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            # Send the robot towards the target pose
            self.sawyer.move_towards_pose_cartesian(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_first_image(self, bottleneck_height):
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        is_first_image = True
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # If this is the first step, then predict the pose
            if is_first_image:
                # Say that from now on, it is not the first image
                is_first_image = False
                # Calculate the current prediction of the bottleneck pose
                predicted_bottleneck_pose_3dof = self._predict_bottleneck_pose_3dof()
                # Calculate the latest estimate of the bottleneck pose
                estimated_bottleneck_pose_3dof = predicted_bottleneck_pose_3dof
                estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Otherwise, the estimated pose is the first pose
            else:
                estimated_bottleneck_pose_3dof = estimated_bottleneck_poses_3dof[0]
                estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_best_image_dropout(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        normalised_uncertainties = np.zeros(0, dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the pose prediction, together with the uncertainty prediction
            predicted_bottleneck_pose_3dof, predicted_endpoint_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_dropout_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Compute the uncertainty scores
            normalised_uncertainty = np.mean(predicted_endpoint_pose_3dof_var / self.validation_error_variance)
            normalised_uncertainties = np.concatenate((normalised_uncertainties, [normalised_uncertainty]), axis=0)
            # Get the index of the image which has the lowest uncertainty
            index_best_image = np.argmin(normalised_uncertainties)
            # Calculate the latest estimate of the bottleneck pose
            estimated_bottleneck_pose_3dof = predicted_bottleneck_poses_3dof[index_best_image]
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_best_image_predicted(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        normalised_uncertainties = np.zeros(0, dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the pose prediction, together with the uncertainty prediction
            predicted_bottleneck_pose_3dof, predicted_endpoint_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_predicted_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Compute the uncertainty scores
            normalised_uncertainty = np.mean(predicted_endpoint_pose_3dof_var / self.validation_error_variance)
            normalised_uncertainties = np.concatenate((normalised_uncertainties, [normalised_uncertainty]), axis=0)
            # Get the index of the image which has the lowest uncertainty
            index_best_image = np.argmin(normalised_uncertainties)
            # Calculate the latest estimate of the bottleneck pose
            estimated_bottleneck_pose_3dof = predicted_bottleneck_poses_3dof[index_best_image]
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_batch(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Calculate the current prediction of the bottleneck pose
            predicted_bottleneck_pose_3dof = self._predict_bottleneck_pose_3dof()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Calculate the latest estimate of the bottleneck pose, as the mean of all the estimated poses so far
            # But you can't just take the average of the angle, since there is a discontinuity, so we convert back to sin-cos and compute the average there
            sin_thetas = np.sin(predicted_bottleneck_poses_3dof[:, 2])
            cos_thetas = np.cos(predicted_bottleneck_poses_3dof[:, 2])
            mean_theta = np.arctan2(np.mean(sin_thetas), np.mean(cos_thetas))
            mean_xy = np.mean(predicted_bottleneck_poses_3dof[:, :2], axis=0)
            estimated_bottleneck_pose_3dof = np.array([mean_xy[0], mean_xy[1], mean_theta])
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Compute the target pose in the live frame
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            # Send the robot towards the target pose
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_batch_with_dropout_uncertainty(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        predicted_bottleneck_pose_3dof_vars = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the pose prediction, together with the uncertainty prediction
            predicted_bottleneck_pose_3dof, predicted_bottleneck_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_dropout_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            predicted_bottleneck_pose_3dof_vars = np.concatenate((predicted_bottleneck_pose_3dof_vars, [predicted_bottleneck_pose_3dof_var]), axis=0)
            # Calculate the latest estimate of the bottleneck pose, which is the inverse-variance weighting average of the bottleneck pose predictions
            # First, the position
            estimated_x = np.sum(predicted_bottleneck_poses_3dof[:, 0] / predicted_bottleneck_pose_3dof_vars[:, 0]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 0])
            estimated_y = np.sum(predicted_bottleneck_poses_3dof[:, 1] / predicted_bottleneck_pose_3dof_vars[:, 1]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 1])
            # Calculate the latest estimate of the bottleneck pose, as the mean of all the estimated poses so far
            # But you can't just take the average of the angle, since there is a discontinuity, so we convert back to sin-cos and compute the average there
            sin_thetas = np.sin(predicted_bottleneck_poses_3dof[:, 2])
            cos_thetas = np.cos(predicted_bottleneck_poses_3dof[:, 2])
            estimated_sin_theta = np.sum(sin_thetas / predicted_bottleneck_pose_3dof_vars[:, 2]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 2])
            estimated_cos_theta = np.sum(cos_thetas / predicted_bottleneck_pose_3dof_vars[:, 2]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 2])
            estimated_theta = np.arctan2(estimated_sin_theta, estimated_cos_theta)
            estimated_bottleneck_pose_3dof = np.array([estimated_x, estimated_y, estimated_theta])
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)

    def run_episode_batch_with_predicted_uncertainty(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        predicted_bottleneck_pose_3dof_vars = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the pose prediction, together with the uncertainty prediction
            predicted_bottleneck_pose_3dof, predicted_bottleneck_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_predicted_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            predicted_bottleneck_pose_3dof_vars = np.concatenate((predicted_bottleneck_pose_3dof_vars, [predicted_bottleneck_pose_3dof_var]), axis=0)
            # Calculate the latest estimate of the bottleneck pose, which is the inverse-variance weighting average of the bottleneck pose predictions
            # First, the position
            estimated_x = np.sum(predicted_bottleneck_poses_3dof[:, 0] / predicted_bottleneck_pose_3dof_vars[:, 0]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 0])
            estimated_y = np.sum(predicted_bottleneck_poses_3dof[:, 1] / predicted_bottleneck_pose_3dof_vars[:, 1]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 1])
            # Calculate the latest estimate of the bottleneck pose, as the mean of all the estimated poses so far
            # But you can't just take the average of the angle, since there is a discontinuity, so we convert back to sin-cos and compute the average there
            sin_thetas = np.sin(predicted_bottleneck_poses_3dof[:, 2])
            cos_thetas = np.cos(predicted_bottleneck_poses_3dof[:, 2])
            estimated_sin_theta = np.sum(sin_thetas / predicted_bottleneck_pose_3dof_vars[:, 2]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 2])
            estimated_cos_theta = np.sum(cos_thetas / predicted_bottleneck_pose_3dof_vars[:, 2]) / np.sum(1 / predicted_bottleneck_pose_3dof_vars[:, 2])
            estimated_theta = np.arctan2(estimated_sin_theta, estimated_cos_theta)
            estimated_bottleneck_pose_3dof = np.array([estimated_x, estimated_y, estimated_theta])
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)
            
    def run_episode_filtering_with_static_uncertainty(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        step_num = 0

        
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the prediction
            predicted_bottleneck_pose_3dof = self._predict_bottleneck_pose_3dof()
            predicted_bottleneck_pose_3dof_var = self.validation_error_variance
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Now compute the updated estimate, using Bayesian filtering
            if step_num == 0:
                estimated_bottleneck_pose_3dof = predicted_bottleneck_pose_3dof
                estimated_bottleneck_pose_variance_3dof = predicted_bottleneck_pose_3dof_var
            else:
                estimated_x = ((predicted_bottleneck_pose_3dof[0] / predicted_bottleneck_pose_3dof_var[0]) + (estimated_bottleneck_pose_3dof[0] / estimated_bottleneck_pose_variance_3dof[0])) / ((1 / predicted_bottleneck_pose_3dof_var[0]) + (1 / estimated_bottleneck_pose_variance_3dof[0]))
                estimated_y = ((predicted_bottleneck_pose_3dof[1] / predicted_bottleneck_pose_3dof_var[1]) + (estimated_bottleneck_pose_3dof[1] / estimated_bottleneck_pose_variance_3dof[1])) / ((1 / predicted_bottleneck_pose_3dof_var[1]) + (1 / estimated_bottleneck_pose_variance_3dof[1]))
                predicted_sin_theta = np.sin(predicted_bottleneck_pose_3dof[2])
                predicted_cos_theta = np.cos(predicted_bottleneck_pose_3dof[2])
                estimated_sin_theta = np.sin(estimated_bottleneck_pose_3dof[2])
                estimated_cos_theta = np.cos(estimated_bottleneck_pose_3dof[2])
                estimated_sin_theta = ((predicted_sin_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_sin_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_cos_theta = ((predicted_cos_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_cos_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_theta = np.arctan2(estimated_sin_theta, estimated_cos_theta)
                estimated_bottleneck_pose_3dof = np.array([estimated_x, estimated_y, estimated_theta])
                estimated_bottleneck_pose_variance_3dof = 1 / ((1 / predicted_bottleneck_pose_3dof_var) + (1 / estimated_bottleneck_pose_variance_3dof))
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)
            step_num += 1

    def run_episode_filtering_with_dropout_uncertainty(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        step_num = 0
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the prediction
            predicted_bottleneck_pose_3dof, predicted_bottleneck_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_dropout_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Now compute the updated estimate, using Bayesian filtering
            if step_num == 0:
                estimated_bottleneck_pose_3dof = predicted_bottleneck_pose_3dof
                estimated_bottleneck_pose_variance_3dof = predicted_bottleneck_pose_3dof_var
            else:
                estimated_x = ((predicted_bottleneck_pose_3dof[0] / predicted_bottleneck_pose_3dof_var[0]) + (estimated_bottleneck_pose_3dof[0] / estimated_bottleneck_pose_variance_3dof[0])) / ((1 / predicted_bottleneck_pose_3dof_var[0]) + (1 / estimated_bottleneck_pose_variance_3dof[0]))
                estimated_y = ((predicted_bottleneck_pose_3dof[1] / predicted_bottleneck_pose_3dof_var[1]) + (estimated_bottleneck_pose_3dof[1] / estimated_bottleneck_pose_variance_3dof[1])) / ((1 / predicted_bottleneck_pose_3dof_var[1]) + (1 / estimated_bottleneck_pose_variance_3dof[1]))
                predicted_sin_theta = np.sin(predicted_bottleneck_pose_3dof[2])
                predicted_cos_theta = np.cos(predicted_bottleneck_pose_3dof[2])
                estimated_sin_theta = np.sin(estimated_bottleneck_pose_3dof[2])
                estimated_cos_theta = np.cos(estimated_bottleneck_pose_3dof[2])
                estimated_sin_theta = ((predicted_sin_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_sin_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_cos_theta = ((predicted_cos_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_cos_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_theta = np.arctan2(estimated_sin_theta, estimated_cos_theta)
                estimated_bottleneck_pose_3dof = np.array([estimated_x, estimated_y, estimated_theta])
                estimated_bottleneck_pose_variance_3dof = 1 / ((1 / predicted_bottleneck_pose_3dof_var) + (1 / estimated_bottleneck_pose_variance_3dof))
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)
            step_num += 1

    def run_episode_filtering_with_predicted_uncertainty(self, bottleneck_height):
        predicted_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        estimated_bottleneck_poses_3dof = np.zeros([0, 3], dtype=np.float32)
        step_num = 0
        while not utils.check_for_key('q') and not self._is_bottleneck_reached(bottleneck_height) and self.is_ros_running:
            # Make the prediction
            predicted_bottleneck_pose_3dof, predicted_bottleneck_pose_3dof_var = self._predict_bottleneck_pose_3dof_with_predicted_uncertainty()
            predicted_bottleneck_poses_3dof = np.concatenate((predicted_bottleneck_poses_3dof, [predicted_bottleneck_pose_3dof]), axis=0)
            # Now compute the updated estimate, using Bayesian filtering
            if step_num == 0:
                estimated_bottleneck_pose_3dof = predicted_bottleneck_pose_3dof
                estimated_bottleneck_pose_variance_3dof = predicted_bottleneck_pose_3dof_var
            else:
                estimated_x = ((predicted_bottleneck_pose_3dof[0] / predicted_bottleneck_pose_3dof_var[0]) + (estimated_bottleneck_pose_3dof[0] / estimated_bottleneck_pose_variance_3dof[0])) / ((1 / predicted_bottleneck_pose_3dof_var[0]) + (1 / estimated_bottleneck_pose_variance_3dof[0]))
                estimated_y = ((predicted_bottleneck_pose_3dof[1] / predicted_bottleneck_pose_3dof_var[1]) + (estimated_bottleneck_pose_3dof[1] / estimated_bottleneck_pose_variance_3dof[1])) / ((1 / predicted_bottleneck_pose_3dof_var[1]) + (1 / estimated_bottleneck_pose_variance_3dof[1]))
                predicted_sin_theta = np.sin(predicted_bottleneck_pose_3dof[2])
                predicted_cos_theta = np.cos(predicted_bottleneck_pose_3dof[2])
                estimated_sin_theta = np.sin(estimated_bottleneck_pose_3dof[2])
                estimated_cos_theta = np.cos(estimated_bottleneck_pose_3dof[2])
                estimated_sin_theta = ((predicted_sin_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_sin_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_cos_theta = ((predicted_cos_theta / predicted_bottleneck_pose_3dof_var[2]) + (estimated_cos_theta / estimated_bottleneck_pose_variance_3dof[2])) / ((1 / predicted_bottleneck_pose_3dof_var[2]) + (1 / estimated_bottleneck_pose_variance_3dof[2]))
                estimated_theta = np.arctan2(estimated_sin_theta, estimated_cos_theta)
                estimated_bottleneck_pose_3dof = np.array([estimated_x, estimated_y, estimated_theta])
                estimated_bottleneck_pose_variance_3dof = 1 / ((1 / predicted_bottleneck_pose_3dof_var) + (1 / estimated_bottleneck_pose_variance_3dof))
            estimated_bottleneck_poses_3dof = np.concatenate((estimated_bottleneck_poses_3dof, [estimated_bottleneck_pose_3dof]), axis=0)
            # Send the robot towards the bottleneck
            target_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(estimated_bottleneck_pose_3dof[0], estimated_bottleneck_pose_3dof[1], bottleneck_height, estimated_bottleneck_pose_3dof[2])
            print('traget pose: {}'.format(target_pose))
            self.sawyer.move_towards_pose(target_pose, self.max_velocity_scale, self.max_accleration_scale)
            step_num += 1

    #####################
    # PRIVATE FUNCTIONS #
    #####################
    
    def _predict_bottleneck_pose_3dof(self):
        # Capture an image
        rgb_image = self.camera.capture_cv_image(resize_image=True, show_image=True, show_big_image=True)
        # Get the true endpoint pose
        true_endpoint_pose = self.sawyer.get_endpoint_pose()
        z = true_endpoint_pose.p[2]
        # Make the pose prediction, together with the uncertainty prediction
        predicted_current_to_bottleneck_pose_3dof = self._predict_endpoint_to_bottleneck_pose_from_rgb_image(rgb_image, true_endpoint_pose.p[2])
        predicted_current_to_bottleneck_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(predicted_current_to_bottleneck_pose_3dof[0], predicted_current_to_bottleneck_pose_3dof[1], z, predicted_current_to_bottleneck_pose_3dof[2])
        # Convert this prediction into the live robot frame
        predicted_bottleneck_pose = true_endpoint_pose * predicted_current_to_bottleneck_pose
        # Convert this prediction into a 3-dof pose
        predicted_bottleneck_pose_3dof = kdl_utils.create_pose_3dof_from_pose(predicted_bottleneck_pose)
        # Return this prediction
        return predicted_bottleneck_pose_3dof
    
    def _predict_bottleneck_pose_3dof_with_dropout_uncertainty(self):
        # Capture an image
        rgb_image = self.camera.capture_cv_image(resize_image=True, show_image=True, show_big_image=True)
        # Get the true endpoint pose
        true_endpoint_pose = self.sawyer.get_endpoint_pose()
        z = true_endpoint_pose.p[2]
        # Set the network to use dropout
        self.image_to_pose_network.set_eval_dropout()
        # Convert the RGB image from 0->255 to 0->1
        rgb_image = rgb_image / 255.0
        # Create a batch of normalised tensors
        torch_image = np.moveaxis(rgb_image, 2, 0)
        image_tensor = torch.tensor(torch_image, dtype=torch.float32).repeat(self.num_uncertainty_samples, 1, 1, 1)
        # Create the z tensor, which needs to go from no dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        z_tensor = torch.tensor(z, dtype=torch.float32).repeat(self.num_uncertainty_samples, 1)
        # Make the prediction samples
        prediction_samples = self.image_to_pose_network.forward(image_tensor, z_tensor).detach().cpu().numpy()
        predicted_theta_samples = np.arctan2(prediction_samples[:, 2], prediction_samples[:, 3])
        # Combine the position and orientation samples into an array of pose samples
        predicted_pose_samples = np.concatenate((prediction_samples[:, :2], np.expand_dims(predicted_theta_samples, axis=1)), axis=1)
        # Calculate the mean and uncertainty
        predicted_current_to_bottleneck_pose_3dof = np.mean(predicted_pose_samples, axis=0)
        prediction_var = np.var(predicted_pose_samples, axis=0)
        # Return the predicted pose
        # Make the pose prediction, together with the uncertainty prediction
        predicted_current_to_bottleneck_pose = kdl_utils.create_vertical_pose_from_x_y_z_theta(predicted_current_to_bottleneck_pose_3dof[0], predicted_current_to_bottleneck_pose_3dof[1], z,predicted_current_to_bottleneck_pose_3dof[2])
        # Convert this prediction into the live robot frame
        predicted_bottleneck_pose = true_endpoint_pose * predicted_current_to_bottleneck_pose
        # Convert this prediction into a 3-dof pose
        predicted_bottleneck_pose_3dof = kdl_utils.create_pose_3dof_from_pose(predicted_bottleneck_pose)
        # Return this prediction
        return predicted_bottleneck_pose_3dof, prediction_var

    def _predict_bottleneck_pose_3dof_with_predicted_uncertainty(self):
        # Make the prediction
        predicted_bottleneck_pose = self._predict_bottleneck_pose_3dof()
        # Get the true endpoint pose
        true_endpoint_pose = self.sawyer.get_endpoint_pose()
        z = true_endpoint_pose.p[2]
        # Predict the uncertainty
        regressor_input = np.array([[predicted_bottleneck_pose[0], predicted_bottleneck_pose[1], z, predicted_bottleneck_pose[2]]])
        predicted_uncertainty = self.pose_to_uncertainty_regressor.predict(regressor_input)[0]
        predicted_uncertainty[2] *= 10  # This is because we divided the orientation error by 10 to balance the loss function
        # We need to make sure the predicted uncertainty is greater than a minimum, so that there is no "divided by zero" issues (can result in very fast velocities)
        predicted_uncertainty = np.clip(predicted_uncertainty, a_min=[0.001, 0.001, 0.0175], a_max=None)
        predicted_variance = np.square(predicted_uncertainty)
        return predicted_bottleneck_pose, predicted_variance

    def _predict_endpoint_to_bottleneck_pose_from_rgb_image(self, rgb_image, z):
        # Convert the RGB image from 0->255 to 0->1
        rgb_image = rgb_image / 255.0
        # Create a Torch image by moving the channel axis
        torch_image = np.moveaxis(rgb_image, 2, 0)
        image_tensor = torch.unsqueeze(torch.tensor(torch_image, dtype=torch.float32), 0)
        # Create the z tensor, which needs to go from no dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)
        # Send the image through the network
        prediction = self.image_to_pose_network.forward(image_tensor, z_tensor).detach().cpu().numpy()[0]
        # Convert the unnormalised prediction to real values
        predicted_theta = np.arctan2(prediction[2], prediction[3])
        predicted_pose = np.array([prediction[0], prediction[1], predicted_theta])
        # Return the predicted pose
        return predicted_pose
        
    def _is_bottleneck_reached(self, bottleneck_height):
        true_endpoint_pose = self.sawyer.get_endpoint_pose()
        if true_endpoint_pose.p[2] < bottleneck_height:
            return True
        else:
            return False
        
    def _compute_3dof_error(self, prediction, ground_truth):
        error = np.zeros(3, dtype=np.float32)
        error[0] = np.fabs(prediction[0] - ground_truth[0])
        error[1] = np.fabs(prediction[1] - ground_truth[1])
        error[2] = utils.compute_absolute_angle_difference(prediction[2], ground_truth[2])
        return error

    def _is_target_outside_task_space(self, target):
        if np.fabs(target[0] - config.DEMO_START_MID_POS[0]) > 0.5 * config.TASK_SPACE_WIDTH:
            return True
        if np.fabs(target[1] - config.DEMO_START_MID_POS[1]) > 0.5 * config.TASK_SPACE_WIDTH:
            return True

    # Function that is called when Control-C is pressed
    # It is better to use is_ros_running rather than rospy.is_shutdown(), because sometimes rospy.is_shutdown() isn't triggered (e.g. if you do Control-C outside of the main ROS control loop, such as with doing position control with Intera, it does not flag that ROS has been shutdown until it is too late)
    def _shutdown(self):
        print('\nControl-C detected: Shutting down ...')
        #utils.reset_terminal()
        self.is_ros_running = False
        print('Shut down complete.\n')
