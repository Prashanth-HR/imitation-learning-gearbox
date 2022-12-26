import numpy as np
import torch.cuda
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
plt.ion()
import pickle

from training.networks import ImageToPoseNetworkCoarse
from training.datasets import ImageToPoseDatasetCoarse
from common import utils


class PoseToUncertaintyTrainer:

    def __init__(self, task_name, num_trajectories):
        self.task_name = task_name
        self.num_trajectories = num_trajectories

    def run(self):

        # Load the image-to-pose dataset
        image_to_pose_dataset = ImageToPoseDatasetCoarse(self.task_name, self.num_trajectories)

        # Load the pose network
        image_to_pose_network = ImageToPoseNetworkCoarse(self.task_name, self.num_trajectories)
        image_to_pose_network.load()
        image_to_pose_network.eval()
        #image_to_pose_network

        # Loop through all validation examples and create the dataset
        num_examples = len(image_to_pose_dataset.validation_indices)
        poses = np.zeros([num_examples, 4], dtype=np.float32)
        errors = np.zeros([num_examples, 3], dtype=np.float32)
        for i, example_id in enumerate(image_to_pose_dataset.validation_indices):
            # Compute the predictions
            image_tensor = torch.unsqueeze(torch.tensor(image_to_pose_dataset.images[example_id]), dim=0)
            endpoint_height_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(image_to_pose_dataset.endpoint_heights[example_id]), dim=0), dim=0)
            prediction = image_to_pose_network.forward(image_tensor, endpoint_height_tensor).detach().cpu().numpy()[0]
            real_prediction = np.array([prediction[0], prediction[1], np.arctan2(np.sin(prediction[2]), np.cos(prediction[2]))])
            # Compute the error
            true_real_pose = np.array([image_to_pose_dataset.endpoint_to_bottleneck_poses[example_id, 0], image_to_pose_dataset.endpoint_to_bottleneck_poses[example_id, 1], np.arctan2(np.sin(image_to_pose_dataset.endpoint_to_bottleneck_poses[example_id, 2]), np.cos(image_to_pose_dataset.endpoint_to_bottleneck_poses[example_id, 2]))])
            x_error = np.fabs(real_prediction[0] - true_real_pose[0])
            y_error = np.fabs(real_prediction[1] - true_real_pose[1])
            theta_error = utils.compute_absolute_angle_difference(real_prediction[2], true_real_pose[2])
            error = np.array([x_error, y_error, theta_error])
            # Add this to the dataset
            poses[i, 0] = real_prediction[0]
            poses[i, 1] = real_prediction[1]
            poses[i, 2] = image_to_pose_dataset.endpoint_heights[example_id]
            poses[i, 3] = real_prediction[2]
            errors[i] = error

        # Compute the average error
        average_error = np.mean(errors, axis=0)

        # Plot uncertainty vs z
        if 0:
            fig, (ax_x, ax_y, ax_theta) = plt.subplots(nrows=3, ncols=1, num=0, figsize=(10, 20), tight_layout=True)
            ax_x.set_title('X Error vs Z')
            ax_x.scatter(poses[:, 2], errors[:, 0])
            ax_y.set_title('Y Error vs Z')
            ax_y.scatter(poses[:, 2], errors[:, 1])
            ax_theta.set_title('Theta Error vs Z')
            ax_theta.scatter(poses[:, 2], errors[:, 2])
            plt.show()
            plt.waitforbuttonpress()

        # Reweight the labels
        errors[:, 2] *= 0.1

        # Perform the regression
        reg = MLPRegressor(hidden_layer_sizes=(100, 100), verbose=True)
        reg.fit(poses, errors)

        # Make some predictions
        test_poses = np.zeros([100, 4])
        test_error_predictions = np.zeros([100, 3])
        for i in range(100):
            r = np.random.choice(num_examples)
            pose = poses[r].reshape(1, -1)
            predicted_error = reg.predict(pose)
            true_error = errors[r]
            test_poses[i] = poses[r]
            test_error_predictions[i] = predicted_error

        # Plot uncertainty vs z
        if 0:
            fig, (ax_x, ax_y, ax_theta) = plt.subplots(nrows=3, ncols=1, num=1, figsize=(10, 20), tight_layout=True)
            ax_x.set_title('X Error vs Z')
            ax_x.scatter(test_poses[:, 2], test_error_predictions[:, 0])
            ax_y.set_title('Y Error vs Z')
            ax_y.scatter(test_poses[:, 2], test_error_predictions[:, 1])
            ax_theta.set_title('Theta Error vs Z')
            ax_theta.scatter(test_poses[:, 2], test_error_predictions[:, 2])
            plt.show()
            plt.waitforbuttonpress()

        # Save
        utils.create_directory_if_not_exist('../Data/' + str(self.task_name) + '/Automatic_Coarse/Pose_To_Uncertainty_Predictor')
        filename = '../Data/' + str(self.task_name) + '/Automatic_Coarse/Pose_To_Uncertainty_Predictor/regressor_' + str(self.num_trajectories) + '.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(reg, handle)
        np.save('../Data/' + str(self.task_name) + '/Automatic_Coarse/Pose_To_Uncertainty_Predictor/average_error_' + str(self.num_trajectories) + '.npy', average_error)
