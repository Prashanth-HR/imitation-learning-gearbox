import numpy as np
import torch
from torch.utils.data import Dataset

from Common import config


class ImageToPoseDatasetCoarse(Dataset):

    def __init__(self, task_name, num_trajectories):
        self.task_name = task_name
        self.dataset_directory = '../Data/' + str(self.task_name) + '/Automatic_Coarse/Image_To_Pose_Dataset'
        # Load the data from the hard drive
        self.images = np.load(self.dataset_directory + '/images_' + str(num_trajectories) + '.npy').astype(np.float32)
        self.endpoint_to_bottleneck_poses = np.load(self.dataset_directory + '/endpoint_to_bottleneck_poses_3dof_sin_cos_' + str(num_trajectories) + '.npy').astype(np.float32)
        self.endpoint_heights = np.load(self.dataset_directory + '/endpoint_heights_' + str(num_trajectories) + '.npy')
        self.training_indices = np.load(self.dataset_directory + '/training_indices_' + str(num_trajectories) + '.npy')
        # Optionally, reduce the amount of data used for training
        if 0:
            num_training_examples_to_use = int(len(self.training_indices) * 0.1)
            self.training_indices = self.training_indices[:num_training_examples_to_use]
        self.validation_indices = np.load(self.dataset_directory + '/validation_indices_' + str(num_trajectories) + '.npy')
        self.num_examples = len(self.training_indices) + len(self.validation_indices)
        print('Image-to-pose dataset loaded with ' + str(self.num_examples) + ' examples (' + str(len(self.training_indices)) + ' training and ' + str(len(self.validation_indices)) + ' validation).')

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        image = np.copy(self.images[index])
        if 1:
            noise = 0.1
            b_rand = np.random.uniform(-noise, noise)
            g_rand = np.random.uniform(-noise, noise)
            r_rand = np.random.uniform(-noise, noise)
            image[0] += np.tile(b_rand, image.shape[1:])
            image[1] += np.tile(g_rand, image.shape[1:])
            image[2] += np.tile(r_rand, image.shape[1:])
            image = image.clip(0, 1)
        endpoint_to_bottleneck_pose = np.copy(self.endpoint_to_bottleneck_poses[index])
        endpoint_height = np.copy(self.endpoint_heights[index])
        example = {'image': image, 'endpoint_to_bottleneck_pose': endpoint_to_bottleneck_pose, 'endpoint_height': endpoint_height, 'example_id': index}
        return example


class ImageToPoseDatasetFine(Dataset):

    def __init__(self, task_name, num_trajectories):
        self.task_name = task_name
        self.dataset_directory = '../Data/' + str(self.task_name) + '/Automatic_Fine/Image_To_Pose_Dataset'
        # Load the data from the hard drive
        self.images = np.load(self.dataset_directory + '/images_' + str(num_trajectories) + '.npy').astype(np.float32)
        self.endpoint_to_bottleneck_poses = np.load(self.dataset_directory + '/endpoint_to_bottleneck_poses_3dof_sin_cos_' + str(num_trajectories) + '.npy').astype(np.float32)
        self.training_indices = np.load(self.dataset_directory + '/training_indices_' + str(num_trajectories) + '.npy')
        self.validation_indices = np.load(self.dataset_directory + '/validation_indices_' + str(num_trajectories) + '.npy')
        self.num_examples = len(self.training_indices) + len(self.validation_indices)
        print('Image-to-pose dataset loaded with ' + str(self.num_examples) + ' examples (' + str(len(self.training_indices)) + ' training and ' + str(len(self.validation_indices)) + ' validation).')

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        image = np.copy(self.images[index])
        if 1:
            noise = 0.1
            b_rand = np.random.uniform(-noise, noise)
            g_rand = np.random.uniform(-noise, noise)
            r_rand = np.random.uniform(-noise, noise)
            image[0] += np.tile(b_rand, image.shape[1:])
            image[1] += np.tile(g_rand, image.shape[1:])
            image[2] += np.tile(r_rand, image.shape[1:])
            image = image.clip(0, 1)
        endpoint_to_bottleneck_pose = np.copy(self.endpoint_to_bottleneck_poses[index])
        endpoint_height = torch.tensor(0, dtype=torch.float)
        example = {'image': image, 'endpoint_to_bottleneck_pose': endpoint_to_bottleneck_pose, 'endpoint_height': endpoint_height, 'example_id': index}
        return example


class ImageToVelocitySingleDataset(Dataset):

    def __init__(self, task_name):
        self.task_name = task_name
        self.dataset_directory = '../Data/' + str(task_name) + '/Single_Demo_Long/Image_To_Velocity_Dataset'
        # Load the data from the hard drive
        self.images = np.load(self.dataset_directory + '/images.npy').astype(np.float32)
        self.velocities = np.load(self.dataset_directory + '/velocities.npy').astype(np.float32)
        self.endpoint_heights = np.load(self.dataset_directory + '/endpoint_heights.npy')
        self.training_indices = np.load(self.dataset_directory + '/training_indices.npy')
        self.validation_indices = np.load(self.dataset_directory + '/validation_indices.npy')
        self.num_examples = len(self.training_indices) + len(self.validation_indices)
        print('Image-to-velocity dataset loaded with ' + str(self.num_examples) + ' examples (' + str(len(self.training_indices)) + ' training and ' + str(len(self.validation_indices)) + ' validation).')

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        image = np.copy(self.images[index])
        if 1:
            noise = 0.1
            b_rand = np.random.uniform(-noise, noise)
            g_rand = np.random.uniform(-noise, noise)
            r_rand = np.random.uniform(-noise, noise)
            image[0] += np.tile(b_rand, image.shape[1:])
            image[1] += np.tile(g_rand, image.shape[1:])
            image[2] += np.tile(r_rand, image.shape[1:])
            image = image.clip(0, 1)
            velocity = np.copy(self.velocities[index])
        endpoint_height = np.copy(self.endpoint_heights[index])
        example = {'image': image, 'velocity': velocity, 'height': endpoint_height, 'example_id': index}
        return example
