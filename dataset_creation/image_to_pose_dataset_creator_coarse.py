import numpy as np
import cv2

from common import utils
from common import config
from robot import kdl_utils


class ImageToPoseDatasetCreatorCoarse:

    def __init__(self, task_name, demo_or_target='demo'):
        self.task_name = task_name
        self.demo_or_target = demo_or_target
        self.raw_data_dir = '../Data/' + str(task_name) + '/Automatic_Coarse/Raw'
        self.dataset_dir = '../Data/' + str(task_name) + '/Automatic_Coarse/Image_To_Pose_Dataset'
        utils.create_or_clear_directory(self.dataset_dir)

    def run(self):

        # Load the images and poses
        endpoint_pose_vectors = np.load(self.raw_data_dir + '/endpoint_pose_vectors.npy').astype(np.float32)
        num_examples = len(endpoint_pose_vectors)
        images = np.zeros([num_examples, 3, config.RESIZED_IMAGE_SIZE, config.RESIZED_IMAGE_SIZE], dtype=np.float32)
        for example_num in range(num_examples):
            if example_num % 100 == 0:
                print(str(example_num) + ' / ' + str(num_examples))
            original_cv_image = cv2.imread(self.raw_data_dir + '/Images/image_' + str(example_num) + '.png')
            resized_cv_image = cv2.resize(original_cv_image, dsize=(config.RESIZED_IMAGE_SIZE, config.RESIZED_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            torch_image = np.moveaxis(resized_cv_image, 2, 0)
            normalised_torch_image = torch_image / 255.0
            images[example_num] = normalised_torch_image
        assert(len(endpoint_pose_vectors) == len(images))
        num_examples = len(images)

        # Load the trajectory data (required for deciding on the training and validation indices)
        trajectory_start_indices = np.load(self.raw_data_dir + '/trajectory_start_indices.npy')
        trajectory_lengths = np.load(self.raw_data_dir + '/trajectory_lengths.npy')
        num_trajectories = len(trajectory_lengths)

        # For each example, compute the bottleneck pose relative to the end-effector pose
        if self.demo_or_target == 'demo':
            bottleneck_pose_vector_vertical = np.load('../Data/' + self.task_name + '/Single_Demo/Raw/bottleneck_pose_vector_vertical.npy')
        elif self.demo_or_target == 'target':
            bottleneck_pose_vector_vertical = np.load('../Data/' + self.task_name + '/Target/target_vector.npy')
        bottleneck_pose_vertical = kdl_utils.create_pose_from_vector(bottleneck_pose_vector_vertical)
        endpoint_to_bottleneck_poses_3dof_sin_cos = np.zeros([num_examples, 4], dtype=np.float32)
        for example_num in range(num_examples):
            endpoint_pose_vector = endpoint_pose_vectors[example_num]
            endpoint_pose = kdl_utils.create_pose_from_vector(endpoint_pose_vector)
            endpoint_pose_vertical = kdl_utils.create_vertical_pose_from_pose(endpoint_pose)
            endpoint_to_bottleneck_pose = endpoint_pose_vertical.Inverse() * bottleneck_pose_vertical
            endpoint_to_bottleneck_pose_3dof = kdl_utils.create_pose_3dof_from_pose(endpoint_to_bottleneck_pose)
            # Create the unnormalised sine-cosine encoding of the pose angle
            endpoint_to_bottleneck_poses_3dof_sin_cos[example_num, :2] = endpoint_to_bottleneck_pose_3dof[:2]
            endpoint_to_bottleneck_poses_3dof_sin_cos[example_num, 2] = np.sin(endpoint_to_bottleneck_pose_3dof[2])
            endpoint_to_bottleneck_poses_3dof_sin_cos[example_num, 3] = np.cos(endpoint_to_bottleneck_pose_3dof[2])

        # For each example, compute the height of the endpoint
        endpoint_heights = np.zeros([num_examples], dtype=np.float32)
        for example_num in range(num_examples):
            endpoint_height = endpoint_pose_vectors[example_num, 2]
            endpoint_heights[example_num] = endpoint_height

        # Create and save the training and validation indices
        # 200
        num_training_examples = int(0.8 * num_examples)
        training_trajectory_indices = []
        trajectory_index = 0
        num_training_examples_so_far = 0
        while num_training_examples_so_far < num_training_examples:
            training_trajectory_indices.append(trajectory_index)
            num_training_examples_so_far += trajectory_lengths[trajectory_index]
            trajectory_index += 1
        validation_trajectory_indices = range(trajectory_index, num_trajectories)
        training_indices = []
        for training_trajectory_index in training_trajectory_indices:
            global_index_start = trajectory_start_indices[training_trajectory_index]
            global_index_end = global_index_start + trajectory_lengths[training_trajectory_index]
            training_indices.extend(np.arange(global_index_start, global_index_end))
        validation_indices = []
        for validation_trajectory_index in validation_trajectory_indices:
            global_index_start = trajectory_start_indices[validation_trajectory_index]
            global_index_end = global_index_start + trajectory_lengths[validation_trajectory_index]
            validation_indices.extend(np.arange(global_index_start, global_index_end))
        np.save(self.dataset_dir + '/training_indices_200.npy', training_indices)
        np.save(self.dataset_dir + '/validation_indices_200.npy', validation_indices)
        np.save(self.dataset_dir + '/images_200.npy', images)
        np.save(self.dataset_dir + '/endpoint_to_bottleneck_poses_3dof_sin_cos_200.npy', endpoint_to_bottleneck_poses_3dof_sin_cos)
        np.save(self.dataset_dir + '/endpoint_heights_200.npy', endpoint_heights)
        
        # 50
        num_trajectories = config.NO_OF_TRAJECTORIES
        num_examples = 0
        for i in range(num_trajectories):
            num_examples += trajectory_lengths[i]
        num_training_examples = int(0.8 * num_examples)
        training_trajectory_indices = []
        trajectory_index = 0
        num_training_examples_so_far = 0
        while num_training_examples_so_far < num_training_examples:
            training_trajectory_indices.append(trajectory_index)
            num_training_examples_so_far += trajectory_lengths[trajectory_index]
            trajectory_index += 1
        validation_trajectory_indices = range(trajectory_index, num_trajectories)
        training_indices = []
        for training_trajectory_index in training_trajectory_indices:
            global_index_start = trajectory_start_indices[training_trajectory_index]
            global_index_end = global_index_start + trajectory_lengths[training_trajectory_index]
            training_indices.extend(np.arange(global_index_start, global_index_end))
        validation_indices = []
        for validation_trajectory_index in validation_trajectory_indices:
            global_index_start = trajectory_start_indices[validation_trajectory_index]
            global_index_end = global_index_start + trajectory_lengths[validation_trajectory_index]
            validation_indices.extend(np.arange(global_index_start, global_index_end))
        np.save(self.dataset_dir + '/training_indices_50.npy', training_indices)
        np.save(self.dataset_dir + '/validation_indices_50.npy', validation_indices)
        np.save(self.dataset_dir + '/images_50.npy', images)
        np.save(self.dataset_dir + '/endpoint_to_bottleneck_poses_3dof_sin_cos_50.npy', endpoint_to_bottleneck_poses_3dof_sin_cos)
        np.save(self.dataset_dir + '/endpoint_heights_50.npy', endpoint_heights)

        print('Created coarse dataset with ' + str(len(training_indices)) + ' training examples and ' + str(len(validation_indices)) + ' validation examples.')
