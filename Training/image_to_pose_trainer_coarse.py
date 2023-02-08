import numpy as np
import cv2
import torch.cuda
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
import matplotlib.pyplot as plt

from Common import utils
from Common import config
from Common import graphs
from Training.datasets import ImageToPoseDatasetCoarse
from Training.networks import ImageToPoseNetworkCoarse


class ImageToPoseTrainerCoarse:

    def __init__(self, task_name, num_trajectories):
        self.task_name = task_name
        self.num_trajectories = num_trajectories
        self.image_to_pose_dataset = ImageToPoseDatasetCoarse(task_name, num_trajectories)
        self.image_to_pose_network = ImageToPoseNetworkCoarse(task_name, num_trajectories)
        self.results_directory = '../Results/' + str(task_name) + '/Image_To_Pose_Training_Coarse'
        utils.create_directory_if_not_exist(self.results_directory)
        utils.create_directory_if_not_exist(self.results_directory + '/Graphs')
        self.minibatch_size = config.MINIBATCH_SIZE
        self.init_learning_rate = 0.001
        self.loss_orientation_coefficient = 0.01
        # INITIALISE THE DATASET
        # Create the dataset samplers
        self.training_sampler = SubsetRandomSampler(indices=self.image_to_pose_dataset.training_indices)
        self.validation_sampler = SubsetRandomSampler(indices=self.image_to_pose_dataset.validation_indices)
        # Create the data loaders
        self.training_loader = DataLoader(self.image_to_pose_dataset, batch_size=self.minibatch_size, sampler=self.training_sampler, drop_last=True)
        self.validation_loader = DataLoader(self.image_to_pose_dataset, batch_size=self.minibatch_size, sampler=self.validation_sampler, drop_last=True)
        # Choose some random indices for debugging when printing out the predictions
        self.num_debug_training_examples = 10
        self.num_debug_validation_examples = 10
        self.debug_training_examples = np.random.choice(self.image_to_pose_dataset.training_indices, size=self.num_debug_training_examples, replace=False)
        self.debug_validation_examples = np.random.choice(self.image_to_pose_dataset.validation_indices, size=self.num_debug_validation_examples, replace=False)
        self.debug_validation_examples = self.image_to_pose_dataset.validation_indices[:self.num_debug_validation_examples]
        # INITIALISE THE NETWORK
        # Set the GPU

        # Define the optimiser
        self.loss_function = torch.nn.MSELoss(reduction='none')
        self.optimiser = torch.optim.Adam(self.image_to_pose_network.parameters(), lr=self.init_learning_rate)
        self.lr_patience = 10
        self.patience = 15

    def train(self):
        # Loop over epochs
        training_losses = []
        validation_losses = []
        validation_errors = []
        min_validation_loss = np.inf
        num_bad_epochs_since_lr_change = 0
        num_bad_epochs = 0
        epoch_num = 0
        while True:

            # Increment the epoch num
            epoch_num += 1
            # TRAINING
            # Set to training mode
            self.image_to_pose_network.train()
            # Set some variables to store the training results
            training_epoch_loss_sum = 0
            # Loop over minibatches
            num_minibatches = 0
            for minibatch_num, examples in enumerate(self.training_loader):
                # Do a forward pass on this minibatch
                minibatch_loss = self._train_on_minibatch(examples, epoch_num)
                # Update the loss sums
                training_epoch_loss_sum += minibatch_loss
                # Update the number of minibatches processed
                num_minibatches += 1
            # Store the training losses
            training_loss = training_epoch_loss_sum / num_minibatches
            training_losses.append(training_loss)

            # VALIDATION
            # Set to validation mode
            self.image_to_pose_network.eval()
            # Set some variables to store the training results
            validation_epoch_loss_sum = 0
            validation_epoch_x_error_sum = 0
            validation_epoch_y_error_sum = 0
            validation_epoch_theta_error_sum = 0
            if 0:
                xy_error_sum = np.zeros([10, 10], dtype=np.float32)
                xy_error_count = np.zeros([10, 10], dtype=np.uint8)
                bins = np.linspace(-0.04, 0.05, 10)  # The bin numbers represent the value on the right of the bin (i.e. the maximum value in that bin)
            # Loop over minibatches
            num_minibatches = 0
            for minibatch_num, examples in enumerate(self.validation_loader):
                # Do a forward pass on this minibatch
                minibatch_loss, minibatch_x_error, minibatch_y_error, minibatch_theta_error, minibatch_poses = self._validate_on_minibatch(examples, epoch_num)
                # Update the loss sums
                validation_epoch_loss_sum += minibatch_loss
                validation_epoch_x_error_sum += minibatch_x_error
                validation_epoch_y_error_sum += minibatch_y_error
                validation_epoch_theta_error_sum += minibatch_theta_error
                # Update the errors for each position
                if 0:
                    minibatch_x_bins = np.digitize(minibatch_poses[:, 0], bins)
                    minibatch_y_bins = np.digitize(minibatch_poses[:, 1], bins)
                    xy_error_sum[minibatch_x_bins, minibatch_y_bins] += 0.5 * (minibatch_x_error + minibatch_y_error)
                    xy_error_count[minibatch_x_bins, minibatch_y_bins] += 1
                # Update the number of minibatches processed
                num_minibatches += 1
            # Store the validation losses
            validation_loss = validation_epoch_loss_sum / num_minibatches
            validation_losses.append(validation_loss)
            validation_epoch_x_error = validation_epoch_x_error_sum / num_minibatches
            validation_epoch_y_error = validation_epoch_y_error_sum / num_minibatches
            validation_epoch_theta_error = validation_epoch_theta_error_sum / num_minibatches
            validation_error = [validation_epoch_x_error, validation_epoch_y_error, validation_epoch_theta_error]
            validation_errors.append(validation_error)

            # Decide whether to update the number of epochs that have elapsed since the loss decreased
            # A 'bad epoch' is one where the loss does not decrease by at least 1% of the current minimum loss
            if validation_loss > 0.99 * min_validation_loss:
                num_bad_epochs_since_lr_change += 1
                num_bad_epochs += 1
            else:
                num_bad_epochs_since_lr_change = 0
                num_bad_epochs = 0
            print('Epoch ' + str(epoch_num) + ': num bad epochs = ' + str(num_bad_epochs))
            # Decide whether to reduce the learning rate
            if num_bad_epochs_since_lr_change > self.lr_patience:
                for p in self.optimiser.param_groups:
                    old_lr = p['lr']
                    new_lr = 0.5 * old_lr
                    p['lr'] = new_lr
                print('Dropping learning rate to ' + str(new_lr))
                num_bad_epochs_since_lr_change = 0
            # Decide whether this loss is the minimum so far. If so, set the minimum loss and save the network.
            # This needs to come after updating the number of bad epochs, otherwise the comparison of the loss and min loss could be the same number
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                min_validation_error = validation_error
                self.image_to_pose_network.save()
            # Decide whether to do early stopping
            if num_bad_epochs > self.patience:
                break

            # Compute the running average losses and show the graphs
            window = 10
            mask = np.ones(window) / window
            if len(training_losses) >= window:
                running_average_training_losses = np.convolve(training_losses, mask, mode='valid')
                running_average_validation_losses = np.convolve(validation_losses, mask, mode='valid')
                running_average_epochs = np.arange(start=1 + 0.5 * (window - 1), stop=len(training_losses) - 0.5 * (window - 1) + 1, step=1)
                # Show and save the graphs
                epochs = range(1, epoch_num + 1)
                graphs.show_loss_curves(training_losses, validation_losses, epochs, running_average_training_losses, running_average_validation_losses, running_average_epochs, self.results_directory + '/Graphs/loss_curves.png')
                graphs.show_error_curves(validation_errors, epochs, self.results_directory + '/Graphs/validation_error_curves.png')
                # Print out the losses for this epoch
                if 0:
                    print('Epoch ' + str(epoch_num) + ':')
                    print('\tRunning Average: Training loss: ' + str(running_average_training_losses[-1]) + ', Validation loss: ' + str(running_average_validation_losses[-1]))
                    print('\tTraining loss: ' + str(training_loss) + ', Validation loss: ' + str(validation_loss))
                    print('\tValidation position error: ' + str(validation_position_error) + ', Validation orientation error: ' + str(validation_orientation_error))

        # Save the error, so it can be used as a prior on uncertainty
        np.save('../Networks/' + str(self.task_name) + '/pose_to_uncertainty_validation_error.npy', min_validation_error)

        # Return the minimum loss and error
        return min_validation_loss, min_validation_error

    def _train_on_minibatch(self, examples, epoch_num):
        # Do a forward pass
        image_tensor = examples['image']
        # Create the z tensor, which needs to go from one dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        endpoint_height_tensor = torch.unsqueeze(examples['endpoint_height'], 1)
        predictions = self.image_to_pose_network.forward(image_tensor, endpoint_height_tensor)
        # Compute the loss
        ground_truths = examples['endpoint_to_bottleneck_pose']
        loss = self._compute_loss(predictions, ground_truths)
        # Set the gradients to zero
        self.optimiser.zero_grad()
        # Do a backward pass, which computes and stores the gradients
        loss.backward()
        # Do a weight update
        self.optimiser.step()
        # Debugging: print the predictions for some examples
        if 0 and epoch_num % 1 == 0:
            for i, example_id in enumerate(examples['example_id']):
                for j in range(self.num_debug_training_examples):
                    if example_id == self.debug_training_examples[j]:
                        prediction = predictions[i].cpu().data.numpy()
                        ground_truth = ground_truths[i].cpu().data.numpy()
                        x_error = np.fabs(prediction[0] - ground_truth[0])
                        y_error = np.fabs(prediction[1] - ground_truth[1])
                        theta_prediction = np.arctan2(prediction[2], prediction[3])
                        theta_ground_truth = np.arctan2(ground_truth[2], ground_truth[3])
                        theta_error = utils.compute_absolute_angle_difference(theta_prediction, theta_ground_truth)
                        error = np.array([x_error, y_error, theta_error])
                        if 1:
                            print('TRAINING DEBUG NUM ' + str(j) + ' (example id ' + str(example_id.item()) + ') :')
                            print('Prediction:\t ' + str(prediction))
                            print('Ground Truth:\t ' + str(ground_truth))
                            print('Error:\t ' + str(error))
        # Debugging: draw the keypoints (only for the spatial soft argmax method)
        # if 0 and epoch_num % 1 == 0:
        #     minibatch_num_examples = len(examples['image'])
        #     for i in range(minibatch_num_examples):
        #         example_id = examples['example_id'][i].numpy()
        #         for j in range(self.num_debug_training_examples):
        #             if example_id == self.debug_training_examples[j]:
        #                 example_keypoints = keypoints[i]
        #                 image_path = '../Data/' + str(self.task_name) + '/Automatic/Raw/Images/image_' + str(example_id) + '.png'
        #                 image = cv2.imread(image_path)
        #                 big_image = cv2.resize(image, (320, 320), cv2.INTER_NEAREST)
        #                 for k in range(16):
        #                     x = example_keypoints[k * 2]
        #                     x = 10 * (5 + x * 22)
        #                     y = example_keypoints[k * 2 + 1]
        #                     y = 10 * (5 + y * 22)
        #                     cv2.circle(big_image, (x, y), 10, color=utils.get_colour(k), thickness=1)
        #                     cv2.imwrite('../Results/' + str(self.task_name) + '/Image_To_Pose_Training/Keypoint_Images/image_' + str(example_id) + '.png', big_image)

        # Return the loss
        minibatch_loss = loss.item()
        return minibatch_loss

    def _validate_on_minibatch(self, examples, epoch_num):
        # Do a forward pass
        image_tensor = examples['image']
        # Create the z tensor, which needs to go from one dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        endpoint_height_tensor = torch.unsqueeze(examples['endpoint_height'], 1)
        predictions = self.image_to_pose_network.forward(image_tensor, endpoint_height_tensor)
        # Compute the loss
        ground_truths = examples['endpoint_to_bottleneck_pose']
        # Note that you need to call item() in the below, otherwise the loss will never be freed from cuda memory
        minibatch_loss = self._compute_loss(predictions, ground_truths).item()
        # Debugging: print the predictions for some examples
        if 0 and epoch_num % 10 == 0:
            for i, example_id in enumerate(examples['example_id']):
                for j in range(self.num_debug_validation_examples):
                    if example_id == self.debug_validation_examples[j]:
                        prediction = predictions[i].cpu().data.numpy()
                        ground_truth = ground_truths[i].cpu().data.numpy()
                        x_error = np.fabs(prediction[0] - ground_truth[0])
                        y_error = np.fabs(prediction[1] - ground_truth[1])
                        theta_prediction = np.arctan2(prediction[2], prediction[3])
                        theta_ground_truth = np.arctan2(ground_truth[2], ground_truth[3])
                        theta_error = utils.compute_absolute_angle_difference(theta_prediction, theta_ground_truth)
                        error = np.array([x_error, y_error, theta_error])
                        if 1:
                            print('VALIDATION DEBUG NUM ' + str(j) + ' (example id ' + str(example_id.item()) + ') :')
                            print('Prediction:\t ' + str(prediction))
                            print('Ground Truth:\t ' + str(ground_truth))
                            print('Error:\t ' + str(error))
        # Calculate the error
        minibatch_x_error, minibatch_y_error, minibatch_theta_error = self._compute_errors(predictions.detach().cpu().numpy(), ground_truths.detach().cpu().numpy())
        # Get the x, y, z positions, so that we can plot the validation error at each position
        minibatch_poses = examples['endpoint_to_bottleneck_pose'].numpy()
        return minibatch_loss, minibatch_x_error, minibatch_y_error, minibatch_theta_error, minibatch_poses

    def _compute_loss(self, predictions, ground_truths):
        position_loss = self.loss_function(predictions[:, :2], ground_truths[:, :2]).mean()
        orientation_loss = self.loss_function(predictions[:, 2:], ground_truths[:, 2:]).mean()
        loss = position_loss + self.loss_orientation_coefficient * orientation_loss
        return loss

    def _compute_errors(self, predictions, ground_truths):
        x_errors = np.fabs(predictions[:, 0] - ground_truths[:, 0])
        x_error = x_errors.mean(axis=0)
        y_errors = np.fabs(predictions[:, 1] - ground_truths[:, 1])
        y_error = y_errors.mean(axis=0)
        theta_predictions = np.arctan2(predictions[:, 2], predictions[:, 3])
        theta_ground_truths = np.arctan2(ground_truths[:, 2], ground_truths[:, 3])
        num_examples = len(predictions)
        theta_errors = np.zeros([num_examples], dtype=np.float32)
        for example_num in range(num_examples):
            theta_errors[example_num] = utils.compute_absolute_angle_difference(theta_predictions[example_num], theta_ground_truths[example_num])
        theta_error = np.mean(theta_errors)
        return x_error, y_error, theta_error

