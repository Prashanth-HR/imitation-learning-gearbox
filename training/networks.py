import numpy as np
import torch
import torchvision
import cv2
import torch.nn.functional as F

from common import config
from common import utils


class Network(torch.nn.Module):

    def __init__(self, network_path):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Set the class variables from the arguments
        self.network_path = network_path

    def save(self):
        torch.save(self.state_dict(), self.network_path)

    def load(self):
        print('Loading network from: ' + str(self.network_path))
        state_dict = torch.load(self.network_path)
        self.load_state_dict(state_dict)

    def set_eval_dropout(self):
        self.apply(self.apply_dropout)

    @staticmethod
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    def freeze_features(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def get_num_trainable_params(self):
        # Each param in the below loop is one part of one layer
        # e.g. the first param is all the CNN weights in the first layer, the second param is all the biases in the first layer, the third param is all the CNN weights in the second layer, etc.
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params

    def print_architecture(self):
        print('Network architecture:')
        print(self)

    def print_weights(self):
        print('Network weights:')
        print('Feature extractor:')
        for name, param in self.feature_extractor.named_parameters():
            if 'weight' in name:
                print('name = ' + str(name))
                print('shape = ' + str(param.shape))
                print('values = ' + str(param[0, 0, 0]))
        print('Predictor:')
        for name, param in self.predictor.named_parameters():
            if 'weight' in name:
                print('name = ' + str(name))
                print('shape = ' + str(param.shape))
                print('values = ' + str(param[0, 0]))

    def print_gradients(self):
        print('Network gradients:')
        print('CNNs')
        for name, param in self.cnns[1].named_parameters():
            if 'weight' in name:
                print(param.grad[0, 0, 0, :3])
        print('FCs')
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                print(param.grad[0, :3])


# Image size: 64
class ImageToPoseNetworkCoarse(Network):

    def __init__(self, task_name, num_training_trajectories):
        # If the network directory doesn't exist (it has not been trained yet), then create a new one.
        # Otherwise (it has already been trained), use the existing directory.
        utils.create_directory_if_none('../Networks/' + str(task_name))
        # Call the parent constructor, which will set the save path.
        image_to_pose_network_path = '../Networks/' + str(task_name) + '/image_to_pose_network_coarse_' + str(num_training_trajectories) + '.torch'
        Network.__init__(self, image_to_pose_network_path)
        # Define the network layers
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),

        )
        self.flat_size = 2 * 2 * 128
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.flat_size, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(201, 50),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(50, 4)
        )

    def forward(self, input_image, height):
        # Compute the cnn features
        image_features = self.conv(input_image)
        image_features_flat = torch.reshape(image_features, (input_image.shape[0], -1))
        # Compute the mlp features
        mlp_features = self.mlp(image_features_flat)
        # Concatenate the z value
        combined_features = torch.cat((mlp_features, height), dim=1)
        # Make the prediction
        prediction = self.predictor(combined_features)
        return prediction


# Image size: 64
class ImageToPoseNetworkFine(Network):

    def __init__(self, task_name, num_training_trajectories):
        # If the network directory doesn't exist (it has not been trained yet), then create a new one.
        # Otherwise (it has already been trained), use the existing directory.
        utils.create_directory_if_none('../Networks/' + str(task_name))
        # Call the parent constructor, which will set the save path.
        image_to_pose_network_path = '../Networks/' + str(task_name) + '/image_to_pose_network_fine_' + str(num_training_trajectories) + '.torch'
        Network.__init__(self, image_to_pose_network_path)
        # Define the network layers
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),

        )
        self.flat_size = 2 * 2 * 128
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.flat_size, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(201, 50),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(50, 4)
        )

    def forward(self, input_image, height):
        # Compute the cnn features
        image_features = self.conv(input_image)
        image_features_flat = torch.reshape(image_features, (input_image.shape[0], -1))
        # Compute the mlp features
        mlp_features = self.mlp(image_features_flat)
        # Concatenate the z value
        combined_features = torch.cat((mlp_features, height), dim=1)
        # Make the prediction
        prediction = self.predictor(combined_features)
        return prediction
