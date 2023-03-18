import numpy as np
import torch
import rospy
import pickle
import PyKDL

from Common import utils
from Common import config
from Training.networks import ImageToPoseNetworkFine
from Robot import kdl_utils


# CoarseController allows us to perform an episode of coarse control, for a particular task
class CorrectionController:

    def __init__(self, task_name, sawyer, camera, ros_rate, num_training_trajectories):
        self.task_name = task_name
        self.sawyer = sawyer
        self.camera = camera
        self.ros_rate = ros_rate
        self.num_training_trajectories = num_training_trajectories
        self.max_translation_speed = config.MAX_TRANSLATION_SPEED
        self.max_rotation_speed = config.MAX_ROTATION_SPEED
        self.is_ros_running = True
        self.image_to_pose_network = None
        rospy.on_shutdown(self._shutdown)

        # Load the network
        self.image_to_pose_network = ImageToPoseNetworkFine(task_name=self.task_name, num_training_trajectories=num_training_trajectories)
        self.image_to_pose_network.load()
        self.image_to_pose_network.eval()

    def run(self, bottleneck_height):
        predicted_bottleneck_pose = self._predict_bottleneck_pose(bottleneck_height)
        self.sawyer.move_to_pose(predicted_bottleneck_pose)

    # PRIVATE FUNCTIONS #
    #####################

    def _predict_bottleneck_pose(self, bottleneck_height):
        # Capture an image
        rgb_image = self.camera.capture_cv_image(resize_image=True, show_image=True, show_big_image=True)
        # Get the true endpoint pose
        true_endpoint_pose = self.sawyer.get_endpoint_pose()
        # Make the pose prediction, together with the uncertainty prediction
        predicted_current_to_bottleneck_pose_3dof = self._predict_endpoint_to_bottleneck_pose_from_rgb_image(rgb_image)
        translation = PyKDL.Vector(predicted_current_to_bottleneck_pose_3dof[0], predicted_current_to_bottleneck_pose_3dof[1], 0)
        rotation = PyKDL.Rotation.RPY(0, 0, predicted_current_to_bottleneck_pose_3dof[2])
        current_to_bottleneck_pose = PyKDL.Frame(rotation, translation)
        # Convert this prediction into the live robot frame
        predicted_bottleneck_pose = true_endpoint_pose * current_to_bottleneck_pose
        # Return this prediction
        return predicted_bottleneck_pose

    def _predict_endpoint_to_bottleneck_pose_from_rgb_image(self, rgb_image):
        # Convert the RGB image from 0->255 to 0->1
        rgb_image = rgb_image / 255.0
        # Create a Torch image by moving the channel axis
        torch_image = np.moveaxis(rgb_image, 2, 0)
        image_tensor = torch.unsqueeze(torch.tensor(torch_image, dtype=torch.float32), 0)
        # Send the image through the network
        prediction = self.image_to_pose_network.forward(image_tensor, torch.tensor([[0]], dtype=torch.float32)).detach().cpu().numpy()[0]
        # Convert the unnormalised prediction to real values
        predicted_theta = np.arctan2(prediction[2], prediction[3])
        predicted_pose = np.array([prediction[0], prediction[1], predicted_theta])
        # Return the predicted pose
        return predicted_pose

    # Function that is called when Control-C is pressed
    # It is better to use is_ros_running rather than rospy.is_shutdown(), because sometimes rospy.is_shutdown() isn't triggered (e.g. if you do Control-C outside of the main ROS control loop, such as with doing position control with Intera, it does not flag that ROS has been shutdown until it is too late)
    def _shutdown(self):
        print('\nControl-C detected: Shutting down ...')
        # utils.reset_terminal()
        self.is_ros_running = False
        print('Shut down complete.\n')
