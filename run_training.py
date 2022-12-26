from training.image_to_pose_trainer_coarse import ImageToPoseTrainerCoarse
from training.pose_to_uncertainty_trainer import PoseToUncertaintyTrainer
from training.image_to_pose_trainer_fine import ImageToPoseTrainerFine
from common import config


# FIRST, DO THIS TO TRAIN THE COARSE IMAGE CONTROLLER
if 0:
    num_trajectories = config.NO_OF_TRAJECTORIES
    trainer = ImageToPoseTrainerCoarse(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.train()
    trainer = PoseToUncertaintyTrainer(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.run()


# THEN, DO THIS TO TRAIN THE LAST-INCH IMAGE CONTROLLER
if 1:
    num_trajectories = config.NO_OF_TRAJECTORIES
    trainer = ImageToPoseTrainerFine(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.train()
