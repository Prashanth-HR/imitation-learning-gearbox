from Training.image_to_pose_trainer_coarse import ImageToPoseTrainerCoarse
from Training.pose_to_uncertainty_trainer import PoseToUncertaintyTrainer
from Training.image_to_pose_trainer_fine import ImageToPoseTrainerFine
from Common import config


# FIRST, DO THIS TO TRAIN THE COARSE IMAGE CONTROLLER
if 0:
    num_trajectories = 50
    trainer = ImageToPoseTrainerCoarse(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.train()
    trainer = PoseToUncertaintyTrainer(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.run()


# THEN, DO THIS TO TRAIN THE LAST-INCH IMAGE CONTROLLER
if 0:
    num_trajectories = 50
    trainer = ImageToPoseTrainerFine(task_name=config.TASK_NAME, num_trajectories=num_trajectories)
    trainer.train()
