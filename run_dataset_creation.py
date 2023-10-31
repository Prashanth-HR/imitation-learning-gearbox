from Common import config
from Common import utils
from Dataset_Creation.image_to_pose_dataset_creator_coarse import ImageToPoseDatasetCreatorCoarse
from Dataset_Creation.image_to_pose_dataset_creator_fine import ImageToPoseDatasetCreatorFine


# FIRST, DO THIS TO CREATE THE DATASET FOR THE COARSE IMAGE CONTROLLER CONTROLLER
if 0:
    dataset_creator = ImageToPoseDatasetCreatorCoarse(task_name=config.TASK_NAME)
    dataset_creator.run()


# THEN, DO THIS TO CREATE THE DATASET FOR THE LAST-INCH IMAGE CONTROLLER
if 0:
    dataset_creator = ImageToPoseDatasetCreatorFine(task_name=config.TASK_NAME)
    dataset_creator.run()
