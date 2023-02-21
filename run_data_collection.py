from Common import config
from Data_Collection.single_demo_collector_short import SingleDemoCollectorShort
from Data_Collection.automatic_demo_collector import AutomaticDemoCollector
from Data_Collection.automatic_demo_collector_fine import AutomaticDemoCollectorFine


task_name = config.TASK_NAME

# FIRST, DO THIS TO RECORD A DEMO
# This allows the user to collect the single demo, as used by my method
# First, the robot requests the bottleneck, and then, the robot requests the demo
# It saves both the bottleneck and the demo
if 0:
    data_collector = SingleDemoCollectorShort(task_name=task_name)
    data_collector.run()

# THEN, DO THIS TO COLLECT THE COARSE IMAGE DATASET
# This collects images automatically by moving the robot into the bottleneck over multiple automatic demos
# This is for the coarse part of the trajectory
if 0:
    num_trajectories = config.NO_OF_TRAJECTORIES
    data_collector = AutomaticDemoCollector(task_name=task_name, total_num_trajectories=num_trajectories, num_timesteps_per_image=1, max_translation_speed=config.MAX_TRANSLATION_SPEED, max_rotation_speed=config.MAX_ROTATION_SPEED)
    data_collector.run()

# THEN, DO THIS TO COLLECT THE LAST-INCH DATASET
# This collects images automatically by moving the robot into the bottleneck over multiple automatic demos
# This is for the last-inch correction part of the trajectory
if 0:
    num_trajectories = config.NO_OF_TRAJECTORIES
    data_collector = AutomaticDemoCollectorFine(task_name=task_name, total_num_trajectories=num_trajectories, num_timesteps_per_image=1, max_translation_speed=config.MAX_TRANSLATION_SPEED, max_rotation_speed=config.MAX_ROTATION_SPEED)
    data_collector.run()
