## Intro
This repository can record robot demonstrations, create datasets for learning to reach bottleneckpose in a self-supervised manner and run tests from the learned model to perform the demonstrated tasks.
Since the Franka robot is interfaced using python, `franka -interface.launch` file in `franka-ros-interface` has to be evecuted before starting the project. 


****

To train and test the method, there are four steps:

1. "python run_data_collection.py" enables you to record a demonstration, then collect the two image datasets.

2. "python run_dataset_creation.py" enables you to create the self-supervised image dataset, ready for training.

3. "python run_training.py" enables you to train the coarse bottleneck pose estimator, and the last-inch bottleneck pose estimator.

4. "python run_testing.py" enables you to test the controller on the task.

****






## Additional - useful cmds
### Joint Error Recovery
- `rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"`
- `rostopic pub -1 /franka_ros_interface/franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"`

### Graphical frontend for load, start, stop controllers 
- `rosrun rqt_controller_manager rqt_controller_manager`

### Launch F/T sensor
- `roslaunch rokubimini_ethercat rokubimini_ethercat.launch`
- additionally `roslaunch bota_demo BFT_SENS_ECAT_M8.launch`

### Matlab
- `matlab`
- if graphs doesnot work `export MESA_LOADER_DRIVER_OVERRIDE=i965 && matlab`
- if have any issues with CVX, open ~/Matlab/cvx in matlab and run `cvx_setup`