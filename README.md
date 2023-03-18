### Run the real sense camera

- `realsense-viewer`

### Setup conda env
- execute `conda env create -f environment_droplet.yml` to create the env with the necessary dependencies..
- activate the created env using `conda activate im_ln`

### Camera Controller
- `robot/cam_controller.py` 
- it has methods to test the camera alond with saving and displaying the images in the opencv window.

### Robot Controller
- #### TODO
- `robot/robot_controller.py`
- To include the code related to robot communication.

### Rviz & Movit panda controller together
- run `roslaunch panda_moveit_config franka_control.launch  robot_ip:=10.162.83.120 `
- we can move the robot in rviz and it is executen ion the actual robot.

## Rviz launch
- run `roslaunch panda_moveit_config demo.launch `
- if we want to activate the tutorials execute the above command with `rviz_tutorial:=true` flag in the end.


****

To train and test the method, there are four steps:

1. "python run_data_collection.py" enables you to record a demonstration, then collect the two image datasets.

2. "python run_dataset_creation.py" enables you to create the self-supervised image dataset, ready for training.

3. "python run_training.py" enables you to train the coarse bottleneck pose estimator, and the last-inch bottleneck pose estimator.

4. "python run_testing.py" enables you to test the controller on the task.

****

## Robot useful cmds
### Joint Error Recovery
- `rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"`
- `rostopic pub -1 /franka_ros_interface/franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"`

### Graphical frontend for load, start, stop controllers 
- `rosrun rqt_controller_manager rqt_controller_manager`
