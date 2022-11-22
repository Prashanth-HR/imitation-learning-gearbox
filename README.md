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

### Rviz launch
- run `roslaunch panda_moveit_config demo.launch `
- if we want to activate the tutorials execute the above command with `rviz_tutorial:=true` flag in the end.