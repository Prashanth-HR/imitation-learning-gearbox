import rospy
import numpy as np

import sys
# adding to the system path
sys.path.insert(0, '/home/prashanth/Thesis/Imitation-Learning/')

import rosbag
from Common import config
from Robot.sawyer import Sawyer
from Robot.ft_sensor import FTSensor
from Robot import kdl_utils
from Common import utils
from  franka_core_msgs.msg import RobotState


class Recorder:
    def __init__(self, rate, folder_name = 'default') -> None:
        self.robot = Sawyer()
        self.ftsensor = FTSensor(should_plot=False)
        self.rate = rate
        self.folder_name = folder_name
        self.save_dir = 'Trial_Code/data/' 

        self.rob_state_sub = rospy.Subscriber('/franka_ros_interface/custom_franka_state_controller/robot_state', RobotState, self._rob_state_callback)
        self.robot_state = None
        #utils.create_or_clear_directory(self.save_dir)
        utils.set_up_terminal_for_key_check()

    def _rob_state_callback(self, msg):
        self.robot_state = msg

    def record_bottelneck_pose(self):
        bottleneck_pose = self.robot.get_endpoint_pose()
        bottleneck_pose_vertical = kdl_utils.create_vertical_pose_from_x_y_z_theta(bottleneck_pose.p[0], bottleneck_pose.p[1], bottleneck_pose.p[2], bottleneck_pose.M.GetRPY()[2])
        bottleneck_pose_vector = kdl_utils.create_vector_from_pose(bottleneck_pose)
        np.save( self.save_dir + '/random_pose_vector.npy', bottleneck_pose_vector)

    def record_ee_forces(self):
        # using the franka F/T sensors in the joints
        force, torque = self.robot.get_endpoint_effort_in_base_frame()
        return [*force,*torque]
                

    def record_ee_velocities(self):
        translation_velocity, rotation_velocity = self.robot.get_endpoint_velocity_in_endpoint_frame()
        velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[0], rotation_velocity[1], rotation_velocity[2]])
        return velocity_vector   

    def record_ee_poses(self):
        # return position vector
        endpoint_pose = self.robot.get_endpoint_pose()
        return kdl_utils.create_vector_from_pose(endpoint_pose)
                

    def record_FT_sensor_forces(self):
        # return force vector
        ft_to_list = self.ftsensor.data    
        return ft_to_list      
                

    def record(self):
        poses, velocities, forces = ([] for i in range(3))
        file_name = input('\nEnter the filename\n')
        print(f'\nYou entered {file_name}')
        print('Starting recording in 2 secs')
        rospy.sleep(2.0)
        print('Recording started... Enter \'x\' to stop...')
        bag = rosbag.Bag(self.save_dir + self.folder_name +'/'+ file_name+'_robot_state.bag', 'w')
        try:
            while True:
                # poses
                endpoint_pose = self.record_ee_poses()
                poses.append(endpoint_pose)
                # forces
                ft_to_list = self.record_FT_sensor_forces()
                forces.append(ft_to_list)

                # robot state
                bag.write('robot_state', self.robot_state)

                # Check if the demo has ended
                if utils.check_for_key('x'):
                    # Save when the robot stops moving
                    
                    np.save(self.save_dir + self.folder_name +'/'+ file_name + '_ee_poses.npy', poses)
                    np.save(self.save_dir + self.folder_name +'/'+ file_name + '_ee_forces.npy', forces)
                    print(f'\nData saved {self.save_dir , self.folder_name , file_name}')
                    print('###### Recording ended ######')
                    break
                # Sleep until the next loop
                self.rate.sleep()
        finally:
            utils.reset_terminal()
            bag.close()
    
def main():
    rospy.init_node('record')
    rate = rospy.Rate(100)
    recorder = Recorder(rate)

    print('Starting in 5 secs')
    rospy.sleep(5.0)
    try: 
        while not rospy.is_shutdown():
            # move to init pose
            if utils.check_for_key('b'):
                print('Moved to B-Pose Start')
                bottleneck_pose_vector = np.load('Trial_Code/data/bottleneck_pose_vector.npy', allow_pickle=True)
                bottelneck_pose = kdl_utils.create_pose_from_vector(bottleneck_pose_vector)
                recorder.robot.move_to_pose(bottelneck_pose)
                print('Moved to B-Pose Done')
                #utils.reset_terminal()

            # move to neutral
            if utils.check_for_key('n'):
                print('Moved to Neutral Start')
                recorder.robot.robot.move_to_neutral()
                print('Moved to Neutral Done')

            # error recovery
            if utils.check_for_key('e'):
                print('Error Recovery Started')
                recorder.robot.error_recovery()
                rospy.sleep(2.0)
                print('Error Recovery Done')

            # record 
            if utils.check_for_key('r'):
                recorder.record()

            if utils.check_for_key('f'):
                folder_name = input('\nEnter the foldername')
                utils.create_directory_if_not_exist(recorder.save_dir + folder_name)
                recorder.folder_name = folder_name
                print(f'\nYou entered {folder_name}')
            
    finally:
            utils.reset_terminal()
    


if __name__ == "__main__":
    main()