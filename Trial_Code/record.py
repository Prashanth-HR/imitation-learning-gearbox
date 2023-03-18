import rospy
import numpy as np

import sys
# adding to the system path
sys.path.insert(0, '/home/prashanth/Thesis/Imitation-Learning/')

from Common import config
from Robot.sawyer import Sawyer
from Common import utils

class Recorder:
    def __init__(self) -> None:
        self.robot = Sawyer()
        self.rate = rospy.Rate(config.CONTROL_RATE)
        self.save_dir = 'Trial_Code/data/test'
        utils.create_or_clear_directory(self.save_dir)
        utils.set_up_terminal_for_key_check()

    def record_ee_forces(self):
        efforts = []
        print('Recording started... Enter \'x\' to stop...')
        try:
            while not rospy.is_shutdown():
                force, torque = self.robot.get_endpoint_effort_in_base_frame()
                efforts.append([*force,*torque])
                
                # Check if the demo has ended
                if utils.check_for_key('x'):
                    print('###### Recording stopped ######')
                    np.save(self.save_dir + '/ee_efforts.npy', efforts)
                    break
                # Sleep until the next loop
                self.rate.sleep()
        finally:
            utils.reset_terminal()

    def record_ee_velocities(self):
        velocities = []
        print('Recording started... Enter \'x\' to stop...')
        try:
            while not rospy.is_shutdown():
                translation_velocity, rotation_velocity = self.robot.get_endpoint_velocity_in_endpoint_frame()
                velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[0], rotation_velocity[1], rotation_velocity[2]])
                velocities.append(velocity_vector)
                
                # Check if the demo has ended
                if utils.check_for_key('x'):
                    print('###### Recording ended ######')
                    np.save(self.save_dir + '/ee_velocities.npy', velocities)
                    break
                # Sleep until the next loop
                self.rate.sleep()
        finally:
            utils.reset_terminal()

    def record_ee_poses(self):
        poses = []
        print('Recording started... Enter \'x\' to stop...')
        try:
            while not rospy.is_shutdown():
                endpoint_pose = self.robot.get_endpoint_pose()
                poses.append(endpoint_pose)
                
                # Check if the demo has ended
                if utils.check_for_key('x'):
                    print('###### Recording ended ######')
                    np.save(self.save_dir + '/ee_poses.npy', poses)
                    break
                # Sleep until the next loop
                self.rate.sleep()
        finally:
            utils.reset_terminal()

    def record(self):
        poses, velocities, forces, joint_positions, joint_efforts = ([] for i in range(5))
        print('Recording started... Enter \'x\' to stop...')
        try:
            while not rospy.is_shutdown():
                # poses
                endpoint_pose = self.robot.get_endpoint_pose()
                poses.append(endpoint_pose)
                # velocities
                translation_velocity, rotation_velocity = self.robot.get_endpoint_velocity_in_endpoint_frame()
                velocity_vector = np.array([translation_velocity[0], translation_velocity[1], translation_velocity[2], rotation_velocity[0], rotation_velocity[1], rotation_velocity[2]])
                velocities.append(velocity_vector)
                # forces
                force, torque = self.robot.get_endpoint_effort_in_base_frame()
                forces.append([*force,*torque])
                # joint_positions
                joint_position = self.robot.get_joint_angles()
                joint_positions.append(joint_position)
                # joint_efforts
                joint_effort = self.robot.get_joint_efforts()
                joint_efforts.append(joint_effort)

                # Check if the demo has ended
                if utils.check_for_key('x'):
                    print('###### Recording ended ######')
                    np.save(self.save_dir + '/ee_poses.npy', poses)
                    np.save(self.save_dir + '/ee_velocities.npy', velocities)
                    np.save(self.save_dir + '/ee_forces.npy', forces)
                    np.save(self.save_dir + '/joint_positions.npy', joint_positions)
                    np.save(self.save_dir + '/joint_efforts.npy', joint_efforts)
                    break
                # Sleep until the next loop
                self.rate.sleep()
        finally:
            utils.reset_terminal()
    
def main():
    rospy.init_node('record')
    recorder = Recorder()
    print('Starting in 2 secs')
    rospy.sleep(2.0)
    recorder.record_ee_poses()

if __name__ == "__main__":
    main()