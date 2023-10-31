import rospy
import numpy as np
import sys
import traceback
# adding to the system path
sys.path.insert(0, '/home/prashanth/Thesis/Imitation-Learning/')


from Common import config
from Robot.sawyer import Sawyer

class Replay:
    def __init__(self) -> None:
        self.robot = Sawyer()
        self.rate = rospy.Rate(config.CONTROL_RATE)

    def replay_joint_positions(self, joint_positions):
        for position in joint_positions:
            self.robot.robot.exec_position_cmd(position)
            self.rate.sleep()
        
    def replay_ee_velocities(self, ee_velocities):
        for ee_velocity in ee_velocities:
            self.robot.set_endpoint_velocity_in_endpoint_frame(ee_velocity)
            self.rate.sleep()
        self.robot.stop()

    def replay_ee_efforts(self, ee_efforts):
        for ee_effort in ee_efforts:
            jacobian_T = self.robot.compute_jacobian().T
            jacobian_T_inv = np.linalg.pinv(jacobian_T)
            joint_torques = np.dot(jacobian_T_inv.T, ee_effort)
            self.robot.robot.exec_torque_cmd(joint_torques)
            self.rate.sleep()
        self.robot.robot.exec_torque_cmd([0.0]*7)

def main():
    rospy.init_node('replay')
    replay = Replay()
    print('Starting in 2 secs')
    rospy.sleep(2.0)
    dir = 'Trial_Code/data/contact'

    #joint_positions = np.load(dir + '/joint_positions.npy', allow_pickle=True)
    #replay.replay_joint_positions(joint_positions)

    # ee_velocities = np.load(dir + '/ee_velocities.npy', allow_pickle=True)
    # replay.replay_ee_velocities(ee_velocities)

    ee_efforts = np.load(dir + '/ee_efforts.npy', allow_pickle=True)
    replay.replay_ee_efforts(ee_efforts)

if __name__ == "__main__":
    main()
    
