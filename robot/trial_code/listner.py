#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from moveit_msgs.msg import PlanningScene

def callback(data):
    rospy.loginfo_throttle(1, data)
    # rospy.loginfo_once(data.position)
    # rospy.loginfo_once(data.velocity)
    # rospy.loginfo_once(data.effort)
    
def listener():
    rospy.init_node('listener', anonymous=True)

    #rospy.Subscriber("/move_group/fake_controller_joint_states", JointState, callback)
    rospy.Subscriber("/move_group/monitored_planning_scene", PlanningScene, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()