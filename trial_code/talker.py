
import rospy
from sensor_msgs.msg import JointState
from moveit_msgs.msg import PlanningScene
from std_msgs.msg import Header

def talker():
    pub = rospy.Publisher('/move_group/monitored_planning_scene', PlanningScene, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 10hz

    #while not rospy.is_shutdown():
    plan = PlanningScene()
    
    joint_state = JointState()
    # joint_state.header = Header(0, rospy.Time.now(), '')
    joint_state.position = [-0.01,-0.72, 0.01, -2.3, 0.01,1.57, 1.2 ]
    joint_state.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    
    plan.robot_state.joint_state = joint_state
    rospy.loginfo(joint_state)
    pub.publish(plan)
    rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass