#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('pose_ekf_slam')
import rospy
import tf
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from pose_ekf_slam.srv import SetPositionRequest, SetPosition
import PyKDL
import numpy as np

class Navigator :
    
    def __init__(self, name):
        """ Merge different navigation sensor values  """
        self.name = name
        
        # Create publisher
        self.pub_imu = rospy.Publisher("/pose_ekf_slam/imu_input", 
                                       Imu)
        self.pub_vel = rospy.Publisher("/pose_ekf_slam/velocity_update", 
                                       TwistWithCovarianceStamped)
        self.pub_pose = rospy.Publisher("/pose_ekf_slam/pose_update", 
                                        PoseWithCovarianceStamped)
        self.pub_landmark = rospy.Publisher("/pose_ekf_slam/landmark_update/landmark_0", 
                                            PoseWithCovarianceStamped)
        
        # Create Subscriber Inputs (u)
        rospy.Subscriber("/dataNavigator", Odometry, self.odomInput)
 
        
        # o = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0, 'sxyz')

        # Create timers
        rospy.Timer(rospy.Duration(0.5), self.pubImu)
        # rospy.Timer(rospy.Duration(0.5), self.pubVel)
        rospy.Timer(rospy.Duration(2.0), self.pubPose)
        # rospy.Timer(rospy.Duration(2.0), self.pubLandmark)
        
        # Init pose
        self.position = 0
        self.odom = Odometry()

        
    def pubImu(self, event):
        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = 'imu'
        imu.orientation.x = 0.0
        imu.orientation.y = 0.0
        imu.orientation.z = 0.0
        imu.orientation.w = 1.0
        imu.linear_acceleration.x = 0.00
        self.pub_imu.publish(imu)
        
        # Publish TF
        imu_tf = tf.TransformBroadcaster()
        o = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0, 'sxyz')
        imu_tf.sendTransform((0.0, 0.0, 0.0), o, imu.header.stamp, 
                             imu.header.frame_id, 'robot')


    def pubVel(self, event):
        v = TwistWithCovarianceStamped()
        v.header.stamp = rospy.Time.now()
        v.header.frame_id = 'vel'
        
        # create a new velocity
        v.twist.twist.linear.x = 0.9
        v.twist.twist.linear.y = 0.0
        v.twist.twist.linear.z = 0.0
        
        # Only the number in the covariance matrix diagonal 
        # are used for the updates!
        v.twist.covariance[0] = 0.01
        v.twist.covariance[7] = 0.01
        v.twist.covariance[14] = 0.01
        
        print 'twist msg:', v
        self.pub_vel.publish(v)
        
        # Publish TF
        vel_tf = tf.TransformBroadcaster()
        o = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0, 'sxyz')
        vel_tf.sendTransform((0.0, 0.0, 0.0), o, v.header.stamp, 
                             v.header.frame_id, 'robot')      
        
        
    def pubPose(self, event):
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = 'pose'
        
        # Create a new pose
        p.pose.pose.position.x = self.position
        p.pose.pose.position.y = 0.0
        p.pose.pose.position.z = 0.0
        self.position = self.position + 2
       
        # Only the number in the covariance matrix diagonal
        # are used for the updates!
        # Example of X, Y update without modifying the Z
        p.pose.covariance[0] = 1.0
        p.pose.covariance[7] = 1.0
        p.pose.covariance[14] = 9999.0
        
        print 'pose msg:', p
        self.pub_pose.publish(p)
        
        # Publish TF
        pose_tf = tf.TransformBroadcaster()
        o = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0, 'sxyz')
        pose_tf.sendTransform((0.0, 0.0, 0.0), o, p.header.stamp, 
                              p.header.frame_id, 'robot')      
        
        
    def odomInput(self, odom):
        self.odom = odom

        
    def pubLandmark(self, event):
            
        # Create transformation
        angle = tf.transformations.euler_from_quaternion(
            [self.odom.pose.pose.orientation.x,
             self.odom.pose.pose.orientation.y,
             self.odom.pose.pose.orientation.z,
             self.odom.pose.pose.orientation.w])

        r = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
        t = PyKDL.Vector(self.odom.pose.pose.position.x, 
                         self.odom.pose.pose.position.y, 
                         self.odom.pose.pose.position.z)
        
        
        p_t = PyKDL.Vector(10.0, 5.0, 3.0)
        
        rel_pose = r * (p_t - t)
        
        print 'landmark_0 pose wrt robot: ', rel_pose
            
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = 'landmark_0'
        p.pose.covariance[0] = 0.05
        p.pose.covariance[7] = 0.05
        p.pose.covariance[14] = 0.05

        # Publish TF
        pose_tf = tf.TransformBroadcaster()
        o = tf.transformations.quaternion_from_euler(0., 0., 0., 'sxyz')
        pose_tf.sendTransform((rel_pose.x() + np.random.normal(0.0, 0.05), 
                               rel_pose.y() + np.random.normal(0.0, 0.05), 
                               rel_pose.z() + np.random.normal(0.0, 0.05)), 
                              o, 
                              p.header.stamp, 
                              p.header.frame_id, 
                              'robot')
        self.pub_landmark.publish(p)
        
              
if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('test_ekf_slam')
        navigator = Navigator(rospy.get_name())
        
        # Init pose_ekf_slam server
        try:
            rospy.wait_for_service('/pose_ekf_slam/set_position', 5)
            set_pose = rospy.ServiceProxy('pose_ekf_slam/set_position', 
                                          SetPosition)
            sp = SetPositionRequest()
            sp.position.x = 0.0
            sp.position.y = 0.0
            sp.position.z = 0.0
            set_pose(sp)
        except rospy.exceptions.ROSException:
            rospy.logerr('Error initializing Pose-EKF-SLAM.')
            
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass
