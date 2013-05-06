#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('pose_ekf_slam')
import rospy
import tf
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
# from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# Custom msgs
from pose_ekf_slam.srv import SetPosition, SetPositionResponse, SetPositionRequest
from pose_ekf_slam.srv import SetLandmark, SetLandmarkResponse
from pose_ekf_slam.msg import Landmark, Map

# More imports
from numpy import delete, dot, zeros, eye, cos, sin, array, diag, sqrt
# from numpy import linalg
from numpy import matrix, asarray, squeeze, mean, var
from collections import deque

import threading
import motion_transformations as mt
import PyKDL
import math

class PoseEkfSlam :
    def __init__(self, name, p_var, q_var):
        self.name = name
        self.p_var = p_var
        
        # Output odometry
        self.odom = Odometry()
        
        # Input last Imu
        self.imu = Imu()
        
        # Global vars
        self.last_prediction = rospy.Time.now()
        self.lock = threading.RLock()
        self.listener = tf.TransformListener()
        self.world_frame_name = 'world'
        self.robot_frame_name = 'robot'
        self.getConfig()

        # Init state, P, Landmarks & TFs
        sp = SetPositionRequest()
        sp.position.x = 0.0
        sp.position.y = 0.0
        sp.position.z = 0.0
        self.initEkf(sp.position)

        # Init Q
        self.Q = self.computeQ(q_var)
         
        # Filter status
        self.is_ekf_init = False
        self.is_imu_init = False
        
        # Create publisher
        self.pub_odom = rospy.Publisher('/pose_ekf_slam/odometry', Odometry)
        self.pub_map = rospy.Publisher('/pose_ekf_slam/map', Map)
        self.pub_landmarks = rospy.Publisher('/pose_ekf_slam/landmarks', 
                                              MarkerArray)
        self.covariance_marker = rospy.Publisher(
            '/pose_ekf_slam/covariance_marker', Marker)
         
        # Create Subscriber Updates (z)
        rospy.Subscriber('/pose_ekf_slam/pose_update', 
                         PoseWithCovarianceStamped, 
                         self.poseUpdate)
        rospy.Subscriber('/pose_ekf_slam/velocity_update', 
                         TwistWithCovarianceStamped, 
                         self.velocityUpdate)
                         
        # Create Subscriber Inputs (u)
        rospy.Subscriber("/pose_ekf_slam/imu_input", Imu, self.imuInput)
 
        # Create Service SetPosition
        self.set_position = rospy.Service('/pose_ekf_slam/set_position', 
                                          SetPosition, 
                                          self.setPosition)
                                          
        self.set_landmark = rospy.Service('/pose_ekf_slam/set_landmark', 
                                          SetLandmark, 
                                          self.setLandmark)

        rospy.Timer(rospy.Duration(10.0), self.searchForLandmarkUpdates)
   
 
    def searchForLandmarkUpdates(self, event):
        # rospy.loginfo("%s: Search for new landmark updates ...", self.name)
        topics = rospy.get_published_topics('/pose_ekf_slam/landmark_update')
        for t in topics:
            if t[1] == 'geometry_msgs/PoseWithCovarianceStamped':
                if t[0] not in self.landmark_update_topics:
                    rospy.loginfo('Subscribe to landmark update: %s', t[0])
                    self.landmark_update_topics.append(t[0])
                    rospy.Subscriber(t[0], PoseWithCovarianceStamped, 
                                     self.landmarkUpdate, t[0])
                    
        
    def setPosition(self, req):
        self.initEkf(req.position)
        
        # Init filter
        self.is_ekf_init = True
        
        return SetPositionResponse()

    
    def setLandmark(self, req):
        self.lock.acquire()
        self.mapped_lamdmarks[req.topic_name] = self.number_of_landmarks
        self.landmark_values[self.number_of_landmarks] = req.topic_name
        self.landmark_last_update[self.number_of_landmarks] = rospy.Time().now()
                                
        rospy.loginfo('%s, Add feature %s', 
                      self.name, req.topic_name)
                                  
        cov = matrix(req.landmark.covariance).reshape(6,6)
        self.addLandmark(req.landmark.pose.position.x, 
                         req.landmark.pose.position.y, 
                         req.landmark.pose.position.z, 
                         cov[0:3, 0:3])
        self.lock.release()
        return SetLandmarkResponse()
        
        
    def imuInput(self, data):        
        self.imu = data
        tf_done = True
        
        if data.header.frame_id != self.robot_frame_name:
            # Better publish imu in robot_frame! 
            if data.header.frame_id in self.tf_cache:
                # Recover TF from cached TF                
                trans = self.tf_cache[data.header.frame_id][0]
                rot = self.tf_cache[data.header.frame_id][1]
            
                # Transform orientations
                qt = mt.orientation(data.orientation, rot)
                self.imu.orientation.x = qt[0]
                self.imu.orientation.y = qt[1]
                self.imu.orientation.z = qt[2]
                self.imu.orientation.w = qt[3]
                
                # Transform orientation covariance
                # TODO: To be check and slow! 
                o_cov = mt.orientationCov(
                    array(data.orientation_covariance).reshape(3,3),
                    self.imu.orientation,
                    rot)
                    
                # TODO: Check that!
                self.imu.orientation_covariance = o_cov.ravel()                
                
                # Transform angular_velocity
                w_r = mt.angularVelocity(data.angular_velocity, rot)
                self.imu.angular_velocity.x = w_r[0]
                self.imu.angular_velocity.y = w_r[1]
                self.imu.angular_velocity.z = w_r[2]
                
                # Transform angular velocity covariance
                w_cov = mt.angularVelocityCov(
                    array(data.angular_velocity_covariance).reshape(3,3), 
                    rot)
                    
                # TODO: Check that!
                self.imu.angular_velocity_covariance = w_cov.ravel()
                
            else:
                try:
                    self.listener.waitForTransform(self.robot_frame_name, 
                                                   data.header.frame_id, 
                                                   data.header.stamp, 
                                                   rospy.Duration(2.0))
                                                       
                    (trans, rot) = self.listener.lookupTransform(
                                        self.robot_frame_name, 
                                        data.header.frame_id, 
                                        data.header.stamp)
                                        
                    self.tf_cache[data.header.frame_id] = [trans, rot]
                    
                except (tf.LookupException, 
                    tf.ConnectivityException, 
                    tf.ExtrapolationException):
                
                    tf_done = False
                    rospy.logerr('%s, Define TF for %s input data!', 
                                 self.name, 
                                 data.header.frame_id)    
                
        if tf_done:
            # Copy input orientation/angular_velocity 
            # into odometry message to be published
            self.lock.acquire()
            self.odom.pose.pose.orientation.x = self.imu.orientation.x
            self.odom.pose.pose.orientation.y = self.imu.orientation.y
            self.odom.pose.pose.orientation.z = self.imu.orientation.z
            self.odom.pose.pose.orientation.w = self.imu.orientation.w
            self.odom.twist.twist.angular.x = self.imu.angular_velocity.x
            self.odom.twist.twist.angular.y = self.imu.angular_velocity.y
            self.odom.twist.twist.angular.z = self.imu.angular_velocity.z
            
            self.is_imu_init = True
            # Make a prediction
            
            if self.makePrediction(data.header.stamp):
                self.updatePrediction()
                self.publishData(data.header.stamp)
            
            self.lock.release()


    def poseUpdate(self, pose_msg):
        """ pose_msg is a geometry_msgs/PoseWithCovarianceStamped msg """
        tf_done = True        
        print 'pose update:\n', pose_msg
        self.lock.acquire()
                
        # Try to make a prediction if 'u' is present and the filter is init
        if self.makePrediction(pose_msg.header.stamp):
            covariance_pose = self.takeCovariance(pose_msg.pose.covariance)
            if pose_msg.header.frame_id != self.robot_frame_name:
                if pose_msg.header.frame_id in self.tf_cache:
                    # Recover TF from cached TF                
                    trans = self.tf_cache[pose_msg.header.frame_id][0]
                    rot = self.tf_cache[pose_msg.header.frame_id][1]
                    
                    # Transform position
                    p = mt.position(pose_msg.pose.pose.position, 
                                    self.imu.orientation, 
                                    trans)
                                    
                    # Transform position covariance       
                    covariance_pose = mt.positionCovariance(
                                        covariance_pose, 
                                        array(self.imu.orientation_covariance).reshape(3,3), 
                                        self.imu.orientation, 
                                        trans)
                                        
                else:
                    tf_done = False
                    # Wait for the tf between the pose sensor and the robot
                    try:
                        self.listener.waitForTransform(self.robot_frame_name, 
                                                       pose_msg.header.frame_id, 
                                                       pose_msg.header.stamp,
                                                       rospy.Duration(2.0))
                                                       
                        (trans, rot) = self.listener.lookupTransform(
                                           self.robot_frame_name, 
                                           pose_msg.header.frame_id, 
                                           pose_msg.header.stamp)
                                           
                        # Pre cache TF
                        print 'get pose TF cached!'
                        self.tf_cache[pose_msg.header.frame_id] = [trans, rot]
                        
                    except (tf.LookupException, 
                        tf.ConnectivityException, 
                        tf.ExtrapolationException,
                        tf.Exception):
                        
                        rospy.logerr('%s, Define TF for %s update!', 
                                     self.name, pose_msg.header.frame_id)   
                
            else:
                p = [pose_msg.pose.pose.position.x, 
                     pose_msg.pose.pose.position.y, 
                     pose_msg.pose.pose.position.z]
            
            if tf_done:
                # Create measurement z and matrices r & h
                z, r, h, v = self.createPoseMeasures(
                                  p, covariance_pose)
    
                # Compute Filter Update
                self.applyUpdate(z, r, h, v, 10.0)
                
                # Publish Data
                self.publishData(pose_msg.header.stamp)
        # else:
        #     rospy.logerr('%s, Impossible to apply pose_update!', self.name)
                         
        self.lock.release()
                
           
    def velocityUpdate(self, twist_msg):
        """ twist_msg is a geometry_msgs/TwistWithCovariance msg """
        tf_done = True
        self.lock.acquire()
        if self.makePrediction(twist_msg.header.stamp):
            velocity_r = self.takeCovariance(twist_msg.twist.covariance)
            if twist_msg.header.frame_id != self.robot_frame_name:
                if twist_msg.header.frame_id in self.tf_cache:
                    # Recover TF from cached TF                
                    trans = self.tf_cache[twist_msg.header.frame_id][0]
                    rot = self.tf_cache[twist_msg.header.frame_id][1]
                    
                    # Transform linear velocity
                    v = mt.linearVelocity(twist_msg.twist.twist.linear, 
                                          self.imu.angular_velocity, 
                                          trans, rot)
                                          
                    # Transform linear velocity covariance     
                    velocity_r = mt.linearVelocityCov(
                                    velocity_r,
                                    array(self.imu.angular_velocity_covariance).reshape(3,3), 
                                    trans, rot)
                else:                    
                    tf_done = False
                    print 'waiting for ', twist_msg.header.frame_id, ' message'
                    # Wait for the tf between the velocity sensor and the robot
                    try:
                        self.listener.waitForTransform(self.robot_frame_name, 
                                                       twist_msg.header.frame_id, 
                                                       twist_msg.header.stamp,  
                                                       rospy.Duration(2.0))
                                                       
                        (trans, rot) = self.listener.lookupTransform(
                                            self.robot_frame_name, 
                                            twist_msg.header.frame_id, 
                                            twist_msg.header.stamp)
                        
                        # Pre cache TF
                        self.tf_cache[twist_msg.header.frame_id] = [trans, rot]
                    
                    except (tf.LookupException, 
                            tf.ConnectivityException, 
                            tf.ExtrapolationException,
                            tf.Exception):
                        rospy.logerr('%s, Define TF for %s update!', 
                                     self.name, 
                                     twist_msg.header.frame_id)
            else:
                v = [twist_msg.twist.twist.linear.x, 
                     twist_msg.twist.twist.linear.y, 
                     twist_msg.twist.twist.linear.z]
            
            if tf_done:
                # Create measurement z and R matrix
                z = array([v[0], v[1], v[2]])
                velocity_h = self.velocityH()
                
                # Compute Filter Update
                self.applyUpdate(z, velocity_r, velocity_h, eye(3), 8.0)
         
                # Publish Data           
                self.publishData(twist_msg.header.stamp)
                
        self.lock.release()
           
    
    def landmarkUpdate(self, landmark_msg, topic_name):
        """ landmark_msg is a geometry_msgs/PoseWithCovariance msg wrt 
        objects frame. Then, position is (0, 0, 0) and orientation is 
        (0, 0, 0, 0,). It is necessary to check its TF to know the object
        position wrt the vehicle """
        
        # print 'Received message: ', landmark_msg
        # print 'from: ', topic_name
        
        ################## Check landmark orientation #######################
        #                                                                   #
        # angle = tf.transformations.euler_from_quaternion([landmark_msg.pose.pose.orientation.x,
        #                                                   landmark_msg.pose.pose.orientation.y,
        #                                                   landmark_msg.pose.pose.orientation.z,
        #                                                   landmark_msg.pose.pose.orientation.w])        
        # for i in [0,2]:
        #     if abs(angle[i]) < deg2rad(35):
        #         pass # Around 0 degrees [-35, 35]
        #     elif abs(abs(angle[i]) - math.pi) < deg2rad(35):
        #         pass # Around 180 degress [145, 215]
        #     else:
        #         print 'invalid landmark ', topic_name
        #         return False
        #                                                                   #
        #####################################################################
            
        self.lock.acquire()
        if self.makePrediction(landmark_msg.header.stamp):
            if topic_name in self.mapped_lamdmarks:
                if landmark_msg.header.frame_id in self.tf_cache:
                    # print 'landmark ', topic_name, ' mapped and TF cached'
                    trans = self.tf_cache[landmark_msg.header.frame_id][0]
                    rot = self.tf_cache[landmark_msg.header.frame_id][1]
                    
                    # Transform landmark pose to vehicle frame
                    pose = mt.landmarkPosition(
                        landmark_msg.pose.pose.position, rot, trans)                    
                    
                    # Create measurement z and R matrix
                    z = array([pose[0], pose[1], pose[2]])
                    # print 'measured z: ', z
                    
                    landmark_id = self.mapped_lamdmarks[topic_name]
                    
                    angle = tf.transformations.euler_from_quaternion(
                        [self.imu.orientation.x, self.imu.orientation.y,
                         self.imu.orientation.z, self.imu.orientation.w])
                    
                    Or = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
                    rot = matrix([Or[0,0], Or[0,1], Or[0,2], 
                                  Or[1,0], Or[1,1], Or[1,2], 
                                  Or[2,0], Or[2,1], Or[2,2]]).reshape(3,3)
    
                    h = self.landmarkH2(landmark_id, rot.T)
                    
                    r = self.takeCovariance(landmark_msg.pose.covariance)             
                    
                    # Compute Filter Update
                    self.applyUpdate(z, r, h, eye(3), 9.0)
                    
                    # Save last update time
                    self.landmark_last_update[self.mapped_lamdmarks[topic_name]] = landmark_msg.header.stamp
                    
                    # Publish Data           
                    self.publishData(landmark_msg.header.stamp)
                else:
                    # print 'landmark ', topic_name, ' waiting for TF...'
                    [trans, rot] = self.getTF(self.robot_frame_name,
                                              landmark_msg.header.frame_id,
                                              landmark_msg.header.stamp)
                    if trans != None:
                        self.tf_cache[landmark_msg.header.frame_id] = [trans,
                                                                       rot]
                      
            else:
                if topic_name in self.candidate_landmarks:
                    print 'landmark ', topic_name, ' is a candidate...'
                    # Add candidate into list
                    [trans, rot] = self.getTF(self.world_frame_name,
                                          landmark_msg.header.frame_id,
                                          rospy.Time())
                    if trans != None:
                        ## Transform landmark pose to world frame
                        pose = mt.landmarkPosition(
                            landmark_msg.pose.pose.position, rot, trans)                    
                        
                        self.candidate_landmarks[topic_name].appendleft(
                            array([pose[0], pose[1], pose[2]]))
                        
                        if len(self.candidate_landmarks[topic_name]) > 4:
                            print '... and have been seen more than 4 times'
                            [pose, cov, update] = self.computeCandidate(self.candidate_landmarks[topic_name])
                            if update:
                                self.mapped_lamdmarks[topic_name] = self.number_of_landmarks
                                self.landmark_values[self.number_of_landmarks] = topic_name
                                self.landmark_last_update[self.number_of_landmarks] = landmark_msg.header.stamp
                                
                                # Get landmark covariance
                                measured_covariance = eye(3)
                                measured_covariance[0, 0] = 0.25
                                measured_covariance[1, 1] = 0.25
                                measured_covariance[2, 2] = 0.25
                                
                                self.addLandmark(pose[0], pose[1], pose[2], 
                                                 measured_covariance)
                                                        
                                rospy.loginfo('%s, Add feature %s', 
                                              self.name, topic_name)
                else:
                    # Add first candidate into list
                    print 'landmark ', topic_name, ' is a candidate seen by first time'
                    [trans, rot] = self.getTF(self.world_frame_name,
                                          landmark_msg.header.frame_id,
                                          rospy.Time())
                    if trans != None:
                        ## Transform landmark pose to world frame
                        pose = mt.landmarkPosition(
                            landmark_msg.pose.pose.position, rot, trans)                    
                        
                        self.candidate_landmarks[topic_name] = deque(maxlen = 5)
                        self.candidate_landmarks[topic_name].appendleft(array([pose[0], pose[1], pose[2]]))
                
        self.lock.release()                         
    
    
    def getTF(self, origin, destination, stamp):
        try:
            self.listener.waitForTransform(origin, 
                                           destination, 
                                           stamp, #rospy.Time(), 
                                           rospy.Duration(0.2))
                                           
            (trans, rot) = self.listener.lookupTransform(
                              origin, destination, stamp)
            
            return [trans, rot]
        
        except (tf.LookupException, 
                tf.ConnectivityException, 
                tf.ExtrapolationException,
                tf.Exception):
            rospy.logerr('%s, Define TF for %s update!', 
                         self.name, 
                         destination)
            return [None, None]
                     
    
    def computeCandidate(self, candidates):
        for i in candidates:
            print '--> ', i
            
        max_cov = 0.0025
        
        cov = var(candidates, axis=0)
        print 'cov: ', cov
        
        if cov[0] < max_cov and cov[1] < max_cov and cov[2] < max_cov:
            return [mean(candidates, axis=0), cov, True]
        else:
            return[candidates[0], cov, False]
            
            
    
    def applyUpdate(self, z, r, h, v, d):
        
        # if d == 6.0:
            # print ' =========== Apply update: ===================\n \n \n \n'       
            # print 'z: ', z
            # print 'd: ', d
            # print '_P_: ', self._P_
            # print 'P: ', diag(self.P)
        
        distance = self.mahalanobisDistance(z, h, r)
        # print ' --> distance: ', distance
        if self.filter_updates < 100 or distance < d: 
            self.lock.acquire()
            innovation = z - dot(h, self._x_).T
            temp_K = dot(dot(h, self._P_), h.T) + dot(dot(v, r), v.T)
            inv_S = squeeze(asarray(matrix(temp_K).I))
            K = dot(dot(self._P_, h.T), inv_S)
            self.x = self._x_ + dot(K, innovation)
            self.P = dot((eye(6 + 3*self.number_of_landmarks)-dot(K, h)), 
                         self._P_)
                         
            # Check covariabce matrix integrity
            p_diag = self.P.diagonal()
            for i in range(len(p_diag)):
                if p_diag[i] <= 0.0:
                    self.P[i,i] = 0.01
                    rospy.logfatal('%s, NEGATIVE VALUES IN THE DIAGONAL OF P!', self.name)
            # ---------------------------------
            
            self.filter_updates = self.filter_updates + 1
            self.lock.release()
        else:
            rospy.loginfo('%s, Invalid update. Mahalanobis distance = %s > %s',
                          self.name, distance, d)
      
      
    def makePrediction(self, now):
        if self.is_ekf_init and self.is_imu_init:
            # Build input array
            orientation = euler_from_quaternion([self.imu.orientation.x,
                                                 self.imu.orientation.y,
                                                 self.imu.orientation.z,
                                                 self.imu.orientation.w])
            u = list(orientation)

            # Take current time/period
            t = (now - self.last_prediction).to_sec()
            
            self.last_prediction = now
            if t > 0.0 and t < 1.0:
                # Make a prediction                
                self.lock.acquire()
                self.prediction(u, t)
                self.lock.release()
                return True
            # All the messages receiverd during the last 50 ms use the 
            # same prediction
            elif t > -0.1: 
                self._x_ = self.x
                self._P_ = self.P
                return True
            else:
                rospy.logerr('Invalid time: %s', t)
                return False
        else:
            return False
        
        
    def publishData(self, now):
        if self.is_ekf_init:
            self.lock.acquire()
        
            # Create header
            self.odom.header.stamp = now
            self.odom.header.frame_id = self.world_frame_name
            self.odom.child_frame_id = '' # self.world_frame_name
            
            # Pose
            self.odom.pose.pose.position.x = self.x[0]
            self.odom.pose.pose.position.y = self.x[1]
            self.odom.pose.pose.position.z = self.x[2]
            
            # Pose covariance
            p = self.P[0:3,0:3].tolist()
            self.odom.pose.covariance[0:3] = p[0]
            self.odom.pose.covariance[6:9] = p[1]
            self.odom.pose.covariance[12:15] = p[2]
            self.odom.pose.covariance[21:24] = self.imu.orientation_covariance[0:3] 
            self.odom.pose.covariance[27:30] = self.imu.orientation_covariance[3:6]
            self.odom.pose.covariance[33:36] = self.imu.orientation_covariance[6:9]
        
            # Twist
            self.odom.twist.twist.linear.x = self.x[3]
            self.odom.twist.twist.linear.y = self.x[4]
            self.odom.twist.twist.linear.z = self.x[5]
            
            # Twist covariance
            p = self.P[3:6,3:6].tolist()
            self.odom.twist.covariance[0:3] = p[0]
            self.odom.twist.covariance[6:9] = p[1]
            self.odom.twist.covariance[12:15] = p[2]
            self.odom.twist.covariance[21:24] = self.imu.angular_velocity_covariance[0:3] 
            self.odom.twist.covariance[27:30] = self.imu.angular_velocity_covariance[3:6]
            self.odom.twist.covariance[33:36] = self.imu.angular_velocity_covariance[6:9]
        
            # Publish Localization
            self.pub_odom.publish(self.odom)
            
            # Publish TF
            br = tf.TransformBroadcaster()
            br.sendTransform((self.x[0], self.x[1], self.x[2]),                   
                             (self.odom.pose.pose.orientation.x, 
                              self.odom.pose.pose.orientation.y, 
                              self.odom.pose.pose.orientation.z, 
                              self.odom.pose.pose.orientation.w),
                             self.odom.header.stamp,
                             self.robot_frame_name,
                             self.world_frame_name)
            
            # Publish covariance marker
            marker = Marker()
            marker.header.stamp = self.odom.header.stamp
            marker.header.frame_id = self.world_frame_name
            
            marker.ns = self.robot_frame_name + '_cov'
            marker.id = 0
            marker.type = 2 # SPHERE
            marker.action = 0 # Add/Modify an object
            marker.pose.position.x = self.odom.pose.pose.position.x
            marker.pose.position.y = self.odom.pose.pose.position.y
            marker.pose.position.z = self.odom.pose.pose.position.z
            marker.scale.x = math.sqrt(self.odom.pose.covariance[0])
            marker.scale.y = math.sqrt(self.odom.pose.covariance[7])
            marker.scale.z = math.sqrt(self.odom.pose.covariance[14])
            marker.color.r = 0.7
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 0.7
            marker.lifetime = rospy.Duration(2.0)
            marker.frame_locked = False
            self.covariance_marker.publish(marker)
                 
            # Publish Mapping
            if self.number_of_landmarks > 0:
                map = Map()
                map.header.stamp = rospy.Time.now()
                map.header.frame_id = self.world_frame_name
                marker_array = MarkerArray()
                
                for i in range(self.number_of_landmarks):
                    # Create Map
                    landmark = Landmark()
                    landmark.position.x = self.x[6 + i*3]
                    landmark.position.y = self.x[7 + i*3]
                    landmark.position.z = self.x[8 + i*3]
                    landmark.landmark_id = self.landmark_values[i]
                    landmark.last_update = self.landmark_last_update[i]
                    p = self.P[6 + i*3:9 + i*3, 6 + i*3:9 + i*3].tolist()
                    landmark.position_covariance[0:3] = p[0]
                    landmark.position_covariance[3:6] = p[1]
                    landmark.position_covariance[6:9] = p[2]
                    
                    map.landmark.append(landmark)
                
                    # Create Markers
                    marker = Marker()
                    marker.header.frame_id = self.world_frame_name
                    marker.header.stamp = self.landmark_last_update[i]
                    marker.ns = self.landmark_values[i]
                    marker.id = 0
                    marker.type = 1 # CUBE
                    marker.action = 0 # Add/Modify an object
                    marker.pose.position.x = landmark.position.x
                    marker.pose.position.y = landmark.position.y
                    marker.pose.position.z = landmark.position.z
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    marker.color.r = 0.1
                    marker.color.g = 0.1
                    marker.color.b = 1.0
                    marker.color.a = 0.6
                    marker.lifetime = rospy.Duration(2.0)
                    marker.frame_locked = False
                    marker_array.markers.append(marker)
                
                self.pub_map.publish(map)
                self.pub_landmarks.publish(marker_array)
                
            self.lock.release()
            
        
    def f(self, x_1, u, t):
        """ The model takes as state 3D position (x, y, z) and linear 
            velocity (vx, vy, vz). The input is the orientation 
            (roll, pitch yaw) and the linear accelerations (ax, ay, az). """
            
        roll = u[0]
        pitch = u[1]
        yaw = u[2]
        x1 = x_1[0]
        y1 = x_1[1]
        z1 = x_1[2]
        vx1 = x_1[3]
        vy1 = x_1[4]
        vz1 = x_1[5]
        
        x = list(x_1)
        
        # Compute Prediction Model with constant velocity
        x[0] = x1 + cos(pitch)*cos(yaw)*(vx1*t) - cos(roll)*sin(yaw)*(vy1*t) + sin(roll)*sin(pitch)*cos(yaw)*(vy1*t) + sin(roll)*sin(yaw)*(vz1*t) + cos(roll)*sin(pitch)*cos(yaw)*(vz1*t)
        x[1] = y1 + cos(pitch)*sin(yaw)*(vx1*t) + cos(roll)*cos(yaw)*(vy1*t) + sin(roll)*sin(pitch)*sin(yaw)*(vy1*t) - sin(roll)*cos(yaw)*(vz1*t) + cos(roll)*sin(pitch)*sin(yaw)*(vz1*t)
        x[2] = z1 - sin(pitch)*(vx1*t) + sin(roll)*cos(pitch)*(vy1*t) + cos(roll)*cos(pitch)*(vz1*t)
        x[3] = vx1
        x[4] = vy1
        x[5] = vz1
        return x


    def computeA(self, u, t):
        """ A is the jacobian matrix of f(x) """
        
        roll = u[0]
        pitch = u[1]
        yaw = u[2]
        
        A = eye(6 + 3*self.number_of_landmarks)
        A[0,3] = cos(pitch)*cos(yaw)*t
        A[0,4] = -cos(roll)*sin(yaw)*t + sin(roll)*sin(pitch)*cos(yaw)*t
        A[0,5] = sin(roll)*sin(yaw)*t + cos(roll)*sin(pitch)*cos(yaw)*t
        
        A[1,3] = cos(pitch)*sin(yaw)*t
        A[1,4] = cos(roll)*cos(yaw)*t + sin(roll)*sin(pitch)*sin(yaw)*t
        A[1,5] = -sin(roll)*cos(yaw)*t + cos(roll)*sin(pitch)*sin(yaw)*t
        
        A[2,3] = -sin(pitch)*t
        A[2,4] = sin(roll)*cos(pitch)*t
        A[2,5] = cos(roll)*cos(pitch)*t
        
        return A
        
    
    def computeW(self, u, t):
        """ The noise in the system is a term added to the acceleration:
            e.g. x[0] = x1 + cos(pitch)*cos(yaw)*(vx1*t +  Eax) *t^2/2)-..  
            then, dEax/dt of x[0] = cos(pitch)*cos(yaw)*t^2/2 """
                 
        roll = u[0]
        pitch = u[1]
        yaw = u[2]
        t2 = (t**2)/2
        
        W = zeros((6 + 3*self.number_of_landmarks, 3))
        W[0,0] = cos(pitch)*cos(yaw)*t2
        W[0,1] = -cos(roll)*sin(yaw)*t2 + sin(roll)*sin(pitch)*cos(yaw)*t2
        W[0,2] = sin(roll)*sin(yaw)*t2 + cos(roll)*sin(pitch)*cos(yaw)*t2
        
        W[1,0] = cos(pitch)*sin(yaw)*t2
        W[1,1] = cos(roll)*cos(yaw)*t2 + sin(roll)*sin(pitch)*sin(yaw)*t2
        W[1,2] = -sin(roll)*cos(yaw)*t2 + cos(roll)*sin(pitch)*sin(yaw)*t2
        
        W[2,0] = -sin(pitch)*t2
        W[2,1] = sin(roll)*cos(pitch)*t2
        W[2,2] = cos(roll)*cos(pitch)*t2
        
        W[3,0] = t
        W[4,1] = t
        W[5,2] = t
        return W
    
    
    def computeQ(self, q_var):
        Q = eye(3)
        return Q*q_var
    
    
    def createPoseMeasures(self, pose, covariance):
        # TODO: WARNING! This functions only uses the diagonal values 
        # in the covariance matrix. We can lose information!
        j = 0 # counter
        m = 0 # counter 2
        
        # Init values
        z = []
        r_diag = []
        h = zeros((3, 6 + 3*self.number_of_landmarks))
        
        for i in range(3):
            if covariance[i, i] < 999:
                z.append(pose[j])
                r_diag.append(covariance[i, i])
                h[m, j] = 1.0
                m = m + 1
            else:
                h = delete(h,j,0)
            j = j + 1
        r = eye(len(z))*r_diag
        v = eye(len(z))
        return z, r, h, v
    
    
    def velocityH(self):
        velocity_h = zeros((3, 6 + 3*self.number_of_landmarks))
        velocity_h[0, 3] = 1.0
        velocity_h[1, 4] = 1.0
        velocity_h[2, 5] = 1.0
        return velocity_h
    
 
    def landmarkH2(self, landmark_id, rot):
        # print 'landmark_id: ', landmark_id
        landmark_h = zeros((3, 6 + 3*self.number_of_landmarks))
        if landmark_id < self.number_of_landmarks:
            landmark_h[0, 0:3] = -1.0*rot[0,0:3]
            landmark_h[1, 0:3] = -1.0*rot[1,0:3]
            landmark_h[2, 0:3] = -1.0*rot[2,0:3]
            landmark_h[0, 6 + landmark_id*3:9 + landmark_id*3] = rot[0,0:3]
            landmark_h[1, 6 + landmark_id*3:9 + landmark_id*3] = rot[1,0:3]
            landmark_h[2, 6 + landmark_id*3:9 + landmark_id*3] = rot[2,0:3]
        else:
            rospy.loginfo('%s, Invalid landmark update. Out of range.', 
                          self.name)
        return landmark_h
            
            
    def landmarkH(self, landmark_id):
        # print 'landmark_id: ', landmark_id
        
        landmark_h = zeros((3, 6 + 3*self.number_of_landmarks))
        if landmark_id < self.number_of_landmarks:
            #landmark_h[0, 0] = 1.0
            #landmark_h[1, 1] = 1.0
            #landmark_h[2, 2] = 1.0
            landmark_h[0, 6 + landmark_id*3] = 1.0
            landmark_h[1, 7 + landmark_id*3] = 1.0
            landmark_h[2, 8 + landmark_id*3] = 1.0
        else:
            rospy.loginfo('%s, Invalid landmark update. Out of range.', 
                          self.name)
        return landmark_h
            
    
    def takeCovariance(self, covariance):
        c = array(covariance).reshape(6, 6)
        return c[0:3, 0:3]

        
    def initEkf(self, position):
        self.lock.acquire()        
        rospy.loginfo("%s, Init POSE_EKF_SLAM", self.name)
               
        # Init state vector
        self.x = zeros(6)
        self.x[0] = position.x
        self.x[1] = position.y
        self.x[2] = position.z
        self._x_ = array(self.x)   
        
        # Init P 
        self.P = eye((6))
        self.P[0, 0] = self.p_var[0]
        self.P[1, 1] = self.p_var[1]
        self.P[2, 2] = self.p_var[2]
        self.P[3, 3] = self.p_var[3]
        self.P[4, 4] = self.p_var[4]
        self.P[5, 5] = self.p_var[5]        
        self._P_ = array(self.P)

        # Init landmarks, TFs and others        
        self.number_of_landmarks = 0
        self.mapped_lamdmarks = {}
        self.candidate_landmarks = {}
        self.landmark_values = {}
        self.landmark_last_update = {}
        self.filter_updates = 0
        self.landmark_update_topics = []
        self.tf_cache = dict()
        
        rospy.loginfo('Init x: %s', self.x)
        rospy.loginfo('Init P: %s', self.P)
        self.lock.release()

    
    def addLandmark(self, x, y, z, measured_covariance):
        self.lock.acquire()
        print 'add landmark at position: ', x, y, z
        print 'with covariance: ', measured_covariance
        print 'vehicle uncertainty P: ', self.P
        
        # Increase P matrix
        new_P = eye(len(self.x) + 3)
        new_P[0:len(self.x), 0:len(self.x)] = self.P
        
        new_P[6 + 3*self.number_of_landmarks: 9 + 3*self.number_of_landmarks, 0:3] = self.P[0:3, 0:3]
        new_P[0:3, 6 + 3*self.number_of_landmarks: 9 + 3*self.number_of_landmarks] = self.P[0:3, 0:3]
        
        angle = tf.transformations.euler_from_quaternion(
            [self.imu.orientation.x, self.imu.orientation.y, 
             self.imu.orientation.z, self.imu.orientation.w])
             
        R = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
        rot_m = matrix([R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]]).reshape(3,3)
        
        m = dot(dot(rot_m, measured_covariance), rot_m.T) + self.P[0:3, 0:3]
        
        new_P[6 + 3*self.number_of_landmarks: 9 + 3*self.number_of_landmarks, 6 + 3*self.number_of_landmarks: 9 + 3*self.number_of_landmarks] = m
        print 'Old P matrix : ', diag(self.P)        
        print 'New P matrix : ', diag(new_P)
        
        self.P = new_P
        
       # Increase state vector
        new_x = zeros(len(self.x) + 3)
        new_x[0:len(self.x)] = self.x
        new_x[-3] = x
        new_x[-2] = y
        new_x[-1] = z
        self.x = new_x

        # Increase number of landmarks registered
        self.number_of_landmarks = self.number_of_landmarks + 1
        
        self.lock.release()
        
        
    def prediction(self, u, t):
        A = self.computeA(u, t)
        W = self.computeW(u, t)
        self._x_ = self.f(self.x, u, t)
        self._P_ = dot(dot(A, self.P), A.T) + dot(dot(W, self.Q), W.T)
        # print diag(dot(dot(W, self.Q), W.T))
        # print random.multivariate_normal(zeros(6), dot(dot(W, self.Q), W.T))
            
        # print diag(self._P_) - diag(self.P) < 0

    def updatePrediction(self): 
        self.lock.acquire()
        self.x = self._x_
        self.P = self._P_ 
        self.lock.release()
        

    def getStateVector(self):
        return self.x
        
        
    def getConfig(self):
        if rospy.has_param('pose_ekf_slam/world_frame_name') :
            self.world_frame_name = rospy.get_param(
                                    'pose_ekf_slam/world_frame_name')
        else:
            rospy.logfatal('pose_ekf_slam/world_frame_name')

        if rospy.has_param('pose_ekf_slam/robot_frame_name') :
            self.robot_frame_name = rospy.get_param(
                                    'pose_ekf_slam/robot_frame_name')
        else:
            rospy.logfatal('pose_ekf_slam/robot_frame_name')
            
            
    def mahalanobisDistance(self, z, h, r):        
#        print 'x: ', self.x
#        print 'P: ', self.P
#        print 'h: ', h
#        print 'z: ', z
#        print 'r: ', r
        
        v = z - dot(h, self.x)
        # print 'v: ', v
        S = matrix(dot(dot(h, self.P), h.T) + r)
        # print 'S: ', S
        d = dot(dot(v.T, S.I), v)
        # print 'Mahalanobis distance: ', d
        return sqrt(d[0,0])
    
    
if __name__ == '__main__':
    try:
        # Init node
        rospy.init_node('pose_ekf_slam')
        pose_3d_ekf = PoseEkfSlam(rospy.get_name(), 
                                  [0.01, 0.01, 0.5, 0.02, 0.02, 0.02], 
                                  [0.05, 0.05, 0.05])
                                  
        # Initialize vehicle pose (should be done by the "navigator" node) 
#        position = Point()
#        position.x = 0.
#        position.y = 0.
#        position.z = 0.
#        pose_3d_ekf.initEkf(position)
        
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass
