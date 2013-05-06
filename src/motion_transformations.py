#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('pose_ekf_slam')
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# More imports
from numpy import array, cross, dot
import PyKDL
import math

def position(measured_pose, vehicle_orientation, trans_sensor_from_vehicle):
    p = PyKDL.Vector(measured_pose.x, measured_pose.y, measured_pose.z)
    angle = tf.transformations.euler_from_quaternion([vehicle_orientation.x,
                                                      vehicle_orientation.y,
                                                      vehicle_orientation.z,
                                                      vehicle_orientation.w])
    O = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
    t = PyKDL.Vector(trans_sensor_from_vehicle[0], 
                     trans_sensor_from_vehicle[1], 
                     trans_sensor_from_vehicle[2])
    return p - (O * t)


def landmarkPosition(landmark_pose, rot_tf, trans_tf):
    p = PyKDL.Vector(landmark_pose.x, landmark_pose.y, landmark_pose.z)
    angle = tf.transformations.euler_from_quaternion([rot_tf[0],
                                                      rot_tf[1],
                                                      rot_tf[2],
                                                      rot_tf[3]])
    O = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
    t = PyKDL.Vector(trans_tf[0], 
                     trans_tf[1], 
                     trans_tf[2])
    f = PyKDL.Frame(O, t)

    return f*p
    

def orientation(measured_orientation, rot_sensor_from_vehicle):
    angle = euler_from_quaternion([measured_orientation.x, 
                                   measured_orientation.y, 
                                   measured_orientation.z, 
                                   measured_orientation.w])
    orientation_data =  PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
    
    rot_angle = euler_from_quaternion([rot_sensor_from_vehicle[0],
                                       rot_sensor_from_vehicle[1],
                                       rot_sensor_from_vehicle[2],
                                       rot_sensor_from_vehicle[3]])
    R = PyKDL.Rotation.RPY(rot_angle[0], rot_angle[1], rot_angle[2])
    # R.Inverse() == transpose R
    orientation_data = orientation_data * R.Inverse() 
    
    new_angle = orientation_data.GetRPY()
    
    return quaternion_from_euler(new_angle[0], new_angle[1], new_angle[2])


def linearVelocity(measured_linear_velocity, 
                   vehicle_angular_velocity, 
                   trans_sensor_from_vehicle, 
                   rot_sensor_from_vehicle):

    v = PyKDL.Vector(measured_linear_velocity.x, 
                     measured_linear_velocity.y, 
                     measured_linear_velocity.z)
                     
    angle = euler_from_quaternion([rot_sensor_from_vehicle[0],
                                   rot_sensor_from_vehicle[1],
                                   rot_sensor_from_vehicle[2],
                                   rot_sensor_from_vehicle[3]])
                                 
    R = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
    v_r = R * v
            
    #Once the velocity is rotates to the vehicle frame it is necessary to add
    #the linear velocity that appears when combining vehicle's angular velocity
    #and displacement between sensor and vehicle's frame
    w = array([vehicle_angular_velocity.x,
               vehicle_angular_velocity.y,
               vehicle_angular_velocity.z])

    return array([v_r[0], v_r[1], v_r[2]]) - cross(w, array(trans_sensor_from_vehicle))


def angularVelocity(measured_angular_velocity, rot_sensor_from_vehicle):
    angle = tf.transformations.euler_from_quaternion([rot_sensor_from_vehicle[0],
                                                      rot_sensor_from_vehicle[1],
                                                      rot_sensor_from_vehicle[2],
                                                      rot_sensor_from_vehicle[3]])
    R = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
    w = PyKDL.Vector(measured_angular_velocity.x, 
                     measured_angular_velocity.y, 
                     measured_angular_velocity.z)
    return R * w 


def positionCovariance(m_cov, vh_or_cov, vh_or, trans):
    # print 'vh_or_cov: ', vh_or_cov
    angle = euler_from_quaternion([vh_or.x, 
                                   vh_or.y, 
                                   vh_or.z, 
                                   vh_or.w])
    r = angle[0]
    p = angle[1]
    y = angle[2]
    # print 'r, p, y: ', r, p, y
    
    dx = trans[0]
    dy = trans[1]
    dz = trans[2]
    # print 'dx, dy, dz: ', dx, dy, dz

    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cy = math.cos(y)
    sy = math.sin(y)

#    RPY rotation matrix
#    rpy = array([cy*cp,  cy*sp*sr - sy*cr,   cy*sp*cr + sy*sr,
#                 sy*cp,  sy*sp*sr + cy*cr,   sy*sp*cr - cy*sr,
#                 -sp,    cp*sr,              cp*cr]).reshape(3,3)

#    Z = measured_pose - RPY*trans
#    Zx = Mx - cy*cp*dx + (cy*sp*sr - sy*cr)*dy + (cy*sp*cr + sy*sr)*dz
#    Zy = My - sy*cp*dx + (sy*sp*sr + cy*cr)*dy + (sy*sp*cr - cy*sr)*dz
#    Zz = Mz - (-sp*dx) + cp*sr*dy + cp*cr*dz
    
    d_r_x = (cy*sp*cr + sy*sr)*dy + (-cy*sp*sr + sy*cr)*dz
    d_r_y = (sy*sp*cr - cy*sr)*dy + (-sy*sp*sr - cy*cr)*dz
    d_r_z = -cp*cr*dy + cp*sr*dz
    d_p_x = -cy*sp*dx + cy*cp*sr*dy + cy*cp*cr*dz
    d_p_y = -sy*sp*dx + sy*cp*sr*dy + sy*cp*cr*dz
    d_p_z = -cp*dx - sp*sr*dy - sp*cr*dz
    d_y_x = -sy*cp*dx + (-sy*sp*sr - cy*cr)*dy + (-sy*sp*cr + cy*sr)*dz
    d_y_y = cy*cp*dx + (cy*sp*sr - sy*cr)*dy + (cy*sp*cr + sy*sr)*dz
    d_y_z = 0
    
    rpy_cov = array([d_r_x, d_p_x, d_y_x,
                     d_r_y, d_p_y, d_y_y,
                     d_r_z, d_p_z, d_y_z]).reshape(3,3)
    
    return m_cov + dot(rpy_cov, dot(vh_or_cov, rpy_cov.T))


def linearVelocityCov(v_cov, vh_w_cov, trans, rot):
    angle = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
    r = angle[0]
    p = angle[1]
    y = angle[2]
    tx = trans[0]
    ty = trans[1]
    tz = trans[2]    

#    Z = RPY*Measured_v - cross(w, trans)
#    Take first RPY*M_v
#    Z'vx = cy*cp*vx + (cy*sp*sr - sy*cr)*vy + (cy*sp*cr + sy*sr)*vz
#    Z'vy = sy*cp*vx + (sy*sp*sr + cy*cr)*vy + (sy*sp*cr - cy*sr)*vz
#    Z'vz = -sp*vx + cp*sr*vy + cp*cr*vz
#    This is a linera operation because the covariance is in M_v then:
    rpy = PyKDL.Rotation.RPY(r, p, y)
    rpy_v_cov = array([rpy[0,0], rpy[0,1], rpy[0,2],
                       rpy[1,0], rpy[1,1], rpy[1,2],
                       rpy[2,0], rpy[2,1], rpy[2,2]]).reshape(3,3)
    
#    Take second cross(W, T)
#    x = wy*tz - wz*ty
#    y = wx*tz - wz*tx
#    z = wx*ty - wy*tx
#    This is also linear

    d_wx_x = 0
    d_wx_y = tz
    d_wx_z = ty
    d_wy_x = tz
    d_wy_y = 0
    d_wy_z = -tx
    d_wz_x = tz
    d_wz_y = -tx
    d_wz_z = 0
    
    w_cov = array([d_wx_x, d_wy_x, d_wz_x,
                   d_wx_y, d_wy_y, d_wz_y,
                   d_wx_z, d_wy_z, d_wz_z]).reshape(3,3)
                     
    return (dot(rpy_v_cov, dot(v_cov, rpy_v_cov.T)) + 
            dot(w_cov, dot(vh_w_cov, w_cov.T)))


def angularVelocityCov(w_cov, rot):
    angle = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
    r = angle[0]
    p = angle[1]
    y = angle[2]

#    Z = RPY*W
#    Zwx = cy*cp*wx + (cy*sp*sr - sy*cr)*wy + (cy*sp*cr + sy*sr)*wz
#    Zwy = sy*cp*wx + (sy*sp*sr + cy*cr)*wy + (sy*sp*cr - cy*sr)*wz
#    Zwz = -sp*wx + cp*sr*wy + cp*cr*wz
#   As the covariance is in the W the function is linear
    
    rpy = PyKDL.Rotation.RPY(r, p, y)
    rpy_w_cov = array([rpy[0,0], rpy[0,1], rpy[0,2],
                       rpy[1,0], rpy[1,1], rpy[1,2],
                       rpy[2,0], rpy[2,1], rpy[2,2]]).reshape(3,3)

    return dot(rpy_w_cov, dot(w_cov, rpy_w_cov.T)) 
           

# TODO: To be check!
def orientationCov(orientation_cov, 
                   measured_orientation, 
                   rot_sensor_from_vehicle):
                       
    angle = euler_from_quaternion([measured_orientation.x, 
                                   measured_orientation.y, 
                                   measured_orientation.z, 
                                   measured_orientation.w])

    r = angle[0]
    p = angle[1]
    y = angle[2]
    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cy = math.cos(y)
    sy = math.sin(y)
    
    rot_angle = euler_from_quaternion([rot_sensor_from_vehicle[0],
                                       rot_sensor_from_vehicle[1],
                                       rot_sensor_from_vehicle[2],
                                       rot_sensor_from_vehicle[3]])
    rot_angle = rot_sensor_from_vehicle
    rR = rot_angle[0]
    pR = rot_angle[1]
    yR = rot_angle[2]
    crR = math.cos(rR)
    srR = math.sin(rR)
    cpR = math.cos(pR)
    spR = math.sin(pR)
    cyR = math.cos(yR)
    syR = math.sin(yR)
    
#    |cy*cp,  cy*sp*sr - sy*cr,   cy*sp*cr + sy*sr|
#    |sy*cp,  sy*sp*sr + cy*cr,   sy*sp*cr - cy*sr| *
#    |-sp,    cp*sr,              cp*cr           |
#    
#    |cyR*cpR,               syR*cpR,                -spR   |
#    |cyR*spR*srR - syR*crR, syR*spR*srR + cyR*crR,  cpR*srR| ==>
#    |cyR*spR*crR + syR*srR, cpR*srR,                cpR*crR|
#    
    a11 = cyR*cpR
    a12 = syR*cpR
    a13 = -spR
    a21 = cyR*spR*srR - syR*crR
    a22 = syR*spR*srR + cyR*crR
    a23 = cpR*srR
    a31 = cyR*spR*crR + syR*srR
    a32 = cpR*srR
    a33 = cpR*crR
#    
#    R32 = -sp*(a12) + (cp*sr)*(a22) + (cp*cr)*(a32)
#    R33 = -sp*(a13) + (cp*sr)*(a23) + (cp*cr)*(a33)
#    R21 = (sy*cp)*(a11) + (sy*sp*sr + cy*cr)*(a21) + (sy*sp*cr - cy*sr)*(a31)
#    R11 = (cy*cp)*(a11) + (cy*sp*sr - sy*cr)*(a21) + (cy*sp*cr + sy*sr)*(a31)
#    R31 = -sp*(a11) + (cp*sr)*(a21) + (cp*cr)*(a31)
#    
#    fr = atan2(R32, R33)
#    fr = atan((-sp*(a12) + (cp*sr)*(a22) + (cp*cr)*(a32))/(-sp*(a13) + (cp*sr)*(a23) + (cp*cr)*(a33)))
    fr_dr = ((a22*cp*cr - a32*cp*sr)/(a33*cp*cr - a13*sp + a23*cp*sr) - ((a23*cp*cr - a33*cp*sr)*(a32*cp*cr - a12*sp + a22*cp*sr))/(a33*cp*cr - a13*sp + a23*cp*sr)**2)/((a32*cp*cr - a12*sp + a22*cp*sr)**2/(a33*cp*cr - a13*sp + a23*cp*sr)**2 + 1)
    fr_dp = -((a12*cp + a32*cr*sp + a22*sp*sr)/(a33*cp*cr - a13*sp + a23*cp*sr) - ((a32*cp*cr - a12*sp + a22*cp*sr)*(a13*cp + a33*cr*sp + a23*sp*sr))/(a33*cp*cr - a13*sp + a23*cp*sr)**2)/((a32*cp*cr - a12*sp + a22*cp*sr)**2/(a33*cp*cr - a13*sp + a23*cp*sr)**2 + 1)
    fr_dy = 0
    
#    fy = atan2(R21, R11)
#    fy = atan(((sy*cp)*(a11) + (sy*sp*sr + cy*cr)*(a21) + (sy*sp*cr - cy*sr)*(a31))/((cy*cp)*(a11) + (cy*sp*sr - sy*cr)*(a21) + (cy*sp*cr + sy*sr)*(a31)))
    YAW = math.atan2((sy*cp)*(a11) + (sy*sp*sr + cy*cr)*(a21) + (sy*sp*cr - cy*sr)*(a31), (cy*cp)*(a11) + (cy*sp*sr - sy*cr)*(a21) + (cy*sp*cr + sy*sr)*(a31))    
    cYAW = math.cos(YAW)
    sYAW = math.sin(YAW)    
    fy_dr = -((a21*(cy*sr - cr*sp*sy) + a31*(cr*cy + sp*sr*sy))/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + ((a21*(sr*sy + cr*cy*sp) + a31*(cr*sy - cy*sp*sr))*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy)**2)/((a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy)**2/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy)**2 + 1)
    fy_dp = ((a31*cp*cr*sy - a11*sp*sy + a21*cp*sr*sy)/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) - ((a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy)*(a31*cp*cr*cy - a11*cy*sp + a21*cp*cy*sr))/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy)**2)/((a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy)**2/(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy)**2 + 1)
    fy_dy = 1

#    fp = atan2(-R31, (cos(y)*R11 + sin(y)*R21))
#    fp = atan(-(-sp*(a11) + (cp*sr)*(a21) + (cp*cr)*(a31))/(cos(YAW)*((cy*cp)*(a11) + (cy*sp*sr - sy*cr)*(a21) + (cy*sp*cr + sy*sr)*(a31)) + sin(YAW)*((sy*cp)*(a11) + (sy*sp*sr + cy*cr)*(a21) + (sy*sp*cr - cy*sr)*(a31)))
#    fp = atan(-(-sin(p)*(a11) + (cos(p)*sin(r))*(a21) + (cos(p)*cos(r))*(a31))/(cos(YAW)*((cos(y)*cos(p))*(a11) + (cos(y)*sin(p)*sin(r) - sin(y)*cos(r))*(a21) + (cos(y)*sin(p)*cos(r) + sin(y)*sin(r))*(a31)) + sin(YAW)*((sin(y)*cos(p))*(a11) + (sin(y)*sin(p)*sin(r) + cos(y)*cos(r))*(a21) + (sin(y)*sin(p)*cos(r) - cos(y)*sin(r))*(a31))))
    fp_dr = -((a21*cp*cr - a31*cp*sr)/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy)) - ((cYAW*(a21*(sr*sy + cr*cy*sp) + a31*(cr*sy - cy*sp*sr)) - sYAW*(a21*(cy*sr - cr*sp*sy) + a31*(cr*cy + sp*sr*sy)))*(a31*cp*cr - a11*sp + a21*cp*sr))/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2)/((a31*cp*cr - a11*sp + a21*cp*sr)**2/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2 + 1)
    fp_dp = ((a11*cp + a31*cr*sp + a21*sp*sr)/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy)) + ((cYAW*(a31*cp*cr*cy - a11*cy*sp + a21*cp*cy*sr) + sYAW*(a31*cp*cr*sy - a11*sp*sy + a21*cp*sr*sy))*(a31*cp*cr - a11*sp + a21*cp*sr))/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2)/((a31*cp*cr - a11*sp + a21*cp*sr)**2/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2 + 1)
    fp_dy = -((cYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy) - sYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy))*(a31*cp*cr - a11*sp + a21*cp*sr))/((cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2*((a31*cp*cr - a11*sp + a21*cp*sr)**2/(cYAW*(a31*(sr*sy + cr*cy*sp) - a21*(cr*sy - cy*sp*sr) + a11*cp*cy) + sYAW*(a21*(cr*cy + sp*sr*sy) - a31*(cy*sr - cr*sp*sy) + a11*cp*sy))**2 + 1))
   
    rpy_orientation_cov = array([fr_dr, fr_dp, fr_dy,
                       fp_dr, fp_dp, fp_dy,
                       fy_dr, fy_dp, fy_dy]).reshape(3,3)

    return dot(rpy_orientation_cov, dot(orientation_cov, rpy_orientation_cov.T)) 
