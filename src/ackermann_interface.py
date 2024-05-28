#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64
import tf2_ros
import tf_conversions
from math import sin, cos, tan, fabs

# Global variables
cmd_vel_out = Twist()
odom_out = Odometry()
odom = Odometry()
tf_msg = TFMessage()
yaw_from_imu = 0.0

min_x = 0.1  # khusus ackermann, robot ga boleh sampe berhenti
min_rot = 0.7  # radian (maksimal rotasi yang diizinkan)

last_cmd_vel_time = rospy.Time()
front_steer_angle = 0.0
rear_wheel_speed = 0.0
rear_left_wheel_speed = 0.0
rear_right_wheel_speed = 0.0

def cmd_vel_callback(cmd_vel_in):
    global cmd_vel_out
    rot = cmd_vel_in.angular.z
    x = cmd_vel_in.linear.x

    cmd_vel_out.angular.z = rot / 2

    if (fabs(rot) > min_rot) and x < min_x:
        cmd_vel_out.linear.x = min_x
    else:
        cmd_vel_out.linear.x = x

    pub_cmd_vel.publish(cmd_vel_out)

def jointStateCallback(joint_state_in):
    global front_steer_angle, rear_wheel_speed, rear_left_wheel_speed, rear_right_wheel_speed
    wheel_rad = 0.1
    
    indices = {name: idx for idx, name in enumerate(joint_state_in.name)}
    
    front_steer_angle = joint_state_in.position[indices.get("front_steer_joint", -1)]
    rear_wheel_speed = joint_state_in.velocity[indices.get("rear_wheel_joint", -1)] * wheel_rad
    rear_left_wheel_speed = joint_state_in.velocity[indices.get("rear_left_wheel_joint", -1)] * wheel_rad
    rear_right_wheel_speed = joint_state_in.velocity[indices.get("rear_right_wheel_joint", -1)] * wheel_rad

def imuCallback(imu_msg):
    global yaw_from_imu
    imu_quaternion = tf_conversions.transformations.quaternion_from_euler(
        imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z
    )
    roll, pitch, yaw = tf_conversions.transformations.euler_from_quaternion(imu_quaternion)
    
    yaw_from_imu = yaw
    yaw_msg = Float64()
    yaw_msg.data = yaw
    pub_yaw.publish(yaw_msg)

def odom_callback(odom_in):
    global odom_out, last_cmd_vel_time, tf_msg, yaw_from_imu

    if odom_out.header.stamp == rospy.Time(0):
        odom_out = odom_in
        last_cmd_vel_time = odom_in.header.stamp
        return

    dt = (odom_in.header.stamp - last_cmd_vel_time).to_sec()
    last_cmd_vel_time = odom_in.header.stamp

    linear_vel = (rear_left_wheel_speed + rear_right_wheel_speed) / 2.0
    steering_angle = front_steer_angle
    correction_factor = 0.80
    
    cx = linear_vel * sin(odom_out.pose.pose.orientation.z) * sin(steering_angle) * correction_factor * dt
    cy = linear_vel * cos(odom_out.pose.pose.orientation.z) * sin(steering_angle) * 0.5 * dt

    delta_x = linear_vel * cos(odom_out.pose.pose.orientation.z) * dt
    delta_y = linear_vel * sin(odom_out.pose.pose.orientation.z) * dt

    delta_x -= cx
    delta_y += cy

    odom_out.pose.pose.position.x += delta_x
    odom_out.pose.pose.position.y += delta_y
    odom_out.pose.pose.orientation.z = yaw_from_imu

    odom.pose.pose.position.x = odom_out.pose.pose.position.x
    odom.pose.pose.position.y = odom_out.pose.pose.position.y
    odom.pose.pose.orientation.z = sin(odom_out.pose.pose.orientation.z / 2.0)
    odom.pose.pose.orientation.w = cos(odom_out.pose.pose.orientation.z / 2.0)

    tf_msg.transforms = [tf2_ros.TransformStamped()]
    tf_msg.transforms[0].header.stamp = rospy.Time.now()
    tf_msg.transforms[0].header.frame_id = "odom"
    tf_msg.transforms[0].child_frame_id = "base_footprint"
    tf_msg.transforms[0].transform.translation.x = odom_out.pose.pose.position.x
    tf_msg.transforms[0].transform.translation.y = odom_out.pose.pose.position.y
    tf_msg.transforms[0].transform.rotation.z = sin(odom_out.pose.pose.orientation.z / 2.0)
    tf_msg.transforms[0].transform.rotation.w = cos(odom_out.pose.pose.orientation.z / 2.0)

    pub_tf.publish(tf_msg)
    pub_odom.publish(odom)

if __name__ == "__main__":
    rospy.init_node("converter", anonymous=True)

    sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    sub_odom = rospy.Subscriber("/ackermann_steering_controller/odom", Odometry, odom_callback)
    sub_joint_state = rospy.Subscriber("/joint_states", JointState, jointStateCallback)
    sub_imu = rospy.Subscriber("/imu", Imu, imuCallback)

    pub_cmd_vel = rospy.Publisher("/ackermann_steering_controller/cmd_vel", Twist, queue_size=100)
    pub_odom = rospy.Publisher("/odom", Odometry, queue_size=100)
    pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=100)
    pub_yaw = rospy.Publisher("/imu_yaw", Float64, queue_size=100)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        rospy.spin()

