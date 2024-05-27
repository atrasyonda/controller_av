#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "tf2_msgs/TFMessage.h"
#include "sensor_msgs/JointState.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64.h"  // Pesan untuk menerbitkan data posisi yaw
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"




ros::Subscriber sub_cmd_vel;
ros::Subscriber sub_odom;
ros::Subscriber sub_joint_state;
ros::Subscriber sub_imu;
ros::Publisher pub_cmd_vel;
ros::Publisher pub_odom;
ros::Publisher pub_tf;
ros::Publisher pub_yaw;


geometry_msgs::Twist cmd_vel_out;
nav_msgs::Odometry odom_out;
nav_msgs::Odometry odom;
tf2_msgs::TFMessage tf_msg;

double min_x = 0.1; //khusus ackermann, robot ga boleh sampe berhenti
double min_rot = 0.7; //radian (maksudnya maksimal rotasi yang diizinkan)


void cmd_vel_callback(const geometry_msgs::Twist &cmd_vel_in)
{
    // cmd_vel_out = cmd_vel_in;
    double rot = cmd_vel_in.angular.z;
    double x = cmd_vel_in.linear.x;

    cmd_vel_out.angular.z = rot/2;
    // cmd_vel_out.angular.z = rot;

    // // ini bukan rumus konversi ackermann gak se :3
    if( (fabs(rot) > min_rot) && x < min_x)
    {
        cmd_vel_out.linear.x = min_x;
    }
    else
    {
        cmd_vel_out.linear.x = x;
    }

    pub_cmd_vel.publish(cmd_vel_out);
}

ros::Time last_cmd_vel_time;
double front_steer_angle;
double rear_wheel_speed;
double rear_left_wheel_speed;
double rear_right_wheel_speed;


void jointStateCallback(const sensor_msgs::JointState::ConstPtr& joint_state_in)
{
    // Find the indices of the relevant joints in the 'name' array
    int rear_wheel_index = -1;
    int front_steer_index = -1;
    int rear_left_wheel_index = -1;
    int rear_right_wheel_index = -1;

    for (size_t i = 0; i < joint_state_in->name.size(); ++i)
    {
        if (joint_state_in->name[i] == "rear_wheel_joint")
            rear_wheel_index = i;
        else if (joint_state_in->name[i] == "front_steer_joint")
            front_steer_index = i;
        else if (joint_state_in->name[i] == "rear_left_wheel_joint")
            rear_left_wheel_index = i;
        else if (joint_state_in->name[i] == "rear_right_wheel_joint")
            rear_right_wheel_index = i;
            
    }

    double wheel_rad = 0.1;

   front_steer_angle = joint_state_in->position[front_steer_index] ;
   rear_wheel_speed = joint_state_in->velocity[rear_wheel_index] * wheel_rad;
   rear_left_wheel_speed = joint_state_in->velocity[rear_left_wheel_index] * wheel_rad;
   rear_right_wheel_speed = joint_state_in->velocity[rear_right_wheel_index] * wheel_rad;

//    ROS_INFO("steer_ang: %f, speed: %f", front_steer_angle, rear_wheel_speed);
//  ROS_INFO("l_speed: %f, r_speed: %f", rear_left_wheel_speed, rear_right_wheel_speed);
}

double yaw_from_imu= 0.0;  // Inisialisasi nilai awal



void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
    // Mengonversi pesan orientasi menjadi tipe data Quaternion
    tf2::Quaternion imu_quaternion;
    tf2::convert(imu_msg->orientation, imu_quaternion);

    // Mendapatkan nilai yaw dari quaternion
    double roll, pitch, yaw;
    tf2::Matrix3x3(imu_quaternion).getRPY(roll, pitch, yaw);

    // Menerbitkan data posisi yaw ke topik yang diinginkan
    std_msgs::Float64 yaw_msg;
    yaw_msg.data = yaw;
    yaw_from_imu = yaw;
    pub_yaw.publish(yaw_msg);
}


void odom_callback(const nav_msgs::Odometry &odom_in)
{
    // // Copy the received odometry message to the output message
    odom = odom_in;
    // // Publish the modified odometry message

    // pub_odom.publish(odom_out);
    
    tf_msg.transforms.resize(1);

    if (odom_out.header.stamp == ros::Time(0))
    {
        odom_out = odom_in;
        last_cmd_vel_time = odom_in.header.stamp;
        return;
    }

    // double wheel_radius = 0.1;
    // double wheel_separation_w = 0.4;

    // Update the odometry based on velocity commands
    double dt = (odom_in.header.stamp - last_cmd_vel_time).toSec();  // Calculate time since the last velocity command
    last_cmd_vel_time = odom_in.header.stamp;  // Update the last command time

    // ROS_INFO("Delta Time: %f", dt);
    // double linear_vel =  rear_wheel_speed ;
    // double linear_vel =  rear_wheel_speed + (rear_wheel_speed*sin(fabs(front_steer_angle)) * 0.3); //ok
    double linear_vel = (rear_left_wheel_speed+rear_right_wheel_speed)/2.0;
    double steering_angle = front_steer_angle;
    // double angular_vel =  linear_vel * tan(steering_angle) / 0.4;

    double correction_factor = 0.80;  // You may need to adjust this factor based on experimentation
    double cx = linear_vel * sin(odom_out.pose.pose.orientation.z) * sin(steering_angle) * correction_factor * dt;
    double cy = linear_vel * cos(odom_out.pose.pose.orientation.z) * sin(steering_angle) * 0.5 * dt;
    // double cy = 0;

    // Use an Ackermann steering model to calculate the incremental position
    // double delta_x = (linear_vel * cos(odom_out.pose.pose.orientation.z) +  linear_vel * cos(odom_out.pose.pose.orientation.z + steering_angle)*1) * dt;
    // double delta_y = (linear_vel * sin(odom_out.pose.pose.orientation.z) +  linear_vel * cos(odom_out.pose.pose.orientation.z + steering_angle)*1) * dt;
    double delta_x = linear_vel * cos(odom_out.pose.pose.orientation.z)  * dt;
    double delta_y = linear_vel * sin(odom_out.pose.pose.orientation.z)  * dt;
    // ROS_INFO("delta_x: %f, delta_y: %f", delta_x, delta_y);

    delta_x -= cx;
    delta_y += cy;



    ROS_INFO("cx: %f, cy: %f", cx, cy);



    // Update the position in the odometry message
    odom_out.pose.pose.position.x += delta_x;
    odom_out.pose.pose.position.y += delta_y;
    odom_out.pose.pose.orientation.z = yaw_from_imu;

    odom.pose.pose.position.x = odom_out.pose.pose.position.x ;
    odom.pose.pose.position.y = odom_out.pose.pose.position.y;
    odom.pose.pose.orientation.z = sin(odom_out.pose.pose.orientation.z/2.0);
    odom.pose.pose.orientation.w = cos(odom_out.pose.pose.orientation.z/2.0);

    // Update the transformation message
    tf_msg.transforms[0].header.stamp = ros::Time::now();
    tf_msg.transforms[0].header.frame_id = "odom";
    tf_msg.transforms[0].child_frame_id = "base_footprint";
    tf_msg.transforms[0].transform.translation.x = odom_out.pose.pose.position.x;
    tf_msg.transforms[0].transform.translation.y = odom_out.pose.pose.position.y;
    // tf_msg.transforms[0].transform.rotation = odom_out.pose.pose.orientation;
    tf_msg.transforms[0].transform.rotation.z = sin(odom_out.pose.pose.orientation.z/2.0);
    tf_msg.transforms[0].transform.rotation.w = cos(odom_out.pose.pose.orientation.z/2.0);


    pub_tf.publish(tf_msg);
    pub_odom.publish(odom);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "converter");
    ros::NodeHandle n;

    sub_cmd_vel = n.subscribe("/cmd_vel", 100, cmd_vel_callback);
    sub_odom = n.subscribe("/ackermann_steering_controller/odom", 100, odom_callback);
    sub_joint_state = n.subscribe("/joint_states", 100, jointStateCallback);
    sub_imu = n.subscribe("/imu", 100, imuCallback);  // Ganti "/imu_topic" dengan topik IMU yang sesuai


    pub_cmd_vel = n.advertise<geometry_msgs::Twist>("/ackermann_steering_controller/cmd_vel", 100);
    pub_odom = n.advertise<nav_msgs::Odometry>("/odom", 100);
    pub_tf = n.advertise<tf2_msgs::TFMessage>("/tf", 100);
    pub_yaw = n.advertise<std_msgs::Float64>("/imu_yaw", 100);

    ros::Rate loop_rate(100);

    while (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;
}