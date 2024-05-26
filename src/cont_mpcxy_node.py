#!/usr/bin/env python3
import rospy
import random
from av_atras.msg import state
import matplotlib.pyplot as plt
from constants import *
from function import Kinematic

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist

import os
import sys
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs

class State_Publisher:
    def __init__(self):
        rospy.init_node('controller_state_publisher')

        # Subscribers
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.path_callback)
        rospy.Subscriber("/move_base/cmd_vel", Twist, self.cmd_vel_callback)

        # Variables to store latest data
        self.latest_odom = None
        self.latest_path_pose = None
        self.latest_path_twist = None

        self.path_x = None
        self.path_y = None
        self.path_psi = None
        self.path_x_dot = None
        self.path_psi_dot = None

        self.odom_x = None
        self.odom_y = None
        self.odom_psi = None
        self.odom_x_dot = None
        self.odom_y_dot = None
        self.odom_psi_dot = None

    def odom_callback(self, data):
        self.latest_odom = data
        # print(data)
        self.odom_x = self.latest_odom.pose.pose.position.x
        self.odom_y = self.latest_odom.pose.pose.position.y
        self.odom_psi = self.latest_odom.pose.pose.orientation.z

        self.odom_x_dot = self.latest_odom.twist.twist.linear.x
        self.odom_y_dot = self.latest_odom.twist.twist.linear.y
        self.odom_psi_dot = self.latest_odom.twist.twist.angular.z
        # print("=============================")
        # print("Odom_x : ", self.odom_x)
        # print("=============================")
        # print("Odom_y : ", self.odom_y)
        # print("=============================")
        # print("Odom_psi : ", self.odom_psi)
        # print("=============================")
        # print("Odom_x_dot : ", self.odom_x_dot)
        # print("=============================")
        # print("Odom_y_dot : ", self.odom_y_dot)
        # print("=============================")
        # print("Odom_psi_dot : ", self.odom_psi_dot)
        # print("=============================")
        return self.odom_x, self.odom_y, self.odom_psi, self.odom_x_dot, self.odom_y_dot, self.odom_psi_dot, self.latest_odom

    def path_callback(self, data):
        self.latest_path_pose = data.poses
        self.path_x = []
        self.path_y = []
        self.path_psi = []
        for pose_stamped in self.latest_path_pose:
            pose = pose_stamped.pose
            position = pose.position
            orientation = pose.orientation

            # Extract x, y, and z coordinates
            x = position.x
            y = position.y
            z = orientation.z  # Assuming orientation z represents the z component of quaternion
            
            self.path_x.append(x)
            self.path_y.append(y)
            self.path_psi.append(z)
        print("=============================")
        print("Path_x : ", self.path_x)
        print("=============================")
        print("Path_y : ", self.path_y)
        print("=============================")
        print("Path_psi : ", self.path_psi)
        print("=============================")
        print("Length : ", len(self.path_x))
        return self.path_x, self.path_y, self.path_psi, self.latest_path_pose
    def collect_car_state(self):
        refPos = None
        velocity_state = None
        velocity_reference = None
        if self.latest_path_pose is not None and self.latest_odom is not None:
            refPos = np.zeros([3,len(self.path_x)])
            velocity_state = []
            velocity_reference = np.zeros([2,len(self.path_psi)])

            X_error = [self.path_x[i]- self.odom_x for i in range(len(self.path_x))] 
            Y_error = [self.path_y[i] - self.odom_y for i in range(len(self.path_y))] 
            Psi_error = [self.path_psi[i] - self.odom_psi for i in range(len(self.path_psi))]
            Vx = self.odom_x_dot
            Vy = self.odom_y_dot
            Psi_dot = self.odom_psi_dot
            # position_error = [X_error,Y_error,Psi_error]

            refPos[0] = X_error
            refPos[1] = Y_error
            refPos[2] = Psi_error
            velocity_state = [Vx, Vy, Psi_dot]

            Vx_ref = [np.sqrt((self.path_x[i+1]-self.path_x[i])**2 + (self.path_y[i+1]-self.path_y[i])**2)/0.1 for i in range(len(self.path_x)-1)]
            Omega_ref = [(self.path_psi[i+1]-self.path_psi[i])/0.1 for i in range(len(self.path_psi)-1)]

            X_dot_ref = [self.path_x_dot*np.cos(Psi_error[i]) for i in range(len(self.path_psi))]
            Psi_dot_ref = [self.path_psi_dot for _ in range(len(X_error))]
            velocity_reference[0,0] = self.path_x_dot
            velocity_reference[0,1:] = Vx_ref
            velocity_reference[1,0] = self.path_psi_dot
            velocity_reference[1,1:] = Omega_ref 

            print("============================")
            print("Vx_reference", velocity_reference[0])
            print("Omega_reference", velocity_reference[1])
            print("X_error", position_error[0])
            print("Y_error", position_error[1])
            print("Psi_error", position_error[2])
            print("============================")
        else:
            rospy.logwarn("Either Path/Odom/Cmd_vel Message is Missing, check /move_base local plan,/odom and /move_base/cmd_vel")
            """
            NOTE : Velocity ref dari cmd_vel cuma 1 step, yang saya ulang 20 kali sbg referensi
            sepanjang horizon. Sepertinya ada yg salah karena seharusnya 20 step kedepan nilainya
            beda-beda. kalau dari simulasi kemarin xdot= (x2-x1)/T begitu juga psidot= (psi2-psi1)/T
            """
        return position_error, velocity_state, velocity_reference

    def run(self):
        pub = rospy.Publisher("/car/state", state, queue_size=10 )
        rate = rospy.Rate(10)  # 10 Hz
        car_msg = state()
        while not rospy.is_shutdown():
            position_error, velocity_state, velocity_reference = self.collect_car_state()

            if position_error is not None:
                # rospy.loginfo("Position Error: %s", position_error)
                car_msg.x = position_error[0]
                car_msg.y = position_error[1]
                car_msg.psi = position_error[2]

            if velocity_state is not None:
                # rospy.loginfo("Velocity State: %s", velocity_state)
                car_msg.x_dot = velocity_state[0]
                car_msg.y_dot = velocity_state[1]
                car_msg.psi_dot = velocity_state[2]

            if velocity_reference is not None:
                # rospy.loginfo("Velocity Reference: %s", velocity_reference)
                car_msg.x_dot_ref = velocity_reference[0]
                car_msg.psi_dot_ref = velocity_reference[1]

            pub.publish(car_msg)

            rate.sleep()
class config:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 6  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 1.0  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * config.dt
        self.y += self.v * math.sin(self.yaw) * config.dt
        self.yaw += self.v / config.WB * math.tan(delta) * config.dt
        self.direct = direct
        self.v += self.direct * a * config.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= config.steer_max:
            return config.steer_max

        if delta <= -config.steer_max:
            return -config.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= config.speed_max:
            return config.speed_max

        if v <= config.speed_min:
            return config.speed_min

        return v


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[self.ind_old: (self.ind_old + config.N_IND)]]
        dy = [node.y - y for y in self.cy[self.ind_old: (self.ind_old + config.N_IND)]]
        dist = np.hypot(dx, dy)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory
    """

    z_ref = np.zeros((config.NX, config.T + 1))
    length = ref_path.length

    ind, _ = ref_path.nearest_index(node)

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = sp[ind]
    z_ref[3, 0] = ref_path.cyaw[ind]

    dist_move = 0.0

    for i in range(1, config.T + 1):
        dist_move += abs(node.v) * config.dt
        ind_move = int(round(dist_move / config.d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = sp[index]
        z_ref[3, i] = ref_path.cyaw[index]

    return z_ref, ind


def linear_mpc_control(z_ref, z0, a_old, delta_old):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps of last time
    :param delta_old: delta of T steps of last time
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * config.T
        delta_old = [0.0] * config.T

    x, y, yaw, v = None, None, None, None

    for k in range(config.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < config.du_res:
            break

    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(z0, a, delta, z_ref):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: initial state
    :param a: acceleration strategy of last time
    :param delta: delta strategy of last time
    :param z_ref: reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(config.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, config.T + 1)):
        node.update(ai, di, 1.0)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar


def calc_linear_discrete_model(v, phi, delta):
    """
    calc linear and discrete time dynamic model.
    :param v: speed: v_bar
    :param phi: angle of vehicle: phi_bar
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    A = np.array([[1.0, 0.0, config.dt * math.cos(phi), - config.dt * v * math.sin(phi)],
                  [0.0, 1.0, config.dt * math.sin(phi), config.dt * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, config.dt * math.tan(delta) / config.WB, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [config.dt, 0.0],
                  [0.0, config.dt * v / (config.WB * math.cos(delta) ** 2)]])

    C = np.array([config.dt * v * math.sin(phi) * phi,
                  -config.dt * v * math.cos(phi) * phi,
                  0.0,
                  -config.dt * v * delta / (config.WB * math.cos(delta) ** 2)])

    return A, B, C


def solve_linear_mpc(z_ref, z_bar, z0, d_bar):
    """
    solve the quadratic optimization problem using cvxpy, solver: OSQP
    :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
    :param z_bar: predicted states in T steps
    :param z0: initial state
    :param d_bar: delta_bar
    :return: optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((config.NX, config.T + 1))
    u = cvxpy.Variable((config.NU, config.T))

    cost = 0.0
    constrains = []

    for t in range(config.T):
        cost += cvxpy.quad_form(u[:, t], config.R)
        cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], config.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < config.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], config.Rd)
            constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= config.steer_change_max * config.dt]

    cost += cvxpy.quad_form(z_ref[:, config.T] - z[:, config.T], config.Qf)

    constrains += [z[:, 0] == z0]
    constrains += [z[2, :] <= config.speed_max]
    constrains += [z[2, :] >= config.speed_min]
    constrains += [cvxpy.abs(u[0, :]) <= config.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= config.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def main():
    ax = [0.0, 15.0, 30.0, 50.0, 60.0]
    ay = [0.0, 40.0, 15.0, 30.0, 0.0]
    cx, cy, cyaw, ck, s = cs.calc_spline_course(
        ax, ay, ds=config.d_dist)

    sp = calc_speed_profile(cx, cy, cyaw, config.target_speed)

    ref_path = PATH(cx, cy, cyaw, ck) # ==>> UBAH REF PATH DARI LOCAL PLANNER
    node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    time = 0.0
    x = [node.x]
    y = [node.y]
    yaw = [node.yaw]
    v = [node.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]

    delta_opt, a_opt = None, None
    a_exc, delta_exc = 0.0, 0.0

    while time < config.time_max:
        z_ref, target_ind = calc_ref_trajectory_in_T_step(node, ref_path, sp)

        z0 = [node.x, node.y, node.v, node.yaw]

        a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = linear_mpc_control(z_ref, z0, a_opt, delta_opt)

        if delta_opt is not None:
            delta_exc, a_exc = delta_opt[0], a_opt[0]

        node.update(a_exc, delta_exc, 1.0)
        time += config.dt

        x.append(node.x)
        y.append(node.y)
        yaw.append(node.yaw)
        v.append(node.v)
        t.append(time)
        d.append(delta_exc)
        a.append(a_exc)

        dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

        if dist < config.dist_stop and abs(node.v) < config.speed_stop:
            break

        dy = (node.yaw - yaw[-2]) / (node.v * config.dt)
        steer = rs.pi_2_pi(-math.atan(config.WB * dy))

        plt.cla()
        draw.draw_car(node.x, node.y, node.yaw, steer, config)
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event:
                                     [exit(0) if event.key == 'escape' else None])

        if x_opt is not None:
            plt.plot(x_opt, y_opt, color='darkviolet', marker='*')

        plt.plot(cx, cy, color='gray')
        plt.plot(x, y, '-b')
        plt.plot(cx[target_ind], cy[target_ind])
        plt.axis("equal")
        plt.title("Linear MPC, " + "v = " + str(round(node.v * 3.6, 2)))
        plt.pause(0.001)

    plt.show()



if __name__ == '__main__':
    state_publisher_node = State_Publisher()
    state_publisher_node.run()
    main()