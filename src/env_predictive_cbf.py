import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import osqp
from scipy import sparse
import collections
from math import inf



GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.3
TIME_DELTA = 0.1

WALKER_SPEED = 0.2  # m/s, adjust as you like

# New distance constraints
MIN_GOAL_ROBOT_DIST = 1.5     # min distance between robot and goal
MIN_BOX_ROBOT_DIST = 1.5      # already used implicitly
MIN_BOX_GOAL_DIST = 1.5       # already used implicitly
WORLD_MIN_X, WORLD_MAX_X = -4.0, 4.0
WORLD_MIN_Y, WORLD_MAX_Y = -4.0, 4.0



class RobotEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 1
        self.goal_y = 0.0

        # For debugging plots
        self.debug_log = {
            "t": [],
            "raw_rate": [],
            "rob_rate": [],
            "rho": [],
            "action": [],
            "safe_action": [],
        }

        # --- walkers (start -> destination) ---
        # (model_name, start(x,y), end(x,y))
        self.walkers = [
            ("person_walking_10",   ( 4.0,  0.0), (-4.0,  0.0)),
            ("person_walking_0", (-3.0,  4.0), (-3.0, -4.0)),
            ("person_walking_1", ( 4.0, -3.0), (-1.0,  2.0)),
            ("person_walking_2", (-1.0, -3.0), ( 4.0, -3.0)),
            ("person_walking_3", (-2.0,  1.0), ( 1.0,  4.0)),
            ("person_walking_4", ( 5.0,  5.0), ( 5.0, -2.0)),
            ("person_walking_5", ( 0.0,  3.0), ( 5.0,  3.0)),
            ("person_walking_6", (-5.0, -4.0), (-5.0,  4.0)),
            ("person_walking_7", (-4.0,  5.0), ( 4.0,  5.0)),
            ("person_walking_8", (-5.0, -1.0), ( 4.0, -1.0)),
            ("person_walking_9", ( 5.0, -5.0), (-4.0, -5.0)),
        ]

                # --- Walker definitions (start -> destination) ---
        # Each walker moves in a straight line from start to end.
        self.walker_paths = [
            {
                "name": "person_walking_10",
                "start": np.array([ 4.0,  0.0], dtype=float),
                "end":   np.array([-4.0,  0.0], dtype=float),
            },
            {
                "name": "person_walking_0",
                "start": np.array([-3.0,  4.0], dtype=float),
                "end":   np.array([-3.0, -4.0], dtype=float),
            },
            {
                "name": "person_walking_1",
                "start": np.array([ 4.0, -3.0], dtype=float),
                "end":   np.array([-1.0,  2.0], dtype=float),
            },
            {
                "name": "person_walking_2",
                "start": np.array([-1.0, -3.0], dtype=float),
                "end":   np.array([ 4.0, -3.0], dtype=float),
            },
            {
                "name": "person_walking_3",
                "start": np.array([-2.0,  1.0], dtype=float),
                "end":   np.array([ 1.0,  4.0], dtype=float),
            },
            {
                "name": "person_walking_4",
                "start": np.array([ 5.0,  5.0], dtype=float),
                "end":   np.array([ 5.0, -2.0], dtype=float),
            },
            {
                "name": "person_walking_5",
                "start": np.array([ 0.0,  3.0], dtype=float),
                "end":   np.array([ 5.0,  3.0], dtype=float),
            },
            {
                "name": "person_walking_6",
                "start": np.array([-5.0, -4.0], dtype=float),
                "end":   np.array([-5.0,  4.0], dtype=float),
            },
            {
                "name": "person_walking_7",
                "start": np.array([-4.0,  5.0], dtype=float),
                "end":   np.array([ 4.0,  5.0], dtype=float),
            },
            {
                "name": "person_walking_8",
                "start": np.array([-5.0, -1.0], dtype=float),
                "end":   np.array([ 4.0, -1.0], dtype=float),
            },
            {
                "name": "person_walking_9",
                "start": np.array([ 5.0, -5.0], dtype=float),
                "end":   np.array([-4.0, -5.0], dtype=float),
            },
        ]

        # Progress parameter s in [0,1] for each walker
        # s = 0 -> at start, s = 1 -> at destination
        self.walker_progress = {
            w["name"]: 0.0 for w in self.walker_paths
        }

        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        # --- CBF parameters ---
        self.cbf_d_min = 0.3   # safety distance [m] (must be > COLLISION_DIST)
        self.cbf_gamma = 1.0   # how aggressively to slow down near obstacles
                # --- CBF / PA-CBF extra config ---
        self.cbf_front_angle = math.pi         # +/- 90deg sector in front
        self.cbf_window_W = 10                 # sliding window length
        self.cbf_k_sigma = 2.0                 # robustness multiplier

        # history for range-rate estimation
        self.cbf_prev_ranges = None            # previous velodyne_data
        self.cbf_rate_history = None           # list of deques, one per beam
        self.cbf_prev_v = None              # Store the speed applied during the previous interval
        self.cbf_horizon = 3          # H: number of steps into the future
        self.cbf_gamma_pred = 1.0     # predictive CBF gain (can reuse cbf_gamma)
        self.cbf_dt = TIME_DELTA      # prediction time step (same as control step)


        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def _sector_centers(self):
        """Return one angle beta_j (center) for each velodyne sector j."""
        betas = []
        for (a0, a1) in self.gaps:
            betas.append(0.5 * (a0 + a1))
        return np.array(betas, dtype=float)

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def cbf_filter(self, action):
        """
        Filter the RL action [v_nom, w_nom] using Predictive PA-CBF.
        Returns safe [v_star, w_star].
        """
        v_nom, w_nom = float(action[0]), float(action[1])

        rho = np.array(self.velodyne_data, dtype=float)  # current ranges (per sector)
        n_beams = len(rho)

        # If we don't have previous data yet, we can't estimate range-rates -> just pass through
        if self.cbf_prev_ranges is None:
            self.cbf_prev_ranges = rho.copy()
            # init history arrays
            self.cbf_rate_history = [
                collections.deque(maxlen=self.cbf_window_W) for _ in range(n_beams)
            ]
            return [v_nom, w_nom]

        # ----- 1) Range-rate estimation -----
        dt = TIME_DELTA  # your env uses a fixed propagation step
        raw_rate = (rho - self.cbf_prev_ranges) / dt
        betas = self._sector_centers()

        if not hasattr(self, "cbf_prev_v") or self.cbf_prev_v is None:
            v_prev = v_nom  # initialize previous speed if not set
        else:
            v_prev = self.cbf_prev_v
        w_hat = raw_rate + v_prev * np.cos(betas)

        # update histories
        for j in range(n_beams):
            self.cbf_rate_history[j].append(w_hat[j])

        # robust rate: rho_dot_rob = raw - k_sigma * sigma(history)
        
        robust_rate = np.zeros(n_beams)
        for j in range(n_beams):
            hist = np.array(self.cbf_rate_history[j])
            sigma = np.std(hist) if len(hist) > 1 else 0.0
            robust_rate[j] = w_hat[j] - self.cbf_k_sigma * sigma

        self.debug_log["t"].append(rospy.get_time())
        self.debug_log["raw_rate"].append(raw_rate.copy())        # or raw_rate[j]
        self.debug_log["rob_rate"].append(robust_rate.copy())     # or robust_rate[j]
        self.debug_log["rho"].append(rho.copy())


        self.cbf_prev_ranges = rho.copy()

        # ----- 2) Build linear constraints A u <= b for u = [v, w] -----
        betas = self._sector_centers()
        A = []
        b = []

        dt = self.cbf_dt
        H  = self.cbf_horizon

        for j in range(n_beams):
            beta_j = betas[j]

            # only forward sector
            if abs(beta_j) > self.cbf_front_angle / 2.0:
                continue

            c = math.cos(beta_j)
            rho_j = rho[j]
            rho_dot_rob = robust_rate[j]

            # current barrier h_{j,0}
            h0 = rho_j - self.cbf_d_min

            # affine coefficients for h_{j,0}(v) = alpha * v + beta
            alpha = 0.0
            beta_ = h0

            epsilon_j = np.zeros(H + 1)  # robustness margin for each step
            epsilon_step = 0.01 # [m] margin per step, can be tuned
            for ell_eps in range(H + 1):
                epsilon_j[ell_eps] = ell_eps * epsilon_step

            for ell in range(H):
                # propagate barrier one step forward:
                # h_{ℓ+1} = h_{ℓ} + dt * (-v * c + rho_dot_rob)
                alpha_next = alpha - dt * c
                beta_next  = beta_ + dt * rho_dot_rob - (epsilon_j[ell + 1] - epsilon_j[ell])  

                gamma_l = self.cbf_gamma_pred  # could vary with ell if you like

                # PPA-CBF: h_{ℓ+1} >= (1 - gamma_l * dt) * h_{ℓ}
                # -> A_coef * v <= b_coef
                A_coef = -(alpha_next - (1.0 - gamma_l * dt) * alpha)
                b_coef =  beta_next  - (1.0 - gamma_l * dt) * beta_

                A.append([A_coef, 0.0])
                b.append(b_coef)

                # prepare for next step
                alpha, beta_ = alpha_next, beta_next

        # If we got no forward beams (should not happen), just pass through
        if len(A) == 0:
            return [v_nom, w_nom]

        v_min, v_max = -0.1, 1.0    # or -0.5 to 0.5, depending on your scaling
        w_min, w_max = -1.0, 1.0

        A.append([1.0, 0.0]);  b.append(v_max)
        A.append([-1.0, 0.0]); b.append(-v_min)
        A.append([0.0, 1.0]);  b.append(w_max)
        A.append([0.0,-1.0]);  b.append(-w_min)

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        # ----- 3) Solve QP: min 0.5 ||u - u_nom||^2  s.t. A u <= b -----
        P = sparse.eye(2, format="csc")        # 2x2 identity
        q = -np.array([v_nom, w_nom], dtype=float)

        A_mat = sparse.csc_matrix(A)
        l = -inf * np.ones_like(b)            # no lower bounds
        u_vec = b

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A_mat, l=l, u=u_vec, verbose=False)
        res = prob.solve()

        if res.info.status_val not in (1, 2):  # not solved
            # fall back to nominal control
            r#eturn [v_nom, w_nom]
            return [v_min, np.clip(w_nom, w_min, w_max)]

        v_star, w_star = res.x[0], res.x[1]
        self.cbf_prev_v = float(v_star)
        return [float(v_star), float(w_star)]

    def update_walkers(self, dt):
        """
        Move every walker along its straight line from start to end
        with speed WALKER_SPEED (global variable).
        """
        for w in self.walker_paths:
            name = w["name"]
            start = w["start"]
            end = w["end"]

            # current progress s in [0,1]
            s = self.walker_progress[name]
            if s >= 1.0:
                # already at destination; don't move further
                continue

            # distance of the full segment
            seg_vec = end - start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue  # degenerate segment

            # how much of the segment to advance this step
            # distance = v * dt  =>  ds = (v * dt) / seg_len
            ds = (WALKER_SPEED * dt) / seg_len
            s_new = min(1.0, s + ds)
            self.walker_progress[name] = s_new

            # new position
            pos = start + s_new * seg_vec

            ms = ModelState()
            ms.model_name = name
            ms.pose.position.x = float(pos[0])
            ms.pose.position.y = float(pos[1])
            ms.pose.position.z = 0.0

            # keep same heading (along segment)
            yaw = math.atan2(seg_vec[1], seg_vec[0])
            q = Quaternion.from_euler(0.0, 0.0, yaw)
            ms.pose.orientation.x = q.x
            ms.pose.orientation.y = q.y
            ms.pose.orientation.z = q.z
            ms.pose.orientation.w = q.w

            self.set_state.publish(ms)

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # --- move walker for this step ---
        self.update_walkers(TIME_DELTA)

        safe_action = self.cbf_filter(action)
        self.debug_log["action"].append(np.array(action).copy())
        self.debug_log["safe_action"].append(np.array(safe_action).copy())

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = safe_action[0]   
        vel_cmd.angular.z = safe_action[1]  
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(safe_action)   

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, safe_action[0], safe_action[1]]
        state = np.append(laser_state, robot_state)

        # ---Choose reward function here---
        reward = self.get_reward(target, collision, safe_action, min_laser)
        # reward = self.get_cbf_reward(target, collision, action, safe_action, min_laser)

        # --- CBF & safety logging info (for training script) ---
        cbf_override = abs(action[0] - safe_action[0]) + abs(action[1] - safe_action[1])
        info = {
            "cbf_override": float(cbf_override),      # magnitude of CBF intervention
            "min_laser": float(min_laser),            # current min distance to obstacle
            "collision": bool(collision),             # True if in collision band
            "distance_to_goal": float(distance),      # current distance to goal
        }

#         np.savez("~/cbf_debug_log.npz",
#             t=np.array(self.debug_log["t"]),
#             raw_rate=np.array(self.debug_log["raw_rate"]),
#             rob_rate=np.array(self.debug_log["rob_rate"]),
#             rho=np.array(self.debug_log["rho"]),
#             action=np.array(self.debug_log["action"]),
#             safe_action=np.array(self.debug_log["safe_action"]),
# )

        return state, reward, done, target, info

    def sample_goal_away_from_walkers(self, min_dist_to_walkers=1.0):
        """
        Randomly sample a goal (x, y) within WORLD_MIN/MAX,
        making sure it is at least `min_dist_to_walkers` meters
        away from every walker's destination AND not too close
        to the robot start.
        """
        # Collect all destination points
        walker_destinations = [w[2] for w in self.walkers]  # (name, start, dest) -> dest

        while True:
            gx = random.uniform(WORLD_MIN_X, WORLD_MAX_X)
            gy = random.uniform(WORLD_MIN_Y, WORLD_MAX_Y)

            # 1) keep some distance from walker destinations
            ok_walkers = True
            for (dx, dy) in walker_destinations:
                if math.hypot(gx - dx, gy - dy) < min_dist_to_walkers:
                    ok_walkers = False
                    break

            if not ok_walkers:
                continue

            # 2) keep some distance from robot start (0,0) using MIN_GOAL_ROBOT_DIST
            if math.hypot(gx - 0.0, gy - 0.0) < MIN_GOAL_ROBOT_DIST:
                continue

            # If we get here, the sampled goal is valid
            return gx, gy

    def reset(self, episode):
        if episode >= 600:
            WALKER_SPEED = 0.25
        if episode >= 1000:
            WALKER_SPEED = 0.3
        if episode >= 2000:
            WALKER_SPEED = 0.35

        self.cbf_prev_ranges = None
        self.cbf_rate_history = None
        self.cbf_prev_v = None
        
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # --- reset walkers to their start positions ---
        for w in self.walker_paths:
            name = w["name"]
            start = w["start"]

            # progress back to 0
            self.walker_progress[name] = 0.0

            ms = ModelState()
            ms.model_name = name
            ms.pose.position.x = float(start[0])
            ms.pose.position.y = float(start[1])
            ms.pose.position.z = 0.0

            # orientation along the direction of motion
            direction = w["end"] - w["start"]
            yaw = math.atan2(direction[1], direction[0])
            q = Quaternion.from_euler(0.0, 0.0, yaw)
            ms.pose.orientation.x = q.x
            ms.pose.orientation.y = q.y
            ms.pose.orientation.z = q.z
            ms.pose.orientation.w = q.w

            # zero velocity
            ms.twist.linear.x = 0.0
            ms.twist.linear.y = 0.0
            ms.twist.linear.z = 0.0
            ms.twist.angular.x = 0.0
            ms.twist.angular.y = 0.0
            ms.twist.angular.z = 0.0

            self.set_state.publish(ms)

        angle = 0.0 
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0.0
        y = 0.0

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # Random goal, but not closer than 1 m to any walker destination
        gx, gy = self.sample_goal_away_from_walkers(min_dist_to_walkers=1.0)
        self.goal_x = gx
        self.goal_y = gy

        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

   
