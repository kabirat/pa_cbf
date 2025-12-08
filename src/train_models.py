#!/usr/bin/env python3
import os
import sys
import time
import csv
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

import rospy
from tf.transformations import euler_from_quaternion

# Ensure local `src/` directory (package workspace) is on sys.path so sibling modules
# like `replay_buffer.py` can be imported when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_buffer import ReplayBuffer

# Ensure a ROS node is initialized. Some environments/launchers expect the script
# to call rospy.init_node; others (custom env wrappers) may have already done so.
# We try to initialize and ignore errors if ROS is already initialized.
try:
    rospy.init_node('train_instantaneous', anonymous=True)
except Exception:
    pass

# Use package-root-relative absolute paths for results and model saving so files are
# written to a predictable location even when the script is launched from another
# working directory (roslaunch sets a different CWD).
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ======== Evaluation function ======
def evaluate(network, epoch,episode_num, eval_episodes=10,):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset(episode_num)
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state), deterministic=True)
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


# ===== SAC networks =====
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy with reparameterization and log-prob correction."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 800)
        self.fc2 = nn.Linear(800, 600)
        self.mean = nn.Linear(600, action_dim)
        self.log_std = nn.Linear(600, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mean, std, log_std

    def sample(self, s):
        mean, std, log_std = self.forward(s)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        a = torch.tanh(pre_tanh)

        # Log prob (Gaussian) with tanh correction
        log_prob = (
            -0.5 * ((pre_tanh - mean) / (std + 1e-6)) ** 2
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - a.pow(2) + 1e-6), dim=1, keepdim=True)

        return a, log_prob, torch.tanh(mean)  # also return deterministic action

    def act(self, s, deterministic=False):
        with torch.no_grad():
            if not torch.is_tensor(s):
                s = torch.as_tensor(s, dtype=torch.float32, device=device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            if deterministic:
                mean, _, _ = self.forward(s)
                a = torch.tanh(mean)
            else:
                a, _, _ = self.sample(s)
        return a.cpu().numpy().flatten()


class QCritic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 800)
        self.q1_fc2 = nn.Linear(800, 600)
        self.q1_out = nn.Linear(600, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 800)
        self.q2_fc2 = nn.Linear(800, 600)
        self.q2_out = nn.Linear(600, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)

        x1 = F.relu(self.q1_fc1(x))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.q2_fc1(x))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=3e-4, tau=0.005, gamma=0.99, target_entropy=None, auto_alpha=True):
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        self.critic = QCritic(state_dim, action_dim).to(device)
        self.critic_target = QCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy

        if auto_alpha:
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = 0.2

        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma
        self.writer = SummaryWriter()
        self.iter_count = 0

    @property
    def alpha(self):
        if self.auto_alpha:
            return self.log_alpha.exp().item()
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val

    def get_action(self, state, deterministic=False):
        return self.actor.act(state, deterministic=deterministic)

    def train(self, replay_buffer, iterations, batch_size=256):
        av_q = 0.0
        av_loss_q = 0.0
        av_loss_pi = 0.0
        av_alpha = 0.0
        for _ in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            s = torch.tensor(batch_states, dtype=torch.float32, device=device)
            a = torch.tensor(batch_actions, dtype=torch.float32, device=device)
            r = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            d = torch.tensor(batch_dones, dtype=torch.float32, device=device)
            s2 = torch.tensor(batch_next_states, dtype=torch.float32, device=device)

            if r.ndim == 1:
                r = r.unsqueeze(-1)
            if d.ndim == 1:
                d = d.unsqueeze(-1)

            # --- Critic update ---
            with torch.no_grad():
                a2, logp_a2, _ = self.actor.sample(s2)
                q1_t, q2_t = self.critic_target(s2, a2)
                q_t_min = torch.min(q1_t, q2_t) - self.alpha * logp_a2
                target_q = r + (1.0 - d) * self.gamma * q_t_min

            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # --- Actor update ---
            a_pi, logp_pi, _ = self.actor.sample(s)
            q1_pi, q2_pi = self.critic(s, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * logp_pi - q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # --- Alpha update ---
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                current_alpha = self.log_alpha.exp().item()
            else:
                current_alpha = self.alpha

            # --- Soft update targets ---
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

            # Aggregates
            av_q += q_pi.mean().item()
            av_loss_q += critic_loss.item()
            av_loss_pi += actor_loss.item()
            av_alpha += current_alpha

        self.iter_count += 1
        iters = float(iterations)
        self.writer.add_scalar("loss/critic", av_loss_q / iters, self.iter_count)
        self.writer.add_scalar("loss/actor", av_loss_pi / iters, self.iter_count)
        self.writer.add_scalar("stats/avg_Q_pi", av_q / iters, self.iter_count)
        self.writer.add_scalar("stats/alpha", av_alpha / iters, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        if self.auto_alpha:
            torch.save(self.log_alpha.detach().cpu(), f"{directory}/{filename}_log_alpha.pt")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.auto_alpha:
            try:
                self.log_alpha = torch.load(f"{directory}/{filename}_log_alpha.pt").to(device)
                self.log_alpha.requires_grad_(True)
            except:
                pass


# ====== Training script ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('***DEVICE:', device,'***')
seed = 0
eval_freq = 5e3
max_ep = 500
eval_ep = 10
batch_size = 256
updates_per_step = 1
discount = 0.99
tau = 0.005
buffer_size = 1_000_000
save_model = True
load_model = False
random_near_obstacle = True
max_episodes = 10000
min_buffer_size = 10000


# Select environment to train on
select_env = rospy.get_param('~select_env', 'predictive')  # options: "instantaneous", "predictive", "cbf_qp"

if select_env == "instantaneous":
    from env_instantaneous import RobotEnv 
    file_name = "model_instantaneous"
elif select_env == "predictive":
    from env_predictive import RobotEnv 
    file_name = "model_predictive"
elif select_env == "cbf_qp":
    from env_cbf_qp import RobotEnv
    file_name = "model_cbf_qp"

# Folders (use absolute package-root paths)
os.makedirs(RESULTS_DIR, exist_ok=True)
if save_model:
    os.makedirs(MODELS_DIR, exist_ok=True)

# Episode/epoch CSVs
csv_path = os.path.join(RESULTS_DIR, "episode_rewards_" + select_env + ".csv")

with open(csv_path, "w", newline="") as f:
    # episode-level metrics for paper plots
    csv.writer(f).writerow([
        "episode",
        "total_reward",
        "collisions",
        "goal_reached",
        "avg_cbf_override",
        "avg_min_laser",
    ])


csv_path2 = os.path.join(RESULTS_DIR, "epoch_" + select_env + ".csv")
with open(csv_path2, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "reward"])

# ========== Robot data CSV (ROS time; matches server columns) ==========

def _wait_for_ros_time():
    """If /use_sim_time is true, wait until /clock starts publishing (ROS time > 0)."""
    try:
        use_sim = rospy.get_param("/use_sim_time", False)
    except Exception:
        use_sim = False
    if not use_sim:
        return
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() == 0.0:
        time.sleep(0.01)

# ================== Build env ==================
environment_dim = 20
robot_dim = 4

# Use the launchfile your env expects (it looks under <env module>/assets/)
env = RobotEnv(environment_dim)
time.sleep(5)  # give Gazebo a moment to come up


# Now that the env has initialized the ROS node, it's safe to query ROS time
_wait_for_ros_time()

# Seeds
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim 
action_dim = 2
max_action = 1.0

# Agent
network = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    lr=3e-4,
    tau=tau,
    gamma=discount,
    target_entropy=-float(action_dim),
    auto_alpha=True
)

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./models")
        print("***PREVIOUS MODEL LOADED***")
    except Exception as e:
        print("Could not load the stored model parameters, initializing training with random parameters")
        print(e)

# Eval storage
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 0

count_rand_actions = 0
random_action = []

episode_override_sum = 0.0
episode_min_laser_sum = 0.0
episode_collision_count = 0
episode_goal_reached = 0


# ===== Training Loop (updates every step) =====
while True:
    if done:
        if timestep != 0:
            # --- compute episode-level averages ---
            ep_len = max(1, episode_timesteps)
            avg_override = episode_override_sum / ep_len
            avg_min_laser = episode_min_laser_sum / ep_len

            # --- log to CSV (for post-hoc analysis / paper plots) ---
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode_num,
                    episode_reward,
                    episode_collision_count,
                    episode_goal_reached,
                    avg_override,
                    avg_min_laser,
                ])

            # --- log to TensorBoard as episode-level scalars ---
            network.writer.add_scalar("episode/total_reward", episode_reward, episode_num)
            network.writer.add_scalar("episode/collisions", episode_collision_count, episode_num)
            network.writer.add_scalar("episode/goal_reached", episode_goal_reached, episode_num)
            network.writer.add_scalar("episode/avg_cbf_override", avg_override, episode_num)
            network.writer.add_scalar("episode/avg_min_laser", avg_min_laser, episode_num)

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            epoch_reward = evaluate(network=network, epoch=epoch, episode_num=episode_num,eval_episodes=eval_ep)
            evaluations.append(epoch_reward)

            with open(csv_path2, "a", newline="") as f:
                csv.writer(f).writerow([epoch, epoch_reward])

            network.save(file_name, directory=MODELS_DIR)
            np.save(os.path.join(RESULTS_DIR, file_name), evaluations)
            epoch += 1

        state = env.reset(episode_num)
        done = False

        episode_reward = 0.0
        episode_timesteps = 0
        episode_num += 1

        # reset episode-level logging accumulators
        episode_override_sum = 0.0
        episode_min_laser_sum = 0.0
        episode_collision_count = 0
        episode_goal_reached = 0

    # ===== Collect action from stochastic policy =====
    action = network.get_action(np.array(state), deterministic=False)

    # Optional obstacle-randomization block (kept)
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Scale to env control: [0,1] linear & [-1,1] angular
    a_in = [(action[0] + 1) / 2, action[1]]    
    next_state, reward, done, target, info = env.step(a_in)

    # --- step-level CBF & safety logging ---
    cbf_override = info.get("cbf_override", 0.0)
    min_laser = info.get("min_laser", 10.0)
    collision_flag = 1 if info.get("collision", False) else 0
    dist_goal = info.get("distance_to_goal", 0.0)

    # log to TensorBoard per timestep
    network.writer.add_scalar("cbf/override", cbf_override, timestep)
    network.writer.add_scalar("cbf/min_distance", min_laser, timestep)
    network.writer.add_scalar("safety/collision_flag", collision_flag, timestep)
    network.writer.add_scalar("task/distance_to_goal", dist_goal, timestep)

    # accumulate for episode-level stats
    episode_override_sum += cbf_override
    episode_min_laser_sum += min_laser
    episode_collision_count += collision_flag
    if target:
        episode_goal_reached = 1

    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save to buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # ===== Train every step once the buffer is warm =====
    if replay_buffer.size() > min_buffer_size:
        network.train(
            replay_buffer,
            iterations=updates_per_step,
            batch_size=batch_size,
        )

    # Counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# ===== End Training Loop =====
print("Training completed.")