# run_and_log.py
import time
import csv
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rospy

# Make sure we can import env_proposed.py from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    rospy.init_node('train_sac', anonymous=True)
except Exception:
    pass
from env_test import RobotEnv 


# ---------- SAC actor definition (must match training) ----------
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
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
        return mean

class SACActorOnly:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)

    @torch.no_grad()
    def get_action(self, state, deterministic=True):
        s = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        mean = self.actor(s)
        a = torch.tanh(mean)  # SAC deterministic action
        return a.cpu().numpy().flatten()  # in [-1, 1]

    def load(self, filename, directory):
        path = os.path.join(directory, filename + "_actor.pth")
        self.actor.load_state_dict(torch.load(path, map_location=self.device))#, weight_only=True))

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

def main():
    # rospy.init_node("test_sac_logging", anonymous=True)

    # ---- Config ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    environment_dim = 20
    robot_dim = 4
    state_dim = environment_dim + robot_dim
    action_dim = 2

    max_ep_steps = 500        # per episode
    num_episodes = 20         # how many episodes to log
    model_dir = "./models/"
    file_name = "model_predictive"

    # ---- Env ----
    env = RobotEnv(environment_dim=environment_dim)
    time.sleep(5.0)

    _wait_for_ros_time()

    # Enable logging inside env
    env.log_enabled = True

    # ---- Policy ----
    policy = SACActorOnly(state_dim, action_dim, device=device)
    try:
        policy.load(file_name, model_dir)
        print("Loaded actor from:", os.path.join(model_dir, file_name + "_actor.pth"))
    except Exception as e:
        print("Failed to load actor model:", e)
        return

    all_rows = []
    csv_out = "test_logs" + time.strftime("_%Y%m%d-%H%M%S") + ".csv"

    for ep in range(num_episodes):
        state = env.reset()
        env.reset_step_log()
        done = False
        step_count = 0

        print(f"Episode {ep+1}/{num_episodes}")

        while not done and step_count < max_ep_steps:
            # Get deterministic action from policy in [-1, 1]
            action = policy.get_action(np.array(state), deterministic=True)

            # Rescale to env's expected input [0,1] x [-1,1]
            a_in = np.array([(action[0] + 1.0) / 2.0, action[1]], dtype=float)

            next_state, reward, done, target = env.step(a_in)
            step_count += 1
            state = next_state

            # If you want strictly time-limited episodes:
            if step_count >= max_ep_steps:
                done = True

        # After episode, pull logs from env
        ep_log = env.get_step_log()
        for k, entry in enumerate(ep_log):
            entry_with_meta = dict(entry)
            entry_with_meta["episode"] = ep
            entry_with_meta["step"] = k
            all_rows.append(entry_with_meta)

        print(f"  steps: {step_count}, target_reached: {ep_log[-1]['target'] if ep_log else False}")

    # ---- Save CSV ----
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Saved logs to", csv_out)
    else:
        print("No data collected!")


if __name__ == "__main__":
    main()
