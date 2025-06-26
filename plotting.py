import os
import json
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Constants
CONFIG_DIR = "experiments"
LOG_DIR = "logs"
PLOT_DIR = "final_plots"
MOVING_AVG_WINDOW = 20
REWARD_THRESHOLD = 5.0

os.makedirs(PLOT_DIR, exist_ok=True)

def parse_config_file(filepath):
    """Parses a config text file into a flat list of CLI arguments."""
    args = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            args.extend(parts)
    return args

def run_training(args):
    """Run training with subprocess using parsed args."""
    command = ['python', 'main.py'] + args
    print(f"Running: {' '.join(command)}")
    start_time = time.time()
    subprocess.run(command)
    elapsed = time.time() - start_time
    return elapsed

def load_episodes(log_dir):
    """Load episode logs from JSON files in order."""
    logs = sorted(glob(os.path.join(log_dir, "episode_*.json")),
                  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    all_data = []
    for path in logs:
        with open(path, "r") as f:
            episode_data = json.load(f)
        all_data.append(episode_data)
    return all_data

def compute_metrics(episodes):
    """Compute metrics from episode data."""
    rewards, lengths, successes, collisions, carpets = [], [], [], [], []
    for episode in episodes:
        reward_vals = [step["reward"] for step in episode]
        total_reward = sum(reward_vals)
        rewards.append(total_reward)
        lengths.append(len(episode))
        successes.append(total_reward >= REWARD_THRESHOLD)
        collisions.append(sum(1 for r in reward_vals if r == -10))
        carpets.append(sum(1 for r in reward_vals if r == -1))
    return {
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes,
        "collisions": collisions,
        "carpets": carpets,
    }

def plot_metric(values, title, ylabel, filename, smooth=False):
    plt.figure(figsize=(8, 5))
    if smooth:
        smoothed = np.convolve(values, np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW, mode='valid')
        plt.plot(smoothed, label="Smoothed", color='blue')
    else:
        plt.plot(values, label="Raw", color='green')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_success_rate(successes, filename):
    rates = [np.mean(successes[max(0, i - MOVING_AVG_WINDOW):i + 1]) for i in range(len(successes))]
    plt.figure(figsize=(8, 5))
    plt.plot(rates, label="Success Rate", color='purple')
    plt.title("Success Rate over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved success rate plot to {filename}")

def parse_plot_filename(config_name):
    """Extracts plot filename details based on config name."""
    base = os.path.basename(config_name).replace(".txt", "")
    algo = "DQN" if "dqn" in base.lower() else "PPO"
    obst = "no_obst" if "no_obst" in base.lower() else "obst"
    rayc = "no_rayc" if "no_rayc" in base.lower() else "rayc"
    return f"{algo}_walls_{obst}_{rayc}"

def analyze_and_plot(config_name):
    episodes = load_episodes(LOG_DIR)
    if not episodes:
        print("No episode logs found. Skipping analysis.")
        return

    metrics = compute_metrics(episodes)
    plot_base = parse_plot_filename(config_name)

    reward_plot_file = os.path.join(PLOT_DIR, f"reward_plot_{plot_base}.png")
    success_plot_file = os.path.join(PLOT_DIR, f"success_rate_plot_{plot_base}.png")

    plot_metric(metrics["rewards"], f"Total Reward - {plot_base}", "Reward", reward_plot_file, smooth=True)
    plot_success_rate(metrics["successes"], success_plot_file)

def main():
    config_files = sorted(glob(os.path.join(CONFIG_DIR, "*.txt")))
    timing_log = {}

    for config_file in config_files:
        print(f"\n=== Running config: {config_file} ===")
        args = parse_config_file(config_file)
        elapsed = run_training(args)
        analyze_and_plot(config_file)
        timing_log[os.path.basename(config_file)] = elapsed
        print(f"Elapsed time: {elapsed:.2f} seconds")

    # Save timing log
    with open("training_times.json", "w") as f:
        json.dump(timing_log, f, indent=4)
    print("Saved timing information to training_times.json")

if __name__ == "__main__":
    main()