import numpy as np
import heapq
import os
import torch
import time
import logging
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv
from agents.DQN import DQNAgent
from util.helpers import hausdorff_distance, compute_optimal_path, compute_tortuosity

def parse_eval_args(argv=None):
    p = ArgumentParser(description="DQN Agent Evaluation Script")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the trained PyTorch model file (e.g., saved_Qnets/dqn_model.pth).")
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Number of episodes to run for evaluation.")
    p.add_argument("--render_mode", type=str, default=None,
                   choices=[None, "human", "rgb_array"],
                   help="Render mode for the environment (e.g., 'human', None).")
    p.add_argument("--n_actions", type=int, default=8,
                   help="Number of discrete actions the agent was trained with.")
    p.add_argument("--eval_epsilon", type=float, default=0.05,
                   help="Epsilon for exploration during evaluation (0 = greedy).")
    p.add_argument("--max_episode_steps", type=int, default=500,
                   help="Maximum steps per evaluation episode.")
    p.add_argument("--render_delay", type=float, default=0.03,
                   help="Delay between frames when rendering (in seconds). Only applies if render_mode='human'.")
    return p.parse_args(argv)

# TODO: during training save the agent and environment so we can call here in evaluate
def evaluate_agent(args):
    env = DeliveryRobotEnv(
        config="open_office_simple",
        show_walls=True,
        show_obstacles=True,
        show_carpets=False,
        render_mode="human"
    )
    obs_dim = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=args.n_actions,
        gamma=0.99,
        epsilon=args.eval_epsilon,
        epsilon_min=args.eval_epsilon,
        epsilon_decay_rate=1.0,
        alpha=0.0,
        batch_size=1,
        buffer_size=1,
        min_replay_size=1,
        target_update_freq=1e8,
        device=device,
        goal_buffer_size=50000,
        goal_fraction=0.4,
        logger=None
    )

    try:
        agent.load_model(args.model_path)
        agent.q_net.eval()
        agent.epsilon = args.eval_epsilon
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episode_rewards = []
    episode_lengths = []
    hausdorff_distances = [] # For deviation from optimal path
    tortuosities = [] # For unnecessary rotations

    # Precompute optimal path once for same environment layout
    CELL_SIZE = 1 # To discretize the environment, for a-star optimal path
    optimal_path = compute_optimal_path(env, CELL_SIZE)
    print(f"Optimal path length: {len(optimal_path)} cells")
    # Precompute tortuosity of the optimal path
    baseline_tort = compute_tortuosity(optimal_path)
    print(f"Optimal path tortuosity: {baseline_tort:.4f}")

    print(f"\nStarting evaluation for {args.n_episodes} episodes...")

    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        agent_path = []

        while not done and steps < args.max_episode_steps:
            if args.render_mode == "human":
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)
            # Record agent's position for Hausdorff distance
            agent_pos = tuple(env.robot_pos.tolist())
            agent_path.append(agent_pos)

            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Compute Hausdorff distance for this episode
        if not agent_path or not optimal_path:
            hd = 0.0
        else:
            hd = hausdorff_distance(agent_path, optimal_path)
        hd_norm = hd / np.hypot(env.width, env.height) # normalize by environment size
        tort = compute_tortuosity(agent_path) # Compute tortuosity of the agent's path

        # ----- Store results for episode -----
        hausdorff_distances.append(hd) # Hausdorff distance for this episode
        episode_rewards.append(total_reward) # total reward for this episode
        episode_lengths.append(steps) # steps taken in this episode
        tortuosities.append(tort) # tortuosity of the agent's path
        print(f"Episode {episode + 1}/{args.n_episodes} - Reward: {total_reward:.2f}, Steps: {steps}, "
              f"Hausdorff Dist: {hd:.2f} (norm: {hd_norm:.4f}), Tortuosity: {tort:.4f}")
    env.close()

    # ----- Evaluation summary -----
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_lengths)
    std_steps = np.std(episode_lengths)
    avg_hd = np.mean(hausdorff_distances) # Hausdorff distance
    std_hd = np.std(hausdorff_distances)
    x = np.arange(1, len(episode_rewards) + 1)
    #TODO: sanity check if auc_reward is correct
    auc_reward = np.trapezoid(episode_rewards, x) # Area Under Learning Curve (AUC)
    #TODO: sanity check for tortuosities
    avg_tort = np.mean(tortuosities)
    std_tort = np.std(tortuosities)

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min Reward:     {min_reward:.2f}")
    print(f"Max Reward:     {max_reward:.2f}")
    print(f"Avg Steps:      {avg_steps:.2f} +/- {std_steps:.2f}")
    print(f"Avg Hausdorff Dist: {avg_hd:.2f} ±{std_hd:.2f}")
    print(f"Avg Tortuosity: {avg_tort:.4f} ±{std_tort:.4f} (baseline: {baseline_tort:.4f})")
    print(f"AUC Reward: {auc_reward:.2f}")
    print("--------------------------")

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_agent(args)
