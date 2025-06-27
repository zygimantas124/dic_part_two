import numpy as np
import heapq
import os
import json
import torch
import time
import glob

import logging
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv
from agents.DQN import DQNAgent
from agents.PPO import PPOAgent
from util.helpers import hausdorff_distance, compute_optimal_path, compute_tortuosity


def parse_eval_args(argv=None):
    # Search for a .pt file inside the logs directory
    default_model_path = None
    pt_files = glob.glob("logs/*.pth")
    if pt_files:
        default_model_path = pt_files[0]

    p = ArgumentParser(description="Unified Agent Evaluation Script (DQN/PPO)")
    p.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help="Path to trained model (.pt). If not specified, uses first found in logs/.",
    )
    p.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["dqn", "ppo"],
        help="Algorithm type. If not specified, will try to detect from filename.",
    )
    p.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run for evaluation.")
    p.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Render mode for the environment (e.g., 'human', None).",
    )
    p.add_argument("--n_actions", type=int, default=8, help="Number of discrete actions the agent was trained with.")
    p.add_argument(
        "--eval_epsilon", type=float, default=0.05, help="Epsilon for exploration during evaluation (DQN only)."
    )
    p.add_argument("--max_episode_steps", type=int, default=500, help="Maximum steps per evaluation episode.")
    p.add_argument("--render_delay", type=float, default=0.03, help="Delay between frames when rendering (in seconds).")

    args = p.parse_args(argv)

    if args.model_path is None:
        raise FileNotFoundError("No model file provided and no *.pt file found in logs/")

    return args


def detect_algorithm_from_filename(filename):
    base = os.path.basename(filename).lower()
    if "ppo" in base:
        return "ppo"
    elif "dqn" in base:
        return "dqn"
    else:
        raise ValueError("Could not detect algorithm type from filename. Please specify --algo explicitly.")


def parse_env_config_from_filename(filename):
    base = os.path.basename(filename)
    config = {
        "env_name": "open_office_simple",
        "show_walls": False,
        "show_obstacles": False,
        "show_carpets": False,
        "use_flashlight": False,
        "use_raycasting": False,
        "reward_step": -0.01,
        "reward_collision": -1.0,
        "reward_delivery": 50.0,
        "reward_carpet": -0.2,
        "render_mode": None,
        "obs_dim": 5,  # default to no raycasting
    }

    if "walls" in base:
        config["show_walls"] = True
    if "obs" in base:
        config["show_obstacles"] = True
    if "ray" in base:
        config["use_raycasting"] = True
        config["obs_dim"] = 35  # raycasting obs dim
    if "noray" in base:
        config["use_raycasting"] = False
        config["obs_dim"] = 5
    if "flash" in base:
        config["use_flashlight"] = True

    return config


def load_environment_config(model_path):
    try:
        with open("logs/env_config.json", "r") as f:
            config = json.load(f)
        print("Loaded config from env_config.json")
        return config, config.get("algo", "dqn")
    except FileNotFoundError:
        print("env_config.json not found, parsing from filename")
        config = parse_env_config_from_filename(model_path)
        algo = detect_algorithm_from_filename(model_path)
        return config, algo


def create_agent(algo, config, args, obs_dim, device):
    if algo == "ppo":
        return PPOAgent(
            obs_dim=config.get("obs_dim", obs_dim),
            n_actions=args.n_actions,
            gamma=0.99,
            clip_eps=0.2,
            update_epochs=0,  # Not used during evaluation
            batch_size=1,
            epsilon=0.01,  # Not really used in PPO
            epsilon_min=0.01,
            epsilon_decay_rate=1.0,
            device=device,
            logger=None,
        )
    else:  # DQN
        return DQNAgent(
            obs_dim=obs_dim,
            n_actions=args.n_actions,
            gamma=0.99,
            epsilon=args.eval_epsilon,
            epsilon_min=args.eval_epsilon,
            epsilon_decay_rate=1.0,
            alpha=0.0,  # Not used during evaluation
            batch_size=1,
            buffer_size=1,
            min_replay_size=1,
            target_update_freq=1e8,
            device=device,
            goal_buffer_size=50000,
            goal_fraction=0.4,
            logger=None,
        )


def load_model(agent, algo, model_path, eval_epsilon):
    """Load model with algorithm-specific setup."""
    try:
        agent.load_model(model_path)

        if algo == "dqn":
            agent.q_net.eval()
            agent.epsilon = eval_epsilon

        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading {algo.upper()} model: {e}")
        return False


def select_action(agent, algo, obs):
    """Select action with algorithm-specific interface."""
    if algo == "ppo":
        action, _, _ = agent.select_action(obs)
        return action
    else:  # DQN
        return agent.select_action(obs)


def evaluate_agent(args):
    # Auto-detect algorithm if not specified
    algo = args.algo
    if algo is None:
        algo = detect_algorithm_from_filename(args.model_path)
        print(f"Auto-detected algorithm: {algo.upper()}")

    # Load environment configuration
    config, config_algo = load_environment_config(args.model_path)

    # Use config algo if we couldn't detect from args
    if args.algo is None and config_algo:
        algo = config_algo
        print(f"Using algorithm from config: {algo.upper()}")

    # Override render mode if specified
    render_mode = args.render_mode or config.get("render_mode")

    # Create environment
    env = DeliveryRobotEnv(
        config=config["env_name"],
        render_mode=render_mode,
        show_walls=config["show_walls"],
        show_obstacles=config["show_obstacles"],
        show_carpets=config["show_carpets"],
        use_flashlight=config.get("use_flashlight", False),
        use_raycasting=config.get("use_raycasting", False),
        reward_step=config.get("reward_step", -0.01),
        reward_collision=config.get("reward_collision", -1.0),
        reward_delivery=config.get("reward_delivery", 50.0),
        reward_carpet=config.get("reward_carpet", -0.2),
    )

    obs_dim = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Environment obs_dim: {obs_dim}")

    # Create and load agent
    agent = create_agent(algo, config, args, obs_dim, device)

    if not load_model(agent, algo, args.model_path, args.eval_epsilon):
        return

    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    hausdorff_distances = []
    tortuosities = []

    # Compute optimal path for comparison
    CELL_SIZE = 5
    optimal_path = compute_optimal_path(env, CELL_SIZE)
    baseline_tort = compute_tortuosity(optimal_path)
    print(f"Optimal path length: {len(optimal_path)} cells")
    print(f"Optimal path tortuosity: {baseline_tort:.4f}")
    print(f"\nStarting {algo.upper()} evaluation for {args.n_episodes} episodes...\n")

    # Run evaluation episodes
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

            # Track agent position
            agent_pos = tuple(env.robot_pos.tolist())
            agent_path.append(agent_pos)

            # Select action (algorithm-specific)
            action = select_action(agent, algo, obs)

            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        # Calculate metrics
        hd = hausdorff_distance(agent_path, optimal_path) if agent_path and optimal_path else 0.0
        hd_norm = hd / np.hypot(env.width, env.height)
        tort = compute_tortuosity(agent_path)

        # Store results
        hausdorff_distances.append(hd)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        tortuosities.append(tort)

        print(
            f"Episode {episode + 1}/{args.n_episodes} - Reward: {total_reward:.2f}, Steps: {steps}, "
            f"Hausdorff Dist: {hd:.2f} (norm: {hd_norm:.4f}), Tortuosity: {tort:.4f}"
        )

    env.close()

    # Calculate summary statistics
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_lengths)
    std_steps = np.std(episode_lengths)
    avg_hd = np.mean(hausdorff_distances)
    std_hd = np.std(hausdorff_distances)
    avg_tort = np.mean(tortuosities)
    std_tort = np.std(tortuosities)

    # Calculate AUC
    episodes_x = np.arange(1, len(episode_rewards) + 1)
    episodes_normalized = (episodes_x - 1) / (len(episode_rewards) - 1) if len(episode_rewards) > 1 else [0]

    try:
        auc_learning_curve = np.trapezoid(episode_rewards, episodes_normalized)
    except AttributeError:
        print("Could not use np.trapezoid, try using np.trapz for AUC calculation (however its deprecated in numpy 2.0+)")

    # Print summary
    print(f"\n--- {algo.upper()} Evaluation Summary ---")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:     {min_reward:.2f}")
    print(f"Max Reward:     {max_reward:.2f}")
    print(f"Avg Steps:      {avg_steps:.2f}")
    print(
        f"Avg Hausdorff Dist: {avg_hd:.2f} ± {std_hd:.2f} (normalized: {avg_hd / np.hypot(env.width, env.height):.4f})"
    )
    print(f"Avg Tortuosity: {avg_tort:.4f} ± {std_tort:.2f} (baseline: {baseline_tort:.4f})")
    print(f"AUC Reward: {auc_learning_curve:.2f}")
    print("-----------------------------")

    return {
        "avg_hausdorff": avg_hd,
        "std_hausdorff": std_hd,
        "avg_tortuosity": avg_tort,
        "std_tortuosity": std_tort,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "avg_steps": avg_steps,
        "std_steps": std_steps,
        "auc_reward": auc_learning_curve,
    }


if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_agent(args)
