import numpy as np
import torch
from argparse import ArgumentParser
import time # For optional rendering delay
from envs.ball_env import WhiteBallEnv
from agents.DQN import *

def action_to_angle(action_idx, n_actions):
    """
    Convert a discrete action index to a continuous angle for the WhiteBallEnv.
    Args:
        action_idx (int): The discrete action index.
        n_actions (int): The total number of discrete actions.
    Returns:
        np.ndarray: A numpy array containing the angle in degrees.
    """
    return np.array([action_idx * (360.0 / n_actions)], dtype=np.float32)

def parse_eval_args(argv=None):
    p = ArgumentParser(description="DQN Agent Evaluation Script")
    p.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained PyTorch model file (e.g., dqn_ball_env_model.pth).")
    p.add_argument("--n_episodes", type=int, default=10,
                       help="Number of episodes to run for evaluation.")
    p.add_argument("--render_mode", type=str, default=None,
                       choices=[None, "human", "rgb_array"],
                       help="Render mode for the environment (e.g., 'human', None).")
    p.add_argument("--n_actions", type=int, default=8,
                       help="Number of discrete actions the agent was trained with.")
    p.add_argument("--eval_epsilon", type=float, default=0.05,
                       help="Epsilon for exploration during evaluation (e.g., 0.01 for near-greedy, 0 for purely greedy).")
    p.add_argument("--max_episode_steps", type=int, default=500,
                       help="Maximum steps per evaluation episode.")
    p.add_argument("--render_delay", type=float, default=0.03,
                       help="Delay in seconds between frames when rendering (e.g., 0.03). Only applies if render_mode='human'.")
    return p.parse_args(argv)

def evaluate_agent(args):
    """
    Evaluate the trained DQN agent.
    """
    # Initialize environment
    render_mode = args.render_mode
    try:
        env = WhiteBallEnv(n_angles=args.n_actions, render_mode=render_mode)
    except Exception as e:
        print(f"Error initializing WhiteBallEnv: {e}")
        return
        
    obs_dim = env.observation_space.shape[0]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent with dummy parameters for non-training aspects
    # The critical parts are obs_dim, n_actions, and device for network loading.
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=args.n_actions,
        gamma=0.99, # Not strictly used in eval
        epsilon=args.eval_epsilon,
        epsilon_min=args.eval_epsilon, # Ensure it stays at eval_epsilon
        epsilon_decay_rate=1.0, # No decay during evaluation
        alpha=0, # Learning rate not used
        batch_size=1, # Not used for inference
        buffer_size=1, # Replay buffer not used
        min_replay_size=1, # Not used
        target_update_freq=1e8, # Effectively disable target updates
        device=device
    )

    # Load the trained model weights
    try:
        agent.load_model(args.model_path)
        # Ensure the Q-network is in evaluation mode (affects layers like Dropout, BatchNorm)
        agent.q_net.eval()
        # The agent's load_model method should also handle target_q_net and set it to eval if necessary.
        # DQNAgent.epsilon is set during __init__, but we re-affirm it here based on args.
        agent.epsilon = args.eval_epsilon
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episode_rewards = []
    episode_lengths = []

    print(f"\nStarting evaluation for {args.n_episodes} episodes...")

    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        while not (terminated or truncated) and current_episode_length < args.max_episode_steps:
            if render_mode == "human":
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)

            # Agent selects an action (using agent.epsilon set to args.eval_epsilon)
            action_idx = agent.select_action(obs, explore=True)

            # Convert action to environment format
            angle_action = action_to_angle(action_idx, args.n_actions)

            next_obs, reward, terminated, truncated, _ = env.step(angle_action)
            
            obs = next_obs
            current_episode_reward += reward
            current_episode_length += 1

        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)
        print(f"Episode {episode + 1}/{args.n_episodes}: Reward = {current_episode_reward:.2f}, Length = {current_episode_length}")

    env.close()

    # Calculate and print statistics
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    min_reward = np.min(episode_rewards) if episode_rewards else 0
    max_reward = np.max(episode_rewards) if episode_rewards else 0

    mean_length = np.mean(episode_lengths) if episode_lengths else 0
    std_length = np.std(episode_lengths) if episode_lengths else 0

    print("\n--- Evaluation Summary ---")
    print(f"Number of episodes evaluated: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Average Reward:           {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Min Reward:               {min_reward:.2f}")
        print(f"Max Reward:               {max_reward:.2f}")
        print(f"Average Episode Length:   {mean_length:.2f} +/- {std_length:.2f}")
    else:
        print("No episodes were completed to show statistics.")
    print("--- ------------------ ---")

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_agent(args)
    # Can be run using:  python evaluate.py --model_path="saved_Qnets/dqn_ball_env_model.pth" --render_mode="human"