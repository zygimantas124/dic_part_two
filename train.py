import numpy as np
import torch
from argparse import ArgumentParser
from envs.ball_env import WhiteBallEnv
from envs.delivery_env import DeliveryRobotEnv
from agents.DQN import *

def parse_args(argv=None):
    """
    Parse command line arguments.
    """
    p = ArgumentParser(description="Deep Q-Learning Training Script")
    # Environment and Action Space
    p.add_argument("--env_render_mode", type=str, default=None, help="Render mode for the environment (e.g., 'human').")
    p.add_argument("--n_actions", type=int, default=8, help="Number of discrete actions for the agent.")
    
    # DQN Hyperparameters
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    p.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate.")
    p.add_argument("--epsilon_min", type=float, default=0.01, help="Minimal exploration rate.")
    p.add_argument("--epsilon_decay", type=float, default=0.99995, help="Epsilon decay rate per episode.")
    p.add_argument("--alpha", type=float, default=1e-4, help="Learning rate for the Adam optimizer.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for DQN learning.")
    p.add_argument("--buffer_size", type=int, default=int(4e4), help="Size of the replay buffer.")
    p.add_argument("--min_replay_size", type=int, default=int(2e4), help="Minimum replay buffer size before training.")
    p.add_argument("--target_update_freq", type=int, default=int(5e3), 
                        help="Frequency (in learning steps) to update the target network.")
    
    # Training Control
    p.add_argument("--max_frames", type=int, default=int(15e4), help="Maximum number of frames to train for.")
    p.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum steps per episode.")
    p.add_argument("--log_interval", type=int, default=10, help="Interval (in episodes) for printing logs.")
    p.add_argument("--save_model_path", type=str, default="saved_Qnets/dqn_ball_env_model.pth", help="Path to save the trained model.")
    p.add_argument("--load_model_path", type=str, default=None, help="Path to load a pre-trained model.")

    return p.parse_args(argv)

def action_to_angle(action_idx, n_actions):
    """
    Convert a discrete action index to a continuous angle for the WhiteBallEnv.
    Args:
        action_idx (int): The discrete action index.
        n_actions (int): The total number of discrete actions.
    Returns:
        np.ndarray: A numpy array containing the angle in degrees.
    """
    # Maps action_idx (0 to n_actions-1) to an angle (0 to 360 degrees)
    return np.array([action_idx * (360.0 / n_actions)], dtype=np.float32)

# Main training loop
def train(args):

    # Initialize environment
    render_mode = args.env_render_mode if args.env_render_mode in [None, "human", "rgb_array"] else None
    env = WhiteBallEnv(n_angles=args.n_actions, render_mode=render_mode)
    #env = DeliveryRobotEnv(render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=args.n_actions,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay_rate=args.epsilon_decay,
        alpha=args.alpha,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
        target_update_freq=args.target_update_freq,
        device=device
    )

    if args.load_model_path:
        try:
            agent.load_model(args.load_model_path)
            agent.set_epsilon_for_eval()  # Or a small epsilon if continuing training
            print(f"Model loaded from {args.load_model_path}. Epsilon set to {agent.epsilon}.")
        except FileNotFoundError:
            print(f"No model found at {args.load_model_path}, starting training from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting training from scratch.")


    obs, _ = env.reset()
    
    total_frames = 0
    episode_count = 0
    all_episode_rewards = []

    print(f"Starting training for {args.max_frames} frames...")

    while total_frames < args.max_frames:
        episode_reward = 0.0
        episode_steps = 0
        done = False
        obs, _ = env.reset()  # Reset environment at the start of each episode

        while not done and episode_steps < args.max_episode_steps and total_frames < args.max_frames :
            if args.env_render_mode == "human":
                env.render()

            # Agent selects an action
            action_idx = agent.select_action(obs)
            
            # Convert discrete action index to environment-specific action (angle)
            angle_action = action_to_angle(action_idx, args.n_actions)
            
            # Environment steps
            next_obs, reward, terminated, truncated, _ = env.step(angle_action)
            done = terminated or truncated # Episode ends if terminated or truncated

            # Store transition in agent's replay buffer
            agent.store_transition(obs, action_idx, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            total_frames += 1
            episode_steps += 1

            # Train the agent
            if agent._can_learn():
                loss = agent.learn() # This also handles target network updates internally
                agent.decay_epsilon_multiplicative()
            
            if done:
                break
        
        # End of episode
        episode_count += 1
        all_episode_rewards.append(episode_reward)

        # Decay epsilon (e.g., per episode) - using multiplicative decay as an example
        agent.decay_epsilon_multiplicative() # Call this for episode-wise decay

        if episode_count % args.log_interval == 0:
            avg_reward = np.mean(all_episode_rewards[-args.log_interval:])
            print(f"Total Frames: {total_frames}/{args.max_frames} | Episode: {episode_count} | "
                  f"Avg Reward (last {args.log_interval} eps): {avg_reward:.2f} | "
                  f"Current Epsilon: {agent.epsilon:.3f}")
        
        if total_frames >= args.max_frames:
            print("Maximum frames reached. Training finished.")
            break
            
    env.close()
    print("Training complete.")
    
    # Save the final model
    if args.save_model_path:
        agent.save_model(args.save_model_path)

if __name__ == "__main__":
    args = parse_args()
    train(args)
    # Can be run using:  python train.py