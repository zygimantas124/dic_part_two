import numpy as np
import torch
import time
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv
from agents.DQN import DQNAgent

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

def evaluate_agent(args):
    env = DeliveryRobotEnv(config = 'simple', render_mode=args.render_mode, show_walls=False, show_carpets=False, show_obstacles=False)
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
        device=device
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

    print(f"\nStarting evaluation for {args.n_episodes} episodes...")

    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < args.max_episode_steps:
            if args.render_mode == "human":
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)

            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode+1}/{args.n_episodes}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min Reward:     {np.min(episode_rewards):.2f}")
    print(f"Max Reward:     {np.max(episode_rewards):.2f}")
    print(f"Avg Steps:      {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print("--------------------------")

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_agent(args)
