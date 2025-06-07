import numpy as np
import torch
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv
from agents.DQN import DQNAgent
from tqdm import tqdm
import logging

def set_global_seed(seed):
    random.seed(seed)                 # Python built-in
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed_all(seed) # PyTorch GPU (if used)
    os.environ["PYTHONHASHSEED"] = str(seed)  # For consistent hashing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Logging Setup ----------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_level=logging.INFO, log_file="training.log"):
    logger = logging.getLogger("dqn_trainer")
    logger.setLevel(log_level)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# ---------- Argument Parsing ----------
def parse_args(argv=None):
    p = ArgumentParser(description="Deep Q-Learning Training Script")
    
    p.add_argument("--render_mode", type=str, default=None)
    p.add_argument("--n_actions", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_decay", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=int(1e5))
    p.add_argument("--min_replay_size", type=int, default=int(1e5))
    p.add_argument("--target_update_freq", type=int, default=int(5e4))
    p.add_argument("--max_episodes", type=int, default=200)
    p.add_argument("--max_episode_steps", type=int, default=1e4)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_model_path", type=str, default="saved_Qnets/bare_delivery_env_model.pth")
    p.add_argument("--load_model_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

    return p.parse_args(argv)


# ---------- Main Training Loop ----------
def train(args):
    logger = setup_logger(log_level=logging.DEBUG)  # Change to INFO if you want less verbosity
    set_global_seed(args.seed)
    render_mode = args.render_mode if args.render_mode in [None, "human", "rgb_array"] else None
    env = DeliveryRobotEnv(show_walls=False, show_carpets=False, show_obstacles=False, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    device = "cpu"

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
            agent.set_epsilon_for_eval()
            logger.info(f"Model loaded from {args.load_model_path}. Epsilon set to {agent.epsilon}.")
        except FileNotFoundError:
            logger.warning(f"No model found at {args.load_model_path}, starting from scratch.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Starting from scratch.")

    obs, _ = env.reset()
    total_steps = 0
    all_episode_rewards = []

    logger.info(f"Starting training for {args.max_episodes} episodes...")

    for episode_count in tqdm(range(args.max_episodes), desc="Training Episodes"):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps < args.max_episode_steps:
            if args.render_mode == "human":
                env.render()

            action_idx = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.store_transition(obs, action_idx, reward, next_obs, done)

            # Learn immediately (if ready)
            if total_steps % 4 == 0 and agent._can_learn():
                agent.learn()

            obs = next_obs
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if done:
                termination_type = "terminated" if terminated else "truncated"
                logger.debug(f"Episode {episode_count} ended via {termination_type} at step {episode_steps}")
                break

        all_episode_rewards.append(episode_reward)

        if episode_count > 20:
            agent.decay_epsilon_multiplicative()

        # if np.mean(all_episode_rewards[-5:]) < 0 and agent.epsilon < 0.8:
        #     agent.epsilon = 1
        #     logger.warning("Resetting epsilon to encourage recovery from collapse")

        if episode_count % args.log_interval == 0:
            recent_rewards = all_episode_rewards[-args.log_interval:]
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            logger.info(f"Total Steps: {total_steps} | Episode: {episode_count} | "
                        f"Avg Reward (last {args.log_interval} eps): {avg_reward:.2f} | "
                        f"Max Reward: {max_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        if episode_count >= args.max_episodes:
            logger.info("Maximum episodes reached. Training finished.")
            break



    env.close()
    logger.info("Training complete.")

    if args.save_model_path:
        agent.save_model(args.save_model_path)
        logger.info(f"Model saved to {args.save_model_path}.")

# ---------- Entry Point ----------
if __name__ == "__main__":
    args = parse_args()
    train(args)
