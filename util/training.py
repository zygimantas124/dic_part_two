import torch
import numpy as np
from tqdm import tqdm
from office.delivery_env import DeliveryRobotEnv
from agents.PPO import PPOAgent
from agents.DQN import DQNAgent
from util.helpers import set_global_seed

def initialize_environment(args):
    """Initialize the DeliveryRobotEnv with validated render mode."""
    render_mode = args.render_mode if args.render_mode in [None, "human", "rgb_array"] else None
    env_config = {
        "config": "open_office_simple",
        "show_walls": True if args.algo == "ppo" else False,
        "show_carpets": False,
        "show_obstacles": False,
        "render_mode": render_mode
    }
    return DeliveryRobotEnv(**env_config)

def initialize_agent(args, obs_dim):
    """Initialize the appropriate agent (PPO or DQN) based on args.algo."""
    device = args.device
    common_params = {
        "obs_dim": obs_dim,
        "n_actions": args.n_actions,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "device": device
    }

    if args.algo == "ppo":
        ppo_params = {
            "clip_eps": args.eps_clip,
            "update_epochs": args.k_epochs
        }
        return PPOAgent(**common_params, **ppo_params)
    else:  # dqn
        dqn_params = {
            "epsilon": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay_rate": args.epsilon_decay,
            "alpha": args.alpha,
            "buffer_size": args.buffer_size,
            "min_replay_size": args.min_replay_size,
            "target_update_freq": args.target_update_freq
        }
        return DQNAgent(**common_params, **dqn_params)

def load_model_if_needed(agent, args, logger):
    """Load a pre-trained model for DQN if specified."""
    if args.algo == "dqn" and args.load_model_path:
        try:
            agent.load_model(args.load_model_path)
            agent.set_epsilon_for_eval()
            logger.info(f"Model loaded from {args.load_model_path}. Epsilon set to {agent.epsilon}.")
        except FileNotFoundError:
            logger.warning(f"No model found at {args.load_model_path}, starting from scratch.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Starting from scratch.")

def run_episode(env, agent, args, logger, episode, total_steps=0):
    """Run a single training episode and return relevant metrics."""
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False

    while not done and episode_steps < args.max_episode_steps:
        if args.render_mode == "human":
            env.render()

        # Action selection
        action_data = agent.select_action(obs)
        if args.algo == "ppo":
            action, log_prob, value = action_data
        else:
            action = action_data

        # Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        if args.algo == "ppo":
            agent.store_transition(obs, action, log_prob, reward, done, value)
        else:  # DQN
            # DQNAgent internally handles adding to both buffers (goal and replay)
            agent.store_transition(obs, action, reward, next_obs, done)

        # Learning step (DQN learns every 4 steps if possible)
        if args.algo == "dqn" and total_steps % 4 == 0 and agent._can_learn():
            agent.learn()
        elif args.algo == "ppo" and done:
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
                _, last_value = agent.actor_critic(state_tensor)
                agent.compute_returns_and_advantages(last_value=last_value.item())
                agent.learn()

        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        total_steps += 1

        if done and args.algo == "dqn":
            termination_type = "terminated" if terminated else "truncated"
            logger.debug(f"Episode {episode} ended via {termination_type} at step {episode_steps}")
            break

    return episode_reward, episode_steps, total_steps

def log_progress(args, episode, all_episode_rewards, total_steps, agent, logger):
    """Log training progress at specified intervals."""
    if episode % args.log_interval == 0:
        recent_rewards = all_episode_rewards[-args.log_interval:]
        avg_reward = np.mean(recent_rewards)
        max_reward = np.max(recent_rewards)
        log_message = (f"Episode: {episode} | Avg Reward (last {args.log_interval} eps): {avg_reward:.2f} | "
                       f"Max Reward: {max_reward:.2f}")
        if args.algo == "dqn":
            log_message += f" | Total Steps: {total_steps} | Epsilon: {agent.epsilon:.3f}"
        logger.info(log_message)

def save_model_if_needed(agent, args, logger):
    """Save the model if a save path is provided."""
    if args.save_model_path:
        agent.save_model(args.save_model_path)
        logger.info(f"{args.algo.upper()} model saved to {args.save_model_path}")

def train(args, logger):
    """Unified training function for PPO and DQN."""
    set_global_seed(args.seed)
    env = initialize_environment(args)
    agent = initialize_agent(args, env.observation_space.shape[0])
    load_model_if_needed(agent, args, logger)

    logger.info(f"Starting {args.algo.upper()} training for {args.max_episodes} episodes...")
    all_episode_rewards = []
    total_steps = 0

    for episode in tqdm(range(args.max_episodes), desc=f"{args.algo.upper()} Training Episodes"):
        episode_reward, episode_steps, total_steps = run_episode(env, agent, args, logger, episode, total_steps)

        all_episode_rewards.append(episode_reward)
        if args.algo == "dqn" and episode > 50:
            agent.decay_epsilon_multiplicative()

        log_progress(args, episode, all_episode_rewards, total_steps, agent, logger)

    env.close()
    logger.info(f"{args.algo.upper()} Training complete.")
    save_model_if_needed(agent, args, logger)