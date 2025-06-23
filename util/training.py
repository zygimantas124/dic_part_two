import os
import json
import numpy as np
from tqdm import tqdm
from office.delivery_env import DeliveryRobotEnv
from agents.PPO import PPOAgent
from agents.DQN import DQNAgent
from util.helpers import set_global_seed
from office.components.env_configs import get_config


def initialize_environment(args):
    env_config = get_config(args.env_name)

    return DeliveryRobotEnv(
        custom_config=env_config,
        render_mode=args.render_mode,
        show_walls=args.show_walls,
        show_obstacles=args.show_obstacles,
        show_carpets=args.show_carpets,
        use_flashlight=args.use_flashlight,
        use_raycasting=args.use_raycasting,
        reward_step=args.reward_step,
        reward_collision=args.reward_collision,
        reward_delivery=args.reward_delivery,
        reward_carpet=args.reward_carpet,
    )


def initialize_agent(args, obs_dim, logger):
    """Initialize the appropriate agent (PPO or DQN) based on args.algo."""
    device = args.device
    common_params = {
        "obs_dim": obs_dim,
        "n_actions": args.n_actions,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "device": device,
        "epsilon": args.epsilon_start,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay_rate": args.epsilon_decay,
    }

    if args.algo == "ppo":
        ppo_params = {"clip_eps": args.eps_clip, "update_epochs": args.k_epochs}
        return PPOAgent(**common_params, **ppo_params, logger=logger)
    else:  # dqn
        dqn_params = {
            "alpha": args.alpha,
            "buffer_size": args.buffer_size,
            "min_replay_size": args.min_replay_size,
            "target_update_freq": args.target_update_freq,
            "goal_buffer_size": args.goal_buffer_size,
            "goal_fraction": args.goal_fraction,
        }
        return DQNAgent(**common_params, **dqn_params, logger=logger)


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


def run_episode(env, agent, args, logger, episode, total_steps=0, record=True, record_path=None, seen_termination=False):
    """Run a single training episode and optionally record transitions."""
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    trajectory = [] if record else None

    while not done and episode_steps < args.max_episode_steps:
        if args.render_mode == "human":
            env.render()

        action_data = agent.select_action(obs)
        action = action_data if args.algo == "dqn" else action_data[0]

        next_obs, reward, terminated, _, _ = env.step(action)
        done = terminated or episode_steps == args.max_episode_steps - 1

        if terminated:
            seen_termination = True

        if args.algo == "ppo":
            agent.store_transition(obs, action, action_data[1], reward, done, action_data[2], terminated)
        else:
            agent.store_transition(obs, action, reward, next_obs, done)

        if args.algo == "dqn" and total_steps % 4 == 0 and agent._can_learn():
            agent.learn()
        elif args.algo == "ppo" and seen_termination and done:
            agent.learn()

        if record:
            trajectory.append(
                {
                    "state": obs.tolist(),
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": next_obs.tolist(),
                    "done": bool(done),
                }
            )

        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        total_steps += 1

    if record and record_path:
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        termination_status = "terminated" if terminated else "truncated"
        logger.debug(f"Episode {episode} recorded ({termination_status}) at step {episode_steps}")

    return episode_reward, episode_steps, total_steps, seen_termination


def log_progress(args, episode, all_episode_rewards, total_steps, agent, logger):
    """Log training progress at specified intervals."""
    if episode % args.log_interval == 0:
        recent_rewards = all_episode_rewards[-args.log_interval :]
        avg_reward = np.mean(recent_rewards)
        max_reward = np.max(recent_rewards)
        log_message = (
            f"Episode: {episode} | Avg Reward (last {args.log_interval} eps): {avg_reward:.2f} | Max Reward: {max_reward:.2f}"
        )
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

    # --- Handle auto-save for evaluation ---
    temp_model_path = None
    if args.evaluate_after_training and not args.save_model_path:
        temp_model_path = "logs/temp_eval_model.pth"
        args.save_model_path = temp_model_path
        logger.info("No model save path specified. Will auto-save for evaluation then delete.")

    # --- Initialize environment ---
    env = initialize_environment(args)
    obs_dim = env.observation_space.shape[0]

    # --- Initialize agent ---
    agent = initialize_agent(args, obs_dim, logger)
    load_model_if_needed(agent, args, logger)

    logger.info(f"Starting {args.algo.upper()} training for {args.max_episodes} episodes...")
    logger.info(f"Environment: {args.env_name}")
    logger.info(f"Render mode: {args.render_mode}")
    logger.info(f"Raycasting: {args.use_raycasting} | Flashlight: {args.use_flashlight}")

    # save config
    config_dump = {
        "env_name": args.env_name,
        "show_walls": args.show_walls,
        "show_obstacles": args.show_obstacles,
        "show_carpets": args.show_carpets,
        "use_raycasting": args.use_raycasting,
        "use_flashlight": args.use_flashlight,
        "render_mode": args.render_mode,
        "seed": args.seed,
        "algo": args.algo,
        "reward_step": args.reward_step,
        "reward_collision": args.reward_collision,
        "reward_delivery": args.reward_delivery,
        "reward_carpet": args.reward_carpet,
    }

    os.makedirs("./logs", exist_ok=True)

    # Save environment config before training begins
    with open("./logs/env_config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    # --- Main training loop ---
    all_episode_rewards = []
    total_steps = 0
    seen_termination = False

    for episode in tqdm(range(args.max_episodes), desc=f"{args.algo.upper()} Training Episodes"):
        log_path = f"./logs/episode_{episode}.json"
        episode_reward, episode_steps, total_steps, seen_termination = run_episode(
            env, agent, args, logger, episode, total_steps, record_path=log_path, seen_termination=seen_termination
        )

        all_episode_rewards.append(episode_reward)

        if args.algo == "dqn" and episode > args.warmstart:
            agent.decay_epsilon_multiplicative()

        log_progress(args, episode, all_episode_rewards, total_steps, agent, logger)

    env.close()
    logger.info(f"{args.algo.upper()} Training complete.")
    save_model_if_needed(agent, args, logger)

    # --- Handle evaluation and cleanup ---
    if args.evaluate_after_training:
        logger.info("Starting post-training evaluation...")
        success = evaluate(args, logger)
        
        # Clean up temporary model
        if temp_model_path and os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            logger.info(f"Temporary model {temp_model_path} deleted after evaluation.")
            
        if not success:
            logger.warning("Evaluation failed, but training completed successfully.")


def evaluate(args, logger):
    """Evaluate a trained model using the evaluate.py module."""
    try:
        from evaluate import parse_eval_args, evaluate_agent
        from types import SimpleNamespace
        
        # Create evaluation arguments
        eval_args = SimpleNamespace()
        
        # Determine model path
        if hasattr(args, 'save_model_path') and args.save_model_path:
            eval_args.model_path = args.save_model_path
        elif hasattr(args, 'load_model_path') and args.load_model_path:
            eval_args.model_path = args.load_model_path
        else:
            # Let evaluate.py auto-detect
            import glob
            pt_files = glob.glob("logs/*.pth")
            if pt_files:
                eval_args.model_path = max(pt_files, key=os.path.getmtime)
            else:
                logger.error("No model file found for evaluation.")
                return False
        
        # Map arguments
        eval_args.n_episodes = args.eval_episodes
        eval_args.render_mode = getattr(args, 'eval_render_mode', None) or getattr(args, 'render_mode', None)
        eval_args.n_actions = args.n_actions
        eval_args.eval_epsilon = args.eval_epsilon
        eval_args.max_episode_steps = args.max_episode_steps
        eval_args.render_delay = getattr(args, 'eval_render_delay', 0.03)
        
        logger.info(f"Starting evaluation with {eval_args.n_episodes} episodes")
        evaluate_agent(eval_args)
        return True
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return False