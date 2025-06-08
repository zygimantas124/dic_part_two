import os
import torch
import random
import numpy as np
from agents.PPO import PPOAgent
from office.delivery_env import DeliveryRobotEnv


def set_global_seed(seed):
    random.seed(seed)  # Python built-in
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (if used)
    os.environ["PYTHONHASHSEED"] = str(seed)  # For consistent hashing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(42)

# Hyperparameters
MAX_EPISODE_LENGTH = 512
TOTAL_TIMESTEPS = 20480
ROLLOUT_LENGTH = 256  # Use full episodes for stable PPO updates
UPDATE_EPOCHS = 2  # Not too high to avoid overfitting bad data
BATCH_SIZE = 128
GAMMA = 0.99  # VERY IMPORTANT — long-term planning required
CLIP_EPS = 0.2

# Environment
env = DeliveryRobotEnv(config="open_office", show_walls=True, show_carpets=False, show_obstacles=True, render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n


# Agent
agent = PPOAgent(
    obs_dim=obs_dim,
    n_actions=n_actions,
    gamma=GAMMA,
    clip_eps=CLIP_EPS,
    update_epochs=UPDATE_EPOCHS,
    batch_size=BATCH_SIZE,
    device=None,  # Auto-select device
)

# Training Loop
obs, _ = env.reset()
episode_return = 0
episode_length = 0
episode_returns = []

timestep = 0

while timestep < TOTAL_TIMESTEPS:
    # ---- Rollout Phase ----
    for _ in range(ROLLOUT_LENGTH):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        agent.store_transition(obs, action, log_prob, reward, done, value)

        # === Add this line to render each frame ===
        env.render()

        episode_return += reward
        episode_length += 1
        timestep += 1

        force_reset = episode_length >= MAX_EPISODE_LENGTH

        if done or truncated or force_reset:
            obs, _ = env.reset()
            episode_returns.append(episode_return)
            print(f"Episode finished: Return={episode_return}, Length={episode_length} {'[FORCED RESET]' if force_reset else ''}")
            episode_return = 0
            episode_length = 0
        else:
            obs = next_obs

    # ---- Learning Phase ----
    last_value = agent.actor_critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device))[1].item()
    agent.compute_returns_and_advantages(last_value=last_value)
    agent.learn()

# Save model
agent.save_model("ppo_test.pth")

# Close environment
env.close()

# Final results
print("Training completed.")
print(f"Average return over last 10 episodes: {np.mean(episode_returns[-10:])}")
