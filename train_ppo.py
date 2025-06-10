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
MAX_EPISODE_LENGTH = 2000
TOTAL_EPISODES = 200  # Train for 100 episodes
UPDATE_EVERY_N_EPISODES = 1  # Update policy after every episode
UPDATE_EPOCHS = 2  # Not too high to avoid overfitting bad data
BATCH_SIZE = 128
GAMMA = 0.99  # VERY IMPORTANT — long-term planning required
CLIP_EPS = 0.2

# Environment
env = DeliveryRobotEnv(config="simple", show_walls=True, show_carpets=False, show_obstacles=True)
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
episode_returns = []
obs, _ = env.reset()

for episode in range(TOTAL_EPISODES):
    episode_return = 0
    episode_length = 0
    done = False
    truncated = False

    # ---- Rollout Phase (collect one full episode) ----
    while not (done or truncated or episode_length >= MAX_EPISODE_LENGTH):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        agent.store_transition(obs, action, log_prob, reward, done, value)

        # Render each frame
        env.render()

        episode_return += reward
        episode_length += 1
        obs = next_obs

    # Store episode results
    episode_returns.append(episode_return)
    print(f"Episode {episode + 1}/{TOTAL_EPISODES}: Return={episode_return}, Length={episode_length} {'[FORCED RESET]' if episode_length >= MAX_EPISODE_LENGTH else ''}")

    # ---- Learning Phase ----
    if (episode + 1) % UPDATE_EVERY_N_EPISODES == 0:
        # Compute last value for the final state
        last_value = 0
        if not done and not truncated:  # Bootstrap if episode was forcibly reset
            last_value = agent.actor_critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device))[1].item()
        agent.compute_returns_and_advantages(last_value=last_value)
        agent.learn()

    # Reset environment for the next episode
    obs, _ = env.reset()

# Save model
agent.save_model("ppo_test.pth")

# Close environment
env.close()

# Final results
print("Training completed.")
print(f"Average return over last 10 episodes: {np.mean(episode_returns[-10:]) if episode_returns else 0}")