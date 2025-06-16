import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_size=128):
        super().__init__()

        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_size, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.terminated = []

    def add(self, state, action, log_prob, reward, done, value, terminated=False):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.terminated.append(terminated)

    def clear(self):
        self.__init__()


class GoodBuffer:
    def __init__(self, max_size=512):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.terminations = []

    def add_batch(self, states, actions, log_probs, returns, advantages, terminations=None):
        # Append
        self.states.append(states)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.returns.append(returns)
        self.advantages.append(advantages)
        self.terminations.append(terminations if terminations is not None else torch.zeros_like(actions, dtype=torch.bool))

        # Enforce max size with FIFO
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.returns.pop(0)
            self.advantages.pop(0)
            self.terminations.pop(0)

    def sample(self, n_batches=1):
        if not self.states:
            return None

        indices = np.random.choice(len(self.states), size=n_batches, replace=True)

        states = torch.cat([self.states[i] for i in indices], dim=0)
        actions = torch.cat([self.actions[i] for i in indices], dim=0)
        log_probs = torch.cat([self.log_probs[i] for i in indices], dim=0)
        returns = torch.cat([self.returns[i] for i in indices], dim=0)
        advantages = torch.cat([self.advantages[i] for i in indices], dim=0)
        terminations = torch.cat([self.terminations[i] for i in indices], dim=0)

        return states, actions, log_probs, returns, advantages, terminations


class PPOAgent:
    def __init__(self, obs_dim, n_actions, gamma, clip_eps, update_epochs, batch_size, device, logger=None):
        self.gamma = gamma  # Discount factor, typically around 0.99
        self.clip_eps = clip_eps  # Clipping epsilon for PPO, typically around 0.2
        self.update_epochs = update_epochs  # Number of epochs to update the policy per batch, typically 10
        self.batch_size = batch_size  # Batch size for training, typically 64 or 128

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.actor_critic = ActorCriticNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)

        self.buffer = RolloutBuffer()
        self.logger = logger or logging.getLogger(__name__)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, state_value = self.actor_critic(state_tensor)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action, log_prob, reward, done, value, terminated=False):
        self.buffer.add(state, action, log_prob, reward, done, value, terminated)

    def compute_returns_and_advantages(self, last_value=0, normalize_advantage=True):
        returns = []
        advantages = []
        G = last_value
        A = 0
        values = self.buffer.values + [last_value]

        # TD error → tells the critic how surprised it should be.
        # Discounted return → tells what really happened → used to train the critic.
        # GAE → gives the actor a “smoothed” signal about which actions were good or bad → improves actor learning.

        for t in reversed(range(len(self.buffer.rewards))):
            delta = self.buffer.rewards[t] + self.gamma * values[t + 1] * (1 - self.buffer.dones[t]) - values[t]  # TD
            A = delta + self.gamma * A * (1 - self.buffer.dones[t])  # GAE
            G = self.buffer.rewards[t] + self.gamma * G * (1 - self.buffer.dones[t])  # Discounted Return
            returns.insert(0, G)
            advantages.insert(0, A)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        if normalize_advantage:  # Improve learning stability by normalizing advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def learn(self):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        terminations = torch.tensor(self.buffer.terminated, dtype=torch.bool).to(self.device)

        returns, advantages = self.compute_returns_and_advantages()

        if not hasattr(self, "good_buffer"):
            self.good_buffer = GoodBuffer()

        for _ in range(self.update_epochs):
            idxs = np.arange(len(states))
            np.random.shuffle(idxs)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_terminations = terminations[batch_idx]

                # Store good batches
                if batch_terminations.any():
                    self.good_buffer.add_batch(batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages)

                # Sample from good buffer and add to current batch
                sample_ratio = 0.5
                sample_size = int(len(batch_states) * sample_ratio)

                good_sample = self.good_buffer.sample(1)

                if good_sample:
                    g_states, g_actions, g_log_probs, g_returns, g_advantages, g_terminations = good_sample

                    total_good = g_states.size(0)

                    if total_good > sample_size:
                        perm = torch.randperm(total_good)[:sample_size]
                        g_states = g_states[perm]
                        g_actions = g_actions[perm]
                        g_log_probs = g_log_probs[perm]
                        g_returns = g_returns[perm]
                        g_advantages = g_advantages[perm]
                        g_terminations = g_terminations[perm]

                    # Concatenate to current batch
                    batch_states = torch.cat([batch_states, g_states], dim=0)
                    batch_actions = torch.cat([batch_actions, g_actions], dim=0)
                    batch_old_log_probs = torch.cat([batch_old_log_probs, g_log_probs], dim=0)
                    batch_returns = torch.cat([batch_returns, g_returns], dim=0)
                    batch_advantages = torch.cat([batch_advantages, g_advantages], dim=0)
                    batch_terminations = torch.cat([batch_terminations, g_terminations], dim=0)

                # PPO forward and loss
                action_probs, state_values = self.actor_critic(batch_states)
                dist = torch.distributions.Categorical(action_probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(state_values.squeeze(-1), batch_returns)

                entropy_coef = 0.05
                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


# python main.py --device cpu --algo ppo --gamma 0.995 --max_episodes 1000 --max_episode_steps 4096 --k_epochs 10 --batch_size 512 --eps_clip 0.2 --seed 42 --epsilon_start 1 --epsilon_min 0.01 --epsilon_decay 0.95
