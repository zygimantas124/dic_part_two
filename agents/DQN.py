import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Neural Network for approximating Q-values.
    """
    def __init__(self, input_dim, output_dim, size="small"):
        super().__init__()
        if size == "small":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        elif size == "large":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, output_dim)
            )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """
    Replay buffer to store experiences for DQN training.
    """
    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer.
        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            tuple: A tuple of tensors (states, actions, rewards, next_states, dones).
        """
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        
        # Convert numpy arrays to PyTorch tensors
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),  # Actions are indices
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),  # Dones as float for multiplication
        )

    def __len__(self):
        """
        Return the current size of the buffer.
        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent.
    """
    def __init__(self, obs_dim, n_actions, gamma, epsilon, epsilon_min,
                epsilon_decay_rate, alpha, batch_size, buffer_size,
                min_replay_size, target_update_freq, goal_buffer_size, goal_fraction,
                device=None, qnet_size="small"):
        """
        Initialize the DQNAgent.
        Args:
            obs_dim (int): Dimensionality of the observation space.
            n_actions (int): Number of possible actions.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimal exploration rate.
            epsilon_decay_rate (float): Multiplicative decay factor for epsilon per decay step.
            alpha (float): Learning rate for the optimizer.
            batch_size (int): Batch size for sampling from the replay buffer.
            buffer_size (int): Maximum capacity of the replay buffer.
            min_replay_size (int): Minimum number of experiences in buffer before training starts.
            target_update_freq (int): Frequency (in frames) for updating the target network.
            device (torch.device, optional): Device to run the networks on (e.g., 'cuda' or 'cpu').
            qnet_size (str): "small" or "large" architecture for the Q-network.
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_freq = int(target_update_freq)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Q-Network, Target Q-Network
        self.q_net = QNetwork(obs_dim, n_actions, size=qnet_size).to(self.device)
        self.target_q_net = QNetwork(obs_dim, n_actions, size=qnet_size).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Initialize target with Q-net weights
        self.target_q_net.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)

        # Replay Buffers
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.goal_buffer = ReplayBuffer(50000)  # Buffer for positive-reward transitions
        self.goal_buffer = ReplayBuffer(goal_buffer_size)
        self.goal_fraction = goal_fraction
        self.learn_step_counter = 0  # To track when to update target network

    def select_action(self, state, explore=True):
        """
        Select an action using an epsilon-greedy policy.
        Args:
            state (np.ndarray): Current state.
            explore (bool): Whether to use exploration (epsilon-greedy). If False, always exploit.
        Returns:
            int: The selected action index.
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore: random action
        else:
            # Exploit: action with the highest Q-value
            with torch.no_grad():  # No gradients required for action selection
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store an experience transition in the replay buffer and goal buffer if reward is positive.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        if reward > 0:
            self.goal_buffer.add(state, action, reward, next_state, done)

    def _can_learn(self):
        """
        Check if the agent has enough samples in the buffer to learn.
        """
        return len(self.replay_buffer) >= self.min_replay_size

    def learn(self):
        """
        Train the Q-Network using a batch of experiences from both the main replay buffer
        and the goal buffer to reinforce successful behavior.
        """
        if not self._can_learn():
            return None  # Not enough samples to learn

        # Split batch: sample from goal buffer and main buffer
        goal_fraction = 0.4 # 25% of batch from goal-reaching transitions
        n_goal = int(self.batch_size * self.goal_fraction)
        n_main = self.batch_size - n_goal

        # Sample with fallback if goal_buffer doesn't have enough data
        if len(self.goal_buffer) >= n_goal:
            goal_samples = self.goal_buffer.sample(n_goal)
        else:
            goal_samples = self.replay_buffer.sample(n_goal)

        main_samples = self.replay_buffer.sample(n_main)

        # Concatenate samples
        states = torch.cat([goal_samples[0], main_samples[0]], dim=0).to(self.device)
        actions = torch.cat([goal_samples[1], main_samples[1]], dim=0).to(self.device)
        rewards = torch.cat([goal_samples[2], main_samples[2]], dim=0).to(self.device)
        next_states = torch.cat([goal_samples[3], main_samples[3]], dim=0).to(self.device)
        dones = torch.cat([goal_samples[4], main_samples[4]], dim=0).to(self.device)

        # Calculate target Q-values
        with torch.no_grad():  # Target network calculations don't require gradients
            next_q_values_target = self.target_q_net(next_states)
            max_next_q_values = next_q_values_target.max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Get current Q-values for the chosen actions from the main Q-network
        current_q = self.q_net(states).gather(1, actions)

        # Calculate loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # Periodically update the target network
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-Network by copying the weights from the main Q-Network.
        """
        print(f"Updating target network at step {self.learn_step_counter + self.min_replay_size}")
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon_multiplicative(self):
        """
        Decay epsilon using a multiplicative factor.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def set_epsilon_for_eval(self, eval_epsilon: float = 0.0):
        """
        Sets epsilon to (almost) purely greedy.
        """
        self.epsilon = eval_epsilon

    def load_model(self, path):
        """
        Loads the Q-network weights from a file.
        """
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()  # Ensure target network is also updated
        print(f"Model loaded from {path}")

    def save_model(self, path):
        """
        Saves the Q-network weights to a file.
        """
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
