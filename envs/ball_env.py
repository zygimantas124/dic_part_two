import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class WhiteBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, width=500, height=500):
        self.width = width
        self.height = height
        self.radius = 10
        self.goal_radius = 15
        self.hole_radius = 20

        # Define action space: dx and dy continuous movement
        self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

        # Observation space: position of the ball
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
        )

        self.start_pos = np.array([50.0, 50.0])
        self.goal_pos = np.array([450.0, 450.0])
        self.holes = [np.array([200.0, 200.0]), np.array([300.0, 100.0])]

        self.state = None
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = np.clip(
            self.state + action, self.observation_space.low, self.observation_space.high
        )

        done = False
        reward = -0.01  # small penalty for each step

        if np.linalg.norm(self.state - self.goal_pos) < self.goal_radius:
            reward = 1.0
            done = True
        elif any(
            np.linalg.norm(self.state - hole) < self.hole_radius for hole in self.holes
        ):
            reward = -1.0
            done = True

        return self.state, reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))  # Black background

        # Draw goal
        pygame.draw.circle(
            self.window, (0, 255, 0), self.goal_pos.astype(int), self.goal_radius
        )

        # Draw holes
        for hole in self.holes:
            pygame.draw.circle(
                self.window, (255, 0, 0), hole.astype(int), self.hole_radius
            )

        # Draw ball
        pygame.draw.circle(
            self.window, (255, 255, 255), self.state.astype(int), self.radius
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


env = WhiteBallEnv(render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = np.random.uniform(-5, 5, size=(2,))
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.close()
