import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class WhiteBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None, width=500, height=500):
        self.width = width
        self.height = height
        self.radius = 30
        self.goal_radius = 15
        self.hole_radius = 20

        self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height, 360], dtype=np.float32),
        )

        self.start_pos = np.array([50.0, 50.0])
        self.goal_pos = np.array([450.0, 450.0])
        self.holes = [np.array([200.0, 200.0]), np.array([300.0, 100.0])]

        self.angle = 0.0
        self.state = None
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        self.ball_image = None
        self.target_image = None
        self.background_image = None

        self.step_count = 0
        self.total_reward = 0.0

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        self.angle = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        return np.append(self.state, self.angle), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if np.linalg.norm(action) > 1e-6:
            self.angle = (np.degrees(np.arctan2(action[1], action[0])) + 360) % 360

        self.state = np.clip(
            self.state + action,
            self.observation_space.low[:2],
            self.observation_space.high[:2],
        )

        done = False
        reward = -0.01

        if np.linalg.norm(self.state - self.goal_pos) < self.goal_radius:
            reward = 1.0
            done = True
        elif any(
            np.linalg.norm(self.state - hole) < self.hole_radius for hole in self.holes
        ):
            reward = -1.0
            done = True

        self.step_count += 1
        self.total_reward += reward

        obs = np.append(self.state, self.angle)
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

            self.background_image = pygame.image.load("envs/sprites/bg.jpg")
            self.background_image = pygame.transform.scale(
                self.background_image, (self.width, self.height)
            )

            self.ball_image = pygame.image.load("envs/sprites/trump.png")
            self.ball_image = pygame.transform.scale(
                self.ball_image, (2 * self.radius, 2 * self.radius)
            )

            self.target_image = pygame.image.load("envs/sprites/dollar.png")
            self.target_image = pygame.transform.scale(
                self.target_image, (2 * self.goal_radius, 2 * self.goal_radius)
            )

            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 20)

        self.window.blit(self.background_image, (0, 0))

        # Draw target
        target_pos = self.goal_pos.astype(int) - self.goal_radius
        self.window.blit(self.target_image, target_pos)

        # Draw holes
        for hole in self.holes:
            pygame.draw.circle(
                self.window, (255, 0, 0), hole.astype(int), self.hole_radius
            )

        # Rotate and draw the ball
        rotated_ball = pygame.transform.rotate(self.ball_image, -self.angle)
        rotated_rect = rotated_ball.get_rect(center=self.state.astype(int))
        self.window.blit(rotated_ball, rotated_rect.topleft)

        # Draw text
        step_text = self.font.render(f"Steps: {self.step_count}", True, (255, 255, 255))
        reward_text = self.font.render(
            f"Reward: {self.total_reward:.2f}", True, (255, 255, 0)
        )
        self.window.blit(step_text, (10, 10))
        self.window.blit(reward_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


# Run the environment
env = WhiteBallEnv(render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = np.random.uniform(-5, 5, size=(2,))
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.close()
