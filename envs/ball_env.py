import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class WhiteBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, n_angles: int, render_mode=None, width=500, height=500):
        self.width = width
        self.height = height
        self.radius = 20
        self.goal_radius = 20
        self.hole_radius = 25
        self.step_size = 5.0  # Constant movement per step

        # Discrete action space: only the angle (in degrees)
        self.n_angles = n_angles
        self.action_space = spaces.Discrete(self.n_angles)

        # Observation space: x, y, angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height, 360], dtype=np.float32),
        )

        self.start_pos = np.array([100.0, 250.0])
        self.goal_pos = np.array([400.0, 280.0])
        self.holes = [
            np.array([60.0, 150.0]), 
            np.array([250.0, 250.0]), 
            np.array([120.0, 380.0]),
            np.array([350.0, 50.0]),
        ]

        self.angle = 0.0  # Store angle of last action in degrees
        self.state = None
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        self.ball_image = None
        self.target_image = None
        self.background_image = None

        # To track visitation
        self.grid_resolution = 2  # grid cell size in pixels
        self.visit_counts = np.zeros((self.width // self.grid_resolution,
                              self.height // self.grid_resolution), dtype=np.int32)

        self.step_count = 0
        self.total_reward = 0.0

    def reset(self, seed=0, options=None):
        """Reset environment variables to initial state."""
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        self.angle = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        return np.append(self.state, self.angle), {}

    def step(self, action):
        """
        Take a step in the environment given the action (angle in degrees).
        Updates reward and potential doneness.
        """
        self.angle = action
        original_pos = self.state.copy()  # Store position before attempting to move

        # Convert angle to movement vector
        rad = np.radians(self.angle)
        dx = self.step_size * np.cos(rad)
        dy = self.step_size * np.sin(rad)
        movement = np.array([dx, dy]).flatten()

        # Create "potential" move first to check against obstacles
        potential_next_state = original_pos + movement
        potential_next_state_clipped = np.clip(
            potential_next_state,
            self.observation_space.low[:2],
            self.observation_space.high[:2]
        )

        done = False

        # Determine whether agent hit an obstacle
        hit_a_hole = False
        for hole_center in self.holes:
            agent_to_hole_dist = np.linalg.norm(potential_next_state_clipped - hole_center)
            agent_to_hole_contact = self.radius + self.hole_radius
            if agent_to_hole_dist < agent_to_hole_contact:
                hit_a_hole = True
                break
        
        # Reward agent based on action
        if hit_a_hole:
            reward = -50.0

        else:
            current_dist_to_goal = np.linalg.norm(self.state - self.goal_pos)
            # goal_proximity_reward = current_dist_to_goal / self.start_to_goal_dist
            # # Small stepwise penalty to reward shorter paths  TODO
            # reward = -1.0 * goal_proximity_reward
            reward = -1.0

            # Visited-area penalty
            grid_x, grid_y = self._get_grid_index(self.state)
            self.visit_counts[grid_x-1, grid_y-1] += 1  # Track visits in specified cell
            # First time visit rewarded, second time visit nullified, from then on penalised
            visit_penalty = -0.5 * (self.visit_counts[grid_x-1, grid_y-1] - 2)
            reward += max(visit_penalty, -10)

            self.state = potential_next_state_clipped
            if current_dist_to_goal < (self.goal_radius+self.radius):
                reward = 1000.0
                self.target_reached = True
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
    
    def _get_grid_index(self, pos):
        return (int(pos[0]) // self.grid_resolution,
                int(pos[1]) // self.grid_resolution)