import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class WhiteBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, n_angles: int, render_mode=None, width=150, height=150):
        self.width = width
        self.height = height
        self.radius = 20
        self.subgoal_radius = 5
        self.goal_radius = 10
        self.hole_radius = 10
        self.step_size = 5.0  # Constant movement per step

        # Discrete action space: only the angle (in degrees)
        self.n_angles = n_angles
        self.action_space = spaces.Discrete(self.n_angles)

        # Observation space: x, y, angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height, 360], dtype=np.float32),
        )

        self.start_pos = np.array([25.0, 25.0])
        self.subgoal1_pos = np.array([40.0, 90.0])
        self.subgoal2_pos = np.array([80.0, 115.0])
        self.goal_pos = np.array([130.0, 130.0])
        self.holes = [np.array([80.0, 80.0]), np.array([120.0, 30.0])]

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
        self.cumulative_obstacle_multiplier = 1.0

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        self.angle = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        return np.append(self.state, self.angle), {}

    def step(self, action):
        self.angle = action
        original_pos = self.state.copy()  # Store position before attempting to move

        # Convert angle to movement vector
        rad = np.radians(self.angle)
        dx = self.step_size * np.cos(rad)
        dy = self.step_size * np.sin(rad)
        movement = np.array([dx, dy])

        # Create "potential" move first to check against obstacles
        potential_next_state = original_pos + movement
        potential_next_state_clipped = np.clip(
            potential_next_state,
            self.observation_space.low[:2],
            self.observation_space.high[:2]
        )[:, 0]

        done = False

        # Determine whether agent hit an obstacle
        hit_a_hole = False
        for hole_center in self.holes:
            if np.linalg.norm(potential_next_state_clipped - hole_center) < self.hole_radius:
                hit_a_hole = True
                break
        
        # Reward agent based on action
        if hit_a_hole:
            self.cumulative_obstacle_multiplier = 1.0  # Reset hole multiplier
            reward = -20.0

        else:
            self.cumulative_obstacle_multiplier *= 0.99  # Decrease stepsize penalty when not hitting hole
            reward = min(-1  * self.cumulative_obstacle_multiplier, -0.01)  # Ensure stepsize penalty remains

            # Visited-area penalty
            grid_x, grid_y = self._get_grid_index(self.state)
            self.visit_counts[grid_x, grid_y] += 1  # Track visits in specified cell
            # First time visit rewarded, second time visit nullified, from then on penalised
            visit_penalty = -0.5 * (self.visit_counts[grid_x, grid_y] - 2)
            reward += visit_penalty

            self.state = potential_next_state_clipped
            if np.linalg.norm(self.state - self.subgoal1_pos) < self.subgoal_radius:
                reward = 5000.0
            if np.linalg.norm(self.state - self.subgoal2_pos) < self.subgoal_radius:
                reward = 5000.0
            if np.linalg.norm(self.state - self.goal_pos) < self.goal_radius:
                reward = 100000.0
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

            self.subgoal_image1 = pygame.image.load("envs/sprites/dollar.png")
            self.subgoal_image1 = pygame.transform.scale(
                self.subgoal_image1, (2 * self.goal_radius, 2 * self.goal_radius)
            )
            self.subgoal_image2 = self.subgoal_image1.copy()

            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 20)

        self.window.blit(self.background_image, (0, 0))

        # Draw target
        target_pos = self.goal_pos.astype(int) - self.goal_radius
        subgoal1_pos = self.subgoal1_pos.astype(int) - self.subgoal_radius
        subgoal2_pos = self.subgoal2_pos.astype(int) - self.subgoal_radius
        self.window.blit(self.target_image, target_pos)
        self.window.blit(self.subgoal_image1, subgoal1_pos)
        self.window.blit(self.subgoal_image2, subgoal2_pos)

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


# Run the environment manually with random angle actions
if __name__ == "__main__":
    env = WhiteBallEnv(render_mode="human")
    obs, _ = env.reset()

    done = False
    while not done:
        action = np.random.uniform(0, 360, size=(1,))
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()
