import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Assumed external dependencies
from office.render import render_environment
from office.components.tables import get_target_tables
from office.components.obstacles import get_carpets, get_people, get_furniture
from office.components.walls import get_walls
from office.components.env_configs import get_config, EnvironmentConfig


class DeliveryRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config="simple", render_mode=None, show_walls=True, show_obstacles=True, show_carpets=True, custom_config=None):
        self.width = 800
        self.height = 600
        self.robot_radius = 10
        self.step_size = 10.0  # Reduced for finer control
        self.max_steps = 50000  # TODO: link this to argparse argument
        self.action = None
        self.angle = 0.0  # Track orientation for observation

        self.show_walls = show_walls
        self.show_obstacles = show_obstacles
        self.show_carpets = show_carpets

        # Load configuration
        if custom_config:
            cfg = custom_config
        else:
            cfg = get_config(config)

        # Load components
        self.walls = get_walls(cfg.walls) if show_walls else []
        self.tables = get_target_tables(cfg.tables, scale=cfg.table_scale)
        self.carpets = get_carpets(cfg.carpets) if show_carpets else []
        obstacles = []
        if show_obstacles:
            obstacles.extend(get_people(cfg.people))
            obstacles.extend(get_furniture(cfg.furniture))
        self.obstacles = obstacles

        # Start position
        if cfg.start_pos:
            self.start_pos = np.array(cfg.start_pos, dtype=np.float32)
        else:
            self.start_pos = np.array([100, 500], dtype=np.float32)

        self.robot_pos = self.start_pos.copy()
        self.delivered_tables = set()

        # Action space: discrete directions
        self.n_directions = 8
        angles = np.linspace(0, 360, self.n_directions, endpoint=False)
        self.directions = np.array([
            [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angles
        ], dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_directions)

        # Observation space: [norm_x, norm_y, cos(angle), sin(angle), min_obs_dist, on_carpet] + table_statuses
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0] + [0.0] * len(self.tables), dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * len(self.tables), dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        self.total_reward = 0.0
        self.step_count = 0

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.robot_pos = self.start_pos.copy()
        self.total_reward = 0.0
        self.step_count = 0
        self.delivered_tables = set()
        self.action = None
        self.angle = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.action = action
        self.angle = np.degrees(np.arctan2(self.directions[action][1], self.directions[action][0]))  # Update orientation
        movement = self.directions[action] * self.step_size
        proposed_pos = self.robot_pos + movement

        # Boundary check: block movement if out of bounds
        hit_wall = False
        new_pos = proposed_pos
        if (proposed_pos[0] < self.robot_radius or
            proposed_pos[0] > self.width - self.robot_radius or
            proposed_pos[1] < self.robot_radius or
            proposed_pos[1] > self.height - self.robot_radius):
            new_pos = self.robot_pos  # Block movement
            hit_wall = True

        reward = -0.01  # Small step penalty
        if hit_wall:
            reward -= 1.0  # Boundary penalty

        if not hit_wall and not self._check_collision(new_pos):
            self.robot_pos = new_pos
            if self._on_carpet():
                reward -= 0.1  # Reduced carpet penalty
            reward += self._check_table_delivery()
        else:
            reward -= 1.0  # Collision penalty

        # Clip rewards to stabilize training
        reward = np.clip(reward, -10.0, 10.0)

        self.total_reward += reward
        self.step_count += 1

        terminated = len(self.delivered_tables) == len(self.tables)
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        status = [1 if i in self.delivered_tables else 0 for i in range(len(self.tables))]
        norm_x = self.robot_pos[0] / self.width
        norm_y = self.robot_pos[1] / self.height
        angle_rad = np.radians(self.angle)
        return np.array([norm_x, norm_y, np.cos(angle_rad), np.sin(angle_rad)] + status, dtype=np.float32)

    def _check_collision(self, pos):
        x, y = pos
        # Check walls
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            if (
                wall_x - self.robot_radius <= x <= wall_x + wall_w + self.robot_radius
                and wall_y - self.robot_radius <= y <= wall_y + wall_h + self.robot_radius
            ):
                return True

        # Check obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            if np.linalg.norm(pos - np.array([obs_x, obs_y])) < obs_r + self.robot_radius:
                return True

        # Check table collisions (more lenient)
        collision_margin = self.robot_radius * 0.5  # Reduced for easier delivery
        for tx, ty, tw, th in self.tables:
            collision_x = tx + collision_margin
            collision_y = ty + collision_margin
            collision_w = tw - 2 * collision_margin
            collision_h = th - 2 * collision_margin
            if collision_w > 0 and collision_h > 0:
                if (
                    collision_x - self.robot_radius <= x <= collision_x + collision_w + self.robot_radius
                    and collision_y - self.robot_radius <= y <= collision_y + collision_h + self.robot_radius
                ):
                    return True
        return False

    def _on_carpet(self):
        x, y = self.robot_pos
        for cx, cy, cw, ch in self.carpets:
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                return True
        return False

    def _check_table_delivery(self):
        reward = 0
        delivery_margin = self.robot_radius * 2  # Reduced for tighter delivery
        for i, (tx, ty, tw, th) in enumerate(self.tables):
            if i not in self.delivered_tables:
                rx, ry = self.robot_pos
                expanded_x = tx - delivery_margin
                expanded_y = ty - delivery_margin
                expanded_w = tw + 2 * delivery_margin
                expanded_h = th + 2 * delivery_margin
                if (expanded_x <= rx <= expanded_x + expanded_w) and (expanded_y <= ry <= expanded_y + expanded_h):
                    self.delivered_tables.add(i)
                    reward += 100  
        return reward

    def render(self):
        if self.render_mode != "human":
            return
        if pygame.get_init():
            pygame.event.pump()
        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        render_environment(self)

    def close(self):
        if self.window:
            pygame.quit()
            pygame.font.quit()
            self.window = None