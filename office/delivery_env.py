import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from office.render import render_environment
from office.components.tables import get_target_tables
from office.components.obstacles import get_carpets, get_people, get_furniture
from office.components.walls import get_walls
from office.components.env_configs import get_config, EnvironmentConfig

import math
from office.raycasting import (
    cast_rays,
    cast_rays_with_hits,
    cast_cone_rays,
    _intersect_ray_rect,
    RAY_OFFSET
)


class DeliveryRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config="simple", render_mode=None, show_walls=True, show_obstacles=True, show_carpets=True, custom_config=None, use_flashlight=False, use_raycasting=False):
        # Environment dimensions
        self.width = 800
        self.height = 600
        self.robot_radius = 10
        self.step_size = 30.0
        self.reward = 0.0

        # Orientation tracking in radians
        self.angle_rad = math.pi / 2  # Default: facing down

        # Toggle raycasting and flashlight rendering
        self.use_flashlight = use_flashlight
        self.use_raycasting = use_raycasting

        # Raycasting configuration
        self.ray_config = {
            'num_rays': 30,
            'cone_width_deg': 45,
            'max_distance': 300,
            'render_rays': 30
        }

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        # Visual and physical object flags
        self.show_walls = show_walls
        self.show_obstacles = show_obstacles
        self.show_carpets = show_carpets

        # Load environment configuration
        cfg = custom_config if custom_config else get_config(config)
        self.walls = get_walls(cfg.walls) if show_walls else []
        self.tables = get_target_tables(cfg.tables, scale=cfg.table_scale)
        self.carpets = get_carpets(cfg.carpets) if show_carpets else []

        self.obstacles = []
        if show_obstacles:
            self.obstacles.extend(get_people(cfg.people))
            self.obstacles.extend(get_furniture(cfg.furniture))

        # Robot starting position
        self.start_pos = np.array(cfg.start_pos if cfg.start_pos else [100, 500], dtype=np.float32)
        self.robot_pos = self.start_pos.copy()
        self.delivered_tables = set()

        # Angular movement setup using 8 directions
        self.n_directions = 8
        angles = np.linspace(0, 2 * math.pi, self.n_directions, endpoint=False)
        self.directions = np.array([[math.cos(a), math.sin(a)] for a in angles], dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_directions)

        # Observation space: position, orientation, table statuses, (optional) ray distances
        obs_dim = 4 + len(self.tables)  # x, y, cos(angle), sin(angle), table status
        if self.use_raycasting:
            obs_dim += self.ray_config['num_rays']

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, 0.0, dtype=np.float32),
            high=np.full(obs_dim, 1.0, dtype=np.float32),
            dtype=np.float32
        )

        self.total_reward = 0.0
        self.step_count = 0

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.robot_pos = self.start_pos.copy()
        self.angle_rad = math.pi / 2
        self.total_reward = 0.0
        self.step_count = 0
        self.delivered_tables = set()
        return self._get_obs(), {}

    def step(self, action):
        # Apply movement based on discrete angular direction
        movement = self.directions[action] * self.step_size
        self.angle_rad = math.atan2(movement[1], movement[0])
        proposed_pos = self.robot_pos + movement

        # Check if proposed position is out of bounds
        hit_boundary = (
            proposed_pos[0] < self.robot_radius or
            proposed_pos[0] > self.width - self.robot_radius or
            proposed_pos[1] < self.robot_radius or
            proposed_pos[1] > self.height - self.robot_radius
        )

        reward = -0.1  # Step penalty
        if hit_boundary or self._check_collision(proposed_pos):
            reward -= 1.0  # Penalty for hitting boundary or collision
        else:
            self.robot_pos = proposed_pos
            if self._on_carpet():
                reward -= 0.2
            reward += self._check_table_delivery()

        self.total_reward += reward
        self.step_count += 1
        done = len(self.delivered_tables) == len(self.tables)

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Position + orientation
        obs = [self.robot_pos[0] / self.width, self.robot_pos[1] / self.height,
               math.cos(self.angle_rad), math.sin(self.angle_rad)]

        # Table delivery statuses
        status = [1 if i in self.delivered_tables else 0 for i in range(len(self.tables))]
        obs.extend(status)

        # Add ray distances if enabled
        if self.use_raycasting:
            obs.extend(self.get_cone_ray_distances())

        return np.array(obs, dtype=np.float32)

    def get_cone_ray_distances(self):
        """Cast rays in a cone and return normalized distances to nearest obstacles."""
        px, py = float(self.robot_pos[0]), float(self.robot_pos[1])
        center_ang = self.angle_rad
        cone_rad = math.radians(self.ray_config['cone_width_deg'])

        pts, types = cast_cone_rays(
            robot_pos=(px, py),
            wall_rects=self.walls + self.tables,
            circle_obstacles=self.obstacles,
            center_angle=center_ang,
            cone_width=cone_rad,
            num_rays=self.ray_config['num_rays'],
            max_distance=self.ray_config['max_distance'],
            robot_radius=self.robot_radius
        )

        ray_distances = []
        half_cone = cone_rad / 2
        for i, (ix, iy) in enumerate(pts):
            dist = math.hypot(ix - px, iy - py) if types[i] in ("wall", "circle") else self.ray_config['max_distance']
            ray_ang = center_ang - half_cone + i * (cone_rad / (self.ray_config['num_rays'] - 1))
            dx, dy = math.cos(ray_ang), math.sin(ray_ang)
            for (cx, cy, cw, ch) in self.carpets:
                t = _intersect_ray_rect(px, py, dx, dy, cx, cy, cw, ch)
                if t is not None and 0 <= t < dist:
                    dist = t
            ray_distances.append(min(dist / self.ray_config['max_distance'], 1.0))

        self._last_ray_distances = ray_distances
        return ray_distances

    def _check_collision(self, pos):
        """Check if the given position collides with walls, obstacles, or tables."""
        x, y = pos
        for wx, wy, ww, wh in self.walls:
            if wx - self.robot_radius <= x <= wx + ww + self.robot_radius and wy - self.robot_radius <= y <= wy + wh + self.robot_radius:
                return True
        for ox, oy, orad in self.obstacles:
            if np.linalg.norm(pos - np.array([ox, oy])) < orad + self.robot_radius:
                return True
        for tx, ty, tw, th in self.tables:
            m = self.robot_radius
            if (tx + m <= x <= tx + tw - m) and (ty + m <= y <= ty + th - m):
                return True
        return False

    def _on_carpet(self):
        """Check whether the robot is currently on a carpet."""
        x, y = self.robot_pos
        return any(cx <= x <= cx + cw and cy <= y <= cy + ch for cx, cy, cw, ch in self.carpets)

    def _check_table_delivery(self):
        """Check if the robot is within delivery range of any table."""
        reward = 0
        margin = self.robot_radius * 4
        for i, (tx, ty, tw, th) in enumerate(self.tables):
            if i not in self.delivered_tables:
                rx, ry = self.robot_pos
                if (tx - margin <= rx <= tx + tw + margin) and (ty - margin <= ry <= ty + th + margin):
                    self.delivered_tables.add(i)
                    reward += 50
        return reward

    def render(self):
        """Render the environment and optionally visualize rays."""
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

        if not self.use_raycasting:
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

        # Visualize rays
        px, py = self.robot_pos
        center_ang = self.angle_rad
        cone_rad = math.radians(self.ray_config['cone_width_deg'])

        pts, _ = cast_cone_rays(
            robot_pos=(px, py),
            wall_rects=self.walls + self.tables,
            circle_obstacles=self.obstacles,
            center_angle=center_ang,
            cone_width=cone_rad,
            num_rays=self.ray_config['render_rays'],
            max_distance=self.ray_config['max_distance'],
            robot_radius=self.robot_radius
        )

        if self.use_flashlight:
            poly_pts = [(int(ix), int(iy)) for (ix, iy) in pts]
            overlay = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            if poly_pts:
                pygame.draw.polygon(overlay, (0, 0, 0, 0), poly_pts)
                pygame.draw.polygon(self.window, (255, 255, 200), poly_pts)
            self.window.blit(overlay, (0, 0))
        else:
            for (ix, iy) in pts:
                pygame.draw.line(self.window, (200, 200, 80), (int(px), int(py)), (int(ix), int(iy)), 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close the rendering window and cleanup pygame."""
        if self.window:
            pygame.quit()
            pygame.font.quit()
            self.window = None

    @property
    def ray_distance_dict(self, scaled=True):
        """Get the current ray distances as a dictionary for debugging/visualization."""
        if not hasattr(self, '_last_ray_distances'):
            return {}
        
        ray_dict = {}
        for i, dist in enumerate(self._last_ray_distances):
            if scaled:
                # Scale to [0, 1] range
                dist = min(dist / self.ray_config['max_distance'], 1.0)
            else:
                # Use raw distance
                dist = max(0, min(dist, self.ray_config['max_distance']))
            ray_dict[f"ray_{i}"] = dist * self.ray_config['max_distance']  # Use config
        return ray_dict