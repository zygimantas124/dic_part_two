import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# does not work if starting outside of office
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

    def __init__(self, config="simple", render_mode=None, show_walls=True, show_obstacles=True, show_carpets=True, custom_config=None, use_flashlight=False):
        self.width = 800
        self.height = 600
        self.robot_radius = 10
        self.step_size = 10.0
        self.reward = 0.0
        self.action = None

        self.show_walls = show_walls
        self.show_obstacles = show_obstacles
        self.show_carpets = show_carpets

        self.use_flashlight = use_flashlight
        # Track last movement direction in radians - default facing "down"
        self.last_move_dir = math.pi / 2

        # Load configuration
        if custom_config:
            cfg = custom_config
        else:
            cfg = get_config(config)

        # Load components based on configuration
        self.walls = get_walls(cfg.walls) if show_walls else []
        self.tables = get_target_tables(cfg.tables, scale=cfg.table_scale)
        self.carpets = get_carpets(cfg.carpets) if show_carpets else []

        obstacles = []
        if show_obstacles:
            obstacles.extend(get_people(cfg.people))
            obstacles.extend(get_furniture(cfg.furniture))
        self.obstacles = obstacles

        # Use configuration's start_pos if provided
        if cfg.start_pos:
            self.start_pos = np.array(cfg.start_pos)
        else:
            self.start_pos = np.array([100, 500])  # default

        self.robot_pos = self.start_pos.copy()
        self.delivered_tables = set()

        self.directions = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),  # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0]),  # Right
            4: np.array([-1, -1]),  # Up-Left
            5: np.array([1, -1]),  # Up-Right
            6: np.array([-1, 1]),  # Down-Left
            7: np.array([1, 1]),  # Down-Right
        }
        self.action_space = spaces.Discrete(len(self.directions))

        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * len(self.tables) + [0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height] + [1] * len(self.tables) + [1, 1, 1, 1], dtype=np.float32),
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
        self.consecutive_collisions = 0 # DELETE before submission if not needed
        self.step_count = 0
        self.delivered_tables = set()
        return self._get_obs(), {}

    def step(self, action):
        self.action = action
        movement = (self.directions[self.action] * self.step_size).flatten()

        # Track movement direction for ray casting
        dx, dy = float(movement[0]), float(movement[1])
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self.last_move_dir = math.atan2(dy, dx)

        new_pos = self.robot_pos + movement
        new_pos = np.clip(
            new_pos, [self.robot_radius] * 2, [self.width - self.robot_radius, self.height - self.robot_radius]
        )

        reward = -0.01   # Default step penalty
        if not self._check_collision(new_pos):
            self.robot_pos = new_pos
            self.consecutive_collisions = 0  # Reset collision count on successful move

            if self._on_carpet():
                reward -= 0.2

            reward += self._check_table_delivery()
        else:        
            reward -= 1.0  # Collision penalty, non-consecutive
            # Removed consecutive collision logic

        done = len(self.delivered_tables) == len(self.tables)
        # if done:
        #     reward += 2000.0

        reward = np.clip(reward, -50.0, 50.0) # Clip rewards to stabilise training

        self.total_reward += reward
        self.step_count += 1

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Returns coordinates and delivery status of tables
        status = [1 if i in self.delivered_tables else 0 for i in range(len(self.tables))]
        # Get ray casting sensors (45° cone in front)
        cone_sensors = self.get_cone_sensors(cone_width_deg=45, num_rays=60, max_distance=300)
        # Combine: [x, y, table_statuses..., ray_sensors...]    
        return np.array([self.robot_pos[0], self.robot_pos[1]] + status + cone_sensors, dtype=np.float32)

    def get_cone_sensors(self, cone_width_deg=45, num_rays=60, max_distance=600):
        """Cast rays in a cone in front of robot and detect what they hit."""
        px, py = float(self.robot_pos[0]), float(self.robot_pos[1])
        center_ang = self.last_move_dir
        cone_rad = math.radians(cone_width_deg)

        pts, types = cast_cone_rays(
            robot_pos=(px, py),
            wall_rects=self.walls,
            circle_obstacles=self.obstacles,
            center_angle=center_ang,
            cone_width=cone_rad,
            num_rays=num_rays,
            max_distance=max_distance,
            robot_radius=self.robot_radius
        )

        # Binary sensors: do we see each type of object?
        see_table = 0
        see_wall = 0
        see_obstacle = 0
        see_carpet = 0

        half_cone = cone_rad / 2
        for i, hit_type in enumerate(types):
            if hit_type == "wall":
                see_wall = 1
            elif hit_type == "circle":
                see_obstacle = 1

            # Check if ray passes through tables/carpets before hitting wall/obstacle
            ix, iy = pts[i]
            t_hit = math.hypot(ix - px, iy - py) if hit_type in ("wall", "circle") else max_distance

            ray_ang = center_ang - half_cone + i * (cone_rad / (num_rays - 1))
            dx = math.cos(ray_ang)
            dy = math.sin(ray_ang)

            # Check table intersections
            for (tx, ty, tw, th) in self.tables:
                t_table = _intersect_ray_rect(px, py, dx, dy, tx, ty, tw, th)
                if t_table is not None and 0 <= t_table < t_hit:
                    see_table = 1
                    break

            # Check carpet intersections
            for (cx, cy, cw, ch) in self.carpets:
                t_carpet = _intersect_ray_rect(px, py, dx, dy, cx, cy, cw, ch)
                if t_carpet is not None and 0 <= t_carpet < t_hit:
                    see_carpet = 1
                    break

            # Early exit if all sensors triggered
            if see_table and see_wall and see_obstacle and see_carpet:
                break

        return [see_table, see_wall, see_obstacle, see_carpet] 

    def _check_collision(self, pos):
        x, y = pos
        # Check for walls
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            if (
                wall_x - self.robot_radius <= x <= wall_x + wall_w + self.robot_radius
                and wall_y - self.robot_radius <= y <= wall_y + wall_h + self.robot_radius
            ):
                return True
            
        # Check for obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            if np.linalg.norm(pos - np.array([obs_x, obs_y])) < obs_r + self.robot_radius:
                return True
            
        #Check table collisions (with reduced collision area to allow delivery)
        collision_margin = self.robot_radius * 1  #larger means more lenient delivery area
        for tx, ty, tw, th in self.tables:
            collision_x = tx + collision_margin
            collision_y = ty + collision_margin
            collision_w = tw - 2 * collision_margin
            collision_h = th - 2 * collision_margin
        
            # Check if the reduced area is still positive
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
        for i, (tx, ty, tw, th) in enumerate(self.tables):
            if i not in self.delivered_tables:
                rx, ry = self.robot_pos

                # Expand the table area by robot radius
                expanded_x = tx - self.robot_radius
                expanded_y = ty - self.robot_radius
                expanded_w = tw + 2 * self.robot_radius
                expanded_h = th + 2 * self.robot_radius

                # Check if robot center is inside the expanded rectangle
                if (expanded_x <= rx <= expanded_x + expanded_w) and (expanded_y <= ry <= expanded_y + expanded_h):
                    self.delivered_tables.add(i)
                    reward += 15

        return reward

    def render(self):
        if self.render_mode != "human":
            return

        # Prevents frame freezing
        if pygame.get_init():
            pygame.event.pump()

        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        render_environment(self)

        px, py = float(self.robot_pos[0]), float(self.robot_pos[1])
        center_ang = self.last_move_dir
        cone_rad = math.radians(45)
        num_rays = 60
        max_d = 600

        # Cast rays in front cone
        pts, _ = cast_cone_rays(
            robot_pos=(px, py),
            wall_rects=self.walls,
            circle_obstacles=self.obstacles,
            center_angle=center_ang,
            cone_width=cone_rad,
            num_rays=num_rays,
            max_distance=max_d,
            robot_radius=self.robot_radius
        )

        if self.use_flashlight:
            # Flashlight mode: filled polygon with darkness overlay
            poly_pts = [(int(ix), int(iy)) for (ix, iy) in pts]
            if poly_pts:
                pygame.draw.polygon(self.window, (255, 255, 220), poly_pts)

            overlay = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            if poly_pts:
                pygame.draw.polygon(overlay, (0, 0, 0, 0), poly_pts)
            self.window.blit(overlay, (0, 0))
        else:
            # Ray lines mode: draw individual rays
            for (ix, iy) in pts:
                pygame.draw.line(
                    self.window,
                    (200, 200, 80),
                    (int(px), int(py)),
                    (int(ix), int(iy)),
                    1
                )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            pygame.font.quit()
            self.window = None
