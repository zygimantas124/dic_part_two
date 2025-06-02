import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# does not work if starting outside of office
from office.render import render_environment
from office.components.tables import get_target_tables
from office.components.obstacles import get_carpets, get_people, get_furniture
from office.components.walls import get_walls


class DeliveryRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, show_walls=True, show_obstacles=True, show_carpets=True):
        self.width = 800
        self.height = 600
        self.robot_radius = 10
        self.step_size = 10.0
        self.reward = 0.0
        self.action = None

        self.show_walls = show_walls
        self.show_obstacles = show_obstacles
        self.show_carpets = show_carpets

        self.walls = get_walls() if show_walls else []
        self.tables = get_target_tables()
        self.carpets = get_carpets() if show_carpets else []
        self.obstacles = (get_people() + get_furniture()) if show_obstacles else []

        self.start_pos = np.array([100, 500])
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

        # self.observation_space = spaces.Box(  # Observe coordinates, action, and delivery status
        #     low=np.array([0, 0] + [0] * len(self.tables), dtype=np.float32),
        #     high=np.array([self.width, self.height] + [1] * len(self.tables), dtype=np.float32),
        #     dtype=np.float32,
        # )

        # self.observation_space = spaces.Box(  # Only "observe coordinates"
        #     low=np.array([0, 0], dtype=np.float32),
        #     high=np.array([self.width, self.height], dtype=np.float32),
        #     dtype=np.float32
        # )
        self.observation_space = spaces.Box(
        low=np.array([0, 0, 0], dtype=np.float32),
        high=np.array([self.width, self.height, len(self.tables)], dtype=np.float32),
        dtype=np.float32
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
        return self._get_obs(), {}

    def step(self, action):
        self.action = action
        movement = (self.directions[self.action] * self.step_size).flatten()

        new_pos = self.robot_pos + movement
        new_pos = np.clip(
            new_pos, [self.robot_radius] * 2, [self.width - self.robot_radius, self.height - self.robot_radius]
        )

        reward = -0.01   # Default step penalty
        # TODO: clip rewards
        if not self._check_collision(new_pos):
            self.robot_pos = new_pos

            if self._on_carpet():
                reward -= 0.2

            reward += self._check_table_delivery()
        else:
            reward = -1.0  # Collision penalty

        done = len(self.delivered_tables) == len(self.tables)
        if done:
            reward += 2000.0

        self.total_reward += reward
        self.step_count += 1

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Returns coordinates, last action taken, and delivery status of tables
        # status = [1 if i in self.delivered_tables else 0 for i in range(len(self.tables))]
        # return np.array([self.robot_pos[0], self.robot_pos[1], self.action] + status, dtype=np.float32)

        # Returns coordinates only
        # return np.array([self.robot_pos[0], self.robot_pos[1]])
        return np.array([self.robot_pos[0], self.robot_pos[1], len(self.delivered_tables)], dtype=np.float32)

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
            
        # Check table collisions (with reduced collision area to allow delivery)
        # collision_margin = self.robot_radius * 1  #larger means more lenient delivery area
        # for tx, ty, tw, th in self.tables:
        #     collision_x = tx + collision_margin
        #     collision_y = ty + collision_margin
        #     collision_w = tw - 2 * collision_margin
        #     collision_h = th - 2 * collision_margin
        
        #     # Check if the reduced area is still positive
        #     if collision_w > 0 and collision_h > 0:
        #         if (
        #             collision_x - self.robot_radius <= x <= collision_x + collision_w + self.robot_radius
        #             and collision_y - self.robot_radius <= y <= collision_y + collision_h + self.robot_radius
        #         ):
        #             return True

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
                    reward += 10

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

    def close(self):
        if self.window:
            pygame.quit()
            pygame.font.quit()
            self.window = None
