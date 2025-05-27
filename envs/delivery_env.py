import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import List, Tuple

class DeliveryRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, width=800, height=600, robot_speed=5.0):
        self.width = width
        self.height = height
        self.robot_radius = 10
        
        # Environment elements based on your drawing
        self.walls = [
            # Outer walls
            (0, 0, width, 20),  # Top wall
            (0, 0, 20, height),  # Left wall
            (width-20, 0, 20, height),  # Right wall
            (0, height-20, width, 20),  # Bottom wall
            # Inner walls - you'll need to map these from your drawing
            (100, 100, 200, 20),  # Example inner wall
            (400, 200, 20, 150),  # Example inner wall
        ]
        
        self.carpets = [
            # (x, y, width, height) - areas where robot moves slower
            (50, 150, 100, 80),   # Carpet area 'C' from your drawing
            (600, 400, 120, 60),  # Another carpet area
        ]
        
        self.tables = [
            # Delivery targets (filled rectangles in your drawing)
            (150, 300, 30, 20),
            (450, 100, 30, 20),
            (650, 350, 30, 20),
        ]
        
        self.obstacles = [
            # Circular obstacles (people, poles, etc.)
            (300, 250, 15),  # (x, y, radius)
            (500, 400, 15),
            (200, 450, 15),
        ]
        
        # Robot state
        self.robot_pos = np.array([50.0, height - 50.0])  # Start bottom-left
        self.robot_angle = 0.0
        self.speed = robot_speed  # Now configurable
        self.delivered_tables = set()
        
        # Action space: discrete angles for movement direction
        self.n_actions = 8  # 8 directions (0, 45, 90, 135, 180, 225, 270, 315 degrees)
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation space: robot position + angle + delivery status
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0] * len(self.tables)),
            high=np.array([width, height, 360] + [1] * len(self.tables)),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([50.0, self.height - 50.0])
        self.robot_angle = 0.0
        self.delivered_tables = set()
        return self._get_obs(), {}

    def step(self, action):
        # Convert action to angle
        angle = action * (360 / self.n_actions)
        self.robot_angle = angle
        
        # Calculate movement
        rad = np.radians(angle)
        dx = self.speed * np.cos(rad)
        dy = self.speed * np.sin(rad)
        
        new_pos = self.robot_pos + np.array([dx, dy])
        
        # Check for collisions and apply movement
        reward = 0
        if not self._check_collision(new_pos):
            self.robot_pos = new_pos
            
            # Check if on carpet (slower movement penalty)
            if self._on_carpet():
                reward -= 0.1  # Small penalty for being on carpet
            
            # Check for table deliveries
            table_reward = self._check_table_delivery()
            reward += table_reward
            
            # Small step penalty to encourage efficiency
            reward -= 0.01
        else:
            # Collision penalty
            reward = -1.0
        
        # Check if all tables delivered
        done = len(self.delivered_tables) == len(self.tables)
        if done:
            reward += 100  # Completion bonus
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Robot position, angle, and delivery status
        delivery_status = [1 if i in self.delivered_tables else 0 
                          for i in range(len(self.tables))]
        return np.array([self.robot_pos[0], self.robot_pos[1], self.robot_angle] + 
                       delivery_status, dtype=np.float32)

    def _check_collision(self, pos):
        x, y = pos
        
        # Check wall collisions
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            if (wall_x <= x <= wall_x + wall_w and 
                wall_y <= y <= wall_y + wall_h):
                return True
        
        # Check obstacle collisions
        for obs_x, obs_y, obs_radius in self.obstacles:
            if np.linalg.norm(pos - np.array([obs_x, obs_y])) < obs_radius + self.robot_radius:
                return True
        
        return False

    def _on_carpet(self):
        x, y = self.robot_pos
        for carpet_x, carpet_y, carpet_w, carpet_h in self.carpets:
            if (carpet_x <= x <= carpet_x + carpet_w and 
                carpet_y <= y <= carpet_y + carpet_h):
                return True
        return False

    def _check_table_delivery(self):
        reward = 0
        x, y = self.robot_pos
        
        for i, (table_x, table_y, table_w, table_h) in enumerate(self.tables):
            if i not in self.delivered_tables:
                # Check if robot is close enough to table
                table_center = np.array([table_x + table_w/2, table_y + table_h/2])
                if np.linalg.norm(self.robot_pos - table_center) < 25:
                    self.delivered_tables.add(i)
                    reward += 10  # Delivery reward
        
        return reward

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Clear screen
        self.window.fill((255, 255, 255))  # White background

        # Draw walls
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            pygame.draw.rect(self.window, (0, 0, 0), 
                           (wall_x, wall_y, wall_w, wall_h))

        # Draw carpets
        for carpet_x, carpet_y, carpet_w, carpet_h in self.carpets:
            pygame.draw.rect(self.window, (139, 69, 19), 
                           (carpet_x, carpet_y, carpet_w, carpet_h))
            # Add carpet pattern
            for i in range(0, carpet_w, 10):
                for j in range(0, carpet_h, 10):
                    pygame.draw.line(self.window, (160, 82, 45),
                                   (carpet_x + i, carpet_y + j),
                                   (carpet_x + i + 5, carpet_y + j + 5))

        # Draw tables
        for i, (table_x, table_y, table_w, table_h) in enumerate(self.tables):
            color = (0, 255, 0) if i in self.delivered_tables else (255, 165, 0)
            pygame.draw.rect(self.window, color, 
                           (table_x, table_y, table_w, table_h))

        # Draw obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            pygame.draw.circle(self.window, (255, 0, 0), 
                             (int(obs_x), int(obs_y)), obs_radius)

        # Draw robot
        pygame.draw.circle(self.window, (0, 0, 255), 
                         (int(self.robot_pos[0]), int(self.robot_pos[1])), 
                         self.robot_radius)
        
        # Draw robot direction
        end_x = self.robot_pos[0] + 20 * np.cos(np.radians(self.robot_angle))
        end_y = self.robot_pos[1] + 20 * np.sin(np.radians(self.robot_angle))
        pygame.draw.line(self.window, (0, 0, 255),
                        (int(self.robot_pos[0]), int(self.robot_pos[1])),
                        (int(end_x), int(end_y)), 3)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


# Test the environment
if __name__ == "__main__":
    env = DeliveryRobotEnv(render_mode="human")
    obs, _ = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            obs, _ = env.reset()
    
    env.close()