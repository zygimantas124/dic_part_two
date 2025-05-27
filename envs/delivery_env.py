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
        self.robot_radius = 8
        
        # Scale factors to map your diagram to screen coordinates
        self.scale_x = width / 10  # Assuming 5 grid units wide
        self.scale_y = height / 10  # Assuming 5 grid units tall
        
        def grid_to_pixel(grid_x, grid_y, grid_w=0, grid_h=0):
            """Convert grid coordinates to pixel coordinates"""
            return (int(grid_x * self.scale_x), int(grid_y * self.scale_y), 
                   int(grid_w * self.scale_x), int(grid_h * self.scale_y))
        
        # Walls based on your diagram
        self.walls = [
            # Outer perimeter
            grid_to_pixel(0, 0, 10, 0.1),      # Top wall
            grid_to_pixel(0, 0, 0.1, 10),      # Left wall  
            grid_to_pixel(9.9, 0, 0.1, 10),    # Right wall
            grid_to_pixel(0, 9.9, 10, 0.1),    # Bottom wall
            
            # Horizontal interior walls
            grid_to_pixel(0, 1, 1, 0.1),      # (posx, posy, width, height)
            grid_to_pixel(2, 1, 2, 0.1),      
            grid_to_pixel(5, 1, 5, 0.1),      
            
            grid_to_pixel(5, 6, 5, 0.1),
            grid_to_pixel(0, 7, 4, 0.1),     
            grid_to_pixel(5, 7, 5, 0.1),
            
            # Vertical interior walls
            grid_to_pixel(4, 1 , 0.1, 3),
            grid_to_pixel(4, 5, 0.1, 2),
            grid_to_pixel(4, 8, 0.1, 2),

            grid_to_pixel(5, 1, 0.1, 1),
            grid_to_pixel(5, 3, 0.1, 3),
            grid_to_pixel(5, 8, 0.1, 2),

            # Small interior walls for rooms
           
        ]
        
        # Carpet area (marked 'C' in your diagram)
        self.carpets = [
            grid_to_pixel(2.5, 2.5, 1, 3),  # Carpet room
        ]
        
        # Delivery tables (filled rectangles in your diagram)
        self.tables = [
            grid_to_pixel(0, 2, 1, 2),   # Table in top-left room
            grid_to_pixel(1, 6, 2, 1),   # Another in top-left room  
            grid_to_pixel(5, 4 , 1, 2),   # Table in top-right room
            grid_to_pixel(9, 2, 1, 2),   # Another in top-right room
            grid_to_pixel(6, 9, 3, 1),   # Table in bottom-left room
        ]
        
        # Circular obstacles (people - stick figures in your diagram)
        obstacle_positions = [
            (7, 3.5),  # Person in top middle area (marked with stick figure)
            (6, 8),    # Person in bottom-right area
            (6, 2),  # Printer
        ]
        self.obstacles = [(grid_to_pixel(x, y)[0], grid_to_pixel(x, y)[1], 15) 
                         for x, y in obstacle_positions]
        
        # Hatched areas (obstacles/blocked areas)
        self.blocked_areas = [
            # grid_to_pixel(0, 4.3, 0.8, 1.5),    # Top-left hatched area
            # grid_to_pixel(4.3, 4.3, 1.5, 1.5),  # Top-right hatched area  
            # grid_to_pixel(1.5, 2.3, 1, 0.5),    # Middle hatched area
            # grid_to_pixel(3.2, 0.8, 0.6, 0.4),  # Bottom hatched area
            # grid_to_pixel(5, 0.2, 0.8, 1.6),    # Right side hatched area
        ]
        
        # Robot starting position (bottom-left area)
        start_pixel = grid_to_pixel(0.5, 0.5)
        self.robot_start_pos = np.array([float(start_pixel[0]), float(start_pixel[1])])
        self.robot_pos = self.robot_start_pos.copy()
        self.robot_angle = 0.0
        self.speed = robot_speed
        self.delivered_tables = set()
        
        # Action space: 8 directions
        self.n_actions = 8
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
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = self.robot_start_pos.copy()
        self.robot_angle = 0.0
        self.delivered_tables = set()
        return self._get_obs(), {}

    def step(self, action):
        # Convert action to angle (0, 45, 90, 135, 180, 225, 270, 315 degrees)
        angle = action * (360 / self.n_actions)
        self.robot_angle = angle
        
        # Calculate movement
        rad = np.radians(angle)
        dx = self.speed * np.cos(rad)
        dy = self.speed * np.sin(rad)
        
        new_pos = self.robot_pos + np.array([dx, dy])
        
        # Keep robot within bounds
        new_pos = np.clip(new_pos, [self.robot_radius, self.robot_radius], 
                         [self.width - self.robot_radius, self.height - self.robot_radius])
        
        reward = 0
        
        # Check for collisions
        if not self._check_collision(new_pos):
            self.robot_pos = new_pos
            
            # Check if on carpet (movement penalty)
            if self._on_carpet():
                reward -= 0.2  # Higher penalty for carpet movement
            
            # Check for table deliveries
            table_reward = self._check_table_delivery()
            reward += table_reward
            
            # Small step penalty to encourage efficiency
            reward -= 0.02
        else:
            # Collision penalty
            reward = -2.0
        
        # Check if all tables delivered
        done = len(self.delivered_tables) == len(self.tables)
        if done:
            reward += 200  # Large completion bonus
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        delivery_status = [1 if i in self.delivered_tables else 0 
                          for i in range(len(self.tables))]
        return np.array([self.robot_pos[0], self.robot_pos[1], self.robot_angle] + 
                       delivery_status, dtype=np.float32)

    def _check_collision(self, pos):
        x, y = pos
        
        # Check wall collisions
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            if (wall_x - self.robot_radius <= x <= wall_x + wall_w + self.robot_radius and 
                wall_y - self.robot_radius <= y <= wall_y + wall_h + self.robot_radius):
                return True
        
        # # Check blocked area collisions
        # for block_x, block_y, block_w, block_h in self.blocked_areas:
        #     if (block_x - self.robot_radius <= x <= block_x + block_w + self.robot_radius and 
        #         block_y - self.robot_radius <= y <= block_y + block_h + self.robot_radius):
        #         return True
        
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
        
        for i, (table_x, table_y, table_w, table_h) in enumerate(self.tables):
            if i not in self.delivered_tables:
                table_center = np.array([table_x + table_w/2, table_y + table_h/2])
                distance = np.linalg.norm(self.robot_pos - table_center)
                
                if distance < 30:  # Delivery distance threshold
                    self.delivered_tables.add(i)
                    reward += 50  # Delivery reward
                    print(f"Delivered to table {i+1}!")
        
        return reward

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # Clear screen with light gray background
        self.window.fill((240, 240, 240))

        # Draw blocked areas (hatched pattern)
        for block_x, block_y, block_w, block_h in self.blocked_areas:
            # Fill with gray
            pygame.draw.rect(self.window, (180, 180, 180), 
                           (block_x, block_y, block_w, block_h))
            # Add hatching pattern
            for i in range(0, block_w + block_h, 8):
                start_x = block_x + i
                start_y = block_y
                end_x = block_x
                end_y = block_y + i
                if start_x <= block_x + block_w and end_y <= block_y + block_h:
                    pygame.draw.line(self.window, (120, 120, 120),
                                   (start_x, start_y), (end_x, end_y), 2)

        # Draw carpets with 'C' label
        for carpet_x, carpet_y, carpet_w, carpet_h in self.carpets:
            pygame.draw.rect(self.window, (139, 69, 19), 
                           (carpet_x, carpet_y, carpet_w, carpet_h))
            # Add carpet texture
            for i in range(0, carpet_w, 15):
                for j in range(0, carpet_h, 15):
                    pygame.draw.circle(self.window, (160, 82, 45),
                                     (carpet_x + i + 7, carpet_y + j + 7), 3)
            
            # Draw 'C' label
            text = self.font.render('C', True, (255, 255, 255))
            text_rect = text.get_rect(center=(carpet_x + carpet_w//2, carpet_y + carpet_h//2))
            self.window.blit(text, text_rect)

        # Draw walls (black lines)
        for wall_x, wall_y, wall_w, wall_h in self.walls:
            pygame.draw.rect(self.window, (0, 0, 0), 
                           (wall_x, wall_y, wall_w, wall_h))

        # Draw delivery tables
        for i, (table_x, table_y, table_w, table_h) in enumerate(self.tables):
            color = (0, 200, 0) if i in self.delivered_tables else (255, 165, 0)  # Green if delivered
            pygame.draw.rect(self.window, color, 
                           (table_x, table_y, table_w, table_h))
            pygame.draw.rect(self.window, (0, 0, 0),  # Black border
                           (table_x, table_y, table_w, table_h), 2)
            
            # Add table number
            text = self.font.render(str(i+1), True, (0, 0, 0))
            text_rect = text.get_rect(center=(table_x + table_w//2, table_y + table_h//2))
            self.window.blit(text, text_rect)

        # Draw obstacles (people/circular obstacles)
        for obs_x, obs_y, obs_radius in self.obstacles:
            pygame.draw.circle(self.window, (255, 100, 100), 
                             (int(obs_x), int(obs_y)), obs_radius)
            pygame.draw.circle(self.window, (200, 0, 0),  # Darker border
                             (int(obs_x), int(obs_y)), obs_radius, 2)

        # Draw robot
        pygame.draw.circle(self.window, (0, 100, 255), 
                         (int(self.robot_pos[0]), int(self.robot_pos[1])), 
                         self.robot_radius)
        
        # Draw robot direction indicator
        end_x = self.robot_pos[0] + (self.robot_radius + 5) * np.cos(np.radians(self.robot_angle))
        end_y = self.robot_pos[1] + (self.robot_radius + 5) * np.sin(np.radians(self.robot_angle))
        pygame.draw.line(self.window, (0, 50, 150),
                        (int(self.robot_pos[0]), int(self.robot_pos[1])),
                        (int(end_x), int(end_y)), 3)

        # Draw status information
        delivered_text = f"Delivered: {len(self.delivered_tables)}/{len(self.tables)}"
        text_surface = self.font.render(delivered_text, True, (0, 0, 0))
        self.window.blit(text_surface, (10, 10))
        
        pos_text = f"Position: ({int(self.robot_pos[0])}, {int(self.robot_pos[1])})"
        pos_surface = self.font.render(pos_text, True, (0, 0, 0))
        self.window.blit(pos_surface, (10, 35))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


# Test the environment
if __name__ == "__main__":
    env = DeliveryRobotEnv(render_mode="human", robot_speed=3)
    obs, _ = env.reset()
    
    print("Testing the new environment layout...")
    print("Close the window to stop the test.")
    
    for step in range(2000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            print(f"All deliveries completed in {step} steps!")
            obs, _ = env.reset()
    
    env.close()