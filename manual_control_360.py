from office.delivery_env import DeliveryRobotEnv
import pygame
import numpy as np

# Initialize pygame first
pygame.init()

# Manual control of the delivery robot

env = DeliveryRobotEnv(
    render_mode="human",
    show_walls=True,  # Set to False to hide walls
    show_obstacles=True,  # Set to False to hide obstacles
    show_carpets=True,  # Set to False to hide carpets
)
obs, _ = env.reset()

print("Manual Control:")
print("Arrow Keys: Move robot")
print("ESC: Exit")
print("Action mapping:")
print("0: Right (0°)")
print("1: Down-Right (45°)")
print("2: Down (90°)")
print("3: Down-Left (135°)")
print("4: Left (180°)")
print("5: Up-Left (225°)")
print("6: Up (270°)")
print("7: Up-Right (315°)")

running = True
clock = pygame.time.Clock()

while running:
    move_direction = 0  # +1 for forward, -1 for backward
    rotate_direction = 0  # +1 for right, -1 for left

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_LEFT:
                rotate_direction = -1
            elif event.key == pygame.K_RIGHT:
                rotate_direction = 1
            elif event.key == pygame.K_UP:
                move_direction = 1
            elif event.key == pygame.K_DOWN:
                move_direction = -1
    # Update the angle
    if rotate_direction != 0:
        env.angle = (env.angle + rotate_direction * 5) % 360

    # Move forward or backward
    if move_direction != 0:
        rad = np.radians(env.angle)
        dx = env.step_size * move_direction * np.cos(rad)
        dy = env.step_size * move_direction * np.sin(rad)
        new_pos = env.robot_pos + np.array([dx, dy])
        
        # Clamp to screen boundaries
        new_pos = np.clip(new_pos,
                          [env.robot_radius, env.robot_radius],
                          [env.width - env.robot_radius, env.height - env.robot_radius])
        
        if not env._check_collision(new_pos):
            env.robot_pos = new_pos
        else:
            print("Collision detected.")

    # Update and render
    obs = env._get_obs()
    env.render()
    clock.tick(10)  # 10 FPS

env.close()
print("Manual control ended.")