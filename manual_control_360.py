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
print("D: Debug current position")
print("R: Reset robot position")

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
            elif event.key == pygame.K_d:
                # Debug current position
                print(f"\n=== DEBUG ===")
                print(f"Robot at: ({env.robot_pos[0]:.1f}, {env.robot_pos[1]:.1f})")
                print(f"Angle: {env.angle:.1f}°")
                print(f"Delivered tables: {env.delivered_tables}")
                for i, (tx, ty, tw, th) in enumerate(env.tables):
                    table_center = np.array([tx + tw/2, ty + th/2])
                    distance = np.linalg.norm(env.robot_pos - table_center)
                    status = "DELIVERED" if i in env.delivered_tables else "PENDING"
                    print(f"Table {i}: distance={distance:.1f} [{status}]")
                print("=============\n")
            elif event.key == pygame.K_r:
                # Reset robot
                env.robot_pos = env.start_pos.copy()
                env.angle = 0.0
                print(f"Robot reset to {env.robot_pos}")

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
            
            # *** Check for deliveries after movement ***
            delivery_reward = env._check_table_delivery()
            if delivery_reward > 0:
                print(f"🎉 DELIVERY MADE! Reward: {delivery_reward}")
                print(f"Total delivered: {len(env.delivered_tables)}/{len(env.tables)}")
                
                # Check if all tables delivered
                if len(env.delivered_tables) == len(env.tables):
                    print("🏆 ALL TABLES DELIVERED! MISSION COMPLETE! 🏆")
        else:
            print("Collision detected.")

    # Update and render
    obs = env._get_obs()
    env.render()
    clock.tick(10)  # 10 FPS

env.close()
print("Manual control ended.")