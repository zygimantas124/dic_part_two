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

print("Manual Control (using env.step):")
print("Arrow Keys: Move robot")
print("  UP: Move forward in current direction")
print("  DOWN: Move backward (opposite direction)")
print("  LEFT: Rotate left (decrease angle)")
print("  RIGHT: Rotate right (increase angle)")
print("ESC: Exit")
print("D: Debug current position and rewards")
print("R: Reset environment")
print("SPACE: Take one step in current direction")

running = True
clock = pygame.time.Clock()

# Keep track of current angle for manual control
current_angle = 0

while running:
    action = None  # Will be set if we want to take a step
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_LEFT:
                # Rotate left (decrease angle)
                current_angle = (current_angle - 15) % 360
                env.angle = current_angle
                print(f"Angle: {current_angle}°")
            elif event.key == pygame.K_RIGHT:
                # Rotate right (increase angle)
                current_angle = (current_angle + 15) % 360
                env.angle = current_angle
                print(f"Angle: {current_angle}°")
            elif event.key == pygame.K_UP:
                # Move forward in current direction
                action = current_angle
            elif event.key == pygame.K_DOWN:
                # Move backward (opposite direction)
                action = (current_angle + 180) % 360
            elif event.key == pygame.K_SPACE:
                # Take one step in current direction
                action = current_angle
            elif event.key == pygame.K_d:
                # Debug current position and state
                print(f"\n=== DEBUG ===")
                print(f"Robot at: ({obs[0]:.1f}, {obs[1]:.1f})")
                print(f"Current angle: {current_angle}°")
                print(f"Env angle: {obs[2]:.1f}°")
                print(f"Delivered tables: {env.delivered_tables}")
                print(f"Observation: {obs}")
                for i, (tx, ty, tw, th) in enumerate(env.tables):
                    table_center = np.array([tx + tw/2, ty + th/2])
                    robot_pos = np.array([obs[0], obs[1]])
                    distance = np.linalg.norm(robot_pos - table_center)
                    status = "DELIVERED" if i in env.delivered_tables else "PENDING"
                    print(f"Table {i}: center=({table_center[0]:.1f},{table_center[1]:.1f}) distance={distance:.1f} [{status}]")
                print("=============\n")
            elif event.key == pygame.K_r:
                # Reset environment
                obs, _ = env.reset()
                current_angle = 0
                env.angle = 0 
                print("Environment reset!")
                print(f"Robot reset to ({obs[0]:.1f}, {obs[1]:.1f})")

    # Take action if one was selected
    if action is not None:
        obs, reward, done, truncated, info = env.step(action)
        
        # Print feedback
        if reward > 0:
            print(f"🎉 Positive reward: {reward:.2f}")
        elif reward < -1:
            print(f"💥 Collision! Reward: {reward:.2f}")
        elif reward < 0:
            print(f"Step penalty: {reward:.2f}")
        
        # Check if mission complete
        if done:
            print("🏆 MISSION COMPLETE! All tables delivered! 🏆")
            print("Press 'R' to reset and try again")
        
        # Show delivery status
        if len(env.delivered_tables) > 0:
            print(f"📦 Delivered: {len(env.delivered_tables)}/{len(env.tables)} tables")

    # Render the environment
    env.render()
    clock.tick(10)  # 10 FPS

env.close()
print("Manual control ended.")

# Optional: Print final statistics
print(f"\nFinal Results:")
print(f"Tables delivered: {len(env.delivered_tables)}/{len(env.tables)}")
if len(env.delivered_tables) > 0:
    print(f"Delivered table IDs: {sorted(env.delivered_tables)}")