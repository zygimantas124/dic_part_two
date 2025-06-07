from office.delivery_env import DeliveryRobotEnv
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Create the environment
env = DeliveryRobotEnv(
    config="open_office",
    render_mode="human",
    show_walls=True,
    show_obstacles=True,
    show_carpets=True,
)
obs, _ = env.reset()

print("Manual Control (8 directions):")
print("Use arrow keys (individually or in combination) to move the robot.")
print("ESC: Exit")
print("D: Debug current position and reward")
print("R: Reset environment")

running = True
clock = pygame.time.Clock()

while running:
    action = None
    pressed_keys = pygame.key.get_pressed()

    # Get direction from key combination
    if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_LEFT]:
        action = 4  # Up-Left
    elif pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_RIGHT]:
        action = 5  # Up-Right
    elif pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_LEFT]:
        action = 6  # Down-Left
    elif pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_RIGHT]:
        action = 7  # Down-Right
    elif pressed_keys[pygame.K_UP]:
        action = 0  # Up
    elif pressed_keys[pygame.K_DOWN]:
        action = 1  # Down
    elif pressed_keys[pygame.K_LEFT]:
        action = 2  # Left
    elif pressed_keys[pygame.K_RIGHT]:
        action = 3  # Right

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_d:
                # Debug print
                print(f"\n=== DEBUG ===")
                print(f"Robot at: ({obs[0]:.1f}, {obs[1]:.1f})")
                print(f"Delivered tables: {env.delivered_tables}")
                print(f"Observation: {obs}")
                for i, (tx, ty, tw, th) in enumerate(env.tables):
                    table_center = np.array([tx + tw / 2, ty + th / 2])
                    robot_pos = np.array([obs[0], obs[1]])
                    distance = np.linalg.norm(robot_pos - table_center)
                    status = "DELIVERED" if i in env.delivered_tables else "PENDING"
                    print(
                        f"Table {i}: center=({table_center[0]:.1f},{table_center[1]:.1f}) distance={distance:.1f} [{status}]"
                    )
                print("=============\n")
            elif event.key == pygame.K_r:
                obs, _ = env.reset()
                print("Environment reset!")
                print(f"Robot reset to ({obs[0]:.1f}, {obs[1]:.1f})")

    if action is not None:
        obs, reward, done, truncated, info = env.step(action)

        if reward > 0:
            print(f"🎉 Positive reward: {reward:.2f}")
        elif reward < -1:
            print(f"💥 Collision! Reward: {reward:.2f}")
        elif reward < 0:
            print(f"Step penalty: {reward:.2f}")

        if done:
            print("🏆 MISSION COMPLETE! All tables delivered! 🏆")
            print("Press 'R' to reset and try again")

        if len(env.delivered_tables) > 0:
            print(f"📦 Delivered: {len(env.delivered_tables)}/{len(env.tables)} tables")

    env.render()
    clock.tick(10)  # 10 FPS

env.close()
print("Manual control ended.")

# Final stats
print(f"\nFinal Results:")
print(f"Tables delivered: {len(env.delivered_tables)}/{len(env.tables)}")
if len(env.delivered_tables) > 0:
    print(f"Delivered table IDs: {sorted(env.delivered_tables)}")
