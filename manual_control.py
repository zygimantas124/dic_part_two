from office.delivery_env import DeliveryRobotEnv
import pygame

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
    action = None
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_RIGHT:
                action = 0  # Right
            elif event.key == pygame.K_DOWN:
                action = 2  # Down
            elif event.key == pygame.K_LEFT:
                action = 4  # Left
            elif event.key == pygame.K_UP:
                action = 6  # Up
            # Diagonal movements
            elif event.key == pygame.K_1:
                action = 1  # Down-Right
            elif event.key == pygame.K_3:
                action = 3  # Down-Left
            elif event.key == pygame.K_7:
                action = 5  # Up-Left
            elif event.key == pygame.K_9:
                action = 7  # Up-Right
    
    # Take action if key was pressed
    if action is not None:
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
        
        if done:
            print("All deliveries completed! Resetting...")
            obs, _ = env.reset()
    
    env.render()
    clock.tick(10)  # 10 FPS

env.close()
print("Manual control ended.")