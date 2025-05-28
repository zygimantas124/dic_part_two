from office.delivery_env import DeliveryRobotEnv

if __name__ == "__main__":
    env = DeliveryRobotEnv(
        render_mode="human",
        show_walls=True,  # Set to False to hide walls
        show_obstacles=True,  # Set to False to hide obstacles
        show_carpets=True,  # Set to False to hide carpets
    )
    obs, _ = env.reset()

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        if done:
            print(f"Delivered all tables in {step} steps!")
            obs, _ = env.reset()

    env.close()
