from main import DeliveryRobotEnv

if __name__ == "__main__":
    env = DeliveryRobotEnv(render_mode="human")
    obs, _ = env.reset()

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        if done:
            print(f"Delivered all tables in {step} steps!")
            obs, _ = env.reset()

    env.close()
