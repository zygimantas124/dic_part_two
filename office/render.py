import pygame
import numpy as np

# ==============================================
# CONFIGURATION - All image paths and toggles in one place
# ==============================================
RENDER_CONFIG = {
    # Toggle between images (True) and colors (False) for each object type
    "use_images": {
        "background": False,
        "walls": False,
        "tables": False,
        "obstacles": False,
        "carpets": True,
        "robot": False,
    },
    # Image paths for each object type
    "image_paths": {
        "background": "office/sprites/bing.png",
        "walls": "",
        "tables": {"delivered": "office/sprites/wood_g.jpg", "pending": "officesprites/wood_r.jpg"},
        "obstacles": "",
        "carpets": "office/sprites/carpet.jpg",
        "robot": "",
    },
    # Fallback colors (original colors)
    "colors": {
        "background": (55, 55, 55),
        "walls": (250, 250, 250),
        "tables": {"delivered": (0, 200, 0), "pending": (255, 165, 0), "border": (0, 0, 0)},
        "obstacles": {"fill": (255, 100, 100), "border": (200, 0, 0)},
        "carpets": (139, 69, 19),
        "robot": {"body": (0, 100, 255), "direction": (0, 50, 150)},
        "text": (0, 0, 0),
    },
}


def load_image(env, image_type, subtype=None):
    """Load and cache an image, with fallback handling."""
    cache_key = f"{image_type}_{subtype}" if subtype else image_type

    # Check if already loaded
    if not hasattr(env, "_image_cache"):
        env._image_cache = {}

    if cache_key in env._image_cache:
        return env._image_cache[cache_key]

    # Get image path
    if subtype:
        image_path = RENDER_CONFIG["image_paths"][image_type][subtype]
    else:
        image_path = RENDER_CONFIG["image_paths"][image_type]

    # Try to load image
    try:
        image = pygame.image.load(image_path)
        env._image_cache[cache_key] = image
        return image
    except pygame.error as e:
        print(f"Could not load {image_type} image ({image_path}): {e}")
        env._image_cache[cache_key] = None
        return None


def render_environment(env):
    win = env.window

    # ==============================================
    # BACKGROUND
    # ==============================================
    if RENDER_CONFIG["use_images"]["background"]:
        background_img = load_image(env, "background")
        if background_img:
            # Scale to window size
            if not hasattr(env, "_scaled_background"):
                env._scaled_background = pygame.transform.scale(background_img, (env.width, env.height))
            win.blit(env._scaled_background, (0, 0))
        else:
            win.fill(RENDER_CONFIG["colors"]["background"])
    else:
        win.fill(RENDER_CONFIG["colors"]["background"])

    # ==============================================
    # WALLS
    # ==============================================
    if env.show_walls:
        if RENDER_CONFIG["use_images"]["walls"]:
            wall_img = load_image(env, "walls")
            if wall_img:
                for wall in env.walls:
                    scaled_wall = pygame.transform.scale(wall_img, (wall[2], wall[3]))
                    win.blit(scaled_wall, (wall[0], wall[1]))
            else:
                # Fallback to color
                for wall in env.walls:
                    pygame.draw.rect(win, RENDER_CONFIG["colors"]["walls"], wall)
        else:
            for wall in env.walls:
                pygame.draw.rect(win, RENDER_CONFIG["colors"]["walls"], wall)

    # ==============================================
    # TABLES
    # ==============================================
    if RENDER_CONFIG["use_images"]["tables"]:
        delivered_img = load_image(env, "tables", "delivered")
        pending_img = load_image(env, "tables", "pending")

        for i, table in enumerate(env.tables):
            img = delivered_img if i in env.delivered_tables else pending_img
            if img:
                scaled_table = pygame.transform.scale(img, (table[2], table[3]))
                win.blit(scaled_table, (table[0], table[1]))
            else:
                # Fallback to color
                color = (
                    RENDER_CONFIG["colors"]["tables"]["delivered"]
                    if i in env.delivered_tables
                    else RENDER_CONFIG["colors"]["tables"]["pending"]
                )
                pygame.draw.rect(win, color, table)
                pygame.draw.rect(win, RENDER_CONFIG["colors"]["tables"]["border"], table, 2)
    else:
        for i, table in enumerate(env.tables):
            color = (
                RENDER_CONFIG["colors"]["tables"]["delivered"]
                if i in env.delivered_tables
                else RENDER_CONFIG["colors"]["tables"]["pending"]
            )
            pygame.draw.rect(win, color, table)
            pygame.draw.rect(win, RENDER_CONFIG["colors"]["tables"]["border"], table, 2)

    # ==============================================
    # OBSTACLES
    # ==============================================
    if env.show_obstacles:
        if RENDER_CONFIG["use_images"]["obstacles"]:
            obstacle_img = load_image(env, "obstacles")
            if obstacle_img:
                for obs in env.obstacles:
                    # Scale image to fit obstacle size
                    img_size = obs[2] * 2  # diameter
                    scaled_obs = pygame.transform.scale(obstacle_img, (img_size, img_size))
                    # Center the image on the obstacle position
                    pos = (int(obs[0] - obs[2]), int(obs[1] - obs[2]))
                    win.blit(scaled_obs, pos)
            else:
                # Fallback to color
                for obs in env.obstacles:
                    pygame.draw.circle(
                        win, RENDER_CONFIG["colors"]["obstacles"]["fill"], (int(obs[0]), int(obs[1])), obs[2]
                    )
                    pygame.draw.circle(
                        win, RENDER_CONFIG["colors"]["obstacles"]["border"], (int(obs[0]), int(obs[1])), obs[2], 2
                    )
        else:
            for obs in env.obstacles:
                pygame.draw.circle(
                    win, RENDER_CONFIG["colors"]["obstacles"]["fill"], (int(obs[0]), int(obs[1])), obs[2]
                )
                pygame.draw.circle(
                    win, RENDER_CONFIG["colors"]["obstacles"]["border"], (int(obs[0]), int(obs[1])), obs[2], 2
                )

    # ==============================================
    # CARPETS
    # ==============================================
    if env.show_carpets:
        if RENDER_CONFIG["use_images"]["carpets"]:
            carpet_img = load_image(env, "carpets")
            if carpet_img:
                for carpet in env.carpets:
                    scaled_carpet = pygame.transform.scale(carpet_img, (carpet[2], carpet[3]))
                    win.blit(scaled_carpet, (carpet[0], carpet[1]))
            else:
                # Fallback to color
                for carpet in env.carpets:
                    pygame.draw.rect(win, RENDER_CONFIG["colors"]["carpets"], carpet)
        else:
            for carpet in env.carpets:
                pygame.draw.rect(win, RENDER_CONFIG["colors"]["carpets"], carpet)

    # ==============================================
    # ROBOT
    # ==============================================
    if RENDER_CONFIG["use_images"]["robot"]:
        robot_img = load_image(env, "robot")
        if robot_img:
            # Scale and rotate robot image
            img_size = env.robot_radius * 2
            scaled_robot = pygame.transform.scale(robot_img, (img_size, img_size))
            rotated_robot = pygame.transform.rotate(scaled_robot, -env.angle)  # Negative for correct rotation

            # Center the rotated image
            rect = rotated_robot.get_rect(center=(int(env.robot_pos[0]), int(env.robot_pos[1])))
            win.blit(rotated_robot, rect)
        else:
            # Fallback to color
            pygame.draw.circle(
                win,
                RENDER_CONFIG["colors"]["robot"]["body"],
                (int(env.robot_pos[0]), int(env.robot_pos[1])),
                env.robot_radius,
            )
            # end_x = env.robot_pos[0] + (env.robot_radius + 5) * np.cos(np.radians(env.angle))
            # end_y = env.robot_pos[1] + (env.robot_radius + 5) * np.sin(np.radians(env.angle))
            # pygame.draw.line(
            #     win,
            #     RENDER_CONFIG["colors"]["robot"]["direction"],
            #     (int(env.robot_pos[0]), int(env.robot_pos[1])),
            #     (int(end_x), int(end_y)),
            #     3,
            # )
    else:
        pygame.draw.circle(
            win,
            RENDER_CONFIG["colors"]["robot"]["body"],
            (int(env.robot_pos[0]), int(env.robot_pos[1])),
            env.robot_radius,
        )
        # end_x = env.robot_pos[0] + (env.robot_radius + 5) * np.cos(np.radians(env.angle))
        # end_y = env.robot_pos[1] + (env.robot_radius + 5) * np.sin(np.radians(env.angle))
        # pygame.draw.line(
        #     win,
        #     RENDER_CONFIG["colors"]["robot"]["direction"],
        #     (int(env.robot_pos[0]), int(env.robot_pos[1])),
        #     (int(end_x), int(end_y)),
        #     3,
        # )

    # ==============================================
    # UI TEXT
    # ==============================================
    delivered_text = env.font.render(
        f"Delivered: {len(env.delivered_tables)}/{len(env.tables)}", True, RENDER_CONFIG["colors"]["text"]
    )
    step_reward_text = env.font.render(
        f"Step: {env.step_count}    Reward: {env.total_reward}", True, RENDER_CONFIG["colors"]["text"]
    )
    win.blit(step_reward_text, (10, 10))
    win.blit(delivered_text, (10, 40))

    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])
