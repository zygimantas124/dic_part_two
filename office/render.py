import pygame
import numpy as np


def render_environment(env):
    win = env.window
    win.fill((240, 240, 240))

    if env.show_walls:
        for wall in env.walls:
            pygame.draw.rect(win, (0, 0, 0), wall)

    for i, table in enumerate(env.tables):
        color = (0, 200, 0) if i in env.delivered_tables else (255, 165, 0)
        pygame.draw.rect(win, color, table)
        pygame.draw.rect(win, (0, 0, 0), table, 2)

    if env.show_obstacles:
        for obs in env.obstacles:
            pygame.draw.circle(win, (255, 100, 100), (int(obs[0]), int(obs[1])), obs[2])
            pygame.draw.circle(win, (200, 0, 0), (int(obs[0]), int(obs[1])), obs[2], 2)

    if env.show_carpets:
        for carpet in env.carpets:
            pygame.draw.rect(win, (139, 69, 19), carpet)

    pygame.draw.circle(win, (0, 100, 255), (int(env.robot_pos[0]), int(env.robot_pos[1])), env.robot_radius)
    end_x = env.robot_pos[0] + (env.robot_radius + 5) * np.cos(np.radians(env.angle))
    end_y = env.robot_pos[1] + (env.robot_radius + 5) * np.sin(np.radians(env.angle))
    pygame.draw.line(win, (0, 50, 150), (int(env.robot_pos[0]), int(env.robot_pos[1])), (int(end_x), int(end_y)), 3)

    text = env.font.render(f"Delivered: {len(env.delivered_tables)}/{len(env.tables)}", True, (0, 0, 0))
    win.blit(text, (10, 10))
    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])
