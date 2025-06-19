import os
import random
import logging
import heapq
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv

import numpy as np
import torch
from tqdm import tqdm


# ---------- Argument Parsing ----------
def parse_args(argv=None):
    p = ArgumentParser(
        description="Unified PPO/DQN Training Script",
        fromfile_prefix_chars='@'
    )

    # --- System ---
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="auto",
                   help="Device to use for training (cpu or cuda).")

    # --- Algorithm ---
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True,
                   help="Algorithm to train: 'ppo' or 'dqn'")

    # --- Common Training Settings ---
    p.add_argument("--render_mode", type=str, default=None,
                   choices=["human", "rgb_array", None],
                   help="Render mode (None, 'human', or 'rgb_array').")
    p.add_argument("--n_actions", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--max_episodes", type=int, default=500)
    p.add_argument("--max_episode_steps", type=int, default=512)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_model_path", type=str, default=None)
    p.add_argument("--load_model_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # --- PPO-specific ---
    p.add_argument("--eps_clip", type=float, default=0.3)
    p.add_argument("--k_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)

    # --- DQN-specific ---
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_decay", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--buffer_size", type=int, default=int(1e5))
    p.add_argument("--min_replay_size", type=int, default=int(1e5))
    p.add_argument("--target_update_freq", type=int, default=int(5e4))
    p.add_argument("--goal_buffer_size", type=int, default=50000)
    p.add_argument("--goal_fraction", type=float, default=0.4)

    # --- Environment config ---
    p.add_argument("--env_name", type=str, default="open_office_simple",
                   help="Predefined environment config name (e.g., 'simple', 'complex', 'open_office').")
    p.add_argument("--show_walls", action="store_true", help="Render walls.")
    p.add_argument("--hide_walls", dest="show_walls", action="store_false")
    p.set_defaults(show_walls=True)

    p.add_argument("--show_obstacles", action="store_true", help="Render people and furniture.")
    p.add_argument("--hide_obstacles", dest="show_obstacles", action="store_false")
    p.set_defaults(show_obstacles=True)

    p.add_argument("--show_carpets", action="store_true", help="Render carpets.")
    p.add_argument("--hide_carpets", dest="show_carpets", action="store_false")
    p.set_defaults(show_carpets=False)

    p.add_argument("--use_flashlight", action="store_true", help="Enable flashlight cone rendering.")
    p.add_argument("--use_raycasting", action="store_true", help="Enable raycasting input to agent.")

    return p.parse_args(argv)



# ---------- Seed Setup ----------
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Logging Setup ----------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_level=logging.INFO, log_file="training.log"):
    logger = logging.getLogger("trainer")
    logger.setLevel(log_level)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# ---------- Hausdorff Distance functions ----------
# --- RL Metric: Hausdorff Distance ---
def hausdorff_distance(path_a, path_b):
    """Compute the symmetric Hausdorff distance between two 2D point lists."""
    def directed(a, b):
        # for each p in a, find min dist to any q in b, then take max
        dists = []
        for px, py in a:
            min_d = min((px - qx)**2 + (py - qy)**2 for qx, qy in b)
            dists.append(min_d)
        return max(dists)**0.5 if dists else 0.0
    return max(directed(path_a, path_b), directed(path_b, path_a))

# Supportive functions and A-star to compute optimal path the environment
def build_occupancy_grid(env, cell_size):
    w, h = env.width, env.height
    cols = int(w // cell_size)
    rows = int(h // cell_size)
    grid = np.zeros((rows, cols), dtype=np.uint8)
    def mark_rects(rects):
        for x, y, rw, rh in rects:
            x0 = int(max(0, (x - env.robot_radius) // cell_size))
            y0 = int(max(0, (y - env.robot_radius) // cell_size))
            x1 = int(min(cols - 1, (x + rw + env.robot_radius) // cell_size))
            y1 = int(min(rows - 1, (y + rh + env.robot_radius) // cell_size))
            grid[y0:y1+1, x0:x1+1] = 1
    # walls and obstacles as blocked
    mark_rects(env.walls)
    for ox, oy, orad in env.obstacles:
        cx = int(ox // cell_size)
        cy = int(oy // cell_size)
        r = int(np.ceil((orad + env.robot_radius) / cell_size))
        y0, y1 = max(0, cy-r), min(rows-1, cy+r)
        x0, x1 = max(0, cx-r), min(cols-1, cx+r)
        for yy in range(y0, y1+1):
            for xx in range(x0, x1+1):
                grid[yy, xx] = 1
    return grid

def astar(grid, start, goal):
    """A* on a 4-connected grid: returns list of (x, y) cell centers."""
    rows, cols = grid.shape
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start, goal), 0, start, None)]
    came_from = {}
    cost_so_far = {start: 0}
    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        if current in came_from and parent is not None:
            # visited with a better cost
            continue
        if parent is not None:
            came_from[current] = parent
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nbr = (current[0]+dx, current[1]+dy)
            if 0 <= nbr[0] < cols and 0 <= nbr[1] < rows and grid[nbr[1], nbr[0]]==0:
                new_cost = cost_so_far[current] + 1
                if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                    cost_so_far[nbr] = new_cost
                    priority = new_cost + h(nbr, goal)
                    heapq.heappush(open_set, (priority, new_cost, nbr, current))
    return []

# TODO: seems to go wrong still as distance is 1 for optimal path, seems strange,,,,
def compute_optimal_path(env, cell_size):
    """Compute an A* path (cell centers) from start to first table."""
    grid = build_occupancy_grid(env, cell_size)

    sx, sy = env.start_pos
    tx, ty, tw, th = env.tables[0]
    # cell coords
    start_cell = (int(sx//cell_size), int(sy//cell_size))
    goal_center = (tx + tw/2, ty + th/2)
    goal_cell = (int(goal_center[0]//cell_size), int(goal_center[1]//cell_size))
    print("start_cell =", start_cell, "goal_cell =", goal_cell)   

    # DEBUG for a-star: print grid and start/goal cells
    print(f"Start cell: {start_cell}")
    print(grid[
        max(0, start_cell[1]-2):start_cell[1]+3,
        max(0, start_cell[0]-2):start_cell[0]+3
    ])
    print(f"Goal cell: {goal_cell}")
    print(grid[
        max(0, goal_cell[1]-2):goal_cell[1]+3,
        max(0, goal_cell[0]-2):goal_cell[0]+3
    ])

    cell_path = astar(grid, start_cell, goal_cell)
    print("cell_path:", cell_path[:10], "...", "len=", len(cell_path))
    # convert cells to real positions
    real_path = []
    for cx, cy in cell_path:
        real_x = cx*cell_size + cell_size/2
        real_y = cy*cell_size + cell_size/2
        real_path.append((real_x, real_y))
    return real_path

def compute_tortuosity(path):
    """
    Compute the tortuosity of a path as total angular change divided by path length.
    """
    path = np.array(path)
    if len(path) < 3: # not enough points to compute angles
        return 0.0
    total_turn = 0.0
    total_length = 0.0 
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i] # vectors between points
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2) # avoid division by zero
        if norm1 == 0 or norm2 == 0: 
            continue
        angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        total_turn += abs(angle) # accumulate total angular change
        total_length += norm1   
    # Add the last segment length
    total_length += np.linalg.norm(path[-1] - path[-2])
    return total_turn / total_length if total_length > 0 else 0.0 # return average angle change per unit 

def make_env(env_config, render_mode=None, show_walls=True, show_obstacles=True, show_carpets=False, use_flashlight=False, use_raycasting=False):
    return DeliveryRobotEnv(
        custom_config=env_config,
        render_mode=render_mode,
        show_walls=show_walls,
        show_obstacles=show_obstacles,
        show_carpets=show_carpets,
        use_flashlight=use_flashlight,
        use_raycasting=use_raycasting
    )
