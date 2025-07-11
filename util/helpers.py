import os
import sys
import random
import logging
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
import shlex


class CommentParser(ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        # Skip empty lines and comments
        arg_line = arg_line.strip()
        if not arg_line or arg_line.startswith("#"):
            return []
        return shlex.split(arg_line)


# ---------- Argument Parsing ----------
def parse_args(argv=None):
    p = CommentParser(description="Unified PPO/DQN Training Script", fromfile_prefix_chars="@")
    # --- System ---
    p.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="auto", help="Device to use for training (cpu or cuda)."
    )

    # --- Algorithm ---
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True, help="Algorithm to train: 'ppo' or 'dqn'")

    # --- Common Training Settings ---
    p.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=["human", "rgb_array", None],
        help="Render mode (None, 'human', or 'rgb_array').",
    )
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
    p.add_argument(
        "--warmstart", type=int, default=50, help="Number of episodes before epsilon decay starts (DQN only)"
    )

    # --- Environment config ---
    p.add_argument(
        "--env_name",
        type=str,
        default="open_office_simple",
        help="Predefined environment config name (e.g., 'simple', 'complex', 'open_office').",
    )
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

    # rewards
    p.add_argument("--reward_step", type=float, default=-0.01, help="Reward for taking a step")
    p.add_argument("--reward_collision", type=float, default=-1.0, help="Penalty for collision")
    p.add_argument("--reward_delivery", type=float, default=50.0, help="Reward for delivering")
    p.add_argument("--reward_carpet", type=float, default=-0.2, help="Penalty for moving over carpet")

    # --- Evaluation Control ---
    p.add_argument(
        "--evaluate_only", action="store_true", help="Skip training and only run evaluation on existing model."
    )
    p.add_argument("--evaluate_after_training", action="store_true", help="Run evaluation after training completes.")

    # --- Evaluation Settings ---
    p.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes to run during evaluation.")
    p.add_argument(
        "--eval_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for exploration during evaluation (DQN only, 0 = greedy).",
    )
    p.add_argument(
        "--eval_render_mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Render mode during evaluation (overrides main render_mode).",
    )
    p.add_argument(
        "--eval_render_delay",
        type=float,
        default=0.03,
        help="Delay between frames when rendering during evaluation (in seconds).",
    )

    args = p.parse_args(argv)

    # Capture the config file name if passed via @file.txt
    config_file = next((arg[1:] for arg in (argv or sys.argv[1:]) if arg.startswith("@")), None)
    setattr(args, "config_file", config_file)  # attach it to args

    return args


# ---------- Seed Setup ----------
def set_global_seed(seed):
    """
    Set global random seed for reproducibility across numpy, random, and torch.
    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed) # Set Python random seed
    np.random.seed(seed) # Set numpy random seed
    torch.manual_seed(seed) # Set PyTorch CPU seed
    torch.cuda.manual_seed_all(seed) # Set PyTorch GPU seed
    os.environ["PYTHONHASHSEED"] = str(seed) # Set environment variable for hash seed
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False 


# ---------- Logging Setup ----------
class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler writing log messages to tqdm.
    """
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
    """
    Setup a logger that writes to both console and file.
    Args:
        log_level (int): Logging level (default: logging.INFO).
        log_file (str): File to write logs to (default: "training.log").
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
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
    """
    Compute the symmetric Hausdorff distance between two 2D point lists.
    
    Args:
        path_a (list of tuples): First path as list of (x, y) tuples.
        path_b (list of tuples): Second path as list of (x, y) tuples.
    Returns:
        float: Symmetric Hausdorff distance between the two paths."""

    def directed(a, b):
        # for each p in a, find min dist to any q in b, then take max
        dists = []
        for px, py in a:
            # Find minimum distance from px, py to all on b
            min_d = min((px - qx) ** 2 + (py - qy) ** 2 for qx, qy in b)
            dists.append(min_d)
        # Return the maximum of these minimum distances
        # This is the directed Hausdorff distance
        # Then take the square root to get the actual distance
        return max(dists) ** 0.5 if dists else 0.0

    return max(directed(path_a, path_b), directed(path_b, path_a))


# A-star and support to compute optimal path benchmark the environment
def build_occupancy_grid(env, cell_size):
    """
    Build a grid with obstacles and walls for A* pathfinding.

    Args:
        env: The environment containing walls and obstacles.
        cell_size (float): Size of each grid cell in the environment.
    Returns:
        np.ndarray: Occupancy grid where 0 = free cell, 1 = blocked cell.
    """
    w, h = env.width, env.height
    cols = int(w // cell_size)
    rows = int(h // cell_size)
    grid = np.zeros((rows, cols), dtype=np.uint8)

    def mark_rects(rects):
        for x, y, rw, rh in rects:
            # Add buffer around obstacles for robot radius
            x0 = int(max(0, (x - env.robot_radius - 1) // cell_size))
            y0 = int(max(0, (y - env.robot_radius - 1) // cell_size))
            x1 = int(min(cols - 1, (x + rw + env.robot_radius + 1) // cell_size))
            y1 = int(min(rows - 1, (y + rh + env.robot_radius + 1) // cell_size))
            grid[y0 : y1 + 1, x0 : x1 + 1] = 1

    # walls and obstacles as blocked
    mark_rects(env.walls)

    for ox, oy, orad in env.obstacles:
        cx = int(ox // cell_size)
        cy = int(oy // cell_size)
        # Ensure sufficient buffer around circular obstacles
        r = int(np.ceil((orad + env.robot_radius + 1) / cell_size))
        y0, y1 = max(0, cy - r), min(rows - 1, cy + r)
        x0, x1 = max(0, cx - r), min(cols - 1, cx + r)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2:
                    grid[yy, xx] = 1
    return grid


def astar(grid, start, goal):
    """A* on a 4-connected grid: returns list of (x, y) cell centers.
    Args:
        grid (np.ndarray): 2D occupancy grid where 0 = free cell, 1 = blocked cell.
        start (tuple): Starting cell coordinates (x, y).
        goal (tuple): Goal cell coordinates (x, y).
    Returns:
        list: List of (x, y) tuples representing the path from start to goal.
        Returns an empty list if no path is found or if start/goal are invalid.
    """

    import heapq

    rows, cols = grid.shape

    # Validate start and goal positions
    if (
        start[1] >= rows
        or start[0] >= cols
        or goal[1] >= rows
        or goal[0] >= cols
        or start[0] < 0
        or start[1] < 0
        or goal[0] < 0
        or goal[1] < 0
    ):
        print(f"Invalid start {start} or goal {goal} for grid {grid.shape}")
        return []

    if grid[start[1], start[0]] == 1:
        print(f"Start position {start} is blocked!")
        return []

    if grid[goal[1], goal[0]] == 1:
        print(f"Goal position {goal} is blocked!")
        return []

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # A* algorithm initialization
    counter = 0 # Unique counter for each cell to break ties in priority queue
    open_set = [(0, start)] # Priority queue for open set, using f-score
    came_from = {} # To reconstruct the path
    g_score = {start: 0} # cost from start to current cell
    f_score = {start: h(start, goal)} # heuristic cost from start to goal
    closed_set = set() # Set of cells already evaluated

    # Directions for 4-connected grid (up, right, down, left)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # A* loop
    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)

        # If we reach the goal, reconstruct the path
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Explore neighbours
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check bounds
            if neighbor[0] < 0 or neighbor[0] >= cols or neighbor[1] < 0 or neighbor[1] >= rows:
                continue

            # Check if blocked
            if grid[neighbor[1], neighbor[0]] == 1:
                continue
            
            # Calculate tentative g-score
            tentative_g = g_score[current] + 1

            # If neighbor not in g_score or found a better path
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor)) # push with f-score

    print("A-star failed to find a path")
    return []


def compute_optimal_path(env, cell_size):
    """
    Compute an A* path (cell centers) from start to first table."""
    grid = build_occupancy_grid(env, cell_size)

    sx, sy = env.start_pos
    tx, ty, tw, th = env.tables[0]
    # cell coords
    start_cell = (int(sx // cell_size), int(sy // cell_size))
    goal_center = (tx + tw / 2, ty + th / 2)
    goal_cell = (int(goal_center[0] // cell_size), int(goal_center[1] // cell_size))
    print("start_cell =", start_cell, "goal_cell =", goal_cell)

    # Clear a radius around the start and goal cells
    rows, cols = grid.shape
    clear_radius = max(3, int(env.robot_radius // cell_size) + 2)

    # Clear around goal and start cells
    for cell in [start_cell, goal_cell]:
        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                new_y = cell[1] + dy
                new_x = cell[0] + dx
                if 0 <= new_y < rows and 0 <= new_x < cols:
                    if dx * dx + dy * dy <= clear_radius * clear_radius:
                        grid[new_y, new_x] = 0

    # ensure start position is clear
    if 0 <= start_cell[1] < rows and 0 <= start_cell[0] < cols:
        grid[start_cell[1], start_cell[0]] = 0

    print(f"Grid shape: {grid.shape}, Start: {start_cell}, Goal: {goal_cell}")
    print(f"Start cell blocked: {grid[start_cell[1], start_cell[0]] == 1}")
    print(f"Goal cell blocked: {grid[goal_cell[1], goal_cell[0]] == 1}")

    cell_path = astar(grid, start_cell, goal_cell)

    if not cell_path:
        print("WARNING: A star failed, no path found. Taking direct line.")
        return [(sx, sy), (goal_center[0], goal_center[1])]

    print(f"Cell path length: {len(cell_path)} cells")

    # Convert cells to real positions
    real_path = []
    for cx, cy in cell_path:
        real_x = cx * cell_size + cell_size / 2
        real_y = cy * cell_size + cell_size / 2
        real_path.append((real_x, real_y))

    return real_path

# ---------- Tortuosity function support ----------
# --- RL Metric: tortuosity ---

def compute_tortuosity(path):
    """
    Compute the tortuosity of a path as total angular change divided by path length.
    Args:
        path (list of tuples): List of (x, y) tuples representing the path.
    Returns:
        float: Average angle change per unit length of the path.
    """

    path = np.array(path)
    if len(path) < 3:  # not enough points to compute angles
        return 0.0
    
    total_turn = 0.0
    total_length = 0.0

    # Compute angles between consecutive segments
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]  # vectors between points
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)  # avoid division by zero
        if norm1 == 0 or norm2 == 0:
            continue
        angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        total_turn += abs(angle)  # accumulate total angular change
        total_length += norm1

    # Add the last segment length
    total_length += np.linalg.norm(path[-1] - path[-2])

    # Return average angle change per unit
    return total_turn / total_length if total_length > 0 else 0.0  
