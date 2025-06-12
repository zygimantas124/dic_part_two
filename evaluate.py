import numpy as np
import heapq
import os
import torch
import time
import logging
from argparse import ArgumentParser
from office.delivery_env import DeliveryRobotEnv
from agents.DQN import DQNAgent

def parse_eval_args(argv=None):
    p = ArgumentParser(description="DQN Agent Evaluation Script")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the trained PyTorch model file (e.g., saved_Qnets/dqn_model.pth).")
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Number of episodes to run for evaluation.")
    p.add_argument("--render_mode", type=str, default=None,
                   choices=[None, "human", "rgb_array"],
                   help="Render mode for the environment (e.g., 'human', None).")
    p.add_argument("--n_actions", type=int, default=8,
                   help="Number of discrete actions the agent was trained with.")
    p.add_argument("--eval_epsilon", type=float, default=0.05,
                   help="Epsilon for exploration during evaluation (0 = greedy).")
    p.add_argument("--max_episode_steps", type=int, default=500,
                   help="Maximum steps per evaluation episode.")
    p.add_argument("--render_delay", type=float, default=0.03,
                   help="Delay between frames when rendering (in seconds). Only applies if render_mode='human'.")
    return p.parse_args(argv)

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

# TODO: during training save the agent and environment so we can call here in evaluate
def evaluate_agent(args):
    env = DeliveryRobotEnv(config = 'open_office_simple', render_mode=args.render_mode, show_walls=False, show_carpets=False, show_obstacles=False)
    obs_dim = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=args.n_actions,
        gamma=0.99,
        epsilon=args.eval_epsilon,
        epsilon_min=args.eval_epsilon,
        epsilon_decay_rate=1.0,
        alpha=0.0,
        batch_size=1,
        buffer_size=1,
        min_replay_size=1,
        target_update_freq=1e8,
        device=device,
        goal_buffer_size=50000,
        goal_fraction=0.4,
        logger=None
    )

    try:
        agent.load_model(args.model_path)
        agent.q_net.eval()
        agent.epsilon = args.eval_epsilon
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episode_rewards = []
    episode_lengths = []
    hausdorff_distances = []

    # Precompute optimal path once for same environment layout
    CELL_SIZE = 1 # To discretize the environment, for a-star optimal path
    print(f"Computing optimal path with cell size {CELL_SIZE}...")
    optimal_path = compute_optimal_path(env, CELL_SIZE)
    print(f"Optimal path length: {len(optimal_path)} cells")

    print(f"\nStarting evaluation for {args.n_episodes} episodes...")

    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        agent_path = []

        while not done and steps < args.max_episode_steps:
            if args.render_mode == "human":
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)
            # Record agent's position for Hausdorff distance
            agent_pos = tuple(env.robot_pos.tolist())
            agent_path.append(agent_pos)

            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Compute Hausdorff distance for this episode
        if not agent_path or not optimal_path:
            hd = 0.0
        else:
            hd = hausdorff_distance(agent_path, optimal_path)
        hd_norm = hd / np.hypot(env.width, env.height) # normalize by environment size

        # Store results
        hausdorff_distances.append(hd)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode+1}/{args.n_episodes}: Reward = {total_reward:.2f}, Steps = {steps}, Hausdorff Distance = {hd:.2f} (norm={hd_norm:.2f})")

    env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min Reward:     {np.min(episode_rewards):.2f}")
    print(f"Max Reward:     {np.max(episode_rewards):.2f}")
    print(f"Avg Steps:      {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Avg Hausdorff Dist: {np.mean(hausdorff_distances):.2f} ±{np.std(hausdorff_distances):.2f}")
    print("--------------------------")

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate_agent(args)
