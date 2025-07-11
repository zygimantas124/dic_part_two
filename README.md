# Data Intelligence Challenge (2AMC15)

## Welcome

Welcome to the **Delivery Robot RL Simulation Project**!  
This repository provides a complete framework for developing, training, evaluating, and analyzing reinforcement learning (RL) agents in a simulated office environment. The primary goal is to train agents (using DQN and PPO algorithms) to deliver medication within elderly care facilities while navigating through obstacles, walls, and variable environments. The project is modular, extensible, and designed for both research and educational use.

---

## Project Structure and Module Overview

[General Note on the project: while running the files make sure that you are not using GPU, and instead all the code runs on CPU. Running the project on a GPU device takes much longer than running it on CPU]

### **Root-Level Scripts**
- **`main.py:`**  
The entry point for training or evaluating agents.
- Provides a unified command-line interface for starting both training and evaluation of RL agents.
- Parses command-line arguments for choosing the RL algorithm, environment, hardware, and hyperparameters.
- Sets up logging and manages experiment life cycle.
- Calls training or evaluation routines based on user input.
- Cleans up previous logs, ensuring reproducible runs.
- Includes sample usage comments for easy command-line experimentation.
  - Example usage (For a list of all the arguments that can be passed to the models please check the corresponding file):
    - Training Only:
        ```sh
        python main.py --algo dqn --max_episodes 1000 --env_name open_office_simple --device cpu
        python main.py --algo ppo --gamma 0.99 --max_episodes 1000 --max_episode_steps 4096 --k_epochs 5 --batch_size 512 --eps_clip 0.2 --seed 42 --epsilon_start 1 --epsilon_min 0.1 --epsilon_decay 0.999 --device cpu
        ```
    - Evaluation only (load existing model)
        ```sh
        python main.py --algo name --evaluate_only --load_model_path logs/my_model.pth --eval_episodes 20 --device cpu
        ```
    - Train then Evaluate (complete pipeline - with and without saving)
        ```sh
        python main.py --algo ppo --max_episodes 80 --evaluate_after_training --eval_episodes 10 --device cpu
        python main.py --algo ppo --max_episodes 80 --evaluate_after_training --eval_episodes 10 --save_model_path logs/my_model --device cpu
        ```
---

- **`evaluate.py:`**  
Script for evaluating trained agents (for both DQN and PPO).
  - Loads a trained model and runs evaluation episodes.
  - Runs multiple evaluation episodes, collecting metrics like reward, path optimality (Hausdorff distance), and tortuosity.
  - Prints detailed statistical summaries, including learning curve area-under-curve.
  - Example usage:
    ```sh
    python evaluate.py --model_path logs/my_model.pth --n_episodes 10
    ```

---

- **`manual_control.py:`**
Provides a manual control interface using keyboard input (via pygame)..  
  - Lets a user drive the robot in the simulated environment using the keyboard.
  - Offers debug printing of state, table delivery, and raycasting data.
  - Very helpful for the cases of demonstration, debugging, and environment exploration.

---

- **`plotting.py:`**  
Automates experiment analysis and visualization.
  - Reads log files, computes metrics, and generates plots for reward, success rate, etc.
  - Saves plots to the `final_plots/` directory.
  - Can batch process multiple experiment configurations in one run.


## **Key Directories and Their Contents**

### 1. **Agents (RL Agent Implementation)**
- **`DQN.py`**  
Implements the Deep Q-Network (DQN) agent. 
  - Defines the neural network (QNetwork) for Q-value estimation.
  - Implements a replay buffer to store and sample past experiences, critical for stable training.
  - Defines neural network architectures for value estimation.
  - Encapsulates sampling, learning, and action selection logic.

---

- **`PPO.py`**  
Implements the Proximal Policy Optimization (PPO) agent. 
  - Contains the PPOAgent class and Actor-Critic neural network.
  - Implements the Proximal Policy Optimization algorithm.
  - Uses a rollout buffer for trajectory storage.
  - Handles policy and value updates, action sampling, and advantage calculation.

### 2. **Office**
- **`delivery_env.py`**  
Defines the main delivery robot environment for RL training.
  - Inherits from Gymnasium’s Env class for compatibility with standard RL workflows.
  - Manages all aspects of the simulation: world layout, robot movement, object placement, reward calculation, and state observation.
  - Supports multiple configuration options for walls, obstacles, carpets, and ray-based sensors.
  - Integrates with rendering and raycasting modules for visualization and rich observations.

---

- **`render.py`**  
Handles rendering and visualization of the environment.  
  - Uses pygame to draw the environment, robot, and objects.

---

- **`raycasting.py`**  
Provides mathematical routines for simulating robot sensors.* 
  - Implements ray-segment intersection logic,  simulating lidar-like sensors.
  - Used to provide agents with distance-to-obstacle information in their observations.

### 3. **Experiments**
  - Contains configuration files for batch-running experiments.
  - Each `.txt` file defines a set of hyperparameters and environment settings.

### 4. **Saved Qnets**  
  - Directory for saved Q-network models or checkpoints.

### 5. **Final Plots**  
  - Output directory for plots generated by `plotting.py`.

### 6. **Results**  
  - Stores summary files and raw results from each experiment.

### 7. **Util**
- **`helpers.py`**  
General helper for argument parsing, logging, and reproducibility.
  - Provides argument parsing that supports comments and config files.
  - Sets up loggers and manages random seeds for reproducible experiments.

- **`training.py`**  
Manages the agent-environment training cycle.
  - Initializes environments and agents according to parsed parameters.
  - Manages episodic training, model saving/loading, and evaluation hooks.

- **`significance_analysis.py`**  
Supports statistical analysis of experimental results.
  - Example usage:
  ```sh 
  python util/significance_analysis.py @experiments/DQN_walls_obst_rayc.txt
  python util/significance_analysis.py @experiments/PPO_walls_no_obst_no_rayc.txt
  ```

[Note: running this file might take a long time due to the comprehensive analysis it does, you can see the result of running this file on the report].
  - Runs multiple repeats of training/evaluation to measure metrics variability.
  - Computes confidence intervals and statistical summaries for reporting.
  - Saves results to JSON for later analysis.