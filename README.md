# Data Intelligence Challenge (2AMC15)

## Welcome

Welcome to the **Delivery Robot RL Simulation Project**!  
This repository provides a complete framework for developing, training, evaluating, and analyzing reinforcement learning (RL) agents in a simulated office environment. The primary goal is to train agents (using DQN and PPO algorithms) to deliver medication within elderly care facilities while navigating through obstacles, walls, and variable environments. The project is modular, extensible, and designed for both research and educational use.

---

## Project Structure and Module Overview

### **Root-Level Scripts**
- **`main.py:`**  
The entry point for training or evaluating agents.
- Provides a unified command-line interface for starting both training and evaluation of RL agents.
- Parses commad-line arguments for choosing the RL algorithm, environment, hardware, and hyperparameters.
- Sets up logging and manages experiment life cycle.
- Calls training or evaluation routines based on user input.
- Cleans up previous logs, ensuring reproducible runs.
- Includes sample usage comments for easy command-line experimentation.
  - Example usage:
    ```sh
    python main.py --algo dqn --max_episodes 1000 --env_name open_office_simple --device cuda
    ```

---

- **`evaluate.py:`**  
Script for evaluating trained agents (for both DQN and PPO).
  - Loads a trained model and runs evaluation episodes.
  - Runs multiple evaluation episodes, collecting metrics like reward, path optimality (Hausdorff distance), and tortuosity.
  - Prints detailed statistical summaries, including learning curve area-under-curve.
  - Example usage:
    ```sh
    python evaluate.py --model_path logs/model.pth --n_episodes 10
    ```

---

- **`manual_control.py:`**
Provides a manual control interface using keyboard input (via pygame)..  
  - Lets a user drive the robot in the simulated environment using the keyboard.
  - Offers debug printing of state, table delivery, and raycasting data.
  - Very helpful for the cases of demonstration, debugging, and environment exploration.

---

- **`plotting.py`**  
Automates experiment analysis and visualization.
  - Reads log files, computes metrics, and generates plots for reward, success rate, etc.
  - Saves plots to the `final_plots/` directory.
  - Can batch process multiple experiment configurations in one run.


## **Key Directories and Their Contents**

### 1. **Agents**

### 2. **Office**

### 3. **Experiments**

### 4. **Saved Qnets**  

### 5. **Final Plots**  

### 6. **Results**  

### 7. **Util**