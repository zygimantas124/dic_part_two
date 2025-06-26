# Data Intelligence Challenge (2AMC15)

## Welcome

Welcome to the **Delivery Robot RL Simulation Project**!  
This repository provides a complete framework for developing, training, evaluating, and analyzing reinforcement learning (RL) agents in a simulated office environment. The primary goal is to train agents (using DQN and PPO algorithms) to deliver medication within elderly care facilities while navigating through obstacles, walls, and variable environments. The project is modular, extensible, and designed for both research and educational use.

---

## Project Structure and Module Overview

### **Root-Level Scripts**
#### `main.py`
**Purpose:**  
*The entry point for training or evaluating agents.*    
**Details:**  
- Provides a unified command-line interface for starting both training and evaluation of RL agents.
- Parses extensive arguments for agent type, environment, hardware, and run modes.
- Initializes logging and manages logs.
- Calls the appropriate training or evaluation routine depending on the user's options.
- Cleans up previous logs, ensuring reproducible runs.
- Includes sample usage comments for easy command-line experimentation.
---
## **Key Directories and Their Contents**
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
### 1. **Agents**

### 2. **Office**

### 3. **Experiments**

### 4. **Saved Qnets**  

### 5. **Final Plots**  

### 6. **Results**  

### 7. **Util**