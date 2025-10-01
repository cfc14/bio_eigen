# Eigenbot Reinforcement Learning Repo

This repository documents the Eigenbot team's reinforcement learning (RL) work using **legged_gym**, a derivative of NVIDIA's Isaac Gym.

It includes environments, robot models, and configurations tailored for training and testing legged robots.

## Learning Resources

* **CMU 10-703: Deep RL** - [Course Website](https://cmudeeprl.github.io/703website_f24/lectures/)

* **Sim2Real Lecture (Highly Recommended for Application)** - [Lecture 24](https://www.dropbox.com/scl/fi/aux0gnrw14b6n7q3vleua/sim2realS24.pdf?rlkey=z7ik66imupsz5q55yitvqm8i8&e=2&dl=0)

## Repository Structure

This repo, referred to internally as **bio_eigen**, consists of four main folders:

* ${\color{green}eigenbot/}$

* ${\color{green}isaacgym/}$

* ${\color{green}legged\\_gym/}$

  Core training and inference framework.

  📍Most work happens here: ${\color{green}legged\\_gym/legged\\_gym/env/base/legged\\_robot.py/}$

  
* ${\color{green}rsl\\_rl/}$
  RL algorithm implemnetations, including PPO and other on-policy/off-policy methods

### ${\color{green}legged\\_gym/legged\\_gym/env/base/legged\\_robot.py/}$
* Defines all core environment functions:
  * ${\color{green}step()}$ -> advances simulation by one step
  * ${\color{green}reset()}$ -> resets environment/robot state

* Contains **all reward functions** at the bottom of the file, following the naming pattern:
  
  ```python
  def _reward_<reward_name>(self)
  ```

    Rewards must be defined in the above format, after definition, the corresponding reward scales must be added into ${\color{green}legged\\_robot\\_config/}$
  * Example: * ${\color{green}\\_reward\\_tracking\\_lin\\_vel()}$

### ${\color{green}legged\\_gym/legged\\_gym/env/base/legged\\_robot\\_config.py/}$
* Centralized configuration file for:
  * Environment Setup
  * Terrain generation
  * Reward weights
  * Network architecture
  * Command sampling
  * Initial states
  * Control & assets
  * Domain randomization
    
## Terrain Configuration

This module defines how terrains are **generated, selected, and managed** for the legged robot environments
### ${\color{green}legged\\_gym/legged\\_gym/utils/terrain.py}$
#### Key Parts
* ${\color{green}Terrain}$ class
  * Initializes the terrain grid for multiple robots ${\color{green}(num\\_rows \  x \  num\\_cols)}$
  * Supports different terrain generation modes:
    * ${\color{green}randomized\\_terrain()}$ -> randomly generates terrain pieces
    * ${\color{green}curriculum()}$ -> terrains incresae in difficulty row by row
    * ${\color{green}selected_terrain()}$ -> uses a manually chosen terrain type
  * Stores **height maps** (${\color{green}height\\_field\\_raw}$) and **origins** for each sub-terrain.
* ${\color{green}make\\_terrain(choice,\ difficulty)}$:
  * Builds different terrain types (slopes, stairs, discrete obstacles, stepping stones, gaps, pits, ...) based on proportions + difficulty
* ${\color{green}add\\_terrain\\_to\\_map()}$:
  *   Places generated sub-terrains into the global map
  *   Sets each environment's origin (x, y, z)
* **Helper Functions**:
  * ${\color{green}gap\\_terrain()}$ -> creates a gap in the map
  * ${\color{green}pit\\_terrain()}$ -> creates a pit with depth

## Running

| Argument          | Type | Default         | Description                                                                 |
|-------------------|------|-----------------|-----------------------------------------------------------------------------|
| --task            | str  | "anymal_c_flat" | Name of the task/environment. Overrides config file if provided.            |
| --resume          | flag | False           | Resume training from a checkpoint.                                          |
| --experiment_name | str  | None            | Name of the experiment to run or load. Overrides config file.               |
| --run_name        | str  | "new"           | Name of the run (to distinguish runs within the same experiment). Overrides config file. |
| --expt_id         | str  | "00-001"        | Experiment ID tag (useful for structured naming). Overrides config file.    |
| --load_run        | str  | -1              | Run directory to load when --resume=True. If -1, loads the last run.        |
| --checkpoint      | int  | -1              | Model checkpoint to load. If -1, loads the latest checkpoint.               |
| --headless        | flag | False           | Run simulation without GUI (offscreen/headless mode).                       |
| --horovod         | flag | False           | Enable Horovod for distributed (multi-GPU) training.                        |
| --rl_device       | str  | "cuda:0"        | Device used by the RL algorithm (cpu, cuda:0, etc.).                        |
| --num_envs        | int  | Config default  | Number of environments to create. Overrides config file.                    |
| --seed            | int  | Config default  | Random seed for reproducibility. Overrides config file.                     |
| --max_iterations  | int  | Config default  | Maximum number of training iterations. Overrides config file.               |
| --show_heading    | flag | False           | Visualize robot’s heading direction in the viewer.                          |
| --rough_terrain   | flag | False           | Enable rough terrain (instead of flat ground).                              |
| --debug           | flag | False           | Disable Weights & Biases (wandb) logging (debug mode).                      |
| --no_wandb        | flag | False           | Run without wandb logging entirely.                                         |

Example usage:
* Train eigenbot on flat terrain with 4096 envs:
```bash
python train.py --task eigenbot_flat --num_envs 4096 --experiment_name locomotion_flat
```

# Branch Overview
This is the encoder branch as detailed below:

## Encoder Branch
*	Introduces encoder modules to process observation states.

*	Includes variants such as:

    1.	History Encoder → encodes past states for temporal context.

    2.	Privileged Encoder → leverages extra simulation-only information during training.

*	How to follow the flow:

    1.	Observation states defined in legged_gym/envs/base/legged_robot.py.

    2.	Passed into encoder modules.

    3.	Integrated into training via ${\color{green}rsl\\_rl}$ (${\color{green}on\\_policy\\_runner.py}$, ${\color{green}ppo.py}$, ${\color{green}vec\\_env.py}$, and ${\color{green}actor\\_critic.py}$).

## Depth-Encoder Branch
*	Extends the encoder framework by adding a Depth Encoder Module.

* Depth Encoder:

  *	Processes simulated depth maps (from sensors or render).

  *	Outputs latent features concatenated with standard encoders.

* Integrated seamlessly into the PPO pipeline in ${\color{green}rsl\\_rl}$.

👉 These two branches build upon the legged_robot observation space and connect into rsl_rl training pipelines, but add new ways of representing or enriching observations.


  
