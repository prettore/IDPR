# Architecture and Code Overview

This document describes the architecture and key components of the IDPR project.

## System Architecture

The IDPR system consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                   5GML.py (Main Training Script)             │
│  - Argument parsing and configuration management             │
│  - Environment setup (single and parallel)                   │
│  - PPO/MaskablePPO agent initialization                      │
│  - Training loop with callbacks and monitoring               │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   drone_env.py       │  │ calculaProbLink_otimizado.py │
│  (Gymnasium Env)     │  │  (Link Reliability Calc)     │
│                      │  │                              │
│ - State/Action space │  │ - Friis propagation model    │
│ - Reward computation │  │ - Log-normal fading          │
│ - Zone avoidance     │  │ - Link probability caching   │
│ - Network metrics    │  │ - RadioParams dataclass      │
└──────────────────────┘  └──────────────────────────────┘
```

## Core Modules

### 1. 5GML.py - Main Training Script

**Purpose:** Orchestrates the entire training pipeline for the PPO agent.

**Key Functions:**

- `set_global_seeds(seed)`: Sets random seeds for reproducibility
- `parse_args()`: Parses command-line arguments into TrainConfig
- `TrainConfig`: Dataclass containing all training hyperparameters
- `create_env()`: Initializes single or parallel Gymnasium environments
- `train_ppo()`: Main training loop with callbacks

**Key Classes:**

- `TrainConfig`: Configuration dataclass with 50+ parameters
  - RL infrastructure (timesteps, n_envs, batch_size, etc.)
  - Network architecture (net_width, net_layers, activation)
  - Environment parameters (grid_size, num_drones, zones)
  - Hyperparameter optimization settings (Optuna)

**Training Flow:**

1. Parse configuration from command-line arguments
2. Set random seeds for reproducibility
3. Create vectorized environment (DummyVecEnv or SubprocVecEnv)
4. Optionally apply VecNormalize for observation/reward normalization
5. Initialize PPO or MaskablePPO agent
6. Set up callbacks (EvalCallback, CheckpointCallback)
7. Train for specified number of timesteps
8. Save best model and final model

### 2. drone_env.py - Gymnasium Environment

**Purpose:** Implements the drone positioning and routing problem as a Gymnasium environment.

**Key Classes:**

- `DroneEnv`: Main environment class inheriting from gym.Env
  - Implements `reset()`, `step()`, and `render()` methods
  - Manages drone positions, communication links, and forbidden zones
  - Computes rewards based on network metrics

**State Space:**

The observation includes:
- Drone positions (N×2 array)
- Adjacency matrix (N×N binary matrix)
- Zone map (optional, downsampled)
- Link-zone crossing information (optional)
- Normalized network metrics

**Action Space:**

For each drone type:
- **First drone**: Link creation/removal actions only
- **Intermediate drones**: Movement (8 directions × 6 distances) + link actions
- **Last drone**: No actions (destination only)

**Reward Components:**

The reward function combines multiple objectives:
- Reliability: Penalizes low end-to-end packet delivery probability
- Latency: Penalizes long paths (high hop count)
- Throughput: Rewards high data transmission capacity
- Redundancy: Penalizes excessive drone count
- Structure: Encourages connected, efficient topologies

**Zone Avoidance:**

- Forbidden zones are randomly generated or loaded from file
- Links crossing zones are penalized
- Drones avoid positioning in zones

### 3. calculaProbLink_otimizado.py - Link Reliability Module

**Purpose:** Calculates link reliability using the Friis free-space propagation model with log-normal fading.

**Key Functions:**

- `prob_link_literature(d_m, params)`: Computes link success probability at distance d
  - Uses Friis path loss model: PL(d) = PL(d₀) + 10n·log₁₀(d/d₀)
  - Models fading with log-normal distribution (CDF of standard normal)
  - Returns probability p ∈ [0, 1]

- `calcula_confiabilidade_iterativa(adjacency, drone_positions, params)`: Computes end-to-end reliability
  - Propagates link failure probabilities through the network
  - Returns network reliability (probability packet reaches destination)

**Key Classes:**

- `RadioParams`: Dataclass containing radio parameters
  - `pt_dbm`: Transmit power (default: 19.54 dBm ≈ 90 mW)
  - `gt_db`, `gr_db`: Transmit/receive antenna gains
  - `freq_hz`: Carrier frequency (default: 2.4 GHz)
  - `n`: Path loss exponent (default: 2.3)
  - `pmin_dbm`: Receiver sensitivity (default: -83 dBm)
  - `sigma_db`: Log-normal fading standard deviation (default: 4 dB)

**Performance Optimizations:**

- LRU caching of link probability calculations (65536 entries)
- Vectorized distance calculations using NumPy
- Sparse matrix operations for large networks

## Data Flow

### Training Iteration

```
1. Environment Reset
   ├─ Generate random drone positions
   ├─ Generate random forbidden zones
   └─ Compute initial adjacency matrix

2. Agent Observation
   ├─ Extract state features
   ├─ Normalize observations
   └─ Pass to policy network

3. Action Selection
   ├─ Policy network forward pass
   ├─ Sample action from distribution
   └─ Apply action masking (optional)

4. Environment Step
   ├─ Update drone positions
   ├─ Update communication links
   ├─ Recalculate adjacency matrix
   ├─ Compute link reliabilities
   └─ Calculate network metrics

5. Reward Computation
   ├─ Evaluate reliability
   ├─ Evaluate latency
   ├─ Evaluate throughput
   ├─ Combine objectives (lexicographic)
   └─ Return reward and done flag

6. Policy Update (every n_steps)
   ├─ Compute advantages (GAE)
   ├─ Compute policy loss (PPO)
   ├─ Compute value loss
   ├─ Backpropagation
   └─ Update network weights
```

## Configuration Management

The `TrainConfig` dataclass organizes 50+ hyperparameters into logical groups:

| Group | Parameters | Purpose |
|-------|-----------|---------|
| RL Infrastructure | seed, timesteps, n_envs, mask_actions | Core RL setup |
| PPO Hyperparameters | n_steps, batch_size, n_epochs, gamma, gae_lambda | Algorithm tuning |
| Schedules | lr_start, lr_end, clip_start, clip_end, ent_start, ent_end | Learning dynamics |
| Network Architecture | net_width, net_layers, activation, policy_arch | Neural network design |
| Evaluation | eval_freq, n_eval_episodes, deterministic_eval | Performance monitoring |
| Checkpointing | checkpoint_freq, best_path, ckpt_path, final_path | Model persistence |
| Environment | grid_size, num_drones, variable_drones, square_size_m | Problem definition |
| Zones | zonas_count, zona_w, zona_h, p_sem_zona, zonas_file | Zone parameters |
| Observation | obs_zone_map, zone_map_scale, obs_edge_cross | State features |
| HPO | hpo_optuna, hpo_trials, hpo_budget, hpo_pruner | Hyperparameter optimization |

## Extensibility

The architecture is designed for easy extension:

1. **Custom Reward Functions**: Modify `DroneEnv.compute_reward()` to implement different objectives
2. **Alternative Algorithms**: Replace PPO with other RL algorithms from stable-baselines3
3. **Different Propagation Models**: Extend `calculaProbLink_otimizado.py` with alternative path loss models
4. **Custom Environments**: Create new environment classes inheriting from `DroneEnv`
5. **Visualization**: Add custom rendering in `DroneEnv.render()`

## Performance Considerations

- **Vectorization**: Uses 32 parallel environments by default for efficient sampling
- **Caching**: Link probability calculations are cached to avoid recomputation
- **Sparse Matrices**: Optional use of sparse representations for large networks
- **GPU Support**: PyTorch automatically uses GPU if available
- **Batch Processing**: PPO processes large batches for stable gradient estimates

## Dependencies and Versions

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | ≥0.28.1 | Environment framework |
| numpy | ≥1.24.0 | Numerical computing |
| torch | ≥2.0.0 | Deep learning |
| stable-baselines3 | ≥2.0.0 | PPO algorithm |
| sb3-contrib | ≥2.0.0 | MaskablePPO |
| scipy | ≥1.10.0 | Scientific computing |
| optuna | ≥3.2.0 | Hyperparameter optimization |
| tensorboard | ≥2.13.0 | Training visualization |
