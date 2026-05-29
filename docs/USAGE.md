# Usage Guide

This document provides comprehensive examples and explanations for using the IDPR project.

## Quick Start

To run a basic training session with default parameters:

```bash
cd src
python 5GML.py
```

This will start training a PPO agent on the drone positioning and routing problem with default settings.

## Training Scenarios

### Scenario 1: High-Throughput Optimization

For scenarios prioritizing maximum throughput with relaxed latency constraints:

```bash
python 5GML.py \
  --grid-size 100 \
  --num-drones 13 \
  --timesteps 20000000 \
  --n-envs 32 \
  --batch-size 12288 \
  --ent-coef 0.01
```

**Parameters explained:**
- `--grid-size 100`: Operational area is 100×100 cells (2000×2000 meters)
- `--num-drones 13`: Deploy 13 drones in the network
- `--timesteps 20000000`: Train for 20 million environment steps
- `--n-envs 32`: Use 32 parallel environments for faster training
- `--batch-size 12288`: Process 12,288 samples per gradient update
- `--ent-coef 0.01`: Entropy coefficient for exploration (lower = less exploration)

### Scenario 2: Low-Latency URLLC Optimization

For Ultra-Reliable Low-Latency Communication (URLLC) scenarios:

```bash
python 5GML.py \
  --grid-size 100 \
  --num-drones 25 \
  --timesteps 40000000 \
  --n-envs 32 \
  --batch-size 24576 \
  --ent-coef 0.015 \
  --vf-coef 1.0 \
  --gamma 0.995
```

**Parameters explained:**
- `--num-drones 25`: More drones for redundancy and low latency
- `--timesteps 40000000`: Extended training for convergence
- `--batch-size 24576`: Larger batch for stable learning
- `--ent-coef 0.015`: Higher entropy for better exploration
- `--vf-coef 1.0`: Higher value function coefficient for stability
- `--gamma 0.995`: Higher discount factor for long-term planning

### Scenario 3: Contested Environment with Zone Avoidance

For military operations with forbidden zones:

```bash
python 5GML.py \
  --grid-size 100 \
  --num-drones 20 \
  --timesteps 30000000 \
  --p-sem-zona 0.5 \
  --zonas-count 3 \
  --zona-w 15 \
  --zona-h 15 \
  --obs-zone-map 1 \
  --obs-edge-cross 1
```

**Parameters explained:**
- `--p-sem-zona 0.5`: 50% of episodes include forbidden zones
- `--zonas-count 3`: Generate 3 forbidden zones per episode
- `--zona-w 15`, `--zona-h 15`: Each zone is 15×15 cells
- `--obs-zone-map 1`: Include zone map in agent observation
- `--obs-edge-cross 1`: Include link-zone crossing information

## Advanced Configuration

### Custom Neural Network Architecture

Specify a custom network architecture with different layer sizes:

```bash
python 5GML.py \
  --net-width 512 \
  --net-layers 4 \
  --activation relu \
  --policy-arch "512,512,256,128"
```

**Parameters explained:**
- `--net-width 512`: Base width for hidden layers
- `--net-layers 4`: Use 4 hidden layers
- `--activation relu`: Use ReLU activation (alternatives: tanh)
- `--policy-arch "512,512,256,128"`: Explicit layer sizes

### Learning Rate Scheduling

Use cosine annealing for learning rate schedule:

```bash
python 5GML.py \
  --lr-start 1e-3 \
  --lr-end 1e-5 \
  --clip-start 0.2 \
  --clip-end 0.05
```

This gradually reduces learning rate and PPO clip range during training.

### Entropy Scheduling

Control exploration over time:

```bash
python 5GML.py \
  --ent-start 0.05 \
  --ent-end 0.001
```

This starts with high exploration and gradually reduces it.

## Hyperparameter Optimization

Use Optuna for automatic hyperparameter tuning:

```bash
python 5GML.py \
  --hpo-optuna 1 \
  --hpo-trials 30 \
  --hpo-budget 5000000 \
  --hpo-pruner asha \
  --hpo-n-jobs -1
```

**Parameters explained:**
- `--hpo-optuna 1`: Enable Optuna HPO
- `--hpo-trials 30`: Run 30 optimization trials
- `--hpo-budget 5000000`: Budget of 5 million timesteps per trial
- `--hpo-pruner asha`: Use ASHA pruner for early stopping
- `--hpo-n-jobs -1`: Use all available CPU cores

## Monitoring Training

### TensorBoard Visualization

Monitor training metrics in real-time:

```bash
tensorboard --logdir ./ppo_drone_tensorboard
```

Then open your browser to `http://localhost:6006`

Key metrics to monitor:
- **rollout/ep_rew_mean**: Average episode reward (higher is better)
- **rollout/ep_len_mean**: Average episode length
- **train/policy_loss**: Policy network loss (should decrease)
- **train/value_loss**: Value network loss (should decrease)
- **train/entropy_loss**: Entropy loss (controls exploration)

### Checkpointing and Resuming

Save checkpoints during training:

```bash
python 5GML.py \
  --checkpoint-freq 1000000 \
  --ckpt-path ./my_checkpoints
```

Resume training from a checkpoint:

```bash
python 5GML.py \
  --use-model continue \
  --best-path ./my_checkpoints/rl_model_1000000_steps.zip
```

## Performance Evaluation

After training, evaluate the learned policy:

```bash
python 5GML.py \
  --use-model continue \
  --best-path ./ppo_best_model \
  --n-eval-episodes 500 \
  --deterministic-eval true
```

This runs 500 evaluation episodes with the best learned policy in deterministic mode (no exploration).

## Troubleshooting

### Training is too slow

**Solution:** Increase parallel environments:
```bash
python 5GML.py --n-envs 64
```

### High variance in rewards

**Solution:** Increase batch size and reduce entropy coefficient:
```bash
python 5GML.py --batch-size 32768 --ent-coef 0.005
```

### Agent not learning

**Solution:** Increase learning rate and entropy:
```bash
python 5GML.py --lr-start 5e-4 --ent-coef 0.05
```

### Out of memory errors

**Solution:** Reduce parallel environments and batch size:
```bash
python 5GML.py --n-envs 8 --batch-size 4096
```

## Output Files

After training, the following files are generated:

| File/Directory | Purpose |
|---|---|
| `ppo_best_model/` | Best model checkpoint (lowest eval loss) |
| `ppo_final/model/` | Final trained model |
| `ppo_checkpoints/` | Periodic checkpoints |
| `ppo_eval/` | Evaluation results |
| `ppo_drone_tensorboard/` | TensorBoard logs |

## Next Steps

For more information about the methodology and results, see the main [README.md](../README.md).
