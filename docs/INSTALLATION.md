# Installation Guide

This document provides detailed instructions for setting up the IDPR project on different operating systems.

## Prerequisites

Before installing IDPR, ensure you have the following:

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM (8GB+ recommended for parallel training)
- CUDA 11.8+ (optional, for GPU acceleration)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IDPR-Intelligent-Drone-Positioning-and-Routing.git
cd IDPR-Intelligent-Drone-Positioning-and-Routing
```

### 2. Create a Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following key packages:

- **gymnasium**: Reinforcement learning environment framework
- **numpy**: Numerical computing library
- **torch**: Deep learning framework (CPU version by default)
- **stable-baselines3**: PPO and other RL algorithms
- **sb3-contrib**: Additional RL algorithms including MaskablePPO
- **scipy**: Scientific computing utilities
- **optuna**: Hyperparameter optimization framework
- **tensorboard**: Training visualization tool

### 5. Verify Installation

To verify that all dependencies are correctly installed, run:

```bash
python -c "import gymnasium; import torch; import stable_baselines3; print('All dependencies installed successfully!')"
```

## Optional: GPU Support

If you have an NVIDIA GPU and want to accelerate training, install the GPU version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then verify GPU availability:

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

**Solution:** Ensure you have activated the virtual environment and installed requirements:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" during training

**Solution:** Reduce the number of parallel environments:
```bash
python 5GML.py --n-envs 8  # Default is 32
```

### Issue: "ImportError: cannot import name 'MaskablePPO'"

**Solution:** Ensure sb3-contrib is installed:
```bash
pip install sb3-contrib --upgrade
```

### Issue: Slow training on CPU

**Solution:** Install PyTorch with GPU support (see GPU Support section above) or reduce the number of timesteps:
```bash
python 5GML.py --timesteps 10000000  # Default is 40,000,000
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4GB | 8GB+ |
| CPU Cores | 4 | 8+ |
| Disk Space | 2GB | 10GB+ |
| GPU | None | NVIDIA with 4GB+ VRAM |

## Docker Support (Optional)

If you prefer to use Docker, create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/5GML.py"]
```

Build and run the Docker image:

```bash
docker build -t idpr:latest .
docker run -it --rm idpr:latest
```

## Next Steps

After successful installation, proceed to the [README.md](../README.md) to learn how to run the training script.
