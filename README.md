# DDP Trial: PyTorch Distributed Data Parallel

This repository provides minimal examples demonstrating how to train a simple neural network using PyTorch. It includes both a single-device standard approach and a multi-device Distributed Data Parallel (DDP) approach.

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support if running on NVIDIA GPUs)

## Setup Instructions

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the environment:**
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

### 1. Single Device (Standard Training)

This script trains the model on a single GPU if available, or falls back to the CPU. It is a simplified version that does not use `torchrun` or DDP wrappers.

```bash
python single_gpu_script.py
```

### 2. Multi-GPU (Distributed Data Parallel)

This script is meant to be run using `torchrun` to launch multiple processes (one per GPU).

**To run on a single machine with 2 GPUs:**

```bash
torchrun --nproc_per_node=2 DDP-script-torchrun.py
```

*Note: If you have a different number of GPUs, change the `--nproc_per_node` argument accordingly.*

**To run on a single GPU (or CPU for testing):**

```bash
python DDP-script-torchrun.py
```