# DDP Trial: PyTorch Distributed Data Parallel

This is a minimal example demonstrating how to train a simple neural network using PyTorch's Distributed Data Parallel (DDP). It's designed to run on multiple GPUs using `torchrun`.

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

## Running the Script

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