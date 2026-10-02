#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

module load cuda/12.2

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

uv venv --python=3.10
source .venv/bin/activate
uv pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
uv pip install ninja
# uv sync
uv pip install -e .

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"