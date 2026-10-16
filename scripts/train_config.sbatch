#!/usr/bin/env bash
#SBATCH --job-name=openpi-jax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/public/node01/users/lvrui/sbatch/sbatch_log/%x-%j.out
#SBATCH --error=/public/node01/users/lvrui/sbatch/sbatch_log/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch scripts/train_config.sbatch CONFIG.json" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

exec .venv/bin/python scripts/train.py --config "$1"
