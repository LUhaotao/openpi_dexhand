#!/usr/bin/env bash
set -e

cd /data/openpi_dexhand

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py pi05_franka_xhand_flower_5_6_prompt \
  --exp-name=my_experiment \
  --overwrite
  