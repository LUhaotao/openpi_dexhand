#!/usr/bin/env bash
set -e

cd /data/openpi_dexhand

export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# export TMPDIR=/data/tmp
# export TEST_TMPDIR=/data/tmp
# export XDG_CACHE_HOME=/data/.cache
# export JAX_COMPILATION_CACHE_DIR=/data/jax_cache
# export XLA_FLAGS=--xla_gpu_per_fusion_autotune_cache_dir=/data/jax_cache/xla_gpu_per_fusion_autotune_cache_dir

# mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR/xla_gpu_per_fusion_autotune_cache_dir"

# uv run scripts/train.py pi05_franka_xhand_flower_streaming \
#   --exp-name=pi05_streaming_franka_xhand_flower \
#   --model.streaming \
#   --model.streaming-chunk-size=1 \
#   --model.streaming-constant-weight=0.2 \
#   --model.streaming-chunk-wise-weight=0.8 \
#   --overwrite

uv run --with debugpy python -m debugpy --listen 5678 --wait-for-client scripts/train.py pi05_franka_xhand_flower_streaming \
  --exp-name=pi05_streaming_franka_xhand_flower \
  --model.streaming \
  --model.streaming-chunk-size=1 \
  --model.streaming-constant-weight=0.2 \
  --model.streaming-chunk-wise-weight=0.8 \
  --overwrite
