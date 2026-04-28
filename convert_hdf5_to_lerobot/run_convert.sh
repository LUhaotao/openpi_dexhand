#!/bin/bash
# 将 franka + xhand 遥操 HDF5 数据转换为 LeRobot 格式

# 使用方法:
#   首次运行: bash convert_hdf5_to_lerobot/run_convert.sh
#   断点续传: bash convert_hdf5_to_lerobot/run_convert.sh --resume

uv run python convert_hdf5_to_lerobot/convert_subtask_to_lerobot.py \
  --repo-id flower_franka_xhand \
  --source-dir /home/frankx/docker_share/workspace/data/flower_4_28/raw \
  --lerobot-home /home/frankx/docker_share/workspace/data/flower_4_28/lerobot \
  --subfolders . \
  --robot-type franka_xhand \
  --default-task flower \
  --fps 30 \
  --mode video \
  --image-writer-processes 32 \
  --image-writer-threads 10 \
  "$@"
