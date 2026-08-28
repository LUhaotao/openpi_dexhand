#!/bin/bash
# 将 franka + xhand 遥操 HDF5 数据转换为 LeRobot 格式

# 使用方法:bash convert_hdf5_to_lerobot/run_convert.sh
#   首次运行: 
#   断点续传: bash convert_hdf5_to_lerobot/run_convert.sh --resume

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x /root/miniconda3/bin/uv ]; then
  UV_BIN=/root/miniconda3/bin/uv
fi
if [ -z "$UV_BIN" ]; then
  echo "uv not found. Install uv or add it to PATH." >&2
  exit 1
fi

"$UV_BIN" run python convert_hdf5_to_lerobot/convert_subtask_to_lerobot.py \
  --repo-id ego_whiteboard_20 \
  --source-dir /root/data/xhand_franka_hdf5/whiteboard \
  --lerobot-home /root/data/xhand_franka_lerobot/whiteboard \
  --subfolders . \
  --robot-type franka_xhand \
  --default-task whiteboard \
  --fps 30 \
  --mode video \
  --image-writer-processes 32 \
  --image-writer-threads 10 \
  "$@"
