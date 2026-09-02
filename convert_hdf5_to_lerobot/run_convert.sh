#!/bin/bash
# 将 UniVTAC HDF5 数据转换为 LeRobot 格式

# 使用方法:
#   首次运行: bash convert_hdf5_to_lerobot/run_convert.sh
#   断点续传: bash convert_hdf5_to_lerobot/run_convert.sh --resume
#   不含触觉: bash convert_hdf5_to_lerobot/run_convert.sh --no-tactile
#   图片存储: bash convert_hdf5_to_lerobot/run_convert.sh --mode image

.venv/bin/python convert_hdf5_to_lerobot/convert_univtac_to_lerobot.py \
  --source-dir /public/node01/users/lvrui/datasets/hdf5/univtac/put_bottle_in_shelf/clean \
  --output-dir /public/node01/users/lvrui/datasets/lerobot/univtac/put_bottle_in_shelf \
  --fps 30 \
  --resume \
  --no-tactile \
  --mode video \
  "$@"
