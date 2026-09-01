#!/bin/bash
# 将 UniVTAC grasp_classify HDF5 数据转换为 LeRobot 格式

# 使用方法:
#   首次运行: bash convert_hdf5_to_lerobot/run_convert.sh
#   断点续传: bash convert_hdf5_to_lerobot/run_convert.sh --resume

.venv/bin/python convert_hdf5_to_lerobot/convert_univtac_to_lerobot.py \
  --source-dir /public/node01/users/lvrui/datasets/hdf5/univtac/grasp_classify/clean \
  --output-dir /public/node01/users/lvrui/datasets/lerobot/univtac/grasp_classify \
  --fps 60 \
  "$@"
