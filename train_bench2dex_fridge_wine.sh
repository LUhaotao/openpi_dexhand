#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

CONFIG="pi05_bench2dex_fridge_wine_active38"
EXP_NAME="${EXP_NAME:-bench2dex_fridge_wine_active38}"

NORM_STATS="/public/node01/users/lvrui/datasets/lerobot/bench2dex/34_fridge_wine_interhand_pour_active38/norm_stats.json"
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "Missing Bench2Dex norm stats. Generate them first:"
  echo "  ${NORM_STATS}"
  exit 1
fi

exec uv run scripts/train.py "${CONFIG}" \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  "$@"
