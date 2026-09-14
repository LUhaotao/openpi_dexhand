#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$workspace_dir/.venv/bin/python" ]]; then
  python_bin="$workspace_dir/.venv/bin/python"
else
  python_bin="python3"
fi

# Set these endpoint values for the machines under test.
export VLM_HOST="${VLM_HOST:-127.0.0.1}"
export VLM_PORT="${VLM_PORT:-8001}"
export FM_HOST="${FM_HOST:-127.0.0.1}"
export FM_PORT="${FM_PORT:-8000}"

# The Python client reads configuration from this script's environment; no
# command-line arguments are required.
export FAKE_CLIENT_TARGETS="${FAKE_CLIENT_TARGETS:-vlm,fm,all}"
export FAKE_WARMUP="${FAKE_WARMUP:-3}"
export FAKE_RUNS="${FAKE_RUNS:-20}"
export FAKE_NUM_STEPS="${FAKE_NUM_STEPS:-10}"
export FAKE_NOISE_TOKENS="${FAKE_NOISE_TOKENS-1}"
export FAKE_FM_MODE="${FAKE_FM_MODE:-stream}"
export FAKE_STREAM_CHUNK_SIZE="${FAKE_STREAM_CHUNK_SIZE:-5}"
export FAKE_ENVIRONMENT="${FAKE_ENVIRONMENT:-franka_xhand_continuous_state}"
export FAKE_STATE_DIM="${FAKE_STATE_DIM:-18}"
export FAKE_IMAGE_HEIGHT="${FAKE_IMAGE_HEIGHT:-224}"
export FAKE_IMAGE_WIDTH="${FAKE_IMAGE_WIDTH:-224}"
export FAKE_PROMPT="${FAKE_PROMPT:-pick up the spray bottle and spray the sunflower}"

exec "$python_bin" "$workspace_dir/fake_client.py"
