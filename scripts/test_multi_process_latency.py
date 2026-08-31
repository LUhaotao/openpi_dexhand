"""Measure blocking VLM/FM server latency with synthetic zero observations."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

try:
    from scripts.multi_process_client import MultiProcessClient
except ModuleNotFoundError:  # Direct execution: ``python scripts/test_multi_process_latency.py``.
    from multi_process_client import MultiProcessClient


def build_zero_observation(
    environment: str,
    *,
    state_dim: int = 18,
    height: int = 224,
    width: int = 224,
    prompt: str = "pick up the object",
) -> dict[str, Any]:
    """Build a zero-valued observation for a supported OpenPI input transform."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if environment in {"franka_xhand", "franka_xhand_continuous_state"}:
        return {
            "images": {"cam_side": image.copy(), "cam_wrist": image.copy()},
            "state": np.zeros(state_dim, dtype=np.float32),
            "prompt": prompt,
        }
    if environment == "droid":
        return {
            "observation/exterior_image_1_left": image.copy(),
            "observation/wrist_image_left": image.copy(),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": prompt,
        }
    raise ValueError(f"Unsupported environment: {environment}")


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Return stable summary statistics for measured wall-clock latencies."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(values)), 1),
        "p50_ms": round(float(np.percentile(values, 50)), 1),
        "p95_ms": round(float(np.percentile(values, 95)), 1),
    }


def _measure(name: str, fn, *, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
    result = summarize_latencies(latencies)
    print(f"{name:12s} mean={result['mean_ms']:8.1f} ms  p50={result['p50_ms']:8.1f} ms  p95={result['p95_ms']:8.1f} ms")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--fm-port", type=int, default=8000)
    parser.add_argument("--vlm-port", type=int, default=8001)
    parser.add_argument(
        "--environment",
        choices=("franka_xhand_continuous_state", "franka_xhand", "droid"),
        default="franka_xhand_continuous_state",
    )
    parser.add_argument("--mode", choices=("vlm", "fm", "both"), default="both")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--noise-tokens", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=18)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--prompt", default="pick up the spray bottle and spray the sunflower")
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.num_steps <= 0:
        parser.error("--runs and --num-steps must be positive; --warmup must be non-negative")

    observation = build_zero_observation(
        args.environment,
        state_dim=args.state_dim,
        height=args.height,
        width=args.width,
        prompt=args.prompt,
    )
    print(f"environment={args.environment} mode={args.mode} runs={args.runs} warmup={args.warmup}")

    if args.mode in ("vlm", "both"):
        vlm = WebsocketClientPolicy(args.host, args.vlm_port)
        _measure(
            "VLM",
            lambda: vlm.infer({"op": "encode_prefix", "observation": observation}),
            warmup=args.warmup,
            runs=args.runs,
        )

    if args.mode in ("fm", "both"):
        client = MultiProcessClient(args.host, args.fm_port, args.vlm_port)
        client.update_vlm(observation)
        _measure(
            "FM",
            lambda: client.infer_fm(
                observation,
                num_steps=args.num_steps,
                noise_tokens=args.noise_tokens,
            ),
            warmup=args.warmup,
            runs=args.runs,
        )

    if args.mode == "both":
        client = MultiProcessClient(args.host, args.fm_port, args.vlm_port)
        _measure(
            "VLM+FM",
            lambda: (
                client.update_vlm(observation),
                client.infer_fm(
                    observation,
                    num_steps=args.num_steps,
                    noise_tokens=args.noise_tokens,
                ),
            ),
            warmup=args.warmup,
            runs=args.runs,
        )


if __name__ == "__main__":
    main()
