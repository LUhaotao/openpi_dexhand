"""Measure JAX policy latency with deterministic synthetic observations.

The benchmark is configured through environment variables so that a small
shell wrapper can select the remote VLM and FM endpoints; ALL runs both in
sequence.
The VLM measurement includes prefix-cache materialization on the VLM server;
FM prefix-cache transfer is setup work for FM latency and is included in ALL.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
import time
from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

SUPPORTED_TARGETS = frozenset({"vlm", "fm", "all"})
SUPPORTED_FM_MODES = frozenset({"stream", "infer"})


@dataclass(frozen=True)
class BenchmarkConfig:
    vlm_host: str
    vlm_port: int
    fm_host: str
    fm_port: int
    targets: tuple[str, ...]
    warmup: int
    runs: int
    num_steps: int
    noise_tokens: int | None
    fm_mode: str
    stream_chunk_size: int
    environment: str
    state_dim: int
    image_height: int
    image_width: int
    prompt: str


def build_zero_observation(
    environment: str,
    *,
    state_dim: int,
    height: int,
    width: int,
    prompt: str,
) -> dict[str, Any]:
    """Build a zero-valued observation for a supported OpenPI input schema."""
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


def summarize_latencies(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Return robust summary statistics for wall-clock latency samples."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "min_ms": round(float(np.min(values)), 1),
        "mean_ms": round(float(np.mean(values)), 1),
        "p50_ms": round(float(np.percentile(values, 50)), 1),
        "p95_ms": round(float(np.percentile(values, 95)), 1),
        "p99_ms": round(float(np.percentile(values, 99)), 1),
        "max_ms": round(float(np.max(values)), 1),
    }


def build_fm_request(
    observation: dict[str, Any],
    *,
    num_steps: int,
    noise_tokens: int | None,
) -> dict[str, Any]:
    """Build an FM request without timing prefix-cache transfer."""
    request: dict[str, Any] = {
        "op": "infer",
        "observation": observation,
        "num_steps": num_steps,
    }
    if noise_tokens is not None:
        request["noise_tokens"] = noise_tokens
    return request


def build_stream_request(
    observation: dict[str, Any],
    *,
    session_id: str,
    executed_action_id: int,
    num_steps: int,
) -> dict[str, Any]:
    return {
        "op": "stream_infer",
        "observation": observation,
        "session_id": session_id,
        "executed_action_id": executed_action_id,
        "num_steps": num_steps,
    }


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return None if not value else int(value)


def _parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(targets) - SUPPORTED_TARGETS)
    if not targets or unknown:
        raise ValueError(f"Unknown latency targets: {unknown or 'empty target list'}")
    return targets


def load_config() -> BenchmarkConfig:
    noise_tokens = _env_optional_int("FAKE_NOISE_TOKENS", 1)
    fm_mode = os.environ.get("FAKE_FM_MODE", "stream").strip().lower()
    config = BenchmarkConfig(
        vlm_host=os.environ.get("VLM_HOST", "127.0.0.1"),
        vlm_port=_env_int("VLM_PORT", 8001),
        fm_host=os.environ.get("FM_HOST", "127.0.0.1"),
        fm_port=_env_int("FM_PORT", 8000),
        targets=_parse_targets(os.environ.get("FAKE_CLIENT_TARGETS", "vlm,fm,all")),
        warmup=_env_int("FAKE_WARMUP", 3),
        runs=_env_int("FAKE_RUNS", 20),
        num_steps=_env_int("FAKE_NUM_STEPS", 10),
        noise_tokens=noise_tokens,
        fm_mode=fm_mode,
        stream_chunk_size=_env_int("FAKE_STREAM_CHUNK_SIZE", 5),
        environment=os.environ.get("FAKE_ENVIRONMENT", "franka_xhand_continuous_state"),
        state_dim=_env_int("FAKE_STATE_DIM", 18),
        image_height=_env_int("FAKE_IMAGE_HEIGHT", 224),
        image_width=_env_int("FAKE_IMAGE_WIDTH", 224),
        prompt=os.environ.get("FAKE_PROMPT", "pick up the spray bottle and spray the sunflower"),
    )
    if config.warmup < 0:
        raise ValueError("FAKE_WARMUP must be non-negative")
    if config.runs <= 0:
        raise ValueError("FAKE_RUNS must be positive")
    if config.num_steps <= 0:
        raise ValueError("FAKE_NUM_STEPS must be positive")
    if config.noise_tokens is not None and config.noise_tokens <= 0:
        raise ValueError("FAKE_NOISE_TOKENS must be positive when set")
    if config.fm_mode not in SUPPORTED_FM_MODES:
        raise ValueError(f"FAKE_FM_MODE must be one of {sorted(SUPPORTED_FM_MODES)}")
    if config.stream_chunk_size <= 0:
        raise ValueError("FAKE_STREAM_CHUNK_SIZE must be positive")
    if config.state_dim <= 0 or config.image_height <= 0 or config.image_width <= 0:
        raise ValueError("FAKE_STATE_DIM and image dimensions must be positive")
    return config


def _server_infer_ms(response: dict[str, Any]) -> float | None:
    timing = response.get("server_timing")
    if not isinstance(timing, dict) or "infer_ms" not in timing:
        return None
    return float(timing["infer_ms"])


def _benchmark_components_ms(response: dict[str, Any]) -> dict[str, float]:
    components = response.get("_benchmark_components_ms", {})
    if not isinstance(components, dict):
        return {}
    return {str(name): float(value) for name, value in components.items()}


def _measure(
    name: str,
    request_fn: Callable[[], dict[str, Any]],
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        request_fn()

    wall_latencies: list[float] = []
    server_latencies: list[float] = []
    component_latencies: dict[str, list[float]] = {}
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        response = request_fn()
        wall_latencies.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
        server_ms = _server_infer_ms(response)
        if server_ms is not None:
            server_latencies.append(server_ms)
        for component_name, component_ms in _benchmark_components_ms(response).items():
            component_latencies.setdefault(component_name, []).append(component_ms)

    result: dict[str, Any] = {
        "samples": runs,
        "wall_ms": summarize_latencies(wall_latencies),
    }
    if server_latencies:
        result["server_infer_ms"] = summarize_latencies(server_latencies)
    if component_latencies:
        result["server_components_ms"] = {
            name: summarize_latencies(samples) for name, samples in component_latencies.items()
        }

    wall = result["wall_ms"]
    server = result.get("server_infer_ms", {})
    server_suffix = f" server_p50={server['p50_ms']:.1f} ms" if server else ""
    print(
        f"{name:4s} wall_p50={wall['p50_ms']:8.1f} ms "
        f"wall_p95={wall['p95_ms']:8.1f} ms{server_suffix}"
    )
    return result


def _close_client(client: WebsocketClientPolicy) -> None:
    websocket = getattr(client, "_ws", None)
    if websocket is not None:
        websocket.close()


@dataclass
class _StreamingRequestRunner:
    client: WebsocketClientPolicy
    observation: dict[str, Any]
    session_id: str
    chunk_size: int
    num_steps: int
    executed_action_id: int = 0

    def seed(self) -> dict[str, Any]:
        """Initialize stream state outside the measured steady-state forward."""
        return self.client.infer(
            build_stream_request(
                self.observation,
                session_id=self.session_id,
                executed_action_id=0,
                num_steps=self.num_steps,
            )
        )

    def infer(self) -> dict[str, Any]:
        self.executed_action_id += self.chunk_size
        return self.client.infer(
            build_stream_request(
                self.observation,
                session_id=self.session_id,
                executed_action_id=self.executed_action_id,
                num_steps=self.num_steps,
            )
        )


def _run_vlm(config: BenchmarkConfig, observation: dict[str, Any]) -> dict[str, Any]:
    client = WebsocketClientPolicy(config.vlm_host, config.vlm_port)
    request = {"op": "encode_prefix", "observation": observation}
    try:
        # The server materializes and retains the KV cache before returning this
        # response, so this includes upload, prefix inference, and cache prep.
        return _measure("VLM", lambda: client.infer(request), warmup=config.warmup, runs=config.runs)
    finally:
        _close_client(client)


def _prepare_fm_cache(
    config: BenchmarkConfig,
    observation: dict[str, Any],
    vlm_client: WebsocketClientPolicy,
    fm_client: WebsocketClientPolicy,
) -> dict[str, Any]:
    prefix = vlm_client.infer({"op": "encode_prefix", "observation": observation})
    return fm_client.infer({
        "op": "refresh_prefix",
        "cache_id": prefix["cache_id"],
        "cache_version": prefix["cache_version"],
        "wait_for_activation": True,
    })


def _prepare_stream_runner(
    config: BenchmarkConfig,
    observation: dict[str, Any],
    fm_client: WebsocketClientPolicy,
) -> _StreamingRequestRunner | None:
    if config.fm_mode != "stream":
        return None
    fm_client.infer({"op": "reset_stream", "session_id": "fake-client"})
    runner = _StreamingRequestRunner(
        client=fm_client,
        observation=observation,
        session_id="fake-client",
        chunk_size=config.stream_chunk_size,
        num_steps=config.num_steps,
    )
    runner.seed()
    return runner


def _run_fm(config: BenchmarkConfig, observation: dict[str, Any]) -> dict[str, Any]:
    vlm_client = WebsocketClientPolicy(config.vlm_host, config.vlm_port)
    fm_client = WebsocketClientPolicy(config.fm_host, config.fm_port)
    try:
        refresh = _prepare_fm_cache(config, observation, vlm_client, fm_client)
        print(f"FM cache setup status={refresh.get('status', 'unknown')}")
        stream_runner = _prepare_stream_runner(config, observation, fm_client)
        if stream_runner is None:
            request = build_fm_request(
                observation,
                num_steps=config.num_steps,
                noise_tokens=config.noise_tokens,
            )

            def request_fn() -> dict[str, Any]:
                return fm_client.infer(request)

        else:
            request_fn = stream_runner.infer
        # Cache setup and stream seeding are completed outside this measured
        # steady-state FM forward.
        return _measure("FM", request_fn, warmup=config.warmup, runs=config.runs)
    finally:
        _close_client(vlm_client)
        _close_client(fm_client)


def _run_all(config: BenchmarkConfig, observation: dict[str, Any]) -> dict[str, Any]:
    vlm_client = WebsocketClientPolicy(config.vlm_host, config.vlm_port)
    fm_client = WebsocketClientPolicy(config.fm_host, config.fm_port)
    try:
        refresh = _prepare_fm_cache(config, observation, vlm_client, fm_client)
        print(f"ALL initial cache setup status={refresh.get('status', 'unknown')}")
        stream_runner = _prepare_stream_runner(config, observation, fm_client)
        request = build_fm_request(
            observation,
            num_steps=config.num_steps,
            noise_tokens=config.noise_tokens,
        )

        def request_fn() -> dict[str, Any]:
            start = vlm_client.infer({"op": "encode_prefix", "observation": observation})
            refreshed = fm_client.infer({
                "op": "refresh_prefix",
                "cache_id": start["cache_id"],
                "cache_version": start["cache_version"],
                "wait_for_activation": True,
            })
            fm_response = stream_runner.infer() if stream_runner is not None else fm_client.infer(request)
            components = {
                name: timing
                for name, response in (
                    ("vlm", start),
                    ("fm_refresh", refreshed),
                    ("fm", fm_response),
                )
                if (timing := _server_infer_ms(response)) is not None
            }
            combined = dict(fm_response)
            combined["_benchmark_components_ms"] = components
            if components:
                combined["server_timing"] = {"infer_ms": sum(components.values())}
            return combined

        return _measure("ALL", request_fn, warmup=config.warmup, runs=config.runs)
    finally:
        _close_client(vlm_client)
        _close_client(fm_client)


def main() -> None:
    config = load_config()
    observation = build_zero_observation(
        config.environment,
        state_dim=config.state_dim,
        height=config.image_height,
        width=config.image_width,
        prompt=config.prompt,
    )
    print(
        f"targets={','.join(config.targets)} warmup={config.warmup} runs={config.runs} "
        f"num_steps={config.num_steps} noise_tokens={config.noise_tokens} "
        f"fm_mode={config.fm_mode} stream_chunk_size={config.stream_chunk_size} "
        f"environment={config.environment}"
    )

    results: dict[str, Any] = {}
    if "vlm" in config.targets:
        results["vlm"] = _run_vlm(config, observation)
    if "fm" in config.targets:
        results["fm"] = _run_fm(config, observation)
    if "all" in config.targets:
        results["all"] = _run_all(config, observation)

    report = {
        "config": asdict(config),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
