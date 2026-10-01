"""Start the synthetic JAX server and print a fixed three-layer latency report."""

import dataclasses
import multiprocessing
import time
from typing import Literal

from openpi_client.websocket_client_policy import WebsocketClientPolicy
import tyro

from openpi.fake_env.config import FakeEnvConfig
from openpi.fake_env.observation import make_production_wire_observation
from openpi.fake_env.server import FakeWebsocketServer
from openpi.fake_env.server import make_role_policy
from openpi.fake_env.timing import format_latency_report


@dataclasses.dataclass(frozen=True)
class Args(FakeEnvConfig):
    """Configuration for the standalone fake environment benchmark."""

    mode: Literal["single", "split"] = "split"
    host: str = "127.0.0.1"
    port: int = 8000
    vlm_port: int = 8001
    serve_only: bool = False


def _serve_process(config: FakeEnvConfig, role: Literal["single", "vlm", "fm"], host: str, port: int, vlm_port: int):
    policy = make_role_policy(config, role, vlm_host=host, vlm_port=vlm_port)
    policy.warmup()
    metadata = {
        "fake_weights": True,
        "fake_role": role,
        "action_dim": config.action_dim,
        "action_horizon": config.action_horizon,
    }
    FakeWebsocketServer(policy, host, port, metadata).serve_forever()


def _start_process(config: FakeEnvConfig, role: Literal["single", "vlm", "fm"], args: Args):
    context = multiprocessing.get_context("spawn")
    port = args.port if role != "vlm" else args.vlm_port
    process = context.Process(
        target=_serve_process,
        args=(config, role, args.host, port, args.vlm_port),
        daemon=True,
    )
    process.start()
    return process


def _run_single(args: Args) -> None:
    process = _start_process(args, "single", args)
    client = None
    try:
        if args.serve_only:
            process.join()
            return
        client = WebsocketClientPolicy(args.host, args.port)
        observation = make_production_wire_observation(args)
        start = time.perf_counter()
        response = client.infer({"observation": observation})
        e2e_ms = (time.perf_counter() - start) * 1000
        fake_timing = response.get("fake_timing", {})
        server_ms = float(fake_timing.get("server_ms", 0.0))
        model_ms = float(fake_timing.get("model_ms", 0.0))
        print(format_latency_report({
            "inference": {"vlm_ms": 0.0, "fm_ms": model_ms},
            "server": {"vlm_ms": 0.0, "kv_refresh_ms": 0.0, "fm_ms": server_ms, "total_ms": server_ms},
            "e2e": {"total_ms": e2e_ms},
        }))
    finally:
        if client is not None:
            client._ws.close()  # noqa: SLF001
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


def _run_split(args: Args) -> None:
    vlm_process = _start_process(args, "vlm", args)
    fm_process = _start_process(args, "fm", args)
    vlm_client = None
    fm_client = None
    try:
        if args.serve_only:
            vlm_process.join()
            fm_process.join()
            return
        vlm_client = WebsocketClientPolicy(args.host, args.vlm_port)
        fm_client = WebsocketClientPolicy(args.host, args.port)
        observation = make_production_wire_observation(args)
        e2e_start = time.perf_counter()

        vlm_response = vlm_client.infer({"op": "encode_prefix", "observation": observation})
        vlm_timing = vlm_response.get("fake_timing", {})

        refresh_response = fm_client.infer({
            "op": "refresh_prefix",
            "cache_id": vlm_response["cache_id"],
            "cache_version": vlm_response["cache_version"],
            "wait_for_activation": True,
        })
        refresh_timing = refresh_response.get("fake_timing", {})

        fm_response = fm_client.infer({
            "op": "infer",
            "observation": observation,
            "expected_cache_version": vlm_response["cache_version"],
        })
        fm_timing = fm_response.get("fake_timing", {})
        e2e_ms = (time.perf_counter() - e2e_start) * 1000

        vlm_server_ms = float(vlm_timing.get("server_ms", 0.0))
        kv_refresh_ms = float(refresh_timing.get("kv_refresh_ms", refresh_timing.get("server_ms", 0.0)))
        fm_server_ms = float(fm_timing.get("server_ms", 0.0))
        print(format_latency_report({
            "inference": {
                "vlm_ms": float(vlm_timing.get("model_ms", 0.0)),
                "fm_ms": float(fm_timing.get("model_ms", 0.0)),
            },
            "server": {
                "vlm_ms": vlm_server_ms,
                "kv_refresh_ms": kv_refresh_ms,
                "fm_ms": fm_server_ms,
                "total_ms": vlm_server_ms + kv_refresh_ms + fm_server_ms,
            },
            "e2e": {"total_ms": e2e_ms},
        }))
    finally:
        for client in (vlm_client, fm_client):
            if client is not None:
                client._ws.close()  # noqa: SLF001
        for process in (vlm_process, fm_process):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def main(args: Args) -> None:
    if args.mode == "single":
        _run_single(args)
    else:
        _run_split(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
