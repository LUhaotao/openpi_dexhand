"""Start the managed Runtime endpoint and its private VLM/FM workers.

This is an additional deployment mode.  ``scripts/serve_policy.py`` remains the
entry point for the existing single-process and client-coordinated multi-process
servers.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import os
import socket
from typing import Literal

import tyro

from openpi.serving import runtime as _runtime
from openpi.serving import websocket_policy_server
from openpi.serving.backends import LocalBackend
from openpi.serving.backends import ProcessBackend
from openpi.serving.worker_health import wait_for_worker as _wait_for_worker

# Reuse the existing checkpoint/environment argument types and policy loader only
# inside worker processes.  Importing this module does not create a policy or touch
# a CUDA device in the Runtime parent.
try:
    from scripts import serve_policy as _serve_policy
except ModuleNotFoundError as exc:  # Running this file directly puts ``scripts`` on sys.path.
    if exc.name != "scripts":
        raise
    import serve_policy as _serve_policy

logger = logging.getLogger(__name__)


Checkpoint = _serve_policy.Checkpoint
Default = _serve_policy.Default
EnvMode = _serve_policy.EnvMode


@dataclasses.dataclass
class Args:
    """Arguments for the Runtime-managed deployment."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    mode: Literal["process", "local"] = "process"
    save_gate_logs: bool = False
    worker_start_timeout: float = 300.0
    # Runtime-owned VLM refresh schedule.  The public client only sends
    # observations; action-window refill and cache refresh are server policy.
    streaming_horizon: int = 50
    streaming_watermark: int = 40
    streaming_inference_mode: Literal["chunk", "action_head", "chunk_heads"] = "chunk_heads"
    num_steps: int = 10
    # Optional override; defaults to the selected model's gate history length.
    history_length: int | None = None
    # These are physical CUDA device indices on the host.  The worker process sees
    # its selected device as local device 0 after CUDA_VISIBLE_DEVICES is applied.
    vlm_cuda_device: int = 0
    fm_cuda_device: int = 1
    # Local mode uses the process's normal CUDA visibility.  Set this only when
    # JAX is not initialized by the parent environment yet.
    cuda_device: int | None = None
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def _find_free_port(excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port


def _worker_entrypoint(
    role: str,
    port: int,
    cuda_device: int,
    worker_args: _serve_policy.Args,
) -> None:
    """Load one role after pinning CUDA visibility in the child process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("Starting %s worker on CUDA device %s at port %s", role.upper(), cuda_device, port)
    _serve_policy._serve_multi_process_role(worker_args, role, port)  # noqa: SLF001


def _make_worker_args(
    args: Args,
    *,
    role: str,
    fm_port: int,
    vlm_port: int,
) -> _serve_policy.Args:
    return _serve_policy.Args(
        env=args.env,
        default_prompt=args.default_prompt,
        port=fm_port if role == "fm" else vlm_port,
        record=False,
        save_gate_logs=args.save_gate_logs,
        multi_process=True,
        vlm_port=vlm_port,
        vlm_host="127.0.0.1",
        fm_port=fm_port,
        fm_host="127.0.0.1",
        multi_process_role=role,
        streaming_inference_mode=args.streaming_inference_mode,  # type: ignore[arg-type]
        policy=args.policy,
    )


def _terminate_workers(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=10)


def _serve_local(args: Args) -> None:
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            raise ValueError("cuda_device must be non-negative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    policy_args = _serve_policy.Args(
        env=args.env,
        default_prompt=args.default_prompt,
        port=args.port,
        record=False,
        save_gate_logs=args.save_gate_logs,
        multi_process=False,
        policy=args.policy,
    )
    policy = _serve_policy.create_policy(policy_args)
    backend = LocalBackend(policy, streaming_inference_mode=args.streaming_inference_mode)
    coordinator = _runtime.RuntimeCoordinator(
        backend,
        streaming_horizon=args.streaming_horizon,
        streaming_watermark=args.streaming_watermark,
        num_steps=args.num_steps,
        history_length=args.history_length,
    )
    coordinator.warmup()
    metadata = {**coordinator.metadata, "runtime": "local"}
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=coordinator,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    try:
        server.serve_forever()
    finally:
        coordinator.close()


def main(args: Args) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.mode == "local":
        _serve_local(args)
        return
    if args.vlm_cuda_device < 0 or args.fm_cuda_device < 0:
        raise ValueError("CUDA device indices must be non-negative")
    if args.vlm_cuda_device == args.fm_cuda_device:
        raise ValueError("VLM and FM must use different CUDA devices")
    if args.worker_start_timeout <= 0:
        raise ValueError("worker_start_timeout must be positive")

    fm_port = _find_free_port({args.port})
    vlm_port = _find_free_port({args.port, fm_port})
    process_context = multiprocessing.get_context("spawn")
    fm_process = process_context.Process(
        target=_worker_entrypoint,
        args=(
            "fm",
            fm_port,
            args.fm_cuda_device,
            _make_worker_args(args, role="fm", fm_port=fm_port, vlm_port=vlm_port),
        ),
        daemon=True,
    )
    vlm_process = process_context.Process(
        target=_worker_entrypoint,
        args=(
            "vlm",
            vlm_port,
            args.vlm_cuda_device,
            _make_worker_args(args, role="vlm", fm_port=fm_port, vlm_port=vlm_port),
        ),
        daemon=True,
    )
    processes = [fm_process, vlm_process]
    coordinator = None
    try:
        for process in processes:
            process.start()
        _wait_for_worker(fm_process, fm_port, args.worker_start_timeout)
        _wait_for_worker(vlm_process, vlm_port, args.worker_start_timeout)
        backend = ProcessBackend(
            vlm_host="127.0.0.1",
            vlm_port=vlm_port,
            fm_host="127.0.0.1",
            fm_port=fm_port,
            metadata={
                "runtime": "managed_vlm_fm",
                "vlm_cuda_device": args.vlm_cuda_device,
                "fm_cuda_device": args.fm_cuda_device,
            },
        )
        coordinator = _runtime.RuntimeCoordinator(
            backend,
            streaming_horizon=args.streaming_horizon,
            streaming_watermark=args.streaming_watermark,
            num_steps=args.num_steps,
            history_length=args.history_length,
        )
        coordinator.warmup()
        metadata = coordinator.metadata
        logger.info(
            "Runtime listening on 0.0.0.0:%s (VLM CUDA %s, FM CUDA %s)",
            args.port,
            args.vlm_cuda_device,
            args.fm_cuda_device,
        )
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=coordinator,
            host="0.0.0.0",
            port=args.port,
            metadata=metadata,
        )
        server.serve_forever()
    finally:
        if coordinator is not None:
            coordinator.close()
        _terminate_workers(processes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
