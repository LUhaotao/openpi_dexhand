import asyncio
import dataclasses
import http
import time
import traceback
from typing import Any, Literal

import jax
import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from openpi.fake_env.config import FakeEnvConfig
from openpi.fake_env.observation import WireToModelTransform
from openpi.policies import multi_process_policy as _multi_process_policy
from openpi.policies import policy as _policy


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


class FakePolicy:
    """Single-process policy using deterministic synthetic weights."""

    def __init__(self, config: FakeEnvConfig):
        self.config = config
        model = config.create(jax.random.key(config.seed))
        self._policy = _policy.Policy(
            model,
            transforms=[WireToModelTransform(config)],
            output_transforms=(),
            warmup_observation=config.fake_obs(),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {"fake_weights": True, "model_id": self.model_id, "mode": "single"}

    @property
    def model_id(self) -> str:
        return f"{type(self._policy._model).__name__}:{self.config.action_dim}:{self.config.action_horizon}"  # noqa: SLF001

    def warmup(self) -> None:
        self._policy.warmup()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        prepare_start = time.perf_counter()
        observation = self._policy.prepare_observation(request["observation"], update_tactile_history=True)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000
        sample_start = time.perf_counter()
        actions = self._policy._sample_actions(  # noqa: SLF001
            jax.random.key(self.config.seed), observation, num_steps=self.config.num_steps
        )
        _block_until_ready(actions)
        model_ms = (time.perf_counter() - sample_start) * 1000
        return {
            "actions": np.asarray(actions[0]),
            "fake_timing": {"prepare_ms": prepare_ms, "model_ms": model_ms},
        }


def _instrument_role(role_policy: _multi_process_policy.MultiProcessPolicy) -> None:
    """Attach model-only timing around the role's jitted model calls."""
    role_policy._last_model_ms = 0.0  # noqa: SLF001
    for name in ("_encode_prefix_jit", "_sample_actions_from_prefix_jit", "_advance_streaming_actions_from_prefix_jit"):
        if not hasattr(role_policy, name):
            continue
        original = getattr(role_policy, name)

        def timed(*args, _original=original, **kwargs):
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            _block_until_ready(result)
            role_policy._last_model_ms = (time.perf_counter() - start) * 1000  # noqa: SLF001
            return result

        setattr(role_policy, name, timed)


def make_role_policy(
    config: FakeEnvConfig,
    role: Literal["single", "vlm", "fm"],
    *,
    vlm_host: str = "127.0.0.1",
    vlm_port: int | None = None,
) -> Any:
    if role == "single":
        return FakePolicy(config)
    model = config.create(jax.random.key(config.seed))
    base_policy = _policy.Policy(
        model,
        transforms=[WireToModelTransform(config)],
        output_transforms=(),
        warmup_observation=config.fake_obs(),
    )
    role_policy = _multi_process_policy.MultiProcessPolicy(
        base_policy,
        role,
        vlm_host=vlm_host,
        vlm_port=vlm_port,
    )
    _instrument_role(role_policy)
    return role_policy


@dataclasses.dataclass
class FakeWebsocketServer:
    policy: Any
    host: str
    port: int
    metadata: dict[str, Any]

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        try:
            while True:
                try:
                    request = msgpack_numpy.unpackb(await websocket.recv())
                    start = time.perf_counter()
                    response = await asyncio.to_thread(self.policy.infer, request)
                    server_ms = (time.perf_counter() - start) * 1000
                    timing = response.setdefault("fake_timing", {})
                    timing["server_ms"] = server_ms
                    if request.get("op") == "refresh_prefix":
                        timing["kv_refresh_ms"] = server_ms
                    if hasattr(self.policy, "_last_model_ms"):
                        timing["model_ms"] = float(self.policy._last_model_ms)  # noqa: SLF001
                    await websocket.send(packer.pack(response))
                except websockets.ConnectionClosed:
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Fake server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            reset = getattr(self.policy, "reset_tactile_history", None)
            if callable(reset):
                reset("fake-env")

    @staticmethod
    def _health_check(connection: _server.ServerConnection, request: _server.Request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None
