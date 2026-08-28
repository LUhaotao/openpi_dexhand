import asyncio
import http
import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


@dataclass
class ServerRTCState:
    """Raw action state retained by one client websocket connection."""

    raw_actions: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    chunk_base_action_id: int | None = None

    def build_model_context(self, executed_action_id: int) -> dict:
        if self.raw_actions is None or self.reference_state is None or self.chunk_base_action_id is None:
            return {"prev_action_chunk": None, "inference_delay": 0}

        consumed = max(0, executed_action_id - self.chunk_base_action_id)
        return {
            "prev_action_chunk": self.raw_actions[consumed:].copy(),
            "prev_action_reference_state": self.reference_state.copy(),
            "inference_delay": consumed,
        }

    def record_chunk(
        self,
        raw_actions: np.ndarray,
        *,
        reference_state: np.ndarray,
        base_action_id: int,
    ) -> None:
        self.raw_actions = np.asarray(raw_actions).copy()
        self.reference_state = np.asarray(reference_state).copy()
        self.chunk_base_action_id = base_action_id


class ExpWebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        enable_rtc: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._enable_rtc = enable_rtc
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        rtc_state = ServerRTCState()
        while True:
            try:
                start_time = time.monotonic()
                payload = msgpack_numpy.unpackb(await websocket.recv())
                obs, client_rtc_context = split_rtc_payload(payload)

                infer_time = time.monotonic()
                action = self._infer(obs, client_rtc_context, rtc_state)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _infer(
        self,
        observation: dict,
        client_rtc_context: dict | None,
        rtc_state: ServerRTCState,
    ) -> dict:
        """Run a normal or RTC-aware inference request based on server configuration."""
        if not self._enable_rtc or client_rtc_context is None:
            return self._policy.infer(observation)

        executed_action_id = _parse_executed_action_id(client_rtc_context)
        action = self._policy.infer(
            observation,
            rtc_context=rtc_state.build_model_context(executed_action_id),
        )
        raw_actions = action.pop("raw_model_actions", None)
        if raw_actions is None:
            raise ValueError("Inference RTC policy response is missing raw_model_actions")
        rtc_state.record_chunk(
            raw_actions,
            reference_state=np.asarray(observation["state"]),
            base_action_id=executed_action_id,
        )
        return action


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def split_rtc_payload(payload: dict) -> tuple[dict, dict | None]:
    """Extract client RTC metadata without changing the legacy payload."""
    if "rtc_context" not in payload:
        return payload, None
    observation = dict(payload)
    rtc_context = observation.pop("rtc_context")
    return observation, rtc_context


def _parse_executed_action_id(rtc_context: object) -> int:
    if not isinstance(rtc_context, dict):
        raise ValueError("RTC context must be a mapping")
    executed_action_id = rtc_context.get("executed_action_id")
    if not isinstance(executed_action_id, int) or isinstance(executed_action_id, bool):
        raise ValueError("RTC context must contain an integer executed_action_id")
    if executed_action_id < 0:
        raise ValueError("RTC executed_action_id must be non-negative")
    return executed_action_id
