import asyncio
import http
import logging
import time
import traceback
import uuid

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._retain_session_on_disconnect = bool(
            getattr(policy, "retain_session_on_disconnect", False)
        )
        # The Runtime deployment has one public client stream. Keep this
        # server-owned session across WebSocket reconnects without requiring a
        # client-persisted session token.
        self._resumable_session_id = uuid.uuid4().hex
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _session_id_for_connection(self) -> str:
        if self._retain_session_on_disconnect:
            return self._resumable_session_id
        return uuid.uuid4().hex

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

        session_id = self._session_id_for_connection()
        prev_total_time = None
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    infer_time = time.monotonic()
                    # Keep synchronous model work off the asyncio event loop so a
                    # background cache refresh cannot stall websocket scheduling.
                    infer_with_session = getattr(self._policy, "infer_with_session", None)
                    if callable(infer_with_session) and getattr(self._policy, "tactile_history_enabled", False):
                        action = await asyncio.to_thread(infer_with_session, obs, session_id=session_id)
                    else:
                        action = await asyncio.to_thread(self._policy.infer, obs)
                    infer_time = time.monotonic() - infer_time

                    if not getattr(self._policy, "observation_action_protocol", False):
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
        finally:
            reset_session = getattr(self._policy, "reset_tactile_history", None)
            if (
                not self._retain_session_on_disconnect
                and callable(reset_session)
                and getattr(self._policy, "tactile_history_enabled", False)
            ):
                reset_session(session_id)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
