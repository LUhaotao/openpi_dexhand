"""Blocking client for the optional OpenPI VLM/FM server pair."""

from __future__ import annotations

from typing import Any

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class MultiProcessClient:
    """Synchronously coordinates a VLM server and an FM server."""

    def __init__(self, host: str = "127.0.0.1", fm_port: int = 8000, vlm_port: int | None = None):
        self._fm = WebsocketClientPolicy(host, fm_port)
        self._vlm = WebsocketClientPolicy(host, vlm_port or fm_port + 1)
        self._prefix_cache: Any | None = None

    def update_vlm(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Block until VLM prefix encoding finishes and retain its cache."""
        response = self._vlm.infer({"observation": observation})
        self._prefix_cache = response["prefix_cache"]
        return response

    def infer_fm(self, observation: dict[str, Any], *, num_steps: int = 10) -> dict[str, Any]:
        """Block until FM produces an action chunk using the latest prefix cache."""
        if self._prefix_cache is None:
            raise RuntimeError("Call update_vlm() before infer_fm().")
        return self._fm.infer({
            "observation": observation,
            "prefix_cache": self._prefix_cache,
            "num_steps": num_steps,
        })
