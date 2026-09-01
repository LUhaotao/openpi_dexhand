"""Blocking client for the optional OpenPI VLM/FM server pair."""

from __future__ import annotations

from typing import Any

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def build_fm_request(
    observation: dict[str, Any],
    cache_version: str,
    *,
    num_steps: int = 10,
    noise_tokens: int | None = None,
) -> dict[str, Any]:
    request = {
        "op": "infer",
        "observation": observation,
        "expected_cache_version": cache_version,
        "num_steps": num_steps,
    }
    if noise_tokens is not None:
        request["noise_tokens"] = noise_tokens
    return request


def build_stream_request(
    observation: dict[str, Any],
    cache_version: str,
    *,
    session_id: str,
    executed_action_id: int,
    num_steps: int = 10,
) -> dict[str, Any]:
    return {
        "op": "stream_infer",
        "observation": observation,
        "expected_cache_version": cache_version,
        "session_id": session_id,
        "executed_action_id": executed_action_id,
        "num_steps": num_steps,
    }


class MultiProcessClient:
    """Synchronously coordinates a VLM server and an FM server."""

    def __init__(self, host: str = "127.0.0.1", fm_port: int = 8000, vlm_port: int | None = None):
        self._fm = WebsocketClientPolicy(host, fm_port)
        self._vlm = WebsocketClientPolicy(host, vlm_port or fm_port + 1)

    def update_vlm(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Block until VLM prefix encoding finishes; only metadata is retained."""
        response = self._vlm.infer({"op": "encode_prefix", "observation": observation})
        self._cache_id = response["cache_id"]
        self._cache_version = response["cache_version"]
        refresh = self._fm.infer({
            "op": "refresh_prefix",
            "cache_id": self._cache_id,
            "cache_version": self._cache_version,
        })
        if refresh.get("cache_version") != self._cache_version:
            raise RuntimeError("FM activated an unexpected prefix cache version")
        return response

    def infer_fm(
        self,
        observation: dict[str, Any],
        *,
        num_steps: int = 10,
        noise_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Block until FM produces an action chunk using the latest prefix cache."""
        if not hasattr(self, "_cache_version"):
            raise RuntimeError("Call update_vlm() before infer_fm().")
        return self._fm.infer(
            build_fm_request(
                observation,
                self._cache_version,
                num_steps=num_steps,
                noise_tokens=noise_tokens,
            )
        )

    def reset_stream(self, session_id: str = "default") -> dict[str, Any]:
        return self._fm.infer({"op": "reset_stream", "session_id": session_id})

    def infer_stream(
        self,
        observation: dict[str, Any],
        *,
        session_id: str,
        executed_action_id: int,
        num_steps: int = 10,
    ) -> dict[str, Any]:
        if not hasattr(self, "_cache_version"):
            raise RuntimeError("Call update_vlm() before infer_stream().")
        return self._fm.infer(
            build_stream_request(
                observation,
                self._cache_version,
                session_id=session_id,
                executed_action_id=executed_action_id,
                num_steps=num_steps,
            )
        )
