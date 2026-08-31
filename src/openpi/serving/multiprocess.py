"""Small helpers shared by the multi-process inference endpoints."""

from typing import Any

import numpy as np


def encode_prefix_cache(cache: Any) -> Any:
    """Convert a JAX prefix cache pytree to a msgpack-compatible NumPy pytree."""
    try:
        import jax
    except ImportError:
        jax = None

    if jax is not None:
        cache = jax.device_get(cache)

    if isinstance(cache, dict):
        return {key: encode_prefix_cache(value) for key, value in cache.items()}
    if isinstance(cache, tuple):
        return tuple(encode_prefix_cache(value) for value in cache)
    if isinstance(cache, list):
        return [encode_prefix_cache(value) for value in cache]
    if isinstance(cache, np.ndarray):
        return np.ascontiguousarray(cache)
    if hasattr(cache, "shape") and hasattr(cache, "dtype"):
        return np.asarray(cache)
    return cache


def _manifest(cache: Any, path: str = "") -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    if isinstance(cache, dict):
        for key, value in cache.items():
            leaves.extend(_manifest(value, f"{path}/{key}"))
    elif isinstance(cache, tuple | list):
        for index, value in enumerate(cache):
            leaves.extend(_manifest(value, f"{path}/{index}"))
    elif isinstance(cache, np.ndarray):
        leaves.append({
            "path": path,
            "dtype": cache.dtype.name,
            "shape": list(cache.shape),
            "nbytes": int(cache.nbytes),
        })
    return leaves


def make_cache_envelope(
    cache: Any,
    *,
    model_id: str,
    cache_id: int,
    cache_version: str,
) -> dict[str, Any]:
    """Build a portable cache payload with metadata for FM validation."""
    cache = encode_prefix_cache(cache)
    manifest = _manifest(cache)
    payload_size = sum(item["nbytes"] for item in manifest)
    return {
        "protocol_version": 1,
        "model_id": model_id,
        "cache_id": int(cache_id),
        "cache_version": cache_version,
        "manifest": manifest,
        "payload_size": int(payload_size),
        "prefix_cache": cache,
    }


def validate_cache_envelope(
    envelope: dict[str, Any], *, model_id: str, max_payload_size: int = 512 * 1024 * 1024
) -> None:
    """Validate protocol, model and leaf metadata before activating a cache."""
    if envelope.get("protocol_version") != 1:
        raise ValueError("Unsupported prefix-cache protocol version")
    if envelope.get("model_id") != model_id:
        raise ValueError(
            f"Prefix-cache model mismatch: expected {model_id!r}, got {envelope.get('model_id')!r}"
        )
    if not isinstance(envelope.get("cache_version"), str):
        raise ValueError("Prefix-cache cache_version must be a string")
    payload = envelope.get("prefix_cache")
    manifest = envelope.get("manifest")
    if not isinstance(manifest, list) or payload is None:
        raise ValueError("Malformed prefix-cache envelope")
    actual_manifest = _manifest(payload)
    if actual_manifest != manifest:
        raise ValueError("Prefix-cache manifest mismatch")
    payload_size = int(envelope.get("payload_size", -1))
    actual_size = sum(item["nbytes"] for item in actual_manifest)
    if payload_size != actual_size or payload_size > max_payload_size:
        raise ValueError("Invalid prefix-cache payload size")


def decode_prefix_cache(cache: Any) -> Any:
    """Convert a received NumPy pytree to device arrays at the call site."""
    if isinstance(cache, dict):
        decoded = {key: decode_prefix_cache(value) for key, value in cache.items()}
        # msgpack has no tuple type and decodes the KV pair as a list. Gemma's
        # jaxtyping contract requires the outer KV container to remain a tuple.
        if "kv_cache" in decoded and isinstance(decoded["kv_cache"], list):
            decoded["kv_cache"] = tuple(decoded["kv_cache"])
        return decoded
    if isinstance(cache, tuple):
        return tuple(decode_prefix_cache(value) for value in cache)
    if isinstance(cache, list):
        return [decode_prefix_cache(value) for value in cache]
    if isinstance(cache, np.ndarray):
        return cache
    return cache
