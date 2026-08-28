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
        # msgpack-numpy rejects NumPy's bfloat16 (dtype kind ``V``). Keep the
        # wire format portable; the FM process casts back to its model dtype.
        if cache.dtype.name == "bfloat16":
            return np.ascontiguousarray(cache, dtype=np.float32)
        return np.ascontiguousarray(cache)
    if hasattr(cache, "shape") and hasattr(cache, "dtype"):
        return np.asarray(cache)
    return cache


def decode_prefix_cache(cache: Any) -> Any:
    """Convert a received NumPy pytree to device arrays at the call site."""
    if isinstance(cache, dict):
        return {key: decode_prefix_cache(value) for key, value in cache.items()}
    if isinstance(cache, tuple):
        return tuple(decode_prefix_cache(value) for value in cache)
    if isinstance(cache, list):
        return [decode_prefix_cache(value) for value in cache]
    if isinstance(cache, np.ndarray):
        return cache
    return cache
