"""Small helpers shared by the multi-process inference endpoints."""

from multiprocessing import shared_memory
from typing import Any
import uuid

import numpy as np

_SHARED_MEMORY_PROTOCOL_VERSION = 1


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
    observation_request_id: str | None = None,
    observation_chunk_id: int | None = None,
) -> dict[str, Any]:
    """Build a portable cache payload with metadata for FM validation."""
    cache = encode_prefix_cache(cache)
    manifest = _manifest(cache)
    payload_size = sum(item["nbytes"] for item in manifest)
    envelope = {
        "protocol_version": 1,
        "model_id": model_id,
        "cache_id": int(cache_id),
        "cache_version": cache_version,
        "manifest": manifest,
        "payload_size": int(payload_size),
        "prefix_cache": cache,
    }
    if observation_request_id is not None:
        envelope["observation_request_id"] = observation_request_id
    if observation_chunk_id is not None:
        envelope["observation_chunk_id"] = int(observation_chunk_id)
    return envelope


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


def _shared_memory_tree(cache: Any, buffer: memoryview, offset: int = 0) -> tuple[Any, int]:
    """Copy an array pytree into a shared-memory buffer and return its descriptor."""
    if isinstance(cache, dict):
        items = []
        for key, value in cache.items():
            child, offset = _shared_memory_tree(value, buffer, offset)
            items.append([key, child])
        return {"kind": "dict", "items": items}, offset
    if isinstance(cache, tuple):
        children = []
        for value in cache:
            child, offset = _shared_memory_tree(value, buffer, offset)
            children.append(child)
        return {"kind": "tuple", "items": children}, offset
    if isinstance(cache, list):
        children = []
        for value in cache:
            child, offset = _shared_memory_tree(value, buffer, offset)
            children.append(child)
        return {"kind": "list", "items": children}, offset
    if isinstance(cache, np.ndarray):
        array = np.ascontiguousarray(cache)
        end = offset + int(array.nbytes)
        buffer[offset:end] = array.view(np.uint8).reshape(-1)
        return {
            "kind": "array",
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": int(array.nbytes),
        }, end
    return {"kind": "scalar", "value": cache}, offset


def _shared_memory_nbytes(cache: Any) -> int:
    if isinstance(cache, dict):
        return sum(_shared_memory_nbytes(value) for value in cache.values())
    if isinstance(cache, tuple | list):
        return sum(_shared_memory_nbytes(value) for value in cache)
    if isinstance(cache, np.ndarray):
        return int(np.ascontiguousarray(cache).nbytes)
    return 0


def make_shared_memory_cache(
    cache: Any,
    *,
    model_id: str,
    cache_id: int,
    cache_version: str,
    observation_request_id: str | None = None,
    observation_chunk_id: int | None = None,
) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    """Materialize a cache in POSIX shared memory and return a small descriptor.

    The returned shared-memory object is owned by the caller and must remain alive
    until the consumer has opened and copied the arrays to its device.  The caller
    is responsible for ``close()`` and ``unlink()`` after the consumer acknowledges
    activation.
    """
    cache = encode_prefix_cache(cache)
    payload_size = _shared_memory_nbytes(cache)
    # Keep a non-zero segment for scalar-only pytrees so the descriptor remains
    # valid and the lifecycle is uniform.
    shm = shared_memory.SharedMemory(create=True, size=max(1, payload_size), name=f"openpi-{uuid.uuid4().hex}")
    tree, written = _shared_memory_tree(cache, shm.buf)
    if written != payload_size:
        shm.close()
        shm.unlink()
        raise RuntimeError(f"Shared-memory cache size mismatch: wrote {written}, expected {payload_size}")
    descriptor = {
        "protocol_version": _SHARED_MEMORY_PROTOCOL_VERSION,
        "transport": "shared_memory",
        "artifact_id": shm.name,
        "shared_memory_name": shm.name,
        "payload_size": payload_size,
        "model_id": model_id,
        "cache_id": int(cache_id),
        "cache_version": cache_version,
        "tree": tree,
    }
    if observation_request_id is not None:
        descriptor["observation_request_id"] = observation_request_id
    if observation_chunk_id is not None:
        descriptor["observation_chunk_id"] = int(observation_chunk_id)
    return descriptor, shm


def validate_shared_memory_cache(
    descriptor: dict[str, Any], *, model_id: str, max_payload_size: int = 512 * 1024 * 1024
) -> None:
    """Validate a shared-memory descriptor before opening its segment."""
    if descriptor.get("protocol_version") != _SHARED_MEMORY_PROTOCOL_VERSION:
        raise ValueError("Unsupported shared-memory cache protocol version")
    if descriptor.get("transport") != "shared_memory":
        raise ValueError("Unsupported cache transport")
    if descriptor.get("model_id") != model_id:
        raise ValueError(
            f"Shared-memory cache model mismatch: expected {model_id!r}, got {descriptor.get('model_id')!r}"
        )
    name = descriptor.get("shared_memory_name")
    if not isinstance(name, str) or not name:
        raise ValueError("Shared-memory cache is missing its name")
    payload_size = descriptor.get("payload_size")
    if not isinstance(payload_size, int) or payload_size < 0 or payload_size > max_payload_size:
        raise ValueError("Invalid shared-memory cache payload size")
    if descriptor.get("artifact_id") != name or not isinstance(descriptor.get("tree"), dict):
        raise ValueError("Malformed shared-memory cache descriptor")


def _shared_memory_array(descriptor: dict[str, Any], buffer: memoryview) -> np.ndarray:
    dtype = np.dtype(descriptor["dtype"])
    shape = tuple(int(value) for value in descriptor["shape"])
    offset = int(descriptor["offset"])
    nbytes = int(descriptor["nbytes"])
    if offset < 0 or nbytes < 0 or offset + nbytes > len(buffer):
        raise ValueError("Shared-memory array is outside the artifact buffer")
    if dtype.itemsize * int(np.prod(shape, dtype=np.int64)) != nbytes:
        raise ValueError("Shared-memory array shape, dtype and size do not match")
    return np.ndarray(shape=shape, dtype=dtype, buffer=buffer, offset=offset)


def decode_shared_memory_cache(tree: dict[str, Any], buffer: memoryview) -> Any:
    """Rebuild a NumPy pytree view over a shared-memory cache."""
    kind = tree.get("kind")
    if kind == "dict":
        return {key: decode_shared_memory_cache(value, buffer) for key, value in tree["items"]}
    if kind == "tuple":
        return tuple(decode_shared_memory_cache(value, buffer) for value in tree["items"])
    if kind == "list":
        return [decode_shared_memory_cache(value, buffer) for value in tree["items"]]
    if kind == "array":
        return _shared_memory_array(tree, buffer)
    if kind == "scalar":
        return tree.get("value")
    raise ValueError(f"Unknown shared-memory tree node: {kind!r}")
