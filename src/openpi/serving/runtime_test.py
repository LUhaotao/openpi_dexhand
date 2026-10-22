import threading
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from openpi.policies import multi_process_policy
from openpi.serving import backends
from openpi.serving import runtime
from openpi.serving.runtime import CacheInfo


class _FakeBackend:
    metadata: ClassVar[dict[str, str]] = {"backend": "test"}

    def __init__(self):
        self.calls = []

    def warmup(self):
        self.calls.append(("warmup", {}))

    def update_prefix(self, observation, *, session_id, executed_action_id=None):
        self.calls.append(
            (
                "update_prefix",
                {
                    "observation": observation,
                    "session_id": session_id,
                    "executed_action_id": executed_action_id,
                },
            )
        )
        return CacheInfo(3, f"model:{session_id}:3")

    def infer(self, request, *, session_id, cache_version):
        self.calls.append(
            (
                "infer",
                {
                    "request": request,
                    "session_id": session_id,
                    "cache_version": cache_version,
                },
            )
        )
        return {"actions": [1], "cache_version": cache_version}

    def reset(self, *, session_id, clear_prefix_cache):
        self.calls.append(("reset", {"session_id": session_id, "clear_prefix_cache": clear_prefix_cache}))
        return {"status": "reset", "prefix_cache_cleared": clear_prefix_cache}

    def close(self):
        self.calls.append(("close", {}))


def test_runtime_coordinates_prefix_and_inference_through_backend():
    backend = _FakeBackend()
    coordinator = runtime.RuntimeCoordinator(backend)

    result = coordinator.infer_with_session({"observation": {"state": [0]}}, session_id="robot-1")

    assert result == {"actions": [1], "cache_version": "model:robot-1:3"}
    assert [op for op, _ in backend.calls] == ["update_prefix", "infer"]
    assert backend.calls[0][1]["session_id"] == "robot-1"
    assert backend.calls[1][1]["cache_version"] == "model:robot-1:3"


def test_runtime_stream_reuses_session_prefix_when_requested():
    backend = _FakeBackend()
    coordinator = runtime.RuntimeCoordinator(backend)

    coordinator.infer(
        {
            "op": "stream_infer",
            "session_id": "robot-1",
            "observation": {"state": [0]},
            "executed_action_id": 0,
        }
    )
    coordinator.infer(
        {
            "op": "stream_infer",
            "session_id": "robot-1",
            "observation": {"state": [1]},
            "executed_action_id": 1,
            "update_prefix": False,
        }
    )

    assert len([call for call in backend.calls if call[0] == "update_prefix"]) == 1
    assert len([call for call in backend.calls if call[0] == "infer"]) == 2


def test_observation_action_runtime_queues_chunks_and_refreshes_prefix_on_refill():
    class ChunkBackend(_FakeBackend):
        def update_prefix(self, observation, *, session_id, executed_action_id=None):
            self.calls.append(
                (
                    "update_prefix",
                    {
                        "observation": observation,
                        "session_id": session_id,
                        "executed_action_id": executed_action_id,
                    },
                )
            )
            return CacheInfo(1, f"cache:{executed_action_id}")

        def infer(self, request, *, session_id, cache_version):
            self.calls.append(
                (
                    "infer",
                    {
                        "request": request,
                        "session_id": session_id,
                        "cache_version": cache_version,
                    },
                )
            )
            start = request["executed_action_id"]
            return {"actions": [[start], [start + 1], [start + 2]]}

    backend = ChunkBackend()
    coordinator = runtime.RuntimeCoordinator(backend, streaming_horizon=4, streaming_watermark=2)
    outputs = [coordinator.infer_with_session({"state": [step]}, session_id="robot") for step in range(4)]

    assert [int(output["actions"][0]) for output in outputs] == [0, 1, 2, 3]
    assert [call[0] for call in backend.calls] == ["update_prefix", "infer", "update_prefix", "infer"]
    infer_requests = [call[1]["request"] for call in backend.calls if call[0] == "infer"]
    assert [request["executed_action_id"] for request in infer_requests] == [0, 3]
    assert all(request["op"] == "stream_infer" for request in infer_requests)
    assert all("session_id" not in request for request in infer_requests)


def test_process_runtime_overlaps_vlm_refresh_with_current_action():
    class AsyncRefreshBackend(_FakeBackend):
        supports_async_prefix_refresh = True

        def __init__(self):
            super().__init__()
            self.refresh_started = threading.Event()
            self.finish_refresh = threading.Event()
            self.refresh_finished = threading.Event()

        def begin_prefix_refresh(self, observation, *, session_id, executed_action_id=None):
            self.calls.append(("begin_refresh", {"executed_action_id": executed_action_id}))
            return {"execution_id": executed_action_id}

        def finish_prefix_refresh(self, refresh):
            executed_action_id = refresh["execution_id"]
            self.calls.append(("update_prefix", {"executed_action_id": executed_action_id}))
            if executed_action_id == 1:
                self.refresh_started.set()
                assert self.finish_refresh.wait(timeout=5)
                self.refresh_finished.set()
            return CacheInfo(executed_action_id + 1, f"cache:{executed_action_id + 1}")

        def infer(self, request, *, session_id, cache_version):
            self.calls.append(("infer", {"execution_id": request["executed_action_id"], "cache": cache_version}))
            return {"actions": [request["executed_action_id"]], "cache_version": cache_version}

    backend = AsyncRefreshBackend()
    coordinator = runtime.RuntimeCoordinator(backend, streaming_horizon=2, streaming_watermark=1)
    try:
        coordinator.infer_with_session({"state": [0]}, session_id="robot")

        # At action id 1, prefix encoding blocks in the background while the
        # current action is inferred from the active cache.
        result = coordinator.infer_with_session({"state": [1]}, session_id="robot")
        assert int(result["actions"][0]) == 1
        assert backend.refresh_started.wait(timeout=1)
        call_names = [call[0] for call in backend.calls]
        infer_indices = [index for index, name in enumerate(call_names) if name == "infer"]
        assert len(infer_indices) == 2
        assert call_names.index("begin_refresh") < infer_indices[1]
        assert backend.calls[infer_indices[1]][1]["cache"] == "model:robot:3"

        backend.finish_refresh.set()
        assert backend.refresh_finished.wait(timeout=2)
        coordinator.infer_with_session({"state": [2]}, session_id="robot")
        infer_calls = [call[1] for call in backend.calls if call[0] == "infer"]
        assert infer_calls[-1]["cache"] == "cache:2"
    finally:
        backend.finish_refresh.set()
        coordinator.close()


def test_observation_action_runtime_reset_clears_queued_actions_and_counter():
    class ChunkBackend(_FakeBackend):
        def infer(self, request, *, session_id, cache_version):
            self.calls.append(("infer", {"request": request}))
            return {"actions": [[10], [11], [12]]}

    backend = ChunkBackend()
    coordinator = runtime.RuntimeCoordinator(backend)
    coordinator.infer_with_session({"state": [0]}, session_id="robot")
    reset = coordinator.infer_with_session({"op": "reset_stream", "clear_prefix_cache": True}, session_id="robot")
    result = coordinator.infer_with_session({"state": [1]}, session_id="robot")

    assert reset["prefix_cache_cleared"] is True
    assert int(result["actions"][0]) == 10
    requests = [call[1]["request"] for call in backend.calls if call[0] == "infer"]
    assert [request["executed_action_id"] for request in requests] == [0, 0]


def test_observation_history_includes_frames_received_while_actions_are_queued():
    class HistoryBackend(_FakeBackend):
        def infer(self, request, *, session_id, cache_version):
            self.calls.append(("infer", {"request": request}))
            return {"actions": [[1], [2], [3]]}

    backend = HistoryBackend()
    coordinator = runtime.RuntimeCoordinator(backend, history_length=3)
    for step in range(4):
        coordinator.infer_with_session(
            {
                "state": [step],
                "left_marker": np.full((2, 1, 2), step, dtype=np.float32),
                "right_marker": np.full((2, 1, 2), step + 10, dtype=np.float32),
            },
            session_id="robot",
        )

    requests = [call[1]["request"] for call in backend.calls if call[0] == "infer"]
    request = requests[-1]
    np.testing.assert_array_equal(request["observation"]["state"][:, 0], [1, 2, 3])
    np.testing.assert_array_equal(request["observation"]["left_marker"][:, 0, 0, 0], [1, 2, 3])


def test_runtime_tracks_cache_independently_per_session():
    backend = _FakeBackend()
    coordinator = runtime.RuntimeCoordinator(backend)
    coordinator.infer({"session_id": "a", "observation": {"state": [0]}})
    coordinator.infer({"session_id": "b", "observation": {"state": [0]}})

    versions = [call[1]["cache_version"] for call in backend.calls if call[0] == "infer"]
    assert versions == ["model:a:3", "model:b:3"]


def test_runtime_reset_can_clear_only_session_cache():
    backend = _FakeBackend()
    coordinator = runtime.RuntimeCoordinator(backend)
    coordinator.infer({"session_id": "a", "observation": {"state": [0]}})
    coordinator.infer({"op": "reset_stream", "session_id": "a", "clear_prefix_cache": True})
    coordinator.infer({"op": "stream_infer", "session_id": "a", "observation": {"state": [1]}})

    assert len([call for call in backend.calls if call[0] == "update_prefix"]) == 2


def test_local_backend_uses_local_worker_without_transport(monkeypatch):
    events = []

    class FakeLocalWorker:
        def __init__(self, policy, role, **kwargs):
            events.append(("create", role, kwargs))

        def encode_prefix_local(self, observation, *, session_id, executed_action_id=None):
            events.append(("prefix", observation, session_id, executed_action_id))
            return {"cache_id": 7, "cache_version": "local:7", "cache": object()}

        def infer(self, request):
            events.append(("infer", request))
            return {"actions": [1]}

        def warmup(self):
            events.append(("warmup",))

        def _cleanup_shared_memory_artifacts(self):
            events.append(("cleanup",))

    monkeypatch.setattr(multi_process_policy, "MultiProcessPolicy", FakeLocalWorker)
    local = backends.LocalBackend(SimpleNamespace(metadata={"model": "test"}))
    info = local.update_prefix({"state": [0]}, session_id="task")
    result = local.infer({"op": "infer"}, session_id="task", cache_version=info.cache_version)

    assert info == CacheInfo(7, "local:7")
    assert result == {"actions": [1]}
    assert events[0] == ("create", "fm", {"streaming_inference_mode": "chunk"})
    assert events[1][0] == "prefix"
    assert events[2][0] == "infer"


def test_process_backend_keeps_cache_payload_out_of_worker_websocket(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, role):
            self.role = role
            self._ws = self

        def get_server_metadata(self):
            return {"role": self.role}

        def infer(self, request):
            calls.append((self.role, request))
            if request["op"] == "observation_started":
                return {"observation_chunk_id": 5}
            if request["op"] == "encode_prefix_to_shared_memory":
                return {
                    "artifact_id": "shm-1",
                    "shared_memory_name": "shm-1",
                    "transport": "shared_memory",
                }
            if request["op"] == "activate_shared_memory_prefix":
                return {"cache_id": 8, "cache_version": "model:8"}
            if request["op"] == "release_shared_memory":
                return {"status": "released"}
            return {"actions": [2]}

        def close(self):
            pass

    clients = iter((FakeClient("vlm"), FakeClient("fm")))
    monkeypatch.setattr(backends, "WebsocketClientPolicy", lambda *args: next(clients))
    process = backends.ProcessBackend(vlm_host="127.0.0.1", vlm_port=1, fm_host="127.0.0.1", fm_port=2)

    info = process.update_prefix({"state": [0]}, session_id="task")
    result = process.infer({"op": "infer"}, session_id="task", cache_version=info.cache_version)

    assert info == CacheInfo(8, "model:8")
    assert result == {"actions": [2]}
    assert [call[1]["op"] for call in calls] == [
        "observation_started",
        "encode_prefix_to_shared_memory",
        "activate_shared_memory_prefix",
        "release_shared_memory",
        "infer",
    ]
    assert calls[2][1]["cache"]["artifact_id"] == "shm-1"
