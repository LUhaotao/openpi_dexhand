from types import SimpleNamespace
from typing import ClassVar

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
        self.calls.append(("update_prefix", {
            "observation": observation,
            "session_id": session_id,
            "executed_action_id": executed_action_id,
        }))
        return CacheInfo(3, f"model:{session_id}:3")

    def infer(self, request, *, session_id, cache_version):
        self.calls.append(("infer", {
            "request": request,
            "session_id": session_id,
            "cache_version": cache_version,
        }))
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

    coordinator.infer({
        "op": "stream_infer",
        "session_id": "robot-1",
        "observation": {"state": [0]},
        "executed_action_id": 0,
    })
    coordinator.infer({
        "op": "stream_infer",
        "session_id": "robot-1",
        "observation": {"state": [1]},
        "executed_action_id": 1,
        "update_prefix": False,
    })

    assert len([call for call in backend.calls if call[0] == "update_prefix"]) == 1
    assert len([call for call in backend.calls if call[0] == "infer"]) == 2


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
        def __init__(self, policy, role):
            events.append(("create", role))

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
    assert events[0] == ("create", "fm")
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
