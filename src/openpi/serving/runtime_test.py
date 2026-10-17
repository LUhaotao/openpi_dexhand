from openpi.serving import runtime


class _FakeClient:
    def __init__(self, role: str, calls: list[tuple[str, dict]]) -> None:
        self.role = role
        self.calls = calls
        self._ws = self

    def get_server_metadata(self):
        return {"role": self.role}

    def infer(self, request):
        self.calls.append((self.role, request))
        if self.role == "vlm":
            return {
                "protocol_version": 1,
                "model_id": "model",
                "cache_id": 3,
                "cache_version": "model:3",
                "prefix_cache": {"cache": 1},
            }
        if request["op"] == "observation_started":
            return {"observation_chunk_id": 4}
        if request["op"] == "activate_prefix":
            return {"cache_id": 3, "cache_version": "model:3", "status": "active"}
        if request["op"] == "reset_stream":
            return {"status": "reset", "prefix_cache_cleared": request["clear_prefix_cache"]}
        return {"actions": [1], "cache_version": "model:3"}

    def close(self):
        pass


def test_runtime_coordinates_prefix_activation_and_inference(monkeypatch):
    calls = []
    clients = iter((_FakeClient("vlm", calls), _FakeClient("fm", calls)))
    monkeypatch.setattr(runtime, "WebsocketClientPolicy", lambda *args: next(clients))

    coordinator = runtime.RuntimeCoordinator(
        vlm_host="127.0.0.1",
        vlm_port=1,
        fm_host="127.0.0.1",
        fm_port=2,
    )

    result = coordinator.infer_with_session({"observation": {"state": [0]}}, session_id="robot-1")

    assert result == {"actions": [1], "cache_version": "model:3"}
    assert [role for role, _ in calls] == ["fm", "vlm", "fm", "fm"]
    assert calls[0][1]["op"] == "observation_started"
    assert calls[1][1]["op"] == "encode_prefix_and_get_cache"
    assert calls[1][1]["observation_chunk_id"] == 4
    assert calls[2][1]["op"] == "activate_prefix"
    assert calls[3][1]["session_id"] == "robot-1"


def test_runtime_stream_reuses_prefix_when_requested(monkeypatch):
    calls = []
    clients = iter((_FakeClient("vlm", calls), _FakeClient("fm", calls)))
    monkeypatch.setattr(runtime, "WebsocketClientPolicy", lambda *args: next(clients))
    coordinator = runtime.RuntimeCoordinator(
        vlm_host="127.0.0.1", vlm_port=1, fm_host="127.0.0.1", fm_port=2
    )

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

    assert len([request for role, request in calls if role == "vlm"]) == 1
    assert len([request for role, request in calls if role == "fm" and request["op"] == "stream_infer"]) == 2
