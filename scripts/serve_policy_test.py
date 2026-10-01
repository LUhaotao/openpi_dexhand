import pytest

from scripts import serve_policy


def test_single_role_server_uses_requested_port(monkeypatch):
    served = []
    monkeypatch.setattr(
        serve_policy,
        "_serve_multi_process_role",
        lambda args, role, port: served.append((role, port)),
    )

    serve_policy.main(serve_policy.Args(multi_process_role="vlm", port=8001))

    assert served == [("vlm", 8001)]


def test_fm_single_role_requires_vlm_port():
    with pytest.raises(ValueError, match="vlm_port"):
        serve_policy.main(serve_policy.Args(multi_process_role="fm"))


def test_single_process_server_warms_up_before_serving(monkeypatch):
    events = []

    class FakePolicy:
        metadata = {}

        def warmup(self):
            events.append("warmup")

    class FakeServer:
        def __init__(self, **kwargs):
            events.append("server_init")

        def serve_forever(self):
            events.append("serve")

    monkeypatch.setattr(serve_policy, "create_policy", lambda args: FakePolicy())
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", FakeServer)

    serve_policy.main(serve_policy.Args())

    assert events == ["warmup", "server_init", "serve"]


def test_create_policy_applies_gate_log_runtime_flag(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.enabled = None

        def set_gate_log_enabled(self, enabled):
            self.enabled = enabled

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    monkeypatch.setattr(serve_policy, "_policy_config", type("PolicyConfig", (), {
        "create_trained_policy": staticmethod(lambda *args, **kwargs: policy),
    }))
    monkeypatch.setattr(serve_policy, "_config", type("Config", (), {
        "get_config": staticmethod(lambda name: type("TrainConfig", (), {"model": object()})()),
    }))

    result = serve_policy.create_policy(
        serve_policy.Args(save_gate_logs=True, policy=serve_policy.Checkpoint(config="test", dir="checkpoint"))
    )

    assert result is policy
    assert policy._model.enabled is True
