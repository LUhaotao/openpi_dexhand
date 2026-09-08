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
