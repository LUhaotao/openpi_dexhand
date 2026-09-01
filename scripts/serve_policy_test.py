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
