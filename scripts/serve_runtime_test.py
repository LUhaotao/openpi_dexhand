from openpi.serving import worker_health


class _FakeSocket:
    def __init__(self):
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def sendall(self, data: bytes) -> None:
        self.sent = data

    def recv(self, _size: int) -> bytes:
        return b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nOK\n"


class _LiveProcess:
    def is_alive(self) -> bool:
        return True


def test_worker_readiness_probe_uses_health_http_request(monkeypatch):
    sock = _FakeSocket()
    monkeypatch.setattr(worker_health.socket, "create_connection", lambda *_args, **_kwargs: sock)
    monkeypatch.setattr(worker_health.time, "monotonic", lambda: 0.0)

    worker_health.wait_for_worker(_LiveProcess(), 12345, timeout=1.0)

    assert sock.sent.startswith(b"GET /healthz HTTP/1.1\r\n")
