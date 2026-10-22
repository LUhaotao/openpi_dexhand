"""HTTP readiness checks for local Runtime workers."""

from __future__ import annotations

import socket
import time
from typing import Any


def wait_for_worker(process: Any, port: int, timeout: float) -> None:
    """Wait until a worker serves its HTTP health endpoint."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"Runtime worker exited before opening port {port}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5) as sock:
                sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                status_line = sock.recv(256).split(b"\r\n", 1)[0]
            if status_line.startswith((b"HTTP/1.1 200 ", b"HTTP/1.0 200 ")):
                return
            time.sleep(0.2)
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for Runtime worker on port {port}")
