
import socket

for port in (8000, 8001, 8010):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        print(port, "FREE")
    except OSError as e:
        print(port, "BUSY:", e)
    finally:
        s.close()