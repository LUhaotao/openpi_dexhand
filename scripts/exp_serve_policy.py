import dataclasses
import logging
import socket

import tyro

from openpi.serving import exp_websocket_policy_server
from serve_policy import Args, create_policy


@dataclasses.dataclass
class ExpArgs(Args):
    """Arguments for the experimental server with optional inference RTC."""

    enable_rtc: bool = False


def main(args: ExpArgs) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    policy_metadata["supports_rtc"] = args.enable_rtc

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    logging.info(
        "Creating experimental server (host: %s, ip: %s, RTC: %s)",
        hostname,
        local_ip,
        args.enable_rtc,
    )

    server = exp_websocket_policy_server.ExpWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        enable_rtc=args.enable_rtc,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(ExpArgs))
