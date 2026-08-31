import dataclasses
import enum
import logging
import multiprocessing
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.multi_process_policy import MultiProcessPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Start separate VLM and FM websocket servers. The default path is unchanged.
    multi_process: bool = False
    vlm_port: int | None = None
    vlm_host: str = "127.0.0.1"

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _multi_process_config(args, _config.get_config(args.policy.config))
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT[args.env]
            train_config = _multi_process_config(args, _config.get_config(checkpoint.config))
            return _policy_config.create_trained_policy(
                train_config, checkpoint.dir, default_prompt=args.default_prompt
            )


def _multi_process_config(args: Args, train_config: _config.TrainConfig) -> _config.TrainConfig:
    """Use fresh continuous state in the FM suffix for split Pi05 inference."""
    model = train_config.model
    if (
        args.multi_process
        and isinstance(model, _config.pi0_config.Pi0Config)
        and model.pi05
        and model.discrete_state_input
    ):
        return dataclasses.replace(
            train_config,
            model=dataclasses.replace(model, discrete_state_input=False),
        )
    return train_config


def main(args: Args) -> None:
    if args.multi_process:
        _serve_multi_process(args)
        return

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def _serve_multi_process_role(args: Args, role: str, port: int) -> None:
    policy = create_policy(args)
    role_policy = MultiProcessPolicy(
        policy, role, vlm_host=args.vlm_host, vlm_port=args.vlm_port
    )
    metadata = {**policy.metadata, "multi_process_role": role}
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=role_policy,
        host="0.0.0.0",
        port=port,
        metadata=metadata,
    )
    server.serve_forever()


def _serve_multi_process(args: Args) -> None:
    """Start blocking VLM and FM servers in separate processes."""
    if args.vlm_port is None:
        args.vlm_port = args.port + 1

    process_context = multiprocessing.get_context("spawn")
    vlm_process = process_context.Process(
        target=_serve_multi_process_role, args=(args, "vlm", args.vlm_port), daemon=True
    )
    fm_process = process_context.Process(
        target=_serve_multi_process_role, args=(args, "fm", args.port), daemon=True
    )
    vlm_process.start()
    fm_process.start()
    logging.info("Multi-process servers started: FM=%s, VLM=%s", args.port, args.vlm_port)
    try:
        vlm_process.join()
        fm_process.join()
    except KeyboardInterrupt:
        vlm_process.terminate()
        fm_process.terminate()
        vlm_process.join()
        fm_process.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
