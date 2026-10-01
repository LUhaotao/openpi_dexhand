import dataclasses

import openpi.models.pi0_config as _pi0_config


@dataclasses.dataclass(frozen=True)
class FakeEnvConfig(_pi0_config.Pi0Config):
    """Pi0 model config plus production-wire observation dimensions."""

    pi05: bool = True
    paligemma_variant: str = "dummy"
    action_expert_variant: str = "dummy"
    max_token_len: int = 200
    discrete_state_input: bool = True
    image_height: int = 224
    image_width: int = 224
    image_views: int = 3
    prompt: str = "synthetic latency benchmark"
    seed: int = 0
    num_steps: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("image dimensions must be positive")
        if self.image_views != 3:
            raise ValueError("production_wire requires exactly three image views")
        if self.use_torque and self.torque_dim < 1:
            raise ValueError("torque_dim must be positive when use_torque=True")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
