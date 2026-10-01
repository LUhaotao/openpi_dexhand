import flax.nnx as nnx
import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as training_config


def test_freeze_vlm_keeps_action_expert_trainable():
    config = training_config.TrainConfig(
        name="freeze_vlm_test",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            use_tactile=True,
            use_torque=True,
            torque_dim=32,
            streaming=True,
            streaming_attention_mode="attention_gate",
            gate_sources=("torque", "tactile", "state"),
        ),
        freeze_vlm=True,
    )
    model = config.model.create(jax.random.key(0))
    frozen = {
        "/".join(map(str, path))
        for path in nnx.state(model, nnx.All(nnx.Param, config.effective_freeze_filter)).flat_state()
    }
    trainable = {"/".join(map(str, path)) for path in nnx.state(model, config.trainable_filter).flat_state()}

    assert any(path.startswith("PaliGemma/img/") for path in frozen)
    assert any(path.startswith("PaliGemma/llm/") for path in frozen)
    for prefix in (
        "marker_mlp_in/", "marker_mlp_out/", "marker_fusion/", "torque_mlp_in/", "torque_mlp_out/",
        "state_proj/", "gate_",
    ):
        assert any(path.startswith(prefix) for path in frozen)
    assert any(path.startswith("PaliGemma/llm/") and "_1/" in path for path in trainable)
    assert any(path.startswith("action_in_proj/") for path in trainable)
    assert any(path.startswith("gate_") for path in frozen)
    assert any(path.startswith("action_out_proj/") for path in trainable)
    assert frozen.isdisjoint(trainable)


def test_freeze_vlm_requires_pi05():
    with pytest.raises(ValueError, match="only supported for JAX Pi0.5"):
        training_config.TrainConfig(name="pi0_freeze", model=pi0_config.Pi0Config(), freeze_vlm=True)
