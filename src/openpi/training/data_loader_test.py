import dataclasses
from types import SimpleNamespace

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_delayed_observation_transform_delays_images_and_discrete_state(monkeypatch):
    repack = _transforms.RepackTransform(
        {
            "images": {"camera": "observation.images.camera"},
            "state": "observation.state",
            "actions": "action",
        }
    )
    transform = _data_loader._DelayedObservationTransform(  # noqa: SLF001
        [repack],
        [_transforms.DeltaActions(mask=[True])],
        max_delay_chunks=2,
        discrete_state_input=True,
    )
    monkeypatch.setattr(np.random, "randint", lambda *args: 2)

    result = transform(
        {
            "observation.images.camera": np.asarray(
                [
                    np.full((2, 2, 3), 10, dtype=np.uint8),
                    np.full((2, 2, 3), 20, dtype=np.uint8),
                    np.full((2, 2, 3), 30, dtype=np.uint8),
                ]
            ),
            "observation.state": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "action": np.asarray([[5.0]], dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(result["images"]["camera"], 30)
    np.testing.assert_array_equal(result["state"], [3.0])
    # Delta actions stay relative to the current state (1.0), not the delayed state (3.0).
    np.testing.assert_array_equal(result["actions"], [[4.0]])


def test_delayed_observation_transform_keeps_continuous_state_current(monkeypatch):
    repack = _transforms.RepackTransform(
        {
            "images": {"camera": "observation.images.camera"},
            "state": "observation.state",
        }
    )
    transform = _data_loader._DelayedObservationTransform(  # noqa: SLF001
        [repack],
        [],
        max_delay_chunks=1,
        discrete_state_input=False,
    )
    monkeypatch.setattr(np.random, "randint", lambda *args: 1)

    result = transform(
        {
            "observation.images.camera": np.asarray(
                [
                    np.full((2, 2, 3), 10, dtype=np.uint8),
                    np.full((2, 2, 3), 20, dtype=np.uint8),
                ]
            ),
            "observation.state": np.asarray([7.0], dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(result["images"]["camera"], 20)
    np.testing.assert_array_equal(result["state"], [7.0])


def test_observation_delay_timestamps_use_chunk_size_for_images_and_discrete_state():
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"camera": "observation.images.camera"},
                        "state": "observation.state",
                    }
                )
            ]
        )
    )
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        streaming=True,
        streaming_chunk_size=5,
        observation_delay_max_chunks=2,
        discrete_state_input=True,
    )
    delta_timestamps = {"action": [0.0]}
    dataset_meta = SimpleNamespace(
        fps=10,
        camera_keys=["observation.images.camera"],
        features={"observation.state": {"dtype": "float32"}},
    )

    _data_loader._add_observation_delay_timestamps(  # noqa: SLF001
        delta_timestamps,
        data_config,
        dataset_meta,
        model_config,
    )

    assert delta_timestamps["observation.images.camera"] == [0.0, -0.5, -1.0]
    assert delta_timestamps["observation.state"] == [0.0, -0.5, -1.0]


def test_tactile_history_timestamps_are_current_to_past():
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "left_marker": "observation.tactile.left_marker",
                        "right_marker": "observation.tactile.right_marker",
                    }
                )
            ]
        )
    )
    model_config = pi0_config.Pi0Config(
        pi05=True,
        use_tactile=True,
        streaming=True,
        streaming_attention_mode="tactile_attention_gate",
        tactile_history_length=4,
    )
    delta_timestamps = {}
    dataset_meta = SimpleNamespace(
        fps=20,
        features={
            "observation.tactile.left_marker": {"dtype": "float32"},
            "observation.tactile.right_marker": {"dtype": "float32"},
        },
    )

    _data_loader._add_tactile_history_timestamps(  # noqa: SLF001
        delta_timestamps, data_config, dataset_meta, model_config
    )

    expected = [-0.15, -0.1, -0.05, 0.0]
    assert delta_timestamps["observation.tactile.left_marker"] == expected
    assert delta_timestamps["observation.tactile.right_marker"] == expected


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
