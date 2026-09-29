from openpi.training import config


def test_bench2dex_full48_config_uses_four_images_and_absolute_space():
    train_config = config.get_config("pi05_bench2dex_fridge_wine_full48")
    observation_spec, action_spec = train_config.model.inputs_spec(batch_size=2)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    assert train_config.model.action_dim == 48
    assert train_config.model.action_horizon == 20
    assert tuple(observation_spec.images) == (
        "base_0_rgb",
        "base_1_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert action_spec.shape == (2, 20, 48)
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.use_quantile_norm
