# JAX Pi0.5 experiment configs

`default.json` is the documented source of defaults for file-backed JAX Pi0.5
experiments. Experiment files under `experiments/pi05/` should normally contain
all fields so a reviewer can see the complete run without following inheritance.

Run an experiment with the repository environment:

```bash
.venv/bin/python scripts/train.py \
  --config configs/examples/pi05_univtac_pull_out_key.json
```

The loader accepts JSON. It deep-merges a file with `configs/default.json`,
rejects unknown fields, constructs the existing JAX
`TrainConfig`, and leaves the legacy named-preset CLI available for old jobs.

Every checkpoint writes the resolved experiment to
`<step>/model_config/pi0_config.json`. The `model` object in that file is the
complete model configuration used by JAX inference. Older flat incremental
`pi0_config.json` files remain loadable.

To turn an old incremental config into a complete file:

```bash
.venv/bin/python scripts/expand_config.py path/to/incremental.json \
  --output configs/experiments/pi05/my_experiment.json
```

The input may also be an old flat `model_config/pi0_config.json`; in that case
its fields are placed under `model`. Review `name`, `exp_name`, `data`, and
`weight_loader` after converting a model-only snapshot because those values did
not exist in the old file.

`data.repack_mapping` records the complete LeRobot-to-adapter field mapping. Its
keys are normalized adapter fields and its values are source paths in the
LeRobot sample. For example, `images.cam_side: observation.images.head` maps
the dataset's `head` camera into the Pi0.5 side-camera input path.

Supported data adapter names are `univtac`, `franka_xhand`, `bench2dex`, and
`fake`. They select existing Python data factories; transforms remain in Python.
The JSON path only applies to JAX Pi0.5 training.
