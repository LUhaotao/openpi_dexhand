# JAX Pi0.5 experiment configs

`default.json` is the documented source of defaults for file-backed JAX Pi0.5
experiments. Experiment files under `experiments/pi05/` should normally contain
all fields so a reviewer can see the complete run without following inheritance.

Run an experiment with the repository environment:

```bash
.venv/bin/python scripts/train.py \
  --config configs/experiments/pi05/univtac/pull_out_key_baseline.yaml
```

The loader accepts JSON, YAML, and YML. It deep-merges a file with
`configs/default.json`, rejects unknown fields, constructs the existing JAX
`TrainConfig`, and leaves the legacy named-preset CLI available for old jobs.

Every checkpoint writes the resolved experiment to
`<step>/model_config/pi0_config.json`. The `model` object in that file is the
complete model configuration used by JAX inference. Older flat incremental
`pi0_config.json` files remain loadable.

To turn an old incremental config into a complete file:

```bash
.venv/bin/python scripts/expand_config.py path/to/incremental.yaml \
  --output configs/experiments/pi05/my_experiment.yaml
```

The input may also be an old flat `model_config/pi0_config.json`; in that case
its fields are placed under `model`. Review `name`, `exp_name`, `data`, and
`weight_loader` after converting a model-only snapshot because those values did
not exist in the old file.

Supported data adapter names are `univtac`, `franka_xhand`, `bench2dex`, and
`fake`. They select existing Python data factories; transforms remain in Python.
The YAML path only applies to JAX Pi0.5 training.
