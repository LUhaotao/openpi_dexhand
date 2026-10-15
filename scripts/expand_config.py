#!/usr/bin/env python3
"""Expand an incremental JAX Pi0.5 JSON/YAML config with configs/default.json."""

from __future__ import annotations

import argparse
import pathlib

from openpi.training import experiment_config


def _default_output(input_path: pathlib.Path) -> pathlib.Path:
    return input_path.with_name(f"{input_path.stem}.full.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deep-merge a JSON/YAML fragment with configs/default.json. Legacy flat "
            "model_config/pi0_config.json snapshots are automatically placed under model."
        )
    )
    parser.add_argument("input", type=pathlib.Path, help="Incremental JSON/YAML file")
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Output .json/.yaml path")
    parser.add_argument(
        "--defaults",
        type=pathlib.Path,
        default=experiment_config.DEFAULT_CONFIG_PATH,
        help="Defaults file (default: configs/default.json)",
    )
    args = parser.parse_args()

    output = args.output or _default_output(args.input)
    expanded = experiment_config.load_expanded_config(args.input, defaults_path=args.defaults)
    experiment_config.dump_config(expanded, output)
    print(output)


if __name__ == "__main__":
    main()
