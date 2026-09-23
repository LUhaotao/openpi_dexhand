"""Replay streaming inference with a VLM refresh waterline.

The FM window advances once per action chunk, while the active VLM prefix is
held fixed for frame_gap action frames. At a refresh boundary the new
observation is activated, and all clean chunks produced during the next
refresh interval are evaluated together.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import pathlib
import subprocess
from typing import Any

import jax
import numpy as np

from exp_scripts._probe_common import ProbeContext
from exp_scripts._probe_common import _load_episode_candidates
from exp_scripts._probe_common import metric_mae
from exp_scripts._probe_common import save_tau_result
from exp_scripts._probe_common import streaming_history_offset
from exp_scripts.jax_streaming_transition_probe import StreamingRunner
from exp_scripts.jax_streaming_transition_probe import _encode_observation
from exp_scripts.jax_streaming_transition_probe import _folded_key
from exp_scripts.jax_streaming_transition_probe import _load_observation

DEFAULT_CHECKPOINT = "checkpoints/pi05_franka_xhand_flower_streaming_v2/pi05_franka_xhand_flower_streaming_mask/19999"
DEFAULT_CONFIG_NAME = "pi05_franka_xhand_flower_streaming_v2"
DEFAULT_DATASET = "/public/node01/users/lvrui/datasets/lerobot/flower_xhand_franka"
DEFAULT_OUTPUT_DIR = "results/streaming_waterline_probe"

METRIC_COLUMNS = (
    "pair_id",
    "index_seed",
    "index_b",
    "episode_index",
    "frame_seed",
    "frame_b",
    "noise_id",
    "frame_gap",
    "refresh_advance_count",
    "interval_action_count",
    "mse_ab",
    "mae_ab",
    "mse_b_cold",
    "mae_b_cold",
    "mse_gt_b",
    "mae_gt_b",
    "first_chunk_mse_ab",
    "first_chunk_mse_b_cold",
    "first_chunk_mse_gt_b",
    "transition_mse",
    "transition_rmse",
    "transition_l2_after_b",
    "transition_l2_to_b_seed",
    "output_mse_ab_b_cold",
    "output_mse_ab_gt_b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=20,
        help="VLM refresh interval in action frames. Must be a multiple of chunk_size.",
    )
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save full pre/post-refresh windows and transition vectors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed waterline.npz in the output directory.",
    )
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _git_commit() -> str | None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _prepare_output_dir(args: argparse.Namespace) -> pathlib.Path:
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not (args.overwrite or args.resume):
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --resume or --overwrite.")
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction) - np.asarray(target))))


def _summary(rows: list[dict[str, Any]], frame_gap: int, refresh_advance_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(rows),
        "frame_gap": frame_gap,
        "refresh_advance_count": refresh_advance_count,
        "interval_action_count": frame_gap,
    }
    for name in (
        "mse_ab",
        "mae_ab",
        "mse_b_cold",
        "mae_b_cold",
        "mse_gt_b",
        "mae_gt_b",
        "first_chunk_mse_ab",
        "first_chunk_mse_b_cold",
        "first_chunk_mse_gt_b",
        "transition_mse",
        "transition_rmse",
        "transition_l2_after_b",
        "transition_l2_to_b_seed",
        "output_mse_ab_b_cold",
        "output_mse_ab_gt_b",
    ):
        values = [row[name] for row in rows if row[name] is not None]
        result[f"{name}_mean"] = None if not values else float(np.mean(values))
    return result


def _stack_results(result_lists: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack(values) for name, values in result_lists.items() if values}


def _normal_noise(
    seed: int,
    pair_id: int,
    noise_id: int,
    stream_slot: int,
    shape: tuple[int, ...],
    *,
    step: int = 0,
) -> jax.Array:
    return jax.random.normal(
        _folded_key(seed, pair_id, noise_id, stream_slot, step),
        shape,
        dtype=jax.numpy.float32,
    )


def _run_pair(
    context: ProbeContext,
    runner: StreamingRunner,
    *,
    pair_id: int,
    index_b: int,
    frame_gap: int,
    seed: int,
    noise_id: int,
    save_traces: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    chunk_size = runner.chunk_size
    refresh_advance_count = frame_gap // chunk_size
    old_intervals = math.ceil(runner.num_chunks / refresh_advance_count)
    index_seed = index_b - old_intervals * frame_gap

    target_pair = context.pair(pair_id, index_b, index_b)
    observation_b = target_pair.observation_b
    cache_b, tactile_b = _encode_observation(context, observation_b)

    sample_noise = _normal_noise(
        seed,
        pair_id,
        noise_id,
        stream_slot=0,
        shape=(1, runner.action_horizon, runner.action_dim),
    )
    renoise_noise = _normal_noise(
        seed,
        pair_id,
        noise_id,
        stream_slot=1,
        shape=(1, runner.action_horizon, runner.action_dim),
    )

    observation = _load_observation(context, pair_id, index_seed)
    cache, tactile = _encode_observation(context, observation)
    window = runner.seed_window(
        rng=_folded_key(seed, pair_id, noise_id, stream_slot=0),
        state=observation.state,
        prefix_cache=cache,
        tactile_condition=tactile,
        sample_noise=sample_noise,
        renoise_noise=renoise_noise,
    )
    del observation, cache, tactile

    observation_indices = [index_seed]
    for interval_index in range(old_intervals):
        current_index = index_seed + interval_index * frame_gap
        if interval_index > 0:
            observation_indices.append(current_index)
        observation = _load_observation(context, pair_id, current_index)
        cache, tactile = _encode_observation(context, observation)
        for advance_in_interval in range(refresh_advance_count):
            advance_index = interval_index * refresh_advance_count + advance_in_interval
            fresh_noise = _normal_noise(
                seed,
                pair_id,
                noise_id,
                stream_slot=2,
                step=advance_index,
                shape=(1, chunk_size, runner.action_dim),
            )
            window = runner.advance_window(
                window,
                observation.state,
                cache,
                tactile,
                fresh_noise,
            )
        del observation, cache, tactile

    expected_b_before_index = index_b - frame_gap
    if observation_indices[-1] != expected_b_before_index:
        observation_indices.append(expected_b_before_index)
    window_before_b = window

    fresh_noises_b = [
        _normal_noise(
            seed,
            pair_id,
            noise_id,
            stream_slot=3,
            step=advance_index,
            shape=(1, chunk_size, runner.action_dim),
        )
        for advance_index in range(refresh_advance_count)
    ]

    window_after_b = window_before_b
    ab_chunks = []
    for fresh_noise in fresh_noises_b:
        window_after_b = runner.advance_window(
            window_after_b,
            observation_b.state,
            cache_b,
            tactile_b,
            fresh_noise,
        )
        ab_chunks.append(window_after_b[:, :chunk_size])
    action_ab_model = jax.numpy.concatenate(ab_chunks, axis=1)

    window_b_seed = runner.seed_window(
        rng=_folded_key(seed, pair_id, noise_id, stream_slot=0),
        state=observation_b.state,
        prefix_cache=cache_b,
        tactile_condition=tactile_b,
        sample_noise=sample_noise,
        renoise_noise=renoise_noise,
    )
    window_b_cold = window_b_seed
    cold_chunks = []
    for fresh_noise in fresh_noises_b:
        window_b_cold = runner.advance_window(
            window_b_cold,
            observation_b.state,
            cache_b,
            tactile_b,
            fresh_noise,
        )
        cold_chunks.append(window_b_cold[:, :chunk_size])
    action_b_cold_model = jax.numpy.concatenate(cold_chunks, axis=1)

    window_gt_before_b = runner.gt_window(
        target_pair.ground_truth_b,
        renoise_noise,
    )
    window_gt_after_b = window_gt_before_b
    gt_chunks = []
    for fresh_noise in fresh_noises_b:
        window_gt_after_b = runner.advance_window(
            window_gt_after_b,
            observation_b.state,
            cache_b,
            tactile_b,
            fresh_noise,
        )
        gt_chunks.append(window_gt_after_b[:, :chunk_size])
    action_gt_b_model = jax.numpy.concatenate(gt_chunks, axis=1)

    ground_truth = context.to_output_actions(
        target_pair.ground_truth_b,
        observation_b.state,
    )
    target_interval = ground_truth[:frame_gap]
    output_ab = context.to_output_actions(action_ab_model, observation_b.state)
    output_b_cold = context.to_output_actions(action_b_cold_model, observation_b.state)
    output_gt_b = context.to_output_actions(action_gt_b_model, observation_b.state)

    transition_delta = _as_numpy(window_before_b[:, chunk_size:] - window_gt_before_b[:, chunk_size:])[0]
    transition_abs = np.abs(transition_delta)
    transition_l2_per_action = np.sqrt(np.mean(np.square(transition_delta), axis=-1))
    transition_l2_per_chunk = np.asarray(
        [
            transition_l2_per_action[index : index + chunk_size].mean()
            for index in range(0, runner.action_horizon, chunk_size)
        ],
        dtype=np.float32,
    )
    after_delta = _as_numpy(window_after_b[:, chunk_size:] - window_gt_after_b[:, chunk_size:])
    b_seed_delta = _as_numpy(window_before_b[:, chunk_size:] - window_b_seed[:, chunk_size:])

    row: dict[str, Any] = {
        "pair_id": pair_id,
        "index_seed": index_seed,
        "index_b": target_pair.index_b,
        "episode_index": target_pair.episode_index,
        "frame_seed": target_pair.frame_b - old_intervals * frame_gap,
        "frame_b": target_pair.frame_b,
        "noise_id": noise_id,
        "frame_gap": frame_gap,
        "refresh_advance_count": refresh_advance_count,
        "interval_action_count": frame_gap,
        "mse_ab": _mse(output_ab, target_interval),
        "mae_ab": metric_mae(output_ab, target_interval),
        "mse_b_cold": _mse(output_b_cold, target_interval),
        "mae_b_cold": metric_mae(output_b_cold, target_interval),
        "mse_gt_b": _mse(output_gt_b, target_interval),
        "mae_gt_b": metric_mae(output_gt_b, target_interval),
        "first_chunk_mse_ab": _mse(output_ab[:chunk_size], target_interval[:chunk_size]),
        "first_chunk_mse_b_cold": _mse(
            output_b_cold[:chunk_size],
            target_interval[:chunk_size],
        ),
        "first_chunk_mse_gt_b": _mse(
            output_gt_b[:chunk_size],
            target_interval[:chunk_size],
        ),
        "transition_mse": float(np.mean(np.square(transition_delta))),
        "transition_rmse": float(np.sqrt(np.mean(np.square(transition_delta)))),
        "transition_l2_after_b": float(np.sqrt(np.mean(np.square(after_delta)))),
        "transition_l2_to_b_seed": float(np.sqrt(np.mean(np.square(b_seed_delta)))),
        "output_mse_ab_b_cold": _mse(output_ab, output_b_cold),
        "output_mse_ab_gt_b": _mse(output_ab, output_gt_b),
    }

    arrays = {
        "observation_indices": np.asarray(
            [*observation_indices, index_b],
            dtype=np.int64,
        ),
        "ground_truth_action": target_interval,
        "action_ab": output_ab,
        "action_b_cold": output_b_cold,
        "action_gt_b": output_gt_b,
        "transition_delta": transition_delta,
        "transition_abs": transition_abs,
        "transition_l2_per_action": transition_l2_per_action,
        "transition_l2_per_chunk": transition_l2_per_chunk,
    }
    if save_traces:
        arrays.update(
            {
                "window_before_b": _as_numpy(window_before_b[0]),
                "window_after_b": _as_numpy(window_after_b[0]),
                "window_b_seed": _as_numpy(window_b_seed[0]),
                "window_gt_before_b": _as_numpy(window_gt_before_b[0]),
                "window_gt_after_b": _as_numpy(window_gt_after_b[0]),
            }
        )
    return row, arrays


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.frame_gap <= 0:
        raise ValueError("--frame-gap must be positive.")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive.")
    if args.num_noise_samples <= 0:
        raise ValueError("--num-noise-samples must be positive.")

    context = ProbeContext.load(args)
    runner = StreamingRunner(context, num_steps=args.num_steps)
    if args.frame_gap % runner.chunk_size:
        raise ValueError(
            f"--frame-gap must be a multiple of streaming_chunk_size ({runner.chunk_size}); got {args.frame_gap}."
        )
    if args.frame_gap > context.action_horizon:
        raise ValueError(
            f"--frame-gap must not exceed action_horizon ({context.action_horizon}); got {args.frame_gap}."
        )

    refresh_advance_count = args.frame_gap // runner.chunk_size
    old_intervals = math.ceil(runner.num_chunks / refresh_advance_count)
    history_offset = streaming_history_offset(
        action_horizon=context.action_horizon,
        chunk_size=runner.chunk_size,
        frame_gap=args.frame_gap,
    )
    expected_history_offset = old_intervals * args.frame_gap
    if history_offset != expected_history_offset:
        raise RuntimeError("The shared streaming history offset disagrees with the local waterline calculation.")
    candidate_indices = _load_episode_candidates(
        context.dataset_path,
        frame_gap=history_offset,
        action_horizon=context.action_horizon,
        max_pairs=args.max_pairs,
    )

    output_dir = _prepare_output_dir(args)
    if args.resume and (output_dir / "waterline.npz").exists() and (output_dir / "metrics.csv").exists():
        print(f"[skip] completed result exists: {output_dir / 'waterline.npz'}")
        return

    manifest = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "arguments": {
            **vars(args),
            "refresh_advance_count": refresh_advance_count,
            "old_intervals": old_intervals,
            "history_offset": history_offset,
            "anchor_frame_gap": history_offset,
        },
        "checkpoint": str(context.checkpoint_path),
        "dataset": str(context.dataset_path),
        "norm_stats": str(context.norm_stats_path),
        "model_config": dataclasses.asdict(context.model_config),
        "git_commit": _git_commit(),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    manifest_path = output_dir / "manifest.json"
    if not (args.resume and manifest_path.exists()):
        _write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    result_lists: dict[str, list[np.ndarray]] = {}
    for pair_id, index_b in enumerate(candidate_indices):
        for noise_id in range(args.num_noise_samples):
            print(f"[run] pair={pair_id + 1}/{len(candidate_indices)}, noise={noise_id + 1}/{args.num_noise_samples}")
            row, arrays = _run_pair(
                context,
                runner,
                pair_id=pair_id,
                index_b=index_b,
                frame_gap=args.frame_gap,
                seed=args.seed,
                noise_id=noise_id,
                save_traces=args.save_traces,
            )
            rows.append(row)
            for name, value in arrays.items():
                result_lists.setdefault(name, []).append(np.asarray(value))

    result_arrays = _stack_results(result_lists)
    result_arrays["pair_id"] = np.asarray([row["pair_id"] for row in rows], dtype=np.int64)
    result_arrays["noise_id"] = np.asarray([row["noise_id"] for row in rows], dtype=np.int64)
    result_arrays["frame_gap"] = np.asarray(args.frame_gap, dtype=np.int64)
    result_arrays["refresh_advance_count"] = np.asarray(refresh_advance_count, dtype=np.int64)
    result_arrays["old_intervals"] = np.asarray(old_intervals, dtype=np.int64)
    result_arrays["anchor_frame_gap"] = np.asarray(history_offset, dtype=np.int64)
    save_tau_result(output_dir / "waterline.npz", arrays=result_arrays)
    _write_csv(output_dir / "metrics.csv", rows)
    _write_json(
        output_dir / "summary.json",
        {"summary": _summary(rows, args.frame_gap, refresh_advance_count)},
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
