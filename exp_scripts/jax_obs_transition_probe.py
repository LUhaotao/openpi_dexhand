"""White-box JAX probe for observation transitions during flow matching.

The default all mode runs both experiments:

1. Denoise with observation A for tau steps, switch to B, and compare with
   denoising with B from the beginning.
2. Start from the B/GT interpolation at the same tau and denoise with B.

One tau value is completed and saved before the next tau starts. This keeps
results and per-sample prefix caches from accumulating across tau values.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import pathlib
import subprocess
from typing import Any

import jax
import numpy as np

try:
    from exp_scripts._probe_common import ProbeContext
    from exp_scripts._probe_common import add_probe_arguments
    from exp_scripts._probe_common import block_until_ready
    from exp_scripts._probe_common import latent_rmse
    from exp_scripts._probe_common import metric_mae
    from exp_scripts._probe_common import metric_rmse
    from exp_scripts._probe_common import resolve_tau_steps
    from exp_scripts._probe_common import save_tau_result
    from exp_scripts._probe_common import streaming_history_offset
except ModuleNotFoundError:
    from _probe_common import ProbeContext
    from _probe_common import add_probe_arguments
    from _probe_common import block_until_ready
    from _probe_common import latent_rmse
    from _probe_common import metric_mae
    from _probe_common import metric_rmse
    from _probe_common import resolve_tau_steps
    from _probe_common import save_tau_result
    from _probe_common import streaming_history_offset

DEFAULT_CHECKPOINT = "checkpoints/pi05_franka_xhand_flower_streaming_v2/pi05_franka_xhand_flower_streaming_mask/19999"
DEFAULT_CONFIG_NAME = "pi05_franka_xhand_flower_streaming_v2"
DEFAULT_DATASET = "/public/node01/users/lvrui/datasets/lerobot/flower_xhand_franka"
DEFAULT_OUTPUT_DIR = "results/obs_transition_probe"

METRIC_COLUMNS = (
    "pair_id",
    "index_a",
    "index_b",
    "episode_index",
    "frame_a",
    "frame_b",
    "noise_id",
    "tau_steps",
    "t_switch",
    "rmse_ab",
    "mae_ab",
    "first_chunk_rmse_ab",
    "rmse_bb",
    "mae_bb",
    "first_chunk_rmse_bb",
    "rmse_gt_b",
    "mae_gt_b",
    "first_chunk_rmse_gt_b",
    "latent_rmse_ab_gt",
    "latent_rmse_ab_b",
    "latent_rmse_gt_b",
    "output_rmse_ab_bb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", "exp1", "exp2"),
        default="all",
        help="Run both experiments or only one branch.",
    )
    add_probe_arguments(
        parser,
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_config_name=DEFAULT_CONFIG_NAME,
        default_dataset=DEFAULT_DATASET,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tau files that already exist in the output directory.",
    )
    return parser.parse_args()


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


def _write_manifest(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
    context: ProbeContext,
    *,
    frame_gap: int,
    anchor_frame_gap: int,
    tau_steps: list[int],
) -> None:
    arguments = vars(args).copy()
    arguments["frame_gap"] = frame_gap
    arguments["anchor_frame_gap"] = anchor_frame_gap
    arguments["tau_steps"] = tau_steps
    manifest = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "arguments": arguments,
        "checkpoint": str(context.checkpoint_path),
        "dataset": str(context.dataset_path),
        "norm_stats": str(context.norm_stats_path),
        "model_config": dataclasses.asdict(context.model_config),
        "git_commit": _git_commit(),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        return
    _write_json(manifest_path, manifest)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = tuple(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize_tau(tau_steps: int, rows: list[dict[str, Any]], num_steps: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tau_steps": tau_steps,
        "t_switch": 1.0 - tau_steps / num_steps,
        "count": len(rows),
    }
    for name in (
        "rmse_ab",
        "mae_ab",
        "first_chunk_rmse_ab",
        "rmse_bb",
        "mae_bb",
        "first_chunk_rmse_bb",
        "rmse_gt_b",
        "mae_gt_b",
        "first_chunk_rmse_gt_b",
        "latent_rmse_ab_gt",
        "latent_rmse_ab_b",
        "latent_rmse_gt_b",
        "output_rmse_ab_bb",
    ):
        summary[f"{name}_mean"] = _mean_metric(rows, name)
    return summary


def _stack_or_empty(values: list[np.ndarray], *, dtype: np.dtype | None = None) -> np.ndarray:
    if values:
        return np.stack(values)
    if dtype is None:
        dtype = np.float32
    return np.empty((0,), dtype=dtype)


def _run_tau(
    context: ProbeContext,
    args: argparse.Namespace,
    *,
    tau_steps: int,
    candidate_indices: list[int],
    frame_gap: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ground_truth_actions: list[np.ndarray] = []
    action_ab_values: list[np.ndarray] = []
    action_bb_values: list[np.ndarray] = []
    action_gt_b_values: list[np.ndarray] = []
    model_action_ab_values: list[np.ndarray] = []
    model_action_bb_values: list[np.ndarray] = []
    model_action_gt_b_values: list[np.ndarray] = []
    switch_ab_values: list[np.ndarray] = []
    switch_b_values: list[np.ndarray] = []
    switch_gt_b_values: list[np.ndarray] = []
    pair_ids: list[int] = []
    index_a_values: list[int] = []
    index_b_values: list[int] = []
    episode_values: list[int] = []
    frame_a_values: list[int] = []
    frame_b_values: list[int] = []
    noise_ids: list[int] = []

    include_gt = args.experiment in ("all", "exp2")
    first_chunk = context.chunk_size

    for pair_id, index_b in enumerate(candidate_indices):
        index_a = index_b - frame_gap
        pair = context.pair(pair_id, index_a, index_b)
        cache_a, cache_b, tactile_a, tactile_b = context.encode_pair(pair)
        ground_truth = context.to_output_actions(
            pair.ground_truth_b,
            pair.observation_b.state,
        )

        for noise_id in range(args.num_noise_samples):
            noise = context.noise(
                seed=args.seed,
                pair_id=pair_id,
                noise_id=noise_id,
            )
            result = context.rollout(
                pair,
                cache_a=cache_a,
                cache_b=cache_b,
                tactile_a=tactile_a,
                tactile_b=tactile_b,
                noise=noise,
                tau_steps=tau_steps,
                include_gt=include_gt,
            )
            result = block_until_ready(result)

            action_ab = context.to_output_actions(
                result.final_ab,
                pair.observation_b.state,
            )
            action_bb = context.to_output_actions(
                result.final_b,
                pair.observation_b.state,
            )
            action_gt_b = None
            if include_gt and result.final_gt_b is not None:
                action_gt_b = context.to_output_actions(
                    result.final_gt_b,
                    pair.observation_b.state,
                )

            row: dict[str, Any] = {
                "pair_id": pair_id,
                "index_a": pair.index_a,
                "index_b": pair.index_b,
                "episode_index": pair.episode_index,
                "frame_a": pair.frame_a,
                "frame_b": pair.frame_b,
                "noise_id": noise_id,
                "tau_steps": tau_steps,
                "t_switch": 1.0 - tau_steps / args.num_steps,
                "rmse_ab": metric_rmse(action_ab, ground_truth),
                "mae_ab": metric_mae(action_ab, ground_truth),
                "first_chunk_rmse_ab": metric_rmse(
                    action_ab[:first_chunk],
                    ground_truth[:first_chunk],
                ),
                "rmse_bb": metric_rmse(action_bb, ground_truth),
                "mae_bb": metric_mae(action_bb, ground_truth),
                "first_chunk_rmse_bb": metric_rmse(
                    action_bb[:first_chunk],
                    ground_truth[:first_chunk],
                ),
                "rmse_gt_b": None,
                "mae_gt_b": None,
                "first_chunk_rmse_gt_b": None,
                "latent_rmse_ab_gt": None,
                "latent_rmse_ab_b": latent_rmse(
                    result.switch_ab,
                    result.switch_b,
                ),
                "latent_rmse_gt_b": None,
                "output_rmse_ab_bb": metric_rmse(action_ab, action_bb),
            }
            if include_gt and action_gt_b is not None and result.switch_gt_b is not None:
                row["rmse_gt_b"] = metric_rmse(action_gt_b, ground_truth)
                row["mae_gt_b"] = metric_mae(action_gt_b, ground_truth)
                row["first_chunk_rmse_gt_b"] = metric_rmse(
                    action_gt_b[:first_chunk],
                    ground_truth[:first_chunk],
                )
                row["latent_rmse_ab_gt"] = latent_rmse(
                    result.switch_ab,
                    result.switch_gt_b,
                )
                row["latent_rmse_gt_b"] = latent_rmse(
                    result.switch_gt_b,
                    result.switch_b,
                )

            rows.append(row)
            ground_truth_actions.append(ground_truth)
            action_ab_values.append(action_ab)
            action_bb_values.append(action_bb)
            pair_ids.append(pair_id)
            index_a_values.append(pair.index_a)
            index_b_values.append(pair.index_b)
            episode_values.append(pair.episode_index)
            frame_a_values.append(pair.frame_a)
            frame_b_values.append(pair.frame_b)
            noise_ids.append(noise_id)
            if args.save_traces:
                model_action_ab_values.append(np.asarray(result.final_ab[0]))
                model_action_bb_values.append(np.asarray(result.final_b[0]))
                switch_ab_values.append(np.asarray(result.switch_ab[0]))
                switch_b_values.append(np.asarray(result.switch_b[0]))
            if action_gt_b is not None:
                action_gt_b_values.append(action_gt_b)
            if args.save_traces and result.final_gt_b is not None and result.switch_gt_b is not None:
                model_action_gt_b_values.append(np.asarray(result.final_gt_b[0]))
                switch_gt_b_values.append(np.asarray(result.switch_gt_b[0]))

        del (
            pair,
            cache_a,
            cache_b,
            tactile_a,
            tactile_b,
            noise,
            result,
            ground_truth,
            action_ab,
            action_bb,
            action_gt_b,
        )

    if not rows:
        raise RuntimeError(f"No valid samples were processed for tau={tau_steps}.")

    arrays = {
        "pair_id": np.asarray(pair_ids, dtype=np.int64),
        "index_a": np.asarray(index_a_values, dtype=np.int64),
        "index_b": np.asarray(index_b_values, dtype=np.int64),
        "episode_index": np.asarray(episode_values, dtype=np.int64),
        "frame_a": np.asarray(frame_a_values, dtype=np.int64),
        "frame_b": np.asarray(frame_b_values, dtype=np.int64),
        "noise_id": np.asarray(noise_ids, dtype=np.int64),
        "tau_steps": np.asarray(tau_steps, dtype=np.int64),
        "t_switch": np.asarray(1.0 - tau_steps / args.num_steps, dtype=np.float32),
        "ground_truth_actions": _stack_or_empty(ground_truth_actions),
        "action_ab": _stack_or_empty(action_ab_values),
        "action_bb": _stack_or_empty(action_bb_values),
        "action_gt_b": _stack_or_empty(action_gt_b_values),
        "summary_metrics": np.asarray(
            [
                (np.nan if (value := _mean_metric(rows, "rmse_ab")) is None else value),
                (np.nan if (value := _mean_metric(rows, "rmse_bb")) is None else value),
                (np.nan if (value := _mean_metric(rows, "rmse_gt_b")) is None else value),
                (np.nan if (value := _mean_metric(rows, "latent_rmse_ab_gt")) is None else value),
                (np.nan if (value := _mean_metric(rows, "latent_rmse_ab_b")) is None else value),
            ],
            dtype=np.float32,
        ),
    }
    if args.save_traces:
        arrays.update(
            {
                "action_ab_model": _stack_or_empty(model_action_ab_values),
                "action_bb_model": _stack_or_empty(model_action_bb_values),
                "action_gt_b_model": _stack_or_empty(model_action_gt_b_values),
                "x_switch_ab": _stack_or_empty(switch_ab_values),
                "x_switch_b": _stack_or_empty(switch_b_values),
                "x_switch_gt_b": _stack_or_empty(switch_gt_b_values),
            }
        )
    return arrays, rows


def _write_aggregate_metrics(output_dir: pathlib.Path, tau_steps: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for tau in tau_steps:
        path = output_dir / f"metrics_tau_{tau:03d}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    if rows:
        _write_csv(output_dir / "metrics.csv", rows)


def main() -> None:
    args = parse_args()
    if args.num_noise_samples <= 0:
        raise ValueError("--num-noise-samples must be positive.")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive.")

    context = ProbeContext.load(args)
    context.configure_rollouts(args.num_steps)
    tau_steps = resolve_tau_steps(args.tau_steps, args.num_steps)
    frame_gap = args.frame_gap
    if frame_gap is None:
        frame_gap = context.chunk_size if context.model_config.streaming else 1
    if frame_gap <= 0:
        raise ValueError("--frame-gap must be positive.")

    if context.model_config.streaming:
        anchor_frame_gap = streaming_history_offset(
            action_horizon=context.action_horizon,
            chunk_size=context.chunk_size,
            frame_gap=frame_gap,
        )
    else:
        anchor_frame_gap = frame_gap

    output_dir = _prepare_output_dir(args)
    _write_manifest(
        output_dir,
        args,
        context,
        frame_gap=frame_gap,
        anchor_frame_gap=anchor_frame_gap,
        tau_steps=tau_steps,
    )
    candidate_indices = context.candidate_indices(
        frame_gap=anchor_frame_gap,
        max_pairs=args.max_pairs,
    )
    summary_rows: list[dict[str, Any]] = []

    for tau in tau_steps:
        npz_path = output_dir / f"tau_{tau:03d}.npz"
        csv_path = output_dir / f"metrics_tau_{tau:03d}.csv"
        if args.resume and npz_path.exists() and csv_path.exists():
            print(f"[skip] tau={tau}: {npz_path}")
            with csv_path.open(newline="") as file:
                existing_rows = list(csv.DictReader(file))
            summary_rows.append(_summarize_tau_from_csv(existing_rows, tau, args.num_steps))
            continue

        print(f"[run] tau={tau}/{args.num_steps - 1}, pairs={len(candidate_indices)}")
        arrays, rows = _run_tau(
            context,
            args,
            tau_steps=tau,
            candidate_indices=candidate_indices,
            frame_gap=frame_gap,
        )
        save_tau_result(npz_path, arrays=arrays)
        _write_csv(csv_path, rows)
        summary_rows.append(_summarize_tau(tau, rows, args.num_steps))
        _write_summary_csv(output_dir / "summary.csv", summary_rows)
        _write_aggregate_metrics(output_dir, tau_steps)
        print(
            f"[done] tau={tau}: "
            f"rmse_ab={summary_rows[-1]['rmse_ab_mean']:.6f}, "
            f"rmse_bb={summary_rows[-1]['rmse_bb_mean']:.6f}"
        )

    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_aggregate_metrics(output_dir, tau_steps)
    _write_json(
        output_dir / "summary.json",
        {"by_tau": {str(row["tau_steps"]): row for row in summary_rows}},
    )
    print(f"Results saved to {output_dir}")


def _summarize_tau_from_csv(
    rows: list[dict[str, Any]],
    tau_steps: int,
    num_steps: int,
) -> dict[str, Any]:
    converted = [
        {
            key: (None if row[key] in ("", "None", "nan") else float(row[key]))
            for key in (
                "rmse_ab",
                "mae_ab",
                "first_chunk_rmse_ab",
                "rmse_bb",
                "mae_bb",
                "first_chunk_rmse_bb",
                "rmse_gt_b",
                "mae_gt_b",
                "first_chunk_rmse_gt_b",
                "latent_rmse_ab_gt",
                "latent_rmse_ab_b",
                "latent_rmse_gt_b",
                "output_rmse_ab_bb",
            )
        }
        for row in rows
    ]
    return _summarize_tau(tau_steps, converted, num_steps)


if __name__ == "__main__":
    main()
