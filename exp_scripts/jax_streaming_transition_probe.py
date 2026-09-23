"""Replay the JAX streaming window with a chronological observation history.

The probe starts a streaming window from the oldest observation needed for the
action horizon, advances it once at every chunk boundary, and uses the current
observation on the final advance. This mirrors the seed/advance algebra used
by the policy server while keeping all intermediate windows observable.
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
import jax.numpy as jnp
import numpy as np

from exp_scripts._probe_common import ProbeContext
from exp_scripts._probe_common import _load_episode_candidates
from exp_scripts._probe_common import latent_rmse
from exp_scripts._probe_common import metric_mae
from exp_scripts._probe_common import metric_rmse
from exp_scripts._probe_common import save_tau_result
from openpi.shared import nnx_utils

DEFAULT_CHECKPOINT = "checkpoints/pi05_franka_xhand_flower_streaming_v2/pi05_franka_xhand_flower_streaming_mask/19999"
DEFAULT_CONFIG_NAME = "pi05_franka_xhand_flower_streaming_v2"
DEFAULT_DATASET = "/public/node01/users/lvrui/datasets/lerobot/flower_xhand_franka"
DEFAULT_OUTPUT_DIR = "results/streaming_transition_probe"

METRIC_COLUMNS = (
    "pair_id",
    "index_seed",
    "index_b",
    "episode_index",
    "frame_seed",
    "frame_b",
    "noise_id",
    "frame_gap",
    "history_advance_count",
    "rmse_ab",
    "mae_ab",
    "rmse_b_seed",
    "mae_b_seed",
    "rmse_gt_b",
    "mae_gt_b",
    "latent_rmse_before_ab_gt",
    "latent_rmse_before_ab_b_seed",
    "latent_rmse_after_ab_gt",
    "latent_rmse_after_ab_b_seed",
    "output_rmse_ab_b_seed",
    "output_rmse_ab_gt_b",
)


@dataclasses.dataclass(frozen=True)
class StreamingRollout:
    window_before_b: jax.Array
    window_after_b: jax.Array
    window_b_seed: jax.Array
    window_gt_before_b: jax.Array
    window_gt_after_b: jax.Array
    action_ab: jax.Array
    action_b_seed: jax.Array
    action_gt_b: jax.Array


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
        default=None,
        help="Observation spacing in frames. Defaults to streaming_chunk_size.",
    )
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save full streaming windows in the result npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed streaming.npz in the output directory.",
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


def _mean_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summary(rows: list[dict[str, Any]], history_advance_count: int, frame_gap: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_gap": frame_gap,
        "history_advance_count": history_advance_count,
        "count": len(rows),
    }
    for name in (
        "rmse_ab",
        "mae_ab",
        "rmse_b_seed",
        "mae_b_seed",
        "rmse_gt_b",
        "mae_gt_b",
        "latent_rmse_before_ab_gt",
        "latent_rmse_before_ab_b_seed",
        "latent_rmse_after_ab_gt",
        "latent_rmse_after_ab_b_seed",
        "output_rmse_ab_b_seed",
        "output_rmse_ab_gt_b",
    ):
        result[f"{name}_mean"] = _mean_metric(rows, name)
    return result


def _folded_key(seed: int, pair_id: int, noise_id: int, stream_slot: int, step: int = 0):
    key = jax.random.key(seed)
    key = jax.random.fold_in(key, pair_id)
    key = jax.random.fold_in(key, noise_id)
    key = jax.random.fold_in(key, stream_slot)
    return jax.random.fold_in(key, step)


class StreamingRunner:
    """JAX implementation of the server's seed and one-chunk advance."""

    def __init__(self, context: ProbeContext, *, num_steps: int):
        model = context.model
        if not context.model_config.streaming:
            raise ValueError("The streaming probe requires model.streaming=True.")
        if context.action_horizon % context.chunk_size:
            raise ValueError("action_horizon must be divisible by streaming_chunk_size.")

        self.context = context
        self.num_steps = num_steps
        self.chunk_size = context.chunk_size
        self.action_horizon = context.action_horizon
        self.action_dim = context.action_dim
        self.num_chunks = self.action_horizon // self.chunk_size
        self.timestep = jnp.asarray(model.streaming_timestep(jnp.float32))
        self.sample_actions = nnx_utils.module_jit(model.sample_actions_from_prefix)
        self.advance_window = self._make_advance_window()

    def seed_window(
        self,
        *,
        rng: jax.Array,
        state: jax.Array,
        prefix_cache: dict,
        tactile_condition: Any,
        sample_noise: jax.Array,
        renoise_noise: jax.Array,
    ) -> jax.Array:
        actions = self.sample_actions(
            rng,
            state,
            prefix_cache,
            num_steps=self.num_steps,
            noise=sample_noise,
            tactile_condition=tactile_condition,
        )
        timestep = self.timestep.astype(actions.dtype)
        future_window = (
            timestep[: -self.chunk_size][None, :, None] * renoise_noise[:, : -self.chunk_size]
            + (1.0 - timestep[: -self.chunk_size][None, :, None]) * actions[:, self.chunk_size :]
        )
        return jnp.concatenate(
            [
                actions[:, : self.chunk_size],
                future_window,
                renoise_noise[:, -self.chunk_size :],
            ],
            axis=1,
        )

    def gt_window(self, ground_truth: jax.Array, renoise_noise: jax.Array) -> jax.Array:
        """Build a B-conditioned in-domain window before B's final advance."""
        timestep = self.timestep.astype(ground_truth.dtype)
        future_window = (
            timestep[: -self.chunk_size][None, :, None] * renoise_noise[:, : -self.chunk_size]
            + (1.0 - timestep[: -self.chunk_size][None, :, None]) * ground_truth[:, : -self.chunk_size]
        )
        unused_ready_chunk = jnp.zeros(
            (ground_truth.shape[0], self.chunk_size, ground_truth.shape[-1]),
            dtype=ground_truth.dtype,
        )
        return jnp.concatenate(
            [
                unused_ready_chunk,
                future_window,
                renoise_noise[:, -self.chunk_size :],
            ],
            axis=1,
        )

    def _make_advance_window(self):
        velocity_from_prefix = self.context.velocity_from_prefix
        timestep = self.timestep
        chunk_size = self.chunk_size

        def advance(
            window: jax.Array,
            state: jax.Array,
            prefix_cache: dict,
            tactile_condition: Any,
            fresh_noise: jax.Array,
        ) -> jax.Array:
            actions = window[:, chunk_size:]
            current_timestep = timestep.astype(actions.dtype)
            velocity = velocity_from_prefix(
                state,
                prefix_cache,
                actions,
                current_timestep[None, :],
                tactile_condition,
            )
            clean_actions = (
                actions[:, :chunk_size] - current_timestep[:chunk_size][None, :, None] * velocity[:, :chunk_size]
            )
            next_actions = (
                actions[:, chunk_size:]
                + (current_timestep[:-chunk_size] - current_timestep[chunk_size:])[None, :, None]
                * velocity[:, chunk_size:]
            )
            return jnp.concatenate([clean_actions, next_actions, fresh_noise], axis=1)

        return jax.jit(advance)


def _load_observation(context: ProbeContext, pair_id: int, index: int):
    return context.pair(pair_id, index, index).observation_b


def _encode_observation(context: ProbeContext, observation):
    cache = context.encode_prefix(observation)
    tactile = context.marker_condition(observation) if context.marker_condition is not None else None
    return cache, tactile


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
    target_pair = context.pair(pair_id, index_b, index_b)
    observation_b = target_pair.observation_b
    cache_b, tactile_b = _encode_observation(context, observation_b)
    history_advance_count = runner.num_chunks - 1
    index_seed = index_b - history_advance_count * frame_gap

    sample_key = _folded_key(seed, pair_id, noise_id, stream_slot=0)
    renoise_key = _folded_key(seed, pair_id, noise_id, stream_slot=1)
    sample_noise = jax.random.normal(
        sample_key,
        (1, runner.action_horizon, runner.action_dim),
        dtype=jnp.float32,
    )
    renoise_noise = jax.random.normal(
        renoise_key,
        (1, runner.action_horizon, runner.action_dim),
        dtype=jnp.float32,
    )

    seed_observation = _load_observation(context, pair_id, index_seed)
    cache_seed, tactile_seed = _encode_observation(context, seed_observation)
    window = runner.seed_window(
        rng=sample_key,
        state=seed_observation.state,
        prefix_cache=cache_seed,
        tactile_condition=tactile_seed,
        sample_noise=sample_noise,
        renoise_noise=renoise_noise,
    )
    del seed_observation, cache_seed, tactile_seed

    window_before_b = window
    fresh_noise_b = None
    for advance_index in range(1, history_advance_count + 1):
        current_index = index_seed + advance_index * frame_gap
        if current_index == index_b:
            observation = observation_b
            cache = cache_b
            tactile = tactile_b
        else:
            observation = _load_observation(context, pair_id, current_index)
            cache, tactile = _encode_observation(context, observation)

        fresh_key = _folded_key(
            seed,
            pair_id,
            noise_id,
            stream_slot=2,
            step=advance_index,
        )
        fresh_noise = jax.random.normal(
            fresh_key,
            (1, runner.chunk_size, runner.action_dim),
            dtype=jnp.float32,
        )
        if advance_index == history_advance_count:
            window_before_b = window
            fresh_noise_b = fresh_noise
        window = runner.advance_window(
            window,
            observation.state,
            cache,
            tactile,
            fresh_noise,
        )
        if current_index != index_b:
            del observation, cache, tactile

    window_after_b = window
    window_b_seed = runner.seed_window(
        rng=sample_key,
        state=observation_b.state,
        prefix_cache=cache_b,
        tactile_condition=tactile_b,
        sample_noise=sample_noise,
        renoise_noise=renoise_noise,
    )

    if history_advance_count == 0:
        window_before_b = window_b_seed
        window_after_b = window_b_seed
        window_gt_before_b = window_b_seed
        window_gt_after_b = window_b_seed
        action_gt_b = window_b_seed[:, : runner.chunk_size]
    else:
        if fresh_noise_b is None:
            raise RuntimeError("The final B advance did not produce fresh noise.")
        window_gt_before_b = runner.gt_window(
            target_pair.ground_truth_b,
            renoise_noise,
        )
        window_gt_after_b = runner.advance_window(
            window_gt_before_b,
            observation_b.state,
            cache_b,
            tactile_b,
            fresh_noise_b,
        )
        action_gt_b = window_gt_after_b[:, : runner.chunk_size]

    action_ab = window_after_b[:, : runner.chunk_size]
    action_b_seed = window_b_seed[:, : runner.chunk_size]
    ground_truth = context.to_output_actions(
        target_pair.ground_truth_b,
        observation_b.state,
    )
    output_ab = context.to_output_actions(action_ab, observation_b.state)
    output_b_seed = context.to_output_actions(action_b_seed, observation_b.state)
    output_gt_b = context.to_output_actions(action_gt_b, observation_b.state)
    target_chunk = ground_truth[: runner.chunk_size]

    row: dict[str, Any] = {
        "pair_id": pair_id,
        "index_seed": index_seed,
        "index_b": target_pair.index_b,
        "episode_index": target_pair.episode_index,
        "frame_seed": target_pair.frame_b - history_advance_count * frame_gap,
        "frame_b": target_pair.frame_b,
        "noise_id": noise_id,
        "frame_gap": frame_gap,
        "history_advance_count": history_advance_count,
        "rmse_ab": metric_rmse(output_ab, target_chunk),
        "mae_ab": metric_mae(output_ab, target_chunk),
        "rmse_b_seed": metric_rmse(output_b_seed, target_chunk),
        "mae_b_seed": metric_mae(output_b_seed, target_chunk),
        "rmse_gt_b": metric_rmse(output_gt_b, target_chunk),
        "mae_gt_b": metric_mae(output_gt_b, target_chunk),
        "latent_rmse_before_ab_gt": latent_rmse(
            window_before_b[:, runner.chunk_size :],
            window_gt_before_b[:, runner.chunk_size :],
        ),
        "latent_rmse_before_ab_b_seed": latent_rmse(
            window_before_b[:, runner.chunk_size :],
            window_b_seed[:, runner.chunk_size :],
        ),
        "latent_rmse_after_ab_gt": latent_rmse(
            window_after_b[:, runner.chunk_size :],
            window_gt_after_b[:, runner.chunk_size :],
        ),
        "latent_rmse_after_ab_b_seed": latent_rmse(
            window_after_b[:, runner.chunk_size :],
            window_b_seed[:, runner.chunk_size :],
        ),
        "output_rmse_ab_b_seed": metric_rmse(output_ab, output_b_seed),
        "output_rmse_ab_gt_b": metric_rmse(output_ab, output_gt_b),
    }

    arrays = {
        "observation_indices": np.asarray(
            [index_seed + index * frame_gap for index in range(history_advance_count + 1)],
            dtype=np.int64,
        ),
        "ground_truth_action": target_chunk,
        "action_ab": output_ab,
        "action_b_seed": output_b_seed,
        "action_gt_b": output_gt_b,
    }
    if save_traces:
        arrays.update(
            {
                "window_before_b": np.asarray(window_before_b[0]),
                "window_after_b": np.asarray(window_after_b[0]),
                "window_b_seed": np.asarray(window_b_seed[0]),
                "window_gt_before_b": np.asarray(window_gt_before_b[0]),
                "window_gt_after_b": np.asarray(window_gt_after_b[0]),
            }
        )
    return row, arrays


def _stack_results(result_lists: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack(values) for name, values in result_lists.items() if values}


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive.")
    if args.num_noise_samples <= 0:
        raise ValueError("--num-noise-samples must be positive.")

    context = ProbeContext.load(args)
    runner = StreamingRunner(context, num_steps=args.num_steps)
    frame_gap = context.chunk_size if args.frame_gap is None else args.frame_gap
    if frame_gap != context.chunk_size:
        raise ValueError(
            "For direct streaming replay, frame_gap must equal "
            f"streaming_chunk_size ({context.chunk_size}); got {frame_gap}."
        )

    output_dir = _prepare_output_dir(args)
    manifest_path = output_dir / "manifest.json"
    if args.resume and (output_dir / "streaming.npz").exists() and (output_dir / "metrics.csv").exists():
        print(f"[skip] completed result exists: {output_dir / 'streaming.npz'}")
        return

    history_advance_count = runner.num_chunks - 1
    candidate_indices = _load_episode_candidates(
        context.dataset_path,
        frame_gap=history_advance_count * frame_gap,
        action_horizon=context.action_horizon,
        max_pairs=args.max_pairs,
    )
    manifest = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "arguments": {
            **vars(args),
            "frame_gap": frame_gap,
            "history_advance_count": history_advance_count,
        },
        "checkpoint": str(context.checkpoint_path),
        "dataset": str(context.dataset_path),
        "norm_stats": str(context.norm_stats_path),
        "model_config": dataclasses.asdict(context.model_config),
        "git_commit": _git_commit(),
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
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
                frame_gap=frame_gap,
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
    result_arrays["frame_gap"] = np.asarray(frame_gap, dtype=np.int64)
    result_arrays["history_advance_count"] = np.asarray(history_advance_count, dtype=np.int64)
    save_tau_result(output_dir / "streaming.npz", arrays=result_arrays)
    _write_csv(output_dir / "metrics.csv", rows)
    _write_json(
        output_dir / "summary.json",
        {"summary": _summary(rows, history_advance_count, frame_gap)},
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
