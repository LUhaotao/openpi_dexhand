"""Shared utilities for white-box JAX observation-transition probes.

These helpers intentionally run the same JAX model primitives used by the
policy server while exposing intermediate flow-matching states to the
experiment driver.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class ProbePair:
    """One previous/current observation pair after model preprocessing."""

    pair_id: int
    index_a: int
    index_b: int
    episode_index: int
    frame_a: int
    frame_b: int
    observation_a: _model.Observation
    observation_b: _model.Observation
    ground_truth_b: jax.Array


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    """Intermediate and final states for one tau value."""

    switch_ab: jax.Array
    final_ab: jax.Array
    switch_b: jax.Array
    final_b: jax.Array
    switch_gt_b: jax.Array | None
    final_gt_b: jax.Array | None


def add_probe_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_checkpoint: str,
    default_config_name: str,
    default_dataset: str,
    default_output_dir: str,
) -> None:
    """Add arguments shared by both probe modes."""
    parser.add_argument("--checkpoint", default=default_checkpoint)
    parser.add_argument("--config-name", default=default_config_name)
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--tau-steps",
        type=parse_int_list,
        default=None,
        help="Comma-separated denoising step indices. Defaults to 0..num_steps-1.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=None,
        help="Gap between A and B in dataset frames. Defaults to streaming_chunk_size.",
    )
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save model-space actions and switch-point latents in each tau file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list for argparse."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("The integer list must not be empty.")
    return values


def resolve_tau_steps(tau_steps: list[int] | None, num_steps: int) -> list[int]:
    """Resolve tau values, excluding the clean endpoint by default."""
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    resolved = list(range(num_steps)) if tau_steps is None else tau_steps
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate tau values are not allowed: {resolved}")
    if any(tau < 0 or tau > num_steps for tau in resolved):
        raise ValueError(f"tau_steps must be in [0, num_steps]; got {resolved} with num_steps={num_steps}")
    return resolved


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/JAX/Torch-like leaves to NumPy without importing PyTorch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _batch_array(value: Any) -> jax.Array:
    """Convert one unbatched transformed value into a JAX batch."""
    array = _as_numpy(value)
    if np.issubdtype(array.dtype, np.floating):
        return jnp.asarray(array, dtype=jnp.float32)[None, ...]
    return jnp.asarray(array)[None, ...]


def observation_from_transformed(data: dict[str, Any]) -> _model.Observation:
    """Build a model Observation from a transformed, unbatched sample."""
    observation_data: dict[str, Any] = {
        "image": {key: _batch_array(value) for key, value in data["image"].items()},
        "image_mask": {key: _batch_array(value) for key, value in data["image_mask"].items()},
        "state": _batch_array(data["state"]),
    }
    for key in (
        "tactile_left_marker",
        "tactile_right_marker",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
    ):
        if key in data:
            observation_data[key] = _batch_array(data[key])
    return _model.Observation.from_dict(observation_data)


def _resolve_norm_stats_path(
    requested_path: str | None,
    dataset_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
) -> pathlib.Path:
    """Resolve norm_stats.json from an explicit path, dataset, or checkpoint."""
    candidates: list[pathlib.Path] = []
    if requested_path is not None:
        requested = pathlib.Path(requested_path).expanduser()
        candidates.append(requested / "norm_stats.json" if requested.is_dir() else requested)
    candidates.append(dataset_path / "norm_stats.json")
    assets_dir = checkpoint_path / "assets"
    if assets_dir.exists():
        candidates.extend(sorted(assets_dir.rglob("norm_stats.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find norm_stats.json. Checked:\n{formatted}")


def _load_episode_candidates(
    dataset_path: pathlib.Path,
    *,
    frame_gap: int,
    action_horizon: int,
    max_pairs: int,
) -> list[int]:
    """Select globally indexed B frames away from episode and action boundaries."""
    if frame_gap <= 0:
        raise ValueError("frame_gap must be positive.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Expected LeRobot episode metadata at {episodes_path}")

    candidates: list[int] = []
    global_start = 0
    for line in episodes_path.read_text().splitlines():
        episode = json.loads(line)
        length = int(episode["length"])
        first_b_frame = frame_gap
        last_b_frame = length - action_horizon
        if first_b_frame <= last_b_frame:
            candidates.extend(global_start + frame for frame in range(first_b_frame, last_b_frame + 1))
        global_start += length

    if not candidates:
        raise ValueError(f"No valid A/B pairs for frame_gap={frame_gap} and action_horizon={action_horizon}.")
    count = min(max_pairs, len(candidates))
    selected_positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[int(position)] for position in selected_positions]


def streaming_history_offset(
    *,
    action_horizon: int,
    chunk_size: int,
    frame_gap: int,
) -> int:
    """Return the history required by the streaming waterline for one B anchor.

    This is used only to choose common B anchors between scalar and streaming
    probes. The scalar probe still uses ``frame_gap`` as its A-to-B interval.
    """
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if frame_gap <= 0:
        raise ValueError("frame_gap must be positive.")
    if action_horizon % chunk_size:
        raise ValueError("action_horizon must be divisible by chunk_size.")
    if frame_gap % chunk_size:
        raise ValueError(
            f"frame_gap ({frame_gap}) must be a multiple of chunk_size ({chunk_size}) "
            "when selecting common streaming anchors."
        )

    num_chunks = action_horizon // chunk_size
    refresh_advance_count = frame_gap // chunk_size
    old_intervals = math.ceil(num_chunks / refresh_advance_count)
    return old_intervals * frame_gap


class ProbeContext:
    """Loads the model and data pipeline used by a JAX probe run."""

    def __init__(
        self,
        *,
        model_config: _pi0_config.Pi0Config,
        data_config: _config.DataConfig,
        dataset: Any,
        input_transform: Any,
        output_transform: Any,
        model: Any,
        dataset_path: pathlib.Path,
        checkpoint_path: pathlib.Path,
        norm_stats_path: pathlib.Path,
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.dataset = dataset
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.model = model
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.norm_stats_path = norm_stats_path
        self.encode_prefix = nnx_utils.module_jit(model.encode_prefix)
        self.velocity_from_prefix = nnx_utils.module_jit(model._velocity_from_prefix)  # noqa: SLF001
        self.marker_condition = (
            nnx_utils.module_jit(model._marker_condition)  # noqa: SLF001
            if model_config.use_tactile
            else None
        )
        self._scalar_switch_rollout = None
        self._gt_rollout = None

    @classmethod
    def load(cls, args: argparse.Namespace) -> ProbeContext:
        """Load model, checkpoint assets, and a transformed LeRobot dataset."""
        checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

        base_config = _config.get_config(args.config_name)
        snapshot = _pi0_config.load_snapshot(checkpoint_path / "model_config")
        model_config = snapshot or base_config.model
        if not isinstance(model_config, _pi0_config.Pi0Config):
            raise TypeError("The JAX observation-transition probe requires a Pi0Config model.")

        dataset_path = pathlib.Path(args.dataset).expanduser().resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

        data_config = base_config.data.create(base_config.assets_dirs, model_config)
        norm_stats_path = _resolve_norm_stats_path(args.norm_stats, dataset_path, checkpoint_path)
        norm_stats = _normalize.load(norm_stats_path.parent)
        data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

        metadata = LeRobotDatasetMetadata(str(dataset_path))
        delta_timestamps = {
            key: [index / metadata.fps for index in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        }
        dataset = LeRobotDataset(str(dataset_path), delta_timestamps=delta_timestamps)
        if data_config.prompt_from_task:
            dataset = _data_loader.TransformedDataset(
                dataset,
                [_transforms.PromptFromLeRobotTask(metadata.tasks)],
            )

        input_transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                _transforms.InjectDefaultPrompt(args.default_prompt),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        output_transform = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )

        params = _model.restore_params(checkpoint_path / "params", dtype=jnp.bfloat16)
        model = model_config.load(params)

        return cls(
            model_config=model_config,
            data_config=data_config,
            dataset=dataset,
            input_transform=input_transform,
            output_transform=output_transform,
            model=model,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            norm_stats_path=norm_stats_path,
        )

    @property
    def action_dim(self) -> int:
        return self.model_config.action_dim

    @property
    def action_horizon(self) -> int:
        return self.model_config.action_horizon

    @property
    def chunk_size(self) -> int:
        return min(
            int(self.model_config.streaming_chunk_size),
            self.model_config.action_horizon,
        )

    def pair(self, pair_id: int, index_a: int, index_b: int) -> ProbePair:
        """Load and transform one A/B pair plus B's GT action chunk."""
        raw_a = self.dataset[index_a]
        raw_b = self.dataset[index_b]
        episode_a = int(_as_numpy(raw_a["episode_index"]))
        episode_b = int(_as_numpy(raw_b["episode_index"]))
        frame_a = int(_as_numpy(raw_a["frame_index"]))
        frame_b = int(_as_numpy(raw_b["frame_index"]))
        if episode_a != episode_b or frame_b - frame_a != index_b - index_a:
            raise ValueError(
                f"Invalid pair ({index_a}, {index_b}): "
                f"episodes=({episode_a}, {episode_b}), frames=({frame_a}, {frame_b})"
            )
        if "action_is_pad" in raw_b and np.any(_as_numpy(raw_b["action_is_pad"])):
            raise ValueError(f"B frame {index_b} has padded GT actions.")

        transformed_a = self.input_transform(raw_a)
        transformed_b = self.input_transform(raw_b)
        if "actions" not in transformed_b:
            raise ValueError("The transformed B sample does not contain GT actions.")
        return ProbePair(
            pair_id=pair_id,
            index_a=index_a,
            index_b=index_b,
            episode_index=episode_b,
            frame_a=frame_a,
            frame_b=frame_b,
            observation_a=observation_from_transformed(transformed_a),
            observation_b=observation_from_transformed(transformed_b),
            ground_truth_b=jnp.asarray(
                _batch_array(transformed_b["actions"]),
                dtype=jnp.float32,
            ),
        )

    def candidate_indices(self, *, frame_gap: int, max_pairs: int) -> list[int]:
        return _load_episode_candidates(
            self.dataset_path,
            frame_gap=frame_gap,
            action_horizon=self.action_horizon,
            max_pairs=max_pairs,
        )

    def encode_pair(self, pair: ProbePair) -> tuple[Any, Any, Any, Any]:
        """Encode A and B prefixes and optional tactile conditions."""
        cache_a = self.encode_prefix(pair.observation_a)
        cache_b = self.encode_prefix(pair.observation_b)
        if self.marker_condition is None:
            return cache_a, cache_b, None, None
        return (
            cache_a,
            cache_b,
            self.marker_condition(pair.observation_a),
            self.marker_condition(pair.observation_b),
        )

    def configure_rollouts(self, num_steps: int) -> None:
        """Create rollout executables after the CLI chooses num_steps."""
        self._scalar_switch_rollout = _make_scalar_switch_rollout(
            self.velocity_from_prefix,
            num_steps,
        )
        self._gt_rollout = _make_gt_rollout(
            self.velocity_from_prefix,
            num_steps,
        )

    def rollout(
        self,
        pair: ProbePair,
        *,
        cache_a: dict,
        cache_b: dict,
        tactile_a: Any,
        tactile_b: Any,
        noise: jax.Array,
        tau_steps: int,
        include_gt: bool = True,
    ) -> RolloutResult:
        """Run the requested probe branches for one tau."""
        if self._scalar_switch_rollout is None or self._gt_rollout is None:
            raise RuntimeError("Call configure_rollouts before running a probe.")
        tau = jnp.asarray(tau_steps, dtype=jnp.int32)
        switch_ab, final_ab = self._scalar_switch_rollout(
            noise,
            pair.observation_a.state,
            cache_a,
            tactile_a,
            pair.observation_b.state,
            cache_b,
            tactile_b,
            tau,
        )
        switch_b, final_b = self._scalar_switch_rollout(
            noise,
            pair.observation_b.state,
            cache_b,
            tactile_b,
            pair.observation_b.state,
            cache_b,
            tactile_b,
            tau,
        )
        if include_gt:
            switch_gt_b, final_gt_b = self._gt_rollout(
                noise,
                pair.ground_truth_b,
                pair.observation_b.state,
                cache_b,
                tactile_b,
                tau,
            )
        else:
            switch_gt_b, final_gt_b = None, None
        return RolloutResult(
            switch_ab=switch_ab,
            final_ab=final_ab,
            switch_b=switch_b,
            final_b=final_b,
            switch_gt_b=switch_gt_b,
            final_gt_b=final_gt_b,
        )

    def to_output_actions(self, actions: jax.Array, state: jax.Array) -> np.ndarray:
        """Apply the same output transforms as the policy."""
        transformed = self.output_transform(
            {
                "state": _as_numpy(state[0]),
                "actions": _as_numpy(actions[0]),
            }
        )
        return np.asarray(transformed["actions"], dtype=np.float32)

    def noise(self, *, seed: int, pair_id: int, noise_id: int) -> jax.Array:
        """Generate deterministic common random numbers for paired branches."""
        key = jax.random.key(seed)
        key = jax.random.fold_in(key, pair_id)
        key = jax.random.fold_in(key, noise_id)
        return jax.random.normal(
            key,
            (1, self.action_horizon, self.action_dim),
            dtype=jnp.float32,
        )


def _make_scalar_switch_rollout(velocity, num_steps: int):
    """Create a JIT-compiled scalar-timestep A-to-B rollout."""
    dt = -1.0 / num_steps

    def rollout(
        noise: jax.Array,
        state_a: jax.Array,
        cache_a: dict,
        tactile_a: Any,
        state_b: jax.Array,
        cache_b: dict,
        tactile_b: Any,
        tau_steps: jax.Array,
    ):
        def update(
            x: jax.Array,
            state: jax.Array,
            cache: dict,
            tactile: Any,
            step_index: jax.Array,
        ) -> jax.Array:
            time = jnp.asarray(1.0, dtype=x.dtype) - (jnp.asarray(step_index, dtype=x.dtype) / num_steps)
            timestep = jnp.broadcast_to(time, (state.shape[0],))
            velocity_value = velocity(state, cache, x, timestep, tactile)
            return x + jnp.asarray(dt, dtype=x.dtype) * velocity_value

        def step(step_index: int, carry: tuple[jax.Array, jax.Array]):
            x, switch = carry
            x_next = jax.lax.cond(
                step_index < tau_steps,
                lambda value: update(value, state_a, cache_a, tactile_a, step_index),
                lambda value: update(value, state_b, cache_b, tactile_b, step_index),
                x,
            )
            switch_next = jax.lax.cond(
                step_index + 1 == tau_steps,
                lambda _: x_next,
                lambda _: switch,
                operand=None,
            )
            return x_next, switch_next

        final, switch = jax.lax.fori_loop(0, num_steps, step, (noise, noise))
        return switch, final

    return jax.jit(rollout)


def _make_gt_rollout(velocity, num_steps: int):
    """Create a JIT-compiled rollout from the GT interpolation at tau."""
    dt = -1.0 / num_steps

    def rollout(
        noise: jax.Array,
        ground_truth: jax.Array,
        state_b: jax.Array,
        cache_b: dict,
        tactile_b: Any,
        tau_steps: jax.Array,
    ):
        switch_time = jnp.asarray(1.0, dtype=noise.dtype) - (jnp.asarray(tau_steps, dtype=noise.dtype) / num_steps)
        x_switch = switch_time * noise + (1.0 - switch_time) * ground_truth

        def update(x: jax.Array, step_index: jax.Array) -> jax.Array:
            time = jnp.asarray(1.0, dtype=x.dtype) - (jnp.asarray(step_index, dtype=x.dtype) / num_steps)
            timestep = jnp.broadcast_to(time, (state_b.shape[0],))
            velocity_value = velocity(state_b, cache_b, x, timestep, tactile_b)
            return x + jnp.asarray(dt, dtype=x.dtype) * velocity_value

        def step(step_index: int, x: jax.Array) -> jax.Array:
            return jax.lax.cond(
                step_index >= tau_steps,
                lambda value: update(value, step_index),
                lambda value: value,
                x,
            )

        final = jax.lax.fori_loop(0, num_steps, step, x_switch)
        return x_switch, final

    return jax.jit(rollout)


def block_until_ready(value: Any) -> Any:
    """Synchronize JAX values before converting them to NumPy."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: block_until_ready(getattr(value, field.name)) for field in dataclasses.fields(value)},
        )
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def metric_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    return float(np.sqrt(np.mean(np.square(error))))


def metric_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def latent_rmse(first: jax.Array, second: jax.Array) -> float:
    first_np = np.asarray(first)
    second_np = np.asarray(second)
    return float(np.sqrt(np.mean(np.square(first_np - second_np))))


def save_tau_result(
    output_path: pathlib.Path,
    *,
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically save one tau result file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(output_path)
