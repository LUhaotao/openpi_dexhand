import logging

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

_TACTILE_GATE_BIAS = -1.38629436112


def _tactile_gate_bias_init(key, shape, dtype=jnp.float32):
    del key
    return jnp.full(shape, _TACTILE_GATE_BIAS, dtype=dtype)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def _action_ar_mask(action_horizon: int, chunk_size: int, mode: str = "causal") -> list[bool]:
    """Make action-block boundaries for the selected inter-chunk attention mode."""
    if mode in ("mask", "bidirectional", "tactile_attention_gate"):
        return [index == 0 for index in range(action_horizon)]
    return [index % chunk_size == 0 for index in range(action_horizon)]


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "*b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "*b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = pos[..., None] * (1.0 / period * 2 * jnp.pi)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _chunk_wise_timestep(action_horizon: int, chunk_size: int, dtype=jnp.float32) -> at.Float[at.Array, "..."]:
    """Return FlashVLA's deterministic equal-width chunk schedule."""
    num_chunks, remainder = divmod(int(action_horizon), int(chunk_size))
    if remainder:
        raise ValueError("action_horizon must be divisible by chunk_size")
    chunk_time = (jnp.arange(num_chunks, dtype=jnp.float32) + 1.0) / num_chunks
    return jnp.repeat(chunk_time, chunk_size).astype(dtype)


def _sample_chunk_wise_timestep(
    rng: at.KeyArrayLike,
    batch_shape: tuple[int, ...],
    action_horizon: int,
    chunk_size: int,
    dtype=jnp.float32,
) -> at.Float[at.Array, "..."]:
    """Sample one Beta timestep per chunk from its equal-width interval."""
    num_chunks, remainder = divmod(int(action_horizon), int(chunk_size))
    if remainder:
        raise ValueError("action_horizon must be divisible by chunk_size")
    samples = jax.random.beta(rng, 1.5, 1.0, (*batch_shape, num_chunks))
    starts = jnp.arange(num_chunks, dtype=samples.dtype) / num_chunks
    chunk_time = 0.001 + 0.999 * (starts + samples / num_chunks)
    return jnp.repeat(chunk_time, chunk_size, axis=-1).astype(dtype)


def _shift_streaming_window(
    actions: _model.Actions,
    velocity: _model.Actions,
    timestep: at.Float[at.Array, "..."],
    fresh_noise: _model.Actions,
) -> _model.Actions:
    """Denoise the next chunk and roll the future action window."""
    chunk_size = fresh_noise.shape[1]
    clean_actions = actions[:, :chunk_size] - timestep[:chunk_size][None, :, None] * velocity[:, :chunk_size]
    next_actions = actions[:, chunk_size:] + (
        timestep[:-chunk_size] - timestep[chunk_size:]
    )[None, :, None] * velocity[:, chunk_size:]
    return jnp.concatenate([clean_actions, next_actions, fresh_noise], axis=1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.discrete_state_input = config.discrete_state_input
        self.streaming = config.streaming
        self.streaming_chunk_size = config.streaming_chunk_size
        self.streaming_attention_mode = config.streaming_attention_mode
        self.streaming_constant_weight = config.streaming_constant_weight
        self.streaming_chunk_wise_weight = config.streaming_chunk_wise_weight
        self.streaming_token_wise_weight = config.streaming_token_wise_weight
        self.use_tactile = config.use_tactile
        self.use_tactile_adarms = config.use_tactile_adarms
        self.tactile_history_length = config.tactile_history_length
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            if not self.discrete_state_input:
                # Continuous Pi05 state uses its own projection so it can learn a
                # state-specific representation independently from noisy actions.
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if self.use_tactile:
                self.marker_mlp_in = nnx.Linear(pi0_config.TACTILE_MARKER_INPUT_DIM, 512, rngs=rngs)
                self.marker_mlp_out = nnx.Linear(512, action_expert_config.width, rngs=rngs)
                self.marker_fusion = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.streaming_attention_mode == "tactile_attention_gate":
                self.tactile_position_dim = 32
                self.tactile_tcn_width = 256
                self.tactile_tcn_in = nnx.Linear(
                    action_expert_config.width + self.tactile_position_dim, self.tactile_tcn_width, rngs=rngs
                )
                self.tactile_tcn_1 = nnx.Linear(3 * self.tactile_tcn_width, self.tactile_tcn_width, rngs=rngs)
                self.tactile_tcn_2 = nnx.Linear(3 * self.tactile_tcn_width, self.tactile_tcn_width, rngs=rngs)
                self.tactile_gate_out = nnx.Linear(
                    self.tactile_tcn_width,
                    1,
                    kernel_init=nn.initializers.zeros,
                    bias_init=_tactile_gate_bias_init,
                    rngs=rngs,
                )
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _marker_condition(self, obs: _model.Observation) -> at.Float[at.Array, "b emb"] | None:
        if not self.use_tactile or not self.use_tactile_adarms:
            return None
        if obs.tactile_left_marker is None or obs.tactile_right_marker is None:
            raise ValueError("use_tactile=True requires both tactile marker fields")

        left_embedding = self._encode_marker_frames(obs.tactile_left_marker)
        right_embedding = self._encode_marker_frames(obs.tactile_right_marker)
        return self.marker_fusion(jnp.concatenate([left_embedding, right_embedding], axis=-1))

    def _encode_marker_frames(self, marker: at.Float[at.Array, "*b 2 63 2"]) -> at.Float[at.Array, "*b emb"]:
        marker = marker / jnp.asarray((320.0, 240.0), dtype=marker.dtype)
        marker = marker.reshape((*marker.shape[:-3], -1))
        marker = self.marker_mlp_in(marker)
        marker = jax.nn.gelu(marker)
        return self.marker_mlp_out(marker)

    def _tactile_attention_log_gates(self, obs: _model.Observation) -> at.Float[at.Array, "b k"] | None:
        if self.streaming_attention_mode != "tactile_attention_gate":
            return None
        return self._tactile_attention_log_gates_from_history(
            obs.tactile_left_marker_history, obs.tactile_right_marker_history
        )

    def _tactile_attention_log_gates_from_history(
        self,
        left_history: at.Float[at.Array, "b th 2 63 2"] | None,
        right_history: at.Float[at.Array, "b th 2 63 2"] | None,
    ) -> at.Float[at.Array, "b k"] | None:
        if self.streaming_attention_mode != "tactile_attention_gate":
            return None
        if left_history is None or right_history is None:
            raise ValueError("tactile_attention_gate requires left and right marker histories")
        if (
            left_history.shape[1] != self.tactile_history_length
            or right_history.shape[1] != self.tactile_history_length
        ):
            raise ValueError(
                f"Expected tactile history length {self.tactile_history_length}, "
                f"got left={left_history.shape[1]}, right={right_history.shape[1]}"
            )

        left = self._encode_marker_frames(left_history)
        right = self._encode_marker_frames(right_history)
        tactile = self.marker_fusion(jnp.concatenate([left, right], axis=-1))
        batch_size, history_length, _ = tactile.shape
        chunk_count = self.action_horizon // self.streaming_chunk_size
        chunk_positions = (jnp.arange(chunk_count, dtype=jnp.float32) + 0.5) / chunk_count
        position = posemb_sincos(chunk_positions, self.tactile_position_dim, min_period=0.01, max_period=1.0)
        position = jnp.broadcast_to(
            position[None, :, None, :], (batch_size, chunk_count, history_length, self.tactile_position_dim)
        )
        tactile = jnp.broadcast_to(tactile[:, None, :, :], (batch_size, chunk_count, history_length, tactile.shape[-1]))
        x = jnp.concatenate([tactile, position], axis=-1).reshape(
            batch_size * chunk_count, history_length, tactile.shape[-1] + self.tactile_position_dim
        )
        x = jax.nn.gelu(self.tactile_tcn_in(x))
        x = jax.nn.gelu(x + self.tactile_tcn_1(self._causal_conv_inputs(x, dilation=1)))
        x = jax.nn.gelu(x + self.tactile_tcn_2(self._causal_conv_inputs(x, dilation=2)))
        logits = self.tactile_gate_out(x[:, -1, :]).reshape(batch_size, chunk_count)
        return jax.nn.log_sigmoid(logits)

    @staticmethod
    def _causal_conv_inputs(x: at.Float[at.Array, "b t d"], *, dilation: int) -> at.Float[at.Array, "b t d3"]:
        length = x.shape[1]
        taps = []
        for lag in (2 * dilation, dilation, 0):
            if lag == 0:
                taps.append(x)
            elif lag >= length:
                taps.append(jnp.pad(x[:, :0], ((0, 0), (length, 0), (0, 0))))
            else:
                taps.append(jnp.pad(x[:, :-lag], ((0, 0), (lag, 0), (0, 0))))
        return jnp.concatenate(taps, axis=-1)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b ..."],
        tactile_condition: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ..."] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05 or not self.discrete_state_input:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        action_horizon = noisy_actions.shape[1]
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            if tactile_condition is None:
                tactile_condition = self._marker_condition(obs)
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            if tactile_condition is not None:
                time_emb = time_emb + (
                    tactile_condition[:, None, :] if time_emb.ndim == 3 else tactile_condition
                )
            action_expert_tokens = action_tokens
            if not self.discrete_state_input and timestep.ndim > 1:
                # The state token is a static condition, so use the clean flow
                # timestep while action tokens retain their token-wise times.
                clean_timestep = jnp.zeros((timestep.shape[0],), dtype=timestep.dtype)
                clean_time_emb = posemb_sincos(
                    clean_timestep,
                    self.action_in_proj.out_features,
                    min_period=4e-3,
                    max_period=4.0,
                )
                clean_time_emb = self.time_mlp_in(clean_time_emb)
                clean_time_emb = nnx.swish(clean_time_emb)
                clean_time_emb = self.time_mlp_out(clean_time_emb)
                clean_time_emb = nnx.swish(clean_time_emb)
                if tactile_condition is not None:
                    clean_time_emb = clean_time_emb + tactile_condition
                adarms_cond = jnp.concatenate([clean_time_emb[:, None, :], time_emb], axis=1)
            else:
                adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = time_emb if self.streaming else einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # Streaming action chunks are bidirectional internally; inter-chunk mode is configurable.
        chunk_size = self.streaming_chunk_size if self.streaming else action_horizon
        ar_mask += _action_ar_mask(action_horizon, chunk_size, self.streaming_attention_mode)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        (
            preprocess_rng,
            noise_rng,
            constant_time_rng,
            chunk_time_rng,
            token_time_rng,
            regime_rng,
        ) = jax.random.split(rng, 6)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        tactile_log_gates = self._tactile_attention_log_gates(observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if self.streaming:
            total_weight = (
                self.streaming_constant_weight
                + self.streaming_chunk_wise_weight
                + self.streaming_token_wise_weight
            )
            regime = jax.random.uniform(regime_rng) * total_weight
            constant_time = jax.random.beta(constant_time_rng, 1.5, 1.0, batch_shape) * 0.999 + 0.001
            constant_time = jnp.broadcast_to(constant_time[..., None], (*batch_shape, self.action_horizon))
            chunk_time = _sample_chunk_wise_timestep(
                chunk_time_rng,
                batch_shape,
                self.action_horizon,
                self.streaming_chunk_size,
                dtype=actions.dtype,
            )
            token_time = jax.random.beta(
                token_time_rng, 1.5, 1.0, (*batch_shape, self.action_horizon)
            ) * 0.999 + 0.001
            time = jnp.where(
                regime < self.streaming_constant_weight,
                constant_time,
                jnp.where(
                    regime < self.streaming_constant_weight + self.streaming_chunk_wise_weight,
                    chunk_time,
                    token_time,
                ),
            ).astype(actions.dtype)
        else:
            time = jax.random.beta(constant_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None] if self.streaming else time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = self._mask_action_chunks(
            attn_mask, prefix_tokens.shape[1] + suffix_tokens.shape[1] - self.action_horizon
        )
        action_start = prefix_tokens.shape[1] + suffix_tokens.shape[1] - self.action_horizon
        attn_bias = self._make_action_attention_bias(
            tactile_log_gates,
            query_length=attn_mask.shape[1],
            key_length=attn_mask.shape[2],
            action_query_start=action_start,
            action_key_start=action_start,
        )
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            attn_bias=attn_bias,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _mask_action_chunks(self, attn_mask, action_start: int):
        if self.streaming_attention_mode != "mask":
            return attn_mask
        chunk_size = self.streaming_chunk_size if self.streaming else self.action_horizon
        chunk_ids = jnp.arange(self.action_horizon) // chunk_size
        same_chunk = chunk_ids[:, None] == chunk_ids[None, :]
        action_end = action_start + self.action_horizon
        return attn_mask.at[:, action_start:action_end, action_start:action_end].set(
            attn_mask[:, action_start:action_end, action_start:action_end] & same_chunk[None]
        )

    def _make_action_attention_bias(
        self,
        log_gates: at.Float[at.Array, "b k"] | None,
        *,
        query_length: int,
        key_length: int,
        action_query_start: int,
        action_key_start: int,
    ) -> at.Float[at.Array, "b q s"] | None:
        if log_gates is None:
            return None
        chunk_size = self.streaming_chunk_size
        chunk_ids = jnp.arange(self.action_horizon) // chunk_size
        same_chunk = chunk_ids[:, None] == chunk_ids[None, :]
        query_gate = log_gates[:, chunk_ids]
        action_bias = jnp.where(same_chunk[None], 0.0, query_gate[:, :, None])
        bias = jnp.zeros((log_gates.shape[0], query_length, key_length), dtype=jnp.float32)
        return bias.at[
            :,
            action_query_start : action_query_start + self.action_horizon,
            action_key_start : action_key_start + self.action_horizon,
        ].set(action_bias)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        tactile_condition = self._marker_condition(observation)
        tactile_log_gates = self._tactile_attention_log_gates(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), tactile_condition
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            suffix_attn_mask = self._mask_action_chunks(
                suffix_attn_mask, suffix_tokens.shape[1] - self.action_horizon
            )
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            action_query_start = suffix_tokens.shape[1] - self.action_horizon
            full_attn_bias = self._make_action_attention_bias(
                tactile_log_gates,
                query_length=full_attn_mask.shape[1],
                key_length=full_attn_mask.shape[2],
                action_query_start=action_query_start,
                action_key_start=prefix_tokens.shape[1] + action_query_start,
            )
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                attn_bias=full_attn_bias,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def encode_prefix(self, observation: _model.Observation) -> dict:
        """Encode image/language prefix and return its KV cache for another process."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return {"kv_cache": kv_cache, "prefix_pad_mask": prefix_mask}

    @at.typecheck
    def sample_actions_from_prefix(
        self,
        rng: at.KeyArrayLike,
        state: at.Float[at.Array, "b s"],
        prefix_cache: dict,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        tactile_condition: at.Float[at.Array, "b emb"] | None = None,
        tactile_left_marker_history: at.Float[at.Array, "b th 2 63 2"] | None = None,
        tactile_right_marker_history: at.Float[at.Array, "b th 2 63 2"] | None = None,
    ) -> _model.Actions:
        """Run flow matching using a prefix KV cache produced by ``encode_prefix``."""
        if getattr(self, "use_tactile_adarms", False) and tactile_condition is None:
            raise ValueError("use_tactile=True requires tactile_condition for prefix sampling")
        tactile_log_gates = (
            self._tactile_attention_log_gates_from_history(tactile_left_marker_history, tactile_right_marker_history)
            if self.streaming_attention_mode == "tactile_attention_gate"
            else None
        )
        batch_size = state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            v_t = self._velocity_from_prefix(
                state,
                prefix_cache,
                x_t,
                jnp.broadcast_to(time, batch_size),
                tactile_condition,
                tactile_log_gates,
            )
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0)))
        return actions

    def streaming_timestep(self, dtype=jnp.float32) -> at.Float[at.Array, "..."]:
        return _chunk_wise_timestep(self.action_horizon, self.streaming_chunk_size, dtype=dtype)

    def advance_streaming_actions_from_prefix(
        self,
        rng: at.KeyArrayLike,
        state: at.Float[at.Array, "b s"],
        prefix_cache: dict,
        action_window: _model.Actions,
        advance: at.Int[at.Array, ""],
        tactile_condition: at.Float[at.Array, "b emb"] | None = None,
        tactile_left_marker_history: at.Float[at.Array, "b th 2 63 2"] | None = None,
        tactile_right_marker_history: at.Float[at.Array, "b th 2 63 2"] | None = None,
    ) -> _model.Actions:
        """Advance a token-wise diffusion-forcing window by completed action chunks."""
        if not self.streaming:
            raise ValueError("Streaming inference requires a model trained with streaming=True")
        if getattr(self, "use_tactile_adarms", False) and tactile_condition is None:
            raise ValueError("use_tactile=True requires tactile_condition for streaming")
        tactile_log_gates = (
            self._tactile_attention_log_gates_from_history(tactile_left_marker_history, tactile_right_marker_history)
            if getattr(self, "streaming_attention_mode", None) == "tactile_attention_gate"
            else None
        )

        timestep = self.streaming_timestep(action_window.dtype)
        chunk_size = self.streaming_chunk_size

        def step(_, carry):
            step_rng, action_buffer = carry
            step_rng, noise_rng = jax.random.split(step_rng)
            actions = action_buffer[:, chunk_size:]
            if tactile_condition is None and tactile_log_gates is None:
                velocity = self._velocity_from_prefix(state, prefix_cache, actions, timestep[None, :])
            else:
                velocity = self._velocity_from_prefix(
                    state,
                    prefix_cache,
                    actions,
                    timestep[None, :],
                    tactile_condition,
                    tactile_log_gates,
                )
            fresh_noise = jax.random.normal(
                noise_rng,
                (actions.shape[0], chunk_size, self.action_dim),
                dtype=actions.dtype,
            )
            return step_rng, _shift_streaming_window(actions, velocity, timestep, fresh_noise)

        _, action_window = jax.lax.fori_loop(0, advance, step, (rng, action_window))
        return action_window

    def _velocity_from_prefix(
        self,
        state: at.Float[at.Array, "b s"],
        prefix_cache: dict,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b ..."],
        tactile_condition: at.Float[at.Array, "b emb"] | None = None,
        tactile_log_gates: at.Float[at.Array, "b k"] | None = None,
    ) -> _model.Actions:
        """Predict flow velocity for suffix tokens against an encoded prefix."""
        kv_cache = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.bfloat16), prefix_cache["kv_cache"]
        )
        prefix_pad_masks = jnp.asarray(prefix_cache["prefix_pad_mask"], dtype=jnp.bool_)
        suffix_observation = _model.Observation(images={}, image_masks={}, state=state)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            suffix_observation, noisy_actions, timestep, tactile_condition
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        suffix_attn_mask = self._mask_action_chunks(suffix_attn_mask, suffix_tokens.shape[1] - self.action_horizon)
        prefix_attn_mask = einops.repeat(prefix_pad_masks, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        action_query_start = suffix_tokens.shape[1] - self.action_horizon
        full_attn_bias = self._make_action_attention_bias(
            tactile_log_gates,
            query_length=full_attn_mask.shape[1],
            key_length=full_attn_mask.shape[2],
            action_query_start=action_query_start,
            action_key_start=prefix_pad_masks.shape[1] + action_query_start,
        )
        positions = jnp.sum(prefix_pad_masks, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            attn_bias=full_attn_bias,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -noisy_actions.shape[1] :])
