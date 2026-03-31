# jax_gla.py
# JAX/Flax NNX reimplementation of GatedLinearAttention (fla/fla/layers/gla.py).
# All Triton/CUDA kernels replaced with pure JAX operations.
# Only depends on: jax, flax, numpy.
#
# JAX 版本的 GLA 层完整实现。
# 所有 Triton/CUDA 内核替换为纯 JAX 操作。

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================

from tops.modules.layernorm import RMSNorm
from tops.modules.fused_norm_gate import FusedRMSNormGated
from tops.modules.convolution import ShortConvolution
from tops.ops.gla import chunk_gla, fused_recurrent_gla, fused_chunk_gla

ACT2FN = {
    "swish": jax.nn.swish,
    "silu": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jax.nn.tanh,
}


def _rearrange_to_heads(x: jax.Array, head_dim: int) -> jax.Array:
    """Reshape: ... (h d) -> ... h d"""
    *leading, last = x.shape
    assert last % head_dim == 0
    num_heads = last // head_dim
    return x.reshape(*leading, num_heads, head_dim)


def _rearrange_from_heads(x: jax.Array) -> jax.Array:
    """Reshape: ... h d -> ... (h d)"""
    *leading, h, d = x.shape
    return x.reshape(*leading, h * d)


def _repeat_kv(x: jax.Array, num_groups: int, head_dim: int) -> jax.Array:
    """repeat(x, '... (h d) -> ... (h g) d', g=num_groups, d=head_dim).

    Input: ... (h d)  where h = num_kv_heads
    Output: ... (h*g) d  where total heads = num_kv_heads * num_groups
    """
    *leading, last = x.shape
    h = last // head_dim
    # Reshape to ... h d
    x = x.reshape(*leading, h, head_dim)
    # Repeat: ... h 1 d -> ... h g d
    x = jnp.repeat(x[..., None, :], num_groups, axis=-2)
    # Merge: ... h g d -> ... (h*g) d
    x = x.reshape(*leading, h * num_groups, head_dim)
    return x


# =============================================================================
# GatedLinearAttention (nnx.Module, 完整层)
# =============================================================================


class GatedLinearAttention(nnx.Module):
    r"""
    JAX/Flax NNX implementation of GatedLinearAttention.

    Gated Linear Attention Transformers with Hardware-Efficient Training
    (https://arxiv.org/abs/2312.06635)

    所有 Triton/CUDA 内核替换为纯 JAX 朴素递归实现。
    """

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        feature_map: str | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: float | None = None,
        fuse_norm: bool = True,
        layer_idx: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent", "fused_chunk"], (
            f"Not supported mode `{mode}`."
        )
        assert self.key_dim % num_heads == 0, (
            f"key dim must be divisible by num_heads of {num_heads}"
        )
        assert self.value_dim % num_heads == 0, (
            f"value dim must be divisible by num_heads of {num_heads}"
        )

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Linear projections
        self.q_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(
            hidden_size, self.key_dim_per_group, use_bias=False, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            hidden_size, self.value_dim_per_group, use_bias=False, rngs=rngs
        )
        if self.use_output_gate:
            self.g_proj = nnx.Linear(
                hidden_size, self.value_dim, use_bias=False, rngs=rngs
            )

        # Short convolutions
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
                rngs=rngs,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
                rngs=rngs,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
                rngs=rngs,
            )

        # Gate projection (low-rank)
        self.gk_proj = nnx.Sequential(
            nnx.Linear(hidden_size, gate_low_rank_dim, use_bias=False, rngs=rngs),
            nnx.Linear(
                gate_low_rank_dim, self.key_dim_per_group, use_bias=True, rngs=rngs
            ),
        )

        # Output projection
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)

        # Normalization + gating
        if gate_fn == "swish" and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormGated(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
                rngs=rngs,
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
                rngs=rngs,
            )
            self.gate_fn_act = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[jax.Array, None, Any | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]."
            )

        batch_size, q_len, _ = hidden_states.shape
        # 对于短序列用 fused_recurrent (这里都映射到同一个 naive 实现)
        mode = "fused_recurrent" if q_len <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        chunk_size = 16

        # NOTE: 不实现 attention_mask → unpad 逻辑 (JAX 不使用动态 index scatter)
        # 如果需要变长打包，直接传 cu_seqlens

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        # Reshape to multi-head format
        q = _rearrange_to_heads(q, self.head_k_dim)  # ... H K
        if self.num_kv_groups > 1:
            k = _repeat_kv(
                k, self.num_kv_groups, self.head_k_dim
            )  # ... (H*G) K → ... H K
            gk = _repeat_kv(gk, self.num_kv_groups, self.head_k_dim)
            v = _repeat_kv(v, self.num_kv_groups, self.head_v_dim)
        else:
            k = _rearrange_to_heads(k, self.head_k_dim)
            gk = _rearrange_to_heads(gk, self.head_k_dim)
            v = _rearrange_to_heads(v, self.head_v_dim)

        gk = jax.nn.log_sigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = jnp.clip(gk, a_min=self.clamp_min)

        if self.feature_map_fn is not None:
            q = self.feature_map_fn(q)
            k = self.feature_map_fn(k)

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_chunk":
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "chunk":
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv
                else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = _rearrange_to_heads(g, self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = _rearrange_from_heads(o)
            else:
                o = _rearrange_from_heads(self.g_norm(o))
                o = o * self.gate_fn_act(g)
        else:
            o = _rearrange_from_heads(self.g_norm(o))
        o = self.o_proj(o)

        return o, None, past_key_values


# =============================================================================
# Smoke tests
# =============================================================================
