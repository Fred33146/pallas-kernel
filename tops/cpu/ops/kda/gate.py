"""JAX CPU reference for KDA gate operations.

This module mirrors the public API exposed by ``fla.ops.kda.gate`` but keeps
the implementation simple and fully differentiable in JAX. It provides:

- raw gate activation: ``fused_kda_gate`` / ``kda_gate_fwd``
- a convenience backward helper: ``kda_gate_bwd``
- chunk-local cumulative sums over the activated gate:
  ``kda_gate_chunk_cumsum``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype, chunk_local_cumsum


def _expand_headwise_params(
    A_log: jax.Array,
    dt_bias: jax.Array | None,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array | None]:
    """Broadcast head-wise gate params to the effective head count."""
    assert A_log.ndim == 1, f"A_log must be 1D, got shape {A_log.shape}"
    base_heads = A_log.shape[0]
    assert num_heads % base_heads == 0, (
        f"Effective heads {num_heads} must be divisible by A_log heads {base_heads}"
    )
    repeat = num_heads // base_heads

    A = A_log.astype(dtype)
    if repeat > 1:
        A = jnp.repeat(A, repeat, axis=0)

    if dt_bias is None:
        return A, None

    assert dt_bias.ndim == 1, f"dt_bias must be 1D, got shape {dt_bias.shape}"
    expected_base = base_heads * head_dim
    expected_full = num_heads * head_dim
    assert dt_bias.shape[0] in (expected_base, expected_full), (
        f"dt_bias shape {dt_bias.shape} incompatible with "
        f"(base={expected_base}, expanded={expected_full})"
    )

    if dt_bias.shape[0] == expected_full:
        bias = dt_bias.reshape(num_heads, head_dim)
    else:
        bias = dt_bias.reshape(base_heads, head_dim)
        if repeat > 1:
            bias = jnp.repeat(bias, repeat, axis=0)
    return A, bias.astype(dtype).reshape(-1)


def naive_kda_gate(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    output_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Reference implementation of the standard KDA gate."""
    assert g.ndim >= 2, f"g must have at least 2 dims, got shape {g.shape}"
    H, K = g.shape[-2:]
    acc = _acc_dtype(g.dtype)
    g_f = g.astype(acc)
    A_f, bias_f = _expand_headwise_params(A_log, dt_bias, H, K, acc)
    if bias_f is not None:
        g_f = g_f + bias_f.reshape(H, K)
    out = -jnp.exp(A_f.reshape(H, 1)) * jax.nn.softplus(g_f)
    return out.astype(output_dtype)


def naive_kda_lowerbound_gate(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    lower_bound: float = -5.0,
    output_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Reference implementation of the lower-bounded KDA gate."""
    assert g.ndim >= 2, f"g must have at least 2 dims, got shape {g.shape}"
    H, K = g.shape[-2:]
    acc = _acc_dtype(g.dtype)
    g_f = g.astype(acc)
    A_f, bias_f = _expand_headwise_params(A_log, dt_bias, H, K, acc)
    if bias_f is not None:
        g_f = g_f + bias_f.reshape(H, K)
    out = lower_bound * jax.nn.sigmoid(jnp.exp(A_f.reshape(H, 1)) * g_f)
    return out.astype(output_dtype)


@cpu_reference
def kda_gate_fwd(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    lower_bound: float | None = None,
    output_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Forward helper matching the FLA gate API."""
    if lower_bound is None:
        return naive_kda_gate(g, A_log, dt_bias=dt_bias, output_dtype=output_dtype)
    return naive_kda_lowerbound_gate(
        g,
        A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_dtype=output_dtype,
    )


@cpu_reference
def kda_gate_bwd(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    dyg: jax.Array | None = None,
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Backward helper using JAX autodiff.

    This is not a custom kernel; it exists so callers can validate gate
    gradients without manually wiring ``jax.vjp`` each time.
    """
    assert dyg is not None, "dyg must be provided"
    acc = _acc_dtype(g.dtype)

    if dt_bias is None:
        def gate_only(g_in, A_in):
            return kda_gate_fwd(
                g_in,
                A_in,
                dt_bias=None,
                lower_bound=lower_bound,
                output_dtype=acc,
            )

        y, pullback = jax.vjp(gate_only, g, A_log)
        dg, dA = pullback(dyg.astype(y.dtype))
        return dg.astype(g.dtype), dA.astype(A_log.dtype), None

    def gate_with_bias(g_in, A_in, b_in):
        return kda_gate_fwd(
            g_in,
            A_in,
            dt_bias=b_in,
            lower_bound=lower_bound,
            output_dtype=acc,
        )

    y, pullback = jax.vjp(gate_with_bias, g, A_log, dt_bias)
    dg, dA, dbias = pullback(dyg.astype(y.dtype))
    return dg.astype(g.dtype), dA.astype(A_log.dtype), dbias.astype(dt_bias.dtype)


@cpu_reference
def fused_kda_gate(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    lower_bound: float | None = None,
    output_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Public gate API matching FLA."""
    return kda_gate_fwd(
        g,
        A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_dtype=output_dtype,
    )


@cpu_reference
def kda_gate_chunk_cumsum(
    g: jax.Array,
    A_log: jax.Array,
    chunk_size: int,
    scale: float | None = None,
    dt_bias: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    output_dtype: jnp.dtype = jnp.float32,
    chunk_indices: jax.Array | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> jax.Array:
    """Apply KDA gate activation then compute chunk-local cumulative sums."""
    del chunk_indices, kwargs
    gated = fused_kda_gate(
        g,
        A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_dtype=output_dtype,
    )
    g_cumsum = chunk_local_cumsum(gated, chunk_size, cu_seqlens=cu_seqlens)
    if scale is not None:
        g_cumsum = g_cumsum * scale
    return g_cumsum.astype(output_dtype)


__all__ = [
    "naive_kda_gate",
    "naive_kda_lowerbound_gate",
    "kda_gate_fwd",
    "kda_gate_bwd",
    "fused_kda_gate",
    "kda_gate_chunk_cumsum",
]
