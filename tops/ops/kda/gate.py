"""KDA gate: chunk-local cumulative sum in log2 space.

KDA uses per-element gates g: [B, T, H, K] in natural log space.
This module converts to log2 space and applies chunk-local cumsum.

The conversion is: g_log2 = g / ln(2), so that exp2(cumsum(g_log2)) == exp(cumsum(g)).
The kernel then uses exp2() for all gate computations — matching FLA convention.
"""

import jax
import jax.numpy as jnp

from tops.ops.common.cumsum import chunk_local_cumsum_vector
from tops.utils import assert_shape

_LN2 = float(jnp.log(jnp.array(2.0, dtype=jnp.float64)))


def kda_gate_chunk_cumsum(
    g: jax.Array,
    chunk_size: int,
    cu_seqlens: jax.Array | None = None,
) -> jax.Array:
    """Chunk-local cumulative sum of KDA gates in log2 space.

    Converts g from natural log space to log2 space, then applies
    chunk-local cumsum. The result g_cumsum satisfies:
        exp2(g_cumsum[..., t, :]) = prod_{s=0}^{t} exp(g[..., s, :])
    within each chunk.

    Args:
        g: [B, T, H, K] — per-element gate in natural log space.
        chunk_size: chunk size for tiling.
        cu_seqlens: [N+1] cumulative sequence lengths for varlen. Optional.

    Returns:
        g_cumsum: [B, T, H, K] — chunk-local cumsum in log2 space (float32).
    """
    B, T, H, K = g.shape
    assert_shape(g, (B, T, H, K), "g")

    # Convert natural log → log2: g_log2 = g / ln(2)
    g_log2 = g.astype(jnp.float32) / _LN2

    return chunk_local_cumsum_vector(
        g_log2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=jnp.float32,
    )


def kda_gate_bwd(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    dyg: jax.Array | None = None,
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Backward pass for the KDA gate function.

    Computes gradients w.r.t. the original gate input g, the log-parameter
    A_log, and the optional dt_bias.

    Forward semantics (for reference):
      - Without lower_bound:  yg = -exp(A_log) * softplus(g + dt_bias)
      - With lower_bound:     yg = lower_bound * sigmoid(exp(A_log) * g)

    Args:
        g:           [..., H, K] — original gate input (before activation).
        A_log:       [H]         — log of the A parameter.
        dt_bias:     [H * K] or None — optional bias added to g.
        dyg:         [..., H, K] — upstream gradient w.r.t. yg.
        lower_bound: float or None — if set, use sigmoid mode.

    Returns:
        dg:    same shape as g, gradient w.r.t. g (cast to g.dtype).
        dA:    same shape as A_log [H], gradient w.r.t. A_log.
        dbias: same shape as dt_bias [H * K] or None.
    """
    H, K = g.shape[-2], g.shape[-1]
    g_f = g.astype(jnp.float32)
    dyg_f = dyg.astype(jnp.float32)

    # Apply bias if present
    if dt_bias is not None:
        g_f = g_f + dt_bias.reshape(H, K).astype(jnp.float32)

    if lower_bound is None:
        # Forward: yg = -exp(A_log) * softplus(g + bias)
        # softplus(x) = log(1 + exp(x)), d/dx softplus(x) = sigmoid(x)
        b_A = -jnp.exp(A_log.astype(jnp.float32))              # [H]
        b_yg = b_A.reshape(H, 1) * jax.nn.softplus(g_f)        # [..., H, K]
        dg_f = b_A.reshape(H, 1) * (dyg_f * jax.nn.sigmoid(g_f))  # [..., H, K]
        # dA = sum over all dims except H: sum(dyg * yg) per head
        dA_per_elem = dyg_f * b_yg                               # [..., H, K]
    else:
        # Forward: yg = lower_bound * sigmoid(exp(A_log) * g)
        b_A = jnp.exp(A_log.astype(jnp.float32))                # [H]
        b_inner = b_A.reshape(H, 1) * g_f                       # [..., H, K]
        b_sig = jax.nn.sigmoid(b_inner)
        b_dsig = b_sig * (1.0 - b_sig)
        dg_f = dyg_f * (lower_bound * b_dsig) * b_A.reshape(H, 1)  # [..., H, K]
        dA_per_elem = dg_f * g_f                                    # [..., H, K]

    # dA: reduce over all dims except H → [H]
    # FLA does sum over K first, then sum over time-block, then sum over blocks
    # Equivalent: sum over all axes except the H axis
    reduce_axes = tuple(range(len(g.shape) - 2)) + (len(g.shape) - 1,)
    dA = jnp.sum(dA_per_elem, axis=reduce_axes)  # [H]
    dA = dA.astype(A_log.dtype)

    # Cast dg back to input dtype
    dg = dg_f.astype(g.dtype)

    # dbias: sum dg over all dims except (H, K) → [H*K]
    if dt_bias is not None:
        dbias_axes = tuple(range(len(g.shape) - 2))
        dbias = jnp.sum(dg_f, axis=dbias_axes).reshape(-1)  # [H*K]
        dbias = dbias.astype(dt_bias.dtype)
    else:
        dbias = None

    return dg, dA, dbias
