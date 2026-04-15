# tops/ops/kda/fused_recurrent.py
"""KDA fused recurrent: step-by-step delta-rule recurrence via lax.scan.

Cannot reuse common/fused_recurrent.py because the delta rule correction
(v_t - k_t^T @ S) creates a dependency on the full state matrix S at each
step, which is not expressible in the standard GLA recurrence.

This implementation uses jax.lax.scan for efficient compilation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.utils import assert_shape, assert_shape_or_none


def fused_recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """KDA fused recurrent via lax.scan.

    Implements the delta rule recurrence step-by-step:
        S' = S * exp(g_t)
        residual = v_t - k_t^T @ S'
        S = S' + beta_t * k_t (x) residual
        o_t = scale * q_t^T @ S

    Args:
        q: [B, T, H, K] — Queries.
        k: [B, T, H, K] — Keys.
        v: [B, T, H, V] — Values.
        g: [B, T, H, K] — Per-element gate in natural log space.
        beta: [B, T, H] — Learning rate.
        scale: Attention scale. Defaults to K ** -0.5.
        initial_state: [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return final state.

    Returns:
        o: [B, T, H, V] — Output.
        final_state: [B, H, K, V] or None.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(g, (B, T, H, K), "g")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (B, H, K, V), "initial_state")

    if scale is None:
        scale = K ** -0.5

    # Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise (matching naive_kda)
    acc_dt = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    # Transpose to [B, H, T, D] for scan
    q_f = jnp.transpose(q, (0, 2, 1, 3)).astype(acc_dt) * scale
    k_f = jnp.transpose(k, (0, 2, 1, 3)).astype(acc_dt)
    v_f = jnp.transpose(v, (0, 2, 1, 3)).astype(acc_dt)
    g_f = jnp.transpose(g, (0, 2, 1, 3)).astype(acc_dt)
    beta_f = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt)  # [B, H, T]

    S0 = jnp.zeros((B, H, K, V), dtype=acc_dt)
    if initial_state is not None:
        S0 = S0 + initial_state.astype(acc_dt)

    def scan_fn(S, inputs):
        q_t, k_t, v_t, g_t, b_t = inputs
        # Decay state
        S = S * jnp.exp(g_t)[..., None]  # [B, H, K, V]
        # Delta correction
        v_pred = jnp.einsum('bhk,bhkv->bhv', k_t, S)
        residual = v_t - v_pred
        # State update
        S = S + jnp.einsum('bhk,bhv->bhkv', b_t[..., None] * k_t, residual)
        # Output
        o_t = jnp.einsum('bhk,bhkv->bhv', q_t, S)
        return S, o_t

    # Stack time-axis inputs: scan over T
    inputs = (
        jnp.moveaxis(q_f, 2, 0),   # [T, B, H, K]
        jnp.moveaxis(k_f, 2, 0),   # [T, B, H, K]
        jnp.moveaxis(v_f, 2, 0),   # [T, B, H, V]
        jnp.moveaxis(g_f, 2, 0),   # [T, B, H, K]
        jnp.moveaxis(beta_f, 2, 0),  # [T, B, H]
    )

    final_S, o_scan = jax.lax.scan(scan_fn, S0, inputs)
    # o_scan: [T, B, H, V] → [B, T, H, V]
    o = jnp.moveaxis(o_scan, 0, 1).astype(q.dtype)

    final_state = final_S if output_final_state else None
    return o, final_state
