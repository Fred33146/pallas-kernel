"""JAX CPU reference for Simple GLA fused recurrent with FLA-exact dtype behavior.

Precisely matches the FLA Triton fused_recurrent kernels dtype behavior for
Simple GLA, which uses per-head scalar gates instead of per-element gates.

Simple GLA gate shapes:
  g:       [B, T, H]  — Data-dependent scalar gate in log-space (per head per step)
  g_gamma: [H]        — Fixed per-head log-decay (same across B and T)
  Combined: decay_t = exp(g[b, t, h] + g_gamma[h]) broadcast to [H, K, V]

Dtype contract (matching FLA fused_recurrent for bf16/fp16/fp32; all fp64 for fp64):
  Forward:
    All inputs loaded as fp32:  q, k, v, g, g_gamma cast to fp32
    Hidden state h:             fp32 accumulator
    Output o:                   fp32 (fused_recurrent_simple_gla casts to q.dtype)
    Final state ht:             fp32
  fp64 mode: all computation in fp64, no precision casts.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tops.cpu.ops import cpu_reference


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


@cpu_reference
def fused_recurrent_simple_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray | None = None,
    g_gamma: jnp.ndarray | None = None,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Fused recurrent Simple GLA — JAX CPU reference with FLA-exact dtype behavior.

    Simple GLA recurrence (per timestep):
        gate_t = exp(g[b, t]) [* exp(g_gamma)]   scalar per (B, H)
        h_t = h_{t-1} * gate_t + k_t^T @ v_t     gate broadcast over [K, V]
        o_t = (q_t * scale)^T @ h_t               sum over K

    Internally the gate is applied as `h = h * decay[:, None, None]` where
    decay is [H], matching Simple GLA's scalar-per-head semantics.

    Dtype behavior (matching FLA Triton fused_recurrent):
      - All inputs cast to fp32 (or fp64) before computation
      - Hidden state h: fp32 accumulator
      - Output o: q.dtype (cast from fp32 accumulator)
      - Final state ht: fp32/fp64

    Args:
        q:                 [B, T, H, K] — Queries
        k:                 [B, T, H, K] — Keys
        v:                 [B, T, H, V] — Values
        g:                 [B, T, H]    — Per-head scalar log-gate (optional)
        g_gamma:           [H]          — Fixed per-head log-decay (optional)
        scale:             Scalar query scale. Defaults to K^{-0.5}.
        initial_state:     [N, H, K, V] — Initial hidden state per sequence.
                           N = B for non-varlen, N = num_segments for varlen.
        output_final_state: Whether to return the final hidden state.
        reverse:           If True, iterate time steps from T-1 to 0.
        cu_seqlens:        [N+1] cumulative sequence lengths for variable-length
                           segments. When provided, B must be 1.

    Returns:
        o:   [B, T, H, V] in q.dtype
        ht:  [N, H, K, V] in fp32/fp64, or None
    """
    orig_dtype = q.dtype
    acc_dt = _acc_dtype(orig_dtype)

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
        f"v shape {v.shape} incompatible with q"
    )
    assert g is not None or g_gamma is not None, (
        "At least one of g or g_gamma must be provided"
    )
    if g is not None:
        assert g.ndim == 3 and g.shape == (B, T, H), (
            f"g shape {g.shape} != {(B, T, H)}"
        )
    if g_gamma is not None:
        assert g_gamma.ndim == 1 and g_gamma.shape[0] == H, (
            f"g_gamma shape {g_gamma.shape} != ({H},)"
        )
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    if initial_state is not None:
        assert initial_state.shape == (N, H, K, V), (
            f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"
        )

    if scale is None:
        scale = K ** -0.5

    # Cast inputs to accumulator dtype
    q_f = q.astype(acc_dt)
    k_f = k.astype(acc_dt)
    v_f = v.astype(acc_dt)
    g_f = g.astype(acc_dt) if g is not None else None
    g_gamma_f = g_gamma.astype(acc_dt) if g_gamma is not None else None

    o = jnp.zeros((B, T, H, V), dtype=acc_dt)
    ht_list = [] if output_final_state else None

    for i_n in range(N):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
            b = 0
        else:
            bos, eos = 0, T
            b = i_n

        h = jnp.zeros((H, K, V), dtype=acc_dt)
        if initial_state is not None:
            h = h + initial_state[i_n].astype(acc_dt)

        time_range = range(eos - 1, bos - 1, -1) if reverse else range(bos, eos)

        for t in time_range:
            # Compute scalar decay per head: [H]
            if g_f is not None:
                decay = g_f[b, t]          # [H]
                if g_gamma_f is not None:
                    decay = decay + g_gamma_f  # [H] + [H]
            else:
                decay = g_gamma_f          # [H]

            # Apply gate: broadcast [H] -> [H, 1, 1] -> [H, K, V]
            h = h * jnp.exp(decay)[:, None, None]

            # Outer product: k_t^T @ v_t  ([H,K,1] * [H,1,V] = [H,K,V])
            h = h + k_f[b, t, :, :, None] * v_f[b, t, :, None, :]

            # Output: sum_k(h * q_t * scale) -> [H, V]
            o_t = jnp.sum(h * (q_f[b, t, :, :, None] * scale), axis=1)
            o = o.at[b, t].set(o_t)

        if output_final_state:
            ht_list.append(h)

    ht = jnp.stack(ht_list, axis=0) if output_final_state else None

    return o.astype(orig_dtype), ht
