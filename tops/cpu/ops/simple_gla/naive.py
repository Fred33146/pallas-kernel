"""JAX CPU reference for Simple GLA naive recurrent operations with FLA-exact dtype behavior.

Precisely matches the FLA naive recurrent Simple GLA dtype behavior:

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation:
    q, k, v, g: cast to fp32 for computation     [fp64 mode: fp64]
    h (hidden state): fp32 accumulator             [fp64 mode: fp64]
    o (output buffer): fp32 during computation     [fp64 mode: fp64]
  Final output:
    o: cast back to original input dtype           [fp64 mode: fp64]
    final_state h: fp32                            [fp64 mode: fp64]

Simple GLA vs GLA:
  - GLA uses per-element gates gk: [B, T, H, K]
  - Simple GLA uses per-head scalar gates g: [B, T, H] broadcast over K
  - Optionally combines with fixed per-head decay g_gamma: [H]
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype


@cpu_reference
def naive_simple_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray | None = None,
    g_gamma: jnp.ndarray | None = None,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Naive recurrent Simple GLA — JAX CPU reference with FLA-exact dtype behavior.

    Simple GLA uses per-head scalar gates (g: [B,T,H]) instead of per-element
    gates (gk: [B,T,H,K]) as in standard GLA. The scalar gate is broadcast
    over both K and V dimensions when updating the hidden state.

    Core recurrence (per timestep):
        gate_t = exp(g_t) [* exp(g_gamma)]       scalar per (B, H)
        h_t = h_{t-1} * gate_t + k_t^T @ v_t    gate broadcast over [K, V]
        o_t = (q_t * scale)^T @ h_t              sum over K dimension

    Dtype behavior (matching FLA):
      - All inputs cast to fp32 for computation (`.float()` in PyTorch)
      - Hidden state h is fp32 accumulator
      - Output o computed in fp32, cast back to original dtype
      - Final state h stays in fp32
      - fp64 mode: all computation in fp64, no precision cast

    Args:
        q:               [B, T, H, K] — Queries
        k:               [B, T, H, K] — Keys
        v:               [B, T, H, V] — Values
        g:               [B, T, H]    — Per-head scalar gate in log-space (e.g., logsigmoid)
                         Optional. If None and g_gamma is None, gate defaults to 1.0.
        g_gamma:         [H]          — Fixed per-head log-decay, applied multiplicatively
                         with g (i.e., combined gate = exp(g + g_gamma)).
                         Must be in acc_dtype (fp32 or fp64). Optional.
        scale:           Scalar query scale. Defaults to K ** -0.5.
        initial_state:   [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return the final hidden state.

    Returns:
        o:           [B, T, H, V] — Output (original input dtype)
        final_state: [B, H, K, V] in fp32 (or fp64), or None
    """
    orig_dtype = q.dtype
    acc_dt = _acc_dtype(orig_dtype)

    # Shape assertions (project coding standard)
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
        f"v shape {v.shape} incompatible with q shape {q.shape}"
    )
    if g is not None:
        assert g.ndim == 3 and g.shape == q.shape[:3], (
            f"g shape {g.shape} != {q.shape[:3]}"
        )
    if g_gamma is not None:
        assert g_gamma.ndim == 1 and g_gamma.shape[0] == q.shape[2], (
            f"g_gamma shape {g_gamma.shape} != (H={q.shape[2]},)"
        )

    # FLA: q, k, v = map(lambda x: x.transpose(1, 2).float(), ...)
    # Transpose [B, T, H, D] -> [B, H, T, D] and cast to accumulator dtype
    q_f, k_f, v_f = (
        jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt)
        for x in (q, k, v)
    )
    if g is not None:
        # Transpose [B, T, H] -> [B, H, T]
        g_f = jnp.transpose(g, (0, 2, 1)).astype(acc_dt)
    else:
        g_f = None

    # Cast g_gamma to acc_dt if provided
    if g_gamma is not None:
        g_gamma_f = g_gamma.astype(acc_dt)  # [H]
    else:
        g_gamma_f = None

    B, H, T, K = q_f.shape
    V = v_f.shape[-1]

    # Default scale: K ** -0.5
    if scale is None:
        scale = K ** -0.5

    # Output buffer and hidden state in acc_dt
    o = jnp.zeros_like(v_f)       # [B, H, T, V]
    h = jnp.zeros((B, H, K, V), dtype=acc_dt)
    if initial_state is not None:
        h = h + initial_state.astype(acc_dt)

    for t in range(T):
        q_t = q_f[:, :, t] * scale   # [B, H, K]
        k_t = k_f[:, :, t]            # [B, H, K]
        v_t = v_f[:, :, t]            # [B, H, V]

        # Compute gate: scalar per (B, H), broadcast to [B, H, K, V]
        if g_f is not None:
            gate = jnp.exp(g_f[:, :, t])   # [B, H]
            if g_gamma_f is not None:
                # Combine with fixed per-head decay multiplicatively
                gate = gate * jnp.exp(g_gamma_f)[None, :]  # [B, H]
        elif g_gamma_f is not None:
            gate = jnp.broadcast_to(jnp.exp(g_gamma_f)[None, :], (B, H))  # [B, H]
        else:
            gate = None

        # Outer product k_t^T @ v_t
        kv_t = k_t[..., None] * v_t[..., None, :]   # [B, H, K, V]

        if gate is not None:
            h = h * gate[:, :, None, None] + kv_t
        else:
            h = h + kv_t

        o = o.at[:, :, t].set(
            (q_t[..., None] * h).sum(
                -2,
                # No explicit precision kwarg needed; acc_dt controls precision
            )                                        # [B, H, V]
        )

    final_state = h if output_final_state else None
    # Transpose [B, H, T, V] -> [B, T, H, V] and cast back to original dtype
    return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state
