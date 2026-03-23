"""JAX CPU reference for GLA naive recurrent operations with FLA-exact dtype behavior.

Precisely matches the FLA naive recurrent GLA implementation
(fla/ops/gla/naive.py) dtype behavior:

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation:
    q, k, v, gk: cast to fp32 for computation     [fp64 mode: fp64]
    h (hidden state): fp32 accumulator             [fp64 mode: fp64]
    o (output buffer): fp32 during computation     [fp64 mode: fp64]
  Final output:
    o: cast back to original input dtype           [fp64 mode: fp64]
    final_state h: fp32                            [fp64 mode: fp64]
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


@cpu_reference
def naive_recurrent_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Naive recurrent GLA — JAX CPU reference with FLA-exact dtype behavior.

    Core recurrence (per timestep):
        h_t = h_{t-1} * exp(gk_t) + k_t^T @ v_t   (outer product)
        o_t = (q_t * scale)^T @ h_t                (sum over K dimension)

    Dtype behavior (matching FLA fla/ops/gla/naive.py):
      - All inputs cast to fp32 for computation (`.float()` in PyTorch)
      - Hidden state h is fp32 accumulator
      - Output o computed in fp32, cast back to original dtype
      - Final state h stays in fp32
      - fp64 mode: all computation in fp64, no precision cast

    Args:
        q:  [B, T, H, K] — Queries
        k:  [B, T, H, K] — Keys
        v:  [B, T, H, V] — Values
        gk: [B, T, H, K] — Per-key gating in log-space (i.e., after logsigmoid)
        initial_state: [B, H, K, V] — Initial hidden state
        output_final_state: Whether to return the final hidden state

    Returns:
        o:           [B, T, H, V] — Output (original input dtype)
        final_state: [B, H, K, V] in fp32 (or fp64), or None
    """
    orig_dtype = q.dtype
    acc_dt = _acc_dtype(orig_dtype)

    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
        f"v shape {v.shape} incompatible with q shape {q.shape}"
    )
    assert gk.shape == q.shape, f"gk shape {gk.shape} != q shape {q.shape}"

    # FLA: q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), ...)
    # Transpose [B, T, H, D] -> [B, H, T, D] and cast to accumulator dtype
    q_f, k_f, v_f, gk_f = (
        jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt)
        for x in (q, k, v, gk)
    )
    B, H, T, K = q_f.shape
    V = v_f.shape[-1]

    # FLA: scale = K ** -0.5
    scale = K**-0.5

    # FLA: o = torch.zeros_like(v)  (v is already float32 after transpose)
    o = jnp.zeros_like(v_f)  # [B, H, T, V] in acc_dt
    # FLA: h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    h = jnp.zeros((B, H, K, V), dtype=acc_dt)
    if initial_state is not None:
        h = h + initial_state.astype(acc_dt)

    for t in range(T):
        q_t = q_f[:, :, t] * scale               # [B, H, K]
        k_t = k_f[:, :, t]                        # [B, H, K]
        v_t = v_f[:, :, t]                        # [B, H, V]
        gk_t = jnp.exp(gk_f[:, :, t])            # [B, H, K]
        kv_t = k_t[..., None] * v_t[..., None, :]  # [B, H, K, V]
        h = h * gk_t[..., None] + kv_t
        o = o.at[:, :, t].set(
            (q_t[..., None] * h).sum(-2)          # [B, H, V]
        )

    final_state = h if output_final_state else None
    # FLA: return o.transpose(1, 2).to(dtype), h
    return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state
