"""JAX CPU reference for KDA (Kernel Delta Attention) naive recurrent operations.

Precisely matches FLA's naive_recurrent_kda dtype behavior.

KDA implements the Delta Rule with per-element gating:
    S_t = S_{t-1} * exp(g_t) + beta_t * k_t ⊗ (v_t - k_t^T (S_{t-1} * exp(g_t)))
    o_t = (q_t * scale)^T @ S_t

Key differences from Simple GLA:
  - Gate shape: KDA uses g: [B,T,H,K] (per-element), Simple GLA uses g: [B,T,H] (per-head)
  - Delta rule: KDA has beta: [B,T,H] learning rate and subtracts k^T S from v
  - State update: KDA uses delta rule correction, not simple outer product

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation:
    q, k, v, g, beta: cast to fp32                [fp64 mode: fp64]
    h (hidden state): fp32 accumulator             [fp64 mode: fp64]
    o (output buffer): fp32 during computation     [fp64 mode: fp64]
  Final output:
    o: cast back to original input dtype           [fp64 mode: fp64]
    final_state h: fp32                            [fp64 mode: fp64]
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype


@cpu_reference
def naive_kda(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  g: jnp.ndarray,
  beta: jnp.ndarray,
  scale: float | None = None,
  initial_state: jnp.ndarray | None = None,
  output_final_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Naive recurrent KDA — JAX CPU reference with FLA-exact dtype behavior.

  KDA (Kernel Delta Attention) implements a gated delta rule where the
  state matrix S is updated by decaying the previous state and applying
  a delta correction weighted by the learning rate beta.

  Core recurrence (per timestep):
      S' = S_{t-1} * exp(g_t)                           decay
      residual = v_t - k_t^T @ S'                        prediction error
      S_t = S' + beta_t * k_t ⊗ residual                delta update
      o_t = (q_t * scale)^T @ S_t                        output

  Dtype behavior (matching FLA):
    - All inputs cast to fp32 for computation
    - Hidden state S is fp32 accumulator
    - Output o computed in fp32, cast back to original dtype
    - Final state S stays in fp32
    - fp64 mode: all computation in fp64, no precision cast

  Args:
      q:               [B, T, H, K] — Queries
      k:               [B, T, H, K] — Keys
      v:               [B, T, H, V] — Values
      g:               [B, T, H, K] — Per-element gate in log-space (e.g., -exp(A)*softplus(g))
      beta:            [B, T, H]    — Learning rate / step size for delta rule
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
  B, T, H, K = q.shape
  V = v.shape[-1]
  assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
  assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
  assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
    f"v shape {v.shape} incompatible with q shape {q.shape}"
  )
  assert g.ndim == 4 and g.shape == q.shape, (
    f"g shape {g.shape} != q shape {q.shape}"
  )
  assert beta.ndim == 3 and beta.shape == q.shape[:3], (
    f"beta shape {beta.shape} != {q.shape[:3]}"
  )
  if initial_state is not None:
    assert initial_state.shape == (B, H, K, V), (
      f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"
    )

  if scale is None:
    scale = K ** -0.5

  # FLA: q, k, v, g, beta = map(lambda x: x.float(), ...)
  # Transpose [B, T, H, D] -> [B, H, T, D] and cast to accumulator dtype
  q_f, k_f, v_f, g_f = (
    jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt)
    for x in (q, k, v, g)
  )
  # beta: [B, T, H] -> [B, H, T]
  beta_f = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt)

  q_f = q_f * scale

  # Hidden state [B, H, K, V] and output buffer [B, H, T, V]
  S = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if initial_state is not None:
    S = S + initial_state.astype(acc_dt)
  o = jnp.zeros((B, H, T, V), dtype=acc_dt)

  for i in range(T):
    q_i = q_f[:, :, i]     # [B, H, K]
    k_i = k_f[:, :, i]     # [B, H, K]
    v_i = v_f[:, :, i]     # [B, H, V]
    g_i = g_f[:, :, i]     # [B, H, K]
    b_i = beta_f[:, :, i]  # [B, H]

    # 1. Decay the state
    S = S * jnp.exp(g_i)[..., None]

    # 2. Delta rule update
    # v_predicted = k_i^T @ S = sum_k(k_i[k] * S[k, v])
    v_predicted = (k_i[..., None] * S).sum(-2)  # [B, H, V]
    residual = v_i - v_predicted                  # [B, H, V]

    # S += beta * k ⊗ residual
    S = S + jnp.einsum(
      'bhk,bhv->bhkv', b_i[..., None] * k_i, residual
    )

    # 3. Compute output
    o = o.at[:, :, i].set(
      jnp.einsum('bhk,bhkv->bhv', q_i, S)
    )

  final_state = S if output_final_state else None
  # Transpose [B, H, T, V] -> [B, T, H, V] and cast back to original dtype
  return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state
