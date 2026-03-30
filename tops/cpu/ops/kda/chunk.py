"""JAX CPU reference for KDA (Kernel Delta Attention) chunk operations.

This module implements the chunk-based KDA algorithm, directly porting
FLA's naive_chunk_kda to JAX. The algorithm processes sequences in chunks
and uses a triangular system solve for intra-chunk dependencies arising
from the delta rule.

Key algorithm steps:
  1. Reshape inputs into chunks and compute chunk-local cumsum of g
  2. Build intra-chunk interaction matrix A (key-key alignment with decay)
  3. Solve the lower-triangular dependency system iteratively
  4. Compute effective keys (w) and values (u) via the solved system
  5. Propagate inter-chunk state with delta rule correction

KDA differences from Simple GLA:
  - Gate: per-element g: [B,T,H,K] instead of per-head scalar g: [B,T,H]
  - Delta rule: beta learning rate, state update subtracts k^T S from v
  - Intra-chunk: requires solving a linear system, not just masked attention

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation: fp32 (or fp64 for fp64 inputs)
  Final output:
    o: cast back to original v.dtype               [fp64 mode: fp64]
    final_state S: fp32                             [fp64 mode: fp64]
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common.utils import acc_dtype as _acc_dtype
from tops.cpu.ops.common.utils import cdiv as _cdiv
from tops.cpu.ops.common.utils import pad_to_multiple as _pad_to_multiple


@cpu_reference
def chunk_kda(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  g: jnp.ndarray,
  beta: jnp.ndarray,
  scale: float | None = None,
  initial_state: jnp.ndarray | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Chunk KDA with FLA-exact dtype behavior.

  Processes the sequence in chunks of size `chunk_size`, solving the
  intra-chunk delta rule dependencies via a lower-triangular system.

  Algorithm overview:
    1. Split sequence into chunks, cumsum g within each chunk
    2. Build interaction matrix A[c,i] = sum_k k[c,k]*exp(g[c,k]-g[i,k])*k[i,k]
    3. Apply beta, mask upper triangle, negate, solve triangular system
    4. Compute effective updates: w = A @ (exp(g)*k), u = A @ v
    5. For each chunk: output = (q*exp(g)) @ S + A_qk @ (u - w @ S)
       then update S for next chunk

  Core recurrence (equivalent to naive):
      S' = S_{t-1} * exp(g_t)
      S_t = S' + beta_t * k_t ⊗ (v_t - k_t^T @ S')
      o_t = (q_t * scale)^T @ S_t

  Args:
      q:               [B, T, H, K] — Queries
      k:               [B, T, H, K] — Keys
      v:               [B, T, H, V] — Values
      g:               [B, T, H, K] — Per-element gate in log-space
      beta:            [B, T, H]    — Learning rate for delta rule
      scale:           Scalar query scale. Defaults to K ** -0.5.
      initial_state:   [B, H, K, V] — Initial hidden state. Optional.
      output_final_state: Whether to return the final hidden state.
      chunk_size:      Block size for chunked computation.

  Returns:
      o:           [B, T, H, V] — Output (v.dtype)
      final_state: [B, H, K, V] in fp32 (or fp64), or None
  """
  orig_dtype = v.dtype
  acc_dt = _acc_dtype(q.dtype)
  B, T_orig, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # Shape assertions (project coding standard)
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

  # --- Pad T to multiple of chunk_size ---
  T = T_orig
  T_padded = _cdiv(T_orig, C) * C
  if T_padded > T_orig:
    q = _pad_to_multiple(q, C, axis=1)
    k = _pad_to_multiple(k, C, axis=1)
    v = _pad_to_multiple(v, C, axis=1)
    g = _pad_to_multiple(g, C, axis=1)
    beta = _pad_to_multiple(beta, C, axis=1)
    T = T_padded

  NT = T // C

  # --- Reshape to [B, H, NT, C, D] and cast to accumulator dtype ---
  # [B, T, H, K] -> [B, H, T, K] -> [B, H, NT, C, K]
  q_c = jnp.transpose(q, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K) * scale
  k_c = jnp.transpose(k, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K)
  v_c = jnp.transpose(v, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, V)
  g_c = jnp.transpose(g, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K)
  # beta: [B, T, H] -> [B, H, T] -> [B, H, NT, C]
  beta_c = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt).reshape(B, H, NT, C)

  # --- Chunk-local cumulative sum of g ---
  g_c = g_c.cumsum(axis=3)

  # =========================================================================
  # Step 1: Build interaction matrix A
  # A[..., c, i] = sum_k k[c,k] * exp(g[c,k] - g[i,k]) * k[i,k]
  # =========================================================================
  # Upper triangular mask (diagonal=0): masks c <= i positions
  mask_upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))

  A = jnp.zeros((B, H, NT, C, C), dtype=acc_dt)
  for i in range(C):
    k_i = k_c[..., i, :]        # [B, H, NT, K]
    g_i = g_c[..., i:i + 1, :]  # [B, H, NT, 1, K]
    # A[..., :, i] = k[c] · k[i] weighted by exp(g[c] - g[i])
    A = A.at[..., i].set(
      jnp.einsum('...cd,...d->...c', k_c * jnp.exp(g_c - g_i), k_i)
    )

  # Apply beta (learning rate) per column
  A = A * beta_c[..., None]

  # Mask upper triangle (including diagonal) and negate
  A = jnp.where(mask_upper[None, None, None], 0, -A)

  # =========================================================================
  # Step 2: Solve lower-triangular dependency system
  # Forward substitution: A[i, :i] += sum_j A[i, j] * A[j, :i]
  # =========================================================================
  for i in range(1, C):
    # A[..., i, :] has shape [..., C], A[..., :, :i] has shape [..., C, i]
    update = jnp.einsum(
      '...j,...ji->...i', A[..., i, :], A[..., :, :i]
    )
    A = A.at[..., i, :i].set(A[..., i, :i] + update)

  # Add identity and scale by beta
  A = (A + jnp.eye(C, dtype=acc_dt)) * beta_c[..., None, :]

  # =========================================================================
  # Step 3: Compute effective keys (w) and values (u)
  # w = A @ (exp(g) * k): how much S changes per unit of past S
  # u = A @ v: accumulated effective values
  # =========================================================================
  w = jnp.einsum('...ij,...jk->...ik', A, jnp.exp(g_c) * k_c)  # [B,H,NT,C,K]
  u = jnp.einsum('...ij,...jv->...iv', A, v_c)                   # [B,H,NT,C,V]

  # =========================================================================
  # Step 4: Inter-chunk recurrence with delta rule correction
  # =========================================================================
  S = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if initial_state is not None:
    S = S + initial_state.astype(acc_dt)
  o = jnp.zeros_like(v_c)  # [B, H, NT, C, V]

  # Strict upper triangular mask (diagonal=1) for intra-chunk attention
  mask_strict_upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=1)

  for i in range(NT):
    q_i = q_c[:, :, i]   # [B, H, C, K]
    k_i = k_c[:, :, i]   # [B, H, C, K]
    u_i = u[:, :, i]     # [B, H, C, V]
    g_i = g_c[:, :, i]   # [B, H, C, K]
    w_i = w[:, :, i]     # [B, H, C, K]

    # --- Intra-chunk attention matrix ---
    # A_qk[..., c, j] = q[c] · k[j] weighted by exp(g[c] - g[j])
    A_qk = jnp.zeros((B, H, C, C), dtype=acc_dt)
    for j in range(C):
      k_j = k_c[:, :, i, j]        # [B, H, K]
      g_j = g_c[:, :, i, j:j + 1, :]  # [B, H, 1, K]
      A_qk = A_qk.at[..., j].set(
        jnp.einsum('...cd,...d->...c', q_i * jnp.exp(g_i - g_j), k_j)
      )
    # Mask future positions (strict upper triangle)
    A_qk = jnp.where(mask_strict_upper[None, None], 0, A_qk)

    # --- Delta rule correction ---
    # v_corrected = u_i - w_i @ S: effective values minus state projection
    v_corrected = u_i - jnp.einsum('...ck,...kv->...cv', w_i, S)

    # --- Output ---
    # o = (q * exp(g)) @ S (inter-chunk) + A_qk @ v_corrected (intra-chunk)
    o = o.at[:, :, i].set(
      jnp.einsum('...ck,...kv->...cv', q_i * jnp.exp(g_i), S)
      + jnp.einsum('...ij,...jv->...iv', A_qk, v_corrected)
    )

    # --- Update state for next chunk ---
    g_last = g_i[:, :, -1]  # [B, H, K]
    # Decay S by exp(g_last)
    S = S * jnp.exp(g_last)[..., None]
    # Add delta-corrected key-value outer product
    k_weighted = jnp.exp(g_last[:, :, None, :] - g_i) * k_i  # [B, H, C, K]
    S = S + jnp.einsum('...ck,...cv->...kv', k_weighted, v_corrected)

  # --- Reshape output back ---
  # [B, H, NT, C, V] -> [B, H, T, V] -> [B, T, H, V]
  o = o.reshape(B, H, T, V)
  o = jnp.transpose(o, (0, 2, 1, 3))[:, :T_orig].astype(orig_dtype)

  final_state = S if output_final_state else None
  return o, final_state
